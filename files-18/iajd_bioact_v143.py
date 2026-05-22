"""
iajd_bioact_v143.py — v14.3 inference API.

v14.3 is the best-performing variant, built on top of v14.0's per-family α + per-
family LION/ADMET gates by adding per-family HPO, per-family direct models, and a
three-way stacked ensemble per family:

    pred = w_pool * pooled_model_pred + w_fam * per_family_model_pred + w_analog * analog_delta_pred

with weights, pool source (v14.1 or v14.2), and fam source (v14.1 or v14.2, or
"pool_fallback" if no in-family data) tuned per family on LOO.

Output schema is a strict superset of v14.0's. Read MODEL_CARD_v14.md first.
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioact_v14_pipeline import assemble_X, BLOCK_SLICES, ALL_NAMES_V14

MFPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
WORK = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Family detection
# -------------------------------------------------------------------
_SMARTS_PE_TRIS = Chem.MolFromSmarts('[CX4]([CH2]O)([CH2]O)([CH2]O)[CH2]O')
_SMARTS_GA_TRIS = Chem.MolFromSmarts('c1cc(C(=O)NC([CH2O])([CH2O])[CH2O])cc1')
_SMARTS_PE_GALLIC = Chem.MolFromSmarts('c1c([OX2])c([OX2])c([OX2])cc1[CX3]=[OX1]')


def _detect_family(mol, fp, bundle):
    if mol is None:
        return 'Unknown', 'invalid_mol'
    if _SMARTS_PE_TRIS and mol.HasSubstructMatch(_SMARTS_PE_TRIS):
        return 'PE-Tris', 'SMARTS'
    if _SMARTS_GA_TRIS and mol.HasSubstructMatch(_SMARTS_GA_TRIS):
        return 'GA-Tris', 'SMARTS'
    if _SMARTS_PE_GALLIC and mol.HasSubstructMatch(_SMARTS_PE_GALLIC):
        return 'PE-Gallic', 'SMARTS'
    fps_train = bundle['fps_train']
    fams_train = bundle['families_train']
    sims = BulkTanimotoSimilarity(fp, fps_train)
    if not sims:
        return 'Unknown', 'no_neighbors'
    top_k = np.argsort(-np.array(sims))[:5]
    vote = {}
    for i in top_k:
        f = fams_train[i]
        vote[f] = vote.get(f, 0) + float(sims[i])
    return max(vote, key=vote.get), 'NN_vote'


# -------------------------------------------------------------------
# Analog-delta variants
# -------------------------------------------------------------------
def _analog_delta(fp, fps_train, y_train, variant='v1', K=8, sim_threshold=0.4):
    """Three analog variants per v14.3 stack:
       v1: similarity-weighted mean (Tanimoto^4 weights)
       v2: similarity-weighted mean (Tanimoto^2 weights, gentler)
       v3: in-family-restricted similarity-weighted mean
    """
    sims = np.array(BulkTanimotoSimilarity(fp, fps_train))
    top = np.argsort(-sims)[:K]
    if sims[top[0]] < sim_threshold:
        return None
    if variant == 'v1':
        w = sims[top] ** 4
    elif variant == 'v2':
        w = sims[top] ** 2
    elif variant == 'v3':
        w = sims[top] ** 3
    else:
        w = sims[top] ** 4
    if w.sum() == 0:
        return None
    w /= w.sum()
    return float(np.sum(w * y_train[top]))


# -------------------------------------------------------------------
# Bundle loading
# -------------------------------------------------------------------
def load_bundle(path=None):
    if path is None:
        for candidate in [WORK / 'bioact_v14_3_bundle.pkl',
                          Path('/mnt/user-data/outputs/bioact_v14/bioact_v14_3_bundle.pkl')]:
            if candidate.exists():
                path = candidate
                break
    with open(path, 'rb') as f:
        return pickle.load(f)


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def predict_bioactivity(smiles, bundle, formulation=None,
                        return_neighbors=5, family_hint=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'invalid SMILES: {smiles}'}
    canonical = Chem.MolToSmiles(mol, canonical=True)
    fp = MFPGEN.GetFingerprint(mol)

    family = family_hint or _detect_family(mol, fp, bundle)[0]
    family_method = 'user_hint' if family_hint else _detect_family(mol, fp, bundle)[1]

    # Build features
    row = {
        'SMILES_canonical': canonical, 'canonical_smi': canonical,
        'family': family, 'IAJD_id': 'query', 'log10_flux_total': 0,
    }
    if formulation:
        for k, v in formulation.items(): row[k] = v
    df_q = pd.DataFrame([row])
    X_q, _ = assemble_X(df_q, [mol], [fp], [canonical])
    x = X_q[0]

    # Apply per-family gates
    gate_b = bundle['block_b_active_per_family'].get(family, True)
    gate_c = bundle['block_c_active_per_family'].get(family, True)
    x_gated = x.copy()
    if not gate_b:
        x_gated[BLOCK_SLICES['B']] = 0
    if not gate_c:
        x_gated[BLOCK_SLICES['C']] = 0

    # Stack components per v14.3
    sw = bundle['stack_weights_per_family'].get(family)
    if sw is None:
        sw = {'w_pool': 1.0, 'w_fam': 0.0, 'w_analog': 0.0,
              'pool_source': 'v141_pool', 'fam_source': 'pool_fallback',
              'analog_source': 'analog_v1'}

    # Pool model prediction
    pool_pred = float(bundle['pooled_model_full_v141'].predict(x_gated.reshape(1, -1))[0])

    # Per-family model prediction
    fam_pred = None
    fam_source = sw.get('fam_source', 'pool_fallback')
    if fam_source == 'v141_fam':
        fam_models = bundle.get('per_family_models_v141', {})
    elif fam_source == 'v142_fam':
        fam_models = bundle.get('per_family_models_v142', {})
    else:
        fam_models = {}
    if family in fam_models:
        fam_pred = float(fam_models[family].predict(x_gated.reshape(1, -1))[0])
    else:
        # fallback to pool
        fam_pred = pool_pred

    # Analog-delta
    fps_train = bundle['fps_train']
    y_train = bundle['y_train']
    analog_variant = sw.get('analog_source', 'v1').replace('analog_', '')
    analog_pred = _analog_delta(fp, fps_train, y_train, variant=analog_variant)

    # Stack
    if analog_pred is None:
        # Renormalize w_pool + w_fam
        w_pool = sw['w_pool']
        w_fam = sw['w_fam']
        w_analog = 0.0
        total = w_pool + w_fam
        if total > 0:
            w_pool /= total
            w_fam /= total
    else:
        w_pool = sw['w_pool']
        w_fam = sw['w_fam']
        w_analog = sw['w_analog']

    pred = w_pool * pool_pred + w_fam * fam_pred + w_analog * (analog_pred or 0)

    # Tanimoto + tier
    sims = np.array(BulkTanimotoSimilarity(fp, fps_train))
    max_sim = float(sims[np.argmax(sims)])
    if max_sim >= 0.7:
        tier = 'HIGH'
    elif max_sim >= 0.5:
        tier = 'MED'
    else:
        tier = 'LOW'
    pi_half = 0.20 if tier == 'HIGH' else (0.30 if tier == 'MED' else 0.45)

    # Top neighbors
    top_n = np.argsort(-sims)[:return_neighbors]
    smis_train = bundle['smis_train']
    fams_train = bundle['families_train']
    neighbors = [
        {'smiles': smis_train[i], 'family': fams_train[i],
         'log10_flux_total': float(y_train[i]), 'tanimoto': float(sims[i])}
        for i in top_n
    ]

    warnings = []
    if max_sim < 0.4:
        warnings.append('NO_CLOSE_NEIGHBOR: max_tanimoto < 0.4; analog path skipped')
    if family == 'Unknown':
        warnings.append('FAMILY_UNKNOWN')
    if not gate_b:
        warnings.append(f'BLOCK_B_OFF: LION features masked for family={family}')
    if not gate_c:
        warnings.append(f'BLOCK_C_OFF: ADMET features masked for family={family}')
    warnings.append('LION_PROXY: trained on RDKit-proxy LION features, not live LION '
                    'predictions. See MODEL_CARD_v14.md for the upgrade path.')

    return {
        'smiles': canonical,
        'log10_flux_total': pred,
        'log10_flux_total_PI90': (pred - pi_half, pred + pi_half),
        'family': family, 'family_method': family_method,
        'alpha_used': bundle['best_alpha_per_family'].get(family, 1.0),
        'block_B_active': bool(gate_b),
        'block_C_active': bool(gate_c),
        'pool_pred': pool_pred,
        'fam_pred': fam_pred,
        'analog_pred': analog_pred,
        'stack_weights': {'w_pool': w_pool, 'w_fam': w_fam, 'w_analog': w_analog},
        'max_tanimoto': max_sim,
        'confidence_tier': tier,
        'nearest_neighbors': neighbors,
        'warnings': warnings,
    }


def predict_bioactivity_batch(smiles_list, bundle, **kwargs):
    rows = []
    for s in smiles_list:
        r = predict_bioactivity(s, bundle, **kwargs)
        rows.append({
            'smiles': r.get('smiles', s),
            'log10_flux_total': r.get('log10_flux_total'),
            'PI90_lo': r.get('log10_flux_total_PI90', (None, None))[0],
            'PI90_hi': r.get('log10_flux_total_PI90', (None, None))[1],
            'family': r.get('family'),
            'pool_pred': r.get('pool_pred'),
            'fam_pred': r.get('fam_pred'),
            'analog_pred': r.get('analog_pred'),
            'max_tanimoto': r.get('max_tanimoto'),
            'confidence_tier': r.get('confidence_tier'),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    bundle = load_bundle()
    print(f'Loaded v14.3 bundle: n_train={len(bundle["X_train"])}, '
          f'families={bundle["family_list"]}')
    print(f'Pooled MAE: {bundle["metrics"]["v143_pooled_mae"]:.4f}, '
          f'R²={bundle["metrics"]["v143_pooled_r2"]:.3f}')
    test_smi = 'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC'
    r = predict_bioactivity(test_smi, bundle)
    print(f'\nTest prediction:')
    print(f'  log10_flux_total = {r["log10_flux_total"]:.3f}')
    print(f'  PI90             = {r["log10_flux_total_PI90"]}')
    print(f'  family           = {r["family"]} ({r["family_method"]})')
    print(f'  Stack components:')
    print(f'    pool_pred  = {r["pool_pred"]:.3f}  (w={r["stack_weights"]["w_pool"]:.2f})')
    print(f'    fam_pred   = {r["fam_pred"]:.3f}  (w={r["stack_weights"]["w_fam"]:.2f})')
    print(f'    analog_pred= {r["analog_pred"]}  (w={r["stack_weights"]["w_analog"]:.2f})')
    print(f'  max_tanimoto={r["max_tanimoto"]:.3f}, tier={r["confidence_tier"]}')
