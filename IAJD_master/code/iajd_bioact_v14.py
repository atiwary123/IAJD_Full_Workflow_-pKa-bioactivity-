"""
iajd_bioact_v14.py — v14 inference API.

Public surface:
    load_bundle(path='bioact_v14_bundle.pkl') -> dict
    predict_bioactivity(smiles, bundle, ...) -> dict
    predict_bioactivity_batch(smiles_list, bundle, ...) -> pandas.DataFrame

The v14 prediction pipeline:
  1. Canonicalize SMILES + Morgan-2 fingerprint.
  2. Detect family (SMARTS-based with nearest-neighbor fallback).
  3. Compute 88-d feature vector using the same assemble_X as training.
  4. Apply per-family Block B/Block C gating (mask the relevant slice to 0
     if that family's gate decision is "OFF").
  5. Compute direct prediction from full-data XGBoost.
  6. Compute analog-delta prediction from training neighbors.
  7. Blend with per-family α.
  8. Compute Tanimoto-to-train, confidence tier, neighbor list.
  9. Return rich diagnostic dict.

Output fields:
  log10_flux_total          : the v14 prediction
  log10_flux_total_PI90     : ±half-width (heuristic, see notes)
  family                    : detected family
  family_method             : how family was decided
  alpha_used                : the per-family α blend weight
  block_B_active            : whether LION features contributed (per family gate)
  block_C_active            : whether ADMET features contributed (per family gate)
  direct_pred               : direct XGBoost prediction (always computed)
  delta_pred                : analog-delta prediction (None if no neighbor ≥ 0.4)
  max_tanimoto              : similarity to nearest training compound
  confidence_tier           : HIGH / MED / LOW
  nearest_neighbors         : top-K list of (smiles, family, y, tanimoto)
  lion_proxy_or_cached      : whether LION block was real cache or RDKit proxy
  warnings                  : any flags
"""
import os, sys, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# RDKit compat shim (project root, not code/) must load before GetMorganGenerator call
_proj_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
import rdkit_compat  # noqa: F401
from bioact_v14_pipeline import (
    assemble_X, BLOCK_SLICES, ALL_NAMES_V14,
)

MFPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
WORK = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Family detection
# -------------------------------------------------------------------
_SMARTS_PE_TRIS = Chem.MolFromSmarts('[CX4]([CH2]O)([CH2]O)([CH2]O)[CH2]O')
_SMARTS_GA_TRIS = Chem.MolFromSmarts('c1cc(C(=O)NC([CH2O])([CH2O])[CH2O])cc1')
_SMARTS_PE_GALLIC = Chem.MolFromSmarts('c1c([OX2])c([OX2])c([OX2])cc1[CX3]=[OX1]')
_SMARTS_DIALKOXY_3_5 = Chem.MolFromSmarts('[OX2]c1cc([CX4])cc([OX2])c1')


def _detect_family(mol, fp, bundle):
    """Heuristic family detection: SMARTS first, NN fallback."""
    if mol is None:
        return 'Unknown', 'invalid_mol'
    if _SMARTS_PE_TRIS and mol.HasSubstructMatch(_SMARTS_PE_TRIS):
        return 'PE-Tris', 'SMARTS'
    if _SMARTS_GA_TRIS and mol.HasSubstructMatch(_SMARTS_GA_TRIS):
        return 'GA-Tris', 'SMARTS'
    if _SMARTS_PE_GALLIC and mol.HasSubstructMatch(_SMARTS_PE_GALLIC):
        return 'PE-Gallic', 'SMARTS'
    # Nearest-neighbor fallback
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
    best_fam = max(vote, key=vote.get)
    return best_fam, 'NN_vote'


# -------------------------------------------------------------------
# Bundle loading
# -------------------------------------------------------------------
def load_bundle(path=None):
    if path is None:
        path = WORK / 'bioact_v14_bundle.pkl'
        if not Path(path).exists():
            # Canonical location in this repo: IAJD_master/bundles_caches/
            path = WORK.parent / 'bundles_caches' / 'bioact_v14_bundle.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def predict_bioactivity(smiles, bundle, formulation=None,
                        return_neighbors=5, family_hint=None):
    """v14 prediction. Returns a dict (see module docstring)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'invalid SMILES: {smiles}'}
    canonical = Chem.MolToSmiles(mol, canonical=True)
    fp = MFPGEN.GetFingerprint(mol)

    # Family
    if family_hint:
        family = family_hint
        family_method = 'user_hint'
    else:
        family, family_method = _detect_family(mol, fp, bundle)

    # Build single-row dataframe and assemble features
    row = {
        'SMILES_canonical': canonical, 'canonical_smi': canonical,
        'family': family, 'IAJD_id': 'query', 'log10_flux_total': 0,
    }
    if formulation:
        for k, v in formulation.items(): row[k] = v
    df_q = pd.DataFrame([row])
    # Use the training pipeline's assembler for consistency
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

    # Direct prediction from full-data model
    direct_pred = float(bundle['direct_model_full'].predict(x_gated.reshape(1, -1))[0])

    # Analog-delta from neighbors
    fps_train = bundle['fps_train']
    y_train = bundle['y_train']
    sims = np.array(BulkTanimotoSimilarity(fp, fps_train))
    top = np.argsort(-sims)[:8]
    max_sim = float(sims[top[0]])
    if max_sim >= 0.4:
        w = sims[top] ** 4
        w /= w.sum()
        delta_pred = float(np.sum(w * y_train[top]))
    else:
        delta_pred = None

    # Blend with per-family α
    alpha = bundle['best_alpha_per_family'].get(family, 0.6)
    if delta_pred is None:
        pred = direct_pred
        alpha_used = 1.0
    else:
        pred = alpha * direct_pred + (1 - alpha) * delta_pred
        alpha_used = alpha

    # Confidence tier
    if max_sim >= 0.7:
        tier = 'HIGH'
    elif max_sim >= 0.5:
        tier = 'MED'
    else:
        tier = 'LOW'

    # Heuristic PI half-width
    pi_half = 0.20 if tier == 'HIGH' else (0.30 if tier == 'MED' else 0.45)

    # Top neighbors
    smis_train = bundle['smis_train']
    fams_train = bundle['families_train']
    top_n = np.argsort(-sims)[:return_neighbors]
    neighbors = [
        {'smiles': smis_train[i], 'family': fams_train[i],
         'log10_flux_total': float(y_train[i]), 'tanimoto': float(sims[i])}
        for i in top_n
    ]

    # Warnings
    warnings = []
    if max_sim < 0.4:
        warnings.append('NO_CLOSE_NEIGHBOR: max_tanimoto < 0.4; delta path skipped')
    if family == 'Unknown':
        warnings.append('FAMILY_UNKNOWN: no SMARTS match, NN vote unclear')
    if not gate_b:
        warnings.append(f'BLOCK_B_OFF: LION features masked for family={family} per gate decision')
    if not gate_c:
        warnings.append(f'BLOCK_C_OFF: ADMET features masked for family={family} per gate decision')
    # Note: in this bundle, the LION/ADMET blocks were computed via RDKit proxy
    # (no live LION checkpoints, no admet-ai install in sandbox).
    lion_origin = bundle.get('lion_modes', ['proxy'])[0]
    if lion_origin == 'proxy':
        warnings.append('LION_PROXY: trained on RDKit-proxy LION features, not live LION '
                        'predictions. Plug lion_cache_v13.json into the pipeline to upgrade.')

    return {
        'smiles': canonical,
        'log10_flux_total': pred,
        'log10_flux_total_PI90': (pred - pi_half, pred + pi_half),
        'family': family, 'family_method': family_method,
        'alpha_used': alpha_used,
        'block_B_active': bool(gate_b),
        'block_C_active': bool(gate_c),
        'direct_pred': direct_pred,
        'delta_pred': delta_pred,
        'max_tanimoto': max_sim,
        'confidence_tier': tier,
        'nearest_neighbors': neighbors,
        'lion_proxy_or_cached': lion_origin,
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
            'alpha_used': r.get('alpha_used'),
            'block_B_active': r.get('block_B_active'),
            'block_C_active': r.get('block_C_active'),
            'max_tanimoto': r.get('max_tanimoto'),
            'confidence_tier': r.get('confidence_tier'),
            'n_warnings': len(r.get('warnings', [])),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    bundle = load_bundle()
    print(f'Loaded v14 bundle: n_train={len(bundle["X_train"])}, '
          f'families={bundle["family_list"]}')
    test_smi = 'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC'
    r = predict_bioactivity(test_smi, bundle)
    print(f'\nTest prediction:')
    print(f'  log10_flux_total = {r["log10_flux_total"]:.3f}')
    print(f'  PI90             = {r["log10_flux_total_PI90"]}')
    print(f'  family           = {r["family"]} ({r["family_method"]})')
    print(f'  α used           = {r["alpha_used"]}, B={r["block_B_active"]}, C={r["block_C_active"]}')
    print(f'  max_tanimoto     = {r["max_tanimoto"]:.3f}, tier = {r["confidence_tier"]}')
    print(f'  nearest neighbors:')
    for nbr in r['nearest_neighbors'][:3]:
        print(f'    Tanimoto={nbr["tanimoto"]:.3f}  fam={nbr["family"]}  '
              f'y={nbr["log10_flux_total"]:.2f}')
    print(f'  warnings:')
    for w in r['warnings']:
        print(f'    - {w}')
