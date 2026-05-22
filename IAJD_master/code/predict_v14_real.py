"""
predict_v14_real.py — v14.0 inference using REAL LION + ADMET predictions.

Auto-extends lion_cache_v13.json and admet_cache_v13.json with new query SMILES
before calling the v14.0 model. This is the recommended production inference path
since v14.0 has the smallest in-sample/honest gap (-0.003, essentially zero).

Usage:
    from predict_v14_real import predict
    r = predict('CCCCCCCCCCCCOc1cc(...)c1OCCCC...')
    print(r['log10_flux_total'])         # 7.480
    print(r['log10_flux_total_PI90'])    # (7.280, 7.680)
    print(r['family'])                    # 'sSS-Nonsym'
    print(r['warnings'])                  # any flags

Pass auto_compute=False to skip real LION/ADMET computation and fall back to
the v14 pipeline's RDKit proxies (faster but less accurate).
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

OUT = Path('/mnt/user-data/outputs/bioact_v14')
sys.path.insert(0, str(OUT))
from bioact_v14_pipeline import assemble_X, BLOCK_SLICES
from iajd_bioact_v14 import _detect_family

MFPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
import pandas as pd


def _ensure_real_features(canonical_smi):
    """Extend caches with this canonical SMILES if not already present."""
    import json
    lion_path = OUT / 'lion_cache_v13.json'
    admet_path = OUT / 'admet_cache_v13.json'
    have_lion = False; have_admet = False
    if lion_path.exists():
        with open(lion_path) as f:
            lion_cache = json.load(f)
            have_lion = canonical_smi in lion_cache
    if admet_path.exists():
        with open(admet_path) as f:
            admet_cache = json.load(f)
            have_admet = canonical_smi in admet_cache
    if have_lion and have_admet:
        return True, []  # already cached
    # Need to compute one or both
    warnings = []
    try:
        sys.path.insert(0, str(OUT))
        from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles
        if not have_admet:
            predict_admet_for_smiles([canonical_smi], verbose=False)
        if not have_lion:
            predict_lion_for_smiles([canonical_smi], verbose=False)
        return True, warnings
    except Exception as e:
        warnings.append(f'real_feature_fetch_failed: {type(e).__name__}: {str(e)[:200]}')
        return False, warnings


def predict(smiles, bundle_path=None, auto_compute=True, formulation=None,
            return_neighbors=5, family_hint=None):
    """Predict log10_flux_total with real LION + ADMET when possible."""
    if bundle_path is None:
        bundle_path = OUT / 'bioact_v14_bundle.pkl'
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'invalid SMILES: {smiles}'}
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Step 1: auto-extend caches if requested
    fetch_warnings = []
    if auto_compute:
        ok, ws = _ensure_real_features(canonical)
        fetch_warnings.extend(ws)

    fp = MFPGEN.GetFingerprint(mol)
    family = family_hint or _detect_family(mol, fp, bundle)[0]
    family_method = 'user_hint' if family_hint else _detect_family(mol, fp, bundle)[1]

    # Build features using the cached/real predictions
    row = {
        'SMILES_canonical': canonical, 'canonical_smi': canonical,
        'family': family, 'IAJD_id': 'query', 'log10_flux_total': 0,
    }
    if formulation:
        for k, v in formulation.items(): row[k] = v
    df_q = pd.DataFrame([row])

    lion_path = OUT / 'lion_cache_v13.json'
    admet_path = OUT / 'admet_cache_v13.json'
    X_q, lion_modes = assemble_X(
        df_q, [mol], [fp], [canonical],
        lion_cache_path=str(lion_path) if lion_path.exists() else None,
        admet_cache_path=str(admet_path) if admet_path.exists() else None,
    )
    x = X_q[0]
    block_b_used_real = (lion_modes[0] == 'cached') if lion_modes else False

    # Apply per-family gates
    gate_b = bundle['block_b_active_per_family'].get(family, True)
    gate_c = bundle['block_c_active_per_family'].get(family, True)
    x_gated = x.copy()
    if not gate_b: x_gated[BLOCK_SLICES['B'][0]:BLOCK_SLICES['B'][1]] = 0
    if not gate_c: x_gated[BLOCK_SLICES['C'][0]:BLOCK_SLICES['C'][1]] = 0

    # Predict
    pred_direct = float(bundle['direct_model_full'].predict(x_gated.reshape(1, -1))[0])

    # Analog-delta
    fps_train = bundle['fps_train']
    y_train = bundle['y_train']
    sims = np.array(BulkTanimotoSimilarity(fp, fps_train))
    top = np.argsort(-sims)[:8]
    if sims[top[0]] >= 0.4:
        weights = sims[top] ** 4
        weights = weights / weights.sum()
        pred_delta = float(np.sum(weights * y_train[top]))
    else:
        pred_delta = None

    # Blend with per-family α
    alpha = bundle['best_alpha_per_family'].get(family, 1.0)
    if pred_delta is None:
        pred = pred_direct
        alpha_used = 1.0  # forced
    else:
        pred = alpha * pred_direct + (1 - alpha) * pred_delta
        alpha_used = alpha

    # Confidence tier from max Tanimoto
    max_sim = float(sims[np.argmax(sims)])
    if max_sim >= 0.7: tier, pi_half = 'HIGH', 0.20
    elif max_sim >= 0.5: tier, pi_half = 'MED', 0.30
    else: tier, pi_half = 'LOW', 0.45

    top_n = np.argsort(-sims)[:return_neighbors]
    smis_train = bundle['smis_train']
    fams_train = bundle['families_train']
    neighbors = [
        {'smiles': smis_train[i], 'family': fams_train[i],
         'log10_flux_total': float(y_train[i]), 'tanimoto': float(sims[i])}
        for i in top_n
    ]

    warnings_out = list(fetch_warnings)
    if max_sim < 0.4:
        warnings_out.append('NO_CLOSE_NEIGHBOR: max_tanimoto < 0.4; analog path skipped')
    if family == 'Unknown':
        warnings_out.append('FAMILY_UNKNOWN')
    if not gate_b: warnings_out.append(f'BLOCK_B_OFF: LION features masked for family={family}')
    if not gate_c: warnings_out.append(f'BLOCK_C_OFF: ADMET features masked for family={family}')
    if not block_b_used_real:
        warnings_out.append('LION_PROXY_USED: real LION cache lookup failed; using RDKit proxy')

    return {
        'smiles': canonical,
        'log10_flux_total': pred,
        'log10_flux_total_PI90': (pred - pi_half, pred + pi_half),
        'family': family, 'family_method': family_method,
        'alpha_used': alpha_used,
        'block_B_active': bool(gate_b),
        'block_C_active': bool(gate_c),
        'block_B_real': bool(block_b_used_real),
        'pred_direct': pred_direct,
        'pred_analog': pred_delta,
        'max_tanimoto': max_sim,
        'confidence_tier': tier,
        'nearest_neighbors': neighbors,
        'warnings': warnings_out,
        'honest_pooled_mae_reference': bundle['metrics'].get('v14_honest_pooled_mae'),
    }


def predict_batch(smiles_list, bundle_path=None, auto_compute=True, **kwargs):
    """Batch prediction."""
    rows = []
    for s in smiles_list:
        r = predict(s, bundle_path=bundle_path, auto_compute=auto_compute, **kwargs)
        rows.append({
            'smiles': r.get('smiles', s),
            'log10_flux_total': r.get('log10_flux_total'),
            'PI90_lo': r.get('log10_flux_total_PI90', (None, None))[0],
            'PI90_hi': r.get('log10_flux_total_PI90', (None, None))[1],
            'family': r.get('family'),
            'max_tanimoto': r.get('max_tanimoto'),
            'confidence_tier': r.get('confidence_tier'),
            'block_B_real': r.get('block_B_real'),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    # Smoke test
    test_cases = [
        ('GA-Tris-like (training in-domain)',
         'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC'),
        ('Novel-ish IAJD (lower Tanimoto)',
         'CCCCCCCCCCCCN(CCO)CCOC(=O)c1ccc(OC)cc1OC'),
    ]
    for label, smi in test_cases:
        print(f'\n=== {label} ===')
        r = predict(smi, auto_compute=True, return_neighbors=2)
        print(f'  SMILES: {smi[:60]}...')
        print(f'  log10_flux_total: {r["log10_flux_total"]:.3f}  PI90: {r["log10_flux_total_PI90"]}')
        print(f'  Family: {r["family"]} ({r["family_method"]})  α: {r["alpha_used"]}')
        print(f'  Block B real: {r["block_B_real"]}  (gates: B={r["block_B_active"]}, C={r["block_C_active"]})')
        print(f'  Tanimoto: {r["max_tanimoto"]:.3f}, tier: {r["confidence_tier"]}')
        for w in r['warnings']: print(f'  ⚠️  {w}')
