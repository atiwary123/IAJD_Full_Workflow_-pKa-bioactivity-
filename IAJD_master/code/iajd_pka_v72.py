"""
iajd_pka_v72.py — IAJD pKa Prediction System v7.2

Adaptive routing by max-Tanimoto similarity to nearest training compound:
  Tanimoto ≥ 0.90:   pure analog-delta (the analog IS essentially the answer)
  0.70 ≤ T < 0.90:   blend with smooth alpha = direct weight scaling 0.20 → 0.65
  0.50 ≤ T < 0.70:   blend with alpha 0.65 → 0.95
  T < 0.50:          pure direct XGBoost (no good analog, full model fallback)

GA-Tris uses the v5.2 hierarchy regardless (n=11 too small for ML).

CIs are now query-conditional based on routing path:
  - Pure delta path:  base CI from analog measurement noise
  - Blend path:       base CI from family-LOO error
  - Pure direct path: wider fallback CI
"""

from __future__ import annotations
import os, sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iajd_pka_v52 import (
    build_bundle as build_v52_bundle,
    compute_features, compute_3d_features_from_mol, tokens_from_mol,
    MFPGEN, predict_pka as predict_v52, _MOLGPKA_CACHE,
)
from iajd_pka_v71 import _try_live_molgpka, V71Bundle

K_NEIGHBORS = 8
XGB_HP = dict(n_estimators=200, max_depth=3, learning_rate=0.08,
              subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
              min_child_weight=5, random_state=42, n_jobs=1)

# Adaptive alpha schedule (alpha = weight on direct, 1-alpha = weight on delta)
def alpha_from_tan(t: float) -> float:
    if t >= 0.90: return 0.0
    if t >= 0.85: return 0.20 + (0.90 - t) * (0.30 - 0.20) / 0.05
    if t >= 0.70: return 0.30 + (0.85 - t) * (0.65 - 0.30) / 0.15
    if t >= 0.50: return 0.65 + (0.70 - t) * (0.95 - 0.65) / 0.20
    if t >= 0.40: return 0.95 + (0.50 - t) * (1.0 - 0.95) / 0.10
    return 1.0


# CI half-widths by routing band and family
# These are calibrated from v7.1 LOO errors but vary by routing path
CI_TABLE = {
    'PE-Tris':       {'pure_delta': 0.10, 'blend': 0.18, 'pure_direct': 0.30, 'OOD': 0.50},
    'GA-Tris':       {'pure_delta': 0.10, 'blend': 0.16, 'pure_direct': 0.25, 'OOD': 0.40},
    'sSS-Nonsym':    {'pure_delta': 0.12, 'blend': 0.20, 'pure_direct': 0.35, 'OOD': 0.50},
    'PE-Gallic':     {'pure_delta': 0.15, 'blend': 0.28, 'pure_direct': 0.45, 'OOD': 0.60},
    'Dialkoxybenzyl':{'pure_delta': 0.18, 'blend': 0.32, 'pure_direct': 0.55, 'OOD': 0.70},
    'default':       {'pure_delta': 0.15, 'blend': 0.28, 'pure_direct': 0.45, 'OOD': 0.60},
}


def load_v72_bundle(xlsx_path: str = "IAJD_pKa_v21_final.xlsx") -> V71Bundle:
    """Same bundle structure as v7.1 — only the prediction logic differs."""
    from iajd_pka_v71 import load_v71_bundle
    return load_v71_bundle(xlsx_path)


def _analog_delta(q_X, q_fp, bundle, K=K_NEIGHBORS):
    """Same as v7.1 analog-delta. Returns (pred, neighbors)."""
    n = len(bundle.train_y)
    sims = np.array([TanimotoSimilarity(q_fp, bundle.train_fps[j]) for j in range(n)])
    top_k = np.argsort(-sims)[:K]
    delta_X, delta_y = [], []
    for j in range(n):
        sims_j = np.array([TanimotoSimilarity(bundle.train_fps[j], bundle.train_fps[k]) for k in range(n)])
        sims_j[j] = -1
        for k in np.argsort(-sims_j)[:K]:
            if k == j: continue
            delta_X.append(bundle.train_X[j] - bundle.train_X[k])
            delta_y.append(float(bundle.train_y[j] - bundle.train_y[k]))
    dX = np.array(delta_X); dy = np.array(delta_y)
    sc = StandardScaler().fit(dX)
    m = xgb.XGBRegressor(**XGB_HP); m.fit(sc.transform(dX), dy, verbose=False)
    ests, wts, info = [], [], []
    for a, s in zip(top_k, sims[top_k]):
        d = (q_X - bundle.train_X[a]).reshape(1,-1)
        dp = float(m.predict(sc.transform(d))[0])
        est = float(bundle.train_y[a]) + dp
        ests.append(est); wts.append(float(s))
        info.append({'IAJD': int(bundle.v52_bundle.ids[a]), 'tanimoto': round(float(s),3),
                     'anchor_pKa': float(bundle.train_y[a]), 'predicted_delta': round(dp,3),
                     'estimated_pKa': round(est,3), 'family': str(bundle.train_fams[a])})
    wts = np.array(wts); wts = wts/wts.sum() if wts.sum() > 0 else wts
    return float(np.dot(wts, ests)), info


def predict_pka_v72(smiles, bundle, architecture=None, family_hint=None,
                    return_diagnostics=True):
    """Adaptive-routing IAJD pKa prediction."""
    out = {'smiles': smiles, 'version': 'v7.2'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out['error'] = 'INVALID_SMILES'; return out
    canonical = Chem.MolToSmiles(mol, canonical=True)
    out['canonical_smiles'] = canonical
    q_tokens = tokens_from_mol(mol, architecture=architecture, family_hint=family_hint)
    out['family_assigned'] = q_tokens.family

    features_3d = compute_3d_features_from_mol(mol)
    if not any(np.isfinite(v) for v in features_3d.values() if v is not None):
        out.setdefault('warnings', []).append('CONFORMER_EMBED_FAILED')

    if canonical not in _MOLGPKA_CACHE:
        try: _try_live_molgpka(mol, canonical)
        except Exception: out.setdefault('warnings', []).append('MOLGPKA_LIVE_FAILED')

    q_X = compute_features(mol, q_tokens, features_3d=features_3d)
    nan_mask = ~np.isfinite(q_X)
    if nan_mask.any():
        med = np.nanmedian(bundle.train_X, axis=0)
        q_X = np.where(nan_mask, med, q_X)
        out.setdefault('warnings', []).append(f'IMPUTED_{int(nan_mask.sum())}_FEATURES')

    q_fp = MFPGEN.GetFingerprint(mol)
    sims = np.array([TanimotoSimilarity(q_fp, fp) for fp in bundle.train_fps])
    max_sim = float(sims.max())
    out['max_tanimoto_to_training'] = round(max_sim, 3)
    out['ood_flag'] = max_sim < 0.50

    # GA-Tris always uses v5.2 hierarchy
    if q_tokens.family == 'GA-Tris':
        v52r = predict_v52(smiles, bundle.v52_bundle, architecture=architecture, family_hint=family_hint)
        out['pKa_pred'] = round(float(v52r.get('pKa_pred', np.nan)), 3)
        out['routing'] = 'v5.2_hierarchy_GA-Tris'
        ci = CI_TABLE['GA-Tris']['blend']
        out['pKa_PI_90'] = [round(out['pKa_pred']-ci, 2), round(out['pKa_pred']+ci, 2)]
        out['confidence_tier'] = 'MEDIUM'
        return out

    # Adaptive routing
    alpha = alpha_from_tan(max_sim)
    out['blend_alpha'] = round(alpha, 3)

    direct_pred = float(bundle.direct_xgb.predict(bundle.direct_scaler.transform(q_X.reshape(1,-1)))[0])
    out['direct_xgb_pred'] = round(direct_pred, 3)

    if alpha == 1.0:
        # Pure direct path - no good analog
        out['pKa_pred'] = round(direct_pred, 3)
        out['routing'] = 'PURE_DIRECT_no_close_analog'
        ci_band = 'pure_direct' if max_sim >= 0.50 else 'OOD'
        tier = 'LOW'
    elif alpha == 0.0:
        # Pure delta path - very close analog
        delta_pred, neighbors = _analog_delta(q_X, q_fp, bundle)
        out['analog_delta_pred'] = round(delta_pred, 3)
        out['nearest_neighbors'] = neighbors
        out['pKa_pred'] = round(delta_pred, 3)
        out['routing'] = 'PURE_ANALOG_DELTA_very_close'
        ci_band = 'pure_delta'
        tier = 'HIGH'
    else:
        # Blend path
        delta_pred, neighbors = _analog_delta(q_X, q_fp, bundle)
        out['analog_delta_pred'] = round(delta_pred, 3)
        out['nearest_neighbors'] = neighbors
        final = alpha * direct_pred + (1-alpha) * delta_pred
        out['pKa_pred'] = round(float(final), 3)
        out['routing'] = f'BLEND_alpha={alpha:.2f}'
        ci_band = 'blend'
        tier = 'HIGH' if max_sim >= 0.85 else 'MEDIUM'

    fam = q_tokens.family if q_tokens.family in CI_TABLE else 'default'
    ci = CI_TABLE[fam][ci_band]
    if out.get('warnings') and any('IMPUTED' in w for w in out['warnings']):
        ci *= 1.2  # widen CI when features were imputed
    out['pKa_PI_90'] = [round(out['pKa_pred']-ci, 2), round(out['pKa_pred']+ci, 2)]
    out['confidence_tier'] = tier
    return out


if __name__ == "__main__":
    print("Loading v7.2 bundle...")
    b = load_v72_bundle('IAJD_pKa_v21_final.xlsx')
    print(f"Loaded {len(b.train_y)} training compounds\n")
    queries = [
        ("A (93360)",  "CCCCC(CC)COc1cc(C(=O)OCCCCN2CCN(CCO)CC2)cc(OCC(CC)CCCC)c1OCC(CC)CCCC"),
        ("B (1348924)","CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"),
        ("C (1349219)","CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"),
        ("D (1351552)","CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"),
        ("E (1356072)","CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"),
    ]
    print(f"{'Compound':<14} {'pKa':>6}  {'90% CI':<14}  {'±':<6} {'maxTan':>7}  {'α':>5}  {'routing':<25} tier")
    print('-' * 110)
    for name, sm in queries:
        r = predict_pka_v72(sm, b, return_diagnostics=False)
        pi_lo, pi_hi = r['pKa_PI_90']; hw = (pi_hi-pi_lo)/2
        a = r.get('blend_alpha', 1.0)
        print(f"{name:<14} {r['pKa_pred']:>6.2f}  [{pi_lo:.2f}, {pi_hi:.2f}]  ±{hw:<5.2f} {r['max_tanimoto_to_training']:>7.3f}  {a:>5.2f}  {r['routing']:<25} {r['confidence_tier']}")
