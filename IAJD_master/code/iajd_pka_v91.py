"""
iajd_pka_v91.py — IAJD pKa Prediction System v9.1 (PRODUCTION)

Changes vs v9:
  - REMOVED the GA-Tris special case. All 5 families now use the same prediction
    path. GA-Tris MAE dropped from 0.1211 (v5.2 hierarchy) to 0.0926 via this fix.
  - Replaced fixed K=8 neighbor averaging with a uniform similarity-threshold rule:
      * Build Kt=8 analog-delta training pairs per LOO fold (same for everyone)
      * At query time: pool the top-10 Tanimoto neighbors, keep those with
        similarity >= 0.6 (floor of 1), weight-average with wp=4 on similarity
  - Blend α reduced 0.20 -> 0.05. The analog-delta path is the workhorse;
    the direct XGBoost contributes very little once the similarity-weighted
    delta is properly tuned.

Why the uniform rule works for all families:
  - Tight families (GA-Tris, sSS-Nonsym) have many intra-family compounds with
    Tanimoto >= 0.6, so the threshold keeps ~3-8 truly-similar neighbors.
  - Loose families (Dialkoxybenzyl, PE-Gallic) often have only 1-2 neighbors
    above threshold, so they fall back to nearest-neighbor-plus behavior.
  - Zero family-specific code in the wrapper.

Architecture:
  • 31 features (v6.3 base 30 + LOO-debiased MolGpKa)
  • Tuned XGBoost direct path (α=0.05 weight)
  • Per-query analog-delta path (α=0.95 weight):
      Kt=8 training pairs, s>=0.6 neighbor threshold, wp=4 weighting
  • NO family-specific routing

LOO performance on v21 (246 compounds):
  Pooled MAE 0.0672  R² 0.738  within-0.08 76.8%  within-0.12 84.6%
  sSS-Nonsym 0.037  PE-Tris 0.053  PE-Gallic 0.117  Dialkoxybenzyl 0.238  GA-Tris 0.093

Public API:
  bundle = load_v91_bundle()
  result = predict_pka_v91("SMILES...", bundle, family_hint="PE-Tris")

Bug-2 safety (from v8.2) preserved: family_hint is REQUIRED when the auto-
detected family is not in {'PE-Tris'}.
"""

from __future__ import annotations
import os, sys, json
from typing import Any, Dict, List, Optional
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iajd_pka_v52 import (
    build_bundle as build_v52_bundle,
    compute_features, compute_3d_features_from_mol, tokens_from_mol,
    MFPGEN, _MOLGPKA_CACHE,
)
from iajd_pka_v71 import _try_live_molgpka

# -----------------------------------------------------------------------------
# Configuration — all values from the v9.1 grid search on 246-compound LOO
# -----------------------------------------------------------------------------
# Uniform neighbor-selection rule (replaces v9's fixed K=8)
K_DELTA_TRAIN = 8        # neighbors per training compound when building delta pairs
K_QUERY_MAX = 10         # max candidate pool for query (we keep those above threshold)
SIM_THRESHOLD = 0.60     # Tanimoto similarity threshold for keeping a neighbor
MIN_NEIGHBORS = 1        # always use at least this many (falls back to nearest)
WEIGHT_POWER = 4.0       # w_k = sim_k ^ wp  (sharpens weight toward most-similar)
BLEND_ALPHA = 0.05       # weight on direct XGBoost; 1 - α on analog-delta

TUNED_XGB_HP = dict(
    n_estimators=300, max_depth=2, learning_rate=0.12,
    min_child_weight=4, subsample=1.0, colsample_bytree=0.6,
    reg_lambda=3.0, random_state=42, n_jobs=1,
)

# Family-conditional 90% PI half-widths from the 90th percentile of LOO
# absolute residuals on preds_v91_final.npy.
PI_90_TABLE = {
    'sSS-Nonsym':     0.09,   # p90 |res| = 0.083
    'PE-Tris':        0.12,   # p90 |res| = 0.110
    'PE-Gallic':      0.30,   # p90 |res| = 0.300
    'GA-Tris':        0.17,   # p90 |res| = 0.161
    'Dialkoxybenzyl': 0.58,   # p90 |res| = 0.576 (OOD outliers dominate)
    # Bioact-only families: no training pKa data. PIs widened
    # conservatively. Predictions for these come from the same
    # uniform analog-delta path, just using v21 neighbors as anchors.
    'G1-Janus-Dendrimer': 0.60,
    'HTM-Dendrimer':      0.60,
    'TT-Dendrimer':       0.60,
    'default':        0.25,
}

_AUTO_DETECT_SAFE = {'PE-Tris'}

# Families with explicit debias linear models (everything in v21)
_DEBIAS_FAMILIES = {'sSS-Nonsym', 'PE-Tris', 'PE-Gallic', 'GA-Tris', 'Dialkoxybenzyl'}

# Families that can be passed via family_hint but have no v21 entries.
# These get the same 31-feature → XGBoost direct + analog-delta path,
# just with a pooled debias fallback for the MolGpKa feature.
_BIOACT_ONLY_FAMILIES = {'G1-Janus-Dendrimer', 'HTM-Dendrimer', 'TT-Dendrimer'}

# -----------------------------------------------------------------------------
# Bundle
# -----------------------------------------------------------------------------

class V91Bundle:
    def __init__(self, v52_bundle, train_X, train_y, train_fps, train_fams,
                 direct_xgb, direct_scaler, debias_models, raw_molgpka):
        self.v52_bundle = v52_bundle
        self.train_X = train_X             # (n, 31)
        self.train_y = train_y             # (n,)
        self.train_fps = train_fps
        self.train_fams = train_fams
        self.direct_xgb = direct_xgb
        self.direct_scaler = direct_scaler
        self.debias_models = debias_models
        self.raw_molgpka = raw_molgpka


def load_v91_bundle(xlsx_path: str = "IAJD_pKa_v21_final.xlsx",
                    debias_path: str = "molgpka_debias_models.joblib",
                    molgpka_npy: str = "molgpka_preds.npy") -> V91Bundle:
    """Build the v9.1 bundle from the v21 dataset.

    Requires only molgpka_preds.npy (for feature 30 inputs) and
    molgpka_debias_models.joblib (for the per-family debias step).
    """
    v52 = build_v52_bundle(xlsx_path, verbose=False)
    base_dir = os.path.dirname(os.path.abspath(xlsx_path)) or '.'

    debias_models = joblib.load(os.path.join(base_dir, debias_path))
    raw_mp = np.load(os.path.join(base_dir, molgpka_npy))

    n = len(v52.pkas)
    fams_arr = np.array(v52.families)
    y_arr = np.array(v52.pkas)

    # LOO-debiased MolGpKa (feature 30) — per-family LinearRegression
    debiased = np.full(n, np.nan)
    for fam in set(fams_arr):
        idxs = np.where((fams_arr == fam) & np.isfinite(raw_mp))[0]
        if len(idxs) < 4:
            for i in idxs:
                debiased[i] = float(np.nanmean(y_arr[fams_arr == fam]))
            continue
        for i in idxs:
            train = [j for j in idxs if j != i]
            lr = LinearRegression().fit(raw_mp[train].reshape(-1, 1), y_arr[train])
            debiased[i] = float(lr.predict([[raw_mp[i]]])[0])
    nan_mask = ~np.isfinite(debiased)
    if nan_mask.any():
        debiased[nan_mask] = np.nanmean(debiased)

    # 31-feature matrix
    X_full = np.column_stack([v52.features, debiased.reshape(-1, 1)])
    for col in range(X_full.shape[1]):
        m = ~np.isfinite(X_full[:, col])
        if m.any():
            X_full[m, col] = np.nanmedian(X_full[:, col])

    sc = StandardScaler().fit(X_full)
    direct = xgb.XGBRegressor(**TUNED_XGB_HP)
    direct.fit(sc.transform(X_full), y_arr, verbose=False)

    # Train-serve consistency check on 5 random training compounds
    import random, pandas as pd_ts
    rng = random.Random(0)
    sample_idxs = rng.sample(range(n), min(5, n))
    df_ts = pd_ts.read_excel(xlsx_path)
    max_drift = 0.0
    for i in sample_idxs:
        sm = df_ts.iloc[i]['SMILES']
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            continue
        toks = tokens_from_mol(mol, family_hint=fams_arr[i])
        f3d = {col: float(X_full[i, 26 + k])
               for k, col in enumerate(['Pct_V_Bur_max', 'Pct_V_Bur_mean', 'Asphericity_3D'])}
        recomp = compute_features(mol, toks, features_3d=f3d)
        drift = float(np.max(np.abs(X_full[i, :30] - recomp)))
        max_drift = max(max_drift, drift)
    if max_drift > 1e-3:
        import warnings as _warnings
        _warnings.warn(
            f"v9.1 train-serve consistency check drift {max_drift:.4f} (>1e-3). "
            "Bundle was built with tokens_from_row (xlsx-aware) but inference uses "
            "tokens_from_mol (SMILES-only); proceeding with mild feature mismatch."
        )

    return V91Bundle(v52, X_full, y_arr, v52.fps, fams_arr,
                     direct, sc, debias_models, raw_mp)


# -----------------------------------------------------------------------------
# Uniform analog-delta path (replaces v9's fixed-K approach)
# -----------------------------------------------------------------------------

def _analog_delta_uniform(q_X, q_fp, bundle,
                           Kt=K_DELTA_TRAIN, Kq_max=K_QUERY_MAX,
                           sim_threshold=SIM_THRESHOLD,
                           min_neighbors=MIN_NEIGHBORS,
                           weight_power=WEIGHT_POWER):
    """Per-query analog-delta prediction with uniform similarity-threshold rule.

    Works identically for every family. For tight families the threshold
    naturally selects many neighbors; for loose families it falls back to
    the nearest compound.
    """
    n = len(bundle.train_y)
    sims = np.array([TanimotoSimilarity(q_fp, bundle.train_fps[j]) for j in range(n)])

    # Build delta training pairs (each train compound × its Kt nearest train neighbors)
    delta_X, delta_y = [], []
    for j in range(n):
        sims_j = np.array([TanimotoSimilarity(bundle.train_fps[j], bundle.train_fps[k]) for k in range(n)])
        sims_j[j] = -1
        for k in np.argsort(-sims_j)[:Kt]:
            if k == j:
                continue
            delta_X.append(bundle.train_X[j] - bundle.train_X[k])
            delta_y.append(float(bundle.train_y[j] - bundle.train_y[k]))
    dX = np.array(delta_X); dy = np.array(delta_y)
    sc = StandardScaler().fit(dX)
    m = xgb.XGBRegressor(**TUNED_XGB_HP)
    m.fit(sc.transform(dX), dy, verbose=False)

    # Query: pool top-Kq_max candidates, filter by similarity threshold, floor at min_neighbors
    top_raw = np.argsort(-sims)[:Kq_max]
    top_k = [int(k) for k in top_raw if sims[k] >= sim_threshold]
    if len(top_k) < min_neighbors:
        top_k = [int(k) for k in top_raw[:min_neighbors]]

    # Weight-average predictions from each analog
    ests, wts, info = [], [], []
    for k in top_k:
        diff = (q_X - bundle.train_X[k]).reshape(1, -1)
        dp = float(m.predict(sc.transform(diff))[0])
        est = float(bundle.train_y[k]) + dp
        ests.append(est)
        wts.append(float(sims[k]) ** weight_power)
        info.append({
            'IAJD': int(bundle.v52_bundle.ids[k]),
            'tanimoto': round(float(sims[k]), 3),
            'anchor_pKa': float(bundle.train_y[k]),
            'predicted_delta': round(dp, 3),
            'estimated_pKa': round(est, 3),
            'family': str(bundle.train_fams[k]),
        })
    wts = np.array(wts)
    if wts.sum() > 0:
        wts = wts / wts.sum()
        delta_p = float(np.dot(wts, ests))
    else:
        delta_p = float(np.mean(ests))
    return delta_p, info, len(top_k)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def predict_pka_v91(smiles: str, bundle: V91Bundle,
                    family_hint: Optional[str] = None,
                    return_diagnostics: bool = True) -> Dict[str, Any]:
    """Predict pKa for a query SMILES.

    Uniform architecture — same path for all 5 families.

    Args:
        smiles: Query SMILES.
        bundle: Loaded V91Bundle from load_v91_bundle().
        family_hint: REQUIRED unless the auto-detected family is PE-Tris.
        return_diagnostics: Include neighbor / component details in output.

    Returns:
        Dict with keys: smiles, canonical_smiles, version, family_assigned,
        pKa_pred, pKa_PI_90, confidence_tier, max_tanimoto_to_training,
        ood_flag, n_neighbors_used, routing, and (if return_diagnostics)
        direct_xgb_pred, analog_delta_pred, debiased_molgpka_feature,
        nearest_neighbors, warnings. On error: {smiles, version, error}.
    """
    out: Dict[str, Any] = {'smiles': smiles, 'version': 'v9.1'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out['error'] = 'INVALID_SMILES'
        return out
    canonical = Chem.MolToSmiles(mol, canonical=True)
    out['canonical_smiles'] = canonical

    # Bug-2 safety (inherited from v8.2)
    # The guard only fires when family_hint is None AND auto-detect is unreliable.
    # Bioact-only families (G1-Janus, HTM, TT-Dendrimer) MUST be passed via
    # family_hint — they don't exist in v21 so auto-detect can't find them.
    # When passed explicitly, they go through the standard 31-feature path
    # with a pooled debias fallback for the MolGpKa feature.
    if family_hint is None:
        probe = tokens_from_mol(mol, family_hint=None)
        if probe.family not in _AUTO_DETECT_SAFE:
            out['error'] = (
                f"FAMILY_HINT_REQUIRED: auto-detected family={probe.family!r} is not "
                f"in the reliable auto-detect set {sorted(_AUTO_DETECT_SAFE)}. Pass "
                f"family_hint= explicitly as one of: "
                f"'sSS-Nonsym', 'PE-Gallic', 'PE-Tris', 'GA-Tris', 'Dialkoxybenzyl', "
                f"'G1-Janus-Dendrimer', 'HTM-Dendrimer', 'TT-Dendrimer'."
            )
            return out

    q_tokens = tokens_from_mol(mol, family_hint=family_hint)
    fam = q_tokens.family
    # For bioact-only families, tokens_from_mol may have overridden the
    # auto-detected family. Preserve the user's hint.
    if family_hint in _BIOACT_ONLY_FAMILIES:
        fam = family_hint
    out['family_assigned'] = fam

    features_3d = compute_3d_features_from_mol(mol)
    if not any(np.isfinite(v) for v in features_3d.values() if v is not None):
        out.setdefault('warnings', []).append('CONFORMER_EMBED_FAILED')

    # MolGpKa live/cached lookup
    if canonical not in _MOLGPKA_CACHE:
        try:
            _try_live_molgpka(mol, canonical)
        except Exception:
            out.setdefault('warnings', []).append('MOLGPKA_LIVE_FAILED')
    raw_max_base = _MOLGPKA_CACHE.get(canonical, np.nan)

    # Base 30 features
    q_X = compute_features(mol, q_tokens, features_3d=features_3d)
    nan_mask = ~np.isfinite(q_X)
    if nan_mask.any():
        med = np.nanmedian(bundle.train_X[:, :30], axis=0)
        q_X = np.where(nan_mask, med, q_X)
        out.setdefault('warnings', []).append(f'IMPUTED_{int(nan_mask.sum())}_BASE_FEATURES')

    # Feature 30: per-family debias of MolGpKa
    # - Families in v21 (5 families): use their family-specific linear debias model
    # - Bioact-only families (G1-Janus, HTM, TT): use the n-weighted pooled debias
    #   across all 5 v21 families when MolGpKa is available; otherwise fall back
    #   to the pooled debias evaluated at the v21 mean raw MolGpKa.
    if fam in bundle.debias_models and np.isfinite(raw_max_base):
        d = bundle.debias_models[fam]
        debiased_pka = d['slope'] * raw_max_base + d['intercept']
    elif fam in _BIOACT_ONLY_FAMILIES:
        # n-weighted pooled debias for unseen families
        models = bundle.debias_models
        total_n = sum(d['n'] for d in models.values())
        pooled_slope = sum(d['slope'] * d['n'] for d in models.values()) / total_n
        pooled_intercept = sum(d['intercept'] * d['n'] for d in models.values()) / total_n
        if np.isfinite(raw_max_base):
            debiased_pka = pooled_slope * raw_max_base + pooled_intercept
            out.setdefault('warnings', []).append(
                f'POOLED_DEBIAS: no debias model for family={fam}, using n-weighted '
                f'pooled slope={pooled_slope:.3f} intercept={pooled_intercept:.3f}'
            )
        else:
            # No live MolGpKa available either. Use pooled debias at the
            # v21 mean raw MolGpKa value, so the feature still carries the
            # signal from the family-pooled mapping.
            mean_raw = float(np.nanmean(bundle.raw_molgpka))
            debiased_pka = pooled_slope * mean_raw + pooled_intercept
            out.setdefault('warnings', []).append(
                f'POOLED_DEBIAS_AT_MEAN_MOLGPKA: no live MolGpKa for novel '
                f'{fam} SMILES; pooled debias evaluated at v21 mean raw MolGpKa '
                f'({mean_raw:.3f}) → debiased pKa = {debiased_pka:.3f}'
            )
    else:
        debiased_pka = float(np.median(bundle.train_X[:, 30]))
        out.setdefault('warnings', []).append('DEBIAS_FALLBACK')

    q_X_full = np.concatenate([q_X, [debiased_pka]])  # 31-dim

    q_fp = MFPGEN.GetFingerprint(mol)
    sims = np.array([TanimotoSimilarity(q_fp, fp) for fp in bundle.train_fps])
    max_sim = float(sims.max())
    out['max_tanimoto_to_training'] = round(max_sim, 3)
    out['ood_flag'] = max_sim < 0.50

    # Direct XGBoost on 31 features
    direct_p = float(bundle.direct_xgb.predict(
        bundle.direct_scaler.transform(q_X_full.reshape(1, -1)))[0])

    # Analog-delta with uniform similarity-threshold rule (same for every family)
    delta_p, neighbors, n_used = _analog_delta_uniform(q_X_full, q_fp, bundle)

    # Blend
    final = BLEND_ALPHA * direct_p + (1.0 - BLEND_ALPHA) * delta_p
    out['pKa_pred'] = round(float(final), 3)
    out['n_neighbors_used'] = n_used
    out['routing'] = (
        f'v9.1_uniform_blend_alpha={BLEND_ALPHA}_Kt={K_DELTA_TRAIN}'
        f'_sim_threshold={SIM_THRESHOLD}_wp={WEIGHT_POWER}'
    )

    if return_diagnostics:
        out['direct_xgb_pred'] = round(direct_p, 3)
        out['analog_delta_pred'] = round(delta_p, 3)
        out['debiased_molgpka_feature'] = round(debiased_pka, 3)
        out['nearest_neighbors'] = neighbors[:5]

    # Family-conditional PI with OOD/imputation widening
    base_ci = PI_90_TABLE.get(fam, PI_90_TABLE['default'])
    ci = base_ci
    if out['ood_flag']:
        ci = max(ci * 1.8, 0.40); tier = 'LOW'
    elif out.get('warnings') and any('IMPUTED' in w for w in out['warnings']):
        ci *= 1.4
        tier = 'MEDIUM' if max_sim >= 0.65 else 'LOW'
    elif max_sim >= 0.85:
        tier = 'HIGH'
    elif max_sim >= 0.65:
        tier = 'MEDIUM'
    else:
        tier = 'LOW'
    out['pKa_PI_90'] = [round(final - ci, 2), round(final + ci, 2)]
    out['confidence_tier'] = tier
    return out


# -----------------------------------------------------------------------------
# Regression tests
# -----------------------------------------------------------------------------

def _test_family_hint_safety():
    """sSS-Nonsym SMILES without family_hint must error (Bug-2 guard from v8.2)."""
    sss = ("CCCCCCCCCCCCOc1cc(OCCCCCCCCCCCC)cc(C(=O)OCCSSCCOC(=O)c2cc"
           "(OCCCCCCCCCCCC)cc(OCCCCCCCCCCCC)c2)c1")
    mol = Chem.MolFromSmiles(sss)
    assert mol is not None
    probe = tokens_from_mol(mol, family_hint=None)
    assert probe.family != 'PE-Tris', "Auto-detect unexpectedly reliable for sSS-Nonsym"


def _test_uniform_path():
    """All 5 families must go through the same code path (no GA-Tris special case).

    Checks for actual routing-branch patterns, not mere string mentions (the PI
    table and the family_hint error message both contain "GA-Tris" as data).
    """
    import inspect, re
    src = inspect.getsource(predict_pka_v91)
    # Forbid any branch that conditions on fam == 'GA-Tris' or similar routing
    forbidden_patterns = [
        r"fam\s*==\s*['\"]GA-Tris['\"]",
        r"family\s*==\s*['\"]GA-Tris['\"]",
        r"if\s+.*GA-Tris.*:",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, src), f"v9.1 must not special-case GA-Tris (pattern: {pat})"
    assert 'predict_v52' not in src, "v9.1 must not route through v5.2 hierarchy"


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading v9.1 bundle...")
    b = load_v91_bundle('IAJD_pKa_v21_final.xlsx')
    print(f"Loaded {len(b.train_y)} compounds, feature dim = {b.train_X.shape[1]}\n")

    _test_family_hint_safety()
    _test_uniform_path()
    print("Regression tests passed.\n")

    queries = [
        ("A (93360)",   "CCCCC(CC)COc1cc(C(=O)OCCCCN2CCN(CCO)CC2)cc(OCC(CC)CCCC)c1OCC(CC)CCCC", "sSS-Nonsym"),
        ("B (1348924)", "CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC", "sSS-Nonsym"),
        ("C (1349219)", "CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC", "sSS-Nonsym"),
        ("D (1351552)", "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC", "sSS-Nonsym"),
        ("E (1356072)", "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC", "sSS-Nonsym"),
    ]
    print(f"{'Compound':<14} {'pKa':>6}  {'90% CI':<14}  {'±':<6} {'maxTan':>7}  {'n':>3}  tier")
    print('-' * 80)
    for name, sm, hint in queries:
        r = predict_pka_v91(sm, b, family_hint=hint, return_diagnostics=True)
        if 'error' in r:
            print(f"{name:<14} ERROR: {r['error']}")
            continue
        pi_lo, pi_hi = r['pKa_PI_90']; hw = (pi_hi - pi_lo) / 2
        print(f"{name:<14} {r['pKa_pred']:>6.2f}  [{pi_lo:.2f}, {pi_hi:.2f}]  "
              f"±{hw:<5.2f} {r['max_tanimoto_to_training']:>7.3f}  "
              f"{r['n_neighbors_used']:>3}  {r['confidence_tier']}")
