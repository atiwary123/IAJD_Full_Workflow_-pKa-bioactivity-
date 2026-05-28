"""
iajd_pka_v71.py — IAJD pKa Prediction System v7.1 (PRODUCTION)

Architecture:
    Final prediction = 0.60 * direct_xgb_pred + 0.40 * analog_delta_pred
where:
    direct_xgb_pred  - tuned XGBoost on 30 features (the v6.3 model)
    analog_delta_pred - K=8 Tanimoto-nearest training analogs, each with a
                       per-pair XGBoost-predicted delta from query, weighted
                       by similarity

GA-Tris compounds use the v5.2 analog hierarchy directly (n=11 too small for
gradient boosting). Other families use the blend.

Returns prediction, 90% PI (family-conditional from LOO error distribution),
OOD flag, nearest neighbors, and per-component breakdown.

Public API:
    bundle = load_v71_bundle()
    result = predict_pka_v71("SMILES...", bundle)
"""

from __future__ import annotations
import os, sys, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb

# Use the v52 module for feature computation, MolGpKa cache, and the GA-Tris hierarchy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iajd_pka_v52 import (
    build_bundle as build_v52_bundle,
    compute_features,
    compute_3d_features_from_mol,
    tokens_from_mol,
    MFPGEN,
    N_FEATURES,
    select_level,
    compute_weights,
    predict_pka as predict_v52,
    _MOLGPKA_CACHE,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

V71_K = 8                # number of Tanimoto neighbors for analog-delta
V71_ALPHA = 0.60         # blend: alpha * direct + (1-alpha) * delta
V71_OOD_THRESHOLD = 0.50 # max-Tanimoto below this -> OOD flag

# Family-conditional 90% PI half-widths from v7.1 LOO errors
FAMILY_PI_90 = {
    "PE-Tris":         0.18,
    "GA-Tris":         0.16,
    "sSS-Nonsym":      0.24,
    "PE-Gallic":       0.32,
    "Dialkoxybenzyl":  0.50,
    "default":         0.30,
}

XGB_HP = dict(
    n_estimators=200, max_depth=3, learning_rate=0.08,
    subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
    min_child_weight=5, random_state=42, n_jobs=1,
)


# -----------------------------------------------------------------------------
# Bundle: load training data + direct XGBoost once
# -----------------------------------------------------------------------------

@dataclass
class V71Bundle:
    v52_bundle: Any                  # v5.2 bundle (carries tokens, fps, features, GA-Tris hierarchy)
    train_X: np.ndarray              # (n, 30) training features
    train_y: np.ndarray              # (n,) training pKa
    train_fps: List                  # Morgan fingerprints
    train_fams: np.ndarray
    direct_xgb: xgb.XGBRegressor
    direct_scaler: StandardScaler


def load_v71_bundle(xlsx_path: str = "IAJD_pKa_v21_final.xlsx") -> V71Bundle:
    """Build the v7.1 bundle from the v21 dataset."""
    v52 = build_v52_bundle(xlsx_path, verbose=False)
    X = v52.features
    y = np.array(v52.pkas)
    sc = StandardScaler().fit(X)
    direct = xgb.XGBRegressor(**XGB_HP)
    direct.fit(sc.transform(X), y, verbose=False)
    return V71Bundle(
        v52_bundle=v52, train_X=X, train_y=y, train_fps=v52.fps,
        train_fams=np.array(v52.families), direct_xgb=direct, direct_scaler=sc,
    )


# -----------------------------------------------------------------------------
# Per-query analog-delta prediction
# -----------------------------------------------------------------------------

def _analog_delta_predict(
    q_X: np.ndarray, q_fp, bundle: V71Bundle, K: int = V71_K
) -> Tuple[float, List[Dict]]:
    """Train a per-query delta model and return blended analog prediction.

    Returns (predicted pKa, list of K neighbor dicts with anchor + delta info).
    """
    n = len(bundle.train_y)
    sims = np.array([TanimotoSimilarity(q_fp, bundle.train_fps[j]) for j in range(n)])
    top_k = np.argsort(-sims)[:K]
    analog_sims = sims[top_k]

    # Build training pairs: each training compound paired with its K nearest training neighbors
    delta_X_train: List[np.ndarray] = []
    delta_y_train: List[float] = []
    for j in range(n):
        sims_j = np.array([TanimotoSimilarity(bundle.train_fps[j], bundle.train_fps[k]) for k in range(n)])
        sims_j[j] = -1
        nbrs = np.argsort(-sims_j)[:K]
        for k in nbrs:
            if k == j:
                continue
            delta_X_train.append(bundle.train_X[j] - bundle.train_X[k])
            delta_y_train.append(float(bundle.train_y[j] - bundle.train_y[k]))

    dX = np.array(delta_X_train)
    dy = np.array(delta_y_train)
    sc = StandardScaler().fit(dX)
    model = xgb.XGBRegressor(**XGB_HP)
    model.fit(sc.transform(dX), dy, verbose=False)

    # Predict delta from query to each analog, estimate pKa, weight-average
    estimates = []
    weights = []
    neighbor_info = []
    for a_idx, a_sim in zip(top_k, analog_sims):
        dq = (q_X - bundle.train_X[a_idx]).reshape(1, -1)
        delta_pred = float(model.predict(sc.transform(dq))[0])
        est = float(bundle.train_y[a_idx]) + delta_pred
        estimates.append(est)
        weights.append(float(a_sim))
        neighbor_info.append({
            "IAJD": int(bundle.v52_bundle.ids[a_idx]),
            "tanimoto": round(float(a_sim), 3),
            "anchor_pKa": float(bundle.train_y[a_idx]),
            "predicted_delta": round(delta_pred, 3),
            "estimated_pKa": round(est, 3),
            "family": str(bundle.train_fams[a_idx]),
        })

    weights = np.array(weights)
    if weights.sum() <= 0:
        return float(np.mean(estimates)), neighbor_info
    weights = weights / weights.sum()
    return float(np.dot(weights, estimates)), neighbor_info


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def predict_pka_v71(
    smiles: str,
    bundle: V71Bundle,
    architecture: Optional[str] = None,
    family_hint: Optional[str] = None,
    return_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Predict pKa for a query SMILES using the v7.1 blended architecture."""
    out: Dict[str, Any] = {"smiles": smiles, "version": "v7.1"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["error"] = "INVALID_SMILES"
        return out

    canonical = Chem.MolToSmiles(mol, canonical=True)
    out["canonical_smiles"] = canonical

    # Compute query features (30-dim, including 3D conformer + MolGpKa)
    q_tokens = tokens_from_mol(mol, architecture=architecture, family_hint=family_hint)
    out["family_assigned"] = q_tokens.family
    out["arch_coarse_assigned"] = q_tokens.arch_coarse

    features_3d = compute_3d_features_from_mol(mol)
    if not any(np.isfinite(v) for v in features_3d.values() if v is not None):
        out.setdefault("warnings", []).append("CONFORMER_EMBED_FAILED")

    # MolGpKa (live call if not in cache; the v52 module's compute_features will inject NaN if missing)
    canon_for_cache = canonical
    if canon_for_cache not in _MOLGPKA_CACHE:
        try:
            _try_live_molgpka(mol, canon_for_cache)
        except Exception:
            out.setdefault("warnings", []).append("MOLGPKA_LIVE_FAILED")

    q_X = compute_features(mol, q_tokens, features_3d=features_3d)
    # Impute any remaining NaNs with training median
    nan_mask = ~np.isfinite(q_X)
    if nan_mask.any():
        med = np.nanmedian(bundle.train_X, axis=0)
        q_X = np.where(nan_mask, med, q_X)
        out.setdefault("warnings", []).append(f"IMPUTED_{int(nan_mask.sum())}_FEATURES")

    q_fp = MFPGEN.GetFingerprint(mol)

    # Compute Tanimoto distances and OOD check
    sims = np.array([TanimotoSimilarity(q_fp, fp) for fp in bundle.train_fps])
    max_sim = float(sims.max())
    out["max_tanimoto_to_training"] = round(max_sim, 3)
    out["ood_flag"] = max_sim < V71_OOD_THRESHOLD

    # GA-Tris uses the v5.2 hierarchy entirely
    if q_tokens.family == "GA-Tris":
        v52_result = predict_v52(smiles, bundle.v52_bundle,
                                  architecture=architecture, family_hint=family_hint)
        out["pKa_pred"] = round(float(v52_result.get("pKa_pred", np.nan)), 3)
        out["prediction_path"] = "v5.2_hierarchy_GA-Tris"
        out["confidence_tier"] = v52_result.get("confidence_tier", "MEDIUM")
        pi = FAMILY_PI_90.get("GA-Tris", FAMILY_PI_90["default"])
        out["pKa_PI_90"] = [round(out["pKa_pred"] - pi, 2),
                            round(out["pKa_pred"] + pi, 2)]
        if return_diagnostics:
            out["v52_details"] = {k: v for k, v in v52_result.items()
                                  if k in ("similarity_level", "anchor_IAJD", "delta_value")}
        return out

    # Direct XGBoost prediction
    direct_pred = float(bundle.direct_xgb.predict(bundle.direct_scaler.transform(q_X.reshape(1, -1)))[0])
    out["direct_xgb_pred"] = round(direct_pred, 3)

    # Analog-delta prediction
    delta_pred, neighbors = _analog_delta_predict(q_X, q_fp, bundle, K=V71_K)
    out["analog_delta_pred"] = round(delta_pred, 3)
    out["nearest_neighbors"] = neighbors

    # Blend
    final = V71_ALPHA * direct_pred + (1.0 - V71_ALPHA) * delta_pred
    out["pKa_pred"] = round(float(final), 3)
    out["prediction_path"] = f"blend_alpha={V71_ALPHA}_K={V71_K}"

    # Confidence tier and PI
    pi = FAMILY_PI_90.get(q_tokens.family, FAMILY_PI_90["default"])
    if out["ood_flag"]:
        pi = max(pi * 1.5, 0.50)
        tier = "LOW"
    elif max_sim >= 0.85:
        tier = "HIGH"
    elif max_sim >= 0.65:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    out["confidence_tier"] = tier
    out["pKa_PI_90"] = [round(final - pi, 2), round(final + pi, 2)]

    return out


def _try_live_molgpka(mol, canon_smiles):
    """Compute MolGpKa pKa for a novel molecule via the local Xundrug/MolGpKa
    install (molgpka_src/ + molgpka_models/) and add to the v52 cache.

    No proxy fallback — if MolGpKa can't run, the cache entry stays unset and
    the downstream debias step uses the family-pooled median, which is the
    standard handling for "no MolGpKa available" already implemented in
    iajd_pka_v91.load_v91_bundle.
    """
    # Resolve project root from this file: …/IAJD_master/code/iajd_pka_v71.py
    proj_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."
    ))
    molgpka_src = os.path.join(proj_root, "molgpka_src")
    if not os.path.isdir(molgpka_src):
        return
    if molgpka_src not in sys.path:
        sys.path.insert(0, molgpka_src)
    try:
        from predict_pka import predict as molgpka_predict  # type: ignore
        base_dict, _ = molgpka_predict(mol)
        if base_dict:
            _MOLGPKA_CACHE[canon_smiles] = float(max(base_dict.values()))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# CLI / smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading v7.1 bundle...")
    b = load_v71_bundle("IAJD_pKa_v21_final.xlsx")
    print(f"Loaded {len(b.train_y)} training compounds")

    test_smiles = [
        # In-distribution: PE-Tris C8 EH MPRZ (similar to v20 IAJD 248-class)
        "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(C)CC1",
        # In-distribution: PE-Gallic singleton
        "O=C(OCC(COC(=O)c1cc(OCCCCCCCCCCCC)cc(OCCCCCCCCCCCC)c1)(COC(=O)c1cc(OCCCCCCCCCCCC)cc(OCCCCCCCCCCCC)c1)COC(=O)c1cc(OCCCCCCCCCCCC)cc(OCCCCCCCCCCCC)c1)CCCN1CCN(C)CC1",
    ]
    for sm in test_smiles:
        r = predict_pka_v71(sm, b, return_diagnostics=False)
        print()
        print(f"SMILES: {sm[:60]}...")
        print(f"  pKa_pred: {r.get('pKa_pred'):.2f}  PI90: {r.get('pKa_PI_90')}  tier: {r.get('confidence_tier')}")
        print(f"  family: {r.get('family_assigned')}  max_tanimoto: {r.get('max_tanimoto_to_training')}  ood: {r.get('ood_flag')}")
        print(f"  direct={r.get('direct_xgb_pred')}  delta={r.get('analog_delta_pred')}")
