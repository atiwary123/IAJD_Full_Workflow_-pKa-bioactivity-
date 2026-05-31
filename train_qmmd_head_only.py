"""
train_qmmd_head_only.py — train just the qmmd head against the current Block
D' cache state, without the full LOO retrain.

This is a sanity-check script for W-D: it validates that the stacker can
consume Block D' features end-to-end, even when most rows are NaN-passthrough
(which is the expected state until the QM/MD caches finish populating).

Saves the qmmd head into a small joblib so the prediction pipeline can
exercise it via _qmmd_predict_single without a full stacker bundle.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.06,
    min_child_weight=4, subsample=0.85, colsample_bytree=0.7,
    reg_lambda=3.0, random_state=42, n_jobs=2, verbosity=0,
    tree_method="hist", objective="reg:squarederror",
)


def main():
    qmmd_block = ROOT / "qmmd_features_v14_train.npy"
    bundle_path = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
    if not qmmd_block.exists() or not bundle_path.exists():
        print(f"ERROR: missing inputs.\n  {qmmd_block} -> {qmmd_block.exists()}\n  {bundle_path} -> {bundle_path.exists()}")
        return 1
    with open(bundle_path, "rb") as f:
        b = pickle.load(f)
    y_all = np.asarray(b["y_train"], dtype=float)
    X = np.load(qmmd_block)
    print(f"qmmd block: {X.shape}, NaN={np.isnan(X).mean():.2%}")
    print(f"target: {len(y_all)} rows, mean={y_all.mean():.3f}, std={y_all.std():.3f}")

    # 5-fold CV — XGBoost handles NaN natively.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y_all)
    for tr, te in kf.split(X):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], y_all[tr])
        preds[te] = m.predict(X[te])
    mae = float(mean_absolute_error(y_all, preds))
    r2 = float(r2_score(y_all, preds))
    print(f"\nqmmd head 5-fold CV: MAE={mae:.4f}  R²={r2:+.3f}")
    print("(With current cache sparsity this is mostly the global mean — "
          "real predictive signal arrives as the cache fills.)")

    # Final model on all data.
    final = XGBRegressor(**XGB_PARAMS)
    final.fit(X, y_all)
    out_path = ROOT / "IAJD_master/bundles_caches/physics/qmmd_head.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_names": [  # mirror physics_cache_io.BLOCK_DPRIME_KEYS
            "md_a_head_prot_nm2", "md_delta_a_head_nm2", "md_cpp_prot", "md_delta_cpp",
            "md_bilayer_thick_nm", "md_order_param", "md_assembles", "md_n_agg",
            "qm_q_ionizableN", "qm_dipole_D", "qm_dGsolv_kJmol", "qm_homo_lumo_eV",
            "dG_escape_helfrich", "head_area_nm2",
        ],
        "n_train": int(len(y_all)),
        "cv_mae": mae, "cv_r2": r2,
        "nan_fraction": float(np.isnan(X).mean()),
    }, out_path)
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
