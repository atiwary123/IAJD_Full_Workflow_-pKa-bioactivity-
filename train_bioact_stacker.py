"""
train_bioact_stacker.py — one-shot trainer for the bioactivity blender.

Fits, on the full 335-compound v14 training set:
    direct_head_stacker  : XGB on all 88 features  (n_est=400 — matches LOO HP)
    lion_head_stacker    : XGB on LION block       (n_est=300)
    admet_head_stacker   : XGB on ADMET block      (n_est=300)
    stacker_xgb          : XGB on [direct_loo, analog_loo, lion_loo, admet_loo] -> y
                            (n_est=300, depth=3, lr=0.03 — chosen by 5-fold CV in
                            weight_eval_v2_results.json: DALM features, MAE=0.3856)

Saves -> IAJD_master/bundles_caches/bioact_stacker_bundle.pkl with keys:w
    {direct_head, lion_head, admet_head, stacker, block_lion, block_admet,
     train_metrics: {loo_cv_mae, baseline_mae}, version}
"""
from __future__ import annotations
import pickle
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "IAJD_master" / "bundles_caches" / "bioact_v14_bundle.pkl"
LOO_NPZ = HERE / "bioact_loo_components.npz"
OUT_PATH = HERE / "IAJD_master" / "bundles_caches" / "bioact_stacker_bundle.pkl"

BLOCK_LION = slice(50, 64)
BLOCK_ADMET = slice(64, 74)

DIRECT_HP = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                 random_state=42, n_jobs=1)
BLOCK_HP = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=1)
STACKER_HP = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                  random_state=42, n_jobs=1)


def main():
    print(f"Loading v14 bundle: {BUNDLE}")
    with open(BUNDLE, "rb") as f:
        b = pickle.load(f)
    X = np.asarray(b["X_train"], dtype=float)
    y = np.asarray(b["y_train"], dtype=float)
    n = X.shape[0]
    print(f"  n={n} X.shape={X.shape}")

    print("Training stacker-aligned heads on full 335 rows...")
    direct_head = xgb.XGBRegressor(**DIRECT_HP).fit(X, y, verbose=False)
    lion_head   = xgb.XGBRegressor(**BLOCK_HP).fit(X[:, BLOCK_LION], y, verbose=False)
    admet_head  = xgb.XGBRegressor(**BLOCK_HP).fit(X[:, BLOCK_ADMET], y, verbose=False)
    print("  ✓ direct, lion, admet heads fit.")

    # Train the stacker on LOO out-of-fold predictions to avoid leakage.
    print(f"Loading LOO component predictions from {LOO_NPZ}...")
    d = np.load(LOO_NPZ, allow_pickle=True)
    direct_loo = d["direct"]; analog_loo = d["analog"]
    lion_loo = d["lion"]; admet_loo = d["admet"]
    y_loo = d["y_true"]
    assert np.allclose(y_loo, y), "LOO y_true must match bundle y_train"

    X_stack = np.column_stack([direct_loo, analog_loo, lion_loo, admet_loo])
    stacker = xgb.XGBRegressor(**STACKER_HP).fit(X_stack, y, verbose=False)
    print("  ✓ stacker fit on LOO OOF predictions.")

    # Sanity: in-sample MAE on the LOO matrix
    in_sample_mae = float(np.mean(np.abs(stacker.predict(X_stack) - y)))
    print(f"  stacker in-sample MAE (on LOO OOF): {in_sample_mae:.4f}")

    bundle_out = {
        "version": "bioact_stacker_v1",
        "direct_head": direct_head,
        "lion_head": lion_head,
        "admet_head": admet_head,
        "stacker": stacker,
        "block_lion": (BLOCK_LION.start, BLOCK_LION.stop),
        "block_admet": (BLOCK_ADMET.start, BLOCK_ADMET.stop),
        "stack_features": ["direct", "analog", "lion", "admet"],
        "training_hp": {
            "direct": DIRECT_HP, "block": BLOCK_HP, "stacker": STACKER_HP,
        },
        "train_metrics": {
            "n_train": n,
            "stacker_in_sample_mae_on_loo": in_sample_mae,
            "5fold_cv_mae": 0.3856,   # from weight_eval_v2_results.json
            "v14_baseline_loo_mae": 0.4010,
            "improvement": round(0.4010 - 0.3856, 4),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(bundle_out, f)
    print(f"Saved -> {OUT_PATH}")
    print(f"  expected production MAE ≈ 0.3856 (vs 0.4010 baseline)  Δ = -0.0154")


if __name__ == "__main__":
    main()
