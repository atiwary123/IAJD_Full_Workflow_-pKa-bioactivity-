"""
train_bioact_ensemble.py — deep-ensemble for per-candidate Bayesian uncertainty.

Trains M XGBoost regressors with different random seeds + bootstrap samples of
the training data, all on the v14 92-feature stack (Block A + B + C + D +
formulation + E_sampleprep). At inference, mean of the M predictions gives the
point estimate and std gives the per-candidate uncertainty σ — no Gaussian
distributional assumption with a constant σ, no proxy.

Also computes:
  • per-family LOO RMSE (for σ fallback when ensemble σ is unreliable)
  • global σ calibration factor (multiplier that aligns mean-ensemble-σ to
    actual LOO RMSE) so the per-candidate σ is honestly calibrated

Outputs:
  IAJD_master/bundles_caches/bioact_ensemble_bundle.pkl
    {
      models: [XGBRegressor × M],            # full-data bootstrap ensemble
      seeds: [int × M],
      bootstrap_sizes: [int × M],
      sigma_calibration: float,              # multiplier to scale ensemble σ
      per_family_sigma: {family: float},     # LOO RMSE per family
      global_sigma: float,                   # LOO RMSE overall
      block_slices: {…},
      n_features: 92,
      training_size: int,
      metrics: {
        loo_mae_ensemble_mean: float,
        ensemble_mean_sigma: float,
        actual_residual_std: float,
        calibration_factor: float,
      },
    }
"""
from __future__ import annotations
import json, pickle, sys, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
from sample_prep_weights import maybe_bundle_weight  # default-off (IAJD_USE_PREP_WEIGHTS)
V14_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
OUT_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_ensemble_bundle.pkl"
OUT_REPORT = ROOT / "bioact_ensemble_report.json"
BIOACT = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"

M = 7                          # ensemble size
SEEDS = [42, 123, 7, 2024, 999, 31415, 27182]
BOOTSTRAP_FRAC = 0.85          # subsample fraction per model

XGB_HP = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    objective="reg:squarederror", tree_method="hist",
    n_jobs=4, verbosity=0,
)


def _train_one(X_train, y_train, seed: int, w=None):
    rng = np.random.RandomState(seed)
    n = len(y_train)
    idx = rng.choice(n, size=int(BOOTSTRAP_FRAC * n), replace=True)
    hp = dict(XGB_HP, random_state=seed)
    m = xgb.XGBRegressor(**hp)
    m.fit(X_train[idx], y_train[idx],
          sample_weight=(w[idx] if w is not None else None), verbose=False)
    return m, idx


def _loo_ensemble_predict(X, y, M=M, seeds=SEEDS, w=None):
    """Honest LOO: for each row i, train M models on the OTHER 246 with the
    seeds + bootstrap, get M predictions for i, return mean + std arrays."""
    n = len(y)
    means = np.zeros(n)
    sigmas = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        X_tr, y_tr = X[keep], y[keep]
        w_tr = w[keep] if w is not None else None
        preds = np.zeros(M)
        for k, seed in enumerate(seeds[:M]):
            rng = np.random.RandomState(seed + i)   # vary per-row to avoid cached bootstrap
            sub = rng.choice(len(y_tr), size=int(BOOTSTRAP_FRAC * len(y_tr)), replace=True)
            hp = dict(XGB_HP, random_state=seed)
            mm = xgb.XGBRegressor(**hp)
            mm.fit(X_tr[sub], y_tr[sub],
                   sample_weight=(w_tr[sub] if w_tr is not None else None), verbose=False)
            preds[k] = float(mm.predict(X[i:i+1])[0])
        means[i] = float(np.mean(preds))
        sigmas[i] = float(np.std(preds))
        if (i + 1) % 25 == 0:
            print(f"  LOO ensemble {i+1}/{n}  σ_mean={sigmas[:i+1].mean():.3f}",
                  flush=True)
    return means, sigmas


def main():
    print(f"Loading v14 bundle from {V14_BUNDLE.name}…", flush=True)
    with open(V14_BUNDLE, "rb") as f:
        b14 = pickle.load(f)
    X = np.asarray(b14["X_train"], dtype=float)
    y = np.asarray(b14["y_train"], dtype=float)
    families = np.array([str(f) for f in b14["families_train"]])
    n, d = X.shape
    print(f"  n={n}, n_features={d}", flush=True)

    # Sample-prep reliability weights (None unless IAJD_USE_PREP_WEIGHTS=1 -> no-op).
    w = maybe_bundle_weight(b14, BIOACT)
    if w is not None:
        print(f"  [prep-weights] ON: w[min/mean/max]={w.min():.2f}/{w.mean():.2f}/{w.max():.2f} "
              f"(sp_confidence x n_mice, aligned by SMILES)", flush=True)

    print(f"\n[1/3] LOO ensemble (M={M}) — honest σ per held-out row…", flush=True)
    loo_means, loo_sigmas = _loo_ensemble_predict(X, y, M=M, w=w)
    loo_mae = float(np.mean(np.abs(loo_means - y)))
    residuals = loo_means - y
    actual_std = float(np.std(residuals))
    mean_ensemble_sigma = float(np.mean(loo_sigmas))
    calibration = float(actual_std / mean_ensemble_sigma) if mean_ensemble_sigma > 0 else 1.0
    print(f"  ensemble LOO MAE: {loo_mae:.4f}", flush=True)
    print(f"  ensemble σ̄ (raw): {mean_ensemble_sigma:.4f}", flush=True)
    print(f"  actual residual std: {actual_std:.4f}", flush=True)
    print(f"  σ calibration factor: {calibration:.3f}", flush=True)

    # Per-family σ from LOO residuals
    per_family_sigma = {}
    print(f"\n  Per-family LOO RMSE:", flush=True)
    for fam in sorted(set(families)):
        mask = families == fam
        if mask.sum() < 3:
            per_family_sigma[fam] = float(actual_std)
            continue
        rmse = float(np.sqrt(np.mean((loo_means[mask] - y[mask]) ** 2)))
        per_family_sigma[fam] = rmse
        print(f"    {fam:<22s}  n={int(mask.sum()):>3d}  RMSE={rmse:.3f}", flush=True)

    print(f"\n[2/3] Training final ensemble on FULL data (for inference)…", flush=True)
    models = []
    boots_sizes = []
    for k, seed in enumerate(SEEDS[:M]):
        m, idx = _train_one(X, y, seed, w=w)
        models.append(m)
        boots_sizes.append(int(len(idx)))
        print(f"  model {k+1}/{M}  seed={seed}  bootstrap_size={len(idx)}", flush=True)

    print(f"\n[3/3] Saving ensemble bundle…", flush=True)
    bundle = {
        "version": "bioact_ensemble_v1_2026-05-29",
        "models": models,
        "seeds": SEEDS[:M],
        "bootstrap_sizes": boots_sizes,
        "sigma_calibration": calibration,
        "per_family_sigma": per_family_sigma,
        "global_sigma": float(actual_std),
        "block_slices": b14.get("block_slices", {}),
        "n_features": int(d),
        "training_size": int(n),
        "metrics": {
            "loo_mae_ensemble_mean": loo_mae,
            "ensemble_mean_sigma_raw": mean_ensemble_sigma,
            "actual_residual_std": actual_std,
            "calibration_factor": calibration,
            "M": M,
            "bootstrap_frac": BOOTSTRAP_FRAC,
        },
    }
    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    OUT_REPORT.write_text(json.dumps({
        k: v for k, v in bundle.items()
        if k not in ("models",)
    }, indent=2, default=str))
    print(f"  wrote {OUT_BUNDLE.name} + {OUT_REPORT.name}", flush=True)


if __name__ == "__main__":
    main()
