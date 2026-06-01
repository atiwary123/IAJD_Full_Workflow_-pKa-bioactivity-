"""
eval/al_vs_random.py — active-learning replay vs random.

Premise: the ensemble's posterior σ is a real signal — picking compounds
with high σ should reduce LOO MAE faster than picking random compounds. If
the QM/MD physics extension makes σ MORE honest (better-calibrated under
novelty), AL should win by a larger margin.

Protocol:
  1. Start with a small seed set (n=20) of random training compounds.
  2. Iteratively grow the training set by 5 compounds, chosen either by
     (a) highest predicted σ on the unlabeled pool (AL), or
     (b) random uniform selection (baseline).
  3. After each batch, refit XGBoost on the new training set and score on
     the rest. Log MAE_AL[t], MAE_random[t].
  4. Repeat with 5 seeds; average.

Outputs:
  eval/figures/al_vs_random.png
  eval/figures/al_vs_random.json
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "al_vs_random.png"
OUT_JSON = FIG_DIR / "al_vs_random.json"

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0,
    min_child_weight=2, objective="reg:squarederror",
    tree_method="hist", n_jobs=2, random_state=42, verbosity=0,
)


def _sigma_from_ensemble(X_train, y_train, X_pool, n_models: int = 5,
                          rng: np.random.Generator = None):
    """Bootstrap ensemble σ — each model trained on a resample, σ = std over
    predictions. Honest no-proxy: this is real posterior σ, not a learned
    estimate."""
    rng = rng or np.random.default_rng(42)
    preds = np.zeros((n_models, len(X_pool)))
    for k in range(n_models):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        params = {**XGB_PARAMS, "random_state": int(rng.integers(0, 2**31 - 1))}
        m = XGBRegressor(**params)
        m.fit(X_train[idx], y_train[idx])
        preds[k] = m.predict(X_pool)
    return preds.mean(axis=0), preds.std(axis=0)


def _replay(X, y, *, seed_size: int, batch_size: int, n_rounds: int,
             mode: str, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    all_idx = np.arange(n)
    seed_idx = rng.choice(all_idx, size=seed_size, replace=False)
    train_idx = set(int(i) for i in seed_idx)
    history = []
    for t in range(n_rounds):
        train_a = np.array(sorted(train_idx))
        pool_a = np.array(sorted(set(all_idx) - train_idx))
        if len(pool_a) == 0:
            break
        mu, sigma = _sigma_from_ensemble(X[train_a], y[train_a], X[pool_a], rng=rng)
        # Final model on current train set, evaluated on the pool.
        final = XGBRegressor(**XGB_PARAMS).fit(X[train_a], y[train_a])
        preds_pool = final.predict(X[pool_a])
        mae = float(mean_absolute_error(y[pool_a], preds_pool))
        history.append({"round": t, "n_train": int(len(train_a)),
                          "n_pool": int(len(pool_a)),
                          "mae_pool": mae, "mean_sigma": float(sigma.mean())})
        if mode == "al":
            order = np.argsort(-sigma)
        else:
            order = rng.permutation(len(pool_a))
        chosen = pool_a[order[:batch_size]]
        train_idx |= set(int(i) for i in chosen)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=20)
    args = parser.parse_args()

    bioact_path = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
    with open(bioact_path, "rb") as f:
        b = pickle.load(f)
    X_all = np.asarray(b["X_train"], dtype=float)
    y_all = np.asarray(b["y_train"], dtype=float)

    # Replace NaN with column medians for AL training only (training-side
    # convenience; XGBoost handles NaN natively, so this is just for the
    # sklearn metric loop).
    col_med = np.nanmedian(X_all, axis=0)
    X_filled = X_all.copy()
    for j in range(X_filled.shape[1]):
        mask = ~np.isfinite(X_filled[:, j])
        X_filled[mask, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0

    print(f"Training set: {len(y_all)} compounds, {X_filled.shape[1]} features")

    al_curves, random_curves = [], []
    for s in range(args.seeds):
        al = _replay(X_filled, y_all, seed_size=args.seed_size,
                      batch_size=args.batch_size, n_rounds=args.n_rounds,
                      mode="al", seed=42 + s)
        ra = _replay(X_filled, y_all, seed_size=args.seed_size,
                      batch_size=args.batch_size, n_rounds=args.n_rounds,
                      mode="random", seed=42 + s)
        al_curves.append([h["mae_pool"] for h in al])
        random_curves.append([h["mae_pool"] for h in ra])
        print(f"  seed {s}: AL final MAE={al[-1]['mae_pool']:.3f}  "
              f"random final MAE={ra[-1]['mae_pool']:.3f}")

    al_mean = np.array(al_curves).mean(axis=0)
    al_std  = np.array(al_curves).std(axis=0)
    ra_mean = np.array(random_curves).mean(axis=0)
    ra_std  = np.array(random_curves).std(axis=0)
    out = {
        "al_curve":     al_mean.tolist(),
        "al_std":       al_std.tolist(),
        "random_curve": ra_mean.tolist(),
        "random_std":   ra_std.tolist(),
        "rounds":       list(range(len(al_mean))),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(al_mean))
    ax.errorbar(x, al_mean, yerr=al_std, label="σ-guided AL", marker="o")
    ax.errorbar(x, ra_mean, yerr=ra_std, label="Random",      marker="s")
    ax.set_xlabel(f"Active-learning round (batch={args.batch_size}, "
                  f"seed_size={args.seed_size})")
    ax.set_ylabel("MAE on remaining pool (log10_flux units)")
    ax.set_title("σ-guided active learning vs random selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
