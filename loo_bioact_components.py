"""
loo_bioact_components.py — per-fold LOO predictions for four standalone heads:
    direct_xgb_full   = XGB on all 88 features (the v14 direct path)
    analog_delta      = Tanimoto-weighted neighbor consensus (the v14 analog path)
    lion_head         = XGB on LION block only (cols 50-63)
    admet_head        = XGB on ADMET block only (cols 64-73)

Outputs bioact_loo_components.npz with arrays:
    direct, analog, lion, admet, max_sim, y_true, families  (object array of strings)
"""
from __future__ import annotations
import pickle
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import RDLogger
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "IAJD_master" / "bundles_caches" / "bioact_v14_bundle.pkl"

# Block ranges from the bundle's block_slices
BLOCK_LION_SLICE = slice(50, 64)
BLOCK_ADMET_SLICE = slice(64, 74)
BLOCK_DPRIME_SLICE = slice(92, 106)

# XGBoost hyperparameters: chosen modest to keep LOO fast.
DIRECT_HP = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                 random_state=42, n_jobs=1)
BLOCK_HP = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=1)


def _analog_pred(i, X, y, fps, sim_power=4, k=8, min_sim=0.3):
    """Tanimoto-weighted neighbor consensus. Returns (pred, max_sim)."""
    sims = np.asarray([TanimotoSimilarity(fps[i], fps[j])
                       for j in range(len(y)) if j != i])
    idxs = np.asarray([j for j in range(len(y)) if j != i])
    order = np.argsort(-sims)
    keep = [int(o) for o in order if sims[o] >= min_sim][:k]
    if not keep:
        keep = [int(order[0])]
    weights = np.asarray([sims[o] for o in keep]) ** sim_power
    ys = np.asarray([y[idxs[o]] for o in keep])
    if weights.sum() <= 0:
        return float(np.mean(ys)), float(sims.max())
    return float(np.dot(weights / weights.sum(), ys)), float(sims.max())


def _fit_predict(X_tr, y_tr, X_q, hp):
    m = xgb.XGBRegressor(**hp)
    m.fit(X_tr, y_tr, verbose=False)
    return float(m.predict(X_q.reshape(1, -1))[0])


def main():
    print(f"Loading {BUNDLE}...", flush=True)
    with open(BUNDLE, "rb") as f:
        b = pickle.load(f)
    X = np.asarray(b["X_train"], dtype=float)
    y = np.asarray(b["y_train"], dtype=float)
    fps = b["fps_train"]
    families = np.asarray([str(f) for f in b["families_train"]])
    n = X.shape[0]
    print(f"  n={n}  X.shape={X.shape}  y range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"  LION cols: {BLOCK_LION_SLICE}  ADMET cols: {BLOCK_ADMET_SLICE}")
    print()

    direct = np.zeros(n); analog = np.zeros(n); lion = np.zeros(n)
    admet = np.zeros(n); max_sim = np.zeros(n)
    has_dprime = X.shape[1] >= 106
    qmmd = np.zeros(n) if has_dprime else None
    t0 = time.time()
    for i in range(n):
        keep = np.array([j for j in range(n) if j != i])
        X_tr = X[keep]; y_tr = y[keep]

        # Full-feature direct head
        direct[i] = _fit_predict(X_tr, y_tr, X[i], DIRECT_HP)
        # LION-only head
        X_tr_l = X_tr[:, BLOCK_LION_SLICE]
        lion[i] = _fit_predict(X_tr_l, y_tr, X[i, BLOCK_LION_SLICE], BLOCK_HP)
        # ADMET-only head
        X_tr_a = X_tr[:, BLOCK_ADMET_SLICE]
        admet[i] = _fit_predict(X_tr_a, y_tr, X[i, BLOCK_ADMET_SLICE], BLOCK_HP)
        # QM/MD physics head (Block D')
        if qmmd is not None:
            X_tr_p = X_tr[:, BLOCK_DPRIME_SLICE]
            qmmd[i] = _fit_predict(X_tr_p, y_tr, X[i, BLOCK_DPRIME_SLICE], BLOCK_HP)
        # Analog-delta path
        analog[i], max_sim[i] = _analog_pred(i, X, y, fps)

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  LOO {i+1}/{n}  elapsed={elapsed:.1f}s  ETA={elapsed*(n-i-1)/(i+1):.0f}s",
                  flush=True)

    out_path = HERE / "bioact_loo_components.npz"
    save_kwargs = dict(direct=direct, analog=analog, lion=lion, admet=admet,
                        max_sim=max_sim, y_true=y, families=families)
    if qmmd is not None:
        save_kwargs["qmmd"] = qmmd
    np.savez(out_path, **save_kwargs)
    mae = lambda p: float(np.mean(np.abs(p - y)))
    print()
    print(f"  pure direct      MAE = {mae(direct):.4f}")
    print(f"  pure analog      MAE = {mae(analog):.4f}")
    print(f"  pure lion-only   MAE = {mae(lion):.4f}")
    print(f"  pure admet-only  MAE = {mae(admet):.4f}")
    if qmmd is not None:
        print(f"  pure qmmd (D')   MAE = {mae(qmmd):.4f}")
    blend = 0.5 * direct + 0.5 * analog
    print(f"  naive 50/50      MAE = {mae(blend):.4f}")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
