"""
loo_pka_components.py — extract per-fold direct / analog / molgpka / sim / y
from the v9.1 pKa LOO pipeline so we can sweep dynamic-weight blends offline.

Saves: pka_loo_components.npz with arrays
    direct, analog, molgpka, max_sim, y_true, n_neighbors
    families  (object array of strings)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE = HERE / "IAJD_master" / "code"
DATA = HERE / "IAJD_master" / "datasets"
CACHE = HERE / "IAJD_master" / "bundles_caches"
sys.path.insert(0, str(CODE))

# The v91 loader reads MolGpKa artifacts via paths relative to the xlsx; the
# main predict path symlinks them. Do the same here to be safe.
for name in ("molgpka_debias_models.joblib", "molgpka_preds.npy"):
    src = CACHE / name; dst = DATA / name
    if not dst.exists() and src.exists():
        try: dst.symlink_to(src)
        except OSError:
            import shutil; shutil.copy2(src, dst)

import numpy as np
from sklearn.preprocessing import StandardScaler
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb

from iajd_pka_v91 import (
    load_v91_bundle, TUNED_XGB_HP, K_DELTA_TRAIN, K_QUERY_MAX,
    SIM_THRESHOLD, MIN_NEIGHBORS, WEIGHT_POWER, BLEND_ALPHA,
)


def loo_components(bundle):
    n = len(bundle.train_y)
    X = bundle.train_X           # (n, 31), feature 30 is LOO-debiased MolGpKa
    y = np.asarray(bundle.train_y)
    fps = bundle.train_fps
    fams = bundle.train_fams

    direct = np.zeros(n); analog = np.zeros(n); max_sim = np.zeros(n)
    n_neigh = np.zeros(n, dtype=int)
    t0 = time.time()
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        X_tr = X[keep]; y_tr = y[keep]

        # Direct XGB refit on 245 → predict held-out compound
        sc = StandardScaler().fit(X_tr)
        direct_m = xgb.XGBRegressor(**TUNED_XGB_HP)
        direct_m.fit(sc.transform(X_tr), y_tr, verbose=False)
        direct[i] = float(direct_m.predict(sc.transform(X[i:i+1]))[0])

        # Tanimoto sims from i to the 245 — same MFPGEN fp generator
        sims = np.asarray([TanimotoSimilarity(fps[i], fps[j]) for j in keep])
        max_sim[i] = float(sims.max())

        # Analog-delta: build LOO delta pairs from the 245 only
        deltaX = []; deltay = []
        for jj, j in enumerate(keep):
            sims_j = np.asarray([TanimotoSimilarity(fps[j], fps[k])
                                 for k in keep if k != j])
            keep_no_j = [k for k in keep if k != j]
            for kk in np.argsort(-sims_j)[:K_DELTA_TRAIN]:
                k = keep_no_j[int(kk)]
                deltaX.append(X[j] - X[k])
                deltay.append(float(y[j] - y[k]))
        dX = np.asarray(deltaX); dy = np.asarray(deltay)
        sc_d = StandardScaler().fit(dX)
        delta_m = xgb.XGBRegressor(**TUNED_XGB_HP)
        delta_m.fit(sc_d.transform(dX), dy, verbose=False)

        # Query analog-delta over the 245 with uniform sim-threshold rule
        order = np.argsort(-sims)[:K_QUERY_MAX]
        top = [int(idx) for idx in order if sims[idx] >= SIM_THRESHOLD]
        if len(top) < MIN_NEIGHBORS:
            top = [int(idx) for idx in order[:MIN_NEIGHBORS]]
        ests = []; wts = []
        for idx in top:
            j_global = keep[idx]
            diff = (X[i] - X[j_global]).reshape(1, -1)
            dp = float(delta_m.predict(sc_d.transform(diff))[0])
            ests.append(float(y[j_global]) + dp)
            wts.append(float(sims[idx]) ** WEIGHT_POWER)
        wts = np.asarray(wts)
        analog[i] = float(np.dot(wts / wts.sum(), ests)) if wts.sum() > 0 else float(np.mean(ests))
        n_neigh[i] = len(top)

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  LOO {i+1}/{n}   elapsed={elapsed:.1f}s   ETA={elapsed*(n-i-1)/(i+1):.0f}s",
                  flush=True)
    return direct, analog, max_sim, n_neigh


def main():
    print("Loading v9.1 bundle (cwd = datasets dir)...", flush=True)
    cwd = os.getcwd()
    os.chdir(DATA)
    try:
        bundle = load_v91_bundle()
    finally:
        os.chdir(cwd)

    n = len(bundle.train_y)
    print(f"  n compounds: {n}")
    print(f"  feature dim: {bundle.train_X.shape[1]}")
    print(f"  α={BLEND_ALPHA}  Kt={K_DELTA_TRAIN}  Kq_max={K_QUERY_MAX}  s≥{SIM_THRESHOLD}  wp={WEIGHT_POWER}")
    print()

    direct, analog, max_sim, n_neigh = loo_components(bundle)
    y_true = np.asarray(bundle.train_y)
    families = np.asarray([str(f) for f in bundle.train_fams])
    molgpka = bundle.train_X[:, 30].copy()

    final_orig = BLEND_ALPHA * direct + (1 - BLEND_ALPHA) * analog
    mae_orig = float(np.mean(np.abs(final_orig - y_true)))
    print(f"\n  Replicated v9.1 LOO pooled MAE: {mae_orig:.4f}  (target 0.0672)")

    out_path = HERE / "pka_loo_components.npz"
    np.savez(
        out_path,
        direct=direct, analog=analog, molgpka=molgpka, max_sim=max_sim,
        y_true=y_true, n_neighbors=n_neigh, families=families,
    )
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
