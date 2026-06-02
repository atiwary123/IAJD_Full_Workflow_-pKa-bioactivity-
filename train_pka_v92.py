"""
train_pka_v92.py — three-head pKa blend with optimal weights.

Heads (independent predictors, no cross-contamination):
  1. analog   : top-10 Tanimoto neighbors with sim≥0.6 floor, sim^4 weighting,
                returns weighted average of neighbor measured pKa
  2. xgb_pure : XGBoost on the 30 base RDKit/structural features only
                (NO MolGpKa) — so it's truly orthogonal to head 3
  3. molgpka  : live MolGpKa raw → per-family linear debias

Blend: minimize LOO MAE over (w1, w2, w3) subject to w_i ≥ 0 and sum=1
(simplex projection). Falls back to a non-negative least-squares blend if
the simplex optimizer hits a degenerate corner.

Outputs:
  IAJD_master/bundles_caches/pka_v92_bundle.joblib  — analog state + xgb +
                                                       debias models + weights
  preds_v91_final.npy                              — LOO predictions
                                                       (overwrites with v92 blend)
  pka_v92_loo_report.json                          — per-head + blend metrics
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.DataStructs import TanimotoSimilarity
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

RDLogger.logger().setLevel(RDLogger.ERROR)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))
from sample_prep_weights import maybe_pka_weight  # default-off (IAJD_USE_PREP_WEIGHTS)

PKA_XLSX = ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
MOLGPKA_NPY = ROOT / "IAJD_master/bundles_caches/molgpka_preds.npy"
DEBIAS_PATH = ROOT / "IAJD_master/bundles_caches/molgpka_debias_models.joblib"
OUT_BUNDLE = ROOT / "IAJD_master/bundles_caches/pka_v92_bundle.joblib"
OUT_PREDS = ROOT / "preds_v91_final.npy"
OUT_REPORT = ROOT / "pka_v92_loo_report.json"

# Analog head hyperparameters (mirrors v9.1)
K_QUERY_MAX = 10
SIM_THRESHOLD = 0.60
MIN_NEIGHBORS = 1
WEIGHT_POWER = 4.0

# XGB head hyperparameters
XGB_HP = dict(
    n_estimators=300, max_depth=2, learning_rate=0.12,
    min_child_weight=4, subsample=1.0, colsample_bytree=0.6,
    reg_lambda=3.0, random_state=42, n_jobs=1,
)


def build_v52_features_and_fps():
    """Build the v52 bundle once to get (features, fps, pkas, families, smiles)."""
    print("[v92] building v52 base bundle…", flush=True)
    cwd = os.getcwd()
    os.chdir(ROOT / "IAJD_master/datasets")
    try:
        from iajd_pka_v52 import build_bundle
        b = build_bundle("IAJD_pKa_v21_final.xlsx", verbose=False)
    finally:
        os.chdir(cwd)
    print(f"  bundle: {len(b.pkas)} compounds, {b.features.shape[1]} features", flush=True)
    return b


def head_analog_loo(b, verbose=True) -> np.ndarray:
    """For each compound i, weighted average of its top-K Tanimoto neighbors in
    the n-1 remaining compounds, weights = sim^4, threshold s≥0.6 with
    fallback to nearest if none."""
    n = len(b.pkas)
    preds = np.full(n, np.nan)
    y = np.array(b.pkas, dtype=float)
    for i in range(n):
        sims = np.array([
            TanimotoSimilarity(b.fps[i], b.fps[j]) if j != i else -1.0
            for j in range(n)
        ])
        order = np.argsort(-sims)
        top = order[:K_QUERY_MAX]
        above = [j for j in top if sims[j] >= SIM_THRESHOLD]
        if len(above) < MIN_NEIGHBORS:
            above = list(top[:MIN_NEIGHBORS])
        s_arr = np.array([sims[j] for j in above])
        w = s_arr ** WEIGHT_POWER
        w = w / w.sum()
        preds[i] = float(np.sum(w * y[np.array(above)]))
        if verbose and (i + 1) % 50 == 0:
            print(f"  [analog] {i+1}/{n}", flush=True)
    return preds


def head_xgb_loo(b, verbose=True, w=None) -> np.ndarray:
    """LOO XGB on the 30 base v52 features (no MolGpKa)."""
    n = len(b.pkas)
    X = np.asarray(b.features, dtype=float)
    # Sanitize NaN/inf in features
    for c in range(X.shape[1]):
        m = ~np.isfinite(X[:, c])
        if m.any():
            X[m, c] = np.nanmedian(X[:, c])
    y = np.array(b.pkas, dtype=float)
    preds = np.full(n, np.nan)
    sc_full = StandardScaler().fit(X)
    Xs_full = sc_full.transform(X)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        sc = StandardScaler().fit(X[keep])
        Xs_tr = sc.transform(X[keep])
        Xs_q = sc.transform(X[i:i+1])
        m = xgb.XGBRegressor(**XGB_HP)
        m.fit(Xs_tr, y[keep],
              sample_weight=(np.asarray(w)[keep] if w is not None else None), verbose=False)
        preds[i] = float(m.predict(Xs_q)[0])
        if verbose and (i + 1) % 50 == 0:
            print(f"  [xgb] {i+1}/{n}", flush=True)
    return preds


def head_molgpka_loo(b, raw_molgpka: np.ndarray, debias_models: dict,
                      verbose=True) -> np.ndarray:
    """For each compound i, apply per-family linear debias to its live MolGpKa
    raw value. The debias model is refit per-fold (leave-i-out) to avoid
    contamination of the i-th observation through the family slope/intercept.
    """
    n = len(b.pkas)
    y = np.array(b.pkas, dtype=float)
    fams = np.array(b.families)
    preds = np.full(n, np.nan)
    for i in range(n):
        fam = fams[i]
        # Refit family debias excluding i
        mask = (fams == fam) & (np.arange(n) != i) & np.isfinite(raw_molgpka) & np.isfinite(y)
        if mask.sum() >= 4:
            lr = LinearRegression().fit(raw_molgpka[mask].reshape(-1, 1), y[mask])
            slope = float(lr.coef_[0])
            intercept = float(lr.intercept_)
        elif fam in debias_models:
            slope = debias_models[fam]["slope"]
            intercept = debias_models[fam]["intercept"]
        else:
            # Pool across all families (n-weighted)
            models = debias_models
            tot_n = sum(d["n"] for d in models.values())
            slope = sum(d["slope"] * d["n"] for d in models.values()) / tot_n
            intercept = sum(d["intercept"] * d["n"] for d in models.values()) / tot_n
        if np.isfinite(raw_molgpka[i]):
            preds[i] = slope * raw_molgpka[i] + intercept
        else:
            preds[i] = np.nan
        if verbose and (i + 1) % 50 == 0:
            print(f"  [molgpka] {i+1}/{n}", flush=True)
    return preds


def optimize_weights(P: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find (w1, w2, w3) ≥ 0, sum=1, minimizing MAE = mean|y - P @ w|."""
    n, k = P.shape
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    Pm = P[mask]; ym = y[mask]
    print(f"  optimization data: {mask.sum()}/{n} valid rows × {k} heads", flush=True)

    def loss(w):
        return float(np.mean(np.abs(ym - Pm @ w)))

    # Simplex: parameterize w = softmax-like with non-negative + sum=1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    # Try multiple starts to avoid local minima
    starts = [
        np.ones(k) / k,             # uniform
        np.array([0.95, 0.05, 0.0]) if k == 3 else np.ones(k) / k,
        np.array([0.7, 0.15, 0.15]) if k == 3 else np.ones(k) / k,
        np.array([0.5, 0.5, 0.0]) if k == 3 else np.ones(k) / k,
    ]
    best_w = None; best_loss = float("inf")
    for w0 in starts:
        r = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                      options={"maxiter": 300, "ftol": 1e-7})
        if r.fun < best_loss:
            best_loss = r.fun
            best_w = r.x
    return best_w, best_loss


def main():
    print("="*70)
    print("[v92] training three-head pKa blend with optimal weights")
    print("="*70)
    b = build_v52_features_and_fps()
    y = np.array(b.pkas, dtype=float)
    n = len(y)
    sw_pka = maybe_pka_weight(list(b.canonical_smiles), PKA_XLSX)  # None unless flag (no-op)
    if sw_pka is not None:
        print(f"  [prep-weights] ON: pKa w[min/mean/max]="
              f"{sw_pka.min():.2f}/{sw_pka.mean():.2f}/{sw_pka.max():.2f}", flush=True)
    raw_molgpka = np.load(MOLGPKA_NPY)
    debias = joblib.load(DEBIAS_PATH)
    print(f"  molgpka cache: {len(raw_molgpka)} entries", flush=True)
    assert len(raw_molgpka) == n, f"molgpka ({len(raw_molgpka)}) ≠ bundle ({n})"

    print("\n[v92] head 1/3: analog (LOO Tanimoto neighbors)…", flush=True)
    p_analog = head_analog_loo(b)
    mae_analog = float(np.mean(np.abs(p_analog - y)))
    print(f"  analog LOO MAE: {mae_analog:.4f}", flush=True)

    print("\n[v92] head 2/3: pure XGB on 30 base features (LOO)…", flush=True)
    p_xgb = head_xgb_loo(b, w=sw_pka)
    mae_xgb = float(np.mean(np.abs(p_xgb - y)))
    print(f"  xgb_pure LOO MAE: {mae_xgb:.4f}", flush=True)

    print("\n[v92] head 3/3: debiased MolGpKa (LOO per-family debias)…", flush=True)
    p_molgpka = head_molgpka_loo(b, raw_molgpka, debias)
    mae_molgpka = float(np.mean(np.abs(p_molgpka - y)))
    print(f"  molgpka LOO MAE: {mae_molgpka:.4f}", flush=True)

    print("\n[v92] optimizing simplex weights…", flush=True)
    P = np.column_stack([p_analog, p_xgb, p_molgpka])
    w, loss = optimize_weights(P, y)
    print(f"  optimal weights: analog={w[0]:.3f}  xgb={w[1]:.3f}  molgpka={w[2]:.3f}", flush=True)
    print(f"  blend LOO MAE: {loss:.4f}", flush=True)
    print(f"  improvement vs best single head: "
          f"{min(mae_analog, mae_xgb, mae_molgpka) - loss:+.4f}", flush=True)

    # Final predictions
    blend = P @ w
    np.save(OUT_PREDS, blend)
    print(f"\n  Saved {OUT_PREDS.name} ({len(blend)} LOO predictions)", flush=True)

    # Per-family metrics for the blend
    fams = np.array(b.families)
    per_fam = {}
    for f in sorted(set(fams)):
        m = fams == f
        per_fam[f] = {
            "n": int(m.sum()),
            "mae": float(np.mean(np.abs(blend[m] - y[m]))),
            "rmse": float(np.sqrt(np.mean((blend[m] - y[m]) ** 2))),
        }
    print(f"\n  per-family blend MAE:", flush=True)
    for f, m in per_fam.items():
        print(f"    {f:<22s} n={m['n']:>3}  MAE={m['mae']:.3f}", flush=True)

    # Save bundle for inference
    # The "analog state" we need to ship: training fps + pkas + families
    bundle = {
        "version": "pka_v92",
        "weights": {"analog": float(w[0]), "xgb_pure": float(w[1]),
                     "molgpka_debiased": float(w[2])},
        "analog": {
            "fps": list(b.fps),
            "pkas": [float(x) for x in b.pkas],
            "k_query_max": K_QUERY_MAX,
            "sim_threshold": SIM_THRESHOLD,
            "min_neighbors": MIN_NEIGHBORS,
            "weight_power": WEIGHT_POWER,
        },
        "xgb_pure": {
            "model": xgb.XGBRegressor(**XGB_HP).fit(
                StandardScaler().fit_transform(
                    np.nan_to_num(np.asarray(b.features, dtype=float),
                                   nan=np.nanmedian(b.features))
                ),
                np.array(b.pkas, dtype=float),
                sample_weight=sw_pka,
                verbose=False,
            ),
            "scaler": StandardScaler().fit(
                np.nan_to_num(np.asarray(b.features, dtype=float),
                                nan=np.nanmedian(b.features))
            ),
            "hp": XGB_HP,
        },
        "molgpka_debiased": {"debias_models": debias},
        "training_smiles": list(b.canonical_smiles),
        "training_families": list(b.families),
        "metrics": {
            "loo_mae_per_head": {
                "analog": mae_analog, "xgb_pure": mae_xgb,
                "molgpka_debiased": mae_molgpka,
            },
            "loo_mae_blend": loss,
            "loo_mae_per_family": per_fam,
            "n_train": n,
        },
    }
    joblib.dump(bundle, OUT_BUNDLE)
    print(f"\n  Saved {OUT_BUNDLE.name}", flush=True)

    OUT_REPORT.write_text(json.dumps(bundle["metrics"], indent=2, default=str))
    print(f"  Saved {OUT_REPORT.name}", flush=True)
    print("\n[v92] DONE.")


if __name__ == "__main__":
    main()
