"""
tune_pka_v92_perfamily.py — joint sweep:
  • analog K ∈ {3, 5, 10}  (neighbors used in the analog head)
  • blend weights:  global vs per-family-with-shrinkage

Picks the configuration that minimizes pooled LOO MAE on the 278-row v9.2
bundle, and overwrites the bundle + preds_v91_final.npy with the winner.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
import joblib
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "IAJD_master/bundles_caches/pka_v92_bundle.joblib"
REPORT = ROOT / "pka_v92_tuning_report.json"
OUT_PREDS = ROOT / "preds_v91_final.npy"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "IAJD_master/code"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.DataStructs import TanimotoSimilarity

from train_pka_v92 import (
    build_v52_features_and_fps, head_xgb_loo, head_molgpka_loo,
    MOLGPKA_NPY, DEBIAS_PATH,
    SIM_THRESHOLD, MIN_NEIGHBORS, WEIGHT_POWER,
)


def head_analog_loo_k(b, K_query: int) -> np.ndarray:
    """LOO analog head with configurable K_query (top-K neighbors)."""
    n = len(b.pkas)
    y = np.array(b.pkas, dtype=float)
    preds = np.full(n, np.nan)
    for i in range(n):
        sims = np.array([
            TanimotoSimilarity(b.fps[i], b.fps[j]) if j != i else -1.0
            for j in range(n)
        ])
        order = np.argsort(-sims)
        top = order[:K_query]
        above = [j for j in top if sims[j] >= SIM_THRESHOLD]
        if len(above) < MIN_NEIGHBORS:
            above = list(top[:MIN_NEIGHBORS])
        s_arr = np.array([sims[j] for j in above])
        w = s_arr ** WEIGHT_POWER
        w = w / w.sum()
        preds[i] = float(np.sum(w * y[np.array(above)]))
    return preds


def _optimize(P, y, n_starts=5):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * P.shape[1]
    starts = [
        np.ones(P.shape[1]) / P.shape[1],
        np.array([0.95, 0.05, 0.00]),
        np.array([0.10, 0.80, 0.10]),
        np.array([0.30, 0.30, 0.40]),
        np.array([0.50, 0.50, 0.00]),
    ]
    best_w = None; best_loss = float("inf")
    for w0 in starts[:n_starts]:
        r = minimize(lambda w: float(np.mean(np.abs(y - P @ w))),
                      w0, method="SLSQP", bounds=bnds, constraints=cons,
                      options={"maxiter": 500, "ftol": 1e-8})
        if r.fun < best_loss:
            best_loss, best_w = r.fun, r.x
    return best_w, best_loss


def main():
    print("[tune] building v52 bundle + heads…", flush=True)
    bundle = build_v52_features_and_fps()
    y = np.array(bundle.pkas, dtype=float)
    fams = np.array(bundle.families)
    raw_molgpka = np.load(MOLGPKA_NPY)
    debias = joblib.load(DEBIAS_PATH)

    # Heads 2/3 are independent of K — compute once
    print("[tune] xgb_pure LOO…", flush=True)
    p_x = head_xgb_loo(bundle, verbose=False)
    print(f"  xgb_pure MAE: {np.mean(np.abs(p_x - y)):.4f}", flush=True)
    print("[tune] molgpka LOO…", flush=True)
    p_m = head_molgpka_loo(bundle, raw_molgpka, debias, verbose=False)
    print(f"  molgpka MAE:  {np.mean(np.abs(p_m - y)):.4f}", flush=True)

    results = {}
    for K in (3, 5, 10):
        print(f"\n[tune] K={K} analog LOO…", flush=True)
        p_a = head_analog_loo_k(bundle, K)
        analog_mae = float(np.mean(np.abs(p_a - y)))
        print(f"  analog (K={K}) MAE: {analog_mae:.4f}", flush=True)

        P = np.column_stack([p_a, p_x, p_m])

        # Global weights
        w_g, mae_g = _optimize(P, y)
        print(f"  global weights: a={w_g[0]:.3f} x={w_g[1]:.3f} m={w_g[2]:.3f}  MAE={mae_g:.4f}",
              flush=True)

        # Per-family with shrinkage
        per_fam_w = {}; per_fam_mae = {}
        for fam in sorted(set(fams)):
            mask = fams == fam
            if mask.sum() < 5:
                per_fam_w[fam] = list(w_g)
                per_fam_mae[fam] = float(np.mean(np.abs(P[mask] @ w_g - y[mask])))
                continue
            w_fam, _ = _optimize(P[mask], y[mask], n_starts=4)
            best_lam, best_mae = 0.0, float(np.mean(np.abs(P[mask] @ w_fam - y[mask])))
            for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                w_b = lam * w_g + (1 - lam) * w_fam
                mae = float(np.mean(np.abs(P[mask] @ w_b - y[mask])))
                if mae < best_mae:
                    best_mae, best_lam = mae, lam
            per_fam_w[fam] = list(best_lam * w_g + (1 - best_lam) * w_fam)
            per_fam_mae[fam] = best_mae
        blend_perfam = np.zeros(len(y))
        for fam in set(fams):
            mask = fams == fam
            blend_perfam[mask] = P[mask] @ np.array(per_fam_w[fam])
        mae_perfam = float(np.mean(np.abs(blend_perfam - y)))
        print(f"  per-family MAE: {mae_perfam:.4f}", flush=True)

        results[K] = {
            "analog_mae": analog_mae,
            "global_w": list(map(float, w_g)),
            "global_mae": mae_g,
            "perfam_w": {f: list(map(float, w)) for f, w in per_fam_w.items()},
            "perfam_mae_per_fam": per_fam_mae,
            "perfam_mae_pooled": mae_perfam,
            "best_mae": min(mae_g, mae_perfam),
            "best_uses_perfam": (mae_perfam < mae_g - 0.001),
        }

    print("\n========== SUMMARY ==========")
    for K, r in results.items():
        print(f"  K={K:>2}  global={r['global_mae']:.4f}  perfam={r['perfam_mae_pooled']:.4f}  "
              f"best={r['best_mae']:.4f}", flush=True)

    # Pick the winning configuration
    winner_K = min(results, key=lambda k: results[k]["best_mae"])
    winner = results[winner_K]
    print(f"\nWINNER: K={winner_K}, MAE={winner['best_mae']:.4f}, "
          f"per-family={winner['best_uses_perfam']}", flush=True)

    # Apply winner to bundle
    b = joblib.load(BUNDLE)
    p_a_winner = head_analog_loo_k(bundle, winner_K)
    P_win = np.column_stack([p_a_winner, p_x, p_m])
    if winner["best_uses_perfam"]:
        blend = np.zeros(len(y))
        for fam in set(fams):
            mask = fams == fam
            blend[mask] = P_win[mask] @ np.array(winner["perfam_w"][fam])
        b["weights"] = {"per_family": winner["perfam_w"],
                         "global": {"analog": winner["global_w"][0],
                                     "xgb_pure": winner["global_w"][1],
                                     "molgpka_debiased": winner["global_w"][2]}}
    else:
        w_g = winner["global_w"]
        blend = P_win @ np.array(w_g)
        b["weights"] = {"analog": w_g[0], "xgb_pure": w_g[1], "molgpka_debiased": w_g[2]}
    b["analog"]["k_query_max"] = winner_K
    b["metrics"]["loo_mae_blend"] = winner["best_mae"]
    b["metrics"]["analog_K"] = winner_K
    b["metrics"]["uses_perfam_weights"] = winner["best_uses_perfam"]
    joblib.dump(b, BUNDLE)
    np.save(OUT_PREDS, blend)
    print(f"  saved {BUNDLE.name}  (K_query_max={winner_K}, weights updated)", flush=True)
    print(f"  saved {OUT_PREDS.name}", flush=True)

    REPORT.write_text(json.dumps(results, indent=2, default=str))
    print(f"  saved {REPORT.name}", flush=True)


if __name__ == "__main__":
    main()
