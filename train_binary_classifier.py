"""
train_binary_classifier.py — train a binary classifier head for
P(log10_flux_total ≥ threshold).

Default threshold = 8.0 (i.e. flux ≥ 10^8), but the threshold is **tunable
at inference time**: the bundle stores LOO probabilities (NOT hard labels),
which are calibrated to the regressor's continuous prediction. The
classification call computes `P(log10_flux ≥ T)` for any T the caller asks
about, by mapping the regression prediction through a calibrated
sigmoid-conditioned-on-T model. Specifically we model:

    P(y ≥ T | x) = σ( a(T) · (ŷ_reg(x) − T) + b(T) )

where a(T), b(T) are fitted from LOO regression residuals at each T in a
fine grid. This is a Mills-ratio style approach that's robust to label
noise (which the user has explicitly flagged is large), since the model
is trained from continuous LOO outputs rather than coarse binary labels.

We ALSO train a separate hard-target XGB classifier at the default
threshold (8.0) for direct comparison, with isotonic calibration. The
two are reported side by side.

Inputs:
  - IAJD_master/bundles_caches/bioact_v14_bundle.pkl  (X_train, y_train, fps_train)

Outputs:
  - IAJD_master/bundles_caches/bioact_binary_bundle.pkl
  - bioact_binary_loo_report.json
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
from sample_prep_weights import maybe_bundle_weight  # default-off (IAJD_USE_PREP_WEIGHTS)
V14_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
OUT_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_binary_bundle.pkl"
REPORT_JSON = ROOT / "bioact_binary_loo_report.json"

DEFAULT_THRESHOLD = 8.0
THRESHOLD_GRID = np.round(np.arange(6.5, 9.51, 0.25), 2)  # tunable inference grid

XGB_CLF_HP = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=1, eval_metric="logloss",
)
XGB_REG_HP = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=1,
)


def loo_predict_clf(X, y_bin, w=None):
    """Leave-one-out probabilities from XGB classifier."""
    n = len(y_bin)
    p = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        clf = xgb.XGBClassifier(**XGB_CLF_HP)
        clf.fit(X[mask], y_bin[mask],
                sample_weight=(w[mask] if w is not None else None), verbose=False)
        p[i] = float(clf.predict_proba(X[i:i+1])[0, 1])
    return p


def loo_predict_reg(X, y, w=None):
    """LOO regression predictions for the noise-aware probabilistic head."""
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        reg = xgb.XGBRegressor(**XGB_REG_HP)
        reg.fit(X[mask], y[mask],
                sample_weight=(w[mask] if w is not None else None), verbose=False)
        yhat[i] = float(reg.predict(X[i:i+1])[0])
    return yhat


def fit_continuous_threshold_calibrator(yhat_loo, y_true, thresholds):
    """For each threshold T, fit a Platt-style sigmoid on (yhat - T) → 1[y≥T].

    Returns dict T → (a, b) coefficients of:  P = σ(a · (ŷ−T) + b).
    Uses logistic regression with a single feature.
    """
    from sklearn.linear_model import LogisticRegression
    calib = {}
    for T in thresholds:
        labels = (y_true >= T).astype(int)
        if labels.sum() < 3 or (1 - labels).sum() < 3:
            # not enough either-class; use a degenerate
            calib[float(T)] = {"a": 1.0, "b": 0.0,
                                "n_pos": int(labels.sum()),
                                "n_neg": int((1-labels).sum()),
                                "degenerate": True}
            continue
        x = (yhat_loo - T).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500).fit(x, labels)
        calib[float(T)] = {"a": float(lr.coef_[0, 0]),
                            "b": float(lr.intercept_[0]),
                            "n_pos": int(labels.sum()),
                            "n_neg": int((1-labels).sum()),
                            "degenerate": False}
    return calib


def main():
    print(f"Loading v14 bundle from {V14_BUNDLE.name}…")
    with open(V14_BUNDLE, "rb") as f:
        b = pickle.load(f)
    X = np.asarray(b["X_train"], dtype=float)
    y = np.asarray(b["y_train"], dtype=float)
    smis = b.get("smis_train", [])
    families = b.get("families_train", [])
    n = X.shape[0]
    print(f"  n={n}, X.shape={X.shape}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]  median={np.median(y):.2f}")
    print(f"  fraction ≥ {DEFAULT_THRESHOLD}: {(y >= DEFAULT_THRESHOLD).mean():.2%}")

    w = maybe_bundle_weight(b, BIO_XLSX)  # None unless IAJD_USE_PREP_WEIGHTS=1 (no-op)
    if w is not None:
        print(f"  [prep-weights] ON: w[min/mean/max]={w.min():.2f}/{w.mean():.2f}/{w.max():.2f}")

    # ── 1. Hard-target classifier @ default threshold ───────────────────
    print(f"\n[1/4] Training direct binary classifier @ T={DEFAULT_THRESHOLD}…")
    y_bin_default = (y >= DEFAULT_THRESHOLD).astype(int)
    print(f"      class balance: pos={y_bin_default.sum()}/{n} ({y_bin_default.mean():.2%})")
    p_loo_direct = loo_predict_clf(X, y_bin_default, w=w)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_loo_direct, y_bin_default)
    p_loo_direct_cal = iso.transform(p_loo_direct)

    roc_auc_d = roc_auc_score(y_bin_default, p_loo_direct_cal) if y_bin_default.sum() > 0 else float("nan")
    pr_auc_d  = average_precision_score(y_bin_default, p_loo_direct_cal)
    brier_d   = brier_score_loss(y_bin_default, p_loo_direct_cal)
    logloss_d = log_loss(y_bin_default, np.clip(p_loo_direct_cal, 1e-6, 1-1e-6))
    print(f"      DIRECT LOO @ T={DEFAULT_THRESHOLD}: ROC-AUC={roc_auc_d:.3f}  "
          f"PR-AUC={pr_auc_d:.3f}  Brier={brier_d:.3f}  logloss={logloss_d:.3f}")

    final_direct = xgb.XGBClassifier(**XGB_CLF_HP).fit(X, y_bin_default, sample_weight=w, verbose=False)

    # ── 2. LOO regression predictions (input to noise-aware head) ───────
    print(f"\n[2/4] LOO regression for the continuous-threshold head…")
    yhat_loo = loo_predict_reg(X, y, w=w)
    reg_mae = float(np.mean(np.abs(yhat_loo - y)))
    reg_r2  = 1 - np.sum((y - yhat_loo)**2) / np.sum((y - y.mean())**2)
    print(f"      regressor LOO: MAE={reg_mae:.3f}  R²={reg_r2:.3f}")

    final_reg = xgb.XGBRegressor(**XGB_REG_HP).fit(X, y, sample_weight=w, verbose=False)

    # ── 3. Threshold-grid calibrator ────────────────────────────────────
    print(f"\n[3/4] Fitting per-threshold sigmoid calibrators over grid {THRESHOLD_GRID.tolist()}…")
    threshold_calib = fit_continuous_threshold_calibrator(yhat_loo, y, THRESHOLD_GRID)

    # Evaluate every grid threshold via LOO
    grid_metrics = {}
    for T in THRESHOLD_GRID:
        T = float(T)
        c = threshold_calib[T]
        if c["degenerate"]:
            grid_metrics[T] = {"roc_auc": None, "pr_auc": None,
                               "brier": None, "n_pos": c["n_pos"]}
            continue
        p = 1.0 / (1.0 + np.exp(-(c["a"] * (yhat_loo - T) + c["b"])))
        labels = (y >= T).astype(int)
        try:
            grid_metrics[T] = {
                "roc_auc": float(roc_auc_score(labels, p)),
                "pr_auc": float(average_precision_score(labels, p)),
                "brier": float(brier_score_loss(labels, p)),
                "logloss": float(log_loss(labels, np.clip(p, 1e-6, 1-1e-6))),
                "n_pos": int(labels.sum()),
                "n_neg": int((1-labels).sum()),
                "a": c["a"], "b": c["b"],
            }
        except Exception as e:
            grid_metrics[T] = {"error": str(e), "n_pos": int(labels.sum())}
    print("      grid (T, roc_auc, pr_auc):")
    for T, m in sorted(grid_metrics.items()):
        print(f"        T={T:>4.2f}  ROC={m.get('roc_auc') if m.get('roc_auc') is None else round(m['roc_auc'],3)}  "
              f"PR={m.get('pr_auc') if m.get('pr_auc') is None else round(m['pr_auc'],3)}  "
              f"n_pos={m.get('n_pos')}")

    # ── 4. Save bundle ──────────────────────────────────────────────────
    print(f"\n[4/4] Saving bundle → {OUT_BUNDLE.name}…")
    bundle = {
        "version": "bioact_binary_v1",
        "default_threshold": DEFAULT_THRESHOLD,
        "direct_classifier": final_direct,
        "direct_isotonic_calibrator": iso,
        "regressor": final_reg,
        "threshold_calibrators": threshold_calib,
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "train_loo_yhat": yhat_loo.tolist(),
        "train_y": y.tolist(),
        "train_smis": list(smis),
        "train_families": list(families),
        "metrics": {
            "n_train": n,
            "regression_loo_mae": reg_mae,
            "regression_loo_r2": reg_r2,
            "direct_classifier_loo_roc_auc": roc_auc_d,
            "direct_classifier_loo_pr_auc": pr_auc_d,
            "direct_classifier_loo_brier": brier_d,
            "direct_classifier_loo_logloss": logloss_d,
            "grid_metrics": grid_metrics,
        },
    }
    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    REPORT_JSON.write_text(json.dumps({k: v for k, v in bundle["metrics"].items()}, indent=2,
                                     default=str))
    print(f"  Wrote {OUT_BUNDLE.name} and {REPORT_JSON.name}")
    print("\nDone.")
    return bundle


if __name__ == "__main__":
    main()
