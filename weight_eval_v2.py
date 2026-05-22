"""
weight_eval_v2.py — push the stacker further:
  • pKa : XGB stackers with various feature subsets and shallower trees
  • bioact: XGB stackers with different HP; also try LinearRegression + Ridge
            on simplex-constrained weights as an interpretable baseline
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

HERE = Path(__file__).resolve().parent


def _mae(p, y): return float(np.mean(np.abs(p - y)))


def _per_fam(p, y, fams):
    out = {}
    for f in np.unique(fams):
        m = fams == f
        out[str(f)] = {"n": int(m.sum()), "mae": _mae(p[m], y[m])}
    return out


def _cv_stacker(X, y, fams, estimator_factory, seed=0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for tr, te in skf.split(y, fams):
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        preds[te] = est.predict(X[te])
    return _mae(preds, y), _per_fam(preds, y, fams), preds


def pka_block():
    d = np.load(HERE / "pka_loo_components.npz", allow_pickle=True)
    y = d["y_true"]; fams = d["families"]; ms = d["max_sim"]
    direct = d["direct"]; analog = d["analog"]; molgpka = d["molgpka"]
    lb = LabelBinarizer().fit(fams); fam_oh = lb.transform(fams)

    feature_sets = {
        "preds_only": np.column_stack([direct, analog, molgpka]),
        "preds+sim":  np.column_stack([direct, analog, molgpka, ms]),
        "preds+sim+fam": np.column_stack([direct, analog, molgpka, ms, fam_oh]),
    }
    estimators = [
        ("xgb_d2_n100", lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            random_state=42, n_jobs=1)),
        ("xgb_d3_n200", lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42, n_jobs=1)),
        ("xgb_d2_n300_lr0.03", lambda: xgb.XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03,
            random_state=42, n_jobs=1)),
        ("ridge_a1.0", lambda: Ridge(alpha=1.0)),
        ("ridge_a5.0", lambda: Ridge(alpha=5.0)),
        ("linreg",     lambda: LinearRegression()),
    ]

    print("=" * 60)
    print(f"pKa stacker sweep  baseline α=0.05 MAE = {_mae(0.05*direct + 0.95*analog, y):.4f}")
    print("=" * 60)
    best = (np.inf, None)
    rows = []
    for fname, X in feature_sets.items():
        for ename, fact in estimators:
            mae, per_fam, preds = _cv_stacker(X, y, fams, fact)
            row = {"features": fname, "estimator": ename, "cv_mae": mae,
                   "per_family": per_fam}
            rows.append(row)
            print(f"  feat={fname:18s} est={ename:18s} MAE={mae:.4f}")
            if mae < best[0]:
                best = (mae, row)
    print(f"\n  BEST: feat={best[1]['features']}  est={best[1]['estimator']}  MAE={best[0]:.4f}")
    return rows, best


def bioact_block():
    d = np.load(HERE / "bioact_loo_components.npz", allow_pickle=True)
    y = d["y_true"]; fams = d["families"]; ms = d["max_sim"]
    direct = d["direct"]; analog = d["analog"]
    lion = d["lion"]; admet = d["admet"]
    lb = LabelBinarizer().fit(fams); fam_oh = lb.transform(fams)

    ALPHA_PER_FAM = {
        "TT-Dendrimer": 1.0, "PE-Tris": 1.0, "G1-Janus-Dendrimer": 1.0,
        "Dialkoxybenzyl": 0.5, "HTM-Dendrimer": 1.0, "sSS-Nonsym": 0.8,
        "GA-Tris": 1.0,
    }
    pred_base = np.zeros_like(y)
    for f, a in ALPHA_PER_FAM.items():
        m = fams == f
        pred_base[m] = a * direct[m] + (1 - a) * analog[m]
    base_mae = _mae(pred_base, y)
    print("\n" + "=" * 60)
    print(f"bioact stacker sweep  v14 per-fam α baseline MAE = {base_mae:.4f}")
    print("=" * 60)

    feature_sets = {
        "DA":         np.column_stack([direct, analog]),
        "DALM":       np.column_stack([direct, analog, lion, admet]),
        "DALM+sim":   np.column_stack([direct, analog, lion, admet, ms]),
        "DALM+sim+fam": np.column_stack([direct, analog, lion, admet, ms, fam_oh]),
    }
    estimators = [
        ("xgb_d2_n100", lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            random_state=42, n_jobs=1)),
        ("xgb_d3_n200", lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42, n_jobs=1)),
        ("xgb_d3_n300_lr0.03", lambda: xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            random_state=42, n_jobs=1)),
        ("xgb_d4_n300_lr0.03", lambda: xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            random_state=42, n_jobs=1)),
        ("xgb_d2_n500_lr0.02", lambda: xgb.XGBRegressor(
            n_estimators=500, max_depth=2, learning_rate=0.02,
            random_state=42, n_jobs=1)),
        ("ridge_a1.0", lambda: Ridge(alpha=1.0)),
        ("ridge_a5.0", lambda: Ridge(alpha=5.0)),
        ("linreg",     lambda: LinearRegression()),
    ]

    best = (np.inf, None)
    rows = []
    for fname, X in feature_sets.items():
        for ename, fact in estimators:
            mae, per_fam, _ = _cv_stacker(X, y, fams, fact)
            row = {"features": fname, "estimator": ename, "cv_mae": mae,
                   "per_family": per_fam}
            rows.append(row)
            print(f"  feat={fname:18s} est={ename:22s} MAE={mae:.4f}")
            if mae < best[0]:
                best = (mae, row)
    print(f"\n  BEST: feat={best[1]['features']}  est={best[1]['estimator']}  MAE={best[0]:.4f}")
    return rows, best, base_mae


def main():
    pka_rows, pka_best = pka_block()
    bioact_rows, bioact_best, bioact_baseline = bioact_block()
    out = {
        "pka": {"rows": pka_rows, "best": pka_best,
                 "baseline_mae": 0.0653},
        "bioact": {"rows": bioact_rows, "best": bioact_best,
                    "baseline_mae": bioact_baseline},
    }
    with open(HERE / "weight_eval_v2_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {HERE / 'weight_eval_v2_results.json'}")


if __name__ == "__main__":
    main()
