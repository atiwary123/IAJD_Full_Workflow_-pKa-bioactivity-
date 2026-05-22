"""
weight_eval.py — honest stratified 5-fold CV of dynamic blend functions for
both the pKa and bioactivity workflows. Each fold:

    1. Hold out 20% of the compounds (stratified by family)
    2. Fit weight-function hyperparameters on the remaining 80%'s LOO
       component predictions
    3. Apply the fitted function to the held-out 20% (using their pre-
       computed LOO component predictions — no leakage)
    4. Record per-fold residuals

Pooled across folds gives an honest generalization estimate for each
candidate weight function.

Candidates:
  pKa       — bounded sigmoid; constrained-monotonic piecewise;
              softmax_lin with L2; XGBoost stacker
  bioact    — per-family static 4-simplex (D, A, L, M);
              per-family α + global similarity sigmoid;
              XGBoost stacker
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mae(p, y): return float(np.mean(np.abs(p - y)))


def _simplex(p):
    e = np.exp(np.asarray(p) - np.max(p)); e = e / e.sum(); return e


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# pKa: weight functions and CV evaluators
# ---------------------------------------------------------------------------

def pka_w_sigmoid_bounded(params, s):
    """w_a = sigmoid(k (s - s0)); rest split (d, m) by softmax(q, r).
    k bounded ∈ [3, 25], s0 ∈ [0.4, 0.85] via squashing."""
    k_raw, s0_raw, q, r = params
    k = 3.0 + 22.0 * _sigmoid(k_raw)      # in (3, 25)
    s0 = 0.4 + 0.45 * _sigmoid(s0_raw)    # in (0.4, 0.85)
    w_a = _sigmoid(k * (s - s0))
    rest = 1.0 - w_a
    z = _simplex(np.array([q, r]))
    return rest * z[0], w_a, rest * z[1]


def pka_w_piecewise(params, s):
    """w_a piecewise-linear over s ∈ {0, .5, 1}, MONOTONIC: w0 ≤ w1 ≤ 1.
    Remaining mass split between (direct, molgpka) by a single ratio."""
    w_low_raw, w_mid_raw, ratio_raw = params
    w_low = _sigmoid(w_low_raw)
    # monotonic: w_mid >= w_low
    w_mid = w_low + (1.0 - w_low) * _sigmoid(w_mid_raw)
    # anchor at s=1 to 1.0 (pure analog when nothing else matters)
    anchors = np.array([0.0, 0.5, 1.0])
    weights = np.array([w_low, w_mid, 1.0])
    w_a = np.interp(s, anchors, weights)
    ratio_direct = _sigmoid(ratio_raw)
    rest = 1.0 - w_a
    return rest * ratio_direct, w_a, rest * (1.0 - ratio_direct)


def pka_w_softmax_lin_reg(params, s, l2=0.001):
    """softmax over (a_i + b_i s) for i ∈ {d, a, m}. Returns 3 weight arrays.
    L2 penalty applied inside the loss to prevent overfit."""
    a_d, b_d, a_a, b_a, a_m, b_m = params
    L = np.stack([a_d + b_d * s, a_a + b_a * s, a_m + b_m * s], axis=1)
    L -= L.max(axis=1, keepdims=True)
    e = np.exp(L); e /= e.sum(axis=1, keepdims=True)
    return e[:, 0], e[:, 1], e[:, 2]


# ---------------------------------------------------------------------------
# bioact: weight functions
# ---------------------------------------------------------------------------

def bioact_w_perfam_static(params, s, families, fam_list):
    """Per-family static 4-simplex. params layout:
        for each family f in fam_list, 3 unconstrained reals → simplex over 4 wts
    Returns (w_d, w_a, w_l, w_m) arrays, each (n,)."""
    n = len(s)
    n_fam = len(fam_list)
    P = np.asarray(params).reshape(n_fam, 3)
    out_d = np.zeros(n); out_a = np.zeros(n); out_l = np.zeros(n); out_m = np.zeros(n)
    for fi, fam in enumerate(fam_list):
        # 3 logits + a fixed 0 (pivot) for 4-way softmax
        logits = np.array([P[fi, 0], P[fi, 1], P[fi, 2], 0.0])
        w = _simplex(logits)
        mask = (families == fam)
        out_d[mask] = w[0]; out_a[mask] = w[1]
        out_l[mask] = w[2]; out_m[mask] = w[3]
    return out_d, out_a, out_l, out_m


def bioact_w_perfam_simmod(params, s, families, fam_list):
    """Per-family base simplex (3 free per family) + global sim-sigmoid that
    scales w_analog. Rest redistributed proportionally to base (w_d, w_l, w_m).

    params: [k_raw, s0_raw, family params (3 each)...]"""
    n = len(s)
    n_fam = len(fam_list)
    k = 3.0 + 22.0 * _sigmoid(params[0])
    s0 = 0.4 + 0.45 * _sigmoid(params[1])
    P = np.asarray(params[2:]).reshape(n_fam, 3)
    g_s = _sigmoid(k * (s - s0))  # in [0, 1]
    out_d = np.zeros(n); out_a = np.zeros(n); out_l = np.zeros(n); out_m = np.zeros(n)
    for fi, fam in enumerate(fam_list):
        logits = np.array([P[fi, 0], P[fi, 1], P[fi, 2], 0.0])
        w = _simplex(logits)            # base (d, a, l, m)
        mask = (families == fam)
        # Modulate: w_a_eff = w_a * g_s, rest scaled to keep sum = 1
        w_a_eff = w[1] * g_s[mask]
        rest_sum = 1.0 - w_a_eff
        base_rest = w[0] + w[2] + w[3]
        if base_rest <= 0:
            out_d[mask] = (1 - g_s[mask]) / 3
            out_a[mask] = w_a_eff
            out_l[mask] = (1 - g_s[mask]) / 3
            out_m[mask] = (1 - g_s[mask]) / 3
            continue
        out_d[mask] = rest_sum * (w[0] / base_rest)
        out_a[mask] = w_a_eff
        out_l[mask] = rest_sum * (w[2] / base_rest)
        out_m[mask] = rest_sum * (w[3] / base_rest)
    return out_d, out_a, out_l, out_m


# ---------------------------------------------------------------------------
# Cross-validation harness
# ---------------------------------------------------------------------------

def _fit_weights_to_data(loss_fn, x0, restarts=8, seed=0):
    """Optimize a loss with Nelder-Mead + random restarts. Returns best params."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    x0 = np.asarray(x0, dtype=float)
    starts = [x0] + [x0 + rng.normal(0, 0.6, x0.shape) for _ in range(restarts - 1)]
    for start in starts:
        try:
            res = minimize(loss_fn, start, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000})
        except Exception:
            continue
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


def _cv_evaluate(name, build_loss, predict_fn, x0, comps, n_splits=5, seed=0):
    """build_loss(tr_idx, full_dict) -> closure(params) -> scalar loss
       predict_fn(params, te_idx, full_dict) -> predicted y values for te_idx
    """
    y = comps["y"]; fams = comps["fams"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds_all = np.zeros_like(y)
    fitted_params = []
    for tr_idx, te_idx in skf.split(y, fams):
        loss = build_loss(tr_idx, comps)
        _, p = _fit_weights_to_data(loss, x0, restarts=6, seed=seed)
        fitted_params.append(p.tolist())
        preds_all[te_idx] = predict_fn(p, te_idx, comps)
    mae = _mae(preds_all, y)
    per_fam = {}
    for f in np.unique(fams):
        m = (fams == f)
        per_fam[str(f)] = {"n": int(m.sum()), "mae": _mae(preds_all[m], y[m])}
    return {"name": name, "cv_mae": mae, "per_family": per_fam,
            "fitted_params_per_fold": fitted_params}


# ---------------------------------------------------------------------------
# pKa
# ---------------------------------------------------------------------------

def pka_run():
    d = np.load(HERE / "pka_loo_components.npz", allow_pickle=True)
    comps = {
        "direct": d["direct"], "analog": d["analog"], "molgpka": d["molgpka"],
        "max_sim": d["max_sim"], "y": d["y_true"], "fams": d["families"],
    }
    n = len(comps["y"])
    print(f"== pKa stratified 5-fold CV ==  n={n}")

    def blend3(wf, params, s, idx, c):
        w_d, w_a, w_m = wf(params, s[idx])
        return (w_d * c["direct"][idx] + w_a * c["analog"][idx]
                + w_m * c["molgpka"][idx])

    # Baseline: fixed α=0.05
    base_pred = 0.05 * comps["direct"] + 0.95 * comps["analog"]
    print(f"  baseline α=0.05 (no CV needed): MAE = {_mae(base_pred, comps['y']):.4f}")

    candidates = [
        ("sigmoid_bounded",  pka_w_sigmoid_bounded,  [0.0, 0.0, 0.0, 0.0]),
        ("piecewise_mono",   pka_w_piecewise,         [0.0, 0.0, 0.0]),
        ("softmax_lin_reg",  pka_w_softmax_lin_reg,   [0, 0, 0, 2.0, 0, 0]),
    ]
    results = {}
    for name, wf, x0 in candidates:
        def make_loss(tr_idx, c, wf=wf, name=name):
            def loss(p):
                pred = blend3(wf, p, c["max_sim"], tr_idx, c)
                err = _mae(pred, c["y"][tr_idx])
                if name == "softmax_lin_reg":
                    err += 0.0005 * float(np.sum(np.asarray(p) ** 2))
                return err
            return loss
        def predict(p, te_idx, c, wf=wf):
            return blend3(wf, p, c["max_sim"], te_idx, c)
        r = _cv_evaluate(name, make_loss, predict, x0, comps)
        results[name] = r
        print(f"  {name:22s} CV_MAE = {r['cv_mae']:.4f}")

    # XGBoost stacker
    import xgboost as xgb
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer().fit(comps["fams"])
    fam_oh = lb.transform(comps["fams"])
    X_stack = np.column_stack([
        comps["direct"], comps["analog"], comps["molgpka"], comps["max_sim"],
        fam_oh,
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds_stack = np.zeros_like(comps["y"])
    for tr, te in skf.split(comps["y"], comps["fams"]):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                              random_state=42, n_jobs=1)
        m.fit(X_stack[tr], comps["y"][tr], verbose=False)
        preds_stack[te] = m.predict(X_stack[te])
    stack_mae = _mae(preds_stack, comps["y"])
    print(f"  {'xgb_stacker':22s} CV_MAE = {stack_mae:.4f}")
    results["xgb_stacker"] = {"name": "xgb_stacker", "cv_mae": stack_mae}

    return results, {"y": comps["y"], "fams": comps["fams"]}


# ---------------------------------------------------------------------------
# bioact
# ---------------------------------------------------------------------------

def bioact_run():
    d = np.load(HERE / "bioact_loo_components.npz", allow_pickle=True)
    comps = {
        "direct": d["direct"], "analog": d["analog"],
        "lion": d["lion"], "admet": d["admet"],
        "max_sim": d["max_sim"], "y": d["y_true"], "fams": d["families"],
    }
    n = len(comps["y"])
    fam_list = sorted(np.unique(comps["fams"]).tolist())
    print(f"\n== bioact stratified 5-fold CV ==  n={n}, families={fam_list}")

    # v14 per-family α baseline (no CV)
    ALPHA_PER_FAM = {
        "TT-Dendrimer": 1.0, "PE-Tris": 1.0, "G1-Janus-Dendrimer": 1.0,
        "Dialkoxybenzyl": 0.5, "HTM-Dendrimer": 1.0, "sSS-Nonsym": 0.8,
        "GA-Tris": 1.0,
    }
    pred_baseline = np.zeros(n)
    for f, a in ALPHA_PER_FAM.items():
        m = (comps["fams"] == f)
        pred_baseline[m] = a * comps["direct"][m] + (1 - a) * comps["analog"][m]
    print(f"  v14 per-fam α baseline (no CV): MAE = {_mae(pred_baseline, comps['y']):.4f}")
    print(f"  pure direct                  : MAE = {_mae(comps['direct'], comps['y']):.4f}")

    def blend4(wf, params, idx, c, fam_list):
        w_d, w_a, w_l, w_m = wf(params, c["max_sim"][idx], c["fams"][idx], fam_list)
        return (w_d * c["direct"][idx] + w_a * c["analog"][idx]
                + w_l * c["lion"][idx] + w_m * c["admet"][idx])

    n_fam = len(fam_list)
    candidates = [
        ("perfam_static_4simplex",  bioact_w_perfam_static,  np.zeros(n_fam * 3)),
        ("perfam_simmod",          bioact_w_perfam_simmod,  np.concatenate([[0.0, 0.0], np.zeros(n_fam * 3)])),
    ]
    results = {}
    for name, wf, x0 in candidates:
        def make_loss(tr_idx, c, wf=wf, fam_list=fam_list):
            def loss(p):
                pred = blend4(wf, p, tr_idx, c, fam_list)
                return _mae(pred, c["y"][tr_idx])
            return loss
        def predict(p, te_idx, c, wf=wf, fam_list=fam_list):
            return blend4(wf, p, te_idx, c, fam_list)
        r = _cv_evaluate(name, make_loss, predict, x0, comps)
        results[name] = r
        print(f"  {name:22s} CV_MAE = {r['cv_mae']:.4f}")

    # XGBoost stacker
    import xgboost as xgb
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer().fit(comps["fams"])
    fam_oh = lb.transform(comps["fams"])
    X_stack = np.column_stack([
        comps["direct"], comps["analog"], comps["lion"], comps["admet"],
        comps["max_sim"], fam_oh,
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds_stack = np.zeros_like(comps["y"])
    for tr, te in skf.split(comps["y"], comps["fams"]):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                              random_state=42, n_jobs=1)
        m.fit(X_stack[tr], comps["y"][tr], verbose=False)
        preds_stack[te] = m.predict(X_stack[te])
    stack_mae = _mae(preds_stack, comps["y"])
    print(f"  {'xgb_stacker':22s} CV_MAE = {stack_mae:.4f}")
    results["xgb_stacker"] = {"name": "xgb_stacker", "cv_mae": stack_mae}

    return results


def main():
    pka_results, _ = pka_run()
    bioact_results = bioact_run()
    out = {"pka": pka_results, "bioact": bioact_results}
    with open(HERE / "weight_eval_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\nSaved -> {HERE / 'weight_eval_results.json'}")


if __name__ == "__main__":
    main()
