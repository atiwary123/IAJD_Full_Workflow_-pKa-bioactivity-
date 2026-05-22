"""
bioact_weight_sweep.py — dynamic-weight blends over four LOO heads:
    direct (full-feature XGB), analog (Tanimoto kNN consensus),
    lion (LION-only XGB),     admet (ADMET-only XGB).

Same scaffolding as pka_weight_sweep.py, generalized to 4 components.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
NPZ = HERE / "bioact_loo_components.npz"


def _load():
    d = np.load(NPZ, allow_pickle=True)
    return {
        "direct": d["direct"], "analog": d["analog"],
        "lion": d["lion"], "admet": d["admet"],
        "max_sim": d["max_sim"], "y_true": d["y_true"],
        "families": d["families"],
    }


def _mae(p, y): return float(np.mean(np.abs(p - y)))


def _per_fam_mae(p, y, fams):
    out = {}
    for f in np.unique(fams):
        m = fams == f
        out[str(f)] = {"n": int(m.sum()), "mae": _mae(p[m], y[m])}
    return out


def _report(label, pred, data):
    pooled = _mae(pred, data["y_true"])
    per_fam = _per_fam_mae(pred, data["y_true"], data["families"])
    fam_line = " ".join(f"{f[:4]}={per_fam[f]['mae']:.3f}(n={per_fam[f]['n']})"
                         for f in sorted(per_fam))
    print(f"  {label:25s} pooled={pooled:.4f}   {fam_line}")
    return pooled, per_fam


# ---------------------------------------------------------------------------
# Weight functions: each returns (w_direct, w_analog, w_lion, w_admet) per row
# ---------------------------------------------------------------------------

def w_baseline_v14_per_fam(params, s, families, alpha_per_fam):
    """v14 baseline: per-family alpha blending direct vs analog, LION/ADMET in zeros."""
    n = len(s)
    w_d = np.zeros(n); w_a = np.zeros(n)
    for fam, a in alpha_per_fam.items():
        m = (families == fam)
        w_d[m] = a; w_a[m] = 1 - a
    return w_d, w_a, np.zeros(n), np.zeros(n)


def w_static_4(params, s):
    """4-simplex via softmax."""
    e = np.exp(params - np.max(params)); e /= e.sum()
    n = len(s)
    return np.full(n, e[0]), np.full(n, e[1]), np.full(n, e[2]), np.full(n, e[3])


def w_sigmoid_analog(params, s):
    """w_a = sigmoid(k (s - s0)); rest distributed (d, l, a) by softmax over 3 logits."""
    k, s0, q_d, q_l, q_m = params
    w_a = 1.0 / (1.0 + np.exp(-k * (s - s0)))
    rest = 1.0 - w_a
    z = np.exp([q_d, q_l, q_m] - np.max([q_d, q_l, q_m])); z = z / z.sum()
    return rest * z[0], w_a, rest * z[1], rest * z[2]


def w_softmax_lin(params, s):
    """softmax over (a_i + b_i · s) for i in 4 components. 8 params."""
    coefs = np.asarray(params).reshape(4, 2)
    L = coefs[:, 0:1] + coefs[:, 1:2] * s[None, :]
    L = L.T  # (n, 4)
    L = L - L.max(axis=1, keepdims=True)
    e = np.exp(L); e = e / e.sum(axis=1, keepdims=True)
    return e[:, 0], e[:, 1], e[:, 2], e[:, 3]


def w_piecewise(params, s):
    """w_a piecewise-linear over s∈{0,.3,.6,.85,1.0}. Remaining mass split
       three ways (d, l, m) by 3-softmax over fixed anchor."""
    anchors = np.array([0.0, 0.3, 0.6, 0.85, 1.0])
    w_a_anch = np.clip(np.asarray(params[:5]), 0.0, 1.0)
    q = np.asarray(params[5:8])
    z = np.exp(q - q.max()); z = z / z.sum()
    w_a = np.interp(s, anchors, w_a_anch)
    rest = 1.0 - w_a
    return rest * z[0], w_a, rest * z[1], rest * z[2]


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _blend(data, wf, params):
    s = data["max_sim"]
    if wf is w_baseline_v14_per_fam:
        # special case — needs alpha_per_fam
        w_d, w_a, w_l, w_m = wf(params, s, data["families"], _ALPHA_PER_FAM)
    else:
        w_d, w_a, w_l, w_m = wf(params, s)
    return (w_d * data["direct"] + w_a * data["analog"]
            + w_l * data["lion"] + w_m * data["admet"])


def _opt(name, wf, x0, data, restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    x0 = np.asarray(x0, dtype=float)
    starts = [x0] + [x0 + rng.normal(0, 0.5, x0.shape) for _ in range(restarts - 1)]
    for start in starts:
        def loss(p):
            return _mae(_blend(data, wf, p), data["y_true"])
        try:
            res = minimize(loss, start, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000})
        except Exception:
            continue
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


_ALPHA_PER_FAM = {
    "TT-Dendrimer": 1.0, "PE-Tris": 1.0, "G1-Janus-Dendrimer": 1.0,
    "Dialkoxybenzyl": 0.5, "HTM-Dendrimer": 1.0, "sSS-Nonsym": 0.8,
    "GA-Tris": 1.0,
}


def main():
    data = _load()
    n = len(data["y_true"])
    print(f"Loaded LOO components for n={n} compounds.\n")

    print("Single-head baselines:")
    _report("pure direct", data["direct"], data)
    _report("pure analog", data["analog"], data)
    _report("pure lion (block B)", data["lion"], data)
    _report("pure admet (block C)", data["admet"], data)
    print()

    print("Reference blend baselines:")
    _report("v14 per-fam alpha (D vs A)",
            _blend(data, w_baseline_v14_per_fam, None), data)
    _report("naive 50/50 D+A", 0.5*data["direct"] + 0.5*data["analog"], data)
    print()

    print("Optimized candidates (Nelder-Mead, 8 restarts):")
    candidates = [
        ("static_4comp", w_static_4,       [0.5, 0.5, 0.0, 0.0]),
        ("sigmoid_4",    w_sigmoid_analog, [10.0, 0.6, 0.0, 0.0, 0.0]),
        ("piecewise_4",  w_piecewise,      [0.1, 0.4, 0.6, 0.8, 0.9, 0.0, 0.0, 0.0]),
        ("softmax_lin",  w_softmax_lin,    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    results = {}
    for name, fn, x0 in candidates:
        mae, params = _opt(name, fn, x0, data, restarts=8)
        pooled, per_fam = _report(name, _blend(data, fn, params), data)
        results[name] = {"mae": pooled, "params": params.tolist(),
                          "per_family": per_fam}

    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"\n  WINNER: {best_name}  pooled={results[best_name]['mae']:.4f}")
    print(f"  params: {results[best_name]['params']}")

    out_path = HERE / "bioact_weight_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_v14_loo_mae": 0.4006,
            "results": results,
            "winner": best_name,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
