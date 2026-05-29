"""
predict_v15.py — inference helper for the v15 hybrid ML+physics predictor.

Exposes:
    predict_v15(smiles, family_hint=None, pka=None, linker_length=None,
                  n_tail_chains=None, kappa_ucb=1.0)
        -> dict with yhat_ml, yhat_physics, yhat_blend, sigma_ml, sigma_physics,
           ucb_score, physics_quality, components

The proposer uses ucb_score (= ŷ + κ·σ) to rank extrapolation candidates that
either the ML predictor likes OR that the model is uncertain about but
physics says are favorable.
"""
from __future__ import annotations
import os, sys, pickle
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from physics_features import compute_all_physics_features

V15_BUNDLE = ROOT / "IAJD_master/bundles_caches/v15_hybrid_bundle.joblib"
V14_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
STACKER_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_stacker_bundle.pkl"

_BUNDLE_CACHE: Optional[Dict[str, Any]] = None


def load_v15_bundle() -> Dict[str, Any]:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    if not V15_BUNDLE.exists():
        raise FileNotFoundError(f"v15 bundle not found at {V15_BUNDLE}")
    _BUNDLE_CACHE = joblib.load(V15_BUNDLE)
    return _BUNDLE_CACHE


def physics_quality(features: dict, pka: float = 6.5) -> float:
    """Mechanism-based quality score for fusogenic IAJD-DNP delivery.

    Combines:
      - CPP proximity to 1 (lamellar/fusogenic optimum):   bell-curve, peak at CPP=1
      - Endosomal protonation differential:                 monotone, [0..1]
      - HLB in optimal surfactant range [6, 11]:            bell-curve, peak at 8.5
      - logKp_membrane > 4 (membrane-affine):               sigmoid

    Returns Q ∈ [0, 1].
    """
    cpp = features.get("cpp_geometric")
    if cpp is None or not np.isfinite(cpp):
        cpp = 1.0
    # Bell curve centered at CPP=1 with σ=0.3
    cpp_score = float(np.exp(-((cpp - 1.0) ** 2) / (2 * 0.3 ** 2)))

    p_e = features.get("protonation_endosome", 0.5)
    p_c = features.get("protonation_cytosol", 0.1)
    if not np.isfinite(p_e):
        p_e = 0.5
    if not np.isfinite(p_c):
        p_c = 0.1
    escape_score = max(0.0, p_e - p_c)

    hlb = features.get("hlb_griffin")
    if hlb is None or not np.isfinite(hlb):
        hlb = 8.5
    hlb_score = float(np.exp(-((hlb - 8.5) ** 2) / (2 * 3.0 ** 2)))

    logKp = features.get("logKp_membrane", 5.0)
    if not np.isfinite(logKp):
        logKp = 5.0
    membrane_score = 1.0 / (1.0 + np.exp(-(logKp - 4.0)))   # sigmoid

    Q = (cpp_score + escape_score + hlb_score + membrane_score) / 4.0
    return float(Q)


def _ml_yhat(smiles: str, family_hint: Optional[str] = None) -> tuple:
    """Run the v14+stacker ML predictor; return (ŷ_ml, σ_ml)."""
    try:
        from predict_binary import _assemble_X, load_binary_bundle
        bin_b = load_binary_bundle()
        X = _assemble_X([smiles], family_hint=family_hint or "GA-Tris")
        yhat = float(bin_b["regressor"].predict(X)[0])
        # σ_ml ≈ in-sample residual std stored in bundle, fallback 0.32
        sigma = 0.32
        return yhat, sigma
    except Exception:
        return float("nan"), 0.32


def predict_v15(smiles: str, family_hint: Optional[str] = None,
                pka: Optional[float] = None,
                linker_length: Optional[int] = None,
                n_tail_chains: int = 3,
                chain_avg_carbons: float = 10.0,
                kappa_ucb: float = 1.0) -> Dict[str, Any]:
    """Hybrid v15 prediction with UCB acquisition score.

    Returns a dict with all components so callers (e.g. the proposer table)
    can show per-layer breakdown.
    """
    b = load_v15_bundle()

    # ML layer
    yhat_ml, sigma_ml = _ml_yhat(smiles, family_hint)

    # Physics layer
    if pka is None:
        # quick v9.2 pKa lookup if available; otherwise default 6.5
        try:
            from predict_pka_v92 import predict_pka_v92
            r = predict_pka_v92(smiles, family_hint=family_hint)
            pka = r.get("pKa_pred") if r and not r.get("error") else 6.5
        except Exception:
            pka = 6.5

    feats = compute_all_physics_features(
        smiles, pka=pka, linker_length=linker_length or 4,
        n_tail_chains=n_tail_chains, chain_avg_carbons=chain_avg_carbons,
    )
    cols = b["physics_cols"]
    X_phys = np.array([[feats.get(c, np.nan) for c in cols]], dtype=float)
    for c in range(X_phys.shape[1]):
        if not np.isfinite(X_phys[0, c]):
            X_phys[0, c] = 0.0
    Xs = b["physics_scaler"].transform(X_phys)
    yhat_physics = float(b["physics_model"].predict(Xs)[0])

    # Quality / mechanism score
    Q = physics_quality(feats, pka=pka)

    # Blend
    w_ml = b["weights"]["ml"]
    w_phys = b["weights"]["physics"]
    yhat_blend = w_ml * yhat_ml + w_phys * yhat_physics

    # σ for UCB: combine in-layer noise with inter-layer disagreement
    sigma_phys = b["sigma"]["physics"]
    sigma_disagreement = b["sigma"]["disagreement"]
    sigma_combined = float(np.sqrt(
        (w_ml * sigma_ml) ** 2
        + (w_phys * sigma_phys) ** 2
        + (sigma_disagreement * abs(yhat_ml - yhat_physics) / 4.0) ** 2
    ))

    ucb_score = yhat_blend + kappa_ucb * sigma_combined * Q

    return {
        "yhat_ml": round(yhat_ml, 3),
        "yhat_physics": round(yhat_physics, 3),
        "yhat_blend": round(yhat_blend, 3),
        "sigma_ml": round(sigma_ml, 3),
        "sigma_physics": round(sigma_phys, 3),
        "sigma_combined": round(sigma_combined, 3),
        "sigma_disagreement": round(sigma_disagreement, 3),
        "physics_quality": round(Q, 3),
        "ucb_score": round(ucb_score, 3),
        "weights": {"ml": w_ml, "physics": w_phys},
        "pka_used": pka,
        "physics_features": {k: round(feats.get(k, 0), 3) for k in cols},
    }


if __name__ == "__main__":
    import json
    sm = "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"
    r = predict_v15(sm, family_hint="GA-Tris", linker_length=2, n_tail_chains=3,
                     chain_avg_carbons=9.0, kappa_ucb=1.0)
    print(json.dumps(r, indent=2, default=str))
