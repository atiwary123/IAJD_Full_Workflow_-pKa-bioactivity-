"""
predict_ensemble.py — per-candidate Bayesian uncertainty inference helper.

For any query SMILES, returns:
  yhat_mean    — mean of the M ensemble models' predictions
  sigma_query  — calibration-corrected per-candidate σ from ensemble disagreement
  sigma_family — LOO RMSE for the SMILES's assigned family (fallback signal)
  sigma_eff    — effective σ used by the binary head:
                 = max(sigma_query · calibration, sigma_family) when σ_query is
                   tiny (model artificially confident); otherwise just calibrated
                 σ_query.

Use cases:
  - Single SMILES tab: feed sigma_eff into the Gaussian P(≥T) calculation
  - Proposer: use sigma_query directly to identify candidates the ensemble
    disagrees about (active-learning targets)

No proxies. The ensemble was trained on real measurements with the new Block E
sample-prep covariates. σ is real ensemble disagreement, not a constant.
"""
from __future__ import annotations
import pickle, sys, math, os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

BUNDLE_PATH = ROOT / "IAJD_master/bundles_caches/bioact_ensemble_bundle.pkl"

_BUNDLE_CACHE: Optional[Dict[str, Any]] = None


def load_ensemble_bundle() -> Dict[str, Any]:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"ensemble bundle not found at {BUNDLE_PATH}")
    with open(BUNDLE_PATH, "rb") as f:
        _BUNDLE_CACHE = pickle.load(f)
    return _BUNDLE_CACHE


def predict_ensemble(X: np.ndarray, family: Optional[str] = None,
                       bundle: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Predict mean + σ for a single query feature row X (shape: (1, n_features))."""
    b = bundle or load_ensemble_bundle()
    preds = np.array([float(m.predict(X)[0]) for m in b["models"]])
    yhat_mean = float(np.mean(preds))
    sigma_raw = float(np.std(preds))
    calibration = float(b.get("sigma_calibration", 1.0))
    sigma_query = sigma_raw * calibration
    fam_sigma = None
    if family is not None:
        fam_sigma = b["per_family_sigma"].get(family)
    if fam_sigma is None:
        fam_sigma = float(b.get("global_sigma", 0.43))
    # Effective σ: when the ensemble is unanimously confident (σ_query very low),
    # fall back to the family-level σ so we don't overstate certainty just
    # because the trees agree.
    sigma_eff = max(sigma_query, fam_sigma * 0.7)
    return {
        "yhat_mean": yhat_mean,
        "sigma_query": sigma_query,
        "sigma_raw": sigma_raw,
        "sigma_family": float(fam_sigma),
        "sigma_eff": sigma_eff,
        "calibration": calibration,
        "M": len(preds),
        "per_model_preds": preds.tolist(),
    }


def p_above_gaussian(yhat: float, sigma: float, threshold: float) -> float:
    """P(y ≥ threshold | y ~ Normal(yhat, sigma²))."""
    if sigma <= 0 or not (math.isfinite(yhat) and math.isfinite(sigma)):
        return float("nan")
    z = (yhat - threshold) / sigma
    # P(Z ≥ -z) where Z is standard normal
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def predict_with_p_above(X: np.ndarray, threshold: float,
                           family: Optional[str] = None,
                           bundle: Optional[Dict[str, Any]] = None,
                           max_tanimoto_to_training: Optional[float] = None) -> Dict[str, Any]:
    """Full inference: mean, σ_eff, P(≥threshold), OOD warnings."""
    r = predict_ensemble(X, family=family, bundle=bundle)
    p = p_above_gaussian(r["yhat_mean"], r["sigma_eff"], threshold)
    r["threshold"] = float(threshold)
    r["p_above"] = float(p) if math.isfinite(p) else None
    r["family"] = family
    r["max_tanimoto_to_training"] = (
        float(max_tanimoto_to_training) if max_tanimoto_to_training is not None else None
    )
    # OOD warning: Tanimoto < 0.5 means we're well outside the model's
    # training neighborhood and σ_eff is likely an underestimate.
    if max_tanimoto_to_training is not None and max_tanimoto_to_training < 0.5:
        r["ood_warning"] = (
            f"max Tanimoto to training = {max_tanimoto_to_training:.2f} (< 0.50): "
            "this candidate is outside the model's training neighborhood; "
            "the reported σ likely underestimates true uncertainty."
        )
    return r


if __name__ == "__main__":
    import pickle as _pkl
    b14 = _pkl.load(open(ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl", "rb"))
    X_train = np.asarray(b14["X_train"], dtype=float)
    y_train = np.asarray(b14["y_train"], dtype=float)
    families = list(b14["families_train"])

    bundle = load_ensemble_bundle()
    print(f"Ensemble bundle loaded: M={bundle['metrics']['M']}, "
          f"calibration={bundle['sigma_calibration']:.3f}", flush=True)

    print("\nSample inference on first 5 training rows:")
    for i in range(5):
        r = predict_with_p_above(
            X_train[i:i+1], threshold=8.0, family=families[i],
            max_tanimoto_to_training=1.0,   # they're in training
        )
        actual = y_train[i]
        print(f"  row {i}: family={families[i]:<22s}  actual={actual:.2f}  "
              f"ŷ={r['yhat_mean']:.2f}  σ_query={r['sigma_query']:.3f}  "
              f"σ_family={r['sigma_family']:.3f}  σ_eff={r['sigma_eff']:.3f}  "
              f"P(≥8.0)={r['p_above']:.0%}")
