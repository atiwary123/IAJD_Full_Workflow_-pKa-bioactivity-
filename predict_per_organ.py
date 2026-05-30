"""
predict_per_organ.py — inference helper for the per-organ XGBoost predictors.

For any query SMILES, returns the predicted log10 flux for each organ
(lung, liver, spleen, LN, heart) using a separate XGBoost regressor trained
on the same 92-feature stack as v14.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

BUNDLE_PATH = ROOT / "IAJD_master/bundles_caches/bioact_per_organ_bundle.pkl"
_CACHE: Optional[Dict[str, Any]] = None


def load_per_organ_bundle() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"per-organ bundle not found at {BUNDLE_PATH}")
    with open(BUNDLE_PATH, "rb") as f:
        _CACHE = pickle.load(f)
    return _CACHE


def predict_per_organ(X: np.ndarray, family: Optional[str] = None,
                       bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
    """For a single query feature row X (shape: (1, n_features)), return:
        {
          "lung":   {"yhat": float, "sigma_family": float},
          "liver":  {...},
          ...
        }
    """
    b = bundle or load_per_organ_bundle()
    out = {}
    for organ_col, info in b["organ_models"].items():
        yhat = float(info["model"].predict(X)[0])
        sigma = info["per_family_rmse"].get(family) if family else None
        if sigma is None:
            sigma = float(info.get("loo_mae", float("nan")))
        organ_name = organ_col.replace("log10_flux_", "")
        out[organ_name] = {
            "yhat": yhat, "sigma_family": float(sigma),
            "loo_mae": info["loo_mae"],
            "n_train": info["n_train"],
        }
    return out
