"""v11 pKa-dominant bioactivity predictor — inference-only wrapper.

Loads the M2 bundle produced by `v11_pka_flux/build_m2_bundle.py` and exposes
a single function `predict_v11_bioact(canonical_smiles, family, predicted_pKa)`
that returns the predicted log10 flux total along with metadata for the UI.

This is an INDEPENDENT baseline to the v14+stacker stack. It is wired into
`app.py` so the Space can show both predictions side-by-side. v14 remains
the primary prediction (lower LOO MAE 0.430 vs M2 0.497).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE_PATH = HERE / "v11_pka_flux" / "m2_bundle.joblib"

_BUNDLE: Dict[str, Any] | None = None


def _load_bundle() -> Dict[str, Any]:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(BUNDLE_PATH)
    return _BUNDLE


def is_available() -> bool:
    """True iff the M2 bundle file exists and loads cleanly."""
    if not BUNDLE_PATH.exists():
        return False
    try:
        _load_bundle()
        return True
    except Exception:  # noqa: BLE001
        return False


def _build_query_row(family: str, predicted_pKa: float,
                      tail_lookup: Dict[str, float] | None,
                      bundle: Dict[str, Any]) -> np.ndarray:
    fam_order: List[str] = bundle["family_order"]
    tail_cols: List[str] = bundle["tail_descriptors"]
    tail_medians: Dict[str, float] = bundle["tail_medians"]
    onehot = np.array([1.0 if f == family else 0.0 for f in fam_order])
    inter = onehot * float(predicted_pKa)
    tail = np.array([
        float(tail_lookup.get(c, np.nan)) if tail_lookup else np.nan
        for c in tail_cols
    ])
    nan_mask = ~np.isfinite(tail)
    if nan_mask.any():
        for i, c in enumerate(tail_cols):
            if nan_mask[i]:
                tail[i] = tail_medians[c]
    return np.concatenate([[float(predicted_pKa)], onehot, inter, tail])


def predict_v11_bioact(canonical_smiles: str, family: str,
                        predicted_pKa: float,
                        tail_descriptors: Dict[str, float] | None = None
                        ) -> Dict[str, Any]:
    """Predict log10 flux total under the v11 M2 pKa-dominant model.

    Args:
      canonical_smiles: SMILES of the query (returned in the result for
        bookkeeping; the model itself does not use it directly).
      family: family label (one of bundle.family_order). Unknown families
        will not match the one-hot and the model will rely on tail
        descriptors + raw pKa only.
      predicted_pKa: the v9.1 predicted pKa for this molecule.
      tail_descriptors: optional dict mapping descriptor name → value
        (MolLogP, FractionCSP3, RotatableBonds, NumAromaticRings, TPSA,
        LabuteASA, HeavyAtomCount, linker_carbons). Missing keys are
        median-imputed from the training distribution.

    Returns:
      dict with keys: point, version, loo_mae, n_train_rows, feature_sha,
      family_used, predicted_pka_used, in_family_set, notes.
    """
    if not is_available():
        return {"error": "v11_bundle_unavailable", "point": None}
    bundle = _load_bundle()
    fam_order: List[str] = bundle["family_order"]
    fam_used = family if family in fam_order else None
    x = _build_query_row(fam_used or "", float(predicted_pKa),
                          tail_descriptors, bundle)
    pred = float(bundle["model"].predict(x.reshape(1, -1))[0])
    return {
        "smiles": canonical_smiles,
        "version": bundle["version"],
        "point": round(pred, 3),
        "linear_flux": round(10 ** pred, 3),
        "loo_mae": bundle["loo_mae"],
        "n_train_rows": bundle["n_train_rows"],
        "feature_sha": bundle["feature_sha"],
        "family_used": fam_used,
        "predicted_pka_used": float(predicted_pKa),
        "in_family_set": fam_used is not None,
        "notes": bundle["notes"],
    }


# Helper to pull tail descriptors from an RDKit Mol so the HF app can feed
# them in without depending on the training-time bioact-table columns.
def tail_descriptors_from_mol(mol) -> Dict[str, float]:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    return {
        "MolLogP": float(Crippen.MolLogP(mol)),
        "FractionCSP3": float(Lipinski.FractionCSP3(mol)),
        "RotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "NumAromaticRings": float(Lipinski.NumAromaticRings(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        # linker_carbons can't be inferred cheaply; let it median-impute.
        "linker_carbons": float("nan"),
    }
