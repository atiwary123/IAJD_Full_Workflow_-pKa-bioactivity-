"""Persist v11 M2 (pKa-dominant) bioactivity model for HF inference.

Refits the exact M2 spec from pka_flux_v11.py on the full bioact table:
  features = [predicted_pKa]
           + family one-hot (8 families)
           + (predicted_pKa × family) interaction (8 cols)
           + tail descriptors (8 cols)

The bundle saves:
  - model            : fitted xgb.XGBRegressor (GBR_HP, depth-4)
  - feature_columns  : exact column order at training time
  - feature_sha      : SHA-256 of the column list (train/serve consistency)
  - family_order     : the 8 families in their one-hot order
  - tail_descriptors : the 8 tail descriptor column names
  - tail_medians     : median value per tail descriptor (for NaN imputation
                       at inference, computed on training rows)
  - loo_mae          : the family-stratified k=5 LOO MAE for context
  - notes            : free-text label that will surface in the UI

Output: v11_pka_flux/m2_bundle.joblib
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v11_pka_flux"))
from pka_flux_v11 import (  # noqa: E402
    GBR_HP, TAIL_DESCRIPTORS, FAMILIES_ORDER,
    build_features_m2, feature_sha,
)

DATA = ROOT / "IAJD_master" / "datasets"
OUT = ROOT / "v11_pka_flux" / "m2_bundle.joblib"
PKA_CACHE = ROOT / "v11_pka_flux" / "predicted_pka_cache.csv"


EXCLUDED_NOVEL_IAJDS = {347, 348, 365, 366, 367, 369, 372, 373}


def main():
    bio = pd.read_excel(DATA / "IAJD_Bioact_v13_clean.xlsx")
    if "IAJD_num" in bio.columns:
        bio = bio[~bio["IAJD_num"].isin(EXCLUDED_NOVEL_IAJDS)].reset_index(drop=True)
    pka_df = pd.read_csv(PKA_CACHE)
    merged = bio.merge(
        pka_df[["row_id", "predicted_pKa", "pka_source"]],
        on="row_id", how="left",
    )
    full = merged.dropna(subset=["log10_flux_total", "predicted_pKa"]).reset_index(drop=True)

    X, cols = build_features_m2(full)
    y = full["log10_flux_total"].to_numpy(dtype=float)
    sha = feature_sha(cols)

    print(f"Training M2 on {len(y)} rows, {X.shape[1]} features, sha={sha}")
    model = xgb.XGBRegressor(**GBR_HP)
    model.fit(X, y, verbose=False)

    # Compute tail descriptor medians on the training rows (for NaN
    # imputation at inference)
    tail_medians = {}
    for c in TAIL_DESCRIPTORS:
        tail_medians[c] = float(np.nanmedian(full[c].to_numpy(dtype=float)))

    metrics = json.loads((ROOT / "v11_pka_flux" / "metrics.json").read_text())
    loo_mae = metrics["pooled"]["M2"]["MAE"]

    bundle = {
        "version": "v11-M2-pKa-dominant",
        "model": model,
        "feature_columns": cols,
        "feature_sha": sha,
        "family_order": FAMILIES_ORDER,
        "tail_descriptors": TAIL_DESCRIPTORS,
        "tail_medians": tail_medians,
        "loo_mae": float(loo_mae),
        "n_train_rows": int(len(y)),
        "notes": (
            "pKa-dominant (M2): pKa + family + pKa×family + 8 tail descriptors. "
            "Family-stratified k=5 LOO MAE 0.497 vs v14 0.430 (v14 wins by 0.067). "
            "Shown as an independent baseline alongside v14+stacker, not as the "
            "primary prediction."
        ),
    }
    joblib.dump(bundle, OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
