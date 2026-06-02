"""
train_per_organ.py — separate v14-style XGBoost regressor for each organ.

The bioact xlsx has measured log10_flux_lung, log10_flux_liver,
log10_flux_spleen, log10_flux_LN, log10_flux_heart in addition to the total.
For lung-targeted vs liver-targeted IAJD design, the per-organ prediction is
more actionable than aggregate flux.

For each organ:
  • Train an XGBoost regressor on the 92-feature stack (Block A..E)
  • Compute LOO MAE per family
  • Save to per_organ_bundle.pkl

No proxies — same Block E policy, NaN handled natively by XGBoost.
"""
from __future__ import annotations
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
from sample_prep_weights import maybe_bundle_weight  # default-off (IAJD_USE_PREP_WEIGHTS)
V14_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
OUT_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_per_organ_bundle.pkl"
OUT_REPORT = ROOT / "per_organ_report.json"

ORGAN_COLS = [
    "log10_flux_lung", "log10_flux_liver", "log10_flux_spleen",
    "log10_flux_LN", "log10_flux_heart",
]

XGB_HP = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=4, verbosity=0,
)


def main():
    print(f"Loading v14 bundle…", flush=True)
    with open(V14_BUNDLE, "rb") as f:
        b14 = pickle.load(f)
    X = np.asarray(b14["X_train"], dtype=float)
    smis_train = list(b14["smis_train"])
    families = np.array([str(f) for f in b14["families_train"]])
    print(f"  X.shape = {X.shape}", flush=True)

    print(f"Loading bioact xlsx for per-organ measurements…", flush=True)
    df = pd.read_excel(BIO_XLSX)
    # Match bioact_v14_pipeline.load_v13 row selection EXACTLY (incl. the
    # 2026-06-01 audit-flagged exclusion) so df aligns positionally with X_train.
    if "audit_status" in df.columns:
        df = df[~df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)]
    df = df.dropna(subset=["log10_flux_total"])
    df = df[df["SMILES_canonical"].notna()].reset_index(drop=True)
    assert len(df) == len(X), f"{len(df)} vs {len(X)}"

    w = maybe_bundle_weight(b14, BIO_XLSX)  # None unless IAJD_USE_PREP_WEIGHTS=1 (no-op)
    if w is not None:
        print(f"  [prep-weights] ON: w[min/mean/max]={w.min():.2f}/{w.mean():.2f}/{w.max():.2f}", flush=True)

    organ_bundles = {}
    organ_metrics = {}
    for organ_col in ORGAN_COLS:
        if organ_col not in df.columns:
            print(f"  {organ_col}: column missing — skipping", flush=True)
            continue
        y = df[organ_col].values.astype(float)
        mask = np.isfinite(y)
        print(f"\n[{organ_col}] n_with_measurement={mask.sum()}/{len(y)}", flush=True)
        if mask.sum() < 30:
            print(f"  too few measurements ({mask.sum()}) — skipping", flush=True)
            continue

        X_m, y_m, fams_m = X[mask], y[mask], families[mask]
        w_m = w[mask] if w is not None else None
        # LOO predictions
        loo = np.zeros(mask.sum())
        for i in range(mask.sum()):
            keep = np.arange(mask.sum()) != i
            m = xgb.XGBRegressor(**XGB_HP)
            m.fit(X_m[keep], y_m[keep],
                  sample_weight=(w_m[keep] if w_m is not None else None), verbose=False)
            loo[i] = float(m.predict(X_m[i:i+1])[0])
            if (i + 1) % 50 == 0:
                print(f"  LOO {i+1}/{mask.sum()}", flush=True)

        loo_mae = float(np.mean(np.abs(loo - y_m)))
        # Per-family LOO RMSE
        per_fam_rmse = {}
        for fam in sorted(set(fams_m)):
            fm = fams_m == fam
            if fm.sum() < 3: continue
            per_fam_rmse[fam] = float(np.sqrt(np.mean((loo[fm] - y_m[fm]) ** 2)))
        # Final model on full data for inference
        final = xgb.XGBRegressor(**XGB_HP)
        final.fit(X_m, y_m, sample_weight=w_m, verbose=False)

        organ_bundles[organ_col] = {
            "model": final,
            "loo_mae": loo_mae,
            "per_family_rmse": per_fam_rmse,
            "n_train": int(mask.sum()),
        }
        organ_metrics[organ_col] = {
            "loo_mae": loo_mae,
            "n_train": int(mask.sum()),
            "per_family_rmse": per_fam_rmse,
        }
        print(f"  LOO MAE = {loo_mae:.3f}", flush=True)
        for fam, rmse in per_fam_rmse.items():
            print(f"    {fam:<22s}  RMSE={rmse:.3f}", flush=True)

    bundle = {
        "version": "per_organ_v1_2026-05-29",
        "organ_models": organ_bundles,
        "organ_columns": ORGAN_COLS,
        "n_features": X.shape[1],
        "block_slices": b14.get("block_slices", {}),
    }
    with open(OUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    OUT_REPORT.write_text(json.dumps(organ_metrics, indent=2, default=str))
    print(f"\nSaved {OUT_BUNDLE.name} + {OUT_REPORT.name}", flush=True)


if __name__ == "__main__":
    main()
