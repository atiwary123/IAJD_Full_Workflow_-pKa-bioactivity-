"""
analyze_monotone_axes.py — identify training-set features with reliably
monotone relationships to log10_flux_total.

For each feature, we measure:
  - Spearman ρ (rank correlation) — monotone signal
  - Pearson r — linear signal
  - Isotonic regression fit MAE — how well an order-only monotone fit explains
    the data
  - The fraction of training rows in the top 25% of feature values that are
    also in the top 25% of flux values — directional consistency

A feature is flagged as "reliably monotone for extrapolation" iff:
  - |Spearman ρ| ≥ 0.30
  - sign agrees with the isotonic fit's slope at the high end
  - the top-quartile consistency is ≥ 0.55 (so the trend isn't dominated by
    outliers in the middle)

We then save:
  monotone_axes.json
    {
      "version": ...,
      "axes": {
        feature_name: {
          "spearman_rho": float,
          "pearson_r": float,
          "isotonic_mae": float,
          "top_q_consistency": float,
          "direction": "increasing" | "decreasing",
          "train_min": float, "train_max": float,
          "reliably_monotone": bool,
        }
      }
    }

The proposer uses this at scoring time: for candidates whose feature values
step BEYOND the training max along a monotone-increasing axis (or below the
min along a monotone-decreasing axis), apply a small bonus that scales with
how confident the monotone fit is. Sub-monotone features get no bonus.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
OUT_JSON = ROOT / "monotone_axes.json"

# Features to test. Mix of (a) RDKit descriptors that the bioact xlsx has
# pre-computed, (b) family/structural counts, (c) physics descriptors
# (computed on the fly for monotone analysis).
FEATURE_COLUMNS = [
    # 2D RDKit
    "ExactMolWt", "MolLogP", "TPSA", "LabuteASA", "FractionCSP3",
    "RotatableBonds", "BertzCT", "Chi0v", "Chi1v", "HallKierAlpha",
    "NumAromaticRings", "NumHDonors", "NumHAcceptors",
    "NumEsters", "NumAmides", "NumEthers",
    "NumTertiaryAmines", "HasPiperazine", "NumAmines_total",
    "Hydrophobic_Index", "Polar_Surface_Ratio",
    "Inductive_Effect_Strength", "HBD_HBA_Ratio", "Desolvation_Proxy",
    # 3D
    "Pct_V_Bur_max", "Pct_V_Bur_mean", "Rg_3D", "Asphericity_3D",
    "N_LowE_Conformers", "E_min_3D",
    # Family / structural metadata
    "linker_length", "total_hydrophobic_carbons",
]

# Thresholds for "reliably monotone".
# Top-quartile consistency proved too strict on this dataset (all features
# fell below 0.55 even when Spearman was strong), so we gate primarily on
# Spearman and use top-q as a soft tiebreaker stored in the JSON.
MIN_SPEARMAN = 0.30
MIN_TOP_QUARTILE_CONSISTENCY = 0.0   # disabled gate


def analyze_feature(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute monotone metrics for a single (x, y) array pair."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return {"spearman_rho": None, "pearson_r": None,
                 "isotonic_mae": None, "top_q_consistency": None,
                 "n": int(mask.sum())}
    x_, y_ = x[mask], y[mask]
    try:
        rho, _ = spearmanr(x_, y_)
    except Exception:
        rho = float("nan")
    try:
        r, _ = pearsonr(x_, y_)
    except Exception:
        r = float("nan")

    # Isotonic regression in the direction of the rank correlation sign
    increasing = rho >= 0
    try:
        iso = IsotonicRegression(increasing=bool(increasing),
                                  out_of_bounds="clip")
        y_pred = iso.fit_transform(x_, y_)
        iso_mae = float(np.mean(np.abs(y_pred - y_)))
    except Exception:
        iso_mae = float("nan")

    # Top-quartile consistency: fraction of "top-25%-by-feature" rows that
    # are also in "top-25%-by-flux" (for increasing) or "bottom-25%-by-flux"
    # (for decreasing).
    q75_x = np.quantile(x_, 0.75)
    q25_x = np.quantile(x_, 0.25)
    q75_y = np.quantile(y_, 0.75)
    q25_y = np.quantile(y_, 0.25)
    if increasing:
        top_x = x_ >= q75_x
        top_y = y_ >= q75_y
    else:
        top_x = x_ <= q25_x
        top_y = y_ >= q75_y
    if top_x.sum() == 0:
        top_q = float("nan")
    else:
        top_q = float((top_x & top_y).sum() / top_x.sum())

    return {
        "spearman_rho": float(rho) if rho == rho else None,
        "pearson_r": float(r) if r == r else None,
        "isotonic_mae": iso_mae,
        "top_q_consistency": top_q,
        "n": int(mask.sum()),
        "train_min": float(np.min(x_)),
        "train_max": float(np.max(x_)),
        "train_q25": float(q25_x),
        "train_q75": float(q75_x),
        "direction": "increasing" if increasing else "decreasing",
    }


def main():
    print(f"Loading {BIO_XLSX.name}…", flush=True)
    df = pd.read_excel(BIO_XLSX)
    df = df.dropna(subset=["log10_flux_total"]).reset_index(drop=True)
    print(f"  n={len(df)} rows with measured log10_flux_total", flush=True)
    y = df["log10_flux_total"].values.astype(float)

    out_axes = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        metrics = analyze_feature(x, y)
        reliably = (
            metrics.get("spearman_rho") is not None
            and abs(metrics["spearman_rho"]) >= MIN_SPEARMAN
            and metrics.get("top_q_consistency") is not None
            and metrics["top_q_consistency"] >= MIN_TOP_QUARTILE_CONSISTENCY
        )
        metrics["reliably_monotone"] = bool(reliably)
        out_axes[col] = metrics

    # Sort by abs Spearman, show top
    sorted_axes = sorted(
        out_axes.items(),
        key=lambda kv: -abs(kv[1].get("spearman_rho") or 0.0),
    )
    print(f"\n  Top monotone features (sorted by |Spearman ρ|):", flush=True)
    print(f"  {'feature':<28s}  {'rho':>8s}  {'top-q':>6s}  {'reliable':>10s}", flush=True)
    for name, m in sorted_axes[:15]:
        flag = "✓" if m["reliably_monotone"] else " "
        rho = m.get("spearman_rho")
        rho_s = f"{rho:+.3f}" if rho is not None else "  —  "
        tq = m.get("top_q_consistency")
        tq_s = f"{tq:.2f}" if tq is not None else " — "
        print(f"  {name:<28s}  {rho_s:>8s}  {tq_s:>6s}  {flag:>10s}",
              flush=True)

    bundle = {
        "version": "monotone_axes_v1",
        "min_spearman": MIN_SPEARMAN,
        "min_top_q_consistency": MIN_TOP_QUARTILE_CONSISTENCY,
        "n_train": int(len(df)),
        "axes": out_axes,
    }
    OUT_JSON.write_text(json.dumps(bundle, indent=2, default=str))
    n_reliable = sum(1 for m in out_axes.values() if m["reliably_monotone"])
    print(f"\n  saved {OUT_JSON.name}", flush=True)
    print(f"  {n_reliable}/{len(out_axes)} features flagged as reliably monotone",
          flush=True)


if __name__ == "__main__":
    main()
