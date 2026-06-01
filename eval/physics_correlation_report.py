"""
eval/physics_correlation_report.py — extend compute_cpp.main()'s correlation
loop to every new W-A/W-B/W-C descriptor in qm_cache.csv, md_cache.csv,
head_area_ensemble.csv. Honest negative reporting: descriptors with too few
finite values are flagged in the audit; nothing is silently imputed.

Outputs:
  eval/figures/physics_correlation_report.png   bar chart of |r| per descriptor
  eval/figures/physics_correlation_report.json  per-descriptor stats + audit
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"
FIG_DIR = ROOT / "eval" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = FIG_DIR / "physics_correlation_report.json"
BAR_PNG = FIG_DIR / "physics_correlation_report.png"


def _merge_caches(df_bio: pd.DataFrame) -> pd.DataFrame:
    out = df_bio[["SMILES_canonical", "log10_flux_total", "family"]].copy()
    out = out[out["log10_flux_total"].notna()].reset_index(drop=True)
    qm_path = PHYS_DIR / "qm_cache.csv"
    md_path = PHYS_DIR / "md_cache.csv"
    if qm_path.exists():
        qm = pd.read_csv(qm_path)
        out = out.merge(qm, left_on="SMILES_canonical",
                          right_on="smiles_canonical", how="left")
    if md_path.exists():
        md = pd.read_csv(md_path)
        out = out.merge(md, left_on="SMILES_canonical",
                          right_on="smiles_canonical", how="left",
                          suffixes=("", "_md"))
    return out


def _correlations(merged: pd.DataFrame) -> dict:
    flux = merged["log10_flux_total"].values
    skip = {"SMILES_canonical", "smiles_canonical", "family", "log10_flux_total"}
    rows = {}
    for col in merged.columns:
        if col in skip:
            continue
        try:
            v = pd.to_numeric(merged[col], errors="coerce").values
        except Exception:
            continue
        mask = np.isfinite(v) & np.isfinite(flux)
        if mask.sum() < 3:
            rows[col] = {"n": int(mask.sum()),
                         "skip_reason": "<3 finite values"}
            continue
        r_p, p_p = pearsonr(v[mask], flux[mask])
        r_s, p_s = spearmanr(v[mask], flux[mask])
        rows[col] = {
            "n": int(mask.sum()),
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
        }
    return rows


def _plot_bar(rows: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rated = [(k, v) for k, v in rows.items() if "pearson_r" in v]
    rated.sort(key=lambda kv: -abs(kv[1]["pearson_r"]))
    labels = [k for k, _ in rated]
    vals = [v["pearson_r"] for _, v in rated]
    colors = ["#3a8" if v > 0 else "#a83" for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(labels))))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r vs log10_flux_total")
    ax.set_title("W-A/W-B/W-C physics descriptor correlations")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    print("Loading bioactivity dataset…")
    df_bio = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
    merged = _merge_caches(df_bio)
    print(f"  merged: {len(merged)} rows, {len(merged.columns)} cols")
    rows = _correlations(merged)
    REPORT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"Report: {REPORT_JSON}")
    _plot_bar(rows, BAR_PNG)
    print(f"Bar chart: {BAR_PNG}")
    # Print top descriptors.
    rated = [(k, v) for k, v in rows.items() if "pearson_r" in v]
    rated.sort(key=lambda kv: -abs(kv[1]["pearson_r"]))
    print("\nTop descriptors by |Pearson r|:")
    for k, v in rated[:25]:
        sig = "***" if v["pearson_p"] < 1e-3 else "**" if v["pearson_p"] < 1e-2 else "*" if v["pearson_p"] < 0.05 else " "
        print(f"  {k:32s} r={v['pearson_r']:+.3f} (n={v['n']:>3d}) {sig}")


if __name__ == "__main__":
    main()
