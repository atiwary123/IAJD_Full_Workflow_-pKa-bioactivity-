"""
analyze_family_sar.py — honest, per-family structure-activity (SAR) priors.

Motivation
----------
The proposer (propose_iajds.beam_search) enumerates EVERY single-step mutation
of a seed (head swap, linker resize, tail swap/extend/shorten, linkage swap,
synthetic-head variants…) and scores each with the ML regressor + physics. The
enumeration is *exhaustive and uninformed*: it spends equal budget on axes that
genuinely move bioactivity and on axes that, for that family, are pure noise.
Because a decent seed sits above its family mean, perturbing a no-signal axis
regresses toward the mean → mostly small NEGATIVE Δ vs seed. That is the
"why is the proposer sending back negative deltas" symptom.

This module measures, **per family, from the training set only**, which
structural axes actually track log10_flux_total, and in which direction. The
proposer then uses these as a prior to (a) prune/deprioritise mutations along
no-signal or proven-adverse directions and (b) add a transparent, data-grounded
term to the ranking score. Nothing here is a proxy or a guess — every number is
a real statistic from measured rows, every axis is significance-gated, and the
features are computed by the SAME fast 2D function (family_sar.sar_features)
that the proposer uses to score candidates, so train-stats and candidate values
live in one consistent space.

What we measure per family (rows with measured log10_flux_total only)
--------------------------------------------------------------------
  Continuous descriptors (family_sar.SAR_FEATURE_NAMES) + linker_length:
    - Spearman ρ (monotone signal), its p-value, and n
    - train_min / max / mean / std  (for standardising a candidate's value)
    - "actionable": |ρ| ≥ MIN_ABS_RHO and p ≤ MAX_P and n ≥ MIN_N
    - "selected_for_prior": actionable AND not collinear (|Pearson r| < 0.7)
      with an already-selected, stronger axis — so the prior doesn't count the
      greasiness/size cluster (MolLogP ≈ LogP_per_HA ≈ …) several times.

  Categorical levers (the discrete mutation operators):
    - head_group : mean flux + n per head (shrunk toward the family mean for
      small n), so a head swap's prior = the observed mean-flux gap.
    - linkage    : mean flux + n per linkage.

Output
------
  family_sar_priors.json  — consumed by family_sar.sar_prior_for_candidate()

CLI:
  python analyze_family_sar.py            # writes family_sar_priors.json
  python analyze_family_sar.py --report   # also prints the full per-family table
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from family_sar import sar_features, SAR_FEATURE_NAMES   # shared feature space

BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
OUT_JSON = ROOT / "family_sar_priors.json"

# Families the grammar can actually assemble/mutate (iajd_grammar.FAMILY_ASSEMBLERS).
MUTABLE_FAMILIES = ["sSS-Nonsym", "GA-Tris", "PE-Tris", "PE-Gallic", "Dialkoxybenzyl"]

# linker_length is a structural integer (not a sar_features descriptor): read
# from the xlsx column for train-stats, and from Seed.linker_n at scoring time.
STRUCTURAL_AXES = ["linker_length"]

# Actionability gate. Effect-size-primary (|ρ|), with a significance floor that
# small families (n≈20) can still clear, and a hard minimum n.
MIN_ABS_RHO = 0.25
MAX_P = 0.10
MIN_N = 12
COLLINEAR_R = 0.70     # |Pearson r| above which two axes are "the same axis"

MIN_LEVEL_N = 2        # need ≥2 measured rows to report a head/linkage level mean
SHRINK_K = 3.0         # James-Stein-ish shrinkage of a level mean toward family mean


def _num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").values.astype(float)


def _axis_stats(x: np.ndarray, y: np.ndarray, source: str) -> dict | None:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 6:
        return None
    xv, yv = x[mask], y[mask]
    if np.std(xv) == 0:                 # constant within family → no signal
        return None
    rho, p = spearmanr(xv, yv)
    if not np.isfinite(rho):
        return None
    actionable = (abs(rho) >= MIN_ABS_RHO) and (p <= MAX_P) and (n >= MIN_N)
    return {
        "rho": float(rho), "p": float(p), "n": n,
        "direction": "increasing" if rho >= 0 else "decreasing",
        "train_min": float(np.min(xv)), "train_max": float(np.max(xv)),
        "train_mean": float(np.mean(xv)), "train_std": float(np.std(xv)),
        "actionable": bool(actionable),
        "selected_for_prior": False,
        "source": source,
    }


def _continuous_axes(feat_df: pd.DataFrame, y: np.ndarray) -> dict:
    axes = {}
    for f in list(SAR_FEATURE_NAMES) + STRUCTURAL_AXES:
        if f not in feat_df.columns:
            continue
        st = _axis_stats(_num(feat_df[f]), y, "structural" if f in STRUCTURAL_AXES else "descriptor")
        if st is not None:
            axes[f] = st
    return axes


def _select_decorrelated(feat_df: pd.DataFrame, axes: dict) -> None:
    """Greedily flag actionable axes that are mutually de-correlated. Picks in
    order of |ρ|, skipping any axis collinear (|Pearson r| ≥ COLLINEAR_R) with
    an already-selected one. Mutates `axes` in place."""
    actionable = sorted([f for f, a in axes.items() if a["actionable"]],
                        key=lambda f: -abs(axes[f]["rho"]))
    selected: list[str] = []
    for f in actionable:
        xf = _num(feat_df[f])
        redundant = False
        for g in selected:
            xg = _num(feat_df[g])
            m = np.isfinite(xf) & np.isfinite(xg)
            if m.sum() < 6 or np.std(xf[m]) == 0 or np.std(xg[m]) == 0:
                continue
            if abs(np.corrcoef(xf[m], xg[m])[0, 1]) >= COLLINEAR_R:
                redundant = True
                break
        if not redundant:
            axes[f]["selected_for_prior"] = True
            selected.append(f)


def _categorical_levels(sub: pd.DataFrame, col: str, fam_mean: float) -> dict:
    out = {"_family_mean": float(fam_mean)}
    if col not in sub.columns:
        return out
    g = sub.groupby(col)["log10_flux_total"].agg(["mean", "count"])
    for level, row in g.iterrows():
        if pd.isna(level):
            continue
        n = int(row["count"])
        if n < MIN_LEVEL_N:
            continue
        raw = float(row["mean"])
        w = n / (n + SHRINK_K)
        shrunk = w * raw + (1 - w) * fam_mean
        out[str(level)] = {
            "mean_raw": raw, "mean_shrunk": float(shrunk),
            "delta_vs_family": float(shrunk - fam_mean), "n": n,
        }
    return out


def analyze() -> dict:
    df = pd.read_excel(BIO_XLSX)
    df = df.dropna(subset=["log10_flux_total"]).reset_index(drop=True)
    smis = df["SMILES_canonical"].fillna(df["SMILES"]).astype(str)
    feats = [sar_features(s) or {} for s in smis]
    feat_df = pd.DataFrame(feats)
    feat_df["linker_length"] = _num(df["linker_length"])
    feat_df["family"] = df["family"].values
    feat_df["log10_flux_total"] = df["log10_flux_total"].values

    bundle = {
        "version": "family_sar_v2",
        "generated_from": BIO_XLSX.name,
        "feature_space": "family_sar.sar_features (fast 2D) + structural linker_length",
        "n_train_total": int(len(df)),
        "gate": {"min_abs_rho": MIN_ABS_RHO, "max_p": MAX_P, "min_n": MIN_N,
                 "collinear_r": COLLINEAR_R},
        "categorical": {"min_level_n": MIN_LEVEL_N, "shrink_k": SHRINK_K},
        "families": {},
    }
    for fam in MUTABLE_FAMILIES:
        m = feat_df["family"] == fam
        sub_feat = feat_df[m]
        sub_meta = df[m.values]
        if len(sub_feat) < 6:
            continue
        y = _num(sub_feat["log10_flux_total"])
        fam_mean = float(np.nanmean(y))
        axes = _continuous_axes(sub_feat, y)
        _select_decorrelated(sub_feat, axes)
        bundle["families"][fam] = {
            "n": int(len(sub_feat)),
            "flux_mean": fam_mean,
            "flux_std": float(np.nanstd(y)),
            "flux_min": float(np.nanmin(y)),
            "flux_max": float(np.nanmax(y)),
            "continuous": axes,
            "head_group": _categorical_levels(sub_meta, "head_group", fam_mean),
            "linkage": _categorical_levels(sub_meta, "linkage", fam_mean),
        }
    return bundle


def _report(bundle: dict) -> None:
    for fam, info in bundle["families"].items():
        print("=" * 78)
        print(f"{fam}   n={info['n']}   flux mean={info['flux_mean']:.2f} "
              f"sd={info['flux_std']:.2f}  range[{info['flux_min']:.2f}, {info['flux_max']:.2f}]")
        ranked = sorted(info["continuous"].items(), key=lambda kv: -abs(kv[1]["rho"]))
        print(f"  {'feature':<20s} {'rho':>7s} {'p':>7s} {'n':>4s}  flag")
        for f, a in ranked[:10]:
            flag = "◀ PRIOR" if a["selected_for_prior"] else ("(collinear)" if a["actionable"] else "")
            print(f"  {f:<20s} {a['rho']:+.3f} {a['p']:7.3f} {a['n']:4d}  {flag}")
        for cat in ("head_group", "linkage"):
            levels = {k: v for k, v in info[cat].items() if k != "_family_mean"}
            if not levels:
                continue
            parts = [f"{lvl}:{v['mean_shrunk']:.2f}(Δ{v['delta_vs_family']:+.2f},n{v['n']})"
                     for lvl, v in sorted(levels.items(), key=lambda kv: -kv[1]["mean_shrunk"])]
            print(f"  {cat:<20s} " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the full per-family table")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    print(f"Loading {BIO_XLSX.name}…", flush=True)
    bundle = analyze()
    Path(args.out).write_text(json.dumps(bundle, indent=2, default=str))
    n_fam = len(bundle["families"])
    n_sel = sum(sum(1 for a in f["continuous"].values() if a["selected_for_prior"])
                for f in bundle["families"].values())
    print(f"  analysed {n_fam} families; {n_sel} de-correlated prior axes total")
    print(f"  wrote {args.out}")
    if args.report:
        print()
        _report(bundle)


if __name__ == "__main__":
    main()
