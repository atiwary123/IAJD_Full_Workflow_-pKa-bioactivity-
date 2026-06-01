"""
analyze_panel.py — "is a physics->flux signal emerging?" the 24-HOUR DECISION TOOL.

Run it anytime while the host-method c0 panel is computing. It joins the converged c0 values
(and the Module-A pKa feature) with organ flux and reports, with HONEST small-n bootstrap CIs:
  - c0  vs flux   (Spearman + Pearson), pooled and per-family
  - pKa vs flux   (the existing feature, for comparison)
  - a GAM response curve c0->flux when n is large enough (per DESIGN_OPTIMIZATION_PROTOCOL)
  - a VERDICT: is an actionable physics->flux correlation emerging, or not?

Decision rule (deliberately conservative): a "signal" requires |Spearman|>=0.35 AND the 90%
bootstrap CI excluding 0 AND n>=8 converged points. Anything else is "weak/inconclusive" —
told honestly so you can decide within 24h whether to keep computing.

Usage:  python analyze_panel.py [--target log10_flux_spleen] [--family GA-Tris]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design"
_DS = ROOT / "IAJD_master" / "datasets"
BIOACT = (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx") if \
    (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx").exists() else \
    (_DS / "IAJD_Bioact_v13_clean.xlsx")


def load_c0() -> pd.DataFrame:
    rows = []
    for f in sorted(DESIGN.glob("IAJD*_host_T*_curvature.json")):
        try:
            r = json.loads(f.read_text())
            rows.append({"iajd": int(r["iajd"]), "c0": r.get("c0_nm_inv"),
                         "converged": bool(r.get("c0_physics_converged")),
                         "method": "host"})
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    # include pure-bilayer runs only if no host run exists for that IAJD (host is preferred)
    have = {d["iajd"] for d in rows}
    for f in sorted(DESIGN.glob("IAJD*_neutral_T*_curvature.json")):
        try:
            r = json.loads(f.read_text())
            n = int(r["iajd"])
            if n not in have:
                rows.append({"iajd": n, "c0": r.get("c0_nm_inv"),
                             "converged": bool(r.get("c0_physics_converged")),
                             "method": "pure(provisional)"})
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return pd.DataFrame(rows)


def boot_ci(x, y, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x)); rs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[b])) > 2 and len(np.unique(y[b])) > 2:
            rs.append(spearmanr(x[b], y[b]).correlation)
    if not rs:
        return (float("nan"), float("nan"))
    return tuple(np.nanpercentile(rs, [5, 95]))


def report(x, y, label):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        print(f"  {label}: n={len(x)} (too few to correlate)"); return None
    rho = spearmanr(x, y).correlation
    r = pearsonr(x, y)[0]
    lo, hi = boot_ci(x, y)
    sig = (abs(rho) >= 0.35) and (lo * hi > 0) and (len(x) >= 8)
    flag = "  <== SIGNAL" if sig else ("  (weak)" if abs(rho) >= 0.35 else "")
    print(f"  {label}: Spearman={rho:+.2f} [90%CI {lo:+.2f},{hi:+.2f}]  "
          f"Pearson={r:+.2f}  n={len(x)}{flag}")
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="log10_flux_spleen")
    p.add_argument("--family", default=None, help="restrict to one family")
    args = p.parse_args()

    c0 = load_c0()
    if c0.empty:
        print("No c0 results yet (panel hasn't produced any IAJD*_curvature.json)."); return 0
    ds = pd.read_excel(BIOACT)
    famcol = next((c for c in ("family", "Family", "architecture") if c in ds.columns), None)
    keep = ["IAJD_num", args.target] + ([famcol] if famcol else []) + \
           (["pKa"] if "pKa" in ds.columns else [])
    ds = ds[keep].rename(columns={"IAJD_num": "iajd", famcol: "family"})
    df = c0.merge(ds, on="iajd", how="left")
    if args.family:
        df = df[df["family"].astype(str).str.contains(args.family, case=False, na=False)]

    conv = df[df["converged"]]
    print(f"\n=== PHYSICS->FLUX SIGNAL CHECK ({args.target}) ===")
    print(f"c0 computed: {len(df)} | converged (trusted physics): {len(conv)} | "
          f"provisional/pure: {int((df['method']!='host').sum())}")
    print(f"\n[c0 vs {args.target}]  (converged only)")
    sig_c0 = report(conv["c0"], conv[args.target], "pooled")
    if "family" in conv and conv["family"].nunique() > 1:
        for fam, g in conv.groupby("family"):
            report(g["c0"], g[args.target], f"family {fam}")

    if "pKa" in ds.columns:
        # pKa is available for ALL rows NOW (no physics needed) — the immediate baseline.
        dsf = ds[df.columns.intersection(ds.columns).tolist()] if False else ds
        if args.family:
            dsf = ds[ds["family"].astype(str).str.contains(args.family, case=False, na=False)]
        else:
            dsf = ds
        print(f"\n[pKa vs {args.target}]  (ALL {len(dsf)} rows — the existing-feature baseline, "
              f"available before any physics)")
        report(dsf["pKa"], dsf[args.target], "pooled")
        if "family" in dsf and dsf["family"].nunique() > 1:
            for fam, g in dsf.groupby("family"):
                if g[["pKa", args.target]].dropna().shape[0] >= 4:
                    report(g["pKa"], g[args.target], f"family {fam}")

    # GAM response curve when enough converged points
    if len(conv) >= 8:
        try:
            from pygam import LinearGAM, s
            xx = conv["c0"].to_numpy(float); yy = conv[args.target].to_numpy(float)
            m = np.isfinite(xx) & np.isfinite(yy)
            if m.sum() >= 8:
                gam = LinearGAM(s(0)).gridsearch(xx[m].reshape(-1, 1), yy[m], progress=False)
                xs = np.linspace(xx[m].min(), xx[m].max(), 5)
                print(f"\n[GAM c0->{args.target} response] pseudo-R2={gam.statistics_['pseudo_r2']['explained_deviance']:.2f}")
                print("  c0:", [round(v, 2) for v in xs], "-> flux:",
                      [round(float(v), 2) for v in gam.predict(xs.reshape(-1, 1))])
        except Exception as exc:
            print(f"  (GAM skipped: {type(exc).__name__})")

    print("\n=== VERDICT ===")
    if sig_c0:
        print("  Actionable c0->flux signal emerging — worth continuing the panel.")
    elif len(conv) < 8:
        print(f"  INCONCLUSIVE — only {len(conv)} converged points. Need ~8+ to judge; keep computing.")
    else:
        print("  No clear c0->flux signal yet at this n. Check pKa/other axes + per-family;")
        print("  if still flat after the family completes, the physics axis may not be the lever here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
