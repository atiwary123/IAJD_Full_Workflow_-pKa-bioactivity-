"""
analyze_curvature.py — inspect/plot a saved lateral pressure profile and (re)compute
the spontaneous curvature with the current kappa, decoupled from the MD run.

Reads the *_profile.npz written by compute_curvature.py (z, dP, dP_nb/bond/angle)
and the matching *_curvature.json (structural data), recomputes c0 over a monolayer
cutoff, prints the moment-vs-zmax convergence curve, and renders the canonical
membrane lateral-pressure-profile figure (and an overlay if several lipids given).

Usage:
  python analyze_curvature.py DOPC DOPE POPC            # overlay + per-lipid c0
  python analyze_curvature.py --temp 300 DOPC --png out.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from physics_design.pressure_profile import ProfileResult, BAR_NM_TO_MNM
from physics_design.curvature import spontaneous_curvature

DESIGN = Path(__file__).resolve().parent / "IAJD_master" / "bundles_caches" / "physics" / "design"


def load_profile(lipid: str, temp: int = 300) -> tuple:
    npz = DESIGN / f"{lipid}_T{temp}_profile.npz"
    js = DESIGN / f"{lipid}_T{temp}_curvature.json"
    d = np.load(npz)
    meta = json.loads(js.read_text()) if js.exists() else {}
    z, dP = d["z"], d["dP"]
    dz = float(z[1] - z[0])
    gamma = -BAR_NM_TO_MNM * float(np.sum(dP) * dz)
    prof = ProfileResult(z=z, dP=dP, dP_nb=d["dP_nb"], dP_bond=d["dP_bond"],
                         dP_angle=d["dP_angle"], n_frames=meta.get("n_frames", 0),
                         surface_tension_mNm=gamma, area_nm2=meta.get("apl_nm2", 0) * 0,
                         gamma_global_mNm=meta.get("gamma_global_mNm", float("nan")))
    return prof, meta


def ascii_profile(z, dP, zrange=3.5, scale=40.0):
    lines = []
    for i in range(len(z)):
        if abs(z[i]) <= zrange:
            n = int(abs(dP[i]) / scale)
            bar = ("+" if dP[i] >= 0 else "-") + ("#" * min(n, 60))
            lines.append(f"{z[i]:+5.2f} {dP[i]:+7.0f} {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lipids", nargs="+")
    ap.add_argument("--temp", type=int, default=300)
    ap.add_argument("--png", default=str(DESIGN / "curvature_profiles.png"))
    args = ap.parse_args()

    profs = {}
    for lip in args.lipids:
        try:
            prof, meta = load_profile(lip, args.temp)
        except FileNotFoundError:
            print(f"{lip}: no profile found"); continue
        thick = meta.get("thickness_PP_nm", 3.7)
        zmax = thick / 2.0 + 1.5
        cr = spontaneous_curvature(prof, lip, zmax=zmax)
        profs[lip] = (prof, meta, cr)
        print(f"\n===== {lip} (T={args.temp}) =====")
        print(f"  APL={meta.get('apl_nm2'):.3f} nm^2  thick={thick:.2f} nm  "
              f"gamma={prof.surface_tension_mNm:+.1f} mN/m  n_frames={meta.get('n_frames')}")
        print(f"  water_baseline={cr.water_baseline_bar:.1f} bar  zmax={zmax:.2f} nm")
        print(f"  tau_moment={cr.tau_moment_bar_nm2:+.1f} bar*nm^2  "
              f"c0={cr.c0_nm_inv:+.3f} nm^-1  R0={cr.R0_nm:+.1f} nm")
        fc = meta.get("force_check", {})
        if fc.get("status") == "ok":
            print(f"  force-check vs GROMACS: max |err|={fc.get('max_abs_pct_err'):.2e} %")
        print("  moment-vs-zmax (bar*nm^2):", cr.moment_curve)
        print("  lateral pressure profile P_L(z)-P_N(z) [bar]:")
        print(ascii_profile(prof.z, prof.dP))

    # matplotlib overlay figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lip, (prof, meta, cr) in profs.items():
            ax.plot(prof.z, prof.dP, label=f"{lip} (c0={cr.c0_nm_inv:+.2f} nm$^{{-1}}$)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-4, 4)
        ax.set_xlabel("z from bilayer midplane (nm)")
        ax.set_ylabel(r"$P_L(z)-P_N(z)$  (bar)")
        ax.set_title("Martini-3 lateral pressure profile (Irving-Kirkwood)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
        print(f"\nsaved figure -> {args.png}")
    except Exception as exc:
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
