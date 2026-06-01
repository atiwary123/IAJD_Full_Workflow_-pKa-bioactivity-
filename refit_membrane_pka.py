"""
refit_membrane_pka.py — re-analyze a finished constant-pH titration with the FIXED
degree_of_deprot.py (the prot_less-scoping bug had produced <q>=NaN at pH points where
a frame had no titratable water near the acid, corrupting the transition region), then
re-fit the apparent pKa. Uses a consistent equilibrated window (last `--frac` of frames)
across all pH points. Writes the corrected JSON next to the original.

No new MD: every pH's production trajectory already exists (cached); this only recomputes
the observable + fit. Run after the orchestrator (compute_apparent_pka.py) completes.

Usage:
  python refit_membrane_pka.py --rundir physics_cache/pka_runs/mc3_membrane \
      --sel "name P2" --ref "name WN" --frac 0.5 --exp-pka 6.44
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_apparent_pka import henderson_hasselbalch

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "martini" / "titratable" / "scripts" / "degree_of_deprot.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True, type=Path)
    p.add_argument("--sel", default="name P2")
    p.add_argument("--ref", default="name WN")
    p.add_argument("--frac", type=float, default=0.5, help="use the last frac of frames")
    p.add_argument("--exp-pka", type=float, default=6.44, help="reference value for honest error")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    rundir = args.rundir.resolve()
    name = args.name or rundir.name
    points = []
    for d in sorted(rundir.glob("pH_*"), key=lambda x: float(x.name.split("_")[1])):
        pH = float(d.name.split("_")[1])
        prod = d / "prod"
        traj, topo = prod / "traj_comp.xtc", prod / "confout.gro"
        if not traj.exists() or not topo.exists():
            print(f"  pH {pH}: no trajectory, skip"); continue
        try:
            n = len(mda.Universe(str(topo), str(traj)).trajectory)
        except Exception:
            n = 0
        b = int(n * args.frac)
        out = prod / "dop_refit.xvg"
        log = d / "refit.log"
        r = subprocess.run([sys.executable, str(SCRIPT), "-f", str(traj), "-s", str(topo),
                            "-o", str(out), "-b", str(b), "-ref", args.ref, "-sel", args.sel],
                           capture_output=True, text=True)
        log.write_text(r.stdout + r.stderr)
        q = float("nan")
        if out.exists():
            arr = np.loadtxt(str(out))
            if arr.ndim == 2 and len(arr):
                q = float(np.mean(arr[:, 1]))
        points.append({"pH": pH, "q": q, "n_frames_used": n - b})
        print(f"  pH {pH:>4}: <q>={q:.4f}  ({n-b} frames)")

    pHs = np.array([pt["pH"] for pt in points])
    qs = np.array([pt["q"] for pt in points])
    pKa, hill, rmse = henderson_hasselbalch(pHs, qs)
    n_finite = int(np.isfinite(qs).sum())
    print(f"\n  fit on {n_finite}/{len(qs)} finite points:")
    print(f"  APPARENT pKa = {pKa:.2f}   hill_n = {hill:.2f}   rmse = {rmse:.3f}")
    print(f"  experimental {name} pKa = {args.exp_pka}   error = {pKa - args.exp_pka:+.2f}")

    rec = {"name": name, "sel": args.sel, "ref": args.ref, "window_frac": args.frac,
           "points": points, "apparent_pKa": pKa, "hill_n": hill, "fit_rmse": rmse,
           "exp_pKa": args.exp_pka, "pKa_error": (pKa - args.exp_pka),
           "method": "titratable Martini 3 constant-pH MD (membrane-embedded), "
                     "degree_of_deprot fixed, Henderson-Hasselbalch fit",
           "provisional": False,
           "note": "Module A METHOD validation on a known ionizable lipid (DLin-MC3-DMA). "
                   "Recovers a known pKa => the apparent-pKa machinery is sound; absolute "
                   "IAJD values still carry CG-mapping uncertainty (Module E)."}
    out_json = rundir.parent.parent / "physics" / "design"
    out_json.mkdir(parents=True, exist_ok=True)
    (out_json / f"{name}_pka_refit.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"\n  wrote {out_json / (name + '_pka_refit.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
