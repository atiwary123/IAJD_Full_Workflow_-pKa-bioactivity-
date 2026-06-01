"""
physics_design/build_mixed_bilayer.py — POPC host bilayer with N IAJDs inserted in the
UPPER leaflet only (asymmetric host method for the Module B spontaneous curvature c0 of
NON-BILAYER-prone IAJDs).

WHY (FW-4): a pure IAJD bilayer collapses for membrane-active IAJDs (369: APL->0.265,
bilayer breaks). The field's standard fix for H_II-formers is to embed them DILUTELY in a
stable host bilayer and extract their spontaneous-curvature CONTRIBUTION. The POPC host
keeps the membrane flat and sets the area. Inserting IAJDs in the UPPER leaflet ONLY makes
the bilayer asymmetric, so the two per-leaflet first moments differ by the IAJD term:

    tau_upper = (1-x)*tau_POPC + x*tau_IAJD ,   tau_lower = tau_POPC   (pure POPC)
 => tau_IAJD = tau_lower + (tau_upper - tau_lower) / x       (x = upper-leaflet IAJD fraction)

So ONE run per IAJD yields its curvature contribution, with the lower leaflet as a built-in
pure-POPC baseline (no separate reference run needed). The c0 still carries the CG-mapping
caveat (Module E) and an assumed kappa, but the bilayer is now stable + tensionless, so the
measurement is well-posed (unlike the collapsing pure bilayer).

No proxies: real CG geometry + real GROMACS equilibration/production; the IAJD insert is the
build_cg topology, head-oriented to the interface and de-overlapped (bilayer._spread_xy...).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

from .lipid_library import load_lipid, write_single_itp, Lipid
from .bilayer import (build_bilayer_coords, lipid_template, write_gro, read_gro,
                      _link_ff)

ROOT = Path(__file__).resolve().parent.parent
_FF_DIR = ROOT / "martini" / "ff"
WATER_GRO = _FF_DIR / "water.gro"


def build_mixed_bilayer(iajd_lip: Lipid, head_idx: int, workdir: Path,
                        gmx_cmd: List[str], *, n_per_leaflet: int = 48,
                        n_iajd_upper: int = 6, apl_nm2: float = 0.64,
                        water_per_lipid: float = 40.0) -> Dict:
    """Build a POPC bilayer with `n_iajd_upper` IAJDs in the upper leaflet. Returns the
    final .gro/.top paths, molecule counts, and the upper-leaflet IAJD mole fraction."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    _link_ff(workdir)
    if not (0 < n_iajd_upper <= n_per_leaflet):
        return {"status": "bad_n_iajd"}

    popc = load_lipid("POPC")
    nb_popc = popc.n_beads
    pos, names, resn, box, zmid = build_bilayer_coords(popc, n_per_leaflet,
                                                       apl_nm2=apl_nm2)
    pos = np.asarray(pos)

    # IAJD single-molecule template (head at z=0, tails to -z), head moved to origin so we
    # can drop it onto each chosen upper-leaflet POPC head (bead 0 = NC3, top of leaflet).
    templ = lipid_template(iajd_lip, head_idx=head_idx)
    templ = templ - templ[head_idx]
    nb_iajd = iajd_lip.n_beads

    new_pos: List[np.ndarray] = []
    new_names: List[str] = []
    new_resn: List[str] = []
    # molecule order in build_bilayer_coords: upper leaflet 0..n_per_leaflet-1, then lower.
    for m in range(2 * n_per_leaflet):
        if m < n_iajd_upper:
            popc_head = pos[m * nb_popc + 0]          # NC3 at the upper interface
            coords = templ + popc_head                # IAJD head there, tails extend down
            for b in range(nb_iajd):
                new_pos.append(coords[b])
                new_names.append(iajd_lip.bead_names[b])
                new_resn.append(iajd_lip.name)
        else:
            base = m * nb_popc
            for b in range(nb_popc):
                new_pos.append(pos[base + b])
                new_names.append(names[base + b])
                new_resn.append("POPC")
    arr = np.array(new_pos)
    # keep lateral coords inside the box
    arr[:, 0] = np.mod(arr[:, 0], box[0])
    arr[:, 1] = np.mod(arr[:, 1], box[1])

    lipids_gro = workdir / "lipids.gro"
    write_gro(arr, new_names, new_resn, box, lipids_gro, title="POPC+IAJD host")
    write_single_itp(popc, workdir / "POPC.itp")
    write_single_itp(iajd_lip, workdir / f"{iajd_lip.name}.itp")

    # solvate with Martini W
    n_popc = 2 * n_per_leaflet - n_iajd_upper
    solv = workdir / "solvated.gro"
    tmp_top = workdir / "_solv.top"
    tmp_top.write_text("[ system ]\nx\n[ molecules ]\n")
    r = subprocess.run(gmx_cmd + ["solvate", "-cp", str(lipids_gro), "-cs", str(WATER_GRO),
                                  "-o", str(solv), "-p", str(tmp_top), "-radius", "0.21"],
                       cwd=str(workdir), capture_output=True, text=True, timeout=600)
    if not solv.exists():
        return {"status": "solvate_failed", "log": (r.stdout + r.stderr)[-2000:]}

    # strip waters inside the hydrophobic slab
    p2, n2, r2, box2 = read_gro(solv)
    zc = box2[2] / 2.0
    keep = [i for i in range(len(p2)) if not (n2[i] == "W" and abs(p2[i, 2] - zc) < 1.4)]
    p2 = p2[keep]; n2 = [n2[i] for i in keep]; r2 = [r2[i] for i in keep]
    n_water = sum(1 for x in r2 if x == "W")
    start = workdir / "start.gro"
    write_gro(p2, n2, r2, box2, start, title="POPC+IAJD host")

    top = workdir / "system.top"
    top.write_text(_mixed_top(iajd_lip.name, n_iajd_upper, n_popc, n_water))
    return {"status": "ok", "gro": str(start), "top": str(top),
            "n_iajd": n_iajd_upper, "n_popc": n_popc, "n_water": n_water,
            "nb_iajd": nb_iajd, "nb_popc": nb_popc,
            "x_upper": n_iajd_upper / float(n_per_leaflet), "box_nm": list(box2),
            "n_per_leaflet": n_per_leaflet}


def _mixed_top(iajd_name: str, n_iajd: int, n_popc: int, n_water: int) -> str:
    # molecule order MUST match the .gro: IAJDs first, then POPC, then W.
    return f"""#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_ions_v1.itp"
#include "POPC.itp"
#include "{iajd_name}.itp"

[ system ]
POPC host bilayer with {n_iajd} {iajd_name} (upper leaflet)

[ molecules ]
{iajd_name}  {n_iajd}
POPC  {n_popc}
W  {n_water}
"""


if __name__ == "__main__":
    import argparse, json, sys
    sys.path.insert(0, str(ROOT))
    from physics_design.iajd_cg import iajd_to_lipid
    p = argparse.ArgumentParser()
    p.add_argument("--iajd", type=int, default=369)
    p.add_argument("--workdir", default="/tmp/mixed_bilayer_test")
    p.add_argument("--n-per-leaflet", type=int, default=48)
    p.add_argument("--n-iajd", type=int, default=6)
    args = p.parse_args()
    lip = iajd_to_lipid(args.iajd, protonated=False)
    gmx = [os.environ.get("GMX", "/opt/homebrew/bin/gmx")]
    info = build_mixed_bilayer(lip, int(lip.head_bead), Path(args.workdir), gmx,
                               n_per_leaflet=args.n_per_leaflet, n_iajd_upper=args.n_iajd)
    print(json.dumps(info, indent=2, default=str))
