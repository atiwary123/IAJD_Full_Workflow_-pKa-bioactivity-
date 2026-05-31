"""
physics_design/build_membrane_pka.py — build a titratable-Martini membrane system for a
constant-pH apparent-pKa titration: a POPC bilayer with ONE ionizable molecule (e.g.
DLin-MC3-DMA = MC3) embedded, titratable water, ready for compute_apparent_pka.py.

Reuses the Module B flat-bilayer machinery (the titratable POPC shares standard POPC's
bead geometry; only the topology/FF differ), then:
  1. tile a POPC bilayer and SWAP one POPC for the ionizable molecule (bead-for-bead),
  2. gmx solvate with regular W,
  3. convert_gro_to_titratable.py: ionizable head bead -> 'base' (+dummy proton), W -> WNA,
  4. write a titratable system.top (titratable martini.itp + lipids.itp + the ionizable
     molecule's titratable .itp + molecules.itp).

No-proxy: real CG geometry; the titration physics is the subsequent constant-pH MD.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lipid_library import load_lipid
from .bilayer import build_bilayer_coords, write_gro, read_gro

ROOT = Path(__file__).resolve().parent.parent
TITR_FF = ROOT / "martini" / "titratable" / "force_fields"
SCRIPTS = ROOT / "martini" / "titratable" / "scripts"
WATER_GRO = ROOT / "martini" / "ff" / "water.gro"


# MC3 (DLin-MC3-DMA) bead layout, mapped onto a POPC site (head->head, tails->tails).
# Standard (pre-convert) MC3 has 12 beads; head bead renamed P2 so the converter finds
# it. Order matches MC3_titratable.itp's post-convert order MINUS the dummies (P2 then
# CN GLA CX tails); the converter inserts D,DP right after P2.
MC3_BEADS = ["P2", "CN", "GLA", "CX", "C1A", "D2A", "D3A", "C4A",
             "C1B", "D2B", "D3B", "C4B"]
# POPC bead order (from build_bilayer_coords / standard POPC):
POPC_BEADS = ["NC3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A",
              "C1B", "C2B", "C3B", "C4B"]


def build(ionizable: str, workdir: Path, gmx: List[str], *,
          n_per_leaflet: int = 32, water_per_lipid: float = 40.0) -> Dict:
    """Build a POPC bilayer with one `ionizable` molecule embedded, titratable.
    Currently supports ionizable='MC3' (DLin-MC3-DMA). Returns paths + counts."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    popc = load_lipid("POPC")
    pos, names, resn, box, zmid = build_bilayer_coords(popc, n_per_leaflet,
                                                       apl_nm2=0.64)
    n_lip = 2 * n_per_leaflet
    nb = popc.n_beads  # 12

    # Swap the FIRST upper-leaflet lipid (beads 0..11) for the ionizable molecule.
    if ionizable.upper() == "MC3":
        ion_names = MC3_BEADS
        ion_res = "MC3T"
    else:
        raise ValueError(f"unsupported ionizable '{ionizable}'")
    for b in range(nb):
        names[b] = ion_names[b]
        resn[b] = ion_res

    # Write lipids+ion .gro (ion is residue 1, POPC are residues 2..n_lip)
    lipids_gro = workdir / "lipids.gro"
    write_gro(np.array(pos), names, resn, box, lipids_gro, title=f"POPC+{ion_res}")

    # solvate with regular W
    for f in ("martini_v3.0.0.itp",):  # not needed for solvate; solvate uses geometry
        pass
    solv = workdir / "solvated.gro"
    n_target = int(water_per_lipid * n_lip)
    # minimal top for solvate (it only needs the molecule counts updated for W)
    tmp_top = workdir / "_solv.top"
    tmp_top.write_text("[ system ]\nx\n[ molecules ]\n")  # solvate appends W
    r = subprocess.run(gmx + ["solvate", "-cp", str(lipids_gro), "-cs", str(WATER_GRO),
                              "-o", str(solv), "-p", str(tmp_top), "-radius", "0.21"],
                       cwd=str(workdir), capture_output=True, text=True, timeout=600)
    if not solv.exists():
        return {"status": "solvate_failed", "log": (r.stdout + r.stderr)[-2000:]}

    # strip waters inside the hydrophobic slab (|z-zc|<1.4 nm)
    p2, n2, r2, box2 = read_gro(solv)
    zc = box2[2] / 2.0
    keep = [i for i in range(len(p2)) if not (n2[i] == "W" and abs(p2[i, 2] - zc) < 1.4)]
    p2 = p2[keep]; n2 = [n2[i] for i in keep]; r2 = [r2[i] for i in keep]
    n_water = sum(1 for x in r2 if x == "W")
    stripped = workdir / "stripped.gro"
    write_gro(p2, n2, r2, box2, stripped, title=f"POPC+{ion_res}+W")

    # convert: ionizable head P2 -> base (+dummy proton), water W -> WNA
    conv = [os.environ.get("PYBIN", "python3")]
    temp = workdir / "temp.gro"
    start = workdir / "start.gro"
    import sys
    py = sys.executable
    r1 = subprocess.run([py, str(SCRIPTS / "convert_gro_to_titratable.py"),
                         "-f", str(stripped), "-o", str(temp), "-sel", "name P2",
                         "-bead", "base"], cwd=str(workdir), capture_output=True, text=True)
    r2c = subprocess.run([py, str(SCRIPTS / "convert_gro_to_titratable.py"),
                          "-f", str(temp), "-o", str(start), "-sel", "name W",
                          "-bead", "water"], cwd=str(workdir), capture_output=True, text=True)
    if not start.exists():
        return {"status": "convert_failed",
                "log": (r1.stdout + r1.stderr + r2c.stdout + r2c.stderr)[-2000:]}

    # system.top (titratable). One ion molecule (MC3T) + (n_lip-1) POPC + WNA + H+.
    n_popc = n_lip - 1
    top = workdir / "system.top"
    top.write_text(_system_top(ion_res, n_popc, n_water))
    return {"status": "ok", "start_gro": str(start), "top": str(top),
            "n_popc": n_popc, "n_water": n_water, "ion": ion_res, "box_nm": list(box2)}


def _system_top(ion_res: str, n_popc: int, n_water: int) -> str:
    ion_itp = ROOT / "martini" / "ionizable" / "MC3_titratable.itp"
    return f"""#define pH<value>
#include "{TITR_FF}/martini.itp"
#include "{TITR_FF}/lipids.itp"
#include "{TITR_FF}/molecules.itp"
#include "{TITR_FF}/ion.itp"
#include "{ion_itp}"

[ system ]
{ion_res} in POPC bilayer (titratable)

[ molecules ]
{ion_res}  1
POPC  {n_popc}
WNA  {n_water}
H+  {n_water}
"""


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--ionizable", default="MC3")
    p.add_argument("--workdir", default="/tmp/membrane_mc3")
    p.add_argument("--n-per-leaflet", type=int, default=32)
    args = p.parse_args()
    gmx = [os.environ.get("GMX", "/opt/homebrew/bin/gmx")]
    info = build(args.ionizable, Path(args.workdir), gmx, n_per_leaflet=args.n_per_leaflet)
    print(json.dumps(info, indent=2, default=str))
