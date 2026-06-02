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

from .lipid_library import load_lipid, Lipid
from .bilayer import build_bilayer_coords, write_gro, read_gro

ROOT = Path(__file__).resolve().parent.parent
TITR_FF = ROOT / "martini" / "titratable" / "force_fields"
SCRIPTS = ROOT / "martini" / "titratable" / "scripts"
WATER_GRO = ROOT / "martini" / "ff" / "water.gro"
FULL_FF = TITR_FF / "martini_titratable_full.itp"   # merged FF (build_titratable_ff.py)


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
    # Analysis selections for compute_apparent_pka.py. NOTE: gmx grompp/mdrun write
    # confout.gro with the TOPOLOGY bead names, so the titratable water is "WN" (the WNA
    # molecule's bead), NOT the pre-grompp "W" in start.gro -> use -ref "name WN".
    return {"status": "ok", "start_gro": str(start), "top": str(top),
            "n_popc": n_popc, "n_water": n_water, "ion": ion_res, "box_nm": list(box2),
            "analysis_sel": "name P2", "analysis_ref": "name WN"}


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


# ======================================================================================
# GENERALIZED IAJD PATH (FW-2): embed ANY IAJD in POPC + make its head titratable.
# ======================================================================================
# The MC3 path above swaps one MC3 bead-for-bead into a single POPC site. IAJDs are
# 24-33-bead dendrimers, so we instead EMBED one IAJD dilutely in a POPC host bilayer
# (reusing the validated Module-B host machinery, build_mixed_bilayer) with its ionizable
# head placed at the interface, then convert that head to the titratable P2/DN/DP motif and
# the water to WNA+H+ — exactly the conversion the MC3 path uses. The merged FF
# (martini_titratable_full.itp) supplies the full IAJD bead-interaction matrix so grompp
# sees no undefined/zero interactions.

# Titratable head motif (Grunewald 2020, as used by MC3_titratable.itp):
#   head bead -> P2 (intrinsic-pKa amine bead, charge -1) + DN (DB1, 0) + DP (DB2, +1).
_HEAD_INTRINSIC = "N2_10.2"   # tertiary-amine intrinsic pKa 10.2 (DMA/piperazine/piperidine
#                               heads); the MEMBRANE-SHIFTED apparent pKa is what we measure,
#                               exactly as the validated MC3 run (generic 10.2 intrinsic).


def _remap(i: int, head_bead: int) -> int:
    """0-based original bead index -> 1-based index after inserting DN,DP right after head."""
    return (i + 1) if i <= head_bead else (i + 3)


def write_titratable_iajd_itp(lip: "Lipid", head_bead: int, path: Path, mol_name: str,
                              intrinsic: str = _HEAD_INTRINSIC) -> Dict:
    """Emit a titratable .itp for IAJD `lip`: the head bead becomes the P2/DN/DP motif
    in-place (matching convert_gro_to_titratable.py's ordering), all bonds/angles renumbered,
    + the DP-P2 proton bond, P2-DN dummy constraint, and dummy exclusions. The IAJD's own
    bonds/angles (incl. those to the head) are preserved (now referencing P2)."""
    p2, dn, dp = head_bead + 1, head_bead + 2, head_bead + 3   # 1-based indices

    atoms: List[Tuple[int, str, str, float]] = []
    for i, (name, btype, charge) in enumerate(lip.beads):
        if i == head_bead:
            atoms.append((p2, intrinsic, "P2", -1.00))
            atoms.append((dn, "DB1", "DN", 0.00))
            atoms.append((dp, "DB2", "DP", 1.00))
        else:
            atoms.append((_remap(i, head_bead), btype, name, float(charge)))
    atoms.sort(key=lambda a: a[0])

    L = [f"; Titratable {mol_name} — IAJD head -> {intrinsic} titratable motif (auto-generated",
         f";  by build_membrane_pka.write_titratable_iajd_itp). Use martini_titratable_full.itp.",
         "[ moleculetype ]", "; name  nrexcl", f"  {mol_name}  1", "", "[ atoms ]",
         "; nr  type  resnr residue atom  cgnr  charge"]
    for (idx, btype, name, charge) in atoms:
        L.append(f"{idx:5d} {btype:10s} 1 {mol_name:8s} {name:5s} {idx:5d} {charge:7.3f}")

    L += ["", "[ bonds ]", "; head titratable motif: DP-P2 proton bond",
          f"{dp:4d} {p2:4d} 1 0.000 4000"]
    for (i, j, r0, k) in lip.bonds:
        L.append(f"{_remap(i, head_bead):4d} {_remap(j, head_bead):4d} 1 {r0:.4f} {k:.1f}")

    L += ["", "[ constraints ]", "; head dummy DN held off P2",
          f"{p2:4d} {dn:4d} 1 0.200"]

    if lip.angles:
        L += ["", "[ angles ]", ";  i j k func theta0 k"]
        for (i, j, k, th, kk) in lip.angles:
            L.append(f"{_remap(i, head_bead):4d} {_remap(j, head_bead):4d} "
                     f"{_remap(k, head_bead):4d} 2 {th:.2f} {kk:.1f}")

    L += ["", "[ exclusions ]", "; head dummies mutually excluded from P2 and each other",
          f"{p2:4d} {dn:4d} {dp:4d}", f"{dn:4d} {dp:4d}"]
    path.write_text("\n".join(L) + "\n")
    return {"mol_name": mol_name, "n_atoms": len(atoms), "p2_idx": p2, "dn_idx": dn,
            "dp_idx": dp, "intrinsic": intrinsic}


def _iajd_system_top(mol_name: str, itp_path: Path, n_popc: int, n_water: int) -> str:
    """Titratable system.top: merged FF (atomtypes+nonbond) + moleculetype libs + IAJD.

    ALL includes are absolute: compute_apparent_pka.run_one_pH runs grompp from a
    pH_X/min/ subdir, so a relative IAJD-.itp include would be unresolvable (fatal)."""
    return f"""#define pH<value>
#include "{FULL_FF}"
#include "{TITR_FF}/lipids.itp"
#include "{TITR_FF}/molecules.itp"
#include "{TITR_FF}/ion.itp"
#include "{Path(itp_path).resolve()}"

[ system ]
{mol_name} in POPC bilayer (titratable, constant-pH)

[ molecules ]
{mol_name}  1
POPC  {n_popc}
WNA  {n_water}
H+  {n_water}
"""


def build_iajd(iajd_num: int, workdir: Path, gmx: List[str], *,
               n_per_leaflet: int = 32, intrinsic: str = _HEAD_INTRINSIC,
               protonated: bool = False) -> Dict:
    """Build a titratable constant-pH system for IAJD `iajd_num`: 1 IAJD embedded head-up in
    a POPC host bilayer, head -> titratable motif, water -> WNA+H+. Returns start_gro / top /
    analysis selections for compute_apparent_pka.titrate()."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from physics_design.iajd_cg import iajd_to_lipid
    from physics_design.build_mixed_bilayer import build_mixed_bilayer

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not FULL_FF.exists():
        from physics_design.build_titratable_ff import build as _build_ff
        _build_ff(verbose=False)

    lip = iajd_to_lipid(iajd_num, protonated=protonated)
    head_bead = int(lip.head_bead)
    mol_name = f"IAJD{iajd_num}T"
    # rename the head bead to "P2" so convert_gro_to_titratable.py (-sel 'name P2') finds it
    lip.beads[head_bead] = ("P2", lip.beads[head_bead][1], lip.beads[head_bead][2])

    # 1. embed ONE IAJD head-up in a POPC host bilayer (reuse validated Module-B machinery)
    mb = build_mixed_bilayer(lip, head_bead, workdir, gmx, n_per_leaflet=n_per_leaflet,
                             n_iajd_upper=1)
    if mb.get("status") != "ok":
        return {"status": f"embed_failed:{mb.get('status')}", "log": mb.get("log", "")}
    embedded = Path(mb["gro"])     # POPC + IAJD(head named P2) + W

    # 2. titratable conversion: head P2 -> base(+DN,DP), water W -> WNA(+H+)
    py = _sys.executable
    temp = workdir / "temp.gro"
    start = workdir / "start.gro"
    r1 = subprocess.run([py, str(SCRIPTS / "convert_gro_to_titratable.py"), "-f",
                         str(embedded), "-o", str(temp), "-sel", "name P2", "-bead", "base"],
                        cwd=str(workdir), capture_output=True, text=True)
    r2 = subprocess.run([py, str(SCRIPTS / "convert_gro_to_titratable.py"), "-f", str(temp),
                         "-o", str(start), "-sel", "name W", "-bead", "water"],
                        cwd=str(workdir), capture_output=True, text=True)
    if not start.exists():
        return {"status": "convert_failed",
                "log": (r1.stdout + r1.stderr + r2.stdout + r2.stderr)[-2000:]}

    # 3. titratable IAJD .itp (head motif placed at head_bead) + system.top
    itp_path = workdir / f"{mol_name}.itp"
    itp_info = write_titratable_iajd_itp(lip, head_bead, itp_path, mol_name,
                                         intrinsic=intrinsic)
    n_popc, n_water = mb["n_popc"], mb["n_water"]
    top = workdir / "system.top"
    top.write_text(_iajd_system_top(mol_name, itp_path, n_popc, n_water))

    return {"status": "ok", "start_gro": str(start), "top": str(top), "ion": mol_name,
            "n_popc": n_popc, "n_water": n_water, "head_bead": head_bead,
            "head_type_orig": lip.beads[head_bead][1], "n_beads_iajd": lip.n_beads,
            "intrinsic": intrinsic, "box_nm": mb.get("box_nm"),
            "analysis_sel": "name P2", "analysis_ref": "name WN", **itp_info}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--ionizable", default="MC3", help="MC3 (legacy) — ignored if --iajd given")
    p.add_argument("--iajd", type=int, default=None, help="IAJD_num: build titratable IAJD system")
    p.add_argument("--workdir", default=None)
    p.add_argument("--n-per-leaflet", type=int, default=32)
    args = p.parse_args()
    gmx = [os.environ.get("GMX", "/opt/homebrew/bin/gmx")]
    if args.iajd is not None:
        wd = Path(args.workdir or f"/tmp/membrane_iajd{args.iajd}")
        info = build_iajd(args.iajd, wd, gmx, n_per_leaflet=args.n_per_leaflet)
    else:
        info = build(args.ionizable, Path(args.workdir or "/tmp/membrane_mc3"), gmx,
                     n_per_leaflet=args.n_per_leaflet)
    print(json.dumps(info, indent=2, default=str))
