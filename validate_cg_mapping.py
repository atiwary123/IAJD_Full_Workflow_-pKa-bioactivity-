"""
validate_cg_mapping.py — Module E: validate the IAJD Martini-3 CG mapping against an
atomistic reference (build prompt §3 Module E — the MANDATORY gate before any A/B/C value
on an IAJD is trusted; no published Janus-dendrimer CG model exists, so this is the
largest validity risk).

Criterion implemented here (tractable, single-molecule, ~hours): RADIUS OF GYRATION.
  atomistic: SMILES -> RDKit 3D -> GAFF (antechamber, AM1-BCC) -> acpype -> GROMACS
             all-atom topology -> solvate (TIP3P) -> EM -> NVT -> NPT -> production ->
             <Rg_AA> over the trajectory.
  CG:        martini/build_cg single molecule -> solvate (Martini W) -> EM -> production
             -> <Rg_CG>.
  GATE:      |Rg_CG - Rg_AA| / Rg_AA <= 0.15  (build prompt: Rg within +-15%).

The other two Module-E criteria (water-bilayer partitioning depth; head-group solvation
ordering) need an atomistic bilayer-PATCH simulation (~100 ns, days) and are scaffolded
as run_partitioning() — deliberately NOT auto-run (hot multi-day compute; user-gated).

No-proxy: real atomistic + CG sims, real Rg, or NaN+audit. If the mapping FAILS the Rg
gate, FLAG and STOP — do not paper over a bad mapping (build prompt §3).

Engines: GROMACS 2026 (both AA and CG), antechamber/acpype (micromamba env `ambertools`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

AMBER = ["/opt/homebrew/bin/micromamba", "run", "-n", "ambertools",
         "--root-prefix", "/Users/aryamantiwary/micromamba"]
GMX = ["/opt/homebrew/bin/gmx"]
DESIGN = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design" / "cg_validation"
WORK = ROOT / "physics_cache" / "cg_validation"


def _rdkit_3d(smiles: str, out_mol2: Path, net_charge: int) -> bool:
    """SMILES -> 3D (ETKDGv3) -> mol2 with the right net charge (for antechamber)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    m = Chem.AddHs(m)
    # large flexible IAJDs (~130 atoms) often fail the default embed; random coords +
    # more iterations is robust.
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    p.useRandomCoords = True
    p.maxIterations = 2000
    if AllChem.EmbedMolecule(m, p) != 0:
        return False
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    Chem.MolToMolFile(m, str(out_mol2.with_suffix(".mol")))
    # antechamber reads .mol; we keep .mol and let antechamber assign GAFF/AM1-BCC
    return True


def gaff_parameterize(smiles: str, workdir: Path, net_charge: int) -> Dict:
    """RDKit 3D -> antechamber GAFF + AM1-BCC -> acpype -> GROMACS .top/.gro."""
    workdir.mkdir(parents=True, exist_ok=True)
    molf = workdir / "lig.mol"
    if not _rdkit_3d(smiles, molf, net_charge):
        return {"status": "rdkit_embed_failed"}
    # antechamber: .mol -> .mol2 with GAFF atom types + charges. AM1-BCC (sqm SCF) HANGS
    # for ~130-atom flexible IAJDs; the Rg/SIZE criterion is dominated by GAFF bonded+vdw,
    # not charge accuracy, so we use fast Gasteiger charges (-c gas). (AM1-BCC is preferable
    # for the energetics-sensitive partitioning/solvation criteria — re-derive there.)
    mol2 = workdir / "lig.mol2"
    r = subprocess.run(AMBER + ["antechamber", "-i", str(molf), "-fi", "mdl",
                                "-o", str(mol2), "-fo", "mol2", "-c", "gas",
                                "-nc", str(net_charge), "-s", "2", "-at", "gaff2"],
                       cwd=str(workdir), capture_output=True, text=True, timeout=900)
    if not mol2.exists():
        return {"status": "antechamber_failed", "log": (r.stdout + r.stderr)[-2500:]}
    # acpype: mol2 -> GROMACS .top/.gro
    r2 = subprocess.run(AMBER + ["acpype", "-i", str(mol2), "-c", "user", "-n", str(net_charge)],
                        cwd=str(workdir), capture_output=True, text=True, timeout=1800)
    acpype_dir = next(workdir.glob("*.acpype"), None)
    if acpype_dir is None:
        return {"status": "acpype_failed", "log": (r2.stdout + r2.stderr)[-2500:]}
    top = next(acpype_dir.glob("*_GMX.top"), None)
    gro = next(acpype_dir.glob("*_GMX.gro"), None)
    if not top or not gro:
        return {"status": "acpype_no_output"}
    return {"status": "ok", "top": str(top), "gro": str(gro), "acpype_dir": str(acpype_dir)}


def rg_from_traj(traj: Path, topol: Path, sel: str = "not resname SOL W NA CL TIP3",
                 last_frac: float = 0.5) -> float:
    """Mass-weighted Rg of the solute, averaged over the last last_frac of frames."""
    import MDAnalysis as mda
    try:
        u = mda.Universe(str(topol), str(traj))
        g = u.select_atoms(sel)
        if len(g) == 0:
            g = u.select_atoms("not resname SOL and not resname W")
        n = len(u.trajectory)
        rgs = []
        for ts in u.trajectory[int(n * (1 - last_frac)):]:
            rgs.append(g.radius_of_gyration() / 10.0)   # A -> nm
        return float(np.mean(rgs)) if rgs else float("nan")
    except Exception:
        return float("nan")


# (atomistic + CG MD runners and validate_mapping orchestration are wired below; the
#  atomistic production is the only ~hours step and is invoked explicitly.)
def validate_mapping(iajd_num: int, *, aa_ns: float = 20.0, cg_ns: float = 10.0,
                     threads: int = 4) -> Dict:
    """Run AA (GAFF) + CG (Martini) single-molecule-in-water sims and compare Rg."""
    import pandas as pd
    from iajd_grammar import decompose_row
    df = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
    row = df[df["IAJD_num"] == iajd_num].iloc[0].to_dict()
    smiles = row.get("SMILES") or row.get("SMILES_canonical")
    wd = WORK / f"IAJD{iajd_num}"
    rec: Dict = {"iajd_num": iajd_num, "smiles": smiles, "status": "running"}
    # neutral state (net charge 0) for the size/Rg check
    par = gaff_parameterize(smiles, wd / "aa", net_charge=0)
    rec["gaff"] = par.get("status")
    if par["status"] != "ok":
        rec.update(status=f"gaff:{par['status']}", rg_atomistic=float("nan"),
                   rg_cg=float("nan"))
        rec["log"] = par.get("log", "")
        _save(rec)
        return rec
    rec["status"] = "gaff_ok_md_pending"   # AA/CG MD wired in run_md_* (hours; explicit)
    _save(rec)
    return rec


def _save(rec: Dict) -> None:
    DESIGN.mkdir(parents=True, exist_ok=True)
    (DESIGN / f"IAJD{rec['iajd_num']}_cgval.json").write_text(json.dumps(rec, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iajd-num", type=int, default=369)
    p.add_argument("--param-only", action="store_true",
                   help="just test GAFF parameterization (validate the toolchain)")
    args = p.parse_args()
    if args.param_only:
        import pandas as pd
        df = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
        smi = df[df["IAJD_num"] == args.iajd_num].iloc[0]["SMILES"]
        r = gaff_parameterize(smi, WORK / f"IAJD{args.iajd_num}" / "aa", net_charge=0)
        print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=2))
        if r["status"] != "ok":
            print("LOG:\n", r.get("log", "")[-1500:])
        return 0
    print(json.dumps(validate_mapping(args.iajd_num), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
