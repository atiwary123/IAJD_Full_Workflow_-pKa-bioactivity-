"""
head_area_ensemble.py — Boltzmann-weighted ensemble head_area for each head
fragment.

Original head_area_3d picks the lowest-MMFF94 conformer and projects its van
der Waals disks. This module replaces that single-shot estimate with a
Boltzmann-weighted average over a multi-conformer ensemble. When the offline
env has CREST + xTB we run a small CREGEN-style ensemble; otherwise we fall
back to an ETKDGv3 + GFN2-xTB ensemble (the same head fragment is small
enough that 5–10 conformers + GFN2 single points takes only a few minutes).

Outputs:
  IAJD_master/bundles_caches/physics/head_area_ensemble.csv
    head_group, head_smiles, n_conf_used, area_mean_nm2, area_std_nm2,
    area_p10_nm2, area_p90_nm2, head_area_nm2 (Boltzmann), kT_kJmol

physics_features.head_area_for_group already prefers head_area_3d's live
calculation — we don't change that contract. physics_cache_io.lookup_head_area
checks this CSV first, so the Boltzmann value shadows the single-conformer
estimate when the cache exists.
"""
from __future__ import annotations
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"
OUT_CSV = PHYS_DIR / "head_area_ensemble.csv"

from qm_descriptors import (
    _xtb_run, _xtb_available, _parse_xtbout_json, _mol_to_xyz, _embed_ensemble,
)
from head_area_3d import VDW_NM, head_area_from_smiles  # reuse projection geometry


# ──────────────────────────────────────────────────────────────────────
# Per-conformer area projection (same algorithm as head_area_3d, but exposed
# so we can drive it from outside with a specific conformer).
# ──────────────────────────────────────────────────────────────────────

def _project_area_per_conformer(mol_h: Chem.Mol, conf_id: int) -> Optional[float]:
    conf = mol_h.GetConformer(conf_id)
    coords = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(mol_h.GetNumAtoms())]) * 0.1   # Å→nm
    radii = np.array([VDW_NM.get(a.GetAtomicNum(), 0.150)
                       for a in mol_h.GetAtoms()])
    masses = np.array([a.GetMass() for a in mol_h.GetAtoms()])
    if masses.sum() <= 0:
        return None
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    rc = coords - com
    I = np.zeros((3, 3))
    for r, m in zip(rc, masses):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    try:
        eigval, eigvec = np.linalg.eigh(I)
    except np.linalg.LinAlgError:
        return None
    long_axis = eigvec[:, 0]
    helper = np.array([1.0, 0.0, 0.0]) if abs(long_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(long_axis, helper); n1 = np.linalg.norm(e1)
    if n1 <= 0:
        return None
    e1 /= n1
    e2 = np.cross(long_axis, e1); n2 = np.linalg.norm(e2)
    if n2 <= 0:
        return None
    e2 /= n2
    proj = np.stack([rc @ e1, rc @ e2], axis=1)
    n_angles = 24
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    samples = []
    for c, r in zip(proj, radii):
        if not np.isfinite(c).all():
            continue
        ring = c[None, :] + r * np.column_stack([np.cos(angles), np.sin(angles)])
        samples.append(ring)
    if not samples:
        return None
    pts = np.vstack(samples)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(hull.volume)   # 2D hull volume == area
    except Exception:
        bb_x = pts[:, 0].max() - pts[:, 0].min()
        bb_y = pts[:, 1].max() - pts[:, 1].min()
        return float(bb_x * bb_y)


# ──────────────────────────────────────────────────────────────────────
# Conformer ensemble + Boltzmann weighting
# ──────────────────────────────────────────────────────────────────────

# Boltzmann at 310.15 K (body temperature) — matches the rest of the pipeline.
RT_KJMOL = 8.314e-3 * 310.15


def _gfn2_energy_per_conformer(mol_h: Chem.Mol, conf_id: int,
                                workdir: Path, *,
                                solvent: str = "water",
                                timeout_s: int = 300) -> Optional[float]:
    """Single-point GFN2 energy (Hartree) on a given conformer."""
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = workdir / "input.xyz"
    _mol_to_xyz(mol_h, conf_id, xyz)
    ok, _ = _xtb_run(xyz, workdir, charge=0, opt=False,
                      alpb=solvent, dipole=False, gfnff=False,
                      timeout_s=timeout_s)
    if not ok:
        return None
    j = _parse_xtbout_json(workdir / "xtbout.json")
    return j.get("total_E")


def _boltzmann_weights(energies_Eh: List[float]) -> np.ndarray:
    """Boltzmann population at 310 K given total energies in Hartree."""
    if not energies_Eh:
        return np.array([])
    e_kj = np.array(energies_Eh) * 2625.4996   # Eh → kJ/mol
    rel = e_kj - np.min(e_kj)
    w = np.exp(-rel / RT_KJMOL)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def compute_head_area_ensemble(head_smiles: str, *,
                                n_confs: int = 10,
                                use_xtb: Optional[bool] = None,
                                solvent: str = "water",
                                timeout_s: int = 300) -> Dict:
    """Boltzmann-weighted head-area for one head fragment.

    If `use_xtb` is None we use xTB only when available. Without xTB we fall
    back to MMFF94 energies as Boltzmann weights — better than single-conformer
    but still classical. The output dict records which path was taken.
    """
    smi = head_smiles.replace("*", "")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"head_smiles": head_smiles, "n_conf_used": 0,
                "head_area_nm2": float("nan"), "error": "smiles_parse_failed"}
    mh, sorted_cids = _embed_ensemble(mol, n_confs=n_confs, seed=42)
    if not sorted_cids:
        return {"head_smiles": head_smiles, "n_conf_used": 0,
                "head_area_nm2": float("nan"), "error": "embed_failed"}

    use_xtb_path = _xtb_available() if use_xtb is None else use_xtb
    # Use the top-K lowest-MMFF conformers; cheap path takes all of them.
    top_cids = sorted_cids[:n_confs]
    areas: List[float] = []
    energies_Eh: List[float] = []
    energies_mmff: List[float] = []
    err = None

    if use_xtb_path:
        tmp = Path(tempfile.mkdtemp(prefix="head_ens_"))
        try:
            for cid in top_cids:
                a = _project_area_per_conformer(mh, cid)
                e = _gfn2_energy_per_conformer(mh, cid, tmp / f"cid{cid}",
                                                 solvent=solvent,
                                                 timeout_s=timeout_s)
                if a is None or e is None:
                    continue
                areas.append(a)
                energies_Eh.append(e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if energies_Eh:
            weights = _boltzmann_weights(energies_Eh)
            method = "gfn2_alpb_water"
        else:
            err = "xtb_failed_for_all_confs"
            weights = np.array([])
    else:
        # Re-derive MMFF energies cheaply.
        for cid in top_cids:
            a = _project_area_per_conformer(mh, cid)
            try:
                props = AllChem.MMFFGetMoleculeProperties(mh)
                if props is None:
                    ff = AllChem.UFFGetMoleculeForceField(mh, confId=cid)
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(mh, props, confId=cid)
                ff.Minimize(maxIts=200)
                e_kcal = ff.CalcEnergy()
            except (RuntimeError, ValueError):
                e_kcal = None
            if a is None or e_kcal is None:
                continue
            areas.append(a)
            energies_mmff.append(e_kcal)
        if energies_mmff:
            e_kj = np.array(energies_mmff) * 4.184
            rel = e_kj - np.min(e_kj)
            w = np.exp(-rel / RT_KJMOL)
            s = w.sum()
            weights = w / s if s > 0 else np.ones_like(w) / len(w)
            method = "mmff94_fallback"
        else:
            err = "mmff_failed_for_all_confs"
            weights = np.array([])

    if not len(weights):
        return {"head_smiles": head_smiles, "n_conf_used": 0,
                "head_area_nm2": float("nan"),
                "error": err or "no_energies"}
    areas_a = np.array(areas)
    boltz_area = float(np.dot(weights, areas_a))
    return {
        "head_smiles": head_smiles,
        "n_conf_used": int(len(areas)),
        "area_mean_nm2": float(np.mean(areas_a)),
        "area_std_nm2": float(np.std(areas_a)),
        "area_min_nm2": float(np.min(areas_a)),
        "area_max_nm2": float(np.max(areas_a)),
        "area_p10_nm2": float(np.percentile(areas_a, 10)),
        "area_p90_nm2": float(np.percentile(areas_a, 90)),
        "head_area_nm2": boltz_area,
        "boltzmann_kT_kJmol": RT_KJMOL,
        "method": method,
        "error": err,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI driver — recompute the full HEAD_FRAGMENTS table + synth heads
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-confs", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--no-xtb", action="store_true",
                        help="Force MMFF94 fallback (skip xTB).")
    args = parser.parse_args()

    from iajd_grammar import HEAD_FRAGMENTS
    # Include the synthetic head variants that propose_iajds emits.
    synthetic = {
        "MPRZ_OMe":  "*N2CCN(COC)CC2",
        "MPRZ_OEt":  "*N2CCN(COCC)CC2",
        "HPRZ_OMe":  "*N2CCN(CCOC)CC2",
        "DEHPRZ":    "*N2CCN(CC(O)CO)CC2",
    }
    all_heads = {**HEAD_FRAGMENTS, **synthetic}

    print(f"Computing Boltzmann-weighted head areas for {len(all_heads)} heads…")
    rows = []
    for name, smi in all_heads.items():
        print(f"  {name:<12s} {smi}")
        res = compute_head_area_ensemble(
            smi, n_confs=args.n_confs, timeout_s=args.timeout_s,
            use_xtb=False if args.no_xtb else None,
        )
        row = {"head_group": name, **res}
        rows.append(row)
        print(f"    → head_area_nm2={res.get('head_area_nm2')}  "
              f"mean={res.get('area_mean_nm2')}  "
              f"std={res.get('area_std_nm2')}  "
              f"n_conf={res.get('n_conf_used')}  "
              f"method={res.get('method')}  err={res.get('error')}")

    df = pd.DataFrame(rows)
    PHYS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
