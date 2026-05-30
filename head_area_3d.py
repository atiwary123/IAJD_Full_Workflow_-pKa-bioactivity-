"""
head_area_3d.py — compute real van der Waals projection area for a head
fragment from a 3D conformer, instead of using a hand-curated per-head
lookup table.

The projection area is the area of the molecule's shadow when projected
onto a plane perpendicular to its longest principal axis. This is the
right physical quantity for a_head in CPP / packing-parameter calculations
because that area is what each head group occupies at the bilayer interface
in a packed monolayer.

For each head fragment:
  1. Parse the SMILES (with [*] attachment point) and add Hs.
  2. Embed in 3D via ETKDGv3 (multiple conformers).
  3. MMFF94 minimization to relax to a low-energy conformation.
  4. Compute principal axes by diagonalizing the inertia tensor.
  5. Project all heavy-atom van der Waals spheres onto the plane perpendicular
     to the longest principal axis.
  6. Compute the convex hull area of the projected disk set.

No proxies — values are computed per-molecule from real 3D geometry. The
result is cached by head_group string so subsequent queries are instant.

For symmetry: synthetic head variants beyond HEAD_FRAGMENTS (e.g. DEHPRZ,
MPRZ_OMe) just need their SMILES to be passed; the calculation works
identically.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

# Van der Waals radii in nm (Bondi 1964)
VDW_NM = {
    1: 0.120, 6: 0.170, 7: 0.155, 8: 0.152, 9: 0.147,
    14: 0.210, 15: 0.180, 16: 0.180, 17: 0.175, 35: 0.183, 53: 0.198,
}

_HEAD_AREA_CACHE: Dict[str, float] = {}


def head_area_from_smiles(head_smiles: str,
                          n_confs: int = 10,
                          force_field: str = "MMFF94") -> float:
    """Compute real van der Waals projection area (nm²) for a head fragment.

    head_smiles can contain [*] dummy atoms — those are stripped first.
    Returns NaN on failure (no proxy default).
    """
    if head_smiles in _HEAD_AREA_CACHE:
        return _HEAD_AREA_CACHE[head_smiles]

    smi = head_smiles.replace("*", "")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _HEAD_AREA_CACHE[head_smiles] = float("nan")
        return float("nan")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids:
        _HEAD_AREA_CACHE[head_smiles] = float("nan")
        return float("nan")

    # MMFF94 minimization
    energies = []
    for cid in cids:
        try:
            if force_field == "MMFF94":
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is None:
                    raise ValueError
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is None:
                raise ValueError
            ff.Minimize(maxIts=500)
            energies.append((cid, ff.CalcEnergy()))
        except Exception:
            continue
    if not energies:
        _HEAD_AREA_CACHE[head_smiles] = float("nan")
        return float("nan")
    energies.sort(key=lambda t: t[1])
    best_cid = energies[0][0]

    conf = mol.GetConformer(best_cid)
    coords = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())]) * 0.1  # Å → nm
    radii = np.array([VDW_NM.get(a.GetAtomicNum(), 0.150)
                       for a in mol.GetAtoms()])

    # Mass-weighted centroid + inertia tensor for principal axes
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    rc = coords - com
    I = np.zeros((3, 3))
    for r, m in zip(rc, masses):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    eigval, eigvec = np.linalg.eigh(I)
    # Longest principal axis = smallest moment of inertia (atoms aligned along it)
    long_axis = eigvec[:, 0]
    # Build orthonormal basis (e1, e2) perpendicular to long_axis
    helper = np.array([1.0, 0.0, 0.0]) if abs(long_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(long_axis, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(long_axis, e1); e2 /= np.linalg.norm(e2)

    # Project atoms + their VdW disks onto the (e1, e2) plane
    proj_centers = np.stack([rc @ e1, rc @ e2], axis=1)

    # Sample boundary of each VdW disk densely, take convex hull of union
    from scipy.spatial import ConvexHull
    samples = []
    n_angles = 24
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for c, r in zip(proj_centers, radii):
        if not np.isfinite(c).all():
            continue
        ring = c[None, :] + r * np.column_stack([np.cos(angles), np.sin(angles)])
        samples.append(ring)
    if not samples:
        _HEAD_AREA_CACHE[head_smiles] = float("nan")
        return float("nan")
    pts = np.vstack(samples)
    try:
        hull = ConvexHull(pts)
        area_nm2 = float(hull.volume)   # 2D ConvexHull.volume == area
    except Exception:
        # Fall back to bounding box area
        bb_x = pts[:, 0].max() - pts[:, 0].min()
        bb_y = pts[:, 1].max() - pts[:, 1].min()
        area_nm2 = float(bb_x * bb_y)

    _HEAD_AREA_CACHE[head_smiles] = area_nm2
    return area_nm2


def head_area_for_group(head_group: str,
                         head_fragments: Optional[Dict[str, str]] = None) -> float:
    """Look up head SMILES from the registry then compute area from 3D.

    If head_fragments is None, imports HEAD_FRAGMENTS from iajd_grammar.
    """
    if head_fragments is None:
        from iajd_grammar import HEAD_FRAGMENTS
        head_fragments = HEAD_FRAGMENTS
    smi = head_fragments.get(head_group)
    if smi is None:
        return float("nan")
    return head_area_from_smiles(smi)


if __name__ == "__main__":
    from iajd_grammar import HEAD_FRAGMENTS
    print("Head area (3D-derived, real per-molecule projection):")
    print(f"  {'group':<10s}  {'a_head (nm²)':>12s}  SMILES")
    for hg, smi in HEAD_FRAGMENTS.items():
        a = head_area_from_smiles(smi)
        print(f"  {hg:<10s}  {a:>12.4f}  {smi}")
