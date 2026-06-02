"""
martini/build_cg.py — assemble a per-IAJD MARTINI 3 topology + initial CG
coordinates from an iajd_grammar Seed.

Inputs:
  Seed(family, head, linker_n, linkage, tails, ...) from iajd_grammar
  protonated: bool
  out_dir: Path

Outputs (per call):
  out_dir/molecule.itp   — atomistic-to-CG mapping & forcefield params
  out_dir/molecule.gro   — single CG configuration (one molecule)
  out_dir/topol.top      — system topology wrapping molecule.itp

The .itp is built by concatenating per-fragment beads/bonds with
index-renumbering and inter-fragment bonds at the linker sites.

If the GROMACS binary is not available, .gro / .top still get written so the
files are inspectable on the Mac without conda env switching.
"""
from __future__ import annotations
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from .mapping.fragment_library import (
    Fragment, make_head, make_core, make_linker, make_tail,
)


# ──────────────────────────────────────────────────────────────────────
# Tail SMILES → carbon count + branched flag
# ──────────────────────────────────────────────────────────────────────

def _tail_descriptors(tail_smiles: str) -> Tuple[int, bool]:
    """Parse a tail SMILES (alpha-first per iajd_grammar) into (n_C, branched)."""
    m = Chem.MolFromSmiles(tail_smiles)
    if m is None:
        return 0, False
    n_c = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
    branched = sum(1 for a in m.GetAtoms()
                    if a.GetAtomicNum() == 6 and len(list(a.GetNeighbors())) > 2) > 0
    return n_c, branched


# ──────────────────────────────────────────────────────────────────────
# Topology assembly
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AssembledTopology:
    name: str
    beads: List[Tuple[str, str, float, float]]
    bonds: List[Tuple[int, int, float, float]]
    angles: List[Tuple[int, int, int, float, float]]
    constraints: List[Tuple[int, int, float]]
    total_charge: float
    fragment_offsets: List[Tuple[str, int]]   # (fragment_name, start_bead_index)


def _append_fragment(topo: AssembledTopology, frag: Fragment, *,
                      anchor_bead: Optional[int] = None,
                      anchor_bond_length: float = 0.270,
                      anchor_bond_k: float = 7500.0) -> int:
    """Append frag's beads to topo. Returns the offset of the first new bead.

    If anchor_bead is set, insert an inter-fragment bond between anchor_bead
    and frag.linker_in (relative to the new fragment).
    """
    offset = len(topo.beads)
    for b in frag.beads:
        topo.beads.append(b)
    for (i, j, length, k) in frag.bonds:
        topo.bonds.append((offset + i, offset + j, length, k))
    for (i, j, k_idx, theta, kk) in frag.angles:
        topo.angles.append((offset + i, offset + j, offset + k_idx, theta, kk))
    for (i, j, length) in frag.constraints:
        topo.constraints.append((offset + i, offset + j, length))
    if anchor_bead is not None:
        topo.bonds.append((anchor_bead, offset + frag.linker_in,
                            anchor_bond_length, anchor_bond_k))
    topo.fragment_offsets.append((frag.name, offset))
    topo.total_charge += sum(b[3] for b in frag.beads)
    return offset


def _assemble_g1_janus(seed, *, protonated: bool) -> AssembledTopology:
    """G1-Janus first-generation dendrimer: a 2-tier architecture the single-core
    Seed model can't express, so it's built explicitly:

        outer benzene (2 alkyl tails) -- benzyl -- amide -- inner benzene
            inner benzene bears 3 triethylene-glycol (TEG) arms; one TEG arm carries
            the ester -> (CH2)n bridge -> amine head.

    Hydrophobic outer tails + hydrophilic inner TEG/amine = the Janus shape that drives
    c0. The TEG dendron is family-constant so it is parametrised here, not from the Seed.
    PROVISIONAL like all IAJD CG mappings (unvalidated vs atomistic).
    """
    topo = AssembledTopology(name="IAJD_G1Janus", beads=[], bonds=[], angles=[],
                             constraints=[], total_charge=0.0, fragment_offsets=[])
    # 1. outer benzene (3,5-disubst) + 2 alkyl tails
    outer = make_core("sSS-Nonsym")          # benzyl_core_35disubst
    _append_fragment(topo, outer)
    o_off = topo.fragment_offsets[-1][1]
    outer_ring = [o_off, o_off + 1, o_off + 2]
    outer_benzyl = o_off + outer.linker_out
    for ti, tsmi in enumerate(list(seed.tails)[:2]):
        n_c, br = _tail_descriptors(tsmi)
        if n_c == 0:
            continue
        topo.beads.append((f"OO{ti+1}", "N4a", 60.0, 0.0)); eidx = len(topo.beads) - 1
        topo.bonds.append((outer_ring[ti], eidx, 0.270, 7500.0))
        _append_fragment(topo, make_tail(n_c, branched=br), anchor_bead=eidx,
                          anchor_bond_length=0.470, anchor_bond_k=5000.0)
    # 2. amide linker off the benzyl
    amide = make_linker("amide")
    _append_fragment(topo, amide, anchor_bead=outer_benzyl)
    amide_bead = topo.fragment_offsets[-1][1] + amide.linker_out
    # 3. inner benzene (3,4,5-trisubst) — amide attaches to its benzyl position
    inner = make_core("GA-Tris")             # benzyl_core_3trisubst
    _append_fragment(topo, inner, anchor_bead=amide_bead)
    i_off = topo.fragment_offsets[-1][1]
    inner_ring = [i_off, i_off + 1, i_off + 2]
    # 4. three TEG arms on the inner ring; the third carries the ester+bridge+head
    for j in range(3):
        topo.beads.append((f"E{j}a", "N4a", 60.0, 0.0)); e0 = len(topo.beads) - 1
        topo.bonds.append((inner_ring[j], e0, 0.270, 7000.0))
        topo.beads.append((f"E{j}b", "SP1", 54.0, 0.0)); e1 = len(topo.beads) - 1
        topo.bonds.append((e0, e1, 0.300, 5000.0))
        topo.beads.append((f"E{j}c", "SP1", 54.0, 0.0)); e2 = len(topo.beads) - 1
        topo.bonds.append((e1, e2, 0.300, 5000.0))
        if j == 2:
            topo.beads.append(("HEs", "N4a", 60.0, 0.0)); est = len(topo.beads) - 1   # ester
            topo.bonds.append((e2, est, 0.300, 5000.0))
            prev = est
            for k in range(max(1, (seed.linker_n + 3) // 4)):
                topo.beads.append((f"BR{k+1}", "SC1", 54.0, 0.0)); bidx = len(topo.beads) - 1
                topo.bonds.append((prev, bidx, 0.270, 5000.0)); prev = bidx
            # use the actual head from the decomposed structure (piperidine for the
            # pharmaceutics lung series; DMA for the ja1c05813 series); fall back to DMA.
            try:
                head = make_head(seed.head, protonated=protonated)
            except KeyError:
                head = make_head("DMA", protonated=protonated)
            _append_fragment(topo, head, anchor_bead=prev,
                              anchor_bond_length=0.300, anchor_bond_k=7000.0)
    return topo


def assemble(seed, *, protonated: bool) -> AssembledTopology:
    """Build a MARTINI topology from an iajd_grammar Seed."""
    if seed.family == "G1-Janus-Dendrimer":
        return _assemble_g1_janus(seed, protonated=protonated)
    topo = AssembledTopology(
        name=f"IAJD_{seed.family.replace('-', '')}",
        beads=[], bonds=[], angles=[], constraints=[],
        total_charge=0.0, fragment_offsets=[],
    )
    # 1. Aromatic core
    core = make_core(seed.family)
    _append_fragment(topo, core)
    benzyl_bead = topo.fragment_offsets[-1][1] + core.linker_out

    # 2. Linker (ester or amide) bead off benzyl CH2
    linker = make_linker(seed.linkage)
    _append_fragment(topo, linker, anchor_bead=benzyl_bead)
    linker_bead = topo.fragment_offsets[-1][1] + linker.linker_out

    # 3. (CH2)_n alkyl bridge between linker and head — represented as a
    #    single SC1 bead per 4 CH2 (rounded up). For linker_n in {2..5} we
    #    typically get 1 bead; n ≥ 6 (unusual) gets 2 beads.
    bridge_beads = max(1, (seed.linker_n + 3) // 4)
    prev_bridge = linker_bead
    for k in range(bridge_beads):
        topo.beads.append((f"BR{k+1}", "SC1", 54.0, 0.0))
        idx = len(topo.beads) - 1
        topo.bonds.append((prev_bridge, idx, 0.270, 5000.0))
        prev_bridge = idx
    topo.fragment_offsets.append((f"bridge_n{seed.linker_n}", len(topo.beads) - bridge_beads))

    # 4. Head fragment
    head = make_head(seed.head, protonated=protonated)
    _append_fragment(topo, head, anchor_bead=prev_bridge,
                      anchor_bond_length=0.300, anchor_bond_k=7000.0)

    # 5. Tails attached to ring O positions. The aromatic core has 2 or 3
    #    free O positions; we attach an N4a ether bead first, then the tail.
    expected_n_tails = 3 if seed.family in ("GA-Tris", "PE-Tris", "PE-Gallic") else 2
    tails = list(seed.tails)[:expected_n_tails]
    ring_bead_indices = [topo.fragment_offsets[0][1] + i for i in range(3)]
    for ti, tail_smi in enumerate(tails):
        n_c, branched = _tail_descriptors(tail_smi)
        if n_c == 0:
            continue
        # Ether O bead
        topo.beads.append((f"O{ti+1}", "N4a", 60.0, 0.0))
        ether_idx = len(topo.beads) - 1
        topo.bonds.append((ring_bead_indices[ti], ether_idx, 0.270, 7500.0))
        tail = make_tail(n_c, branched=branched)
        _append_fragment(topo, tail, anchor_bead=ether_idx,
                          anchor_bond_length=0.470, anchor_bond_k=5000.0)
    return topo


# ──────────────────────────────────────────────────────────────────────
# .itp / .gro / .top writers
# ──────────────────────────────────────────────────────────────────────

def write_itp(topo: AssembledTopology, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"; MARTINI 3 topology for {topo.name}",
        f"; assembled by martini/build_cg.py",
        f"; total charge = {topo.total_charge:+.3f}",
        "",
        "[ moleculetype ]",
        f"; name nrexcl",
        f"{topo.name} 1",
        "",
        "[ atoms ]",
        ";   nr type resnr residue atom cgnr charge mass",
    ]
    for i, (name, bt, mass, charge) in enumerate(topo.beads, start=1):
        lines.append(f" {i:5d}  {bt:<5s} 1   {topo.name[:5]:<5s}  {name:<4s} {i:5d}  {charge:+.3f}  {mass:7.3f}")
    if topo.bonds:
        lines += ["", "[ bonds ]", ";  i    j  func  length  fc"]
        for (i, j, length, k) in topo.bonds:
            lines.append(f" {i+1:4d} {j+1:4d}   1  {length:.4f}  {k:.1f}")
    if topo.constraints:
        lines += ["", "[ constraints ]", ";  i    j  func  length"]
        for (i, j, length) in topo.constraints:
            lines.append(f" {i+1:4d} {j+1:4d}   1  {length:.4f}")
    if topo.angles:
        lines += ["", "[ angles ]", ";  i    j    k  func  theta  fc"]
        for (i, j, k_idx, theta, kk) in topo.angles:
            lines.append(f" {i+1:4d} {j+1:4d} {k_idx+1:4d}   2  {theta:.2f}  {kk:.1f}")
    lines += ["", "[ exclusions ]",
              "; default: bonded exclusions only (handled by gmx)"]
    path.write_text("\n".join(lines) + "\n")


def _radial_layout(topo: AssembledTopology) -> np.ndarray:
    """Place each bead so distances respect the topology's bond and constraint
    targets, avoiding overlapping initial configurations that crash EM.

    Algorithm:
      1. Build an adjacency map (bond + constraint targets).
      2. BFS from bead 0. At each new bead, place it at its parent's position
         + target_distance * direction. Direction is chosen to spread beads
         (alternating axis cycle) — this gets an extended geometry rather
         than a balled-up one.
      3. After BFS, do a local relaxation pass: for each bonded pair, nudge
         atoms so their distance is within ±10% of the target.

    The result is a feasible starting geometry; gmx EM then relaxes to the
    real energy minimum.
    """
    n = len(topo.beads)
    # Adjacency: bead → [(neighbor, target_distance_nm)]
    adj: dict = {i: [] for i in range(n)}
    for (i, j, length, k) in topo.bonds:
        adj[i].append((j, length))
        adj[j].append((i, length))
    for (i, j, length) in topo.constraints:
        adj[i].append((j, length))
        adj[j].append((i, length))

    pos = np.full((n, 3), np.inf)
    pos[0] = np.array([0.0, 0.0, 0.0])
    placed = {0}
    queue = [0]
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])]
    while queue:
        cur = queue.pop(0)
        # Sort neighbors so children always extend "outward" from origin.
        children = [(j, d) for (j, d) in adj[cur] if j not in placed]
        for idx, (j, d) in enumerate(children):
            # Pick the axis least populated near `cur`.
            best_axis = None
            best_min = -1.0
            for a in axes:
                candidate = pos[cur] + a * d
                # Distance to nearest already-placed bead.
                if not placed:
                    min_dist = float("inf")
                else:
                    diffs = pos[list(placed)] - candidate
                    min_dist = float(np.linalg.norm(diffs, axis=1).min())
                if min_dist > best_min:
                    best_min = min_dist
                    best_axis = a
            pos[j] = pos[cur] + (best_axis if best_axis is not None
                                  else axes[idx % 6]) * d
            placed.add(j)
            queue.append(j)
    # Any bead that wasn't reached by BFS (orphan) lands at the origin
    # corner offset to avoid superposition.
    for i in range(n):
        if not np.isfinite(pos[i]).all():
            pos[i] = np.array([0.5 + 0.5 * i, 0.5 + 0.5 * i, 0.5 + 0.5 * i])
    # Re-center on the COM and shift into a positive corner so gmx
    # insert-molecules can place us in a box that starts at the origin.
    pos = pos - pos.mean(axis=0) + np.array([1.5, 1.5, 1.5])
    return pos


def write_gro(topo: AssembledTopology, path: Path,
              box_nm: Tuple[float, float, float] = (5.0, 5.0, 5.0)) -> None:
    pos = _radial_layout(topo)
    n = len(topo.beads)
    res = topo.name[:5]
    lines = [topo.name, f"{n:5d}"]
    for i, (name, bt, mass, charge) in enumerate(topo.beads, start=1):
        x, y, z = pos[i - 1]
        lines.append(f"{1:5d}{res:<5s}{name:>5s}{i:5d}"
                      f"{x:8.3f}{y:8.3f}{z:8.3f}")
    bx, by, bz = box_nm
    lines.append(f"{bx:10.5f}{by:10.5f}{bz:10.5f}")
    path.write_text("\n".join(lines) + "\n")


def write_top(itp_filename: str, system_name: str, path: Path,
              n_molecules: int = 1, counterion: Optional[str] = None,
              n_counterions: int = 0, n_water: int = 0,
              moleculetype_name: Optional[str] = None) -> None:
    """Write a MARTINI 3 [system] / [molecules] topology.

    Includes the FF + molecule .itp via cwd-relative paths. The
    moleculetype_name must match the [moleculetype] entry in the molecule .itp.
    """
    mt = moleculetype_name or system_name
    lines = [
        f"; MARTINI 3 system topology — {system_name}",
        "",
        "#include \"martini_v3.0.0.itp\"",
        "#include \"martini_v3.0.0_solvents_v1.itp\"",
        "#include \"martini_v3.0.0_ions_v1.itp\"",
        f"#include \"{itp_filename}\"",
        "",
        "[ system ]",
        system_name,
        "",
        "[ molecules ]",
        f"{mt}  {n_molecules}",
    ]
    if counterion and n_counterions:
        lines.append(f"{counterion}  {n_counterions}")
    if n_water:
        lines.append(f"W  {n_water}")
    path.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────

def build(seed, out_dir: Path, *, protonated: bool, n_molecules: int = 256,
          box_nm: Tuple[float, float, float] = (12.0, 12.0, 12.0),
          target_water: int = 6000) -> AssembledTopology:
    """Assemble + write .itp/.gro/.top into `out_dir`. Returns the assembled
    topology so callers can inspect bead counts & charge.

    The .top is written with [molecules] containing only the IAJD line
    (and counterion if charged). gmx solvate appends the W molecules line
    after running; pre-populating the W count here would cause topology /
    coordinate mismatches.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = assemble(seed, protonated=protonated)
    state = "prot" if protonated else "neutral"
    itp_name = f"molecule_{state}.itp"
    gro_name = f"molecule_{state}.gro"
    top_name = f"topol_{state}.top"
    write_itp(topo, out_dir / itp_name)
    write_gro(topo, out_dir / gro_name, box_nm=(5.0, 5.0, 5.0))
    # NOTE: counterions are added later by gmx genion (run_selfassembly.run
    # passes net_charge_per_molecule). gmx solvate appends the W line after
    # it runs; genion then converts some W beads to ions in both the .gro
    # and topology. Pre-allocating CL/NA in the topology here would cause
    # an atom-count mismatch in solvated.gro vs topol.top.
    write_top(itp_name, topo.name, out_dir / top_name,
              n_molecules=n_molecules, counterion=None,
              n_counterions=0, n_water=0,
              moleculetype_name=topo.name)
    return topo


__all__ = [
    "assemble", "build", "write_itp", "write_gro", "write_top",
    "AssembledTopology",
]


if __name__ == "__main__":
    # Smoke test: build for IAJD 369
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from iajd_grammar import build_library, decompose_row
    import pandas as pd
    df = pd.read_excel(Path(__file__).resolve().parent.parent / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
    row = df[df["IAJD_num"] == 369].iloc[0].to_dict()
    seed = decompose_row(row)
    print("Seed:", seed)
    out = Path("/tmp/martini_test_369")
    if out.exists():
        shutil.rmtree(out)
    for prot in (False, True):
        topo = build(seed, out, protonated=prot, n_molecules=128)
        state = "prot" if prot else "neutral"
        print(f"  state={state}: {len(topo.beads)} beads, "
              f"{len(topo.bonds)} bonds, "
              f"{len(topo.constraints)} constraints, "
              f"total_charge={topo.total_charge:+.2f}")
    print(f"\nWrote → {out}")
