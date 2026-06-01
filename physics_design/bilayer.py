"""
physics_design/bilayer.py — construct a flat Martini-3 lipid bilayer patch for a
single lipid species, ready for a tensionless GROMACS run.

Pipeline:
  1. Lay out one lipid as a head-up template using a BFS over its bond graph
     (z by graph depth from the head bead; the two acyl tails splayed in x so
     they don't overlap). EM later relaxes the rough geometry.
  2. Tile N_per_leaflet copies on an x-y grid; mirror for the lower leaflet
     (heads pointing outward, tails meeting near the midplane with a small gap).
  3. Write lipids-only .gro + a system .top, then `gmx solvate` to add Martini W
     water above/below the membrane.
  4. Strip any water beads that solvate placed inside the hydrophobic slab, so we
     start from a clean bilayer (waters in the core are expelled within ps anyway,
     but removing them avoids EM instabilities).

No proxies: every coordinate is a real placement; nothing is fabricated to hit a
target observable. The builder only seeds geometry — all physics comes from the
subsequent GROMACS equilibration + production.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lipid_library import Lipid, write_single_itp

_FF_DIR = Path(__file__).resolve().parent.parent / "martini" / "ff"


# ──────────────────────────────────────────────────────────────────────
# Single-lipid template geometry
# ──────────────────────────────────────────────────────────────────────

def _adjacency(lip: Lipid) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {i: [] for i in range(lip.n_beads)}
    for (i, j, _r, _k) in lip.bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def lipid_template(lip: Lipid, *, dz: float = 0.27, splay: float = 0.13,
                   head_idx: int = 0) -> np.ndarray:
    """Return Nx3 local coords with the head bead at the top (z=0) and tails
    descending to negative z. Tails are splayed in x by branch.

    `head_idx` is the bead placed at the interface (z=0); graph depth from it sets
    z. Standard lipids store the head as bead 0 (default); IAJDs from build_cg order
    beads core-first, so their ionizable head (Lipid.head_bead) must be passed here
    or the bilayer is built upside-down (tails at the water interface)."""
    adj = _adjacency(lip)
    n = lip.n_beads
    depth = [-1] * n
    parent = [-1] * n
    depth[head_idx] = 0
    q = deque([head_idx])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                parent[v] = u
                q.append(v)
    # Branch root = lowest-depth bead with >2 bonded neighbours (GL1 for PC/PE).
    branch = None
    for u in order:
        if len(adj[u]) > 2:
            branch = u
            break
    # Assign an x-offset per bead: descendants of each branch-child get a distinct
    # lateral offset so the two acyl tails do not sit on top of each other.
    xoff = [0.0] * n
    if branch is not None:
        kids = [v for v in adj[branch] if parent[v] == branch]  # children, not the parent side
        offsets = np.linspace(-splay, splay, num=max(1, len(kids)))
        for child, off in zip(kids, offsets):
            # BFS the subtree rooted at `child` (not crossing back through branch)
            sub = deque([child])
            seen = {branch, child}
            xoff[child] = off
            while sub:
                u = sub.popleft()
                xoff[u] = off
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        sub.append(v)
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i, 0] = xoff[i]
        coords[i, 1] = 0.0
        coords[i, 2] = -depth[i] * dz
    bonded = {frozenset((i, j)) for (i, j, _r, _k) in lip.bonds}
    coords = _spread_xy_overlaps(coords, bonded)
    return coords


def _spread_xy_overlaps(coords: np.ndarray, bonded_pairs, min_dist: float = 0.25,
                        max_iter: int = 400) -> np.ndarray:
    """Push NON-BONDED beads apart IN THE XY PLANE (preserving the head-up z set by graph
    depth) until no non-bonded pair is closer than `min_dist` nm.

    A true NO-OP for standard lipids: their only sub-`min_dist` contacts are either
    directly bonded (skipped — they're LJ-excluded and held by the bond) or the two
    splayed tails at ~0.26 nm (> min_dist), so the validated DOPE/DOPC path is byte-for-
    byte unchanged. It rescues complex IAJD topologies whose (depth, branch) layout
    collapses several NON-bonded beads (rings, multiple same-depth branches) onto
    IDENTICAL coordinates → infinite LJ force → EM NaN crash (IAJD 369: 460 coincident
    pairs). Only non-bonded pairs cause the crash (bonded pairs are LJ-excluded and feel
    a finite harmonic force); EM + equilibration relax the rest. min_dist=0.25 sits below
    the ~0.26 nm legitimate cross-tail spacing yet matches the proven-OK lipid contact
    regime, so EM handles the separated IAJD start the same way it handles a lipid."""
    n = len(coords)
    if n < 2:
        return coords

    def _close_nonbonded(c):
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((i, j)) in bonded_pairs:
                    continue
                d = c[j] - c[i]
                if float(d @ d) < min_dist * min_dist:
                    return True
        return False

    if not _close_nonbonded(coords):
        return coords                       # standard lipids: nothing to do
    c = coords.astype(float).copy()
    GA = 2.399963322759                      # golden angle (rad), deterministic tie-break
    for i in range(n):                       # break exact ties so push directions exist
        c[i, 0] += 0.04 * np.cos(i * GA)
        c[i, 1] += 0.04 * np.sin(i * GA)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((i, j)) in bonded_pairs:
                    continue
                dx = c[j, 0] - c[i, 0]; dy = c[j, 1] - c[i, 1]; dz = c[j, 2] - c[i, 2]
                r = (dx * dx + dy * dy + dz * dz) ** 0.5
                if r < min_dist:
                    rxy = (dx * dx + dy * dy) ** 0.5
                    if rxy < 1e-6:           # stacked in xy -> deterministic direction
                        ang = (2 * i + j) * GA
                        dx, dy, rxy = np.cos(ang), np.sin(ang), 1.0
                    push = 0.5 * (min_dist - r) + 1e-3
                    c[i, 0] -= push * dx / rxy; c[i, 1] -= push * dy / rxy
                    c[j, 0] += push * dx / rxy; c[j, 1] += push * dy / rxy
                    moved = True
        if not moved:
            break
    return c


# ──────────────────────────────────────────────────────────────────────
# Bilayer assembly
# ──────────────────────────────────────────────────────────────────────

def build_bilayer_coords(lip: Lipid, n_per_leaflet: int, *,
                         apl_nm2: float = 0.66, midgap_nm: float = 0.35,
                         water_pad_nm: float = 4.0, head_idx: int = 0
                         ) -> Tuple[np.ndarray, List[str], List[str], Tuple[float, float, float], float]:
    """Build bilayer bead coordinates (lipids only).

    Returns (positions[M,3] nm, atom_names[M], res_names[M], box(nm), z_mid)."""
    templ = lipid_template(lip, head_idx=head_idx)
    span_z = float(templ[:, 2].max() - templ[:, 2].min())  # leaflet height (nm)
    s = float(np.sqrt(apl_nm2))                              # grid spacing (nm)
    ncol = int(np.ceil(np.sqrt(n_per_leaflet)))
    Lx = Ly = ncol * s
    # leaflet stacking: upper leaflet heads at +z_top, tails down to +midgap/2.
    half_bilayer = span_z + midgap_nm / 2.0
    Lz = 2.0 * (half_bilayer + water_pad_nm)
    z_mid = Lz / 2.0

    pos: List[np.ndarray] = []
    names: List[str] = []
    resn: List[str] = []
    bead_names = lip.bead_names

    def place(leaflet_up: bool, count: int):
        placed = 0
        for gy in range(ncol):
            for gx in range(ncol):
                if placed >= count:
                    return
                cx = (gx + 0.5) * s
                cy = (gy + 0.5) * s
                t = templ.copy()
                if leaflet_up:
                    # head at top: shift so tail tips sit at +midgap/2 above mid
                    t[:, 2] = t[:, 2] - templ[:, 2].min() + midgap_nm / 2.0 + z_mid
                else:
                    # mirror through midplane: heads point down (-z, outward)
                    t[:, 2] = -t[:, 2]
                    t[:, 2] = t[:, 2] - t[:, 2].max() - midgap_nm / 2.0 + z_mid
                t[:, 0] += cx
                t[:, 1] += cy
                for b in range(lip.n_beads):
                    pos.append(t[b])
                    names.append(bead_names[b])
                    resn.append(lip.name)
                placed += 1

    place(True, n_per_leaflet)
    place(False, n_per_leaflet)
    arr = np.array(pos)
    # wrap into box laterally
    arr[:, 0] = np.mod(arr[:, 0], Lx)
    arr[:, 1] = np.mod(arr[:, 1], Ly)
    return arr, names, resn, (Lx, Ly, Lz), z_mid


def write_gro(positions: np.ndarray, atom_names: List[str], res_names: List[str],
              box: Tuple[float, float, float], path: Path, title: str = "bilayer") -> None:
    n = len(positions)
    lines = [title, f"{n:5d}"]
    # residue numbering: increment when res changes (one residue per lipid here)
    resid = 0
    prev_res_start = -1
    beads_per = {}
    # Simpler: assign resid by chunks — each lipid is a fixed bead count block.
    # Detect block size from first contiguous same-name run is unreliable; instead
    # the caller passes res_names with repeats; we group consecutive identical
    # res blocks of equal length. We infer block size as run of identical name
    # until it changes OR until we see the first atom name repeat.
    # Robust approach: number residues so that each set of beads whose atom names
    # form one lipid gets a residue id. We rely on the first atom name marking a
    # new residue.
    first_name = atom_names[0] if atom_names else None
    for i in range(n):
        if atom_names[i] == first_name:
            resid += 1
        x, y, z = positions[i]
        rn = res_names[i][:5]
        an = atom_names[i][:5]
        lines.append(f"{resid % 100000:5d}{rn:<5s}{an:>5s}{(i+1) % 100000:5d}"
                     f"{x:8.3f}{y:8.3f}{z:8.3f}")
    bx, by, bz = box
    lines.append(f"{bx:10.5f}{by:10.5f}{bz:10.5f}")
    path.write_text("\n".join(lines) + "\n")


def write_topology(lip: Lipid, n_lipids: int, n_water: int, path: Path,
                   itp_name: str) -> None:
    lines = [
        f"; flat {lip.name} bilayer system",
        '#include "martini_v3.0.0.itp"',
        '#include "martini_v3.0.0_solvents_v1.itp"',
        '#include "martini_v3.0.0_ions_v1.itp"',
        f'#include "{itp_name}"',
        "",
        "[ system ]",
        f"{lip.name} bilayer",
        "",
        "[ molecules ]",
        f"{lip.name}  {n_lipids}",
    ]
    if n_water:
        lines.append(f"W  {n_water}")
    path.write_text("\n".join(lines) + "\n")


def _link_ff(workdir: Path) -> None:
    for fname in ("martini_v3.0.0.itp", "martini_v3.0.0_solvents_v1.itp",
                  "martini_v3.0.0_ions_v1.itp"):
        src = _FF_DIR / fname
        dst = workdir / fname
        if src.exists() and not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except (OSError, NotImplementedError):
                shutil.copy(src, dst)


def read_gro(path: Path) -> Tuple[np.ndarray, List[str], List[str], Tuple[float, float, float]]:
    txt = path.read_text().splitlines()
    n = int(txt[1])
    pos = np.zeros((n, 3))
    names, resn = [], []
    for i in range(n):
        line = txt[2 + i]
        resn.append(line[5:10].strip())
        names.append(line[10:15].strip())
        pos[i] = [float(line[20:28]), float(line[28:36]), float(line[36:44])]
    box = tuple(float(x) for x in txt[2 + n].split()[:3])
    return pos, names, resn, box


def build_system(lip: Lipid, workdir: Path, gmx_cmd: List[str], *,
                 n_per_leaflet: int = 64, apl_nm2: float = 0.66,
                 water_per_lipid: float = 45.0, seed: int = 1,
                 strip_core_waters: bool = True, head_idx: int = 0) -> Dict:
    """Build the solvated flat bilayer in `workdir`. Returns an info dict with the
    final .gro/.top paths and counts. Uses gmx solvate for water.

    `head_idx` = the bead placed at the water interface (0 for standard lipids;
    Lipid.head_bead for core-first IAJD topologies from build_cg)."""
    workdir.mkdir(parents=True, exist_ok=True)
    _link_ff(workdir)
    itp_name = f"{lip.name}.itp"
    write_single_itp(lip, workdir / itp_name)

    pos, names, resn, box, z_mid = build_bilayer_coords(
        lip, n_per_leaflet, apl_nm2=apl_nm2, head_idx=head_idx)
    n_lipids = 2 * n_per_leaflet
    lipids_gro = workdir / "lipids.gro"
    write_gro(pos, names, resn, box, lipids_gro, title=f"{lip.name} bilayer")

    # topology with zero water first (solvate appends/needs the count updated)
    top = workdir / "system.top"
    write_topology(lip, n_lipids, 0, top, itp_name)

    # gmx solvate: fill with Martini water box
    water_box = _FF_DIR / "water.gro"
    solv_gro = workdir / "solvated.gro"
    target_water = int(water_per_lipid * n_lipids)
    cmd = gmx_cmd + ["solvate", "-cp", str(lipids_gro), "-cs", str(water_box),
                     "-o", str(solv_gro), "-p", str(top), "-radius", "0.21"]
    r = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not solv_gro.exists():
        return {"status": "solvate_failed", "log": (r.stdout + r.stderr)[-3000:]}

    # Read back, optionally strip waters inside the hydrophobic slab.
    pos2, names2, resn2, box2 = read_gro(solv_gro)
    if strip_core_waters:
        templ = lipid_template(lip, head_idx=head_idx)
        span_z = float(templ[:, 2].max() - templ[:, 2].min())
        # hydrophobic half-thickness ~ leaflet span minus head beads (~2 beads)
        hydro_half = max(0.8, span_z - 0.6)
        zc = box2[2] / 2.0
        keep = []
        for i in range(len(pos2)):
            if resn2[i] == "W" and abs(pos2[i, 2] - zc) < hydro_half:
                continue
            keep.append(i)
        pos2 = pos2[keep]
        names2 = [names2[i] for i in keep]
        resn2 = [resn2[i] for i in keep]

    n_water = sum(1 for rn in resn2 if rn == "W")
    final_gro = workdir / "start.gro"
    write_gro(pos2, names2, resn2, box2, final_gro, title=f"{lip.name} bilayer")
    write_topology(lip, n_lipids, n_water, top, itp_name)

    return {"status": "ok", "gro": str(final_gro), "top": str(top),
            "itp": str(workdir / itp_name), "n_lipids": n_lipids,
            "n_water": n_water, "box_nm": list(box2),
            "n_per_leaflet": n_per_leaflet}


if __name__ == "__main__":
    import argparse
    from .lipid_library import load_lipid
    p = argparse.ArgumentParser()
    p.add_argument("lipid")
    p.add_argument("--workdir", default="/tmp/bilayer_test")
    p.add_argument("--n-per-leaflet", type=int, default=64)
    args = p.parse_args()
    gmx = [os.environ.get("GMX", "/opt/homebrew/bin/gmx")]
    lip = load_lipid(args.lipid)
    info = build_system(lip, Path(args.workdir), gmx,
                        n_per_leaflet=args.n_per_leaflet)
    print(info)
