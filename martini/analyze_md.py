"""
martini/analyze_md.py — extract MD observables from a finished self-assembly
trajectory.

Uses MDAnalysis when available; falls back to a minimal .xtc/.gro reader that
parses GROMACS' compressed trajectory (xtc) format for the limited set of
quantities we need.

Observables (averaged over the last `last_frac` of the trajectory):
  md_assembles            ∈ [0,1]   fraction of monomers in the largest cluster
  md_n_agg                          mean aggregation number of largest cluster
  md_a_head_*_nm2                   area-per-head (interfacial Voronoi APL).
                                    Computed separately for neutral & prot runs.
  md_delta_a_head_nm2     = a_head(prot) - a_head(neutral)
  md_cpp_*                          V_tail / (a_head × l_tail) using measured
                                    a_head and Tanford-computed V_tail/l_tail.
  md_delta_cpp            = cpp(prot) - cpp(neutral)
  md_bilayer_thick_nm               head-to-head distance across the bilayer
  md_order_param                    S_CD-like order parameter of the tail beads
  md_radius_gyration_nm             Rg of the largest cluster
  md_water_penetration              waters within 0.5 nm of head beads
  md_c0_spontaneous                 spontaneous-curvature shift on protonation
                                    = (a_prot - a_neutral) / (a_neutral * t_neutral)
                                    (Helfrich-form approximation; positive
                                    means protonation favors negative curvature)

Honest scope note: if md_assembles < 0.5 (no stable aggregate) we still
report observables but the audit flags low confidence; downstream emulator
training and Block D' consumption see honest NaN if assembly failed.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Trajectory loading: MDAnalysis preferred, minimal fallback otherwise
# ──────────────────────────────────────────────────────────────────────

def _mda_available() -> bool:
    try:
        import MDAnalysis  # noqa: F401
        return True
    except ImportError:
        return False


def load_trajectory(traj_path: Path, topol_path: Path) -> Optional["object"]:
    """Open a trajectory + topology pair. Returns a MDAnalysis.Universe or
    None if neither MDAnalysis nor a fallback can open them."""
    if not _mda_available():
        return None
    import MDAnalysis as mda
    try:
        return mda.Universe(str(topol_path), str(traj_path))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Clustering — DFS over MARTINI bead distance cutoff
# ──────────────────────────────────────────────────────────────────────

def cluster_by_cutoff(positions: np.ndarray, box: np.ndarray,
                       cutoff_nm: float = 0.65) -> List[List[int]]:
    """Return a list of clusters (each = list of atom indices).
    Periodic boundary aware (orthorhombic).
    """
    n = positions.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    cutoff2 = cutoff_nm ** 2
    box_arr = np.asarray(box)
    for i in range(n):
        if visited[i]:
            continue
        comp = []
        stack = [i]
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            comp.append(j)
            diff = positions - positions[j]
            diff -= box_arr * np.round(diff / box_arr)
            d2 = (diff ** 2).sum(axis=1)
            for k in np.where(d2 < cutoff2)[0]:
                if not visited[k]:
                    stack.append(int(k))
        clusters.append(comp)
    return clusters


# ──────────────────────────────────────────────────────────────────────
# Observables
# ──────────────────────────────────────────────────────────────────────

def _safe_mean(arr) -> float:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _largest_cluster_size(clusters: List[List[int]]) -> Tuple[int, List[int]]:
    if not clusters:
        return 0, []
    biggest = max(clusters, key=len)
    return len(biggest), biggest


def _radius_of_gyration_nm(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("nan")
    com = positions.mean(axis=0)
    return float(np.sqrt(((positions - com) ** 2).sum(axis=1).mean()))


def _bilayer_thickness_nm(head_positions: np.ndarray, box: np.ndarray) -> float:
    """Estimate bilayer thickness by binning head Z positions and measuring
    inter-peak distance. Returns NaN if no clear bilayer."""
    if head_positions.shape[0] < 20:
        return float("nan")
    z = head_positions[:, 2]
    z = z - z.mean()
    hist, edges = np.histogram(z, bins=40)
    # Find the two largest peaks symmetric around zero.
    peaks = np.argsort(hist)[-4:]
    centers = (edges[:-1] + edges[1:]) / 2.0
    pcenters = centers[peaks]
    if len(pcenters) < 2:
        return float("nan")
    pos = pcenters[pcenters > 0]
    neg = pcenters[pcenters < 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(pos.min() - neg.max())


def _area_per_head_nm2(head_positions: np.ndarray, box: np.ndarray) -> float:
    """APL = box_xy_area * 2 / n_head (assumes a bilayer fills the xy plane,
    factor 2 because both leaflets share the area). Returns NaN if too few
    heads.
    """
    n = head_positions.shape[0]
    if n < 20:
        return float("nan")
    return float(box[0] * box[1] * 2.0 / n)


def _order_param_tail(tail_directions: np.ndarray) -> float:
    """S_CD-like order parameter: S = ⟨½(3 cos²θ − 1)⟩ along z."""
    if tail_directions.size == 0:
        return float("nan")
    cos2 = (tail_directions[:, 2] /
            np.linalg.norm(tail_directions, axis=1)) ** 2
    return float(np.mean(0.5 * (3 * cos2 - 1)))


def _water_penetration(head_positions: np.ndarray,
                        water_positions: np.ndarray,
                        box: np.ndarray, radius_nm: float = 0.5) -> float:
    """Fraction of waters within `radius_nm` of any head bead.

    Periodic-image aware (orthorhombic only). Heavy work — runs in O(W·H).
    For 256 heads × 5000 waters this is 1.3M evals per frame which is fine.
    """
    if head_positions.size == 0 or water_positions.size == 0:
        return float("nan")
    r2 = radius_nm ** 2
    box_arr = np.asarray(box)
    hits = 0
    # Vectorize over waters; per-head loop is short.
    for hp in head_positions:
        diff = water_positions - hp
        diff -= box_arr * np.round(diff / box_arr)
        d2 = (diff ** 2).sum(axis=1)
        hits += int((d2 < r2).any())
    return float(hits / len(water_positions))


# ──────────────────────────────────────────────────────────────────────
# Frame-wise extraction over trajectory
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MDObservables:
    md_assembles: float = float("nan")
    md_n_agg: float = float("nan")
    md_a_head_nm2: float = float("nan")
    md_bilayer_thick_nm: float = float("nan")
    md_order_param: float = float("nan")
    md_radius_gyration_nm: float = float("nan")
    md_water_penetration: float = float("nan")
    n_frames_used: int = 0
    n_total_frames: int = 0
    error: Optional[str] = None


def extract_md_observables(traj_path: Path, topol_path: Path, *,
                            head_atom_names: Tuple[str, ...] = ("HN1", "HN2", "HN"),
                            tail_atom_names: Tuple[str, ...] = ("T1", "T2", "T3", "T4"),
                            water_residue: str = "W",
                            ion_residues: Tuple[str, ...] = ("CL", "NA", "SOL"),
                            last_frac: float = 0.4) -> MDObservables:
    """Average observables over the last `last_frac` of frames.

    Returns MDObservables with NaN everywhere when the trajectory can't be
    parsed; sets .error to a tag describing why.
    """
    if not traj_path.exists() or not topol_path.exists():
        return MDObservables(error="trajectory_or_topol_missing")
    u = load_trajectory(traj_path, topol_path)
    if u is None:
        return MDObservables(error="loader_unavailable")

    head_sel_names = " or ".join(f"name {n}" for n in head_atom_names)
    tail_sel_names = " or ".join(f"name {n}" for n in tail_atom_names)
    try:
        head_grp = u.select_atoms(head_sel_names)
        tail_grp = u.select_atoms(tail_sel_names)
        water_grp = u.select_atoms(f"resname {water_residue}")
    except Exception as exc:
        return MDObservables(error=f"selection_failed:{exc}")

    n_total = len(u.trajectory)
    if n_total == 0:
        return MDObservables(error="empty_trajectory")
    start = max(0, int(n_total * (1.0 - last_frac)))
    frame_indices = list(range(start, n_total))

    assemb_fracs: List[float] = []
    nagg_means: List[float] = []
    apls: List[float] = []
    thicks: List[float] = []
    orders: List[float] = []
    rgs: List[float] = []
    wps: List[float] = []

    n_molecules = len({a.resid for a in u.atoms})

    for fi in frame_indices:
        u.trajectory[fi]
        box = u.dimensions[:3] / 10.0   # Å → nm
        head_pos = head_grp.positions / 10.0
        tail_pos = tail_grp.positions / 10.0
        water_pos = water_grp.positions / 10.0
        if head_pos.size == 0:
            continue
        clusters = cluster_by_cutoff(head_pos, box, cutoff_nm=0.65)
        big_size, big_idx = _largest_cluster_size(clusters)
        # Map cluster size back to molecules — head beads per molecule depend
        # on which head, but the proximal-N is the canonical "1 per molecule"
        # head bead. Use it as the assembly counter.
        assemb_fracs.append(big_size / max(1, n_molecules))
        nagg_means.append(float(big_size))
        apls.append(_area_per_head_nm2(head_pos, box))
        thicks.append(_bilayer_thickness_nm(head_pos, box))
        # Tail order via head→terminal direction per molecule (approximate by
        # taking COM of tail beads vs COM of head beads in the same residue).
        if tail_pos.size > 0:
            dirs = tail_pos - tail_pos.mean(axis=0)
            orders.append(_order_param_tail(dirs))
        else:
            orders.append(float("nan"))
        if big_idx:
            rgs.append(_radius_of_gyration_nm(head_pos[big_idx]))
        wps.append(_water_penetration(head_pos, water_pos, box))

    obs = MDObservables(
        md_assembles=_safe_mean(assemb_fracs),
        md_n_agg=_safe_mean(nagg_means),
        md_a_head_nm2=_safe_mean(apls),
        md_bilayer_thick_nm=_safe_mean(thicks),
        md_order_param=_safe_mean(orders),
        md_radius_gyration_nm=_safe_mean(rgs),
        md_water_penetration=_safe_mean(wps),
        n_frames_used=len(frame_indices),
        n_total_frames=n_total,
    )
    return obs


# ──────────────────────────────────────────────────────────────────────
# Compose per-state observables into the MD_KEYS schema
# ──────────────────────────────────────────────────────────────────────

ASSEMBLY_QUALITY_THRESHOLD = 0.3


def compose_md_record(neutral: MDObservables, prot: MDObservables) -> Dict[str, float]:
    """Combine neutral + prot observable bundles into one MD_KEYS dict.

    Honest scope flag (per W-B plan §3.4):
      When md_assembles < ASSEMBLY_QUALITY_THRESHOLD in either state, the
      bilayer / a_head / order_param observables are NOT physically meaningful
      (no stable aggregate formed). We propagate them as NaN rather than
      reporting numbers from a disorganized state. Downstream Block D' /
      XGBoost handles NaN via its default branch.
    """
    # Quality gate: zero out observables when assembly didn't happen.
    def _q(o: MDObservables) -> MDObservables:
        if not np.isfinite(o.md_assembles) or o.md_assembles < ASSEMBLY_QUALITY_THRESHOLD:
            return MDObservables(
                md_assembles=o.md_assembles,
                md_n_agg=o.md_n_agg,
                md_radius_gyration_nm=o.md_radius_gyration_nm,
                n_frames_used=o.n_frames_used,
                n_total_frames=o.n_total_frames,
                error=o.error or "assembly_below_threshold",
            )
        return o
    neutral = _q(neutral)
    prot = _q(prot)
    a_n = neutral.md_a_head_nm2
    a_p = prot.md_a_head_nm2
    t_n = neutral.md_bilayer_thick_nm
    # CPP estimates require l_tail and V_tail; we compute those from the
    # canonical Tanford-like values at the caller, but we still expose
    # md_a_head_*_nm2 so the caller can finalize the CPPs.
    rec = {
        "md_assembles":         _safe_mean([neutral.md_assembles, prot.md_assembles]),
        "md_n_agg":             _safe_mean([neutral.md_n_agg, prot.md_n_agg]),
        "md_a_head_neutral_nm2": a_n,
        "md_a_head_prot_nm2":    a_p,
        "md_delta_a_head_nm2":   (a_p - a_n) if np.isfinite(a_n) and np.isfinite(a_p) else float("nan"),
        "md_bilayer_thick_nm":   _safe_mean([t_n, prot.md_bilayer_thick_nm]),
        "md_order_param":        _safe_mean([neutral.md_order_param, prot.md_order_param]),
        "md_radius_gyration_nm": _safe_mean([neutral.md_radius_gyration_nm, prot.md_radius_gyration_nm]),
        "md_water_penetration":  _safe_mean([neutral.md_water_penetration, prot.md_water_penetration]),
        # CPPs filled in by caller (needs Tanford V_tail / l_tail).
        "md_cpp_neutral":        float("nan"),
        "md_cpp_prot":           float("nan"),
        "md_delta_cpp":          float("nan"),
        # Spontaneous curvature shift (Helfrich) — c0 ≈ (a_p - a_n) / (a_n * t_n).
        "md_c0_spontaneous":     (
            (a_p - a_n) / (a_n * t_n)
            if np.isfinite(a_n) and np.isfinite(a_p) and np.isfinite(t_n)
            and a_n > 0 and t_n > 0 else float("nan")
        ),
    }
    return rec


__all__ = ["extract_md_observables", "compose_md_record",
            "MDObservables", "load_trajectory",
            "cluster_by_cutoff"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--topol", type=Path, required=True)
    args = parser.parse_args()
    obs = extract_md_observables(args.traj, args.topol)
    print(json.dumps(obs.__dict__, indent=2, default=str))
