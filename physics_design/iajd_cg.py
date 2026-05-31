"""
physics_design/iajd_cg.py — adapt the existing IAJD CG model (martini/build_cg.py, from
the iajd_grammar decomposition) into the `Lipid` abstraction the physics_design modules
consume, so Module B (c0) / Module C (H_II) can run on real IAJDs (not just lipids).

What this DOES (validated): build the per-IAJD Martini-3 AssembledTopology (beads, bonds,
angles, constraints, charge) and expose it as a `Lipid`, with constraints converted to
stiff harmonic bonds so the in-Python Irving-Kirkwood pressure profile (pressure_profile.py,
which has no constraint-virial term) includes their stress. All IAJD bead types are present
in martini_v3.0.0.itp (verified), so the force recompute is well-defined.

What this does NOT yet cover (HONEST — these gate a TRUSTED IAJD c0/H_II; see
docs/PHYSICS_DESIGN_FUTURE_WORK.md FW-2 and the build prompt's Module E):
  1. **Module E AA-reference validation of the CG mapping is NOT done.** build_cg.py is a
     hand-mapped, UNVALIDATED Janus-dendrimer model (no published one exists). Per the build
     prompt, A/B/C values on IAJDs are PROVISIONAL until the CG model reproduces an
     atomistic reference (Rg ±15%, partitioning, head solvation). Do not present IAJD
     numbers as trusted before that gate.
  2. **Head-up bilayer orientation:** build_cg orders beads core-first, not head-first, so a
     correct IAJD bilayer needs the ionizable head placed at the interface (the lipid
     bilayer tiler assumes bead 0 = head). A flag marks the head bead for orientation.
  3. **Counterions:** a protonated (+1) IAJD bilayer needs neutralizing Cl- (gmx genion);
     the neutral state (charge 0) is the tractable first c0.

Use the NEUTRAL state for a first provisional c0 (no counterions); protonated + counterions
+ Module E are the path to a trusted value.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .lipid_library import Lipid

ROOT = Path(__file__).resolve().parent.parent
BIOACT = ROOT / "IAJD_master" / "datasets" / "IAJD_Bioact_v13_clean.xlsx"

STIFF_BOND_K = 50000.0    # kJ/mol/nm^2 — stiff harmonic ~ a Martini constraint (needs dt<=10fs)


@lru_cache(maxsize=1)
def _bioact() -> pd.DataFrame:
    return pd.read_excel(BIOACT)


def iajd_to_lipid(iajd_num: int, *, protonated: bool = False,
                  stiffen_constraints: bool = True) -> Lipid:
    """Build IAJD `iajd_num` as a `Lipid` (constraints -> stiff bonds). NEUTRAL by default
    (no counterions needed). The returned Lipid records `head_bead` (the ionizable bead,
    for interface orientation) and `has_constraints`/charge for downstream handling."""
    from iajd_grammar import decompose_row
    from martini.build_cg import assemble
    df = _bioact()
    rows = df[df["IAJD_num"] == iajd_num]
    if rows.empty:
        raise KeyError(f"IAJD_num {iajd_num} not in dataset")
    seed = decompose_row(rows.iloc[0].to_dict())
    if seed is None:
        raise ValueError(f"decompose_row failed for IAJD {iajd_num}")
    topo = assemble(seed, protonated=protonated)

    beads = [(name, btype, charge) for (name, btype, _mass, charge) in topo.beads]
    bonds = [(i, j, r0, k) for (i, j, r0, k) in topo.bonds]
    n_constraints = len(topo.constraints)
    if stiffen_constraints:
        for (i, j, length) in topo.constraints:
            bonds.append((i, j, length, STIFF_BOND_K))
    angles = [(i, j, k, th0, kk) for (i, j, k, th0, kk) in topo.angles]

    # identify the ionizable head bead (highest |charge| in the protonated assembly) for
    # interface orientation; in the neutral assembly fall back to the same index.
    topo_p = assemble(seed, protonated=True)
    head_idx = int(max(range(len(topo_p.beads)),
                       key=lambda i: abs(topo_p.beads[i][3])))

    lip = Lipid(name=f"IAJD{iajd_num}", beads=beads, bonds=bonds, angles=angles,
                has_constraints=False,   # converted to stiff bonds
                source=f"build_cg:IAJD{iajd_num}:{'prot' if protonated else 'neutral'}")
    lip.head_bead = head_idx               # type: ignore[attr-defined]
    lip.net_charge_actual = topo.total_charge  # type: ignore[attr-defined]
    lip.n_constraints_stiffened = n_constraints  # type: ignore[attr-defined]
    lip.cg_mapping_validated = False       # type: ignore[attr-defined]  Module E NOT done
    return lip


if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 369
    for prot in (False, True):
        lip = iajd_to_lipid(num, protonated=prot)
        print(f"IAJD{num} {'prot' if prot else 'neutral'}: {lip.n_beads} beads, "
              f"{len(lip.bonds)} bonds (incl {lip.n_constraints_stiffened} stiffened), "
              f"{len(lip.angles)} angles, head_bead={lip.head_bead} "
              f"({lip.beads[lip.head_bead][0]}/{lip.beads[lip.head_bead][1]}), "
              f"net_charge={lip.net_charge_actual:+.2f}, cg_validated={lip.cg_mapping_validated}")
