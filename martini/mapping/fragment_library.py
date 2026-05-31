"""
fragment_library.py — MARTINI 3 bead vocabulary for IAJD fragments.

Bead types follow Souza et al. 2021 (MARTINI 3). Each fragment exposes:
  - beads:    [(bead_name, bead_type, mass, charge), ...]
  - bonds:    [(i, j, length_nm, k_kJ_mol_nm2), ...]
  - angles:   [(i, j, k, theta_deg, k_kJ_mol_rad2), ...]
  - dihedrals/constraints/exclusions as needed

These are deliberately conservative MARTINI 3 numbers in line with the
standard parameter tables — they're a starting point and would be re-tuned
during W-B's self-assembly convergence checks. The vocabulary is bounded by
iajd_grammar, so each fragment is parametrized once and reused.

NOTE on protonation: the distal piperazine N has two parametrizations
(_neutral vs _prot) selected via build_cg.py. The protonated form swaps
N5a → Q1 (charge +1) and adds a Q1 counter-ion (Cl−) at the system level,
not the molecule level.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Fragment:
    name: str
    beads: List[Tuple[str, str, float, float]] = field(default_factory=list)
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)
    angles: List[Tuple[int, int, int, float, float]] = field(default_factory=list)
    dihedrals: List[Tuple[int, int, int, int, float, float, int]] = field(default_factory=list)
    constraints: List[Tuple[int, int, float]] = field(default_factory=list)
    # Sites: which bead index acts as the linker to the next fragment.
    linker_in: int = 0
    linker_out: int = -1   # default: last bead
    # Notes on which IAJD positions this fragment is allowed to bind to.
    annotation: str = ""


# ──────────────────────────────────────────────────────────────────────
# Alkyl tails — 4-to-1 atomistic-to-CG mapping (MARTINI 3 standard).
# ──────────────────────────────────────────────────────────────────────

def alkyl_tail(n_carbons: int, *, branched: bool = False) -> Fragment:
    """Linear or branched (2-ethylhexyl-like) alkyl tail.

    Length encoded in bead count: 1 bead per ~4 carbons (rounded up).
    Branched tails (n_carbons==8 only, used as 2-ethylhexyl) get an extra
    branch bead at position 1 (SC1).
    """
    n_beads = max(1, (n_carbons + 3) // 4)
    f = Fragment(name=f"tail_C{n_carbons}{'_br' if branched else ''}")
    for i in range(n_beads):
        name = f"T{i+1}"
        bead_type = "TC1" if i == n_beads - 1 else "C1"
        f.beads.append((name, bead_type, 72.0, 0.0))
    for i in range(n_beads - 1):
        f.bonds.append((i, i + 1, 0.470, 5000.0))
    for i in range(n_beads - 2):
        f.angles.append((i, i + 1, i + 2, 180.0, 35.0))
    if branched and n_beads >= 2:
        # Extra branch bead off bead index 1 (SC1, small mass).
        f.beads.append(("B1", "SC1", 54.0, 0.0))
        branch_idx = len(f.beads) - 1
        f.bonds.append((1, branch_idx, 0.430, 5000.0))
        if n_beads >= 3:
            f.angles.append((0, 1, branch_idx, 109.5, 35.0))
    f.linker_in = 0
    f.linker_out = n_beads - 1
    return f


# ──────────────────────────────────────────────────────────────────────
# Linkers
# ──────────────────────────────────────────────────────────────────────

def ester_linker() -> Fragment:
    """`-O-C(=O)-` ester (3 atoms → 1 CG bead). MARTINI N4a (mild H-bond acceptor)."""
    f = Fragment(name="link_ester")
    f.beads.append(("L1", "N4a", 60.0, 0.0))
    return f


def amide_linker() -> Fragment:
    """`-N-C(=O)-` amide (3 atoms → 1 CG bead). MARTINI P1 (strong H-bond donor)."""
    f = Fragment(name="link_amide")
    f.beads.append(("L1", "P1", 60.0, 0.0))
    return f


# ──────────────────────────────────────────────────────────────────────
# Aromatic cores
# ──────────────────────────────────────────────────────────────────────

def benzyl_core_3trisubst() -> Fragment:
    """3,4,5-trisubstituted benzyl (GA-Tris / PE-Tris / PE-Gallic core).

    3 TC5 ring beads + 1 SC1 benzyl CH2. Three ether O beads (N4a) attach at
    ring positions 3/4/5 as separate fragments; this one only contains the
    aromatic ring + CH2 linker to the rest of the molecule.
    """
    f = Fragment(name="ar_3trisubst")
    f.beads.append(("R1", "TC5", 45.0, 0.0))
    f.beads.append(("R2", "TC5", 45.0, 0.0))
    f.beads.append(("R3", "TC5", 45.0, 0.0))
    f.beads.append(("B1", "SC1", 54.0, 0.0))
    # Equilateral ring constraints (~0.27 nm bead-to-bead is standard MARTINI 3).
    f.constraints += [(0, 1, 0.270), (1, 2, 0.270), (0, 2, 0.270)]
    f.bonds.append((0, 3, 0.270, 7500.0))   # ring → benzyl CH2
    f.linker_in = 3   # benzyl CH2 connects out
    f.linker_out = 3
    return f


def benzyl_core_35disubst() -> Fragment:
    """3,5-disubstituted benzyl (sSS-Nonsym / Dialkoxybenzyl core)."""
    f = Fragment(name="ar_35disubst")
    f.beads.append(("R1", "TC5", 45.0, 0.0))
    f.beads.append(("R2", "TC5", 45.0, 0.0))
    f.beads.append(("R3", "TC5", 45.0, 0.0))
    f.beads.append(("B1", "SC1", 54.0, 0.0))
    f.constraints += [(0, 1, 0.270), (1, 2, 0.270), (0, 2, 0.270)]
    f.bonds.append((0, 3, 0.270, 7500.0))
    f.linker_in = 3
    f.linker_out = 3
    return f


# ──────────────────────────────────────────────────────────────────────
# Head groups (two parametrizations each: neutral + protonated)
# ──────────────────────────────────────────────────────────────────────

def head_DMA(protonated: bool = False) -> Fragment:
    """Dimethylamine head: 1 bead (N6d neutral, Q1 protonated) + 2 TC1 methyls."""
    f = Fragment(name=f"head_DMA_{'prot' if protonated else 'neutral'}")
    f.beads.append(("HN", "Q1" if protonated else "N6d", 50.0,
                     1.0 if protonated else 0.0))
    f.beads.append(("HM1", "TC1", 15.0, 0.0))
    f.beads.append(("HM2", "TC1", 15.0, 0.0))
    f.bonds.append((0, 1, 0.260, 7500.0))
    f.bonds.append((0, 2, 0.260, 7500.0))
    f.angles.append((1, 0, 2, 109.5, 50.0))
    f.linker_in = 0
    f.linker_out = 0
    return f


def head_piperidine(protonated: bool = False) -> Fragment:
    """Piperidine (PIP) head: 1 N-bead + 2 ring-C beads."""
    f = Fragment(name=f"head_PIP_{'prot' if protonated else 'neutral'}")
    f.beads.append(("HN", "Q1" if protonated else "N6d", 50.0,
                     1.0 if protonated else 0.0))
    f.beads.append(("HC1", "SC2", 50.0, 0.0))
    f.beads.append(("HC2", "SC2", 50.0, 0.0))
    f.bonds.append((0, 1, 0.300, 7500.0))
    f.bonds.append((1, 2, 0.300, 7500.0))
    f.bonds.append((0, 2, 0.420, 5000.0))   # ring-closure
    f.linker_in = 0
    f.linker_out = 0
    return f


def head_piperazine_core(protonated: bool = False) -> Fragment:
    """Piperazine 6-ring with proximal N (HN1) and distal N (HN2).

    Proximal N attaches to the linker (linker_in); distal N is where the
    methyl/hydroxyethyl/H2EPRZ substituents go via subsequent fragments.
    Protonation only changes the DISTAL N — the proximal N is already
    substituted by the linker carbon and stays neutral.
    """
    f = Fragment(name=f"head_PZ_core_{'prot' if protonated else 'neutral'}")
    f.beads.append(("HN1", "N6d", 50.0, 0.0))      # proximal N — linker
    f.beads.append(("HC1", "SC2", 50.0, 0.0))
    f.beads.append(("HC2", "SC2", 50.0, 0.0))
    f.beads.append(("HN2", "Q1" if protonated else "N5a", 50.0,
                     1.0 if protonated else 0.0))   # distal N — protonated here
    f.bonds.append((0, 1, 0.300, 7500.0))
    f.bonds.append((1, 3, 0.300, 7500.0))
    f.bonds.append((3, 2, 0.300, 7500.0))
    f.bonds.append((2, 0, 0.300, 7500.0))
    # Cross-ring distance.
    f.constraints += [(0, 3, 0.420)]
    f.linker_in = 0
    f.linker_out = 3   # distal N continues to substituent
    return f


def head_MPRZ(protonated: bool = False) -> Fragment:
    """4-methylpiperazine: piperazine_core + TC1 methyl on distal N."""
    f = head_piperazine_core(protonated)
    f.name = f"head_MPRZ_{'prot' if protonated else 'neutral'}"
    f.beads.append(("HMe", "TC1", 15.0, 0.0))
    f.bonds.append((3, len(f.beads) - 1, 0.260, 7500.0))
    return f


def head_HPRZ(protonated: bool = False) -> Fragment:
    """4-(2-hydroxyethyl)piperazine: piperazine_core + 2 SP1 beads (-CH2CH2OH)."""
    f = head_piperazine_core(protonated)
    f.name = f"head_HPRZ_{'prot' if protonated else 'neutral'}"
    f.beads.append(("HE1", "SP1", 45.0, 0.0))   # -CH2-
    f.beads.append(("HE2", "SP1", 45.0, 0.0))   # -CH2OH (terminal OH bead)
    base = len(f.beads) - 2
    f.bonds.append((3, base, 0.300, 7000.0))
    f.bonds.append((base, base + 1, 0.300, 7000.0))
    return f


def head_H2EPRZ(protonated: bool = False) -> Fragment:
    """4-(2-(2-hydroxyethoxy)ethyl)piperazine: piperazine_core + 4-bead extension."""
    f = head_piperazine_core(protonated)
    f.name = f"head_H2EPRZ_{'prot' if protonated else 'neutral'}"
    f.beads.append(("HE1", "SP1", 45.0, 0.0))
    f.beads.append(("HE2", "N4a", 60.0, 0.0))   # ether O bead
    f.beads.append(("HE3", "SP1", 45.0, 0.0))
    f.beads.append(("HE4", "SP1", 45.0, 0.0))   # terminal OH
    base = len(f.beads) - 4
    f.bonds.append((3, base, 0.300, 7000.0))
    f.bonds.append((base, base + 1, 0.300, 7000.0))
    f.bonds.append((base + 1, base + 2, 0.300, 7000.0))
    f.bonds.append((base + 2, base + 3, 0.300, 7000.0))
    return f


def head_DMBA(protonated: bool = False) -> Fragment:
    """4-(N,N-dimethylamino)benzyl: TC5×3 ring + N6d/Q1 + 2 TC1 methyls."""
    f = Fragment(name=f"head_DMBA_{'prot' if protonated else 'neutral'}")
    f.beads.append(("B1", "SC1", 54.0, 0.0))    # benzyl CH2 (linker)
    f.beads.append(("R1", "TC5", 45.0, 0.0))
    f.beads.append(("R2", "TC5", 45.0, 0.0))
    f.beads.append(("R3", "TC5", 45.0, 0.0))
    f.beads.append(("HN", "Q1" if protonated else "N6d", 50.0,
                     1.0 if protonated else 0.0))
    f.beads.append(("HM1", "TC1", 15.0, 0.0))
    f.beads.append(("HM2", "TC1", 15.0, 0.0))
    f.constraints += [(1, 2, 0.270), (2, 3, 0.270), (1, 3, 0.270)]
    f.bonds.append((0, 1, 0.270, 7500.0))
    f.bonds.append((3, 4, 0.270, 7500.0))   # ring para-position → N
    f.bonds.append((4, 5, 0.260, 7500.0))
    f.bonds.append((4, 6, 0.260, 7500.0))
    f.linker_in = 0
    f.linker_out = 0
    return f


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

HEAD_FACTORIES = {
    "DMA":    head_DMA,
    "PIP":    head_piperidine,
    "MPRZ":   head_MPRZ,
    "HPRZ":   head_HPRZ,
    "H2EPRZ": head_H2EPRZ,
    "DMBA":   head_DMBA,
}

CORE_FACTORIES = {
    "GA-Tris":        benzyl_core_3trisubst,
    "PE-Tris":        benzyl_core_3trisubst,
    "PE-Gallic":      benzyl_core_3trisubst,
    "sSS-Nonsym":     benzyl_core_35disubst,
    "Dialkoxybenzyl": benzyl_core_35disubst,
}

LINKER_FACTORIES = {
    "ester": ester_linker,
    "amide": amide_linker,
}


def make_head(name: str, protonated: bool = False) -> Fragment:
    if name not in HEAD_FACTORIES:
        raise KeyError(f"unknown head: {name}")
    return HEAD_FACTORIES[name](protonated)


def make_core(family: str) -> Fragment:
    if family not in CORE_FACTORIES:
        raise KeyError(f"unknown family core: {family}")
    return CORE_FACTORIES[family]()


def make_linker(name: str) -> Fragment:
    if name not in LINKER_FACTORIES:
        raise KeyError(f"unknown linker: {name}")
    return LINKER_FACTORIES[name]()


def make_tail(n_carbons: int, branched: bool = False) -> Fragment:
    return alkyl_tail(n_carbons, branched=branched)


__all__ = [
    "Fragment", "make_head", "make_core", "make_linker", "make_tail",
    "HEAD_FACTORIES", "CORE_FACTORIES", "LINKER_FACTORIES",
]
