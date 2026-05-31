"""
physics_design/lipid_library.py — extract a single Martini-3 lipid moleculetype
(beads, bonds, angles) from the downloaded lipidome collection .itp files, and
emit a standalone per-lipid .itp for GROMACS.

The collection files (martini/lipidome/*.itp) bundle many [moleculetype] blocks.
We parse one by name and expose:
  Lipid.beads    : list of (name, type, charge)        index 0-based
  Lipid.bonds    : list of (i, j, r0_nm, k)            harmonic, func 1
  Lipid.angles   : list of (i, j, k, theta0_deg, k)    cosine, func 2
  Lipid.net_charge

Martini-3 phospholipids (DOPC/DOPE/POPC) use only bonds + angles (no constraints,
no dihedrals), verified against the source itp — so a faithful force recompute
needs only LJ + reaction-field + harmonic bonds + cosine angles.

If a requested lipid has a [constraints] or [dihedrals] block we record it and
refuse to silently ignore it (the pressure-profile force model would be
incomplete -> that would be a hidden proxy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LIPIDOME_DIR = Path(__file__).resolve().parent.parent / "martini" / "lipidome"

# Where to find each lipid (filename within the lipidome dir). The PC collection
# also contains many PE/PG/etc. entries; we list explicit files we have vetted.
_DEFAULT_FILES = (
    "martini_v3.0_phospholipids.itp",
    "martini_v3.0.0_phospholipids_PE_v2.itp",
)


def _strip(line: str) -> str:
    semi = line.find(";")
    if semi >= 0:
        line = line[:semi]
    return line.strip()


@dataclass
class Lipid:
    name: str
    beads: List[Tuple[str, str, float]] = field(default_factory=list)   # (atomname, btype, charge)
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)
    angles: List[Tuple[int, int, int, float, float]] = field(default_factory=list)
    has_constraints: bool = False
    has_dihedrals: bool = False
    source: str = ""

    @property
    def n_beads(self) -> int:
        return len(self.beads)

    @property
    def net_charge(self) -> float:
        return sum(c for _, _, c in self.beads)

    @property
    def bead_types(self) -> List[str]:
        return [t for _, t, _ in self.beads]

    @property
    def bead_names(self) -> List[str]:
        return [n for n, _, _ in self.beads]


def _iter_blocks(path: Path):
    """Yield (molname, block_lines) for each [moleculetype] in the file."""
    cur_name = None
    cur_lines: List[str] = []
    pending_moltype = False
    with path.open() as fh:
        for raw in fh:
            s = _strip(raw)
            low = s.lower()
            if low.startswith("[") and "moleculetype" in low:
                if cur_name is not None:
                    yield cur_name, cur_lines
                cur_name = None
                cur_lines = []
                pending_moltype = True
                continue
            if pending_moltype:
                if s:
                    cur_name = s.split()[0]
                    pending_moltype = False
                continue
            if cur_name is not None:
                cur_lines.append(raw)
        if cur_name is not None:
            yield cur_name, cur_lines


def load_lipid(name: str,
               files: Optional[Tuple[str, ...]] = None,
               lipidome_dir: Optional[Path] = None) -> Lipid:
    """Find lipid `name` across the lipidome files and parse its block."""
    ld = Path(lipidome_dir) if lipidome_dir else _LIPIDOME_DIR
    for fname in (files or _DEFAULT_FILES):
        fpath = ld / fname
        if not fpath.exists():
            continue
        for mol, lines in _iter_blocks(fpath):
            if mol == name:
                return _parse_block(name, lines, source=str(fpath))
    raise KeyError(f"lipid '{name}' not found in {files or _DEFAULT_FILES}")


def _parse_block(name: str, lines: List[str], source: str) -> Lipid:
    lip = Lipid(name=name, source=source)
    section = None
    for raw in lines:
        s = _strip(raw)
        if not s:
            continue
        if s.startswith("["):
            section = s.strip("[] ").lower()
            if section == "constraints":
                lip.has_constraints = True
            if section.startswith("dihedral"):
                lip.has_dihedrals = True
            continue
        # ignore preprocessor / position-restraint blocks
        if s.startswith("#"):
            section = "_ignore"
            continue
        toks = s.split()
        if section == "atoms":
            # id type resnr residue atom cgnr charge [mass]
            if len(toks) >= 7:
                btype = toks[1]
                aname = toks[4]
                try:
                    charge = float(toks[6])
                except ValueError:
                    charge = 0.0
                lip.beads.append((aname, btype, charge))
        elif section == "bonds":
            if len(toks) >= 5 and toks[2] == "1":
                i, j = int(toks[0]) - 1, int(toks[1]) - 1
                r0, k = float(toks[3]), float(toks[4])
                lip.bonds.append((i, j, r0, k))
        elif section == "angles":
            if len(toks) >= 6 and toks[3] in ("1", "2", "10"):
                i, j, k = int(toks[0]) - 1, int(toks[1]) - 1, int(toks[2]) - 1
                theta0, kk = float(toks[4]), float(toks[5])
                lip.angles.append((i, j, k, theta0, kk))
    return lip


def write_single_itp(lip: Lipid, path: Path) -> None:
    """Emit a standalone .itp with just this lipid's moleculetype (for grompp)."""
    L = [f"; {lip.name} extracted from {Path(lip.source).name}",
         "[moleculetype]", "; name nrexcl", f"{lip.name} 1", "", "[atoms]",
         ";  id type resnr residue atom cgnr charge"]
    for idx, (aname, btype, charge) in enumerate(lip.beads, start=1):
        L.append(f"{idx:5d} {btype:6s} 1 {lip.name:6s} {aname:5s} {idx:5d} {charge:7.3f}")
    L += ["", "[bonds]", ";  i j func r0 k"]
    for (i, j, r0, k) in lip.bonds:
        L.append(f"{i+1:4d} {j+1:4d} 1 {r0:.4f} {k:.1f}")
    if lip.angles:
        L += ["", "[angles]", ";  i j k func theta0 k"]
        for (i, j, k, th, kk) in lip.angles:
            L.append(f"{i+1:4d} {j+1:4d} {k+1:4d} 2 {th:.2f} {kk:.1f}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    for nm in ("DOPC", "DOPE", "POPC"):
        lip = load_lipid(nm)
        print(f"{nm}: {lip.n_beads} beads  net_charge={lip.net_charge:+.2f}  "
              f"bonds={len(lip.bonds)} angles={len(lip.angles)}  "
              f"constraints={lip.has_constraints} dihedrals={lip.has_dihedrals}")
        print("   types:", " ".join(lip.bead_types))
        print("   names:", " ".join(lip.bead_names))
