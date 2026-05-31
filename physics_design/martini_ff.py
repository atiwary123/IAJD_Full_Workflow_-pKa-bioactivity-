"""
physics_design/martini_ff.py — parse the Martini 3 force-field nonbonded table.

Reads martini/ff/martini_v3.0.0.itp:
  [ defaults ]      -> nbfunc, comb-rule (expects "1 2", sigma-epsilon LJ)
  [ atomtypes ]     -> bead name -> mass (g/mol)
  [ nonbond_params ]-> (type_i, type_j) -> (sigma_nm, epsilon_kJmol)

Martini 3 supplies an *explicit* LJ pair for every bead-type pair (355k lines),
so we never fall back to a combination rule. A missing needed pair is a hard
error (no-proxy: real value or audited failure, never an invented default).

The LJ form is V(r) = 4 eps [ (sig/r)^12 - (sig/r)^6 ] = C12/r^12 - C6/r^6 with
  C6  = 4 eps sig^6 ,  C12 = 4 eps sig^12 .
GROMACS' Verlet + Potential-shift modifier shifts only the *energy* to be zero at
the cutoff; the *force* is the bare analytic LJ within rc and exactly zero beyond
rc. We therefore recompute the bare analytic force with a hard cutoff at rvdw —
which is exactly the force GROMACS integrated to produce the trajectory.

This module performs no MD; it is the parameter source for pressure_profile.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

_DEF_FF = Path(__file__).resolve().parent.parent / "martini" / "ff" / "martini_v3.0.0.itp"


def _strip(line: str) -> str:
    """Remove inline ';' comments and surrounding whitespace."""
    semi = line.find(";")
    if semi >= 0:
        line = line[:semi]
    return line.strip()


@dataclass
class MartiniFF:
    """Parsed Martini-3 nonbonded parameters."""
    masses: Dict[str, float]                       # bead type -> mass (g/mol)
    # pair -> (sigma_nm, epsilon_kJmol); key is the *sorted* (type_i, type_j) tuple
    pair_sig_eps: Dict[Tuple[str, str], Tuple[float, float]]
    comb_rule: int
    nbfunc: int
    source: str

    def sig_eps(self, a: str, b: str) -> Tuple[float, float]:
        """Return (sigma_nm, epsilon_kJmol) for bead types a,b. Raises KeyError
        if the explicit pair is absent (no silent combination-rule fallback)."""
        key = (a, b) if a <= b else (b, a)
        return self.pair_sig_eps[key]

    def c6_c12(self, a: str, b: str) -> Tuple[float, float]:
        sig, eps = self.sig_eps(a, b)
        s6 = sig ** 6
        c6 = 4.0 * eps * s6
        c12 = 4.0 * eps * s6 * s6
        return c6, c12

    def has_pair(self, a: str, b: str) -> bool:
        key = (a, b) if a <= b else (b, a)
        return key in self.pair_sig_eps

    def c6_c12_matrix(self, types: Iterable[str]) -> Tuple[list, np.ndarray, np.ndarray]:
        """Build dense C6/C12 lookup matrices over an ordered list of unique
        bead types. Returns (ordered_types, C6[NxN], C12[NxN]). Raises KeyError
        on any missing pair so callers fail loudly rather than guessing."""
        uniq = sorted(set(types))
        n = len(uniq)
        c6 = np.zeros((n, n), dtype=float)
        c12 = np.zeros((n, n), dtype=float)
        for i, a in enumerate(uniq):
            for j in range(i, n):
                b = uniq[j]
                cc6, cc12 = self.c6_c12(a, b)
                c6[i, j] = c6[j, i] = cc6
                c12[i, j] = c12[j, i] = cc12
        return uniq, c6, c12


def parse_ff(path: Optional[Path] = None,
             needed_types: Optional[Iterable[str]] = None) -> MartiniFF:
    """Parse the Martini-3 itp. If `needed_types` is given, only nonbond_params
    lines whose BOTH types are in the set are retained (much smaller dict)."""
    path = Path(path) if path else _DEF_FF
    if not path.exists():
        raise FileNotFoundError(f"Martini FF not found: {path}")
    need = set(needed_types) if needed_types is not None else None

    masses: Dict[str, float] = {}
    pairs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    nbfunc, comb = 1, 2
    section = None
    with path.open() as fh:
        for raw in fh:
            s = _strip(raw)
            if not s:
                continue
            if s.startswith("["):
                section = s.strip("[] ").lower()
                continue
            if section == "defaults":
                toks = s.split()
                if len(toks) >= 2:
                    nbfunc, comb = int(toks[0]), int(toks[1])
            elif section == "atomtypes":
                toks = s.split()
                # name mass charge ptype V W   (Martini: mass in col 2)
                if len(toks) >= 2:
                    try:
                        masses[toks[0]] = float(toks[1])
                    except ValueError:
                        pass
            elif section == "nonbond_params":
                toks = s.split()
                if len(toks) >= 5:
                    a, b = toks[0], toks[1]
                    if need is not None and (a not in need or b not in need):
                        continue
                    try:
                        sig = float(toks[3])
                        eps = float(toks[4])
                    except ValueError:
                        continue
                    key = (a, b) if a <= b else (b, a)
                    pairs[key] = (sig, eps)
    return MartiniFF(masses=masses, pair_sig_eps=pairs,
                     comb_rule=comb, nbfunc=nbfunc, source=str(path))


@lru_cache(maxsize=2)
def load_ff_cached(source: str = "") -> MartiniFF:
    """Cached full-FF load (use when many lipids share the FF)."""
    return parse_ff(Path(source) if source else None)


if __name__ == "__main__":
    ff = parse_ff()
    print(f"parsed {ff.source}")
    print(f"  nbfunc={ff.nbfunc} comb_rule={ff.comb_rule}")
    print(f"  {len(ff.masses)} atomtypes, {len(ff.pair_sig_eps)} explicit pairs")
    for a, b in [("C1", "C1"), ("W", "W"), ("Q1", "W"), ("Q4p", "W"),
                 ("Q5", "W"), ("SN4a", "C1"), ("C4h", "C4h"), ("Q1", "Q5")]:
        try:
            sig, eps = ff.sig_eps(a, b)
            c6, c12 = ff.c6_c12(a, b)
            print(f"  {a:5s}-{b:5s}: sigma={sig:.4f} nm  eps={eps:.4f} kJ/mol"
                  f"  (C6={c6:.4e} C12={c12:.4e})")
        except KeyError:
            print(f"  {a:5s}-{b:5s}: MISSING")
    print(f"  mass(C1)={ff.masses.get('C1')}  mass(SN4a)={ff.masses.get('SN4a')}"
          f"  mass(TC5)={ff.masses.get('TC5')}  mass(W)={ff.masses.get('W')}")
