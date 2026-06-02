"""physics_design/build_titratable_ff.py — generate a COMPLETE, self-contained titratable
Martini 3 force field that covers the FULL IAJD bead palette, so any IAJD can run a
constant-pH apparent-pKa titration (Module A) without silent missing-interaction blow-ups.

WHY THIS EXISTS
---------------
The vendored Grunewald titratable FF (martini/titratable/force_fields/martini.itp) declares
only a REDUCED atomtype set (water + a few lipids + the titratable/dummy/charged beads) —
enough for the MC3 / aniline validation, but it is MISSING the bead types IAJDs use
(N5a, N6d, SC1, SP1, TC1, P1) AND their entire nonbonded interaction matrix. That FF uses
comb-rule 2 with ZERO self sigma/epsilon, so any pair absent from [nonbond_params] is a
*silent* zero interaction (no grompp error) -> the IAJD body has no cohesion/repulsion ->
the run blows up (LINCS failures / exploding box). This is the #1 runtime-error risk.

THE FIX (no proxy — real Martini 3 numbers)
-------------------------------------------
Merge, restricted to the bead types that actually occur in IAJD-titratable systems (small,
fast to grompp, human-inspectable):
  (a) FULL standard Martini 3 atomtypes + nonbond_params (martini/ff/martini_v3.0.0.itp) —
      provides every standard x standard pair (N6d x C1, N5a x SN4a, ... all of it).
  (b) the titratable extensions (dummy / titratable-base / titratable-acid / charged beads
      + pH-dependent proton interactions) — provides titratable x {reduced standard set},
      titratable x titratable, dummy x titratable, proton/water exchange.
  (c) SYNTHESIZED titratable-bead x IAJD-body-bead cross terms for the pairs the reduced
      distribution omitted, taken from the titratable bead's standard PARENT:
        N2_10.2 -> N2 , SN6d_* -> SN6d , N5d_* -> N5d , P2_* -> P2 , WNT -> W
      This is VALIDATED exact: the distribution's own neutral_beads.itp has
      N2_10.2 x C1 == standard N2 x C1 (0.470/2.960), x N4a == N2 x N4a, x SN4a == N2 x SN4a.
      So "parent" is the real bead the titratable type was calibrated onto; copying its row
      is the Martini-consistent completion, not an approximation we invented.

Dummy / proton beads keep the Grunewald design: they LJ-interact only with the titratable /
dummy / ion set (sparse) and are zero-LJ vs bulk standard beads, acting on the system via
their PME charge only. That is exactly how the validated MC3-in-POPC run behaved, so we do
NOT synthesize dummy x standard terms (doing so would change the validated physics).

Output: an .itp that INLINES standard x standard + synthesized cross terms and #INCLUDES the
Grunewald titratable files (absolute paths). pH_dep_interactions.itp is #included, NOT
flattened, so its #ifdef pH<value> conditionality survives — flattening it (an earlier bug
caught by a non-physical titration in local testing) silently kills the pH response. Every
synthesized line is logged for transparency (honesty: no silent fills).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
STD_FF = ROOT / "martini" / "ff" / "martini_v3.0.0.itp"
TITR_FF = ROOT / "martini" / "titratable" / "force_fields"
TITR_MARTINI = TITR_FF / "martini.itp"
# pH-INDEPENDENT titratable extensions (safe to treat as static coverage).
TITR_STATIC_INCLUDES = ["dummy_beads.itp", "neutral_beads.itp", "charged_beads.itp",
                        "titratable_beads_self_proton.itp"]
# pH-DEPENDENT block: MUST stay a #include — its #ifdef pH<value> conditionality (21 pH
# blocks selecting the water<->proton<->titratable-bead strengths) is the entire pH response.
PH_DEP_INCLUDE = "pH_dep_interactions.itp"
OUT = TITR_FF / "martini_titratable_full.itp"

# titratable bead -> the standard Martini bead it was calibrated onto (its "parent").
# Anything matching <PARENT>_<pKa>[F|R] takes PARENT; WNT is titratable water -> standard W.
_SPECIAL_PARENT = {"WNT": "W"}
# beads that are intentionally sparse (Grunewald ghost dummies + the proton). Never
# synthesize their cross terms — they are zero-LJ vs bulk by design (PME charge only).
_DUMMY_LIKE = {"DW", "DA1", "DA2", "DB1", "DB2", "DB3", "DB4", "POS"}


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def parent_of(bead: str) -> Optional[str]:
    """Standard parent bead of a titratable bead (None if it has no _pKa form)."""
    if bead in _SPECIAL_PARENT:
        return _SPECIAL_PARENT[bead]
    m = re.match(r"^([A-Za-z0-9]+?)_\d", bead)   # N2_10.2 -> N2, SN6d_4.8 -> SN6d
    return m.group(1) if m else None


def _read_atomtypes(text: str) -> Dict[str, str]:
    """name -> normalized atomtype line (everything after the name)."""
    out: Dict[str, str] = {}
    in_at = False
    for raw in text.splitlines():
        s = raw.split(";", 1)[0].rstrip()
        st = s.strip()
        if st.startswith("["):
            in_at = st.lower().replace(" ", "").startswith("[atomtypes]")
            continue
        if in_at and st:
            tok = st.split()
            if len(tok) >= 2:
                out[tok[0]] = " ".join(tok[1:])
    return out


def _read_include_pairs(path: Path) -> Dict[Tuple[str, str], str]:
    """Parse a bare-row titratable include (a b func V W ...). Skips #ifdef/#endif/section
    lines, so for pH_dep_interactions.itp this returns the SET of pairs it covers (across all
    pH blocks) — used only to know coverage; the file itself stays a runtime #include."""
    out: Dict[Tuple[str, str], str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        st = line.split(";", 1)[0].strip()
        if not st or st.startswith("#") or st.startswith("["):
            continue
        t = st.split()
        if len(t) >= 5:
            out[_pair_key(t[0], t[1])] = f"{t[2]} {t[3]} {t[4]}"
    return out


def _read_nonbond(text: str, base: Path) -> Dict[Tuple[str, str], str]:
    """(a,b)->'func V W'. Resolves #include lines relative to `base`."""
    out: Dict[Tuple[str, str], str] = {}
    in_nb = False
    for raw in text.splitlines():
        s = raw.split(";", 1)[0].rstrip()
        st = s.strip()
        if st.startswith("#include"):
            inc = st.split('"')[1] if '"' in st else st.split()[1].strip("'<>\"")
            p = (base / inc)
            if p.exists():
                # included files are bare nonbond rows
                for line in p.read_text().splitlines():
                    t = line.split(";", 1)[0].split()
                    if len(t) >= 5:
                        out[_pair_key(t[0], t[1])] = f"{t[2]} {t[3]} {t[4]}"
            continue
        if st.startswith("["):
            in_nb = st.lower().replace(" ", "").startswith("[nonbond_params]")
            continue
        if in_nb and st:
            t = st.split()
            if len(t) >= 5:
                out[_pair_key(t[0], t[1])] = f"{t[2]} {t[3]} {t[4]}"
    return out


def iajd_bead_universe() -> Set[str]:
    """Every bead type used by any non-flagged IAJD (built fresh from corrected SMILES)."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from physics_design.iajd_cg import iajd_to_lipid, BIOACT
    d = pd.read_excel(BIOACT)
    bad = (d["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
           if "audit_status" in d else pd.Series(False, index=d.index))
    used: Set[str] = set()
    for n in (int(x) for x in d[~bad]["IAJD_num"].dropna().unique()):
        try:
            lip = iajd_to_lipid(n, protonated=False)
            used.update(lip.bead_types)
        except Exception:
            pass
    return used


def build(extra_beads: Optional[Set[str]] = None, verbose: bool = True) -> Dict:
    """Generate the merged FF restricted to the IAJD-titratable bead universe."""
    std_text = STD_FF.read_text()
    titr_text = TITR_MARTINI.read_text()
    std_at = _read_atomtypes(std_text)
    titr_at = _read_atomtypes(titr_text)
    std_nb = _read_nonbond(std_text, STD_FF.parent)
    # parse the titratable extensions PER FILE: static (pH-independent) coverage + the set of
    # pairs the pH-dependent file covers. We INLINE only std x std + synthesized cross-terms,
    # and #include the titratable files at runtime so pH_dep_interactions' #ifdef stays intact.
    static_titr_nb: Dict[Tuple[str, str], str] = {}
    for inc in TITR_STATIC_INCLUDES:
        static_titr_nb.update(_read_include_pairs(TITR_FF / inc))
    phdep_keys = set(_read_include_pairs(TITR_FF / PH_DEP_INCLUDE))

    # ---- bead universe actually present in IAJD-titratable systems ----
    titr_specific = set(titr_at)                       # all titratable-declared atomtypes
    # standard beads used by IAJDs + the titratable lipids (POPC/DOPC/DPPC) + N2_10.2 head
    popc_beads = {"Q1p", "Q5n", "SN4a", "N4a", "C1", "C4h"}   # from titratable lipids.itp
    universe = set()
    universe |= iajd_bead_universe()
    universe |= popc_beads
    universe |= titr_specific
    if extra_beads:
        universe |= extra_beads
    # the head motif replaces the IAJD head bead with N2_10.2 + dummies — already in titr.

    # ---- atomtypes for the universe (titratable decl wins for its own beads) ----
    at_lines: List[str] = []
    missing_at = []
    for b in sorted(universe):
        if b in titr_at:
            at_lines.append(f"{b:12s} {titr_at[b]}")
        elif b in std_at:
            at_lines.append(f"{b:12s} {std_at[b]}")
        else:
            missing_at.append(b)

    # ---- nonbond: INLINE std x std + synthesized titratable x IAJD-body;
    #      #include the titratable extension files (so pH_dep #ifdef survives) ----
    beads = sorted(universe)
    nb_lines: List[str] = []
    synthesized: List[str] = []
    zero_by_design: List[str] = []
    truly_missing: List[str] = []
    covered_by_include = 0
    for i, a in enumerate(beads):
        for b in beads[i:]:
            key = _pair_key(a, b)
            # include-coverage FIRST: the reduced martini.itp has an empty inline
            # [nonbond_params] (the 5 includes ARE its matrix), so the includes already define
            # std x std among the reduced bead set — inlining those too -> duplicate-pair errors.
            if key in static_titr_nb or key in phdep_keys:
                covered_by_include += 1                # provided by the #included titratable files
            elif key in std_nb:                        # std x std NOT in includes (IAJD-body) -> INLINE
                nb_lines.append(f"{key[0]:12s} {key[1]:12s} {std_nb[key]}")
            else:
                # missing -> parent synthesis (skip dummy/proton ghosts: sparse by design)
                if a in _DUMMY_LIKE or b in _DUMMY_LIKE:
                    zero_by_design.append(f"{a}-{b}")
                    continue
                pa, pb = parent_of(a) or a, parent_of(b) or b
                pkey = _pair_key(pa, pb)
                src = std_nb.get(pkey) or static_titr_nb.get(pkey)
                if src:
                    nb_lines.append(f"{key[0]:12s} {key[1]:12s} {src}   ; SYNTH parent {pa}x{pb}")
                    synthesized.append(f"{a}x{b} <- {pa}x{pb}")
                else:
                    truly_missing.append(f"{a}-{b}")

    # RELATIVE includes (filename only): the 5 files are siblings of this merged FF, and
    # GROMACS resolves a nested #include relative to the dir of the file containing it — so
    # this stays correct on the pod (absolute laptop paths would break there). The including
    # system.top references THIS file by absolute path (env-derived), which is fine.
    inc_lines = [f'#include "{inc}"' for inc in (TITR_STATIC_INCLUDES + [PH_DEP_INCLUDE])]
    header = (
        "; martini_titratable_full.itp — AUTO-GENERATED by physics_design/build_titratable_ff.py\n"
        "; Titratable Martini 3 FF for IAJD apparent-pKa (Module A). INLINE: standard Martini 3\n"
        "; std x std (IAJD-body palette) + parent-synthesized titratable x IAJD-bead cross terms.\n"
        "; #INCLUDE: the Grunewald titratable extensions — pH_dep_interactions.itp is included (NOT\n"
        "; flattened) so its #ifdef pH<value> blocks stay conditional (the pH response). The\n"
        "; including system.top sets `#define pH<value>` BEFORE this file, selecting the pH block.\n"
        f"; Beads={len(beads)}  inline_pairs={len(nb_lines)} (synth={len(synthesized)})  "
        f"include_covered={covered_by_include}\n"
        "; DO NOT EDIT BY HAND — regenerate: python -m physics_design.build_titratable_ff\n")
    body = ["[ defaults ]", "1 2", "", "[ atomtypes ]", *at_lines,
            "", "[ nonbond_params ]",
            "; --- standard Martini 3 (std x std) + synthesized titratable x IAJD-body ---",
            *nb_lines,
            "; --- Grunewald titratable extensions (#ifdef pH<value> preserved in pH_dep) ---",
            *inc_lines, ""]
    OUT.write_text(header + "\n" + "\n".join(body) + "\n")

    report = {"out": str(OUT), "n_beads": len(beads), "n_inline_pairs": len(nb_lines),
              "n_synthesized": len(synthesized), "n_covered_by_include": covered_by_include,
              "n_zero_by_design_dummy": len(zero_by_design),
              "n_truly_missing": len(truly_missing), "missing_atomtypes": missing_at,
              "truly_missing_pairs": truly_missing[:40], "synthesized_sample": synthesized[:25]}
    if verbose:
        import json
        print(json.dumps(report, indent=2))
        if missing_at:
            print(f"\n!! {len(missing_at)} atomtypes with NO definition: {missing_at}")
        if truly_missing:
            print(f"\n!! {len(truly_missing)} real-bead pairs with NO param and NO parent — "
                  f"these would be ZERO interactions: {truly_missing[:20]}")
        else:
            print("\nOK: every real-bead pair in the universe has an explicit interaction.")
    return report


if __name__ == "__main__":
    build()
