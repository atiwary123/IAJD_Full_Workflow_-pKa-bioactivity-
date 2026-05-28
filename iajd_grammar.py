"""
iajd_grammar.py — IAJD structural grammar + mutation operators.

Approach: every training IAJD is a (family, head, linker_n, linkage, tail_motifs)
tuple. We assemble candidate SMILES from these parameters via family-specific
templates, rather than trying to mutate SMILES strings directly (which breaks
connectivity).

Supported families & templates (covers > 95% of training rows):
  - sSS-Nonsym  : 3,5-bis(O-tail)-benzyl  linkage  (CH2)n  head_amine
  - GA-Tris     : 3,4,5-tris(O-tail)-benzyl  ester  (CH2)n  piperazine_head
  - PE-Tris     : 3,4,5-tris(O-tail)-benzyl  ester  (CH2)n  piperazine_head
  - PE-Gallic   : 3,4,5-tris(O-tail)-benzyl  amide  (CH2)n  DMBA / piperazine
  - Dialkoxybenzyl : 3,5-bis(O-tail)-benzyl  ester  (CH2)n  head

For each family we keep:
  - tail library  (list of canonical alkyl chain SMILES, e.g. `CCCCCCCCCCCC`,
                    `CCCCC(CC)CC`)
  - linker length set (set of small ints, e.g. {2,3,4,5})
  - head library  (e.g. {DMA, MPRZ, HPRZ, H2EPRZ, PIP})
  - linkage set   ({ester, amide})

Single-mutation grammar:
  - swap_head       : head ← any other head in family library
  - resize_linker   : linker_n ± 1 within family's observed range
  - swap_tail_i     : tail at position i ← any other tail in family library
  - lengthen_tail   : add 2C to longest tail
  - shorten_tail    : remove 2C from longest tail (≥ family min)
  - swap_linkage    : ester ↔ amide if both observed in family

Each mutation returns a fully-connected, sanitized SMILES.
"""
from __future__ import annotations
import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Iterator, Tuple, Optional

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)


# Head SMILES with [*] attachment to the linker carbon.
# NOTE: ring closure number 2 used in head fragments because the family
# templates already use ring 1 for the benzene scaffold. Mixing them up
# produces fused / macrocyclic garbage.
HEAD_FRAGMENTS = {
    "DMA":    "*N(C)C",
    "MPRZ":   "*N2CCN(C)CC2",
    "PIP":    "*N2CCCCC2",
    "HPRZ":   "*N2CCN(CCO)CC2",
    "H2EPRZ": "*N2CCN(CCOCCO)CC2",
    "DMBA":   "*Cc2ccc(N(C)C)cc2",   # PE-gallic-style benzyl-DMA
}


# ──────────────────────────────────────────────────────────────────────
# Template assemblers
# ──────────────────────────────────────────────────────────────────────

def _attach_head(head_group: str) -> str:
    """Return the head SMILES with the [*] dummy stripped (caller prepends linker)."""
    sm = HEAD_FRAGMENTS.get(head_group)
    if sm is None:
        return None
    return sm.replace("*", "")   # caller writes `...CH2{head_body}`


def assemble_ga_tris(head_group: str, linker_n: int, linkage: str,
                     tails: Tuple[str, str, str]) -> Optional[str]:
    """3,4,5-tris(O-tail)-benzyl ester/amide-linker-piperazine_head."""
    head_body = _attach_head(head_group)
    if head_body is None or linker_n is None or linker_n < 1 or len(tails) != 3:
        return None
    chain = "C" * (max(linker_n, 1) - 1)
    link = "OC(=O)" if linkage == "ester" else "NC(=O)"
    t1, t2, t3 = tails
    smi = f"{t1}Oc1cc(CO{link}{chain}{head_body})cc({t2})c1O{t3}"
    # Wait: the t2/t3 attachment also needs O. Fix:
    smi = f"{t1}Oc1cc(C{link[::-1] if False else ''}{link}{chain}{head_body})cc(O{t2})c1O{t3}"
    # cleaner build:
    smi = (
        f"{t1}Oc1cc("           # tail1-O-benzene-C
        f"C{link}"              # benzyl-OC(=O)/NC(=O)
        f"{chain}"              # linker (CH2)_{n-1}
        f"{head_body}"          # piperazine head
        f")cc(O{t2})c1O{t3}"    # tails 2,3 on the ring
    )
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def assemble_pe_tris(head_group: str, linker_n: int, linkage: str,
                     tails: Tuple[str, str, str]) -> Optional[str]:
    """PE-Tris architecture mirrors GA-Tris for our purposes."""
    return assemble_ga_tris(head_group, linker_n, linkage, tails)


def assemble_sss_nonsym(head_group: str, linker_n: int, linkage: str,
                        tails: Tuple[str, str]) -> Optional[str]:
    """3,5-bis(O-tail)-benzyl  linkage  linker  head."""
    head_body = _attach_head(head_group)
    if head_body is None or linker_n is None or linker_n < 1 or len(tails) != 2:
        return None
    chain = "C" * (max(linker_n, 1) - 1)
    link = "OC(=O)" if linkage == "ester" else "NC(=O)"
    t1, t2 = tails
    smi = (
        f"{t1}Oc1cc("
        f"C{link}"
        f"{chain}"
        f"{head_body}"
        f")cc(O{t2})c1"
    )
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


FAMILY_ASSEMBLERS = {
    "sSS-Nonsym":     (assemble_sss_nonsym, 2),
    "GA-Tris":        (assemble_ga_tris,    3),
    "PE-Tris":        (assemble_pe_tris,    3),
    "PE-Gallic":      (assemble_pe_tris,    3),    # same template
    "Dialkoxybenzyl": (assemble_sss_nonsym, 2),
}


# ──────────────────────────────────────────────────────────────────────
# Library
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Library:
    heads:     Dict[str, Set[str]]   = field(default_factory=lambda: defaultdict(set))
    tails:     Dict[str, Set[str]]   = field(default_factory=lambda: defaultdict(set))
    linkers:   Dict[str, Set[int]]   = field(default_factory=lambda: defaultdict(set))
    linkages:  Dict[str, Set[str]]   = field(default_factory=lambda: defaultdict(set))
    families:  Set[str]              = field(default_factory=set)

    def summary(self) -> dict:
        return {
            "n_families": len(self.families),
            "per_family": {
                f: {
                    "n_heads":  len(self.heads[f]),
                    "heads":    sorted(self.heads[f]),
                    "n_tails":  len(self.tails[f]),
                    "tails":    sorted(self.tails[f]),
                    "linkers":  sorted(self.linkers[f]),
                    "linkages": sorted(self.linkages[f]),
                }
                for f in sorted(self.families)
            },
        }


def _extract_tails_from_smiles(smiles: str) -> List[str]:
    """Find each `c-O-(CH2)…` substituent on aromatic rings; return tail SMILES
    (just the alkyl chain, no leading O)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    tails = []
    seen = set()
    aro_o_c = Chem.MolFromSmarts("c[OX2][CX4]")
    for ar, o, c0 in mol.GetSubstructMatches(aro_o_c):
        if c0 in seen:
            continue
        tail_atoms = []
        stack = [c0]
        local_seen = set()
        while stack:
            a = stack.pop()
            if a in local_seen:
                continue
            local_seen.add(a)
            atom = mol.GetAtomWithIdx(a)
            if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
                continue
            # Don't follow back through the linker O or into nitrogens
            tail_atoms.append(a)
            for n in atom.GetNeighbors():
                ni = n.GetIdx()
                if ni == o or n.GetIsAromatic() or n.GetAtomicNum() != 6:
                    continue
                stack.append(ni)
        if tail_atoms:
            try:
                sm = Chem.MolFragmentToSmiles(mol, atomsToUse=tail_atoms, canonical=True)
                m = Chem.MolFromSmiles(sm)
                if m is not None and m.GetNumHeavyAtoms() >= 3:
                    tails.append(Chem.MolToSmiles(m))
                    seen.update(tail_atoms)
            except Exception:
                pass
    return tails


def build_library(bioact_xlsx: str | Path) -> Library:
    df = pd.read_excel(bioact_xlsx)
    lib = Library()
    for _, r in df.iterrows():
        smi = r.get("SMILES_canonical") or r.get("SMILES")
        fam = str(r.get("family", "")).strip()
        if pd.isna(smi) or not fam:
            continue
        if fam not in FAMILY_ASSEMBLERS:
            continue
        lib.families.add(fam)
        head = r.get("head_group")
        linkage = str(r.get("linkage") or "").lower()
        linker_n = r.get("linker_length") or r.get("Linker_Length")
        if pd.notna(head) and head in HEAD_FRAGMENTS:
            lib.heads[fam].add(head)
        if linkage in ("ester", "amide"):
            lib.linkages[fam].add(linkage)
        if pd.notna(linker_n):
            try:
                lib.linkers[fam].add(int(round(float(linker_n))))
            except (TypeError, ValueError):
                pass
        for t in _extract_tails_from_smiles(str(smi)):
            lib.tails[fam].add(t)
    return lib


# ──────────────────────────────────────────────────────────────────────
# Seed + mutation grammar
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Seed:
    family:    str
    head:      str                       # e.g. "MPRZ"
    linker_n:  int
    linkage:   str                       # "ester" / "amide"
    tails:     Tuple[str, ...]           # canonical tail SMILES
    iajd_num:  Optional[int] = None
    y_obs:     Optional[float] = None
    # If the seed came from a training row, store the ORIGINAL SMILES
    # so callers can score it via cached features instead of the
    # round-tripped (positional-isomer-shifted) assembled form.
    original_smiles: Optional[str] = None

    def assemble(self) -> Optional[str]:
        assembler, n_tails_expected = FAMILY_ASSEMBLERS[self.family]
        if len(self.tails) != n_tails_expected:
            # pad / truncate to match expected number
            tails = list(self.tails) + [self.tails[-1]] * (n_tails_expected - len(self.tails))
            tails = tuple(tails[:n_tails_expected])
        else:
            tails = self.tails
        return assembler(self.head, self.linker_n, self.linkage, tails)


def decompose_row(row: dict) -> Optional[Seed]:
    fam = str(row.get("family") or "").strip()
    if fam not in FAMILY_ASSEMBLERS:
        return None
    head = row.get("head_group")
    if pd.isna(head) or head not in HEAD_FRAGMENTS:
        return None
    linker_n = row.get("linker_length") or row.get("Linker_Length")
    if pd.isna(linker_n):
        return None
    linkage = str(row.get("linkage") or "ester").lower()
    if linkage not in ("ester", "amide"):
        linkage = "ester"
    smi = row.get("SMILES_canonical") or row.get("SMILES")
    if pd.isna(smi):
        return None
    tails = _extract_tails_from_smiles(str(smi))
    _, n_expected = FAMILY_ASSEMBLERS[fam]
    if len(tails) < 1:
        return None
    if len(tails) < n_expected:
        tails = tails + [tails[-1]] * (n_expected - len(tails))
    tails = tuple(tails[:n_expected])
    try:
        original = Chem.MolToSmiles(Chem.MolFromSmiles(str(smi)))
    except Exception:
        original = None
    return Seed(
        family=fam, head=str(head), linker_n=int(round(float(linker_n))),
        linkage=linkage, tails=tails,
        iajd_num=int(row["IAJD_num"]) if pd.notna(row.get("IAJD_num")) else None,
        y_obs=float(row["log10_flux_total"]) if pd.notna(row.get("log10_flux_total")) else None,
        original_smiles=original,
    )


def _tail_modify(tail_smi: str, delta_c: int) -> Optional[str]:
    """Add or remove CH2 from the end of the tail string."""
    m = Chem.MolFromSmiles(tail_smi)
    if m is None:
        return None
    n = m.GetNumHeavyAtoms()
    if delta_c < 0 and n + delta_c < 3:
        return None
    new_sm = tail_smi + ("C" * delta_c) if delta_c > 0 else tail_smi[:delta_c]
    if not new_sm:
        return None
    try:
        m2 = Chem.MolFromSmiles(new_sm)
        if m2 is None:
            return None
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


def humanize_mutation_tag(tag: str) -> str:
    """Turn a cryptic mutation_tag into a one-line, human-readable description.

    Examples:
        head:HPRZ→MPRZ              → "Head swapped: HPRZ → MPRZ"
        linker:4→3C                  → "Linker shortened: 4C → 3C"
        tail0:CCCCCCCC→CCCCCCCCCC    → "Tail #1 changed from C8 to C10"
        tail_extend_4C               → "Longest tail extended by 4C"
        tail_shorten_2C              → "Longest tail shortened by 2C"
        linkage:ester→amide          → "Linkage changed: ester → amide"
        all_tails→C12                → "All tails unified to C12"
        multi_tail2→C16              → "Two tails simultaneously swapped to C16"
        family:GA-Tris→PE-Tris       → "Architecture switched: GA-Tris → PE-Tris"
        synth_head→MPRZ_OMe          → "Synthetic head added: MPRZ-OMe"
        seed                          → "Original seed (no mutation)"
    """
    if tag == "seed":
        return "Original seed (no mutation)"
    if tag.startswith("head:"):
        a, b = tag[5:].split("→")
        return f"Head swapped: {a} → {b}"
    if tag.startswith("linker:"):
        body = tag[7:].rstrip("C")
        a, b = body.split("→")
        direction = "lengthened" if int(b) > int(a) else "shortened"
        return f"Linker {direction}: {a}C → {b}C"
    if tag.startswith("tail") and "→" in tag:
        # tail0:CCCC→CCCCCCC  or multi_tail2→CCCC
        idx_part, rest = tag.split(":") if ":" in tag else (tag, tag.split("→")[1])
        idx = idx_part[4:] if idx_part.startswith("tail") and idx_part[4:].isdigit() else None
        a, b = tag.split("→")[-2:] if "→" in tag else (tag, tag)
        # parse a/b lengths if pure-C strings
        def _tail_label(s):
            mm = re.fullmatch(r"C+", s)
            if mm:
                return f"C{len(s)}"
            m = Chem.MolFromSmiles(s)
            if m:
                return f"{m.GetNumHeavyAtoms()}-atom branched"
            return s
        a_lab, b_lab = _tail_label(a.split(":")[-1]), _tail_label(b)
        pos = f" #{int(idx)+1}" if idx is not None else ""
        return f"Tail{pos} changed from {a_lab} to {b_lab}"
    if tag.startswith("tail_extend_"):
        n = tag.replace("tail_extend_", "").rstrip("C")
        return f"Longest tail extended by {n}C"
    if tag.startswith("tail_shorten_"):
        n = tag.replace("tail_shorten_", "").rstrip("C")
        return f"Longest tail shortened by {n}C"
    if tag.startswith("linkage:"):
        a, b = tag[8:].split("→")
        return f"Linkage changed: {a} → {b}"
    if tag.startswith("all_tails→"):
        new = tag.split("→")[1]
        m = Chem.MolFromSmiles(new)
        lab = f"C{m.GetNumHeavyAtoms()}" if (m and not any(c in new for c in "()")) else new
        return f"All tails unified to {lab}"
    if tag.startswith("multi_tail2→"):
        new = tag.split("→")[1]
        m = Chem.MolFromSmiles(new)
        lab = f"C{m.GetNumHeavyAtoms()}" if (m and not any(c in new for c in "()")) else new
        return f"Two tails simultaneously swapped to {lab}"
    if tag.startswith("family:"):
        a, b = tag[7:].split("→")
        return f"Architecture switched: {a} → {b}"
    if tag.startswith("synth_head→"):
        new = tag.split("→")[1].replace("_", "-")
        return f"Synthetic head substituent added: {new}"
    return tag  # fallback


def propose_single_mutations(seed: Seed, library: Library) -> Iterator[Tuple[str, str, Seed]]:
    """Yield (SMILES, mutation_tag, mutated_Seed) for each single-step mutation.

    Mutation grammar (expanded from the original 5-op set):
       (1) head_swap         — replace head with another in family library
       (2) linker_resize     — change linker by ±1, ±2 (capped at family observed range)
       (3) tail_swap         — replace any tail with another from family library
       (4) tail_extend_NC    — extend longest tail by 2, 4, 6 C
       (5) tail_shorten_NC   — shorten longest tail by 2, 4 C (if min stays ≥ 5)
       (6) linkage_swap      — ester ↔ amide if both observed
       (7) multi_tail_swap   — change 2 tails at once (high-jump exploration)
       (8) all_tails_uniform — make all tails the same (collapse asymmetry)
       (9) tail_branching    — swap straight tail for known branched analog
       (10) cross_family     — change architecture template (GA-Tris ↔ PE-Tris)
       (11) head_chemistry   — synthetic head variants beyond family library
                                (e.g., add ethanol/methoxyethyl to distal N)
    """
    fam = seed.family
    if fam not in library.families:
        return

    # 1. swap head
    for new_head in sorted(library.heads[fam]):
        if new_head == seed.head:
            continue
        s2 = Seed(fam, new_head, seed.linker_n, seed.linkage, seed.tails)
        sm = s2.assemble()
        if sm:
            yield sm, f"head:{seed.head}→{new_head}", s2

    # 2. resize linker ±1, ±2
    for new_n in sorted(library.linkers[fam]):
        if new_n == seed.linker_n:
            continue
        if abs(new_n - seed.linker_n) > 2:
            continue
        s2 = Seed(fam, seed.head, new_n, seed.linkage, seed.tails)
        sm = s2.assemble()
        if sm:
            yield sm, f"linker:{seed.linker_n}→{new_n}C", s2

    # 3. swap each tail (single-position)
    for i, current_tail in enumerate(seed.tails):
        for new_tail in sorted(library.tails[fam]):
            if new_tail == current_tail:
                continue
            new_tails = list(seed.tails)
            new_tails[i] = new_tail
            s2 = Seed(fam, seed.head, seed.linker_n, seed.linkage, tuple(new_tails))
            sm = s2.assemble()
            if sm:
                yield sm, f"tail{i}:{current_tail}→{new_tail}", s2

    # 4. tail extension (longest tail, +2 / +4 / +6 C)
    longest_i = max(range(len(seed.tails)), key=lambda i: len(seed.tails[i]))
    longest = seed.tails[longest_i]
    for delta in (2, 4, 6):
        new_tail = _tail_modify(longest, delta)
        if new_tail and new_tail != longest:
            new_tails = list(seed.tails)
            new_tails[longest_i] = new_tail
            s2 = Seed(fam, seed.head, seed.linker_n, seed.linkage, tuple(new_tails))
            sm = s2.assemble()
            if sm:
                yield sm, f"tail_extend_{delta}C", s2
    # tail shortening (longest tail, -2, -4 C, only if shrunk ≥ 5)
    for delta in (-2, -4):
        new_tail = _tail_modify(longest, delta)
        if new_tail and new_tail != longest:
            mol = Chem.MolFromSmiles(new_tail)
            if mol is None or mol.GetNumHeavyAtoms() < 5:
                continue
            new_tails = list(seed.tails)
            new_tails[longest_i] = new_tail
            s2 = Seed(fam, seed.head, seed.linker_n, seed.linkage, tuple(new_tails))
            sm = s2.assemble()
            if sm:
                yield sm, f"tail_shorten_{abs(delta)}C", s2

    # 5. swap linkage
    for new_link in sorted(library.linkages[fam]):
        if new_link == seed.linkage:
            continue
        s2 = Seed(fam, seed.head, seed.linker_n, new_link, seed.tails)
        sm = s2.assemble()
        if sm:
            yield sm, f"linkage:{seed.linkage}→{new_link}", s2

    # 6. all-tails-uniform: collapse asymmetric tails to a single shared tail
    # (the library element with longest extension)
    if len(set(seed.tails)) > 1:
        for unified_tail in sorted(library.tails[fam], key=lambda t: -len(t))[:5]:
            new_tails = tuple([unified_tail] * len(seed.tails))
            if new_tails == seed.tails:
                continue
            s2 = Seed(fam, seed.head, seed.linker_n, seed.linkage, new_tails)
            sm = s2.assemble()
            if sm:
                yield sm, f"all_tails→{unified_tail}", s2

    # 7. multi-tail swap: change two tails simultaneously
    for new_tail in sorted(library.tails[fam], key=lambda t: -len(t))[:6]:
        if new_tail in seed.tails:
            continue
        # Swap into positions 0 and 1
        if len(seed.tails) >= 2:
            new_tails = list(seed.tails)
            new_tails[0] = new_tail; new_tails[1] = new_tail
            s2 = Seed(fam, seed.head, seed.linker_n, seed.linkage, tuple(new_tails))
            sm = s2.assemble()
            if sm:
                yield sm, f"multi_tail2→{new_tail}", s2

    # 8. cross-family jump (GA-Tris ↔ PE-Tris; same head/linker/linkage/tails)
    for other_fam in library.families:
        if other_fam == fam:
            continue
        if other_fam not in FAMILY_ASSEMBLERS:
            continue
        # Only jump between architectures that share the same tail count
        _, n_self = FAMILY_ASSEMBLERS[fam]
        _, n_other = FAMILY_ASSEMBLERS[other_fam]
        if n_other != n_self:
            continue
        # Use a head in the new family's library
        if seed.head not in library.heads[other_fam]:
            continue
        new_linker = seed.linker_n
        if new_linker not in library.linkers[other_fam]:
            cands = sorted(library.linkers[other_fam], key=lambda x: abs(x - seed.linker_n))
            if not cands:
                continue
            new_linker = cands[0]
        new_linkage = seed.linkage
        if new_linkage not in library.linkages[other_fam] and library.linkages[other_fam]:
            new_linkage = next(iter(library.linkages[other_fam]))
        s2 = Seed(other_fam, seed.head, new_linker, new_linkage, seed.tails)
        sm = s2.assemble()
        if sm:
            yield sm, f"family:{fam}→{other_fam}", s2

    # 9. synthetic head variants — append a CH2CH2OH / OCH3 / OEt to the distal
    # piperazine N. Only meaningful for piperazine-based heads.
    if seed.head in ("MPRZ", "HPRZ", "H2EPRZ"):
        # Add ether-extension to head (turn MPRZ into MPRZ-OEt-like)
        for synth_tag, new_head_smiles in [
            ("MPRZ_OMe",  "*N2CCN(COC)CC2"),
            ("MPRZ_OEt",  "*N2CCN(COCC)CC2"),
            ("HPRZ_OMe",  "*N2CCN(CCOC)CC2"),
            ("DEHPRZ",    "*N2CCN(CC(O)CO)CC2"),
        ]:
            # Inject directly by swapping HEAD_FRAGMENTS temporarily
            old_frag = HEAD_FRAGMENTS.get(synth_tag)
            HEAD_FRAGMENTS[synth_tag] = new_head_smiles
            try:
                s2 = Seed(fam, synth_tag, seed.linker_n, seed.linkage, seed.tails)
                sm = s2.assemble()
                if sm:
                    yield sm, f"synth_head→{synth_tag}", s2
            finally:
                if old_frag is None:
                    HEAD_FRAGMENTS.pop(synth_tag, None)
                else:
                    HEAD_FRAGMENTS[synth_tag] = old_frag


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BIO = "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
    print(f"Building library from {BIO}…")
    lib = build_library(BIO)
    print(f"  {len(lib.families)} families: {sorted(lib.families)}")
    for f in sorted(lib.families):
        print(f"    {f:<18s}  heads={sorted(lib.heads[f])}  "
              f"linkers={sorted(lib.linkers[f])}  linkages={sorted(lib.linkages[f])}  "
              f"n_tails={len(lib.tails[f])}")

    print(f"\n--- Round-trip test: decompose + reassemble training rows ---")
    df = pd.read_excel(BIO)
    n_ok = n_mismatch = n_fail = 0
    examples = []
    for _, r in df.iterrows():
        seed = decompose_row(r.to_dict())
        if seed is None:
            n_fail += 1
            continue
        reassembled = seed.assemble()
        original = Chem.MolToSmiles(Chem.MolFromSmiles(r["SMILES_canonical"] or r["SMILES"]))
        if reassembled is None:
            n_fail += 1
            continue
        if reassembled == original:
            n_ok += 1
        else:
            n_mismatch += 1
            if len(examples) < 3:
                examples.append((r["IAJD_num"], original, reassembled))
    print(f"  ok={n_ok}  mismatch={n_mismatch}  fail={n_fail}")
    for iajd, orig, reas in examples:
        print(f"    IAJD {iajd}:")
        print(f"      orig: {orig}")
        print(f"      reas: {reas}")

    # Demo mutations
    print(f"\n--- Demo: 5 mutations of IAJD 347 ---")
    ga_row = df[df["IAJD_num"] == 347].iloc[0].to_dict()
    seed = decompose_row(ga_row)
    print(f"  Seed: {seed}")
    for i, (sm, tag, _) in enumerate(propose_single_mutations(seed, lib)):
        if i >= 5:
            break
        print(f"    [{tag}]  {sm}")
