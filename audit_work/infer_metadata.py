"""infer_metadata.py — infer head_group / linker_length / linkage from SMILES structure,
so decompose_row stops failing on rows with blank metadata. VALIDATE against already-filled
rows before writing anything to the dataset.

head_group: matched by substructure (most-specific first).
linker_length: # carbons in the acyl chain from the ester/amide carbonyl to the head N,
               inclusive of the carbonyl C  (e.g. C(=O)CCCN -> 4, matching the dataset).
linkage:    ester if the benzyl/core attaches via O-C(=O); amide if via N-C(=O).
"""
from __future__ import annotations
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# head substructure SMARTS, MOST SPECIFIC FIRST (H2EPRZ before HPRZ before MPRZ; DMBA before DMA).
# Clean ring forms first; then macrocycle-tolerant "piperazine diamine by N-substituent"
# fallbacks (the sSS-Nonsym macrocyclic IAJDs embed the piperazine in a big ring, so the
# ring-1 SMARTS miss it — but the head TYPE is still set by the N-substituent: CH3=MPRZ,
# CCO=HPRZ, CCOCCO=H2EPRZ). DMA last so a piperazine-N-methyl is MPRZ, not DMA.
HEAD_SMARTS = [
    ("H2EPRZ", "[NX3]1CCN(CCOCCO)CC1"),
    ("HPRZ",   "[NX3]1CCN(CCO)CC1"),
    ("MPRZ",   "[NX3]1CCN([CH3])CC1"),
    ("PIP",    "[NX3]1CCCCC1"),
    ("DMBA",   "[CH2]c1ccc(N([CH3])[CH3])cc1"),
    ("H2EPRZ", "[NX3](CCOCCO)[CX4][CX4][NX3]"),   # macrocyclic / embedded piperazine
    ("HPRZ",   "[NX3](CCO)[CX4][CX4][NX3]"),
    ("MPRZ",   "[NX3]([CH3])[CX4][CX4][NX3]"),
    ("DMA",    "[NX3]([CH3])[CH3]"),
]
_HEAD_PATTS = [(name, Chem.MolFromSmarts(s)) for name, s in HEAD_SMARTS]


def infer_head(mol):
    for name, patt in _HEAD_PATTS:
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return None


def _head_N(mol):
    """Return the head amine nitrogen atom idx (the terminal/ring N that carries the head)."""
    for name, patt in _HEAD_PATTS:
        if patt is None:
            continue
        m = mol.GetSubstructMatch(patt)
        if m:
            # first atom of the SMARTS is the N (except DMBA where N is inside) — find an N in match
            for ai in m:
                if mol.GetAtomWithIdx(ai).GetAtomicNum() == 7:
                    return ai, name
    return None, None


def infer_linkage_and_linker(mol):
    """Find the ester/amide carbonyl whose acyl chain reaches the head N; return
    (linkage, linker_length)."""
    headN, _ = _head_N(mol)
    if headN is None:
        return None, None
    # all carbonyl carbons C(=O)
    carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
    cands = [m[0] for m in mol.GetSubstructMatches(carbonyl)]
    best = None
    for cC in cands:
        # BFS from carbonyl C through CARBON chain to headN; count carbons
        from collections import deque
        q = deque([(cC, 1)])           # carbonyl counts as carbon #1
        seen = {cC}
        found = None
        while q:
            a, depth = q.popleft()
            atom = mol.GetAtomWithIdx(a)
            for nb in atom.GetNeighbors():
                ni = nb.GetIdx()
                if ni == headN:
                    found = depth
                    break
                if ni in seen:
                    continue
                if nb.GetAtomicNum() == 6 and not nb.GetIsAromatic():
                    seen.add(ni); q.append((ni, depth + 1))
                elif nb.GetAtomicNum() == 8 and nb.GetDegree() == 2:
                    # cross a single ester/ether O (benzoate: C(=O)-O-(CH2)n-N) WITHOUT
                    # counting it as a linker carbon; keeps depth = carbon count.
                    seen.add(ni); q.append((ni, depth))
            if found:
                break
        if found is not None and (best is None or found < best[0]):
            # linkage: what does this carbonyl attach to on the OTHER side (toward core)?
            link = None
            for nb in mol.GetAtomWithIdx(cC).GetNeighbors():
                if nb.GetAtomicNum() == 8 and nb.GetDegree() == 2:   # ester O
                    link = "ester"
                elif nb.GetAtomicNum() == 7:                          # amide N
                    link = "amide"
            best = (found, link)
    if best is None:
        return None, None
    return best[1], best[0]


def infer_row(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return {"head_group": None, "linkage": None, "linker_length": None}
    head = infer_head(m)
    linkage, linker = infer_linkage_and_linker(m)
    return {"head_group": head, "linkage": linkage, "linker_length": linker}


if __name__ == "__main__":
    d = pd.read_excel("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
    bad = d["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
    d = d[~bad].reset_index(drop=True)
    smicol = "SMILES_canonical" if "SMILES_canonical" in d else "SMILES"
    # VALIDATE: where existing value is present, does inference match?
    agree = {"head_group": [0, 0], "linker_length": [0, 0], "linkage": [0, 0]}
    examples = []
    for _, r in d.iterrows():
        inf = infer_row(r.get(smicol) or r.get("SMILES"))
        for col, ikey in (("head_group", "head_group"), ("linker_length", "linker_length"), ("linkage", "linkage")):
            ev = r.get(col)
            if pd.notna(ev) and inf[ikey] is not None:
                agree[col][1] += 1
                a = (str(int(round(float(ev)))) == str(inf[ikey])) if col == "linker_length" else (str(ev).strip().upper() == str(inf[ikey]).upper())
                if a:
                    agree[col][0] += 1
                elif len(examples) < 8:
                    examples.append(f"  MISMATCH {col} IAJD{r.get('IAJD_num')} fam={r.get('family')}: existing={ev!r} inferred={inf[ikey]!r}")
    print("=== inference accuracy vs existing-filled rows ===")
    for col, (ok, tot) in agree.items():
        print(f"  {col:14s}: {ok}/{tot} agree" + (f"  ({100*ok//max(tot,1)}%)" if tot else ""))
    for e in examples:
        print(e)
    # COVERAGE: how many currently-blank rows could we fill?
    fillable = {"head_group": 0, "linker_length": 0, "linkage": 0}
    for _, r in d.iterrows():
        inf = infer_row(r.get(smicol) or r.get("SMILES"))
        for col, ikey in (("head_group", "head_group"), ("linker_length", "linker_length"), ("linkage", "linkage")):
            if pd.isna(r.get(col)) and inf[ikey] is not None:
                fillable[col] += 1
    print("=== currently-blank rows we could fill ===")
    for col, n in fillable.items():
        print(f"  {col:14s}: +{n}")
