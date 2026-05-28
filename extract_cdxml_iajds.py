"""
Extract IAJDs from `lion_repo/scripts/IAJD for Comp.cdxml`.

Strategy:
- Parse the CDXML.
- For each <fragment>, build a graph from <n> (atoms) and <b> (bonds).
- Convert to an RDKit Mol via a custom CDX-to-RDKit adapter.
- Canonicalize SMILES.
- For each fragment, find the nearest <t>...</t> label (IAJD number annotation)
  by spatial proximity using BoundingBox centroids.
- Validate against the SMILES already recorded for IAJD_347..373 in the
  pre-removal backup; also compute RDKit descriptors used by training.

Writes:
- cdxml_iajds_extracted.csv (the canonical extraction result + validation flags)
"""

from __future__ import annotations

import re
import sys
import math
import csv
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, Descriptors


CDXML_PATH = Path("lion_repo/scripts/IAJD for Comp.cdxml")
BACKUP_XLSX = Path("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx.bak_pre_novel_removal")
OUTPUT_CSV = Path("cdxml_iajds_extracted.csv")

ELEMENT_BY_Z = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def parse_bbox(s: str):
    parts = [float(x) for x in s.split()]
    # CDXML BoundingBox is "left top right bottom"
    x = (parts[0] + parts[2]) / 2.0
    y = (parts[1] + parts[3]) / 2.0
    return parts, (x, y)


def cdx_fragment_to_rdkit(fragment, frag_centroid):
    """Build an RDKit RWMol from a CDXML <fragment> element."""
    mol = Chem.RWMol()
    atom_index = {}  # cdx node id -> rdkit atom idx
    positions = {}   # cdx node id -> (x, y)
    bond_orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                   3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}

    # First pass: collect atom nodes
    for node in fragment.iter("n"):
        nid = node.get("id")
        if nid is None:
            continue
        # NodeType filter: only "Element" or absent (default element)
        nt = node.get("NodeType", "Element")
        if nt not in ("Element", "GenericNickname", "Unspecified"):
            # Skip non-atom nodes
            continue
        z = int(node.get("Element", "6"))  # default C
        symbol = ELEMENT_BY_Z.get(z, "C")
        atom = Chem.Atom(symbol)
        # explicit isotope/charge?
        if node.get("Charge"):
            try:
                atom.SetFormalCharge(int(node.get("Charge")))
            except ValueError:
                pass
        if node.get("Isotope"):
            try:
                atom.SetIsotope(int(node.get("Isotope")))
            except ValueError:
                pass
        idx = mol.AddAtom(atom)
        atom_index[nid] = idx
        if node.get("p"):
            try:
                x, y = [float(v) for v in node.get("p").split()]
                positions[nid] = (x, y)
            except ValueError:
                pass

    # Second pass: bonds
    for bond in fragment.iter("b"):
        b = bond.get("B")
        e = bond.get("E")
        if b is None or e is None:
            continue
        if b not in atom_index or e not in atom_index:
            continue
        order_str = bond.get("Order", "1")
        try:
            order = float(order_str)
        except ValueError:
            order = 1.0
        # CDX uses Order=1.5 for aromatic? Not always; we treat fractional as aromatic
        if order == 1.0:
            bt = Chem.BondType.SINGLE
        elif order == 2.0:
            bt = Chem.BondType.DOUBLE
        elif order == 3.0:
            bt = Chem.BondType.TRIPLE
        elif order == 1.5:
            bt = Chem.BondType.AROMATIC
        else:
            bt = Chem.BondType.SINGLE
        try:
            mol.AddBond(atom_index[b], atom_index[e], bt)
        except RuntimeError:
            # duplicate bond
            pass

    if mol.GetNumAtoms() == 0:
        return None, positions, atom_index

    rdmol = mol.GetMol()
    try:
        Chem.SanitizeMol(rdmol)
    except Exception as exc:
        return None, positions, atom_index
    return rdmol, positions, atom_index


def extract_text_labels(root):
    """Collect ChemDraw text labels along with their centroid."""
    labels = []
    # Text annotations live in <t> with optional <s> children. We look for IAJD\s+\d+.
    for t in root.iter("t"):
        s = "".join(t.itertext()).strip()
        if not s:
            continue
        m = re.search(r"IAJD\s+(\d+)", s)
        if not m:
            continue
        iajd_num = int(m.group(1))
        # Capture surrounding context to detect "(n = 3)" etc.
        bbox = t.get("BoundingBox")
        if not bbox:
            continue
        try:
            _, centroid = parse_bbox(bbox)
        except Exception:
            continue
        labels.append((iajd_num, centroid, s))
    return labels


def nearest_label(frag_centroid, labels):
    best = None
    best_d = float("inf")
    for num, c, text in labels:
        d = math.hypot(frag_centroid[0] - c[0], frag_centroid[1] - c[1])
        if d < best_d:
            best_d = d
            best = (num, text, d)
    return best


def main():
    if not CDXML_PATH.exists():
        print(f"ERROR: {CDXML_PATH} missing", file=sys.stderr)
        sys.exit(1)
    raw = CDXML_PATH.read_bytes()
    idx = raw.find(b"<?xml")
    if idx == -1:
        idx = raw.find(b"<CDXML")
    if idx > 0:
        print(f"Stripping {idx} stray bytes before XML prolog: {raw[:idx]!r}")
        raw = raw[idx:]
    root = ET.fromstring(raw)
    tree = ET.ElementTree(root)

    labels = extract_text_labels(root)
    label_map = defaultdict(list)
    for num, c, txt in labels:
        label_map[num].append((c, txt))
    print(f"Found {len(labels)} IAJD text labels in cdxml; unique numbers: {sorted(set(n for n,_,_ in labels))}")

    # Backup SMILES for validation
    backup = pd.read_excel(BACKUP_XLSX)
    novels = {347, 348, 365, 366, 367, 369, 372, 373}
    backup_smiles = {}
    for _, row in backup[backup["IAJD_num"].isin(novels)].iterrows():
        backup_smiles[int(row["IAJD_num"])] = row["SMILES"]

    # First pass: build list of (fragment, centroid, canonical_smiles)
    fragments = []
    for frag in root.iter("fragment"):
        bbox = frag.get("BoundingBox")
        if not bbox:
            continue
        try:
            _, frag_centroid = parse_bbox(bbox)
        except Exception:
            continue
        rdmol, _, _ = cdx_fragment_to_rdkit(frag, frag_centroid)
        if rdmol is None or rdmol.GetNumAtoms() < 5:
            continue
        smiles = Chem.MolToSmiles(rdmol)
        fragments.append({"centroid": frag_centroid, "smiles": smiles, "mol": rdmol})

    print(f"Extracted {len(fragments)} valid fragments")

    # Canonicalize backup SMILES
    backup_canon_map = {}
    for num, sm in backup_smiles.items():
        bm = Chem.MolFromSmiles(sm)
        if bm is not None:
            backup_canon_map[num] = Chem.MolToSmiles(bm)

    # Assignment: each IAJD num -> the fragment whose SMILES exactly equals the backup canonical.
    # This is the ground-truth assignment by chemistry, not by centroid.
    rows = []
    used_fragment_idx = set()
    for num in sorted(backup_canon_map.keys()):
        backup_canon = backup_canon_map[num]
        # Find a fragment matching this SMILES
        candidates = [(i, f) for i, f in enumerate(fragments)
                      if f["smiles"] == backup_canon and i not in used_fragment_idx]
        if not candidates:
            # No exact match — record the closest centroid candidate among labels of this num
            label_centroids = [c for n, c, _ in labels if n == num]
            if label_centroids:
                # pick fragment closest to any of this num's label centroids
                def label_dist(f):
                    return min(math.hypot(f["centroid"][0]-lc[0], f["centroid"][1]-lc[1])
                               for lc in label_centroids)
                avail = [(i, f) for i, f in enumerate(fragments) if i not in used_fragment_idx]
                avail.sort(key=lambda x: label_dist(x[1]))
                if avail:
                    fi, frag = avail[0]
                    match = "MISMATCH_BUT_NEAREST"
                    dist = label_dist(frag)
                else:
                    continue
            else:
                continue
        else:
            # Pick the candidate closest to a label of this num
            label_centroids = [c for n, c, _ in labels if n == num]
            if label_centroids:
                def label_dist(f):
                    return min(math.hypot(f["centroid"][0]-lc[0], f["centroid"][1]-lc[1])
                               for lc in label_centroids)
                candidates.sort(key=lambda x: label_dist(x[1]))
            fi, frag = candidates[0]
            dist = (label_dist(frag) if label_centroids
                    else float("nan"))
            match = "match"
        used_fragment_idx.add(fi)
        rdmol = frag["mol"]
        smiles = frag["smiles"]
        iajd_num = num
        label_text = next((t for n, c, t in labels if n == num), f"IAJD {num}")
        backup_canon_for_row = backup_canon_map.get(iajd_num)

        # Compute the descriptor set we'll use for training featurization
        n_atoms = rdmol.GetNumHeavyAtoms()
        mw = Descriptors.ExactMolWt(rdmol)
        logp = Descriptors.MolLogP(rdmol)
        tpsa = Descriptors.TPSA(rdmol)
        rot = Descriptors.NumRotatableBonds(rdmol)
        n_n = sum(1 for a in rdmol.GetAtoms() if a.GetAtomicNum() == 7)
        n_o = sum(1 for a in rdmol.GetAtoms() if a.GetAtomicNum() == 8)
        n_arom = rdMolDescriptors.CalcNumAromaticRings(rdmol)

        rows.append({
            "iajd_num": iajd_num,
            "cdxml_label": label_text,
            "smiles_cdxml_canon": smiles,
            "smiles_backup_canon": backup_canon_for_row,
            "validation": match,
            "n_heavy": n_atoms,
            "mol_wt_exact": round(mw, 4),
            "mol_logp": round(logp, 3),
            "tpsa": round(tpsa, 2),
            "rot_bonds": rot,
            "n_N": n_n,
            "n_O": n_o,
            "n_aromatic_rings": n_arom,
            "centroid_dist": round(dist, 2) if dist == dist else None,
        })

    print(f"Produced {len(rows)} IAJD records")
    if not rows:
        print("ERROR: no rows produced", file=sys.stderr)
        sys.exit(1)

    # Write csv
    df = pd.DataFrame(rows).sort_values(["iajd_num", "centroid_dist"])
    # Keep one record per IAJD number (closest label, in case of duplicates)
    df_dedup = df.drop_duplicates(subset=["iajd_num"], keep="first").reset_index(drop=True)
    df_dedup.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(df_dedup)} rows")

    # Print validation summary
    print("\n=== Validation summary ===")
    print(df_dedup[["iajd_num", "cdxml_label", "validation", "n_heavy", "mol_wt_exact"]].to_string(index=False))
    n_match = (df_dedup["validation"] == "match").sum()
    print(f"\nSMILES match vs backup: {n_match}/{len(df_dedup)}")
    return df_dedup


if __name__ == "__main__":
    main()
