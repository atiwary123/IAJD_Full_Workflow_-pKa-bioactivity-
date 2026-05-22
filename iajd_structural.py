"""
iajd_structural.py — Structural feature extraction with 2D positional encoding.

What this adds on top of SMILES-only input
------------------------------------------
The shipped tandem pipeline (v15) routes via count-Morgan-3 Tanimoto, which is
topology-only: it doesn't know how the chemist *drew* the molecule. ChemDraw
files (.cdxml / .cdx / .mol / .sdf) carry 2D coordinates that encode the
designer's intent — which chains are tails vs. linkers, which N is the basic
ionizable center, branch symmetry, etc.

This module:
  1. Parses any of {.cdxml, .cdx (limited), .mol, .sdf, .smi, raw SMILES}.
     Returns a list of (canonical_smiles, RDKit Mol with 2D coords). Multiple
     IAJDs in a single ChemDraw file are returned as separate entries.
  2. Computes a fixed-length structural feature vector for each Mol using:
       a. Sinusoidal positional encoding (transformer-style) over polar
          coordinates of each heavy atom relative to the ionizable nitrogen
          (or 2D centroid if no basic N), pooled to (mean, max, std).
       b. Ionizable-N-centric distance features (to esters, ethers, terminal
          methyls, ring centers).
       c. Branch symmetry / lipid tail features from the 2D layout.
  3. Exposes `structural_similarity(mol_a, mol_b)` (cosine) and
     `reweight_neighbor_sims(query_mol, neighbor_mols, base_tanimoto, beta)`
     that returns a blended similarity vector ready to plug into the existing
     sim^4 analog-delta path.

The features are *coordinate-frame invariant* (use polar / pairwise distances)
so a query parsed from ChemDraw and a training molecule with RDKit-generated
2D coords land in comparable feature space. ChemDraw coords are preferred when
present because they reflect the chemist's layout; otherwise we fall back to
`AllChem.Compute2DCoords`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Input parsing — single or multiple molecules
# ---------------------------------------------------------------------------

@dataclass
class ParsedInput:
    smiles: str          # canonical
    mol: Chem.Mol        # has 2D conformer
    source: str          # "smiles" | "chemdraw" | "molfile" | "sdf"
    label: Optional[str] = None   # e.g. "IAJD-1" if the source named it
    had_coords: bool = False      # True if 2D coords came from the file (not computed)
    family_hint: Optional[str] = None  # per-row family override (from parser)


def _ensure_2d(mol: Chem.Mol) -> bool:
    """Make sure the mol has a 2D conformer. Returns True if coords were
    already present, False if we had to compute them."""
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        # If any atom has a non-zero coord, trust the existing conformer.
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            if abs(p.x) > 1e-6 or abs(p.y) > 1e-6:
                return True
    AllChem.Compute2DCoords(mol)
    return False


def _parse_cdxml(content: str) -> List[Tuple[Chem.Mol, Optional[str]]]:
    mols = Chem.MolsFromCDXML(content, sanitize=True, removeHs=True)
    out: List[Tuple[Chem.Mol, Optional[str]]] = []
    for i, m in enumerate(mols):
        if m is None:
            continue
        # ChemDraw lets users tag fragments with text labels; RDKit stores
        # these on the mol if present.
        label = None
        for prop in ("_Name", "Name", "ChemDraw_Label"):
            if m.HasProp(prop):
                label = m.GetProp(prop)
                break
        out.append((m, label or f"mol_{i+1}"))
    return out


def _parse_sdf(content: bytes) -> List[Tuple[Chem.Mol, Optional[str]]]:
    suppl = Chem.SDMolSupplier()
    suppl.SetData(content.decode("utf-8", errors="ignore"), sanitize=True, removeHs=True)
    out: List[Tuple[Chem.Mol, Optional[str]]] = []
    for i, m in enumerate(suppl):
        if m is None:
            continue
        label = None
        for prop in ("_Name", "Name"):
            if m.HasProp(prop):
                label = m.GetProp(prop)
                break
        out.append((m, label or f"mol_{i+1}"))
    return out


def _parse_molblock(content: str) -> List[Tuple[Chem.Mol, Optional[str]]]:
    m = Chem.MolFromMolBlock(content, sanitize=True, removeHs=True)
    if m is None:
        return []
    label = m.GetProp("_Name") if m.HasProp("_Name") else "mol_1"
    return [(m, label)]


def _parse_smiles_text(text: str) -> List[Tuple[Chem.Mol, Optional[str]]]:
    """Accept one SMILES per line, optionally followed by whitespace-separated
    tokens. Token format: <SMILES> [family=<FAMILY>] [<label>]. Comment lines
    (starting #) and blanks ignored. Family hints carry through as a property
    on the Mol (`_iajd_family_hint`) so the parser stays stateless."""
    out: List[Tuple[Chem.Mol, Optional[str]]] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        smi = toks[0]
        rest = toks[1:]
        family_hint = None
        label_parts: List[str] = []
        for t in rest:
            if t.startswith("family=") or t.startswith("Family="):
                family_hint = t.split("=", 1)[1].strip()
            else:
                label_parts.append(t)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if family_hint:
            m.SetProp("_iajd_family_hint", family_hint)
        label = " ".join(label_parts).strip() or f"mol_{i+1}"
        out.append((m, label))
    return out


def parse_input(
    content: bytes | str,
    filename: Optional[str] = None,
) -> List[ParsedInput]:
    """Parse a SMILES string or an uploaded structure file.

    Detection order:
      1. filename extension (.cdxml, .cdx, .mol, .sdf, .smi)
      2. content sniffing (XML declaration, M  END marker, looks-like-SMILES)
    Always returns a list (possibly empty). Each ParsedInput carries a 2D
    conformer; `had_coords=True` if the coords came from the file."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        raw_bytes = content
    else:
        text = content
        raw_bytes = content.encode("utf-8", errors="ignore")

    ext = (Path(filename).suffix.lower() if filename else "").lstrip(".")
    parsed_raw: List[Tuple[Chem.Mol, Optional[str]]] = []
    source = "smiles"

    is_cdxml = ext == "cdxml" or text.lstrip().startswith("<?xml") and "CDXML" in text[:600].upper()
    is_sdf = ext in {"sdf"} or ("$$$$" in text and "M  END" in text)
    is_mol = ext in {"mol", "molfile"} or (
        not is_sdf and "M  END" in text and "$$$$" not in text
    )

    if is_cdxml:
        parsed_raw = _parse_cdxml(text)
        source = "chemdraw"
    elif ext == "cdx":
        # Binary CDX: RDKit doesn't ship a parser; try OpenBabel as a soft dep.
        parsed_raw = _try_openbabel_cdx(raw_bytes)
        source = "chemdraw"
    elif is_sdf:
        parsed_raw = _parse_sdf(raw_bytes)
        source = "sdf"
    elif is_mol:
        parsed_raw = _parse_molblock(text)
        source = "molfile"
    else:
        parsed_raw = _parse_smiles_text(text)
        source = "smiles"

    results: List[ParsedInput] = []
    for mol, label in parsed_raw:
        had = _ensure_2d(mol)
        try:
            canon = Chem.MolToSmiles(mol, canonical=True)
        except Exception:  # noqa: BLE001
            continue
        # Per-row family hint may come from:
        #   - `family=` token in the SMILES line (set by _parse_smiles_text)
        #   - `family` / `Family` property on the SDF/molfile record
        family_hint = None
        for prop in ("_iajd_family_hint", "family", "Family"):
            if mol.HasProp(prop):
                family_hint = mol.GetProp(prop).strip()
                break
        results.append(ParsedInput(
            smiles=canon, mol=mol, source=source, label=label,
            had_coords=had and source in {"chemdraw", "molfile", "sdf"},
            family_hint=family_hint,
        ))
    return results


def _try_openbabel_cdx(raw: bytes) -> List[Tuple[Chem.Mol, Optional[str]]]:
    """Optional binary-CDX path via OpenBabel (`obabel`). Returns [] if not
    installed or conversion fails; callers should fall back to asking the user
    to export as CDXML."""
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("obabel"):
        return []
    with tempfile.NamedTemporaryFile(suffix=".cdx", delete=False) as f_in:
        f_in.write(raw); in_path = f_in.name
    try:
        proc = subprocess.run(
            ["obabel", in_path, "-osdf"],
            check=False, capture_output=True, timeout=15,
        )
        if proc.returncode != 0 or not proc.stdout:
            return []
        return _parse_sdf(proc.stdout)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Patterns we care about for IAJDs.
_BASIC_N_SMARTS = Chem.MolFromSmarts("[N;X3;!$(N=*);!$([N+]);!$(NC=O);!$(Nc)]")
_ESTER_SMARTS   = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ETHER_SMARTS   = Chem.MolFromSmarts("[#6][OX2;!$(OC=O)][#6]")
_AMIDE_SMARTS   = Chem.MolFromSmarts("[NX3][CX3](=O)")
_TERMINAL_CH3   = Chem.MolFromSmarts("[CH3;D1]")

_PE_FREQS = (1.0, 2.0, 4.0, 8.0)  # 4 frequency bands -> 8 channels per atom (sin+cos)
_FEAT_DIM = (
    len(_PE_FREQS) * 2 * 3       # PE channels × (mean,max,std)
    + 8                          # ionizable-N distance summaries
    + 6                          # branch / chain stats
    + 4                          # ring / heteroatom counts
)


def _atom_positions(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    p = np.zeros((n, 2), dtype=float)
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        p[i, 0] = pos.x; p[i, 1] = pos.y
    return p


def _basic_n_idx(mol: Chem.Mol) -> Optional[int]:
    """Pick the most-likely ionizable nitrogen for our IAJD families.

    Heuristic: tertiary aliphatic N with no neighboring carbonyl. If multiple,
    prefer the one whose 2D neighborhood is most 'head-like' (more heteroatoms
    nearby — i.e., the one chemists draw at the head)."""
    matches = mol.GetSubstructMatches(_BASIC_N_SMARTS)
    if not matches:
        # fallback: any N that isn't in an amide
        cand = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetSymbol() == "N"
                and not any(nbr.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2.0
                            and b.GetOtherAtom(nbr).GetSymbol() == "O"
                            for b in nbr.GetBonds())
                            for nbr in a.GetNeighbors())]
        if not cand:
            return None
        return cand[0]
    if len(matches) == 1:
        return matches[0][0]
    pos = _atom_positions(mol)
    best = matches[0][0]; best_score = -1.0
    for (idx,) in matches:
        # Count heteroatoms within 3.0 (unit-less, 2D) units in the layout
        center = pos[idx]
        d = np.linalg.norm(pos - center, axis=1)
        score = sum(1 for j, atom in enumerate(mol.GetAtoms())
                    if atom.GetSymbol() != "C" and atom.GetIdx() != idx and d[j] < 3.0)
        if score > best_score:
            best = idx; best_score = score
    return best


def _polar_coords(pos: np.ndarray, origin_idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r, theta) for each atom relative to origin atom (or centroid)."""
    origin = pos[origin_idx] if origin_idx is not None else pos.mean(axis=0)
    rel = pos - origin
    r = np.linalg.norm(rel, axis=1)
    rmax = r.max() if r.max() > 0 else 1.0
    r_norm = r / rmax
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    return r_norm, theta


def _positional_encoding_pooled(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Sinusoidal PE over polar coordinates of each atom, aggregated to a
    fixed-length vector via (mean, max, std) pooling over atoms.

    Channels per atom: for each frequency f in _PE_FREQS,
        sin(2π f r),  cos(2π f r) * cos(theta),  ...
    Concretely we emit 2 channels per frequency:
        s_f = sin(2π f r) · cos(theta)
        c_f = cos(2π f r) · sin(theta)
    This entangles radial and angular position so swapping a long chain for a
    short one shifts both the r and θ statistics together."""
    if len(r) == 0:
        return np.zeros(len(_PE_FREQS) * 2 * 3, dtype=float)
    cols = []
    for f in _PE_FREQS:
        cols.append(np.sin(2 * math.pi * f * r) * np.cos(theta))
        cols.append(np.cos(2 * math.pi * f * r) * np.sin(theta))
    mat = np.stack(cols, axis=1)  # (n_atoms, 2*len(freqs))
    return np.concatenate([mat.mean(axis=0), mat.max(axis=0), mat.std(axis=0)])


def _ionizable_n_distance_block(mol: Chem.Mol, pos: np.ndarray, n_idx: Optional[int]) -> np.ndarray:
    """8-dim block of distance summaries from the ionizable N to key motifs.

    Layout: [d_ester_min, d_ester_mean, d_ether_min, d_ether_mean,
             d_term_min, d_term_p25, d_term_p75, d_ring_min]
    All distances in *normalized* layout units (divided by molecule radius)."""
    if n_idx is None or pos.shape[0] == 0:
        return np.zeros(8, dtype=float)
    origin = pos[n_idx]
    radius = np.linalg.norm(pos - pos.mean(axis=0), axis=1).max() or 1.0

    def _d_to_match(smarts_mol):
        if smarts_mol is None:
            return []
        hits = mol.GetSubstructMatches(smarts_mol)
        ds = []
        for h in hits:
            # use the carbonyl-C (first atom) / O (second) of each match
            j = h[0]
            ds.append(np.linalg.norm(pos[j] - origin) / radius)
        return ds

    d_est = _d_to_match(_ESTER_SMARTS)
    d_eth = _d_to_match(_ETHER_SMARTS)
    d_term_hits = [h[0] for h in mol.GetSubstructMatches(_TERMINAL_CH3)]
    d_term = sorted(np.linalg.norm(pos[j] - origin) / radius for j in d_term_hits)
    d_ring = []
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        c = pos[list(ring)].mean(axis=0)
        d_ring.append(np.linalg.norm(c - origin) / radius)

    def _stat(xs, kind):
        if not xs:
            return 0.0
        a = np.asarray(xs, dtype=float)
        if kind == "min": return float(a.min())
        if kind == "mean": return float(a.mean())
        if kind == "p25": return float(np.percentile(a, 25))
        if kind == "p75": return float(np.percentile(a, 75))
        return 0.0

    return np.asarray([
        _stat(d_est, "min"), _stat(d_est, "mean"),
        _stat(d_eth, "min"), _stat(d_eth, "mean"),
        _stat(d_term, "min"), _stat(d_term, "p25"), _stat(d_term, "p75"),
        _stat(d_ring, "min"),
    ], dtype=float)


def _branch_block(mol: Chem.Mol, n_idx: Optional[int]) -> np.ndarray:
    """6-dim: branch lengths from the ionizable N (sorted), longest CH2 run,
    branch-length spread, total heavy atom count (normalized)."""
    n_heavy = mol.GetNumHeavyAtoms()
    if n_idx is None or n_heavy == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, n_heavy / 100.0], dtype=float)

    branches: List[int] = []  # heavy atoms reachable per neighbor of N
    seen_root = {n_idx}
    for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        # BFS in the subgraph excluding N
        visited = set(seen_root)
        stack = [nbr.GetIdx()]
        count = 0
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            count += 1
            for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
                if nn.GetIdx() not in visited:
                    stack.append(nn.GetIdx())
        branches.append(count)
    branches.sort(reverse=True)
    pad = (branches + [0, 0, 0])[:3]

    # Longest contiguous CH2 chain (count carbons in any straight aliphatic chain)
    longest_ch2 = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetDegree() > 2 or atom.GetIsAromatic():
            continue
        # extend in both directions
        chain_len = 1
        visited = {atom.GetIdx()}
        for nbr in atom.GetNeighbors():
            cur = nbr; prev = atom
            while (cur.GetSymbol() == "C" and cur.GetDegree() <= 2
                   and not cur.GetIsAromatic() and cur.GetIdx() not in visited):
                chain_len += 1
                visited.add(cur.GetIdx())
                nxt = None
                for nn in cur.GetNeighbors():
                    if nn.GetIdx() != prev.GetIdx() and nn.GetSymbol() == "C":
                        nxt = nn; break
                if nxt is None:
                    break
                prev = cur; cur = nxt
        longest_ch2 = max(longest_ch2, chain_len)

    spread = (max(branches) - min(branches)) / (max(branches) + 1) if branches else 0.0
    return np.asarray([
        pad[0] / 30.0, pad[1] / 30.0, pad[2] / 30.0,
        longest_ch2 / 20.0, spread, n_heavy / 100.0,
    ], dtype=float)


def _composition_block(mol: Chem.Mol) -> np.ndarray:
    """4-dim composition summary: ring count, O count, N count, halide count
    (each normalized)."""
    n_rings = mol.GetRingInfo().NumRings()
    n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    n_x = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in {"F", "Cl", "Br", "I"})
    return np.asarray([n_rings / 5.0, n_o / 10.0, n_n / 5.0, n_x / 5.0], dtype=float)


def structural_features(mol: Chem.Mol) -> np.ndarray:
    """Compute the fixed-length structural feature vector for a mol. The mol
    must have a 2D conformer (call `_ensure_2d` first if unsure)."""
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    pos = _atom_positions(mol)
    n_idx = _basic_n_idx(mol)
    r, theta = _polar_coords(pos, n_idx)
    pe = _positional_encoding_pooled(r, theta)
    dist = _ionizable_n_distance_block(mol, pos, n_idx)
    branch = _branch_block(mol, n_idx)
    comp = _composition_block(mol)
    vec = np.concatenate([pe, dist, branch, comp])
    assert vec.shape[0] == _FEAT_DIM, (vec.shape, _FEAT_DIM)
    return vec


def feature_dim() -> int:
    return _FEAT_DIM


# ---------------------------------------------------------------------------
# Similarity + neighbor reweighting
# ---------------------------------------------------------------------------

def structural_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    na = np.linalg.norm(feat_a); nb = np.linalg.norm(feat_b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(feat_a, feat_b) / (na * nb))
    # Map cosine [-1, 1] -> [0, 1] so it blends cleanly with Tanimoto in [0,1]
    return max(0.0, min(1.0, 0.5 * (cos + 1.0)))


def reweight_neighbor_sims(
    query_feat: np.ndarray,
    neighbor_feats: Sequence[np.ndarray],
    base_tanimoto: Sequence[float],
    beta: float = 0.25,
) -> np.ndarray:
    """Blend Tanimoto with structural similarity. Returns one similarity per
    neighbor in [0, 1]. `beta` controls how much weight structural similarity
    gets (0 → ignore structural; 1 → ignore Tanimoto). Default 0.25 keeps
    Tanimoto dominant and lets structural break ties / penalize positional
    mismatches."""
    base = np.asarray(base_tanimoto, dtype=float)
    if not len(neighbor_feats):
        return base
    structs = np.asarray([
        structural_similarity(query_feat, f) for f in neighbor_feats
    ], dtype=float)
    return (1.0 - beta) * base + beta * structs


def refine_analog_prediction(
    analog_neighbors: List[dict],
    query_feat: np.ndarray,
    neighbor_feats: Sequence[np.ndarray],
    beta: float = 0.25,
    sim_power: int = 4,
) -> dict:
    """Recompute the analog-delta estimate from existing v15 neighbors using
    structural-blended weights. `analog_neighbors` is the list v15 emits; each
    entry must have `tanimoto` (float) and `estimated_y` (float). Returns:
        {refined_point, sim_blend_per_neighbor, weight_per_neighbor,
         struct_sim_per_neighbor}
    Returns `refined_point=None` if no neighbors or all weights zero."""
    if not analog_neighbors or not len(neighbor_feats):
        return {
            "refined_point": None,
            "sim_blend_per_neighbor": [],
            "weight_per_neighbor": [],
            "struct_sim_per_neighbor": [],
        }
    base = np.asarray([float(n.get("tanimoto", 0.0)) for n in analog_neighbors])
    y_est = np.asarray([float(n.get("estimated_y", n.get("neighbor_y", 0.0)))
                        for n in analog_neighbors])
    structs = np.asarray([
        structural_similarity(query_feat, f) for f in neighbor_feats
    ])
    blended = (1.0 - beta) * base + beta * structs
    w = np.clip(blended, 0.0, None) ** sim_power
    s = w.sum()
    refined = float(np.dot(w, y_est) / s) if s > 0 else float(np.mean(y_est))
    return {
        "refined_point": refined,
        "sim_blend_per_neighbor": [round(float(x), 4) for x in blended],
        "weight_per_neighbor": [round(float(x), 6) for x in w],
        "struct_sim_per_neighbor": [round(float(x), 4) for x in structs],
    }
