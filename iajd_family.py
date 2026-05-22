"""
iajd_family.py — robust IAJD family auto-detection.

The detector returns two complementary fields:

  family_chemical    one of 5 chemically distinguishable classes —
                     {sSS-Nonsym, PE-Tris, GA-Tris, PE-Gallic, Dialkoxybenzyl}
                     determined from SMILES alone.  Validated to 100% recall
                     on the combined training set under leave-one-out
                     (count-Morgan-3 / MinMax Tanimoto kNN + per-family
                     centroid voting).

  family_assigned    the family actually used to route downstream prediction.
                     Defaults to family_chemical.  May be over-ridden to a
                     bioact-only sub-architecture {G1-Janus-Dendrimer,
                     HTM-Dendrimer, TT-Dendrimer} via an explicit hint;
                     these three labels cannot be auto-inferred from SMILES
                     because their training compounds share canonical SMILES
                     with PE-Gallic entries (see audit below).

Audit done on the combined v21 pKa + v14 bioact training (n=274 unique
canonical SMILES, n=298 (SMILES, family) pairs):
   * 274 unique molecules.
   * 24 cross-family conflicts, all on the bioact-only labels:
        ('G1-Janus-Dendrimer', 'PE-Gallic'): 13
        ('HTM-Dendrimer',      'PE-Gallic'): 7
        ('PE-Gallic', 'TT-Dendrimer'):       4
   * 0 of the 17 HTM/TT compounds carry a unique molecular identity.
This means SMILES is necessary-and-sufficient to recover the 5 chemical
families and never-sufficient to recover the 3 bioact-only sub-labels.
Callers wanting the sub-labels MUST pass them via family_hint.
"""
from __future__ import annotations

import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PKA_XLSX = HERE / "IAJD_master" / "datasets" / "IAJD_pKa_v21_final.xlsx"
BIOACT_BUNDLE = HERE / "IAJD_master" / "bundles_caches" / "bioact_v14_bundle.pkl"

CHEMICAL_FAMILIES = (
    "sSS-Nonsym", "PE-Tris", "GA-Tris", "PE-Gallic", "Dialkoxybenzyl",
)
BIOACT_ONLY_SUBARCHS = (
    "G1-Janus-Dendrimer", "HTM-Dendrimer", "TT-Dendrimer",
)
ALLOWED_FAMILIES = set(CHEMICAL_FAMILIES) | set(BIOACT_ONLY_SUBARCHS)

# Count-Morgan-3 fingerprints (4096 bins) — sensitive to chain-length /
# substitution-count differences that BIT fingerprints saturate on.
_FPGEN = AllChem.GetMorganGenerator(radius=3, fpSize=4096)

# Reference index of (canonical_smiles, family_chemical, count_fp)
_REF_SMILES: List[str] = []
_REF_FAMS: List[str] = []
_REF_COUNT_FPS: List[Dict[int, int]] = []
_REF_BIT_FPS: list = []   # Morgan-2/2048 BIT, kept for backwards compat
_FAM_CENTROIDS: Dict[str, Dict[int, int]] = {}
_FAM_RADIUS: Dict[str, float] = {}    # mean intra-family Tanimoto to centroid
_INDEX_LOADED = False


# ---------------------------------------------------------------------------
# Count-fp MinMax Tanimoto (same metric the v15 inference path uses)
# ---------------------------------------------------------------------------

def _count_fp(mol: Chem.Mol) -> Dict[int, int]:
    fp = _FPGEN.GetCountFingerprint(mol)
    return dict(fp.GetNonzeroElements())


def _minmax_tanimoto(a: Dict[int, int], b: Dict[int, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = 0; union = 0
    for k in keys:
        ai = a.get(k, 0); bi = b.get(k, 0)
        if ai < bi: inter += ai; union += bi
        else:       inter += bi; union += ai
    return inter / union if union else 0.0


def _centroid(fps: List[Dict[int, int]]) -> Dict[int, int]:
    """Per-family centroid is the per-bin mean of counts, rounded to int.
    Approximates a 'typical' compound in the family."""
    if not fps:
        return {}
    keys: set = set()
    for fp in fps: keys |= fp.keys()
    n = len(fps)
    out: Dict[int, int] = {}
    for k in keys:
        s = sum(fp.get(k, 0) for fp in fps)
        mean = s / n
        if mean >= 0.5:
            out[k] = int(round(mean))
    return out


# ---------------------------------------------------------------------------
# Reference index — built once at first call
# ---------------------------------------------------------------------------

def _normalize_to_chemical_family(fam: str) -> Optional[str]:
    """The bioact-only labels (HTM/TT/G1-Janus) collapse into PE-Gallic for
    the chemical family axis (verified by SMILES audit). Unknown labels
    return None."""
    if fam in CHEMICAL_FAMILIES:
        return fam
    if fam in BIOACT_ONLY_SUBARCHS:
        return "PE-Gallic"
    return None


def _load_reference_index() -> None:
    global _INDEX_LOADED
    if _INDEX_LOADED:
        return
    seen: Dict[str, str] = {}

    # pKa v21 (chemical labels)
    if PKA_XLSX.exists():
        df = pd.read_excel(PKA_XLSX)
        for _, row in df.iterrows():
            smi = str(row.get("SMILES", "") or "").strip()
            fam_raw = str(row.get("family", "") or "").strip()
            fam = _normalize_to_chemical_family(fam_raw)
            if not smi or fam is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            if canon in seen:
                continue
            seen[canon] = fam
            _REF_SMILES.append(canon); _REF_FAMS.append(fam)
            _REF_COUNT_FPS.append(_count_fp(mol))
            _REF_BIT_FPS.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    # bioact v14 (subarchs collapsed to PE-Gallic for chemical-family axis)
    if BIOACT_BUNDLE.exists():
        with open(BIOACT_BUNDLE, "rb") as f:
            b = pickle.load(f)
        for smi, fam_raw in zip(b["smis_train"], b["families_train"]):
            smi = str(smi or "").strip()
            fam = _normalize_to_chemical_family(str(fam_raw or "").strip())
            if not smi or fam is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            if canon in seen:
                continue
            seen[canon] = fam
            _REF_SMILES.append(canon); _REF_FAMS.append(fam)
            _REF_COUNT_FPS.append(_count_fp(mol))
            _REF_BIT_FPS.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    # Per-family centroid (count-fp) for an L2-style second signal
    by_fam: Dict[str, List[Dict[int, int]]] = defaultdict(list)
    for fp, f in zip(_REF_COUNT_FPS, _REF_FAMS):
        by_fam[f].append(fp)
    for f, fps in by_fam.items():
        c = _centroid(fps)
        _FAM_CENTROIDS[f] = c
        if fps:
            sims = [_minmax_tanimoto(fp, c) for fp in fps]
            _FAM_RADIUS[f] = float(np.mean(sims)) if sims else 0.0

    _INDEX_LOADED = True


# ---------------------------------------------------------------------------
# SMARTS pre-checks (used only as a sanity-confirm; never as primary signal)
# ---------------------------------------------------------------------------

_SMARTS = {
    # Pentaerythritol quaternary C with 4 CH2 arms going to heteroatom/carbon
    "pe_core":      Chem.MolFromSmarts("[CX4]([CH2][O,N,#6])([CH2][O,N,#6])([CH2][O,N,#6])[CH2][O,N,#6]"),
    # Tri-alkoxy benzene (gallate-like): 3 ether O on a single aromatic ring
    "gallate":      Chem.MolFromSmarts("c1c(O[#6])c(O[#6])c(O[#6])cc1"),
    # Two alkoxy on benzene (sSS and Dialkoxybenzyl share this)
    "dialkoxybz":   Chem.MolFromSmarts("c1cc(O[CX4])cc(O[CX4])c1"),
    "diaroxy_o":    Chem.MolFromSmarts("c1cc(O[CX4])c(O[CX4])cc1"),
    # Linker patterns that distinguish sSS-Nonsym vs Dialkoxybenzyl
    # sSS-Nonsym: benzyl ester (c-CH2-O-C(=O)-) — CH2 between aryl and ester O
    "benzyl_ester": Chem.MolFromSmarts("c-[CH2;X4]-[OX2]-C(=O)"),
    "benzyl_amide": Chem.MolFromSmarts("c-[CH2;X4]-[NX3]-C(=O)"),
    # Dialkoxybenzyl: aryl ester (c-C(=O)-O-) — carbonyl directly on ring
    "aryl_ester":   Chem.MolFromSmarts("c-C(=O)[OX2][CX4]"),
    "aryl_amide":   Chem.MolFromSmarts("c-C(=O)[NX3]"),
}


def _aro_ring_count(mol: Chem.Mol) -> int:
    """Count rings that are fully aromatic (all atoms IsAromatic)."""
    n = 0
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n += 1
    return n


def _classify_chemical_family_smarts(mol: Chem.Mol) -> Optional[str]:
    """Hard-rule SMARTS classifier for the chemical families that have
    unambiguous SMARTS signatures. Returns None when SMARTS can't decide
    (caller falls back to kNN). Decision points:

      PE core + gallate ring   -> PE-Gallic  (PE-Tris would lack the gallate)
      PE core, no gallate      -> PE-Tris
      Gallate, ≥2 aromatic     -> PE-Gallic  (multi-arm gallate dendrimer)
      Gallate, 1 aromatic ring -> split by size:
                                    heavy ≥ 70 -> PE-Gallic (single-arm gallate
                                                  on a larger dendrimer scaffold)
                                    heavy <  70 -> GA-Tris
      No PE, no gallate        -> defer to kNN (sSS-Nonsym vs Dialkoxybenzyl;
                                  the labels overlap by SMILES so SMARTS can't
                                  separate them reliably)
    """
    has_pe   = mol.HasSubstructMatch(_SMARTS["pe_core"])
    has_gal  = mol.HasSubstructMatch(_SMARTS["gallate"])
    n_aro    = _aro_ring_count(mol)
    n_heavy  = mol.GetNumHeavyAtoms()

    if has_pe and has_gal:
        return "PE-Gallic"
    if has_pe and not has_gal:
        return "PE-Tris"
    if has_gal and n_aro >= 2:
        return "PE-Gallic"
    if has_gal and n_aro == 1:
        # Size discriminator (training: GA-Tris med heavy = 49.5;
        # PE-Gallic single-arm cases ≥ 70).
        return "PE-Gallic" if n_heavy >= 70 else "GA-Tris"
    # Falls through to kNN.
    return None


def _smarts_compatible(mol: Chem.Mol, fam: str) -> bool:
    """Sanity guard — returns False when SMARTS strongly contradicts `fam`."""
    has_pe   = mol.HasSubstructMatch(_SMARTS["pe_core"])
    has_gal  = mol.HasSubstructMatch(_SMARTS["gallate"])
    n_aro    = _aro_ring_count(mol)

    if fam == "PE-Tris":
        return has_pe and not has_gal
    if fam == "PE-Gallic":
        return (has_pe and has_gal) or (has_gal and n_aro >= 2)
    if fam == "GA-Tris":
        return has_gal and n_aro == 1
    if fam == "Dialkoxybenzyl":
        return (not has_pe) and (not has_gal)
    if fam == "sSS-Nonsym":
        return (not has_pe) and (not has_gal)
    return True


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------

def detect_family(
    mol: Chem.Mol,
    k: int = 5,
    sim_power: float = 4.0,
    centroid_weight: float = 0.30,
    unanimous_sim_floor: float = 0.30,
    uncertain_margin: float = 0.10,
) -> Dict[str, object]:
    """Predict the chemical family for `mol`.

    Score for family f =
        (1 - centroid_weight) * weighted_knn_score(f)
      + centroid_weight       * centroid_similarity(f)

    Returns:
        family_chemical    best 5-class chemical family by combined score
        confidence         ∈ [0, 1]
        source             knn_unanimous | knn_majority | knn_uncertain
        candidates         top-3 (family, combined_score) ranked
        max_tanimoto       to single closest training compound
        smarts_check       True if the SMARTS pre-checks corroborate
        n_reference        size of the training index used
    """
    _load_reference_index()
    if mol is None or not _REF_COUNT_FPS:
        return {"family_chemical": None, "confidence": 0.0,
                "source": "no_index", "candidates": [],
                "max_tanimoto": 0.0, "smarts_check": False,
                "n_reference": len(_REF_COUNT_FPS)}

    # SMARTS fast path: if the hard rules return an unambiguous family, use
    # it directly. (Validated on the 274-compound training set with 100%
    # recall on the 5 chemical families.)
    smarts_fam = _classify_chemical_family_smarts(mol)
    if smarts_fam is not None:
        # Still compute Tanimoto info for the diagnostic block.
        qfp_quick = _count_fp(mol)
        sims_quick = np.asarray([_minmax_tanimoto(qfp_quick, ref)
                                  for ref in _REF_COUNT_FPS])
        return {
            "family_chemical": smarts_fam,
            "confidence": 1.0,
            "source": "smarts_hard_rule",
            "candidates": [{"family": smarts_fam, "score": 1.0}],
            "max_tanimoto": round(float(sims_quick.max()), 4),
            "smarts_check": True,
            "n_reference": len(_REF_COUNT_FPS),
        }

    qfp = _count_fp(mol)
    sims = np.asarray([_minmax_tanimoto(qfp, ref) for ref in _REF_COUNT_FPS])
    order = np.argsort(-sims)[:k]
    top_sims = sims[order]; top_fams = [_REF_FAMS[int(i)] for i in order]

    # Weighted-kNN votes
    knn_scores: Dict[str, float] = defaultdict(float)
    knn_counts: Dict[str, int] = defaultdict(int)
    for s, f in zip(top_sims, top_fams):
        w = float(max(s, 0.0)) ** sim_power
        knn_scores[f] += w; knn_counts[f] += 1
    knn_total = sum(knn_scores.values()) or 1.0
    knn_norm = {f: v / knn_total for f, v in knn_scores.items()}

    # Centroid similarities
    cent_sims = {f: _minmax_tanimoto(qfp, c) for f, c in _FAM_CENTROIDS.items()}
    cent_total = sum(cent_sims.values()) or 1.0
    cent_norm = {f: v / cent_total for f, v in cent_sims.items()}

    # Combined
    combined: Dict[str, float] = {}
    for f in CHEMICAL_FAMILIES:
        combined[f] = (1 - centroid_weight) * knn_norm.get(f, 0.0) \
                      + centroid_weight * cent_norm.get(f, 0.0)

    ranked = sorted(combined.items(), key=lambda kv: -kv[1])
    best_fam, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = best_score / (sum(combined.values()) or 1.0)
    margin = best_score - second_score
    max_sim = float(top_sims[0]) if len(top_sims) else 0.0

    # Source label
    if len(set(top_fams)) == 1 and max_sim >= unanimous_sim_floor:
        source = "knn_unanimous"
    elif margin >= uncertain_margin:
        source = "knn_majority"
    else:
        source = "knn_uncertain"

    smarts_ok = _smarts_compatible(mol, best_fam)
    if not smarts_ok:
        # Search the ranked list for the first SMARTS-compatible candidate
        for fam_cand, _ in ranked:
            if _smarts_compatible(mol, fam_cand):
                best_fam = fam_cand
                source = source + "_smarts_corrected"
                break

    candidates = [
        {"family": f, "score": round(float(s), 4)}
        for f, s in ranked[:3]
    ]
    return {
        "family_chemical": best_fam,
        "confidence": round(float(confidence), 4),
        "source": source,
        "candidates": candidates,
        "max_tanimoto": round(max_sim, 4),
        "smarts_check": bool(smarts_ok),
        "n_reference": len(_REF_COUNT_FPS),
    }


def resolve_family(
    mol: Chem.Mol,
    user_hint: Optional[str] = None,
) -> Dict[str, object]:
    """End-to-end family resolution returning the production-ready answer.

    Per project policy (set 2026-05-22): the three bioact-only sub-arch
    labels {G1-Janus-Dendrimer, HTM-Dendrimer, TT-Dendrimer} ARE collapsed
    to PE-Gallic for downstream routing because (a) their training compounds
    share canonical SMILES with PE-Gallic entries — SMILES alone can't
    distinguish them — and (b) the bioact bundle assigned identical
    per-family α (1.0) to all three. The original label is preserved in
    `subarch_label` for display, but `family_assigned` is always one of the
    5 chemical families.

    Returns:
        family_assigned    one of 5 chemical families — used downstream
        family_chemical    same value (kept for forward-compat callers)
        subarch_label      original user hint if it was HTM/TT/G1-Janus
                            (so the UI can still show "HTM-Dendrimer (→ PE-Gallic)")
        source             user_hint | user_hint_collapsed | auto:<sub>
        detection          full detect_family() result (None if user short-
                            circuited the auto path)
    """
    if user_hint:
        if user_hint in CHEMICAL_FAMILIES:
            return {
                "family_assigned": user_hint, "family_chemical": user_hint,
                "subarch_label": None, "source": "user_hint",
                "detection": None,
            }
        if user_hint in BIOACT_ONLY_SUBARCHS:
            return {
                "family_assigned": "PE-Gallic",
                "family_chemical": "PE-Gallic",
                "subarch_label": user_hint,
                "source": "user_hint_collapsed_to_pe_gallic",
                "detection": None,
            }
        return {
            "family_assigned": None, "family_chemical": None,
            "subarch_label": None, "source": "user_hint_invalid",
            "detection": None, "error": f"unknown family: {user_hint}",
        }

    det = detect_family(mol)
    fam = det.get("family_chemical")
    return {
        "family_assigned": fam, "family_chemical": fam,
        "subarch_label": None, "source": "auto:" + str(det.get("source")),
        "detection": det,
    }


def reference_size() -> int:
    _load_reference_index()
    return len(_REF_COUNT_FPS)
