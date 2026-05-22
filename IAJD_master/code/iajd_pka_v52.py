"""
iajd_pka_v51.py — IAJD pKa Prediction System (v5.1)

Selective 3D conformer features. Adds three 3D descriptors that were guessed
to carry signal independent of the 2D feature set:
    - Pct_V_Bur_max  (buried volume around tertiary N, max)
    - Pct_V_Bur_mean (buried volume around tertiary N, mean)
    - Asphericity_3D (molecular shape asymmetry)

Excluded from v5.0:
    - E_min_3D (r=0.92 with BertzCT)
    - N_basic_N_3D (r=0.82 with NumNitrogens)
    - Rg_3D (r=0.69 with LabuteASA)
    - N_LowE_Conformers (weak signal)

Total features: 26 (v4.1) + 3 = 29.

Top-level API:
    from iajd_pka_v51 import build_bundle, predict_pka, loo_cv
"""
from __future__ import annotations

import re
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Lipinski, GraphDescriptors, MolSurf
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
_warnings.filterwarnings("ignore", category=UserWarning)
_warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MEASUREMENT_SD = 0.051  # triplicate noise floor (spec §1, §5.2)
SD_RELIABILITY_THRESHOLD = 0.10
N_PAIRS_RELIABLE = 5
CORRELATION_CORRECTION = 1.15

FAMILY_PI_90 = {
    "PE-Tris": 0.13,
    "GA-Tris": 0.13,
    "sSS-Nonsym": 0.31,
    "PE-Gallic": 0.45,
    "Dialkoxybenzyl": 0.68,
    "NOVEL": 0.68,
}

DIMENSIONS = ["core_position", "linker_length", "head_group", "linkage", "chain_symmetry"]

# The 29 engineered features (26 from v4 + 3 selective 3D features)
FEATURE_NAMES = [
    # 2D block (indices 0-25)
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "AromRings", "BertzCT",
    "Chi0v", "Chi1v", "Chi0n", "Chi1n", "LabuteASA",
    "GasteigerN_min", "GasteigerN_max", "GasteigerN_sum",
    "linker_length", "chain_min", "chain_max", "chain_diff", "chain_sum",
    "is_HPRZ", "is_MPRZ", "is_H2EPRZ", "is_DMBA", "is_ester",
    # Selective 3D block (indices 26-28)
    "Pct_V_Bur_max", "Pct_V_Bur_mean", "Asphericity_3D", "MolGpKa_pred",
]
N_FEATURES = 30
FEATURE_3D_COLS = ["Pct_V_Bur_max", "Pct_V_Bur_mean", "Asphericity_3D"]

MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# -----------------------------------------------------------------------------
# Section 2.4 / 3.3: chain pair parsing from architecture string
# -----------------------------------------------------------------------------

def parse_chain_pair(arch_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse (chain_min, chain_max) from an IAJD architecture label.

    Handles all conventions in v20: Cn/Cm, Cn, EH (ethylhexyl=C8), C8br, dmN,
    3Cn prefix, and C(n) per spec §2.4. Returns (None, None) if unparseable.
    """
    if arch_str is None or (isinstance(arch_str, float) and np.isnan(arch_str)):
        return (None, None)
    s = str(arch_str)
    # Cn/Cm (nonsymmetric sSS)
    m = re.search(r"C(\d+)\s*/\s*C(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 2 < a <= 22 and 2 < b <= 22:
            return (min(a, b), max(a, b))
    # Ethylhexyl / branched C8
    if re.search(r"\bEH\b", s) or "C8br" in s:
        return (8, 8)
    # dmN
    m = re.search(r"\bdm(\d+)\b", s, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 2 < n <= 22:
            return (n, n)
    # C(n) per spec
    nums = [int(x) for x in re.findall(r"C\((\d+)\)", s) if 2 < int(x) <= 22]
    if nums:
        return (min(nums), max(nums))
    # Generic Cn at token boundary (avoid matching head-group IDs like DMBA13
    # or linker markers like 4C). Requires C immediately followed by digits,
    # preceded by dot/dash/start, followed by dot/dash/end.
    cands = re.findall(r"(?:^|[.\-])\d*C(\d+)(?=[.\-]|$)", s)
    filtered = [int(n) for n in cands if 2 < int(n) <= 22]
    if filtered:
        return (min(filtered), max(filtered))
    return (None, None)


# -----------------------------------------------------------------------------
# Section 2: token system and arch_coarse
# -----------------------------------------------------------------------------

@dataclass
class TokenRecord:
    family: str
    core_position: str  # "35", "34", or "NA"
    linker_length: Optional[int]  # 2..5 or None
    head_group: str  # HPRZ, MPRZ, H2EPRZ, DMBA, PIP, DMA, NOVEL
    linkage: str  # ester, amide, MIXED, NA
    chain_min: Optional[int]
    chain_max: Optional[int]
    arch_coarse: str = ""

    @property
    def chain_symmetry(self) -> str:
        if self.chain_min is None or self.chain_max is None:
            return "NA"
        return "symmetric" if self.chain_min == self.chain_max else "nonsymmetric"

    @property
    def chain_pair(self) -> Tuple[Optional[int], Optional[int]]:
        return (self.chain_min, self.chain_max)


def tokens_from_row(row: pd.Series) -> TokenRecord:
    """Build a TokenRecord from a v20 dataset row (training data)."""
    fam = row["family"]
    arch = row.get("architecture", "") or ""
    # core position
    if "sSS" in arch or "sSS-Nonsym" in fam:
        core = "35" if "-35-" in arch else ("34" if "-34-" in arch else "35")
    elif "Dialkoxybenz" in arch:
        core = "34" if "-34-" in arch else ("35" if "-35-" in arch else "NA")
    elif "GA-tris" in arch or "GA-Tris" in fam:
        core = "345"
    else:
        core = "NA"

    ll = row.get("linker_length", None)
    try:
        ll = int(ll) if ll is not None and not pd.isna(ll) else None
    except (TypeError, ValueError):
        ll = None

    head = str(row.get("head_group", "NOVEL") or "NOVEL")
    link_raw = str(row.get("linkage", "ester") or "ester")
    # Normalize linkage: ester_benzoate, ester_benzyl -> ester; amide_benzyl -> amide
    if link_raw.startswith("ester"):
        linkage = "ester"
    elif link_raw.startswith("amide"):
        linkage = "amide"
    else:
        linkage = "NA"

    cmin, cmax = parse_chain_pair(arch)

    tok = TokenRecord(
        family=fam,
        core_position=core,
        linker_length=ll,
        head_group=head,
        linkage=linkage,
        chain_min=cmin,
        chain_max=cmax,
    )
    tok.arch_coarse = compute_arch_coarse(tok)
    return tok


def compute_arch_coarse(t: TokenRecord) -> str:
    """Build the arch_coarse label per §2.2."""
    abbrev = {
        "sSS-Nonsym": "sSS",
        "PE-Gallic": "PEg",
        "PE-Tris": "PEt",
        "GA-Tris": "GAt",
        "Dialkoxybenzyl": "DAB",
    }.get(t.family, "NOV")
    parts = [abbrev]
    if t.core_position not in ("NA", "", None):
        parts.append(str(t.core_position))
    parts.append(f"{t.linker_length}C" if t.linker_length is not None else "NAC")
    parts.append(str(t.head_group))
    parts.append(str(t.linkage))
    return "-".join(parts)


def token_distance(a: TokenRecord, b: TokenRecord) -> float:
    """Token distance per §2.3. inf if family differs."""
    if a.family != b.family:
        return float("inf")
    d = 0
    if a.core_position != b.core_position:
        d += 1
    if a.linker_length != b.linker_length:
        d += 1
    if a.head_group != b.head_group:
        d += 1
    if a.linkage != b.linkage:
        d += 1
    if a.chain_symmetry != b.chain_symmetry:
        d += 1
    return d


# -----------------------------------------------------------------------------
# Section 3: SMILES validation and token extraction (for novel queries)
# -----------------------------------------------------------------------------

HEAD_SMARTS = {
    # order matters: check H2EPRZ before HPRZ, HPRZ before MPRZ
    "H2EPRZ": "N1(CCO)CCN(CCO)CC1",
    "HPRZ":   "N1CCN(CCO)CC1",
    "MPRZ":   "N1CCN(C)CC1",
    "DMBA":   "c1ccccc1CN(C)C",
    "PIP":    "N1CCCCC1",
    "DMA":    "[CX4][N]([CH3])[CH3]",
}

FAMILY_SMARTS = {
    # evaluation order: sSS, GA-Tris, PE-Tris, PE-Gallic, DAB
    "sSS-Nonsym_acyl2": "c1cc(OC(=O))cc(OC(=O))c1",
    "GA-Tris_acyl3":    "c1cc(OC(=O))c(OC(=O))c(OC(=O))c1",
    "PE-Tris_core":     "[CX4]([CH2][OX2]C(=O))([CH2][OX2]C(=O))([CH2][OX2]C(=O))[CH2]",
    "PE-Gallic_gallate": "c1cc(OC(=O))c(OC(=O))c(OC(=O))c1C(=O)OC",
    "DAB_35": "c1cc(O[CX4])cc(O[CX4])c1",
    "DAB_34": "c1cc(O[CX4])c(O[CX4])cc1",
}


@dataclass
class ValidationResult:
    ok: bool
    mol: Optional[Chem.Mol]
    canonical_smiles: Optional[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_smiles(smiles: str, maldi_mw: Optional[float] = None) -> ValidationResult:
    """Five-stage SMILES validator per §3.1."""
    vr = ValidationResult(ok=False, mol=None, canonical_smiles=None)
    # Stage 1: parse
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        vr.errors.append("SMILES_UNPARSEABLE")
        return vr
    # Stage 2: MW
    if maldi_mw is not None:
        mw_calc = Descriptors.ExactMolWt(mol)
        if abs(mw_calc - maldi_mw) > 1.0:
            vr.errors.append(f"MW_MISMATCH: computed {mw_calc:.4f} vs MALDI {maldi_mw:.4f}")
            return vr
    # Stage 4 (before 3): tertiary amine check
    patt = Chem.MolFromSmarts("[NX3;H0;!$(N=*);!$(N-O)]")
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        vr.errors.append("NO_IONIZABLE_AMINE")
        return vr
    # Stage 3: ring census (soft)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_ali = rdMolDescriptors.CalcNumAliphaticRings(mol)
    if n_ali == 0 and n_arom == 0:
        vr.warnings.append("RING_COUNT_ATYPICAL")
    # Stage 5: canonical + macrocycle
    canon = Chem.MolToSmiles(mol, canonical=True)
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    if any(sz >= 12 for sz in ring_sizes):
        vr.warnings.append("MACROCYCLE_PRESENT")
    vr.ok = True
    vr.mol = mol
    vr.canonical_smiles = canon
    return vr


def classify_family_from_smiles(mol: Chem.Mol) -> str:
    """Ordered SMARTS decision tree per §3.2. Returns family name or 'NOVEL'."""
    # Count acyl-oxy on aromatic ring
    acyloxy = Chem.MolFromSmarts("c-O-C(=O)")
    n_acyloxy = len(mol.GetSubstructMatches(acyloxy))
    # Pentaerythritol quaternary carbon (4 CH2 neighbors)
    pe_core = Chem.MolFromSmarts("[CX4]([CH2])([CH2])([CH2])[CH2]")
    has_pe = mol.HasSubstructMatch(pe_core)
    # Trihydroxy/triacyl gallate ring
    trihydroxy_ring = Chem.MolFromSmarts("c1c(O)c(O)c(O)cc1")
    triacyl_ring = Chem.MolFromSmarts("c1c(O[#6])c(O[#6])c(O[#6])cc1")
    has_gallate = mol.HasSubstructMatch(triacyl_ring) or mol.HasSubstructMatch(trihydroxy_ring)

    # sSS vs GA-Tris: distinguish by number of acyl-oxy on a single aromatic ring
    if n_acyloxy >= 2 and has_gallate and not has_pe:
        return "GA-Tris" if n_acyloxy >= 3 else "sSS-Nonsym"
    if has_pe and has_gallate:
        return "PE-Gallic"
    if has_pe and not has_gallate:
        return "PE-Tris"
    # Dialkoxybenzyl: two alkoxy on benzene, no acyl-oxy, no gallate
    dab = Chem.MolFromSmarts("c1cc(O[CX4])cc(O[CX4])c1")
    dab2 = Chem.MolFromSmarts("c1cc(O[CX4])c(O[CX4])cc1")
    if mol.HasSubstructMatch(dab) or mol.HasSubstructMatch(dab2):
        return "Dialkoxybenzyl"
    return "NOVEL"


def extract_head_group(mol: Chem.Mol) -> str:
    for name, sm in HEAD_SMARTS.items():
        patt = Chem.MolFromSmarts(sm)
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "NOVEL"


def extract_linkage(mol: Chem.Mol) -> str:
    ester = Chem.MolFromSmarts("[#6X3](=O)[OX2][#6]")
    amide = Chem.MolFromSmarts("[#6X3](=O)[NX3][#6]")
    n_e = len(mol.GetSubstructMatches(ester))
    n_a = len(mol.GetSubstructMatches(amide))
    if n_e >= 2 * max(1, n_a):
        return "ester"
    if n_a >= 2 * max(1, n_e):
        return "amide"
    return "MIXED"


def extract_linker_length(mol: Chem.Mol) -> Optional[int]:
    """Walk -O-C(=O)-CH2-...-N, count sp3 carbons between carbonyl and amine.

    The IAJD ester linker is structured as:
        [core]-O-C(=O)-(CH2)_n-N[amine]
    Walking from the carbonyl C (idx 0 of the SMARTS match) along its sp3 C
    neighbor toward the amine N gives the linker length n.
    """
    # Match the carbonyl C itself, not the alkoxy O
    patt = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    for match in mol.GetSubstructMatches(patt):
        carbonyl_c = mol.GetAtomWithIdx(match[0])
        # Walk away from the carbonyl C through its sp3 C neighbor
        for nbr in carbonyl_c.GetNeighbors():
            if (nbr.GetSymbol() == "C" and
                    nbr.GetHybridization() == Chem.HybridizationType.SP3):
                length = _walk_to_nitrogen(mol, carbonyl_c.GetIdx(), nbr.GetIdx(),
                                             depth=1)
                if length is not None and 2 <= length <= 6:
                    return length
    return None


def _walk_to_nitrogen(mol: Chem.Mol, prev_idx: int, cur_idx: int, depth: int = 0) -> Optional[int]:
    if depth > 8:
        return None
    cur = mol.GetAtomWithIdx(cur_idx)
    if cur.GetSymbol() == "N":
        return depth
    if cur.GetSymbol() != "C" or cur.GetHybridization() != Chem.HybridizationType.SP3:
        return None
    # continue to the one sp3 C or N neighbor (excluding where we came from)
    for nbr in cur.GetNeighbors():
        if nbr.GetIdx() == prev_idx:
            continue
        if nbr.GetSymbol() == "N":
            return depth + 1
        if nbr.GetSymbol() == "C" and nbr.GetHybridization() == Chem.HybridizationType.SP3 and nbr.GetDegree() <= 3:
            res = _walk_to_nitrogen(mol, cur_idx, nbr.GetIdx(), depth + 1)
            if res is not None:
                return res
    return None


def _infer_chain_lengths_from_smarts(mol: Chem.Mol) -> Tuple[Optional[int], Optional[int]]:
    """Infer (chain_min, chain_max) from molecular structure when architecture
    label isn't available. Walks from terminal CH3 atoms back through linear
    CH2 chains and counts the run length. Only returns chains of length >=4
    (filters out methyl groups on rings/quaternary carbons)."""
    chains: List[int] = []
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == 'C' and atom.GetDegree() == 1
                and atom.GetTotalNumHs() == 3):
            n = 1
            prev = atom
            neighbors = list(atom.GetNeighbors())
            if not neighbors:
                continue
            cur = neighbors[0]
            while True:
                if (cur.GetSymbol() == 'C' and cur.GetTotalNumHs() == 2
                        and cur.GetDegree() == 2 and not cur.GetIsAromatic()):
                    n += 1
                    nxt = [nbr for nbr in cur.GetNeighbors()
                           if nbr.GetIdx() != prev.GetIdx()]
                    if not nxt:
                        break
                    prev, cur = cur, nxt[0]
                else:
                    break
            if n >= 4:
                chains.append(n)
    if not chains:
        return (None, None)
    return (min(chains), max(chains))


def tokens_from_mol(mol: Chem.Mol, architecture: Optional[str] = None,
                    family_hint: Optional[str] = None) -> TokenRecord:
    fam = family_hint or classify_family_from_smiles(mol)
    head = extract_head_group(mol)
    linkage = extract_linkage(mol)
    ll = extract_linker_length(mol)
    core = "35"  # sensible default
    if fam == "GA-Tris":
        core = "345"
    elif fam in ("PE-Tris", "PE-Gallic"):
        core = "NA"
    cmin, cmax = (None, None)
    if architecture:
        cmin, cmax = parse_chain_pair(architecture)
    if cmin is None or cmax is None:
        # Fallback: infer from molecular structure
        cmin, cmax = _infer_chain_lengths_from_smarts(mol)
    tok = TokenRecord(fam, core, ll, head, linkage, cmin, cmax)
    tok.arch_coarse = compute_arch_coarse(tok)
    return tok


# -----------------------------------------------------------------------------
# Selective 3D feature computation (3 features)
# -----------------------------------------------------------------------------

def compute_3d_features_from_mol(mol: Chem.Mol,
                                 n_confs: int = 5,
                                 random_seed: int = 42) -> Dict[str, float]:
    """Generate 3D conformers and compute the three selective 3D descriptors:
    Pct_V_Bur_max, Pct_V_Bur_mean, Asphericity_3D.

    Uses ETKDGv3 + MMFF94 (UFF fallback). Returns NaN for any feature that
    fails to compute.
    """
    result = {"Pct_V_Bur_max": np.nan, "Pct_V_Bur_mean": np.nan,
              "Asphericity_3D": np.nan}
    try:
        molH = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.numThreads = 1
        params.useRandomCoords = True  # FIX: large IAJDs need random-coord init
        conf_ids = AllChem.EmbedMultipleConfs(molH, numConfs=n_confs, params=params)
        if len(conf_ids) == 0:
            # Last-resort retry: single conformer with maximum permissiveness
            molH = Chem.AddHs(mol)
            cid = AllChem.EmbedMolecule(molH, randomSeed=random_seed,
                                          useRandomCoords=True, maxAttempts=500)
            if cid < 0:
                return result
            conf_ids = [cid]
        # MMFF94 minimize each conformer
        energies: List[Tuple[int, float]] = []
        try:
            mp = AllChem.MMFFGetMoleculeProperties(molH)
        except Exception:
            mp = None
        for cid in conf_ids:
            try:
                ff = (AllChem.MMFFGetMoleculeForceField(molH, mp, confId=cid)
                      if mp is not None else None)
                if ff is None:
                    ff = AllChem.UFFGetMoleculeForceField(molH, confId=cid)
                if ff is None:
                    continue
                ff.Minimize(maxIts=200)
                energies.append((cid, float(ff.CalcEnergy())))
            except Exception:
                continue
        if not energies:
            return result
        energies.sort(key=lambda x: x[1])
        best_cid = energies[0][0]
        # Asphericity from minimum-energy conformer
        result["Asphericity_3D"] = float(Descriptors3D.Asphericity(molH, confId=best_cid))
        # Buried volume around tertiary nitrogens
        conf = molH.GetConformer(best_cid)
        heavy_atoms = [a for a in molH.GetAtoms() if a.GetAtomicNum() > 1]
        heavy_coords = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in heavy_atoms])
        pct_vs: List[float] = []
        for atom in molH.GetAtoms():
            if atom.GetAtomicNum() != 7:
                continue
            if atom.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if atom.GetIsAromatic():
                continue
            # Skip amide N
            is_amide = False
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 6:
                    for bb in nb.GetBonds():
                        if (bb.GetBondType() == Chem.BondType.DOUBLE
                                and bb.GetOtherAtom(nb).GetAtomicNum() == 8):
                            is_amide = True
                            break
                if is_amide:
                    break
            if is_amide:
                continue
            pos = np.array(list(conf.GetAtomPosition(atom.GetIdx())))
            d = np.linalg.norm(heavy_coords - pos, axis=1)
            near = np.sum((d > 0.01) & (d <= 4.0))
            pct = 100.0 * near / max(1, len(heavy_atoms) - 1)
            pct_vs.append(pct)
        if pct_vs:
            result["Pct_V_Bur_max"] = float(max(pct_vs))
            result["Pct_V_Bur_mean"] = float(np.mean(pct_vs))
    except Exception:
        pass
    return result


# -----------------------------------------------------------------------------
# 29-feature computation
# -----------------------------------------------------------------------------

def compute_features(mol: Chem.Mol, tokens: TokenRecord,
                     features_3d: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Compute the 29 engineered features (26 2D + 3 selective 3D)."""
    feats = np.zeros(N_FEATURES, dtype=float)
    try:
        feats[0] = Descriptors.MolWt(mol)
        feats[1] = Descriptors.MolLogP(mol)
        feats[2] = Descriptors.TPSA(mol)
        feats[3] = Lipinski.NumHDonors(mol)
        feats[4] = Lipinski.NumHAcceptors(mol)
        feats[5] = Lipinski.NumRotatableBonds(mol)
        feats[6] = Lipinski.NumAromaticRings(mol)
        feats[7] = GraphDescriptors.BertzCT(mol)
        feats[8] = GraphDescriptors.Chi0v(mol)
        feats[9] = GraphDescriptors.Chi1v(mol)
        feats[10] = GraphDescriptors.Chi0n(mol)
        feats[11] = GraphDescriptors.Chi1n(mol)
        feats[12] = MolSurf.LabuteASA(mol)
    except Exception:
        pass
    try:
        m2 = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(m2)
        tert_n_charges = []
        for atom in m2.GetAtoms():
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 3:
                try:
                    q = float(atom.GetProp("_GasteigerCharge"))
                    if np.isfinite(q):
                        tert_n_charges.append(q)
                except KeyError:
                    pass
        if tert_n_charges:
            feats[13] = min(tert_n_charges)
            feats[14] = max(tert_n_charges)
            feats[15] = sum(tert_n_charges)
    except Exception:
        pass
    feats[16] = tokens.linker_length if tokens.linker_length is not None else np.nan
    feats[17] = tokens.chain_min if tokens.chain_min is not None else np.nan
    feats[18] = tokens.chain_max if tokens.chain_max is not None else np.nan
    if tokens.chain_min is not None and tokens.chain_max is not None:
        feats[19] = tokens.chain_max - tokens.chain_min
        feats[20] = tokens.chain_max + tokens.chain_min
    else:
        feats[19] = np.nan
        feats[20] = np.nan
    feats[21] = 1.0 if tokens.head_group == "HPRZ" else 0.0
    feats[22] = 1.0 if tokens.head_group == "MPRZ" else 0.0
    feats[23] = 1.0 if tokens.head_group == "H2EPRZ" else 0.0
    feats[24] = 1.0 if tokens.head_group == "DMBA" else 0.0
    feats[25] = 1.0 if tokens.linkage == "ester" else 0.0
    # 3D block (indices 26-28)
    if features_3d is None:
        features_3d = compute_3d_features_from_mol(mol)
    for i, col in enumerate(FEATURE_3D_COLS):
        val = features_3d.get(col, np.nan)
        feats[26 + i] = val if val is not None and np.isfinite(val) else np.nan
    # MolGpKa prediction (index 29)
    feats[29] = _MOLGPKA_CACHE.get(Chem.MolToSmiles(mol, canonical=True), np.nan)
    return feats


# Global cache mapping canonical SMILES -> MolGpKa basic pKa prediction
_MOLGPKA_CACHE: Dict[str, float] = {}


def load_molgpka_cache(npy_path: str, smiles_list: List[str]) -> None:
    """Populate the MolGpKa cache from a .npy file of predictions aligned to a SMILES list."""
    global _MOLGPKA_CACHE
    preds = np.load(npy_path)
    for sm, p in zip(smiles_list, preds):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if np.isfinite(p):
            _MOLGPKA_CACHE[canon] = float(p)


def impute_features(X: np.ndarray, families: List[str]) -> np.ndarray:
    """NaN imputation: family median first, then global median."""
    X = X.copy()
    fam_arr = np.asarray(families)
    for col in range(X.shape[1]):
        col_vals = X[:, col]
        for fam in np.unique(fam_arr):
            mask = fam_arr == fam
            sub = col_vals[mask]
            finite = sub[np.isfinite(sub)]
            if len(finite) > 0:
                med = np.median(finite)
                nan_mask = mask & ~np.isfinite(col_vals)
                X[nan_mask, col] = med
        # global fallback
        col_vals = X[:, col]
        if not np.all(np.isfinite(col_vals)):
            finite = col_vals[np.isfinite(col_vals)]
            if len(finite) > 0:
                X[~np.isfinite(col_vals), col] = np.median(finite)
            else:
                X[~np.isfinite(col_vals), col] = 0.0
    return X


# -----------------------------------------------------------------------------
# Section 5: Delta table (matched pairs)
# -----------------------------------------------------------------------------

@dataclass
class DeltaRow:
    delta_id: str
    family: str
    dimension: str
    value_from: str
    value_to: str
    condition: str
    delta_mean: float
    delta_se: float
    delta_sd: float
    n_pairs: int
    reliable: bool
    notes: str = ""


def extract_matched_pairs(tokens: List[TokenRecord], pkas: List[float],
                          dimension: str) -> List[Dict[str, Any]]:
    """Find all matched pairs differing only on `dimension`."""
    other_dims = [d for d in DIMENSIONS if d != dimension]
    groups: Dict[tuple, List[int]] = {}
    for i, t in enumerate(tokens):
        if t.family in (None, "NOVEL"):
            continue
        key = (t.family,) + tuple(
            getattr(t, d) if d != "chain_symmetry" else t.chain_symmetry
            for d in other_dims
        )
        groups.setdefault(key, []).append(i)
    pairs = []
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ti, tj = tokens[idxs[i]], tokens[idxs[j]]
                vi = getattr(ti, dimension) if dimension != "chain_symmetry" else ti.chain_symmetry
                vj = getattr(tj, dimension) if dimension != "chain_symmetry" else tj.chain_symmetry
                if vi == vj:
                    continue
                pairs.append({
                    "family": ti.family,
                    "value_from": vi,
                    "value_to": vj,
                    "delta": pkas[idxs[j]] - pkas[idxs[i]],
                    "condition": key,
                    "i": idxs[i],
                    "j": idxs[j],
                })
    return pairs


def build_delta_table(tokens: List[TokenRecord], pkas: List[float]) -> pd.DataFrame:
    """Populate the delta table per §5.4 with linker-specific head deltas and
    pooled linker slopes."""
    rows: List[DeltaRow] = []

    for dim in ["linker_length", "head_group", "linkage"]:
        pairs = extract_matched_pairs(tokens, pkas, dim)
        # Group by (family, value_from, value_to)
        grouped: Dict[tuple, List[float]] = {}
        for p in pairs:
            # Normalize direction: sort value_from<value_to lexicographically for head/linkage
            vf, vt, delta = p["value_from"], p["value_to"], p["delta"]
            if dim == "linker_length":
                # normalize so larger minus smaller (positive delta = longer linker)
                if vf > vt:
                    vf, vt, delta = vt, vf, -delta
            else:
                if str(vf) > str(vt):
                    vf, vt, delta = vt, vf, -delta
            key = (p["family"], vf, vt)
            grouped.setdefault(key, []).append(delta)

        for (fam, vf, vt), vals in grouped.items():
            vals = np.array(vals)
            n = len(vals)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n >= 2 else float("nan")
            se = sd / np.sqrt(n) if n >= 2 else float("nan")
            reliable = (n >= N_PAIRS_RELIABLE) and (not np.isnan(sd)) and (sd <= SD_RELIABILITY_THRESHOLD)
            rows.append(DeltaRow(
                delta_id=f"{fam}-{dim}-{vf}-{vt}",
                family=fam, dimension=dim,
                value_from=str(vf), value_to=str(vt),
                condition="pooled",
                delta_mean=mean, delta_se=se, delta_sd=sd,
                n_pairs=n, reliable=bool(reliable),
                notes="",
            ))

    # Linker-specific head-group deltas (for PE-Tris interaction handling)
    for fam in ["PE-Tris"]:
        fam_idx = [i for i, t in enumerate(tokens) if t.family == fam]
        by_key = {}
        for i in fam_idx:
            t = tokens[i]
            k = (t.linker_length, t.head_group)
            by_key.setdefault(k, []).append(pkas[i])
        # for each linker length, compute HPRZ->H2EPRZ delta
        llset = sorted({t.linker_length for t in tokens if t.family == fam and t.linker_length is not None})
        for ll in llset:
            hprz = by_key.get((ll, "HPRZ"), [])
            h2e = by_key.get((ll, "H2EPRZ"), [])
            if hprz and h2e:
                d = float(np.mean(h2e) - np.mean(hprz))
                rows.append(DeltaRow(
                    delta_id=f"{fam}-head-HPRZ-H2EPRZ-L{ll}",
                    family=fam, dimension="head_group",
                    value_from="HPRZ", value_to="H2EPRZ",
                    condition=f"linker_length={ll}",
                    delta_mean=d, delta_se=float("nan"), delta_sd=float("nan"),
                    n_pairs=1, reliable=False,
                    notes=f"linker-specific at {ll}C",
                ))

    return pd.DataFrame([r.__dict__ for r in rows])


def lookup_delta(delta_df: pd.DataFrame, family: str, dimension: str,
                 value_from: str, value_to: str,
                 condition: Optional[str] = None) -> Tuple[Optional[float], Optional[float], int, bool]:
    """Returns (delta, se, n_pairs, reliable). Handles sign via normalization."""
    if delta_df is None or delta_df.empty:
        return None, None, 0, False
    sign = 1.0
    vf, vt = value_from, value_to
    if dimension == "linker_length":
        try:
            if float(value_from) > float(value_to):
                vf, vt, sign = value_to, value_from, -1.0
        except (ValueError, TypeError):
            pass
    else:
        if str(value_from) > str(value_to):
            vf, vt, sign = value_to, value_from, -1.0

    candidates = delta_df[
        (delta_df.family == family)
        & (delta_df.dimension == dimension)
        & (delta_df.value_from == str(vf))
        & (delta_df.value_to == str(vt))
    ]
    if condition is not None:
        cond_match = candidates[candidates.condition == condition]
        if not cond_match.empty:
            row = cond_match.iloc[0]
            return sign * float(row.delta_mean), float(row.delta_se), int(row.n_pairs), bool(row.reliable)
    pooled = candidates[candidates.condition == "pooled"]
    if not pooled.empty:
        row = pooled.iloc[0]
        return sign * float(row.delta_mean), float(row.delta_se), int(row.n_pairs), bool(row.reliable)
    return None, None, 0, False


# -----------------------------------------------------------------------------
# Section 5.6: Chain pair lookup
# -----------------------------------------------------------------------------

def build_chain_pair_table(tokens: List[TokenRecord], pkas: List[float]) -> Dict[tuple, tuple]:
    """Key: (family, head_group, chain_min, chain_max) -> (mean, sd, n)."""
    buckets: Dict[tuple, List[float]] = {}
    for t, pka in zip(tokens, pkas):
        if t.chain_min is None or t.chain_max is None:
            continue
        key = (t.family, t.head_group, t.chain_min, t.chain_max)
        buckets.setdefault(key, []).append(pka)
    table = {}
    for key, vals in buckets.items():
        arr = np.array(vals)
        sd = float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")
        table[key] = (float(np.mean(arr)), sd, len(arr))
    return table


def chain_pair_lookup(query_pair: Tuple[int, int], family: str, head_group: str,
                      table: Dict[tuple, tuple]) -> Tuple[Optional[float], Optional[float], str]:
    """Manhattan k-NN interpolation with expanding radius (§5.6)."""
    candidates = [(key, val) for key, val in table.items()
                  if key[0] == family and key[1] == head_group]
    if not candidates:
        return None, None, "NO_CANDIDATES"
    qmin, qmax = query_pair
    distances = []
    for (f, h, cmin, cmax), (m, s, n) in candidates:
        d = abs(qmin - cmin) + abs(qmax - cmax)
        distances.append((d, m, s, n))
    distances.sort(key=lambda x: x[0])
    chosen: List[tuple] = []
    for max_d in (4, 6, 8):
        neighbors = [x for x in distances if x[0] <= max_d]
        if len(neighbors) >= 3:
            chosen = neighbors[:3]
            break
    if not chosen:
        return None, None, "CHAIN_PAIR_NEIGHBORHOOD_EMPTY"
    weights = np.array([1.0 / (d + 0.5) for d, _, _, _ in chosen])
    weights = weights / weights.sum()
    means = np.array([m for _, m, _, _ in chosen])
    mean_pKa = float(np.sum(weights * means))
    se = float(np.sqrt(np.sum((weights * (means - mean_pKa)) ** 2)))
    return mean_pKa, se, "OK"


# -----------------------------------------------------------------------------
# Section 6: Family-stratified GPR
# -----------------------------------------------------------------------------

def fit_pure_model_ensemble(X: np.ndarray, y: np.ndarray, scaler: StandardScaler
                            ) -> Tuple[RandomForestRegressor, GradientBoostingRegressor]:
    """Fit pooled RandomForest and GradientBoosting on the full training set
    for the Level 6 fallback. These do not depend on family structure and are
    robust for out-of-distribution queries."""
    Xs = scaler.transform(X)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(Xs, y)
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42,
    )
    gbr.fit(Xs, y)
    return rf, gbr


def predict_level6_pure_model(query_features: np.ndarray, query_fp,
                              bundle: Bundle) -> Tuple[float, float, Dict[str, float]]:
    """Pure-modeling fallback ensemble for queries with no reliable analog.

    Combines four orthogonal predictors:
        1. Pooled GPR (kernel method)
        2. Pooled Random Forest (tree ensemble)
        3. Pooled Gradient Boosting (boosted trees)
        4. Tanimoto-weighted k-NN over the full training set (instance-based)

    Returns (mean_prediction, ensemble_std, individual_components).
    The ensemble_std is the empirical standard deviation across the four
    component predictions and serves as a query-local uncertainty estimate.
    """
    Xs = bundle.pooled_scaler.transform(query_features.reshape(1, -1))

    # 1. Pooled GPR
    gpr_p = float(bundle.pooled_gpr.predict(Xs)[0])

    # 2. Pooled RandomForest
    if bundle.pooled_rf is not None:
        rf_p = float(bundle.pooled_rf.predict(Xs)[0])
    else:
        rf_p = gpr_p

    # 3. Pooled Gradient Boosting
    if bundle.pooled_gbr is not None:
        gbr_p = float(bundle.pooled_gbr.predict(Xs)[0])
    else:
        gbr_p = gpr_p

    # 4. Tanimoto-weighted k-NN over the entire training set
    sims = np.array([TanimotoSimilarity(query_fp, fp) for fp in bundle.fps])
    k = 5
    top_idx = np.argsort(-sims)[:k]
    top_sims = sims[top_idx]
    top_pkas = np.array([bundle.pkas[i] for i in top_idx])
    # Distance-weighted average; add small epsilon so identical molecules don't div0
    weights = top_sims + 1e-3
    weights = weights / weights.sum()
    knn_p = float(np.sum(weights * top_pkas))

    components = {"gpr": gpr_p, "rf": rf_p, "gbr": gbr_p, "knn": knn_p}
    vals = np.array(list(components.values()))
    mean_pred = float(np.mean(vals))
    ensemble_std = float(np.std(vals, ddof=1))
    return mean_pred, ensemble_std, components


def make_kernel():
    return (ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=0.005, noise_level_bounds=(0.003, 0.05)))


def fit_family_gprs(X: np.ndarray, y: np.ndarray, families: List[str],
                    n_restarts: int = 5
                    ) -> Tuple[Dict[str, GaussianProcessRegressor], Dict[str, StandardScaler],
                               GaussianProcessRegressor, StandardScaler]:
    fam_arr = np.asarray(families)
    family_gpr: Dict[str, GaussianProcessRegressor] = {}
    family_scaler: Dict[str, StandardScaler] = {}
    sorted_fams = sorted(set(families))
    for idx, fam in enumerate(sorted_fams):
        mask = fam_arr == fam
        if mask.sum() < 10:
            continue
        Xf = X[mask]
        yf = y[mask]
        scaler = StandardScaler().fit(Xf)
        gpr = GaussianProcessRegressor(
            kernel=make_kernel(),
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=n_restarts,
            random_state=42 + idx,
            normalize_y=True,
        )
        gpr.fit(scaler.transform(Xf), yf)
        family_gpr[fam] = gpr
        family_scaler[fam] = scaler

    # Pooled fallback (all data)
    pooled_scaler = StandardScaler().fit(X)
    pooled_gpr = GaussianProcessRegressor(
        kernel=make_kernel(), optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=n_restarts, random_state=42, normalize_y=True,
    )
    pooled_gpr.fit(pooled_scaler.transform(X), y)
    return family_gpr, family_scaler, pooled_gpr, pooled_scaler


def predict_gpr(family: str, x_query: np.ndarray, bundle: "Bundle") -> Tuple[float, float]:
    """Returns (pKa, std). Falls back to pooled GPR if family-specific unavailable."""
    if family in bundle.family_gpr:
        xs = bundle.family_scaler[family].transform(x_query.reshape(1, -1))
        mean, std = bundle.family_gpr[family].predict(xs, return_std=True)
    else:
        xs = bundle.pooled_scaler.transform(x_query.reshape(1, -1))
        mean, std = bundle.pooled_gpr.predict(xs, return_std=True)
    return float(mean[0]), float(std[0])


# -----------------------------------------------------------------------------
# Section 7: Blending
# -----------------------------------------------------------------------------

def compute_weights(level: int, reliable_delta: bool, family: str, n_arch: int,
                    sigma_gpr: float) -> Dict[str, float]:
    if level == 1:
        return {"anchor": 1.0, "delta": 0.0, "gpr": 0.0}
    if level == 2:
        if family == "sSS-Nonsym":
            w = {"anchor": 0.55, "delta": 0.0, "gpr": 0.45}
        else:
            w = {"anchor": 0.60, "delta": 0.0, "gpr": 0.40}
    elif level == 3:
        if family in ("PE-Tris", "GA-Tris") and n_arch >= 1:
            w = {"anchor": 0.70, "delta": 0.0, "gpr": 0.30}
        elif reliable_delta:
            w = {"anchor": 0.40, "delta": 0.40, "gpr": 0.20}
        else:
            w = {"anchor": 0.50, "delta": 0.10, "gpr": 0.40}
    elif level == 4:
        if family in ("PE-Tris", "GA-Tris") and n_arch >= 1:
            w = {"anchor": 0.70, "delta": 0.0, "gpr": 0.30}
        elif reliable_delta:
            w = {"anchor": 0.30, "delta": 0.50, "gpr": 0.20}
        else:
            w = {"anchor": 0.50, "delta": 0.10, "gpr": 0.40}
    else:  # Level 5
        return {"anchor": 0.0, "delta": 0.0, "gpr": 1.0}
    if family == "PE-Gallic":
        w_anchor = min(n_arch / 8.0, 0.75) if n_arch > 0 else 0.0
        w = {"anchor": w_anchor, "delta": 0.0, "gpr": 1.0 - w_anchor}
    if family == "Dialkoxybenzyl":
        return {"anchor": 0.0, "delta": 0.0, "gpr": 1.0}
    # sigma_gpr modifier
    if sigma_gpr > 0.15:
        reduce = 0.10
        w["gpr"] = max(0.0, w["gpr"] - reduce)
        if w["delta"] > 0:
            w["anchor"] += reduce / 2
            w["delta"] += reduce / 2
        else:
            w["anchor"] += reduce
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


# -----------------------------------------------------------------------------
# Section 4: Five-level hierarchy dispatcher
# -----------------------------------------------------------------------------

@dataclass
class LevelResult:
    level: int
    anchor_idx: Optional[int]
    anchor_pKa: Optional[float]
    anchor_sd: Optional[float]
    n_arch: int
    delta_applied: bool
    delta_dim: Optional[str]
    delta_value: Optional[float]
    delta_se: Optional[float]
    delta_n_pairs: Optional[int]
    delta_reliable: Optional[bool]
    chain_lookup_pKa: Optional[float]
    chain_lookup_se: Optional[float]
    top1_tanimoto: float
    top3: List[Dict[str, Any]]
    warnings: List[str]


def compute_tanimoto(query_fp, train_fps: List) -> np.ndarray:
    return np.array([TanimotoSimilarity(query_fp, fp) for fp in train_fps])


def select_level(query_tokens: TokenRecord, query_fp, query_canon: str,
                 bundle: "Bundle") -> LevelResult:
    warns: List[str] = []
    train_tokens = bundle.tokens
    train_pkas = bundle.pkas
    train_fps = bundle.fps
    train_canon = bundle.canonical_smiles

    tanimotos = compute_tanimoto(query_fp, train_fps)
    order = np.argsort(-tanimotos)
    top3 = [{"IAJD": bundle.ids[i], "family": train_tokens[i].family,
             "pKa": train_pkas[i], "tanimoto": float(tanimotos[i])} for i in order[:3]]
    # Same-family top1
    same_fam_idx = [i for i, t in enumerate(train_tokens) if t.family == query_tokens.family]
    top1_tanimoto = float(max([tanimotos[i] for i in same_fam_idx], default=0.0))

    # Level 1: canonical SMILES match
    if query_canon:
        exact = [i for i, cs in enumerate(train_canon) if cs == query_canon]
        if exact:
            anchor_pka = float(np.mean([train_pkas[i] for i in exact]))
            anchor_sd = float(np.std([train_pkas[i] for i in exact], ddof=1)) if len(exact) > 1 else 0.0
            return LevelResult(
                level=1, anchor_idx=exact[0], anchor_pKa=anchor_pka,
                anchor_sd=max(anchor_sd, MEASUREMENT_SD),
                n_arch=len(exact),
                delta_applied=False, delta_dim=None, delta_value=None,
                delta_se=None, delta_n_pairs=None, delta_reliable=None,
                chain_lookup_pKa=None, chain_lookup_se=None,
                top1_tanimoto=1.0, top3=top3, warnings=warns,
            )

    # arch_coarse match
    arch_key = query_tokens.arch_coarse
    same_arch = [i for i, t in enumerate(train_tokens) if t.arch_coarse == arch_key]
    n_arch = len(same_arch)

    # Level 2: sSS-Nonsym same arch, chain pair lookup
    if (query_tokens.family == "sSS-Nonsym"
            and n_arch >= 1
            and query_tokens.chain_min is not None):
        # chain pair lookup; if query pair is in training (level 2 still fires),
        # the lookup returns the neighbor-weighted estimate
        cpl, cpse, status = chain_pair_lookup(
            (query_tokens.chain_min, query_tokens.chain_max),
            query_tokens.family, query_tokens.head_group, bundle.chain_pair_table,
        )
        if status == "OK":
            anchor_mean = float(np.mean([train_pkas[i] for i in same_arch]))
            anchor_sd = float(np.std([train_pkas[i] for i in same_arch], ddof=1)) if n_arch >= 2 else MEASUREMENT_SD
            return LevelResult(
                level=2, anchor_idx=same_arch[0],
                anchor_pKa=anchor_mean, anchor_sd=anchor_sd,
                n_arch=n_arch,
                delta_applied=False, delta_dim=None, delta_value=None,
                delta_se=None, delta_n_pairs=None, delta_reliable=None,
                chain_lookup_pKa=cpl, chain_lookup_se=cpse,
                top1_tanimoto=top1_tanimoto, top3=top3, warnings=warns,
            )
        else:
            warns.append(status)

    # For PE-Tris / GA-Tris / PE-Gallic: if arch_coarse matches, use arch_mean as Level-1-style anchor
    if n_arch >= 1 and query_tokens.family in ("PE-Tris", "GA-Tris", "PE-Gallic"):
        anchor_mean = float(np.mean([train_pkas[i] for i in same_arch]))
        anchor_sd = float(np.std([train_pkas[i] for i in same_arch], ddof=1)) if n_arch >= 2 else MEASUREMENT_SD
        anchor_sd = max(anchor_sd, MEASUREMENT_SD)
        return LevelResult(
            level=2, anchor_idx=same_arch[0],
            anchor_pKa=anchor_mean, anchor_sd=anchor_sd,
            n_arch=n_arch,
            delta_applied=False, delta_dim=None, delta_value=None,
            delta_se=None, delta_n_pairs=None, delta_reliable=None,
            chain_lookup_pKa=None, chain_lookup_se=None,
            top1_tanimoto=top1_tanimoto, top3=top3, warnings=warns,
        )

    # Level 3 / 4: same family, token distance 1 or 2
    same_fam = [(i, train_tokens[i]) for i in same_fam_idx]
    if same_fam:
        distances = [(i, token_distance(query_tokens, t)) for i, t in same_fam]
        distances = [(i, d) for i, d in distances if np.isfinite(d)]
        distances.sort(key=lambda x: x[1])
        if distances:
            min_d = distances[0][1]
            if min_d == 1 and top1_tanimoto >= 0.55:
                return _level3(query_tokens, distances, bundle, top1_tanimoto, top3, warns, n_arch)
            if min_d == 2 and top1_tanimoto >= 0.40:
                return _level4(query_tokens, distances, bundle, top1_tanimoto, top3, warns, n_arch)

    # Level 5 fallback
    return LevelResult(
        level=5, anchor_idx=None, anchor_pKa=None, anchor_sd=None,
        n_arch=n_arch,
        delta_applied=False, delta_dim=None, delta_value=None,
        delta_se=None, delta_n_pairs=None, delta_reliable=None,
        chain_lookup_pKa=None, chain_lookup_se=None,
        top1_tanimoto=top1_tanimoto, top3=top3, warnings=warns,
    )


def _pick_anchor_group(query_tokens: TokenRecord, differing_dim: str,
                      bundle: "Bundle") -> Tuple[Optional[float], Optional[float], int, Optional[int]]:
    """Anchor = mean of training compounds in same family sharing all tokens
    except `differing_dim`, fixed to the anchor value (not query's)."""
    idxs = []
    for i, t in enumerate(bundle.tokens):
        if t.family != query_tokens.family:
            continue
        mismatch = 0
        for dim in DIMENSIONS:
            qv = getattr(query_tokens, dim) if dim != "chain_symmetry" else query_tokens.chain_symmetry
            tv = getattr(t, dim) if dim != "chain_symmetry" else t.chain_symmetry
            if dim == differing_dim:
                if qv == tv:
                    mismatch += 99  # force mismatch: we want the DIFFERENT token
                continue
            if qv != tv:
                mismatch += 1
        if mismatch == 0:
            idxs.append(i)
    if not idxs:
        return None, None, 0, None
    vals = np.array([bundle.pkas[i] for i in idxs])
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else MEASUREMENT_SD
    return mean, max(sd, MEASUREMENT_SD), len(idxs), idxs[0]


def _differing_dims(a: TokenRecord, b: TokenRecord) -> List[str]:
    diff = []
    for d in DIMENSIONS:
        va = getattr(a, d) if d != "chain_symmetry" else a.chain_symmetry
        vb = getattr(b, d) if d != "chain_symmetry" else b.chain_symmetry
        if va != vb:
            diff.append(d)
    return diff


def _level3(query_tokens: TokenRecord, distances: List[Tuple[int, float]],
            bundle: "Bundle", top1: float, top3: List[Dict], warns: List[str],
            n_arch: int) -> LevelResult:
    nearest_idx = distances[0][0]
    anchor_tok = bundle.tokens[nearest_idx]
    diffs = _differing_dims(query_tokens, anchor_tok)
    if len(diffs) != 1:
        return LevelResult(level=5, anchor_idx=None, anchor_pKa=None, anchor_sd=None,
                           n_arch=n_arch, delta_applied=False, delta_dim=None,
                           delta_value=None, delta_se=None, delta_n_pairs=None,
                           delta_reliable=None, chain_lookup_pKa=None, chain_lookup_se=None,
                           top1_tanimoto=top1, top3=top3, warnings=warns)
    dim = diffs[0]
    q_val = getattr(query_tokens, dim) if dim != "chain_symmetry" else query_tokens.chain_symmetry
    a_val = getattr(anchor_tok, dim) if dim != "chain_symmetry" else anchor_tok.chain_symmetry
    anchor_mean, anchor_sd, nn, aidx = _pick_anchor_group(query_tokens, dim, bundle)
    if anchor_mean is None:
        anchor_mean = bundle.pkas[nearest_idx]
        anchor_sd = MEASUREMENT_SD
        nn = 1
        aidx = nearest_idx
    delta, delta_se, n_pairs, reliable = lookup_delta(
        bundle.delta_df, query_tokens.family, dim, str(a_val), str(q_val))
    if delta is None:
        warns.append("DELTA_UNRELIABLE")
        return LevelResult(level=5, anchor_idx=None, anchor_pKa=None, anchor_sd=None,
                           n_arch=n_arch, delta_applied=False, delta_dim=None,
                           delta_value=None, delta_se=None, delta_n_pairs=None,
                           delta_reliable=None, chain_lookup_pKa=None, chain_lookup_se=None,
                           top1_tanimoto=top1, top3=top3, warnings=warns)
    if not reliable:
        warns.append("HIGH_UNCERTAINTY")
    return LevelResult(
        level=3, anchor_idx=aidx, anchor_pKa=anchor_mean, anchor_sd=anchor_sd,
        n_arch=nn, delta_applied=True, delta_dim=dim, delta_value=delta,
        delta_se=delta_se, delta_n_pairs=n_pairs, delta_reliable=reliable,
        chain_lookup_pKa=None, chain_lookup_se=None,
        top1_tanimoto=top1, top3=top3, warnings=warns,
    )


def _level4(query_tokens: TokenRecord, distances: List[Tuple[int, float]],
            bundle: "Bundle", top1: float, top3: List[Dict], warns: List[str],
            n_arch: int) -> LevelResult:
    nearest_idx = distances[0][0]
    anchor_tok = bundle.tokens[nearest_idx]
    diffs = _differing_dims(query_tokens, anchor_tok)
    if len(diffs) != 2:
        return LevelResult(level=5, anchor_idx=None, anchor_pKa=None, anchor_sd=None,
                           n_arch=n_arch, delta_applied=False, delta_dim=None,
                           delta_value=None, delta_se=None, delta_n_pairs=None,
                           delta_reliable=None, chain_lookup_pKa=None, chain_lookup_se=None,
                           top1_tanimoto=top1, top3=top3, warnings=warns)
    # Anchor: full nearest training compound's pKa
    anchor_pKa = bundle.pkas[nearest_idx]
    anchor_sd = MEASUREMENT_SD
    total_delta = 0.0
    se_sq = 0.0
    n_pairs_total = 0
    any_reliable = True
    for dim in diffs:
        qv = getattr(query_tokens, dim) if dim != "chain_symmetry" else query_tokens.chain_symmetry
        av = getattr(anchor_tok, dim) if dim != "chain_symmetry" else anchor_tok.chain_symmetry
        # Special case: PE-Tris head+linker interaction
        if (query_tokens.family == "PE-Tris" and dim == "head_group"
                and "linker_length" in diffs and query_tokens.linker_length is not None):
            # Use linker-specific head delta at the QUERY linker length
            cond = f"linker_length={query_tokens.linker_length}"
            d, se, n, rel = lookup_delta(bundle.delta_df, "PE-Tris", "head_group",
                                         str(av), str(qv), condition=cond)
            if d is None:
                d, se, n, rel = lookup_delta(bundle.delta_df, "PE-Tris", "head_group",
                                             str(av), str(qv))
        else:
            d, se, n, rel = lookup_delta(bundle.delta_df, query_tokens.family, dim,
                                         str(av), str(qv))
        if d is None:
            warns.append(f"DELTA_UNRELIABLE:{dim}")
            any_reliable = False
            continue
        if not rel:
            any_reliable = False
        total_delta += d
        if se is not None and np.isfinite(se):
            se_sq += se ** 2
        n_pairs_total += n
    if not np.isfinite(total_delta):
        return LevelResult(level=5, anchor_idx=None, anchor_pKa=None, anchor_sd=None,
                           n_arch=n_arch, delta_applied=False, delta_dim=None,
                           delta_value=None, delta_se=None, delta_n_pairs=None,
                           delta_reliable=None, chain_lookup_pKa=None, chain_lookup_se=None,
                           top1_tanimoto=top1, top3=top3, warnings=warns)
    return LevelResult(
        level=4, anchor_idx=nearest_idx, anchor_pKa=anchor_pKa, anchor_sd=anchor_sd,
        n_arch=1, delta_applied=True, delta_dim="+".join(diffs),
        delta_value=total_delta, delta_se=float(np.sqrt(se_sq)) if se_sq > 0 else None,
        delta_n_pairs=n_pairs_total, delta_reliable=any_reliable,
        chain_lookup_pKa=None, chain_lookup_se=None,
        top1_tanimoto=top1, top3=top3, warnings=warns,
    )


# -----------------------------------------------------------------------------
# Bundle (model artifact)
# -----------------------------------------------------------------------------

@dataclass
class Bundle:
    version: str
    ids: List[Any]
    tokens: List[TokenRecord]
    pkas: List[float]
    canonical_smiles: List[str]
    fps: List
    features: np.ndarray
    families: List[str]
    family_gpr: Dict[str, GaussianProcessRegressor]
    family_scaler: Dict[str, StandardScaler]
    pooled_gpr: GaussianProcessRegressor
    pooled_scaler: StandardScaler
    delta_df: pd.DataFrame
    chain_pair_table: Dict[tuple, tuple]
    # Pure-modeling ensemble components (Level 6 fallback)
    pooled_rf: Optional[RandomForestRegressor] = None
    pooled_gbr: Optional[GradientBoostingRegressor] = None
    conformal_q: Dict[str, Dict[float, float]] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)


def build_bundle(xlsx_path: str, verbose: bool = True) -> Bundle:
    """Build the full model bundle from the v20 Excel dataset."""
    df = pd.read_excel(xlsx_path, sheet_name="Dataset")
    if verbose:
        print(f"[build_bundle] Loaded {len(df)} rows")

    # Load MolGpKa predictions cache (precomputed offline)
    import os.path
    molgpka_npy = os.path.join(os.path.dirname(os.path.abspath(xlsx_path)), "molgpka_preds.npy")
    if os.path.exists(molgpka_npy):
        load_molgpka_cache(molgpka_npy, df["SMILES"].tolist())
        if verbose:
            print(f"[build_bundle] Loaded {len(_MOLGPKA_CACHE)} MolGpKa predictions")

    tokens: List[TokenRecord] = []
    canon: List[str] = []
    fps = []
    feats_list = []
    valid_mask = []

    for _, row in df.iterrows():
        sm = row["SMILES"]
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            valid_mask.append(False)
            tokens.append(None)  # type: ignore
            canon.append("")
            fps.append(None)
            feats_list.append(np.full(N_FEATURES, np.nan))
            continue
        tok = tokens_from_row(row)
        tokens.append(tok)
        canon.append(Chem.MolToSmiles(mol, canonical=True))
        fps.append(MFPGEN.GetFingerprint(mol))
        # Pull 3D features from v20 dataset row
        features_3d = {col: float(row[col]) for col in FEATURE_3D_COLS
                       if col in row and not pd.isna(row[col])}
        feats_list.append(compute_features(mol, tok, features_3d=features_3d))
        valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    if verbose:
        print(f"[build_bundle] {valid_mask.sum()}/{len(df)} compounds validated")

    df_v = df[valid_mask].reset_index(drop=True)
    tokens_v = [t for t, v in zip(tokens, valid_mask) if v]
    canon_v = [c for c, v in zip(canon, valid_mask) if v]
    fps_v = [f for f, v in zip(fps, valid_mask) if v]
    feats = np.array([f for f, v in zip(feats_list, valid_mask) if v])

    families_v = [t.family for t in tokens_v]
    feats_imp = impute_features(feats, families_v)

    pkas = df_v["pKa"].astype(float).tolist()
    ids = df_v["IAJD"].tolist()

    if verbose:
        print("[build_bundle] Building delta table...")
    delta_df = build_delta_table(tokens_v, pkas)
    if verbose:
        print(f"[build_bundle] {len(delta_df)} delta entries, "
              f"{int(delta_df.reliable.sum())} reliable")

    if verbose:
        print("[build_bundle] Building chain pair table...")
    cp_table = build_chain_pair_table(tokens_v, pkas)
    if verbose:
        print(f"[build_bundle] {len(cp_table)} chain-pair cells")

    if verbose:
        print("[build_bundle] Fitting family-stratified GPRs...")
    fam_gpr, fam_scaler, pooled_gpr, pooled_scaler = fit_family_gprs(
        feats_imp, np.array(pkas), families_v)
    if verbose:
        print(f"[build_bundle] GPRs fitted for: {list(fam_gpr.keys())}")

    if verbose:
        print("[build_bundle] Fitting pure-modeling fallback ensemble (RF + GBR)...")
    pooled_rf, pooled_gbr = fit_pure_model_ensemble(
        feats_imp, np.array(pkas), pooled_scaler)
    if verbose:
        print("[build_bundle] Pure-model ensemble ready")

    bundle = Bundle(
        version="v5.2",
        ids=ids, tokens=tokens_v, pkas=pkas,
        canonical_smiles=canon_v, fps=fps_v, features=feats_imp,
        families=families_v,
        family_gpr=fam_gpr, family_scaler=fam_scaler,
        pooled_gpr=pooled_gpr, pooled_scaler=pooled_scaler,
        delta_df=delta_df, chain_pair_table=cp_table,
        pooled_rf=pooled_rf, pooled_gbr=pooled_gbr,
    )
    return bundle


# -----------------------------------------------------------------------------
# Section 10: predict_pka top-level API
# -----------------------------------------------------------------------------

def assign_tier(family: str, n_arch: int, top1_tanimoto: float) -> str:
    if top1_tanimoto < 0.40 or family in ("Dialkoxybenzyl", "NOVEL"):
        return "LOW"
    if family in ("PE-Tris", "GA-Tris") and n_arch >= 2:
        return "HIGH"
    if family == "sSS-Nonsym" and top1_tanimoto >= 0.60:
        return "MEDIUM_A"
    if family == "PE-Gallic" and n_arch >= 5:
        return "MEDIUM_B"
    return "LOW"


def predict_pka(smiles: str, bundle: Bundle,
                maldi_mw: Optional[float] = None,
                architecture: Optional[str] = None,
                family_hint: Optional[str] = None,
                correlation_correction: bool = True) -> Dict[str, Any]:
    """Top-level prediction API per §10."""
    out: Dict[str, Any] = {
        "pKa_pred": None,
        "pKa_PI_90_lower": None, "pKa_PI_90_upper": None,
        "pKa_PI_95_lower": None, "pKa_PI_95_upper": None,
        "similarity_level": None,
        "anchor_IAJD": None, "anchor_pKa": None, "anchor_tanimoto": 0.0,
        "delta_applied": False, "delta_dimension": None,
        "delta_value": None, "delta_se": None, "delta_n_pairs": None, "delta_reliable": None,
        "chain_pair_query": None, "chain_pair_lookup_result": None,
        "gpr_pred": None, "gpr_std": None,
        "blend_weights": None,
        "confidence_tier": None,
        "domain_flag": None,
        "family_assigned": None, "arch_coarse_assigned": None,
        "n_same_arch": 0, "top1_tanimoto": 0.0, "top3_neighbors": [],
        "warnings": [],
        "pure_model_components": None,
    }

    # Validate SMILES
    vr = validate_smiles(smiles, maldi_mw)
    out["warnings"].extend(vr.warnings)
    if not vr.ok:
        out["warnings"].extend(vr.errors)
        out["domain_flag"] = "INVALID"
        return out

    mol = vr.mol
    q_fp = MFPGEN.GetFingerprint(mol)
    q_canon = vr.canonical_smiles
    q_tokens = tokens_from_mol(mol, architecture=architecture, family_hint=family_hint)
    out["family_assigned"] = q_tokens.family
    out["arch_coarse_assigned"] = q_tokens.arch_coarse
    out["chain_pair_query"] = q_tokens.chain_pair if q_tokens.chain_min else None

    # Compute 3D features on-the-fly for novel queries
    features_3d = compute_3d_features_from_mol(mol)
    if not any(np.isfinite(v) for v in features_3d.values()):
        out["warnings"].append("CONFORMER_EMBED_FAILED")

    q_feats = compute_features(mol, q_tokens, features_3d=features_3d)
    # Impute with family median from bundle
    fam_idx = [i for i, t in enumerate(bundle.tokens) if t.family == q_tokens.family]
    for col in range(len(q_feats)):
        if not np.isfinite(q_feats[col]):
            if fam_idx:
                med = np.nanmedian(bundle.features[fam_idx, col])
            else:
                med = np.nanmedian(bundle.features[:, col])
            q_feats[col] = med if np.isfinite(med) else 0.0

    # GPR prediction
    gpr_pred, gpr_std = predict_gpr(q_tokens.family, q_feats, bundle)
    out["gpr_pred"] = gpr_pred
    out["gpr_std"] = gpr_std

    # Level selection
    lr = select_level(q_tokens, q_fp, q_canon, bundle)
    out["warnings"].extend(lr.warnings)

    # ------------------------------------------------------------------
    # Level 6 contingency: pure-modeling fallback for OOD queries
    # Triggers if the query has no reliable analog anywhere in training:
    #   (a) max Tanimoto over the FULL training set < 0.30, OR
    #   (b) family is NOVEL, OR
    #   (c) Level 5 fired AND family GPR sigma > 0.30 (extreme uncertainty)
    # ------------------------------------------------------------------
    global_top1 = float(max([n["tanimoto"] for n in lr.top3], default=0.0))
    use_level6 = (
        global_top1 < 0.30
        or q_tokens.family in ("NOVEL", None)
        or (lr.level == 5 and gpr_std > 0.30)
    )

    if use_level6:
        l6_pred, l6_std, l6_components = predict_level6_pure_model(q_feats, q_fp, bundle)
        out["warnings"].append("PURE_MODEL_FALLBACK")
        out["similarity_level"] = 6
        out["top1_tanimoto"] = global_top1
        out["top3_neighbors"] = lr.top3
        out["n_same_arch"] = 0
        out["anchor_IAJD"] = None
        out["anchor_pKa"] = None
        out["delta_applied"] = False
        out["delta_dimension"] = None
        out["delta_value"] = None
        out["delta_se"] = None
        out["delta_n_pairs"] = None
        out["delta_reliable"] = None
        out["chain_pair_lookup_result"] = None
        out["gpr_pred"] = l6_components["gpr"]
        out["gpr_std"] = l6_std
        out["blend_weights"] = {"anchor": 0.0, "delta": 0.0, "gpr": 0.0,
                                "level6_ensemble": 1.0}
        out["pure_model_components"] = l6_components
        out["confidence_tier"] = "LOW"
        # PI: use family fallback width doubled, but floor at the ensemble disagreement
        # converted to a 90% half-width (1.645 * sigma) so internal disagreement
        # raises uncertainty above the family prior when components disagree.
        pi90_base = FAMILY_PI_90.get(q_tokens.family, 0.45) * 2.0
        pi90 = max(pi90_base, 1.645 * l6_std)
        pred_clipped = max(4.5, min(8.0, float(l6_pred)))
        if abs(pred_clipped - l6_pred) > 1e-6:
            out["warnings"].append("PKA_OUT_OF_RANGE")
        out["pKa_pred"] = pred_clipped
        out["pKa_PI_90_lower"] = pred_clipped - pi90
        out["pKa_PI_90_upper"] = pred_clipped + pi90
        pi95 = pi90 * 1.645 / 1.282
        out["pKa_PI_95_lower"] = pred_clipped - pi95
        out["pKa_PI_95_upper"] = pred_clipped + pi95
        out["domain_flag"] = "LEVEL6_PURE_MODEL"
        return out

    out["similarity_level"] = lr.level
    out["top1_tanimoto"] = lr.top1_tanimoto
    out["top3_neighbors"] = lr.top3
    out["n_same_arch"] = lr.n_arch
    if lr.anchor_idx is not None:
        out["anchor_IAJD"] = bundle.ids[lr.anchor_idx]
        out["anchor_pKa"] = lr.anchor_pKa
        out["anchor_tanimoto"] = lr.top1_tanimoto
    out["delta_applied"] = lr.delta_applied
    out["delta_dimension"] = lr.delta_dim
    out["delta_value"] = lr.delta_value
    out["delta_se"] = lr.delta_se
    out["delta_n_pairs"] = lr.delta_n_pairs
    out["delta_reliable"] = lr.delta_reliable
    out["chain_pair_lookup_result"] = lr.chain_lookup_pKa

    # Weights
    weights = compute_weights(lr.level, bool(lr.delta_reliable),
                              q_tokens.family, lr.n_arch, gpr_std)
    out["blend_weights"] = weights

    # Assemble the blended prediction
    if lr.level == 1:
        pred = lr.anchor_pKa
        sigma_anchor = lr.anchor_sd or MEASUREMENT_SD
        sigma_delta = 0.0
        sigma_g = gpr_std
        domain_flag = "EXACT_MATCH"
    elif lr.level == 2:
        anchor_val = lr.chain_lookup_pKa if lr.chain_lookup_pKa is not None else lr.anchor_pKa
        pred = (weights["anchor"] * anchor_val
                + weights["delta"] * anchor_val
                + weights["gpr"] * gpr_pred)
        sigma_anchor = lr.chain_lookup_se if lr.chain_lookup_se is not None else (lr.anchor_sd or MEASUREMENT_SD)
        sigma_delta = 0.0
        sigma_g = gpr_std
        domain_flag = "LEVEL2_CHAIN_PAIR"
    elif lr.level == 3:
        pred = (weights["anchor"] * lr.anchor_pKa
                + weights["delta"] * (lr.anchor_pKa + (lr.delta_value or 0.0))
                + weights["gpr"] * gpr_pred)
        sigma_anchor = lr.anchor_sd or MEASUREMENT_SD
        sigma_delta = lr.delta_se if (lr.delta_se and np.isfinite(lr.delta_se)) else 0.05
        sigma_g = gpr_std
        domain_flag = "LEVEL3"
    elif lr.level == 4:
        pred = (weights["anchor"] * lr.anchor_pKa
                + weights["delta"] * (lr.anchor_pKa + (lr.delta_value or 0.0))
                + weights["gpr"] * gpr_pred)
        sigma_anchor = lr.anchor_sd or MEASUREMENT_SD
        sigma_delta = lr.delta_se if (lr.delta_se and np.isfinite(lr.delta_se)) else 0.07
        sigma_g = gpr_std
        domain_flag = "LEVEL4" if lr.delta_reliable else "LEVEL4_DEGRADED"
    else:
        pred = gpr_pred
        sigma_anchor = 0.0
        sigma_delta = 0.0
        sigma_g = gpr_std
        domain_flag = "LEVEL5_GPR"

    # sigma_total
    sig2 = (weights["anchor"] ** 2 * sigma_anchor ** 2
            + weights["delta"] ** 2 * sigma_delta ** 2
            + weights["gpr"] ** 2 * sigma_g ** 2)
    sigma_total = float(np.sqrt(sig2))
    if correlation_correction:
        sigma_total *= CORRELATION_CORRECTION

    # PI via family 90% half-width, with doubling for OUT_OF_DOMAIN
    pi90 = FAMILY_PI_90.get(q_tokens.family, 0.45)
    if lr.level == 1:
        # Exact match: high confidence regardless of family, Tanimoto irrelevant
        tier = "HIGH"
        pi90 = max(MEASUREMENT_SD * 1.645, 0.10)
        out["confidence_tier"] = tier
    else:
        tier = assign_tier(q_tokens.family, lr.n_arch, lr.top1_tanimoto)
        out["confidence_tier"] = tier
        if tier == "LOW" or lr.top1_tanimoto < 0.40:
            pi90 *= 2.0
            out["warnings"].append("EXTRAPOLATION_WARNING")
            domain_flag = "OUT_OF_DOMAIN"

    # Clip prediction to physical range
    pred_clipped = max(4.5, min(8.0, float(pred)))
    if abs(pred_clipped - pred) > 1e-6:
        out["warnings"].append("PKA_OUT_OF_RANGE")
    out["pKa_pred"] = pred_clipped
    out["pKa_PI_90_lower"] = pred_clipped - pi90
    out["pKa_PI_90_upper"] = pred_clipped + pi90
    pi95 = pi90 * 1.645 / 1.282  # scale 90% to 95% using normal quantiles
    out["pKa_PI_95_lower"] = pred_clipped - pi95
    out["pKa_PI_95_upper"] = pred_clipped + pi95
    out["domain_flag"] = domain_flag
    return out


# -----------------------------------------------------------------------------
# Section 9: LOO-CV
# -----------------------------------------------------------------------------

def loo_cv(bundle: Bundle, verbose: bool = True,
           checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Proper leave-one-out CV with refitted arch_mean, chain_pair, delta, GPR per fold.

    If checkpoint_path is provided, predictions are saved to a .npz file after
    each fold so the loop can be resumed by calling loo_cv again with the same path.
    """
    import os
    n = len(bundle.pkas)
    preds = np.full(n, np.nan)
    levels = np.full(n, -1, dtype=int)
    start = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        d = np.load(checkpoint_path)
        preds = d["preds"]
        levels = d["levels"]
        done = np.isfinite(preds)
        start = int(done.sum())
        if verbose:
            print(f"[LOO] resuming from checkpoint at fold {start}/{n}")
    for holdout in range(n):
        if np.isfinite(preds[holdout]):
            continue
        keep = [i for i in range(n) if i != holdout]
        tok_tr = [bundle.tokens[i] for i in keep]
        pka_tr = [bundle.pkas[i] for i in keep]
        fps_tr = [bundle.fps[i] for i in keep]
        canon_tr = [bundle.canonical_smiles[i] for i in keep]
        feat_tr = bundle.features[keep]
        fam_tr = [bundle.families[i] for i in keep]
        ids_tr = [bundle.ids[i] for i in keep]

        delta_df = build_delta_table(tok_tr, pka_tr)
        cp_table = build_chain_pair_table(tok_tr, pka_tr)
        fam_gpr, fam_sc, pooled_gpr, pooled_sc = fit_family_gprs(
            feat_tr, np.array(pka_tr), fam_tr, n_restarts=1)
        # Refit pure-model ensemble per fold so the Level 6 path is LOO-honest
        pooled_rf, pooled_gbr = fit_pure_model_ensemble(
            feat_tr, np.array(pka_tr), pooled_sc)

        fold_bundle = Bundle(
            version="loo", ids=ids_tr, tokens=tok_tr, pkas=pka_tr,
            canonical_smiles=canon_tr, fps=fps_tr, features=feat_tr,
            families=fam_tr, family_gpr=fam_gpr, family_scaler=fam_sc,
            pooled_gpr=pooled_gpr, pooled_scaler=pooled_sc,
            delta_df=delta_df, chain_pair_table=cp_table,
            pooled_rf=pooled_rf, pooled_gbr=pooled_gbr,
        )

        q_tok = bundle.tokens[holdout]
        q_fp = bundle.fps[holdout]
        q_canon = bundle.canonical_smiles[holdout]
        q_feats = bundle.features[holdout]

        gpr_pred, gpr_std = predict_gpr(q_tok.family, q_feats, fold_bundle)
        lr = select_level(q_tok, q_fp, q_canon, fold_bundle)

        # Level 6 trigger (mirrors predict_pka): query has no reliable analog
        global_top1 = float(max([n["tanimoto"] for n in lr.top3], default=0.0))
        use_level6 = (
            global_top1 < 0.30
            or q_tok.family in ("NOVEL", None)
            or (lr.level == 5 and gpr_std > 0.30)
        )
        if use_level6:
            l6_pred, _, _ = predict_level6_pure_model(q_feats, q_fp, fold_bundle)
            preds[holdout] = l6_pred
            levels[holdout] = 6
            if checkpoint_path and (holdout + 1) % 10 == 0:
                np.savez(checkpoint_path, preds=preds, levels=levels)
            if verbose and (holdout + 1) % 20 == 0:
                y = np.array(bundle.pkas)
                done = np.isfinite(preds)
                mae = float(np.mean(np.abs(y[done] - preds[done])))
                print(f"  [LOO] {int(done.sum())}/{n}  running MAE={mae:.3f}", flush=True)
            continue

        levels[holdout] = lr.level

        weights = compute_weights(lr.level, bool(lr.delta_reliable),
                                  q_tok.family, lr.n_arch, gpr_std)

        if lr.level == 1:
            p = lr.anchor_pKa
        elif lr.level == 2:
            anchor_val = lr.chain_lookup_pKa if lr.chain_lookup_pKa is not None else lr.anchor_pKa
            p = (weights["anchor"] * anchor_val
                 + weights["delta"] * anchor_val
                 + weights["gpr"] * gpr_pred)
        elif lr.level in (3, 4):
            p = (weights["anchor"] * lr.anchor_pKa
                 + weights["delta"] * (lr.anchor_pKa + (lr.delta_value or 0.0))
                 + weights["gpr"] * gpr_pred)
        else:
            p = gpr_pred
        preds[holdout] = p

        if checkpoint_path and (holdout + 1) % 10 == 0:
            np.savez(checkpoint_path, preds=preds, levels=levels)

        if verbose and (holdout + 1) % 20 == 0:
            y = np.array(bundle.pkas)
            done = np.isfinite(preds)
            mae = float(np.mean(np.abs(y[done] - preds[done])))
            print(f"  [LOO] {int(done.sum())}/{n}  running MAE={mae:.3f}", flush=True)

    if checkpoint_path:
        np.savez(checkpoint_path, preds=preds, levels=levels)

    y_true = np.array(bundle.pkas)
    errs = y_true - preds
    mae = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    within_008 = float(np.mean(np.abs(errs) <= 0.08))
    within_012 = float(np.mean(np.abs(errs) <= 0.12))
    within_015 = float(np.mean(np.abs(errs) <= 0.15))
    ss_res = float(np.sum(errs ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    per_family = {}
    fam_arr = np.array(bundle.families)
    for f in sorted(set(bundle.families)):
        m = fam_arr == f
        if m.sum() > 0:
            per_family[f] = {
                "n": int(m.sum()),
                "mae": float(np.mean(np.abs(errs[m]))),
                "rmse": float(np.sqrt(np.mean(errs[m] ** 2))),
                "within_0.08": float(np.mean(np.abs(errs[m]) <= 0.08)),
            }

    level_dist = {int(lv): int(np.sum(levels == lv)) for lv in np.unique(levels)}

    return {
        "n": n,
        "mae": mae, "rmse": rmse, "r2": r2,
        "within_0.08": within_008,
        "within_0.12": within_012,
        "within_0.15": within_015,
        "per_family": per_family,
        "level_distribution": level_dist,
        "predictions": preds.tolist(),
        "y_true": y_true.tolist(),
    }


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "IAJD_pKa_v20_final.xlsx"
    b = build_bundle(xlsx)
    print("\n=== LOO-CV ===")
    rep = loo_cv(b, verbose=True, checkpoint_path="loo_checkpoint_v52.npz")
    print(f"\nPooled MAE:       {rep['mae']:.4f}")
    print(f"Pooled RMSE:      {rep['rmse']:.4f}")
    print(f"R^2:              {rep['r2']:.4f}")
    print(f"Within +/- 0.08:  {rep['within_0.08']:.1%}")
    print(f"Within +/- 0.12:  {rep['within_0.12']:.1%}")
    print(f"Within +/- 0.15:  {rep['within_0.15']:.1%}")
    print("\nPer-family:")
    for f, r in rep["per_family"].items():
        print(f"  {f:<18} n={r['n']:>3}  MAE={r['mae']:.3f}  "
              f"RMSE={r['rmse']:.3f}  within 0.08={r['within_0.08']:.0%}")
    print("\nLevel distribution:", rep["level_distribution"])
