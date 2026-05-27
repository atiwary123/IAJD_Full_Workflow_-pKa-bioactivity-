"""Compute pH-dependent Critical Packing Parameter (CPP) for IAJDs.

Physics: P = V_tail / (a_head × l_tail)
  - P < 1: cylinder (bilayer)
  - P ≈ 1: flexible bilayer
  - P > 1: cone (favors inverted hexagonal H_II phase → endosomal escape)

At endosomal pH (~5.5), the ionizable amine protonates:
  - Headgroup area (a) increases (charge repulsion, solvation shell)
  - Tail geometry stays ~constant
  - CPP shifts → quantifies the shape change that drives membrane fusion

Features computed per IAJD:
  1. CPP_neutral:    packing parameter at pH 7.4 (uncharged amine)
  2. CPP_protonated: packing parameter at pH 5.5 (charged amine)
  3. delta_CPP:      CPP_protonated - CPP_neutral (shape change magnitude)
  4. V_tail:         total tail volume (Å³)
  5. a_head_neutral: headgroup area at pH 7.4 (Å²)
  6. a_head_charged: headgroup area at pH 5.5 (Å²)
  7. l_tail:         effective tail length (Å)
  8. cone_angle:     approximate cone angle from CPP
  9. protonation_fraction_55: Henderson-Hasselbalch fraction at pH 5.5
  10. protonation_fraction_65: fraction at pH 6.5 (early endosome)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem import Descriptors3D

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")


def identify_head_and_tail_atoms(mol):
    """Split an IAJD molecule into head (ionizable amine region) and tail (hydrophobic chains).

    Head = ionizable nitrogen + atoms within 4 bonds of it (amine, linker, piperazine)
    Tail = everything else (alkyl chains, aromatic core for some families)
    """
    # Find ionizable nitrogens (tertiary amines, piperazine N)
    n_pat = Chem.MolFromSmarts("[NX3;H0]")
    n_matches = mol.GetSubstructMatches(n_pat)
    if not n_matches:
        return None, None

    # Head atoms: all N + atoms within 4 bonds
    head_atoms = set()
    for match in n_matches:
        n_idx = match[0]
        head_atoms.add(n_idx)
        # BFS up to 4 bonds from N
        visited = {n_idx}
        frontier = {n_idx}
        for depth in range(4):
            next_frontier = set()
            for idx in frontier:
                atom = mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    nidx = neighbor.GetIdx()
                    if nidx not in visited:
                        visited.add(nidx)
                        next_frontier.add(nidx)
            frontier = next_frontier
            head_atoms |= frontier

    # Also include ester/amide carbonyl connected to head
    ester_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")
    for match in mol.GetSubstructMatches(ester_pat):
        if any(idx in head_atoms for idx in match):
            head_atoms.update(match)

    tail_atoms = set(range(mol.GetNumAtoms())) - head_atoms

    return head_atoms, tail_atoms


def compute_tail_volume(mol, tail_atoms, conf_id=0):
    """Estimate tail volume using van der Waals radii.
    V_CH2 ≈ 27 Å³, V_CH3 ≈ 54 Å³ (Tanford formula).
    """
    vdw_radii = {6: 1.70, 1: 1.20, 7: 1.55, 8: 1.52, 16: 1.80}

    # Count tail carbons
    n_tail_c = sum(1 for idx in tail_atoms
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
    n_tail_heavy = sum(1 for idx in tail_atoms
                       if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)

    # Tanford formula: V = 27.4 + 26.9 * n_c (for linear chains, Å³)
    # Modified for branched chains
    v_tanford = 27.4 + 26.9 * n_tail_c

    return v_tanford, n_tail_c


def compute_tail_length(mol, tail_atoms, conf):
    """Estimate effective tail length from 3D conformer.
    Maximum distance from core attachment to terminal methyl.
    """
    if conf is None:
        # Estimate from atom count: ~1.265 Å per CH2 (C-C bond projected)
        n_c = sum(1 for idx in tail_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
        return 1.265 * n_c * 0.7  # 0.7 correction for branching/folding

    # Find tail atoms positions
    tail_positions = []
    for idx in tail_atoms:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:  # skip H
            pos = conf.GetAtomPosition(idx)
            tail_positions.append(np.array([pos.x, pos.y, pos.z]))

    if len(tail_positions) < 2:
        return 5.0  # minimum

    # Max pairwise distance as estimate of tail extent
    positions = np.array(tail_positions)
    from scipy.spatial.distance import pdist
    max_dist = np.max(pdist(positions))

    return max_dist / 2.0  # half-extent ≈ effective length


def compute_headgroup_area(mol, head_atoms, charged=False, conf=None):
    """Estimate headgroup cross-sectional area.

    Neutral: SASA of head atoms / 4π (sphere equivalent)
    Charged: increase by ~30-60% due to solvation shell expansion
             and electrostatic repulsion between protonated amines.
    """
    # Count head heavy atoms and estimate area
    n_head_heavy = sum(1 for idx in head_atoms
                       if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)

    # Base area: ~15 Å² per heavy atom in headgroup (empirical)
    # Piperazine head ≈ 50-70 Å², DMBA ≈ 30-40 Å²
    base_area = 12.0 * n_head_heavy

    if conf is not None:
        # Use actual SASA of head atoms
        from rdkit.Chem import rdFreeSASA
        try:
            radii = rdFreeSASA.classifyAtoms(mol)
            sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf.GetId())
            # Get per-atom SASA
            head_sasa = 0
            for idx in head_atoms:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:
                    # Approximate per-atom contribution
                    head_sasa += sasa * (1.0 / mol.GetNumHeavyAtoms())
            if head_sasa > 10:
                base_area = head_sasa
        except Exception:
            pass

    if charged:
        # Protonation increases effective headgroup area
        # Empirical: +40% for single protonation, +60% for dual
        n_ionizable = sum(1 for idx in head_atoms
                         if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
                         and mol.GetAtomWithIdx(idx).GetDegree() == 3)
        expansion = 1.0 + 0.30 * n_ionizable  # 30% per ionizable N
        base_area *= expansion

    return max(base_area, 20.0)  # minimum 20 Å²


def henderson_hasselbalch(pka, ph):
    """Fraction protonated at given pH."""
    return 1.0 / (1.0 + 10**(ph - pka))


def compute_cpp_features(smiles, pka=None):
    """Compute all CPP-related features for one IAJD."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    result = {}

    # Identify head and tail
    head_atoms, tail_atoms = identify_head_and_tail_atoms(mol)
    if head_atoms is None or len(tail_atoms) < 5:
        return None

    result["n_head_atoms"] = len(head_atoms)
    result["n_tail_atoms"] = len(tail_atoms)

    # Generate 3D conformer
    mol_h = Chem.AddHs(mol)
    conf = None
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=1, params=params)
        if cids:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=cids[0], maxIters=500)
            conf = mol_h.GetConformer(cids[0])
    except Exception:
        pass

    # Tail volume and length
    v_tail, n_tail_c = compute_tail_volume(mol, tail_atoms)
    l_tail = compute_tail_length(mol, tail_atoms, conf)

    result["V_tail"] = v_tail
    result["n_tail_carbons"] = n_tail_c
    result["l_tail"] = l_tail

    # Headgroup areas
    a_neutral = compute_headgroup_area(mol, head_atoms, charged=False, conf=conf)
    a_charged = compute_headgroup_area(mol, head_atoms, charged=True, conf=conf)

    result["a_head_neutral"] = a_neutral
    result["a_head_charged"] = a_charged
    result["delta_a_head"] = a_charged - a_neutral

    # CPP values
    cpp_neutral = v_tail / (a_neutral * l_tail) if (a_neutral * l_tail) > 0 else 0
    cpp_charged = v_tail / (a_charged * l_tail) if (a_charged * l_tail) > 0 else 0

    result["CPP_neutral"] = cpp_neutral
    result["CPP_charged"] = cpp_charged
    result["delta_CPP"] = cpp_charged - cpp_neutral
    result["CPP_ratio"] = cpp_charged / cpp_neutral if cpp_neutral > 0 else 1.0

    # Cone angle approximation: θ ≈ arctan(sqrt(a_head) / l_tail)
    result["cone_angle_neutral"] = np.degrees(np.arctan2(np.sqrt(a_neutral), l_tail))
    result["cone_angle_charged"] = np.degrees(np.arctan2(np.sqrt(a_charged), l_tail))
    result["delta_cone_angle"] = result["cone_angle_charged"] - result["cone_angle_neutral"]

    # Henderson-Hasselbalch protonation fractions
    if pka is not None and not np.isnan(pka):
        result["protonation_55"] = henderson_hasselbalch(pka, 5.5)
        result["protonation_65"] = henderson_hasselbalch(pka, 6.5)
        result["protonation_74"] = henderson_hasselbalch(pka, 7.4)
        result["delta_protonation"] = result["protonation_55"] - result["protonation_74"]

        # Effective shape change = geometric change × protonation change
        result["effective_shape_change_55"] = abs(result["delta_CPP"]) * result["protonation_55"]
        result["effective_shape_change_65"] = abs(result["delta_CPP"]) * result["protonation_65"]
    else:
        for k in ["protonation_55","protonation_65","protonation_74",
                   "delta_protonation","effective_shape_change_55","effective_shape_change_65"]:
            result[k] = np.nan

    # Additional geometric descriptors
    result["MolLogP"] = Crippen.MolLogP(mol)
    result["HeavyAtomCount"] = float(mol.GetNumHeavyAtoms())
    result["tail_fraction"] = len(tail_atoms) / mol.GetNumHeavyAtoms()
    result["head_tail_ratio"] = len(head_atoms) / max(len(tail_atoms), 1)

    # Number of tails (count long alkyl chains)
    # Approximate: count terminal CH3 groups
    ch3_pat = Chem.MolFromSmarts("[CH3]")
    result["n_tails_approx"] = len(mol.GetSubstructMatches(ch3_pat))

    # Branching in tails
    branch_pat = Chem.MolFromSmarts("[CX4;H1]([CX4])([CX4])[CX4]")
    result["n_branch_points"] = len(mol.GetSubstructMatches(branch_pat))

    return result


CPP_FEATURE_NAMES = [
    "CPP_neutral", "CPP_charged", "delta_CPP", "CPP_ratio",
    "V_tail", "l_tail", "a_head_neutral", "a_head_charged", "delta_a_head",
    "cone_angle_neutral", "cone_angle_charged", "delta_cone_angle",
    "protonation_55", "protonation_65", "protonation_74", "delta_protonation",
    "effective_shape_change_55", "effective_shape_change_65",
    "n_tail_carbons", "tail_fraction", "head_tail_ratio",
    "n_tails_approx", "n_branch_points",
]


_EXCLUDED_NOVEL_IAJDS = {347, 348, 365, 366, 367, 369, 372, 373}


def main():
    # Compute CPP for all bioactivity training compounds
    bio = pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx', sheet_name='Sheet1')
    if "IAJD_num" in bio.columns:
        bio = bio[~bio["IAJD_num"].isin(_EXCLUDED_NOVEL_IAJDS)].reset_index(drop=True)
    pka_cache = pd.read_csv('v11_pka_flux/predicted_pka_cache.csv')
    merged = bio.merge(pka_cache[["row_id","predicted_pKa"]], on="row_id", how="left")

    print(f"Computing CPP features for {len(bio)} compounds...")

    cpp_rows = []
    for i, (_, row) in enumerate(bio.iterrows()):
        smi = row.get('SMILES_canonical') or row.get('SMILES')
        pka = merged.iloc[i].get('predicted_pKa') if i < len(merged) else None
        if pd.isna(smi):
            cpp_rows.append({k: np.nan for k in CPP_FEATURE_NAMES})
            continue

        feats = compute_cpp_features(str(smi), pka=pka if pd.notna(pka) else None)
        if feats is None:
            cpp_rows.append({k: np.nan for k in CPP_FEATURE_NAMES})
        else:
            cpp_rows.append({k: feats.get(k, np.nan) for k in CPP_FEATURE_NAMES})

        if i % 50 == 0:
            print(f"  {i}/{len(bio)}", flush=True)

    df_cpp = pd.DataFrame(cpp_rows)
    df_cpp.to_csv('cpp_features_bioact.csv', index=False)
    print(f"\nSaved cpp_features_bioact.csv: {df_cpp.shape}")

    # Basic stats
    print(f"\nFeature statistics:")
    for col in CPP_FEATURE_NAMES[:12]:
        vals = df_cpp[col].dropna()
        if len(vals) > 0:
            print(f"  {col:30s} mean={vals.mean():8.3f} std={vals.std():8.3f} range=[{vals.min():.3f}, {vals.max():.3f}]")

    # Correlation with flux
    flux = bio['log10_flux_total'].values
    print(f"\nCorrelation with log10_flux_total:")
    from scipy.stats import pearsonr, spearmanr
    for col in CPP_FEATURE_NAMES:
        vals = df_cpp[col].values
        valid = np.isfinite(vals) & np.isfinite(flux)
        if valid.sum() < 10:
            continue
        r_p, p_p = pearsonr(vals[valid], flux[valid])
        r_s, p_s = spearmanr(vals[valid], flux[valid])
        if abs(r_p) > 0.05:
            sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
            print(f"  {col:30s} r={r_p:+.3f} (p={p_p:.1e}) rho={r_s:+.3f} {sig}")


if __name__ == "__main__":
    main()
