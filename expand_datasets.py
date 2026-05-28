"""Expand pKa and bioactivity datasets with cross-populated + reconstructed IAJDs.

Phase 1: Cross-populate (bioact→pKa for 22 IAJDs with pKa+SMILES)
Phase 0 data: Add 34 reconstructed IAJDs (unique SMILES) to pKa dataset
Compute all 61 columns for new rows.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, Crippen, Descriptors, Lipinski,
    GraphDescriptors, MolSurf, rdMolDescriptors, Descriptors3D,
)

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PKA_XLSX = "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
BIO_XLSX = "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
RECON_CSV = "reconstructed_iajds.csv"

# ── Feature computation ─────────────────────────────────────────────────

def compute_rdkit_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    d = {}
    d["ExactMolWt"] = Descriptors.ExactMolWt(mol)
    d["MolFormula"] = rdMolDescriptors.CalcMolFormula(mol)
    d["HeavyAtomCount"] = mol.GetNumHeavyAtoms()
    d["NumNitrogens"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    d["MolLogP"] = Crippen.MolLogP(mol)
    d["TPSA"] = Descriptors.TPSA(mol)
    d["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
    d["FractionCSP3"] = Lipinski.FractionCSP3(mol)
    d["RotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    d["BertzCT"] = GraphDescriptors.BertzCT(mol)
    d["Chi0v"] = GraphDescriptors.Chi0v(mol)
    d["Chi1v"] = GraphDescriptors.Chi1v(mol)
    d["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)
    d["NumAromaticRings"] = Lipinski.NumAromaticRings(mol)
    d["NumHDonors"] = Lipinski.NumHDonors(mol)
    d["NumHAcceptors"] = Lipinski.NumHAcceptors(mol)

    # Substructure counts
    d["NumEsters"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2]")))
    d["NumAmides"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]")))
    d["NumEthers"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4][OX2][CX4]")))
    tert_n_pat = Chem.MolFromSmarts("[NX3;H0]")
    d["NumTertiaryAmines"] = len(mol.GetSubstructMatches(tert_n_pat))
    piperazine_pat = Chem.MolFromSmarts("C1CNCCN1")
    d["HasPiperazine"] = 1 if mol.HasSubstructMatch(piperazine_pat) else 0
    amine_pat = Chem.MolFromSmarts("[NX3;H0,H1,H2]")
    d["NumAmines_total"] = len(mol.GetSubstructMatches(amine_pat))

    # Gasteiger charges on tertiary nitrogens
    m2 = Chem.RWMol(mol)
    try:
        AllChem.ComputeGasteigerCharges(m2)
        tert_charges = []
        for atom in m2.GetAtoms():
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 3:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isfinite(q):
                    tert_charges.append(q)
        d["GasteigerN_min"] = min(tert_charges) if tert_charges else np.nan
        d["GasteigerN_max"] = max(tert_charges) if tert_charges else np.nan
    except Exception:
        d["GasteigerN_min"] = np.nan
        d["GasteigerN_max"] = np.nan

    # Derived features
    d["Hydrophobic_Index"] = d["MolLogP"] / d["ExactMolWt"] if d["ExactMolWt"] > 0 else 0
    d["Polar_Surface_Ratio"] = d["TPSA"] / d["LabuteASA"] if d["LabuteASA"] > 0 else 0

    # Inductive effect: fraction of sp3 C in the linker region
    n_sp3_c = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() == 6 and a.GetHybridization().name == "SP3")
    total_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    d["Inductive_Effect_Strength"] = n_sp3_c / total_c if total_c > 0 else 0

    # Linker length (carbons between ester C=O and first N)
    linker_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3][CX4]~[CX4]~[CX4]~[NX3]")
    linker_matches = mol.GetSubstructMatches(linker_pat) if linker_pat else []
    d["Linker_Length"] = np.nan  # will be overridden by metadata if available

    # Taft steric sum (approximate from linker carbons)
    taft_per_ch2 = -1.24
    d["Taft_Steric_Sum"] = np.nan

    # Gasteiger charge on N and C-alpha
    d["Gasteiger_Charge_N"] = d["GasteigerN_max"]
    d["Gasteiger_Charge_Calpha"] = np.nan
    try:
        n_alpha_pat = Chem.MolFromSmarts("[NX3][CX4]")
        matches = m2.GetSubstructMatches(n_alpha_pat)
        if matches:
            calpha_charges = []
            for m in matches:
                c_atom = m2.GetAtomWithIdx(m[1])
                q = float(c_atom.GetProp("_GasteigerCharge"))
                if np.isfinite(q):
                    calpha_charges.append(q)
            if calpha_charges:
                d["Gasteiger_Charge_Calpha"] = np.mean(calpha_charges)
    except Exception:
        pass

    # N electron-donor groups within 5 Angstroms (approximated by bond distance)
    d["N_ED_Groups_5A"] = d["NumEthers"]  # simplified proxy

    # HBD/HBA ratio
    d["HBD_HBA_Ratio"] = d["NumHDonors"] / d["NumHAcceptors"] if d["NumHAcceptors"] > 0 else 0

    # Desolvation proxy
    d["Desolvation_Proxy"] = d["MolLogP"] * d["HeavyAtomCount"] / 100.0

    return d


def compute_3d_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    d = {}
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.numThreads = 1

        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=5, params=params)
        if not cids:
            params2 = AllChem.ETKDGv3()
            params2.randomSeed = 42
            params2.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=3, params=params2)

        if not cids:
            d["embed_method_3D"] = "failed"
            d["ff_3D"] = "none"
            d["n_confs_valid_3D"] = 0
            for k in ["Pct_V_Bur_max", "Pct_V_Bur_mean", "Rg_3D",
                       "Asphericity_3D", "N_LowE_Conformers", "E_min_3D",
                       "N_basic_N_3D"]:
                d[k] = np.nan
            return d

        d["embed_method_3D"] = "etkdgv3" if len(cids) > 0 else "random_coords"

        # Minimize with MMFF94
        energies = []
        ff_name = "mmff94"
        for cid in cids:
            try:
                mp = AllChem.MMFFGetMoleculeProperties(mol_h)
                if mp:
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=cid)
                    if ff:
                        ff.Minimize(maxIts=500)
                        energies.append((cid, ff.CalcEnergy()))
                        continue
                ff_uff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
                if ff_uff:
                    ff_uff.Minimize(maxIts=500)
                    energies.append((cid, ff_uff.CalcEnergy()))
                    ff_name = "uff"
            except Exception:
                pass

        d["ff_3D"] = ff_name
        d["n_confs_valid_3D"] = len(energies)

        if not energies:
            for k in ["Pct_V_Bur_max", "Pct_V_Bur_mean", "Rg_3D",
                       "Asphericity_3D", "N_LowE_Conformers", "E_min_3D",
                       "N_basic_N_3D"]:
                d[k] = np.nan
            return d

        energies.sort(key=lambda x: x[1])
        best_cid, best_e = energies[0]
        d["E_min_3D"] = best_e
        d["N_LowE_Conformers"] = sum(1 for _, e in energies if e - best_e < 5.0)

        # 3D descriptors
        d["Asphericity_3D"] = Descriptors3D.Asphericity(mol_h, confId=best_cid)
        d["Rg_3D"] = Descriptors3D.RadiusOfGyration(mol_h, confId=best_cid)

        # Buried volume around tertiary N
        conf = mol_h.GetConformer(best_cid)
        tert_n_smarts = Chem.MolFromSmarts("[NX3;H0;!$(N=*);!$(NC=O)]")
        tert_n_matches = mol_h.GetSubstructMatches(tert_n_smarts)
        pct_vs = []
        n_basic = 0
        for match in tert_n_matches:
            n_idx = match[0]
            n_pos = conf.GetAtomPosition(n_idx)
            n_basic += 1
            heavy_in_sphere = 0
            total_heavy = 0
            for atom in mol_h.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                total_heavy += 1
                pos = conf.GetAtomPosition(atom.GetIdx())
                dist = n_pos.Distance(pos)
                if 0.01 < dist < 4.0:
                    heavy_in_sphere += 1
            if total_heavy > 0:
                pct_vs.append(100.0 * heavy_in_sphere / total_heavy)

        d["Pct_V_Bur_max"] = max(pct_vs) if pct_vs else np.nan
        d["Pct_V_Bur_mean"] = np.mean(pct_vs) if pct_vs else np.nan
        d["N_basic_N_3D"] = n_basic

    except Exception as e:
        for k in ["Pct_V_Bur_max", "Pct_V_Bur_mean", "Rg_3D",
                   "Asphericity_3D", "N_LowE_Conformers", "E_min_3D",
                   "N_basic_N_3D"]:
            d[k] = np.nan
        d["embed_method_3D"] = "failed"
        d["ff_3D"] = "none"
        d["n_confs_valid_3D"] = 0

    return d


def build_full_row(iajd_num, smiles, pka, pka_sd, family, family_original,
                   architecture, head_group, linker_length, linkage, source,
                   architecture_coarse="", architecture_v13="") -> dict:
    row = {}
    row["IAJD"] = iajd_num
    row["SMILES"] = smiles
    row["pKa"] = pka
    row["pKa_sd"] = pka_sd
    row["family"] = family
    row["family_original"] = family_original or family
    row["architecture"] = architecture
    row["architecture_coarse"] = architecture_coarse
    row["architecture_v13"] = architecture_v13
    row["head_group"] = head_group
    row["linker_length"] = linker_length
    row["linkage"] = linkage
    row["source"] = source
    row["v14_status"] = "OK"
    row["v15_status"] = "features_refreshed"
    row["v16_status"] = "features_refreshed"
    row["Validation_Status"] = "RECONSTRUCTED" if source == "reconstructed" else "CROSS_POPULATED"

    feats_2d = compute_rdkit_features(smiles)
    row.update(feats_2d)

    if linker_length and not np.isnan(linker_length):
        row["Linker_Length"] = linker_length
        row["Taft_Steric_Sum"] = -1.24 * linker_length

    feats_3d = compute_3d_features(smiles)
    row.update(feats_3d)

    return row


def main():
    print("Loading existing datasets...")
    df_pka = pd.read_excel(PKA_XLSX, sheet_name="Dataset")
    df_bio = pd.read_excel(BIO_XLSX, sheet_name="Sheet1")
    df_recon = pd.read_csv(RECON_CSV)

    pka_ids = set(df_pka["IAJD"].dropna().astype(int))
    bio_ids = set(df_bio["IAJD_num"].dropna().astype(int))

    existing_canonical = set()
    for smi in df_pka["SMILES"].dropna():
        m = Chem.MolFromSmiles(smi)
        if m:
            existing_canonical.add(Chem.MolToSmiles(m))

    # ── Phase 1a: Cross-populate bioact→pKa ──────────────────────────────
    print("\n=== Phase 1a: Cross-populate bioact→pKa ===")
    bioact_only = sorted(bio_ids - pka_ids)
    new_pka_rows = []

    for iajd_num in bioact_only:
        rows = df_bio[df_bio["IAJD_num"] == iajd_num]
        row0 = rows.iloc[0]

        pka_val = row0.get("pKa")
        if pd.isna(pka_val):
            pka_val = row0.get("pKa_paper")
        if pd.isna(pka_val):
            continue  # skip if no pKa available

        smiles = row0.get("SMILES_canonical") or row0.get("SMILES")
        if pd.isna(smiles):
            continue  # skip if no SMILES

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in existing_canonical:
            print(f"  IAJD {iajd_num}: SMILES already in pKa dataset, skip")
            continue

        family = row0.get("family", "")
        pka_family = family
        if family in ("G1-Janus-Dendrimer", "HTM-Dendrimer", "TT-Dendrimer"):
            pka_family = family  # keep as-is to expand family coverage

        new_row = build_full_row(
            iajd_num=iajd_num,
            smiles=canonical,
            pka=float(pka_val),
            pka_sd=row0.get("pKa_sd", np.nan),
            family=pka_family,
            family_original=row0.get("family_original", family),
            architecture=row0.get("architecture", ""),
            head_group=row0.get("head_group", ""),
            linker_length=row0.get("linker_length", np.nan),
            linkage=row0.get("linkage", ""),
            source=row0.get("source", "bioact_v13"),
            architecture_coarse=row0.get("architecture_coarse", ""),
            architecture_v13=row0.get("architecture_v13", ""),
        )
        new_pka_rows.append(new_row)
        existing_canonical.add(canonical)
        print(f"  IAJD {iajd_num}: pKa={pka_val:.2f}, family={pka_family} ✓")

    print(f"\nCross-populated: {len(new_pka_rows)} IAJDs from bioact→pKa")

    # ── Phase 0 integration: Add reconstructed IAJDs ─────────────────────
    print("\n=== Phase 0: Add reconstructed IAJDs ===")
    recon_rows = []
    for _, r in df_recon.iterrows():
        iajd_num = int(r["iajd_num"])
        canonical = r["smiles_canonical"]
        if canonical in existing_canonical:
            print(f"  IAJD {iajd_num}: duplicate SMILES, skip")
            continue
        if iajd_num in pka_ids:
            print(f"  IAJD {iajd_num}: already in pKa dataset, skip")
            continue

        pka_val = r.get("pka")
        if pd.isna(pka_val):
            pka_val = np.nan

        new_row = build_full_row(
            iajd_num=iajd_num,
            smiles=canonical,
            pka=pka_val if not pd.isna(pka_val) else np.nan,
            pka_sd=np.nan,
            family=r["family"],
            family_original=r["family"],
            architecture=r.get("notes", ""),
            head_group="",
            linker_length=np.nan,
            linkage="",
            source=r.get("source", "reconstructed"),
        )
        new_row["Validation_Status"] = f"RECONSTRUCTED_{r['confidence']}"
        recon_rows.append(new_row)
        existing_canonical.add(canonical)
        pka_str = f"pKa={pka_val:.2f}" if not pd.isna(pka_val) else "pKa=?"
        print(f"  IAJD {iajd_num}: {pka_str}, family={r['family']}, conf={r['confidence']} ✓")

    print(f"\nReconstructed IAJDs added: {len(recon_rows)}")

    # ── Combine and write expanded pKa dataset ───────────────────────────
    all_new = new_pka_rows + recon_rows
    if all_new:
        df_new = pd.DataFrame(all_new)
        # Align columns to match existing
        for col in df_pka.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        df_new = df_new[df_pka.columns]

        df_expanded = pd.concat([df_pka, df_new], ignore_index=True)
        df_expanded = df_expanded.sort_values("IAJD").reset_index(drop=True)

        out_path = PKA_XLSX.replace(".xlsx", "_expanded.xlsx")
        df_expanded.to_excel(out_path, sheet_name="Dataset", index=False)
        print(f"\n=== EXPANDED pKa DATASET ===")
        print(f"Original: {len(df_pka)} rows")
        print(f"Added: {len(all_new)} rows ({len(new_pka_rows)} cross-populated + {len(recon_rows)} reconstructed)")
        print(f"Total: {len(df_expanded)} rows")
        print(f"Written to: {out_path}")

        # Family breakdown
        print(f"\nFamily breakdown:")
        for fam, count in df_expanded["family"].value_counts().items():
            orig = len(df_pka[df_pka["family"] == fam])
            print(f"  {fam}: {count} (was {orig}, +{count - orig})")

        # Validation: no duplicate SMILES
        canonical_list = []
        for smi in df_expanded["SMILES"].dropna():
            m = Chem.MolFromSmiles(smi)
            if m:
                canonical_list.append(Chem.MolToSmiles(m))
        dupes = len(canonical_list) - len(set(canonical_list))
        print(f"\nDuplicate SMILES check: {dupes} duplicates found")

        # Validation: no duplicate IAJD numbers
        id_dupes = df_expanded["IAJD"].duplicated().sum()
        print(f"Duplicate IAJD ID check: {id_dupes} duplicates found")

    else:
        print("No new rows to add!")

    return len(all_new)


if __name__ == "__main__":
    n = main()
    print(f"\nDone. {n} new IAJDs added to expanded pKa dataset.")
