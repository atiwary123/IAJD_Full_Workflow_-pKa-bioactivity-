"""
Integrate the 8 GA-Tris IAJDs from `lion_repo/scripts/IAJD for Comp.cdxml`
(347, 348, 365, 366, 367, 369, 372, 373) into the full training workflow.

Inputs:
  - cdxml_iajds_extracted.csv          (validated cdxml→SMILES extraction)
  - IAJD_Bioact_v13_clean.xlsx.bak_pre_novel_removal  (flux + metadata)
  - existing pKa primary model         (predicts pKa for new SMILES)
  - existing LION + ADMET caches       (already extended for the 8 novels)

Outputs:
  - IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx  (8 rows appended)
  - IAJD_master/datasets/IAJD_pKa_v21_final.xlsx     (8 rows appended)
  - IAJD_master/datasets/IAJD_Bioact_v13_clean.pkl   (regenerated)
  - cdxml_iajds_integration_report.json              (full audit trail)
"""
from __future__ import annotations
import json, sys, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Reuse expand_datasets feature functions (RDKit 2D + 3D, all 61 cols)
from expand_datasets import compute_rdkit_features, compute_3d_features

NOVELS = [347, 348, 365, 366, 367, 369, 372, 373]
BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
BIOACT_PKL  = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.pkl"
BACKUP_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx.bak_pre_novel_removal"
PKA_XLSX    = ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
CDXML_CSV   = ROOT / "cdxml_iajds_extracted.csv"
REPORT_JSON = ROOT / "cdxml_iajds_integration_report.json"


def fit_pka_baseline_and_predict(train_df: pd.DataFrame, query_smiles: list[str]):
    """Train a simple RDKit-descriptor-based pKa regressor on the existing
    training data and predict pKa for the novel SMILES.

    Uses GradientBoosting because the existing trained primary model files
    require auxiliary feature stacks we don't have packaged. The point here is
    to provide a reasonable pKa estimate so the row's pKa column is non-null
    (downstream code keeps a separate pKa cache anyway)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "ExactMolWt", "MolLogP", "TPSA", "LabuteASA", "FractionCSP3",
        "RotatableBonds", "NumAromaticRings", "NumHDonors", "NumHAcceptors",
        "NumTertiaryAmines", "HasPiperazine", "NumAmines_total",
        "Hydrophobic_Index", "Polar_Surface_Ratio", "Inductive_Effect_Strength",
        "HBD_HBA_Ratio", "Desolvation_Proxy",
    ]
    train = train_df.dropna(subset=["pKa"]).copy()
    # Sanitize: replace inf/-inf, fill NaN feature cells with column medians
    for c in feature_cols:
        if c not in train.columns:
            train[c] = np.nan
    X = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = train["pKa"].astype(float).values

    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)
    model = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                      learning_rate=0.03, random_state=42)
    model.fit(Xs, y)

    # Feature matrix for query
    rows = []
    for sm in query_smiles:
        d = compute_rdkit_features(sm)
        rows.append([d.get(c, np.nan) for c in feature_cols])
    Xq = pd.DataFrame(rows, columns=feature_cols)
    Xq = Xq.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    Xq_s = scaler.transform(Xq.values)
    yhat = model.predict(Xq_s)
    # In-sample OOB-ish error proxy = train residual std (very rough sd)
    resid_std = float(np.std(y - model.predict(Xs)))
    return yhat.tolist(), resid_std


def main():
    report = {"timestamp": pd.Timestamp.utcnow().isoformat(),
              "novels": NOVELS, "actions": [], "warnings": []}

    # ── 1. Read backup rows (flux + metadata) ────────────────────────────
    print("\n[1/6] Loading backup rows for novels...")
    bak = pd.read_excel(BACKUP_XLSX)
    nov = bak[bak["IAJD_num"].isin(NOVELS)].copy().reset_index(drop=True)
    if len(nov) != 8:
        raise RuntimeError(f"Expected 8 backup rows; got {len(nov)}")
    print(f"  loaded {len(nov)} backup rows")

    # Cross-check against cdxml extraction
    cdx = pd.read_csv(CDXML_CSV)
    canonical_lookup = {}
    for _, r in nov.iterrows():
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(r["SMILES"]))
        canonical_lookup[int(r["IAJD_num"])] = canonical
    for _, r in cdx.iterrows():
        n = int(r["iajd_num"])
        if canonical_lookup.get(n) != r["smiles_cdxml_canon"]:
            report["warnings"].append(
                f"IAJD {n}: cdxml SMILES {r['smiles_cdxml_canon']} differs "
                f"from backup canonical {canonical_lookup.get(n)}"
            )
    if not report["warnings"]:
        print("  cdxml ↔ backup SMILES: all 8 match")
    report["actions"].append("validated cdxml SMILES against backup")

    # ── 2. Compute full descriptor block (RDKit 2D + 3D) ─────────────────
    print("\n[2/6] Computing full RDKit + 3D descriptor block...")
    enriched_rows = []
    for _, r in nov.iterrows():
        n = int(r["IAJD_num"])
        canonical = canonical_lookup[n]
        feats_2d = compute_rdkit_features(canonical)
        feats_3d = compute_3d_features(canonical)
        row = dict(r)
        # Overwrite with newly computed descriptors (backup had many NaN)
        for k, v in feats_2d.items():
            row[k] = v
        for k, v in feats_3d.items():
            row[k] = v
        # Override Linker_Length from architecture metadata
        ll = r.get("linker_length")
        if pd.notna(ll):
            row["Linker_Length"] = float(ll)
            row["Taft_Steric_Sum"] = -1.24 * float(ll)
        row["SMILES_canonical"] = canonical
        row["v15_status"] = "features_refreshed"
        row["v16_status"] = "features_refreshed"
        row["Validation_Status"] = "REINTEGRATED_FROM_CDXML_2026-05-28"
        enriched_rows.append(row)
        print(f"  IAJD {n}: NumNitrogens={feats_2d['NumNitrogens']}, "
              f"MolLogP={feats_2d['MolLogP']:.2f}, n_confs={feats_3d.get('n_confs_valid_3D','?')}")

    nov_full = pd.DataFrame(enriched_rows)

    # ── 3. Append to Bioact xlsx ─────────────────────────────────────────
    print("\n[3/6] Appending novels to IAJD_Bioact_v13_clean.xlsx...")
    df_bio = pd.read_excel(BIOACT_XLSX)
    before_n = len(df_bio)
    existing_ids = set(df_bio["IAJD_num"].dropna().astype(int))
    overlap = set(NOVELS) & existing_ids
    if overlap:
        raise RuntimeError(f"Novels already present in bioact xlsx: {sorted(overlap)}")

    # Align columns
    for c in df_bio.columns:
        if c not in nov_full.columns:
            nov_full[c] = np.nan
    extra = [c for c in nov_full.columns if c not in df_bio.columns]
    if extra:
        # Drop columns not in bioact schema
        nov_full = nov_full.drop(columns=extra)
    nov_full = nov_full[df_bio.columns]

    # Assign fresh row_ids
    next_row_id = int(df_bio["row_id"].max()) + 1 if "row_id" in df_bio.columns else len(df_bio)
    nov_full["row_id"] = list(range(next_row_id, next_row_id + len(nov_full)))

    df_bio_new = pd.concat([df_bio, nov_full], ignore_index=True)
    df_bio_new = df_bio_new.sort_values("IAJD_num").reset_index(drop=True)
    df_bio_new.to_excel(BIOACT_XLSX, sheet_name="Sheet1", index=False)
    print(f"  bioact: {before_n} → {len(df_bio_new)} rows (+{len(nov_full)})")
    report["actions"].append(f"bioact xlsx: {before_n} → {len(df_bio_new)}")

    # Also regenerate pickle
    df_bio_new.to_pickle(BIOACT_PKL)
    print(f"  bioact pkl regenerated: {BIOACT_PKL.name}")

    # ── 4. Append to pKa xlsx with predicted pKa ─────────────────────────
    print("\n[4/6] Predicting pKa and appending to IAJD_pKa_v21_final.xlsx...")
    df_pka = pd.read_excel(PKA_XLSX)
    before_p = len(df_pka)
    smiles_q = [canonical_lookup[int(r["IAJD_num"])] for _, r in nov.iterrows()]
    yhat, resid_std = fit_pka_baseline_and_predict(df_pka, smiles_q)
    print(f"  fitted GB on {(df_pka['pKa'].notna()).sum()} training pKa rows; resid_std={resid_std:.2f}")
    pka_rows = []
    for (_, r), pka in zip(nov.iterrows(), yhat):
        n = int(r["IAJD_num"])
        canonical = canonical_lookup[n]
        d_2d = compute_rdkit_features(canonical)
        d_3d = compute_3d_features(canonical)
        ll = float(r["linker_length"]) if pd.notna(r["linker_length"]) else np.nan
        row = {"IAJD": n, "SMILES": canonical, "pKa": float(pka),
               "pKa_sd": resid_std,
               "family": r["family"], "family_original": r["family_original"],
               "architecture": r["architecture"], "architecture_coarse": r.get("architecture_coarse",""),
               "architecture_v13": r.get("architecture_v13",""),
               "head_group": r["head_group"], "linker_length": ll, "linkage": r["linkage"],
               "source": "novel_2026_reintegrated",
               "v14_status": "REINTEGRATED",
               "v15_status": "features_refreshed",
               "v16_status": "features_refreshed",
               "Validation_Status": "REINTEGRATED_FROM_CDXML_2026-05-28",
               "Linker_Length": ll,
               "Taft_Steric_Sum": -1.24 * ll if pd.notna(ll) else np.nan}
        row.update(d_2d)
        row.update(d_3d)
        pka_rows.append(row)
    df_pka_new_only = pd.DataFrame(pka_rows)
    for c in df_pka.columns:
        if c not in df_pka_new_only.columns:
            df_pka_new_only[c] = np.nan
    extra = [c for c in df_pka_new_only.columns if c not in df_pka.columns]
    if extra:
        df_pka_new_only = df_pka_new_only.drop(columns=extra)
    df_pka_new_only = df_pka_new_only[df_pka.columns]
    df_pka_full = pd.concat([df_pka, df_pka_new_only], ignore_index=True)
    df_pka_full = df_pka_full.sort_values("IAJD").reset_index(drop=True)
    df_pka_full.to_excel(PKA_XLSX, sheet_name="Dataset", index=False)
    print(f"  pKa: {before_p} → {len(df_pka_full)} rows (+{len(pka_rows)})")
    report["actions"].append(f"pKa xlsx: {before_p} → {len(df_pka_full)}")
    report["pka_predicted"] = {int(r["IAJD"]): round(float(r["pKa"]), 3) for r in pka_rows}
    report["pka_resid_std"] = round(resid_std, 3)

    # ── 5. Write report ──────────────────────────────────────────────────
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[5/6] Wrote integration report → {REPORT_JSON.name}")

    print("\n[6/6] DONE. Next: regenerate AGILE embeddings, CPP features, and remove EXCLUDED_NOVEL_IAJDS guards.")


if __name__ == "__main__":
    main()
