#!/usr/bin/env python
"""
build_sample_prep_features.py
=============================================================================
Add QUANTITATIVE SAMPLE-PREPARATION parameters to the IAJD datasets.

Motivation
----------
The IAJD bioactivity / pKa labels were measured across ~9 source papers whose
DNP (dendrimersome nanoparticle) formulation and in-vivo assay conditions are
*not* identical.  Unmodeled prep variation (assembly pH, IAJD:mRNA ratio,
imaging time, one-component-vs-not, dialysis, measurement method, batch/era)
shows up as label noise.  This script extracts every prep parameter that the
SOURCE PAPERS actually report (or that follows from a paper's single stated
protocol), tags each with a provenance + confidence, and writes a self-contained
`sp_*` ("sample prep") feature block into every live dataset.

Honesty / provenance policy (NO fabricated numbers)
---------------------------------------------------
Every row gets `sp_provenance` and `sp_confidence`:
  reported_per_row       (1.00) - value read per-compound from a paper data table
                                   (only ja1c05813, whose Table S7 gives per-IAJD
                                    buffer/pH/conc/size/PDI/time)
  paper_protocol         (0.85) - paper states ONE Methods protocol applied to all
                                   its IAJDs; we apply that protocol (confirmed by
                                   reading each SI)
  inferred_group_standard(0.35-0.65) - SI lacks a written protocol (bm4c01599) OR
                                   the compound is not in any cited paper
                                   (novel_2026 GA-Tris 347-373); we apply the Percec
                                   one-component standard by same-lab/same-dendron
                                   inference.  LOW confidence - down-weight these.
  unknown_paper          (0.25) - source paper not identifiable (some pKa rows);
                                   only universally-true facts are filled
                                   (prep_method, one-component, pKa titration, cargo);
                                   paper-specific values (assembly pH, dose, route,
                                   imaging time, ratio) are LEFT NULL.

Protocol facts and their provenance are documented in
`docs/SAMPLE_PREP_PARAMETERS.md` and dumped to `sample_prep_protocols.json`.

Source-paper evidence (read this session; see the .md for verbatim quotes)
-------------------------------------------------------------------------
Standard Percec one-component IAJD protocol, confirmed across
pharmaceutics1501572 / ja2c00273 / ja3c07337 / ja3c13569 / ja1c09585 / ja5c07232
(and bm4c01107 / pharmaceutics-2390918 for the GA-Tris dendron):
  rapid ETHANOL INJECTION; IAJD 80 mg/mL in EtOH; Luc-mRNA 4.0 mg/mL in water;
  12.5 uL mRNA + 463 uL acetate buffer (10 mM, pH 4.0) + 25 uL IAJD-EtOH; vortex 5 s
  -> final C_IAJD = 4.0 mg/mL, C_mRNA = 0.10 mg/mL, IAJD:mRNA = 40:1 (mass),
     EtOH vol-frac = 25/500.5 = 0.0500, batch = 50 ug mRNA.
  In vivo: retro-orbital sinus (IV), 100 uL / 10 ug Luc-mRNA, BALB/c 6-8 wk,
  IVIS Spectrum (CT), i.p. D-luciferin 150 mg/kg, image ~4 h (range 4-7 h).
  Cargo: nucleoside-modified firefly Luc-mRNA, m1-pseudouridine, cap1, poly(A)=101 nt
         (poly(A) length explicitly stated only in the 2023+ papers).
  IAJD pKa: half-equivalence-point titration, 1.5 mg/mL IAJD in NaCl-sat ethanol,
            0.1 M HCl in 7.5 uL increments.
Deviations honored:
  ja1c05813  - SCREEN: per-row assembly pH (acetate 4.0 or 5.2; IAJD 9 = citrate 3.0
               then PBS-dialysed, C_mRNA 0.025 -> 160:1). In-vivo Table S7 rows
               otherwise C_mRNA 0.10 -> 40:1.
  ja3c13569  - vortex "5 or 20 s"; assembly pH rises 4.0->~4.8-5.0 after injection.
  ja5c07232  - also DNP-surface pKa by TNS fluorescence; organ panel adds lymph node.
  ja1c09585  - early paper; in-vitro HEK293T (125 ng/well) also present; SI imaging
               4-6 h (some dataset T_hours up to 10.25 are NOT supported by the SI).
  bm4c01599  - synthesis-only SI; volumes/ratio/vortex inferred from group standard.
  novel_2026 - GA-Tris-345-EH (IAJD-97 dendron family); NO citation; standard inferred.

Usage:  python build_sample_prep_features.py            # apply + verify + report
        python build_sample_prep_features.py --dry-run  # report only, no writes
=============================================================================
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "IAJD_master", "datasets")

# Datasets to receive the sp_ block.  (paper_col, id_col, label)
XLSX_TARGETS = [
    ("IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx", "paper", "IAJD_num"),
    ("IAJD_Bioact_v13_clean.xlsx",             "paper", "IAJD_num"),  # canonical (== AUDIT_FIXED)
    ("IAJD_pKa_v21_final.AUDIT_FIXED.xlsx",    "source", "IAJD"),
    ("IAJD_pKa_v21_final.xlsx",                "source", "IAJD"),     # canonical (== AUDIT_FIXED)
]
NN_CSV = os.path.join(ROOT, "nn", "iajd_transfer_input.csv")

# ---------------------------------------------------------------------------
# 1. The Percec one-component IAJD STANDARD (in-vivo Luc-mRNA DNP) protocol.
#    Every value here is reported in / derived from the source-paper Methods.
# ---------------------------------------------------------------------------
STANDARD = {
    # --- formulation / assembly ---
    "sp_prep_method": "ethanol_injection",      # rapid manual injection of IAJD-EtOH into mRNA/buffer
    "sp_prep_method_code": 1,                    # 1=ethanol_injection, 2=microfluidic, 3=robot/LNP
    "sp_n_components": 1,                         # one-component IAJD (no helper lipid/chol/PEG)
    "sp_assembly_buffer": "acetate",
    "sp_assembly_buffer_mM": 10.0,
    "sp_assembly_pH": 4.0,                        # nominal assembly buffer pH  <-- VARIES (ja1c05813)
    "sp_IAJD_conc_final_mg_mL": 4.0,
    "sp_mRNA_conc_final_mg_mL": 0.10,             # <-- VARIES (ja1c05813 IAJD9 = 0.025)
    "sp_IAJD_mRNA_mass_ratio": 40.0,              # C_IAJD/C_mRNA  <-- KEY lever (ja1c05813 IAJD9=160)
    "sp_ethanol_vol_frac": 0.0500,               # 25 / 500.5 uL
    "sp_vortex_time_s": 5.0,
    "sp_assembly_temp_C": 23.0,                   # INFERRED (room temp; never stated for mixing step)
    "sp_dialyzed": 0,                             # <-- VARIES (ja1c05813 IAJD9 = 1)
    # --- mRNA cargo ---
    "sp_cargo": "Luc-mRNA",
    "sp_cargo_code": 1,                           # 1=firefly luciferase mRNA
    "sp_mRNA_polyA_nt": 101.0,                    # stated 2023+; same Weissman construct throughout
    # --- in-vivo assay ---
    "sp_assay_system": "in_vivo",
    "sp_assay_system_code": 1,                    # 1=in_vivo, 0=in_vitro
    "sp_inj_route": "retro-orbital",
    "sp_inj_route_iv": 1,                         # retro-orbital & tail-vein are both intravenous
    "sp_dose_mRNA_ug": 10.0,
    "sp_inj_volume_uL": 100.0,
    "sp_mouse_strain": "BALB/c",
    "sp_imaging_time_h": 4.0,                     # overridden per-row by existing T_hours when present
    "sp_luciferin_mg_kg": 150.0,
    # --- IAJD pKa measurement ---
    "sp_pKa_method": "ethanolic_HCl_titration_half_equiv",
    "sp_pKa_method_code": 1,                      # 1=ethanolic HCl titration, 2=TNS DNP-surface pKa
    "sp_pKa_IAJD_conc_mg_mL": 1.5,
    # --- provenance ---
    "sp_provenance": "paper_protocol",
    "sp_confidence": 0.85,
    "sp_notes": "",
}

# Per-paper overrides merged onto STANDARD.  Only the keys that differ.
PAPERS = {
    "ja1c05813": {
        "sp_provenance": "reported_per_row", "sp_confidence": 1.00,
        "sp_notes": "Library 1-6 screen; buffer/pH/conc/size/PDI/time read per-compound from Table S7. "
                    "In-vivo C_mRNA 0.10 (40:1) for all; assembly acetate pH 4.0 or 5.2 (per Table S7); "
                    "IAJD 9 = citrate pH 3.0 then PBS dialysis, C_mRNA 0.025 (160:1).",
    },
    "ja2c00273": {
        "sp_notes": "Tables S2/S3/S4 = in-vivo (used here). Table S1 = in-vitro HEK293T 125 ng/well "
                    "(not used). Imaging 4-7 h. Symmetric (S3) vs nonsymmetric (S4) alkyl chains.",
    },
    "pharmaceutics1501572": {
        "sp_notes": "Optimized organ-targeting series (lung/liver/spleen). SI states retro-orbital; "
                    "dataset labels route intravenous. poly(A)=101 nt, m1psi, cap1. Imaging 4-6 h.",
    },
    "ja3c07337": {
        "sp_notes": "Standard protocol confirmed from image-SI. Imaging 4-6 h. In-vivo only.",
    },
    "ja3c13569": {
        "sp_notes": "Standard confirmed from image-SI. CAVEAT: vortex stated as '5 or 20 s' "
                    "(sp_vortex_time_s=5 = primary); assembly pH rises 4.0->~4.8-5.0 after IAJD "
                    "injection; encapsulation >97%. In-vivo only.",
    },
    "ja1c09585": {
        "sp_notes": "Early (2021) paper. In-vitro HEK293T (125 ng/well) also present. SI imaging 4-6 h "
                    "- some dataset T_hours up to 10.25 are NOT supported by the SI (treat high "
                    "imaging_time with caution). poly(A) length not stated in this paper.",
    },
    "ja5c07232": {
        "sp_notes": "Standard confirmed. ALSO reports DNP-surface pKa by TNS fluorescence "
                    "(sp_pKa_method_code 2 available). Organ panel adds lymph node. poly(A)=101, m1psi, cap1.",
    },
    "bm4c01599": {
        "sp_provenance": "inferred_group_standard", "sp_confidence": 0.65,
        "sp_notes": "GA-Tris family. Synthesis/characterization-only SI (no written formulation or "
                    "in-vivo methods). Assembly acetate pH 4.0 + final 4.0/0.10 mg/mL from DLS captions; "
                    "mixing volumes/ratio/vortex INFERRED from Percec group standard. Dataset carries "
                    "main-paper in-vivo metadata (retro-orbital, BALB/c, 10 ug).",
    },
    "novel_2026": {
        "sp_provenance": "inferred_group_standard", "sp_confidence": 0.35,
        "sp_notes": "GA-Tris-345-EH dendron family (same dendron as IAJD-97). NOT from any cited paper - "
                    "reconstructed from a ChemDraw CDXML + bare flux spreadsheet. Spleen-dominant. "
                    "ALL prep params INFERRED from the Percec one-component standard - LOW confidence, "
                    "down-weight in modeling.",
    },
}

# ja1c05813 per-IAJD assembly buffer pH (acetate), from Table S7.
JA1C05813_PH40 = {33, 34, 35, 36, 37, 46, 47, 50, 51, 54}   # acetate pH 4.0; all others acetate pH 5.2
JA1C05813_CITRATE_DIALYSED = {9}                            # citrate pH 3.0 -> PBS dialysis; C_mRNA 0.025

# Universal-only block for unknown-paper rows (Percec IAJD facts true regardless of paper).
UNKNOWN_BLOCK = {
    "sp_prep_method": "ethanol_injection", "sp_prep_method_code": 1, "sp_n_components": 1,
    "sp_assembly_buffer": "acetate", "sp_assembly_buffer_mM": 10.0,
    "sp_assembly_pH": np.nan, "sp_IAJD_conc_final_mg_mL": np.nan, "sp_mRNA_conc_final_mg_mL": np.nan,
    "sp_IAJD_mRNA_mass_ratio": np.nan, "sp_ethanol_vol_frac": np.nan, "sp_vortex_time_s": np.nan,
    "sp_assembly_temp_C": np.nan, "sp_dialyzed": np.nan,
    "sp_cargo": "Luc-mRNA", "sp_cargo_code": 1, "sp_mRNA_polyA_nt": np.nan,
    "sp_assay_system": np.nan, "sp_assay_system_code": np.nan, "sp_inj_route": np.nan,
    "sp_inj_route_iv": np.nan, "sp_dose_mRNA_ug": np.nan, "sp_inj_volume_uL": np.nan,
    "sp_mouse_strain": np.nan, "sp_imaging_time_h": np.nan, "sp_luciferin_mg_kg": np.nan,
    "sp_pKa_method": "ethanolic_HCl_titration_half_equiv", "sp_pKa_method_code": 1,
    "sp_pKa_IAJD_conc_mg_mL": 1.5,
    "sp_provenance": "unknown_paper", "sp_confidence": 0.25,
    "sp_notes": "Source paper not identifiable; only universal Percec-IAJD facts filled "
                "(ethanol injection, one-component, Luc-mRNA, pKa titration). Paper-specific "
                "formulation/assay values left null.",
}

SP_COLS = list(STANDARD.keys())  # canonical column order

# Columns that genuinely VARY across the dataset (use these as model features);
# the rest are protocol-documentation constants within this corpus.
VARYING_COLS = [
    "sp_assembly_pH", "sp_assembly_buffer", "sp_mRNA_conc_final_mg_mL",
    "sp_IAJD_mRNA_mass_ratio", "sp_dialyzed", "sp_imaging_time_h",
    "sp_provenance", "sp_confidence",
]


def normalize_paper(paper, source):
    """Coalesce paper<-source, normalize aliases."""
    p = paper
    if pd.isna(p) or str(p).strip() == "" or str(p).lower() == "nan":
        p = source
    if pd.isna(p) or str(p).strip() == "" or str(p).lower() == "nan":
        return None
    p = str(p).strip()
    if p == "pharmaceutics":
        p = "pharmaceutics1501572"
    if "|" in p:                       # e.g. "ja3c13569|ja5c07232" -> first (both standard)
        p = p.split("|")[0]
    return p


def build_row_block(paper, iajd_num, existing_T):
    """Return the sp_ dict for one row."""
    if paper is None or paper not in PAPERS:
        # all 9 known Percec papers are keys in PAPERS; anything else -> unknown
        return dict(UNKNOWN_BLOCK)
    blk = dict(STANDARD)
    blk.update(PAPERS.get(paper, {}))

    if paper == "ja1c05813":
        try:
            n = int(iajd_num)
        except (TypeError, ValueError):
            n = None
        if n in JA1C05813_CITRATE_DIALYSED:
            blk["sp_assembly_buffer"] = "citrate"
            blk["sp_assembly_pH"] = 3.0
            blk["sp_dialyzed"] = 1
            blk["sp_mRNA_conc_final_mg_mL"] = 0.025
            blk["sp_IAJD_mRNA_mass_ratio"] = 160.0
        elif n in JA1C05813_PH40:
            blk["sp_assembly_pH"] = 4.0
        else:
            blk["sp_assembly_pH"] = 5.2   # Table S7 default for the library

    # Imaging time: prefer the row's recorded T_hours (per-compound), else paper default.
    if existing_T is not None and pd.notna(existing_T):
        try:
            blk["sp_imaging_time_h"] = float(existing_T)
        except (TypeError, ValueError):
            pass
    return blk


def apply_to_frame(df, paper_col, id_col, t_col):
    """Compute sp_ columns for a dataframe; return a new dataframe with them added."""
    df = df.copy()
    # drop any prior sp_ columns (idempotent re-run)
    df = df.drop(columns=[c for c in df.columns if c.startswith("sp_")], errors="ignore")

    source_col = "source" if "source" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        paper = normalize_paper(r.get(paper_col), r.get(source_col) if source_col else None)
        T = r.get(t_col) if (t_col and t_col in df.columns) else None
        rows.append(build_row_block(paper, r.get(id_col), T))
    sp = pd.DataFrame(rows, index=df.index)[SP_COLS]
    out = pd.concat([df, sp], axis=1)
    return out


def verify(df, label):
    print(f"\n----- VERIFY: {label}  ({len(df)} rows) -----")
    # coverage
    filled = {c: int(df[c].notna().sum()) for c in SP_COLS}
    print("  provenance:", df["sp_provenance"].value_counts(dropna=False).to_dict())
    print("  confidence:", {round(k, 2): int(v) for k, v in df["sp_confidence"].value_counts(dropna=False).items()})
    print("  sp_assembly_pH:", df["sp_assembly_pH"].value_counts(dropna=False).to_dict())
    print("  sp_IAJD_mRNA_mass_ratio:", df["sp_IAJD_mRNA_mass_ratio"].value_counts(dropna=False).to_dict())
    print("  sp_dialyzed:", df["sp_dialyzed"].value_counts(dropna=False).to_dict())
    print("  sp_assay_system:", df["sp_assay_system"].value_counts(dropna=False).to_dict())
    nulls = {c: len(df) - n for c, n in filled.items() if n < len(df)}
    print("  cols with nulls:", nulls if nulls else "none (full coverage)")
    # cross-check: ja1c05813 assembly_pH vs measured pH_sample (sanity, not equality)
    if "pH_sample" in df.columns and "paper" in df.columns:
        j = df[df["paper"] == "ja1c05813"]
        if len(j):
            lo = j[j["sp_assembly_pH"] == 4.0]["pH_sample"]
            hi = j[j["sp_assembly_pH"] == 5.2]["pH_sample"]
            print(f"  ja1c05813 sanity: assembly pH4.0 -> measured pH_sample median {lo.median():.2f} "
                  f"(expect ~4.5-5.1); assembly pH5.2 -> median {hi.median():.2f} (expect ~6.5-7.8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute + verify, no file writes")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # dump protocol source-of-truth for audit
    audit = {
        "generated_utc": stamp,
        "standard_protocol": STANDARD,
        "paper_overrides": PAPERS,
        "ja1c05813_assembly_pH": {"acetate_pH4.0": sorted(JA1C05813_PH40),
                                  "citrate_pH3.0_dialysed": sorted(JA1C05813_CITRATE_DIALYSED),
                                  "default": "acetate_pH5.2"},
        "varying_columns": VARYING_COLS,
        "all_sp_columns": SP_COLS,
    }
    if not args.dry_run:
        with open(os.path.join(ROOT, "sample_prep_protocols.json"), "w") as fh:
            json.dump(audit, fh, indent=2, default=str)
        print("wrote sample_prep_protocols.json")

    outputs = {}  # fname -> (df_with_sp, id_col)
    for fname, paper_col, id_col in XLSX_TARGETS:
        path = os.path.join(DS, fname)
        if not os.path.exists(path):
            print(f"!! missing {path} - skipped")
            continue
        df = pd.read_excel(path)
        t_col = "T_hours" if "T_hours" in df.columns else None
        out = apply_to_frame(df, paper_col, id_col, t_col)
        outputs[fname] = (out, id_col)
        verify(out, fname)
        if not args.dry_run:
            bak = path.replace(".xlsx", ".PRE_SAMPLEPREP.xlsx")
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            out.to_excel(path, index=False)
            print(f"  WROTE {fname}  (+{len(SP_COLS)} sp_ cols)  backup={os.path.basename(bak)}")

    # ---- NN transfer CSV: it has `iajd_num` but no paper column. Build an
    #      IAJD-number -> sp_block lookup from the computed Bioact (then pKa)
    #      outputs and map it on (every prep value is per-paper+per-IAJD, so the
    #      lookup is unambiguous). ----
    if os.path.exists(NN_CSV):
        nn = pd.read_csv(NN_CSV)
        idc = next((c for c in ("iajd_num", "IAJD_num", "IAJD", "IAJD_id") if c in nn.columns), None)
        print(f"\nNN csv: {len(nn)} rows, id_col={idc}, cols={len(nn.columns)}")
        if idc is not None:
            lookup = {}  # int IAJD id -> sp_ dict  (Bioact first, then pKa fills gaps)
            for src_fname in ("IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx", "IAJD_pKa_v21_final.AUDIT_FIXED.xlsx"):
                if src_fname not in outputs:
                    continue
                sdf, sidc = outputs[src_fname]
                for _, r in sdf.iterrows():
                    try:
                        k = int(r[sidc])
                    except (TypeError, ValueError):
                        continue
                    if k not in lookup:
                        lookup[k] = {c: r[c] for c in SP_COLS}
            nn = nn.drop(columns=[c for c in nn.columns if c.startswith("sp_")], errors="ignore")
            sp_rows = []
            for _, r in nn.iterrows():
                try:
                    k = int(r[idc])
                except (TypeError, ValueError):
                    k = None
                sp_rows.append(lookup.get(k, {c: np.nan for c in SP_COLS}))
            sp = pd.DataFrame(sp_rows, index=nn.index)[SP_COLS]
            merged = pd.concat([nn, sp], axis=1)
            n_match = int(merged["sp_provenance"].notna().sum())
            print(f"  matched {n_match}/{len(merged)} NN rows to an sp_ block")
            verify(merged, "nn/iajd_transfer_input.csv")
            if not args.dry_run:
                bak = NN_CSV.replace(".csv", ".PRE_SAMPLEPREP.csv")
                if not os.path.exists(bak):
                    shutil.copy2(NN_CSV, bak)
                merged.to_csv(NN_CSV, index=False)
                print(f"  WROTE nn/iajd_transfer_input.csv  (+{len(SP_COLS)} sp_ cols)")
        else:
            print("  !! NN csv has no recognizable IAJD id column - NOT updated")
    else:
        print(f"\nNN csv not found at {NN_CSV} - skipped")

    print("\nDONE." + ("  (dry-run, no files written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
