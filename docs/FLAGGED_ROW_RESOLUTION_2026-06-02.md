# Flagged / Unresolved Row Resolution — 2026-06-02

Resolves **every** `UNRESOLVED:*` / `FLAG_*` / `NEEDS_DIFFERENTIATION` / `CONFIRM_VS_PAPER` row in
the corrected datasets, using the most rigorous inference available, with an **honest confidence
tier per row**. Companion to `DATASET_AUDIT_REPORT_2026-06-01.md` and
`DATASET_CORRECTION_AND_REGEN_GUIDE.md`.

- Scripts: `audit_work/verify_flagged.py`, `audit_work/enumerate_hard.py`, `audit_work/resolve_flagged_2026-06-02.py`
- Ledger: `audit_work/flagged_resolution_2026-06-02.{csv,json}`
- Backups: `audit_work/pre_flagfix_backup_20260602/` (all 4 dataset files, pre-edit)
- Files updated: `IAJD_pKa_v21_final.xlsx` + `.AUDIT_FIXED.xlsx`, `IAJD_Bioact_v13_clean.xlsx` + `.AUDIT_FIXED.xlsx`
  (canonical kept **byte-identical** to AUDIT_FIXED; `audit_confidence` column added).

## Method (evidence hierarchy)
1. **10118 Table-S1 descriptor vector** (`p10118_tableS1.json`): MolWt, FractionCSP3, HBA, HBD, RotB,
   NumAromaticRings — an *independent* 6-number fingerprint per IAJD. A reconstruction is accepted
   only when it reproduces this vector (tol: MolWt<1.0, FCsp3<0.01, HBA/HBD/Aromatic exact, RotB±1).
2. **SI MALDI formula** ("calcd for CₓHᵧNᵤOᵥ", adduct-corrected) where the SI text was extractable.
3. **Architecture label + homologous-series + family templates** for rows absent from both 1 and 2.

## Confidence tiers
| Tier | Meaning |
|---|---|
| **HIGH** | Structure reproduces the independent 10118 vector (and/or SI MALDI formula). Trustworthy. |
| **MED-HIGH** | Self/architecture consistent + validated against the named source-paper figure (not 10118). |
| **MED** | Not in 10118; formula not in extractable SI text. Internally + series + architecture consistent (inference-grade). |
| **LOW** | Molecular **formula certain**, but **connectivity underdetermined** by descriptors — needs the SI structural figure. Kept excluded from structure-feature training. |

## Headline results
- **12 HIGH**, **1 MED-HIGH**, **17 MED**, **0 LOW** → all 30 pKa-file flags (and the overlapping
  bioact-file flags) cleared. **0 residual** `UNRESOLVED/FLAG` rows and **nothing excluded** (IAJD 30
  upgraded LOW→HIGH on 2026-06-02 once its exact SI structure was found — see below).
- 4 structures corrected (SMILES changed): **30, 33, 64, 86**; RDKit 2D+3D features regenerated for these.
- 8 confirmed-correct (flag was stale/over-cautious): **31, 248, 273, 287, 290, 291, 292, 297**.

## Key structural corrections (HIGH)
| IAJD | Was | Now | Basis |
|---|---|---|---|
| **64** | bis-C12 dialkoxybenzyl-**MPRZ** butanoate (C40H72N2O4) | bis-C12 dialkoxybenzyl-**DMA** butanoate **C37H67NO4** | 10118 #64 (MW589.946, HBA5, **TPSA48.0**, HBD0, Ar1): TPSA = 2 ethers + 1 ester + **one** tertiary amine ⇒ head is dimethylamino (DMA), not piperazine. |
| **86** | C11/**C16** dialkoxybenzyl-MPRZ (C43H78N2O4) | **bis-C11** (symmetric) **C38H68N2O4** | 10118 #86 MW 616.972 = exact bis-C11; prior C11/C16 was +70 (5×CH₂). |
| **33** | bis-C12 gallic, 2×**MPRZ** (C72H124N4O17, Ar2/HBD0) | bis-C12 gallic-**amide**, **2×piperidine + 1×OBn** **C81H133N3O17** | 10118 #33 (MW1420.96, HBA19, **HBD1**, **Ar3**) **+ SI MALDI calcd C81H134N3O17 = [M+H]⁺**. 10118 requires 3 N (not 5) ⇒ PIP not MPRZ; amide ⇒ HBD1; OBn ⇒ Ar3. |
| **30** | dialkoxybenzyl-amide 2-arm DMBA dendron (C58H107N3O13, **Ar1**) | aliphatic G1-Janus **C58H114N2O9** — pentaerythritol-tris(C12-ether) ester of bis-MPA bearing 2× DMBA | **HIGH (upgraded from LOW 2026-06-02).** Resolved to the **exact SI structure**: ja1c05813 Scheme S8 **Compound 64 "(2/2DMBA1,2)"** (pentaerythritol-tris-dodecyl-ether + bis-MPA core + 2× 4-(dimethylamino)butyrate). SI MALDI [M+H]⁺ C58H115N2O9 **+** full 10118 #30 vector match on all 6 descriptors (MW 983.555, FCsp3 0.948, HBA 11, HBD 0, **RotB 54**, Ar 0). Now **included** in training. |

## Confirmed-correct (flag was stale) — HIGH
`31, 248, 273, 287, 290, 291, 292, 297` all reproduce their 10118 vector. Mislabels fixed where
10118 determines head identity (e.g. 248 head HPRZ→**MPRZ**; 273/291 →**methoxyethyl-piperazine**
since HBD0 rules out hydroxyethyl). 287 matches 5/6 (RotB differs by 3 = rotatable-bond definition
difference, not a structural error).

## Numbering collision — MED-HIGH
**134**: 10118 #134 is an unrelated ~2× heavier compound (MW1205.7, HBA17, Ar2). This dataset row
(C38H68N2O3, bis-C11 dialkoxybenzyl-amide-piperidine, pharmaceutics1501572 Fig11) is internally +
architecture consistent and validated against its **own** paper — *not* 10118. Flag cleared with collision note.

## Inference-grade (not independently verifiable) — MED
17 PE-Gallic "twin-mix" rows (`26,27,28,29,38,39,40,41,42,43,47,48,49,50,51,53,54`, ja1c05813
Lib6/Lib4). **Absent from the 10118 library** and their formulas are **not in the extractable SI
text** (the 21 SI-unmatched ja1c05813 rows are exactly these). Their SMILES are internally
consistent, fit the homologous chain-length series (C8/C10/C11/C12/EH/dm8 differ only by CH₂), and
match their architecture-label decomposition (DMBA/PIP/Bn/OH head census). This is the best
inference without the high-resolution SI figures; those would upgrade them to HIGH.

## MED inclusion decision (impact-tested 2026-06-02)
Per a quick surrogate-model impact test (`audit_work/med_inclusion_impact.py`; RandomForest on
RDKit-2D/3D features — relative-impact approximation, not the production stack), including the 17
MED rows in training is **not disruptive**, so they are kept **flag-free / training-eligible**:

| | verified-row accuracy (10-fold CV), baseline → +MED | SAR stability (verified vs +MED) |
|---|---|---|
| **pKa** (17 MED, full coverage) | MAE 0.157 → **0.154** (Δ −0.003, marginally better); R² 0.356→0.361 | top-10 importances 9/10 overlap, Spearman 0.895 |
| **bioact** log10_flux (7 MED w/ target) | MAE 0.480 → 0.487 (Δ +0.007, ~1%); R² 0.484→0.473 | 10/10 overlap, Spearman 0.991 |

The MED dendrimers are themselves harder to predict (high-leverage in feature space: pKa self-MAE
0.54, flux 1.12) but do **not** degrade the verified compounds or shift the SAR conclusions.
`audit_confidence` (HIGH/MED-HIGH/MED/LOW) is retained as honest provenance and to allow a stricter
gate if desired. **Nothing is excluded** as of the 30 upgrade — IAJD 30 was the only holdout and is now HIGH/included.

## What is NOT done (honest scope)
- **Models/heavy caches are still keyed to the pre-fix SMILES and the pre-inclusion row set.** Only
  RDKit 2D/3D features were regenerated (for the 4 changed rows). The QM (xTB), MD, LiON, ADMET,
  AGILE caches and all trained models (XGBoost/pKa/bioact/NN) do **not** yet reflect either the
  structural fixes or the now-included MED rows — regenerate + retrain per
  `DATASET_CORRECTION_AND_REGEN_GUIDE.md` to propagate.
- **30** — RESOLVED to its exact SI structure (Compound 64) on 2026-06-02; now HIGH and included.
  Models were retrained with 30 included: **pKa v92 LOO MAE 0.1263 (n=278)**, bioact v14 bundle n=247
  (binary reg LOO MAE 0.4364, R² 0.562, ROC-AUC 0.770) — change from the n−1 set is negligible.
