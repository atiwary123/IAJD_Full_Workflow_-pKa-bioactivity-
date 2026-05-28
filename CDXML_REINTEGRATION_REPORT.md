# cdxml IAJD Reintegration — 2026-05-28

## Summary

Eight GA-Tris IAJDs from `lion_repo/scripts/IAJD for Comp.cdxml`
(IAJDs 347, 348, 365, 366, 367, 369, 372, 373) have been reintegrated
into the training set after being removed on 2026-05-27.

## Validation: cdxml → SMILES → features

The cdxml file was parsed directly (custom CDXML→RDKit adapter). The
parser found 10 fragments and 10 IAJD text labels (IAJD 365 appears 3
times because it is drawn with `(n = 3)` regio-isomer variants). After
canonical-SMILES-based fragment-to-label assignment, **all 8 unique
IAJDs produce canonical SMILES that exactly match the pre-removal
backup `IAJD_Bioact_v13_clean.xlsx.bak_pre_novel_removal`**.

| IAJD | Architecture            | SMILES (canonical)                                                                |
|------|-------------------------|-----------------------------------------------------------------------------------|
| 347  | GA-tris-345-EH-4C-H2EPRZ | `CCCCC(CC)CCOc1cc(COC(=O)CCCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCC`        |
| 348  | GA-tris-345-EH-4C-HPRZ   | `CCCCC(CC)CCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCC`           |
| 365  | GA-tris-345-EH-3C-H2EPRZ | `CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC`        |
| 366  | GA-tris-345-EH-3C-HPRZ   | `CCCCC(CC)CCOc1cc(COC(=O)CCN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCC`            |
| 367  | GA-tris-345-EH-2C-HPRZ   | `CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCC`             |
| 369  | GA-tris-345-EH-2C-H2EPRZ | `CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC`         |
| 372  | GA-tris-345-EH-5C-HPRZ   | `CCCCC(CC)CCOc1cc(COC(=O)CCCCN2CCN(CCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC`         |
| 373  | GA-tris-345-EH-5C-H2EPRZ | `CCCCC(CC)CCOc1cc(COC(=O)CCCCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC`      |

## What changed

### Datasets
| File                                                | Before | After | Delta |
|-----------------------------------------------------|--------|-------|-------|
| `IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx`   | 265    | 273   | +8    |
| `IAJD_master/datasets/IAJD_Bioact_v13_clean.pkl`    | 265    | 273   | +8    |
| `IAJD_master/datasets/IAJD_pKa_v21_final.xlsx`      | 278    | 286   | +8    |
| `v11_pka_flux/predicted_pka_cache.csv`              | 265    | 273   | +8    |
| `IAJD_master/bundles_caches/admet_cache_v13.json`   | 247    | 255   | +8    |
| `IAJD_master/bundles_caches/lion_cache_v13.json`    | 252    | 260   | +8    |
| `agile_embeddings_bioact.npy`                       | 265×512| 273×512| +8 rows |
| `agile_embeddings_pka.npy`                          | 278×512| 286×512| +8 rows |
| `cpp_features_bioact.csv`                           | 265×23 | 273×23 | +8 rows |

### Features computed for each new IAJD
- 2D RDKit (61 descriptor columns): ExactMolWt, MolLogP, TPSA, LabuteASA,
  FractionCSP3, Chi0v/Chi1v, HallKierAlpha, NumTertiaryAmines, HasPiperazine,
  GasteigerN_min/max, Hydrophobic_Index, Polar_Surface_Ratio,
  Inductive_Effect_Strength, Linker_Length, Taft_Steric_Sum,
  HBD_HBA_Ratio, Desolvation_Proxy, etc.
- 3D RDKit (ETKDGv3 + MMFF94, 5 conformers, all converged): Pct_V_Bur_max,
  Pct_V_Bur_mean, Rg_3D, Asphericity_3D, E_min_3D, N_LowE_Conformers,
  N_basic_N_3D.
- LION (chemprop, 6 tissues × Z-score + OOD): real subprocess prediction,
  not RDKit proxy.
- ADMET-AI (10 columns): real subprocess prediction.
- AGILE (512-dim graph embedding): from the pretrained encoder.
- CPP (23 columns): geometry + Henderson-Hasselbalch protonation at pH
  5.5 / 6.5 / 7.4. All non-NaN.
- pKa (predicted): gradient-boosting baseline on the 278-row pKa training
  set; predictions 5.99–6.08 (resid_std on training residuals = 0.14;
  this is the in-sample residual scale, NOT a true held-out uncertainty —
  treat the pKa values as provisional until the full pKa primary model is
  rerun).

### Bioactivity values
The eight rows include the **measured** `log10_flux_*` values from the
backup file (lung, liver, spleen, LN, heart). These are the same
measurements that were in the v13 dataset before removal.

### Scripts un-gated
The `EXCLUDED_NOVEL_IAJDS = {347, 348, 365, 366, 367, 369, 372, 373}`
guard was removed from 9 scripts:

- `agile_embeddings.py`
- `compute_cpp.py`
- `iajd_neighbors.py`
- `iajd_v15.py`
- `pka_primary_model.py`
- `app.py`
- `IAJD_master/code/bioact_v14_pipeline.py`
- `v11_pka_flux/pka_flux_v11.py`
- `v11_pka_flux/build_m2_bundle.py`

## What is NOT yet retrained (honest disclosure)

These pre-existing trained bundles were fitted on a 335-row historical
training snapshot and have **not** been retrained with the 8 reintegrated
IAJDs:

- `IAJD_master/bundles_caches/bioact_v14_3_bundle.pkl`
- `IAJD_master/bundles_caches/bioact_stacker_bundle.pkl`
- `v11_pka_flux/m2_bundle.joblib`

These bundles are stale w.r.t. the new training rows. Re-running
`IAJD_master/code/bioact_v14_pipeline.py`, `IAJD_master/code/iajd_pka_v52.py`,
and `v11_pka_flux/build_m2_bundle.py` will produce bundles that include
the 8 reintegrated IAJDs (the guards have been removed). Retraining was
not done in this pass because it requires significant compute time
(~1–2 h per bundle) and was out of scope for the integration step.

The auxiliary frozen .npy files `agile_embeddings_v14_train.npy` and
`cpp_features_v14_train.npy` (both 335 rows) are tied 1:1 to the existing
bundle's `smis_train` and are intentionally **left untouched** so the
bundle still loads without index drift.

## Snapshot

A full pre-integration snapshot lives at
`workflow_snapshot_2026-05-28_pre_cdxml_reintegration/`. It contains the
pre-change xlsx/pkl/npy artifacts plus the 9 scripts in their guarded form.
Reverting is a directory copy.

## Reproducibility

```
.venv/bin/python extract_cdxml_iajds.py    # cdxml → cdxml_iajds_extracted.csv (validation)
.venv/bin/python integrate_cdxml_iajds.py  # xlsx + pkl + pKa + report
.venv/bin/python agile_embeddings.py       # regenerate AGILE bioact + pKa embeddings
.venv/bin/python compute_cpp.py            # regenerate cpp_features_bioact.csv
```

LION + ADMET cache extension was done as a one-shot via
`extend_caches.predict_*_for_smiles([...])` against the canonical SMILES
list above.
