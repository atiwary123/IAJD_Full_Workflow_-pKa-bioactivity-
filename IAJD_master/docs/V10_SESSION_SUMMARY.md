# v10 Session Summary — pharm1572 Expansion

## Quick Status

**v09 → v10 dataset improvement:**
- Rows: 210 → 351 (+67%)
- Unique IAJDs: 156 → 254 (+63%)
- Training-eligible: 147 → 274 (+86%)
- Honest LOO α-routed MAE (collapsed, identical features): 0.4495 → **0.3982** (−11.4%)

## Apples-to-apples LOO comparison

Same MIN_FEATURES (Block A subset, 20 features), same per-family α routing, same leakage-aware delta-pair LOO, replicates collapsed to per-IAJD mean log_flux:

| Metric | v09 collapsed (n=93) | v10 collapsed (n=187) | Δ |
|--------|----------------------|------------------------|------|
| DIRECT MAE | 0.5260 | 0.4532 | −0.0728 (−14%) |
| DELTA MAE | 0.4750 | 0.4212 | −0.0538 (−11%) |
| **α-ROUTED MAE** | **0.4495** | **0.3982** | **−0.0513 (−11.4%)** |

### Per-family α-routed MAE
| Family | v09 | v10 | Δ |
|--------|-----|-----|---|
| sSS-Nonsym | 0.4308 (n=41) | 0.3974 (n=135) | −7.8% |
| PE-Gallic | 0.6871 (n=22) | 0.5802 (n=23) | −16% |
| GA-Tris | 0.3312 (n=10) | 0.2787 (n=10) | −16% |
| PE-Tris | 0.2428 (n=15) | 0.2062 (n=14) | −15% |
| Dialkoxybenzyl | 0.4152 (n=5) | 0.3607 (n=5) | −13% |

## Rich-feature LOO (Block A + 3D + v21-derived)

n=175 unique IAJDs (12 lose v21-derived features):
- DIRECT 0.4453 / DELTA 0.4198 / **α-ROUTED 0.4192**

The plain Block-A version (n=187) beats the rich-feature version (n=175) — feature value < dataset size at this scale, consistent with the "n<300 means simpler features win" principle.

## Pharm1572 extraction (155 measurements, 143 unique IAJDs)

### SI authoritative tables extracted
- 98 IAJDs pKa values from SI text (lines 7590-7620)
- 78 IAJDs DLS size/PDI from SI captions (early IAJDs 112-140 don't have explicit SI captions)

### Validated transcriptions (cross-checked vs SI)
- **Figure 4** (4 rows × ~20 = 78 nsSS IAJDs): screening total flux + DLS + pKa
  - Row 1: 18/19 pKa match perfectly, 3 size mis-reads auto-corrected from SI
  - Row 2: 19/20 perfect, 1 PDI mistype auto-corrected
  - Row 3: 20/20 perfect zero issues
  - Row 4: 19/19 perfect zero issues
- **Figure 6** (20 spleen-targeting): EXACT replicate values, all match SI
- **Figure 8** (15 liver-targeting): EXACT replicate values
- **Figure 11** (20 lung-targeting incl. IAJD 33 = 1.44×10⁸ and IAJD 78 = 1.12×10⁹): 13/13 v21-overlap exact
- **Figure 12** (20 lung with **per-organ flux** — first per-organ data in dataset)
- **Figure 13** (22 sSS IAJDs): 22/22 perfect

### Cross-figure validation
- Caught Fig 4 IAJD 186 flux exponent OCR error (10⁷ → 10⁸) by comparing against Fig 6's 1.52×10⁸ for same compound

### Replicate accounting
- Fig 4 = single screening run for 78 compounds
- Fig 6/8/11 = replicated triplicate confirmatory measurements (different mice from Fig 4)
- Both kept as separate rows for biological-replicate noise floor estimation

## SMILES construction & MALDI validation (22 → 14 ready, 9 deferred)

### ✓ Built and MALDI-validated (Δ < 0.6 Da)
| IAJD | Architecture | SMILES |
|------|--------------|--------|
| 66 | sSS-C12-MP | `CCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCCCCCC)c1` |
| 71 | sSS-C12-HP | `CCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCCCCCCCCCCCC)c1` |
| 74 | sSS-C14-MP | `CCCCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCCCCCCCC)c1` |
| 76 | sSS-EH-MP | `CC(CC)CCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCC(CC)CCCC)c1` |
| 77 | sSS-C9-MP | `CCCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCCC)c1` |
| 87 | sSS-EH-HP | `CC(CC)CCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCC(CC)CCCC)c1` |
| 88 | sSS-C8-MP | `CCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCC)c1` |
| 91 | sSS-C10-HP | `CCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCCCCCCCCCC)c1` |
| 95 | sSS-C4COOEH-HP | `CCCCC(CC)COC(=O)CCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCC(=O)OCC(CC)CCCC)c1` |
| 98 | sSS-C18-MP | `CCCCCCCCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCCCCCCCCCCCC)c1` |
| 99 | sSS-C18-HP | `CCCCCCCCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCCCCCCCCCCCCCCCCCC)c1` |
| 267 | sSS-C9-HP | `CCCCCCCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCCCCCCCCC)c1` |
| 269 | sSS-C13-MP | `CCCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(C)CC1)cc(OCCCCCCCCCCCCC)c1` |
| 270 | sSS-C13-HP | `CCCCCCCCCCCCCOc1cc(COC(=O)CCCN1CCN(CCO)CC1)cc(OCCCCCCCCCCCCC)c1` |

### ⏸ Deferred (G1-Janus dendrimers)
9 compounds (33, 110, 111, 131, 155-159) are 2nd-generation amphiphilic Janus dendrimers (multi-arm tetraethyleneglycol-linked dendritic architecture, MW 1350-1500, formulas C76-C86 with 18 oxygens). Construction requires careful structure decode from Schemes S in ja2c00273/pharm1572 SI; deferred to a dedicated G1-Janus pipeline session.

### 3D conformer ensemble computed
ETKDGv3 + MMFF94 with `useRandomCoords` fallback (needed for IAJD 95). All 14 new compounds have:
- N_LowE_Conformers (within 10 kcal/mol of E_min)
- Rg_3D, Asphericity_3D
- %V_Bur_max, %V_Bur_mean (4000-pt MC, 3.5 Å sphere on basic-N centers)
- N_basic_N_3D

## Files saved (persistent at /home/claude/pharm1572_extract/)

### Validation infrastructure
- `si_pka_table.pkl/.csv` — 98 IAJDs pKa from SI (authoritative)
- `si_dls_table.pkl/.csv` — 78 IAJDs DLS from SI (authoritative)

### Per-figure validated extractions
- `fig{4_row1-4, 6, 8, 11, 12, 13}_validated.pkl`

### Master tables
- `pharm1572_combined.pkl/.xlsx` — 155 rows raw extraction
- `pharm1572_aug_with_features.pkl` — with SMILES + Block A features
- **`IAJD_Bioact_v10.pkl/.xlsx`** — concatenated v09 + pharm1572 (356 rows)
- **`IAJD_Bioact_v10_clean.pkl/.xlsx`** — after B1-B7 cleanup (351 rows, 274 train-eligible)

### SMILES & 3D
- `sss_smiles_built.pkl` — 14 newly-built MALDI-validated SMILES
- `new_iajd_3d.pkl` — 3D features for the 14 new IAJDs
- `deferred_iajds.pkl` — 9 G1-Janus dendrimer registry

### LOO test artifacts
- `v10_loo_preds.npy`, `v10_y.npy`, `v10_families.npy` — DIRECT-only LOO
- `v10_alpha_*_preds.npy` — α-routed LOO outputs
- `v10_collapsed_*.npy` — collapsed-replicates LOO outputs
- `v10_rich_*.npy` — rich-feature LOO outputs

### Scripts
- `cleanup_b1_b7_v10.py` — B1-B7 cleanup pipeline for v10
- `v10_baseline_test.py` — DIRECT-only LOO MAE
- `v10_alpha_routing_test.py` — α-routing LOO with strict leakage avoidance
- `v10_alpha_collapsed.py` — α-routing on collapsed replicates
- `v10_rich_features.py` — α-routing with 3D + v21-derived features

## Next session pickup points

1. **Run full Stage A/B/C cascade on v10** with proper labels and per-family α routing to get production-quality metrics
2. **Refit conformal intervals** on honest v10 LOO residuals (v1.0 PIs are untrustworthy)
3. **Build SMILES for the 9 deferred G1-Janus dendrimers** — would require dedicated session reading Schemes S in ja2c00273 SI carefully
4. **Update bioact_v1.1 deployment bundle** with v10 model + retrain
5. **Address open IAJD 82 SMILES** (still TENTATIVE from session 2)
