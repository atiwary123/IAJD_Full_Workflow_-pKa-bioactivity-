# IAJD Dataset Deep Audit — Findings Log

## Method established
- Dataset IAJD id == Percec paper "IAJD N" label (proven: 50/52 pKa values in
  ja1c05813 match SI Tables S12-S17 exactly).
- SMILES ground truth = molecular formula reported in each paper SI (MALDI-TOF).
  Phase 0 proved dataset MolFormula column == dataset SMILES, so paper-formula ==
  dataset-formula ⟹ SMILES correct (modulo constitutional isomers).
- Engine: audit_work/fm2.py extracts SI neutral formulas (adduct-corrected) and
  tests dataset formula + structural-fix hypotheses.

## Phase 0 (internal consistency) — datasets are internally clean
- 0/278 pKa rows, 0/273 bioact rows have SMILES⇄descriptor mismatch (features fresh).
- 20 pKa rows have NO stored formula/MW; 26 rows have NO source paper; 32 lack
  linker/head/linkage fields.
- Cross-dataset HEAD-GROUP conflicts (bioact vs pKa file) for IAJD 248, 273, 297.
- Duplicate identical SMILES w/ different pKa: 287/292 (6.30 vs 6.48), 290/291
  (6.50 vs 6.42) — all ja3c07337, formula col empty. Need paper.
- 10 flux⇄log10_flux inconsistencies; 5 pKa⇄pKa_paper diffs in bioact set.

## CONFIRMED SMILES ERRORS — ja1c05813 (PE-Gallic, IAJD 1-54)
Systematic, family-wide cap errors (verified by SI formula match):
1. **OH-instead-of-OCH3**: non-ionizable arms that paper caps as methyl ether
   (-O(EO)3-OCH3) were modeled with free -OH. Missing CH2 per capped arm.
   e.g. IAJD 1: dataset C62H108N2O16 -> +2CH2 -> C64H112N2O16 (SI). IAJD 1-5,44,45 ✓.
2. **phenylacetate-instead-of-OBn**: arms paper caps as benzyl ETHER (-O(EO)3-O-CH2-Ph)
   were modeled as phenylacetate ESTER (-O(EO)3-O-C(=O)-CH2-Ph). Extra C=O + wrong
   functional group. e.g. IAJD 6: dataset C78H120N2O18 -> -C2O2 -> C76H120N2O16 (SI).
   Affects IAJD 6-9 (48f-i, OBn caps) and the OPMB-capped variants.
- EO arm length (3 units) and DMBA head (OCO(CH2)3NMe2) are CORRECT.
- Library 5 (twin-twin, IAJD 10-18) & Library 6 use different architecture — TBD.

## pKa minor discrepancies (ja1c05813) — curator took a single replicate not the Avg
- IAJD 38: dataset 6.32 vs SI Avg 6.38 (reps 6.43/6.32)
- IAJD 45: dataset 5.97 vs SI Avg 5.93 (reps 5.88/5.98)
- IAJD 47: dataset 6.64 vs SI Avg 6.59 (reps 6.63/6.54)
