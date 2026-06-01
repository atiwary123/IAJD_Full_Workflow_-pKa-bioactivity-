# IAJD Dataset Deep Audit — Report (2026-06-01)

Audit of `IAJD_master/datasets/IAJD_pKa_v21_final.xlsx` (278 rows) and
`IAJD_Bioact_v13_clean.xlsx` (273 rows) against the original Percec-lab source papers
(SI PDFs restored to `IAJD_master/source_papers/`). Priority per request:
**SMILES → pKa → bioactivity → structural inferences.**

## Headline
1. **pKa data is excellent** — ~155 values cross-checked against paper tables; only **5 minor
   corrections**, all ≤0.06 (a single replicate had been used instead of the table average).
2. **SMILES are systematically broken in the PE-Gallic family (`ja1c05813`, IAJD 1–54)** and
   essentially correct elsewhere. **18 fixed & re-validated**; 33 need reconstruction.
3. **Bioactivity is largely sound** — most flagged flux/log10 "inconsistencies" are a legitimate
   linear-vs-log replicate-averaging difference, not errors.
4. Several **label/duplicate errors** found and fixed/flagged (head groups; a SMILES-collapse of
   distinct compounds; cross-file conflicts; missing source attributions).

## Method
- Dataset `IAJD` id == the paper's "IAJD N" label — **proven** (49/52 `ja1c05813` pKa match SI
  Tables S12–S17 exactly; coincidence ≈ 0).
- SMILES ground truth = the SI's MALDI/HRMS molecular formula. Phase 0 proved stored `MolFormula`
  == stored SMILES, so *paper-formula == dataset-formula ⟹ SMILES correct* (modulo isomers).
- Tooling in `audit_work/`: `fm2.py` (adduct-tolerant SI formula extraction), `recap2.py`
  (deterministic, formula-validated cap fixer), `consolidate_ja1c05813.py`, `apply_fixes*.py`.

## SMILES — corroboration by paper
| Paper | rows | SMILES correct (SI formula) | notes |
|---|---|---|---|
| `ja2c00273` | 40 | **40/40** ✓ | |
| `pharmaceutics1501572` | 98 | **95/98** ✓ | 83, 89 = head MPRZ-label vs **HPRZ**-SMILES; 134 (C11 chain / SI-extraction) |
| `ja1c05813` (PE-Gallic) | 52 | 1 as-is + **18 fixed**; **33 need reconstruction** | see taxonomy below |
| `ja3c07337` (PE-Tris) | 20 | pKa-verified; **4 SMILES errors** (287/290/291/292) | image SI — formulas not machine-checked |
| 5 other image-only SIs | ~40 | not machine-checked | same families as validated papers → likely mostly correct |
| no `source` | 26 | partly attributed (92,100,101 → ja3c07337) | |

### `ja1c05813` PE-Gallic error taxonomy (the dataset's wrong cap encodes the intended one)
1. **Free –OH instead of –OCH₃** methyl ether (missing CH₂/arm). e.g. IAJD 1
   `C62H108N2O16`→`C64H112N2O16` (SI). [6]
2. **Phenylacetate *ester* instead of –OBn benzyl *ether*** (spurious C=O). e.g. IAJD 6
   `C78H120N2O18`→`C76H120N2O16` (SI). [12]
3. **Needs reconstruction [33]**: twin-twin Library 5 (10–18) & twin-mix Library 6 (26–29,38–43,
   46–54) use other architectures; plus **typos** (O–CH₂–O acetal `OCCOCCOCOC` in IAJD 30/31;
   ethyl-capped glycols in 9/24/33/43) and a **wrong linkage** (IAJD 9 ester, paper is amide).
   These were NOT guessed.

### `ja3c07337` SMILES-collapse error (confirmed)
Paper Table S1 lists **287 (6.30) and 292 (6.48)** — and **290 (6.50) and 291 (6.42)** — as
*distinct* compounds, but the dataset gave each pair **identical SMILES**. PE-Tris grid =
chain length (C6–C12) × head (MPRZ/HPRZ/diEG-piperazine); **MPRZ@C7 is missing**, so one of each
pair is mislabeled. pKa values are correct; the 4 SMILES need differentiation from the synthesis.

## pKa — cross-checked against paper tables (~155 values)
| Paper | matched-exact | corrected | not in table |
|---|---|---|---|
| `ja1c05813` (Tables S12–S17) | 49/52 | **38→6.38, 45→5.93, 47→6.59** | — |
| `ja2c00273` (Table S9) | **40/40** | — | — |
| `pharmaceutics1501572` (Table S1) | 90/98 | **266→6.54, 268→6.45** | 78,83,89,96,108,134 |
| `ja3c07337` (Table S1) | 25/27 | — | (93,288 use bm4c01599's value — cross-paper dup) |

## Bioactivity (273 rows)
- **9 of 10** `flux_total`⇄`log10_flux_total` "mismatches" have `n_replicates_averaged>1`: the
  flux is the arithmetic mean of replicates while log10 is the log-mean — **both valid, not an
  error** (log10(flux) > log10_flux, consistent). Only **IAJD 10** (single value; `log10=5.000`
  vs flux→4.918) looks like a rounded entry — flagged.
- `organ_dominant` "mismatches": mostly multi-organ qualitative labels (e.g. "liver+lung") not
  argmax. **IAJD 178** (dominant spleen, argmax liver) and **249** (dominant liver, argmax spleen)
  flagged for paper confirmation.
- Per-organ flux not yet cross-checked vs paper tables (mostly image/complex).

## Structural-inference / label errors
- `head_group`: IAJD **44**=PIP (labeled DMBA), **45**=MPRZ, **83/89**=HPRZ (labeled MPRZ).
- Cross-file SMILES conflict (bioact vs pKa file): **248, 273, 297** — bioact file aligned to the
  grid-consistent pKa-file value (MPRZ); confirm vs paper.
- `source` attributed: **92, 100, 101 → ja3c07337** (appear in its Table S1).

## Changes applied — to COPIES (canonical untouched; overnight physics run active)
`IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx`, `…Bioact…AUDIT_FIXED.xlsx`:
- **18 SMILES** corrected (PE-Gallic, formula re-validated vs SI) + MolFormula/ExactMolWt updated.
- **5 pKa** corrected (38,45,47,266,268). **3 source** attributions. **3 bioact SMILES** aligned.
- Dup pairs (287/290/291/292) flagged "NEEDS_DIFFERENTIATION". `audit_status` column marks every
  touched row; **their other feature columns (custom 2D, 3D, QM) are STALE and must be regenerated
  by the project pipeline** — deliberately NOT recomputed/fabricated.
- Ledgers: `audit_work/AUDIT_corrections_master.csv`, `audit_work/ja1c05813_corrections.csv`.

## Remaining work (honest)
1. **Reconstruct 33 `ja1c05813` SMILES** (twin-twin/twin-mix + typo/linkage specials) from Schemes
   S4/S9–S12 + Figure 2.
2. **Differentiate the 4 `ja3c07337` SMILES** (287/290/291/292) from the synthesis characterization.
3. **Image-only SIs** (`ja1c09585`, `ja3c07337`, `ja3c13569`, `ja5c07232`, `bm4c01599`,
   `bm4c01107`): SMILES-formula + pKa verification needs page-image reads (no text layer; local
   tesseract/Leptonica is broken). pKa tables are short and readable this way.
4. **Bioactivity**: per-organ flux vs paper tables; resolve IAJD 10, 178, 249.
5. **pharmaceutics 83/89/134** and remaining no-source rows.
6. Confirm head-conflict resolutions 248/273/297 against the paper structures.

## Reproduce
```
source .venv/bin/activate
python3 audit_phase0.py
python3 audit_work/recap2.py "<si_txts>" ja1c05813
python3 audit_work/consolidate_ja1c05813.py
python3 audit_work/apply_fixes.py && python3 audit_work/apply_fixes2.py
```
