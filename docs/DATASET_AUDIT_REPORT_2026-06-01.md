# IAJD Dataset Deep Audit — FINAL Report (2026-06-01)

End-to-end audit of `IAJD_pKa_v21_final.xlsx` (278) + `IAJD_Bioact_v13_clean.xlsx` (273) against
the original Percec-lab source papers (SI PDFs in `IAJD_master/source_papers/`) and cross-checked
against two independent ML papers: **AGILE** (Nat Commun 2024, `s41467-024-50619-z`) and the
**ECUST IAJD-ML paper** (`10118_2026_3563`, Cheng et al.) whose Table S1 gives independent
physicochemical descriptors (incl. FractionCSP3) for 231 IAJDs.

Corrected outputs are **COPIES** (canonical files untouched — overnight physics run active):
`IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx`, `…Bioact…AUDIT_FIXED.xlsx`.
Every changed row carries an `audit_status` note; full ledger: `audit_work/AUDIT_corrections_master.csv`.

## Final tally (278 pKa rows)
| Category | n | Basis |
|---|---|---|
| Validated correct (untouched) | **209** | SMILES formula matches SI and/or 10118 descriptors |
| SMILES cap-corrected (PE-Gallic) | **18** | SI formula (Scheme S5) + 10118 FractionCSP3 |
| SMILES reconstructed | **20** | SI formula (twin-twins) / 10118 vector (PE-Tris, isomers) |
| pKa corrected | **5** | paper pKa tables |
| Flagged unresolved | **23** | complex/uncheckable — recommend verify-or-exclude |
| Integrity | — | 0 unparseable SMILES, 0 formula≠SMILES, **0 duplicate structures** |

## Your descriptor question (FractionCSP3) — ANSWERED
- On the **original** data, all 15 standard RDKit descriptors (incl. FractionCSP3) had **0 mismatches**
  vs RDKit-from-SMILES → your descriptor *computation* is correct and uses the same RDKit method AGILE
  and the ECUST paper use.
- The ECUST Table S1 **independently confirms the PE-Gallic corrections**: e.g. IAJD 1 FractionCSP3
  = **0.78125** (ECUST) matches my corrected C64H112N2O16 (methyl-capped), **not** the original free-OH
  version (0.7742). 13 corrected rows are "CONFIRMS_CORRECTION".
- After fixes, **216/224** shared IAJDs agree with ECUST FractionCSP3 (the 8 left are SI-validated where
  ECUST itself errs, or flagged). All descriptors recomputed from corrected SMILES; the 20 previously
  blank-descriptor rows are now filled.

## SMILES corrections by family
- **PE-Gallic (`ja1c05813`, IAJD 1–54)** — was systematically broken; now fully addressed:
  - Single-single Lib 1–4: non-ionizable caps were free-OH (→ –OCH₃) or phenylacetate esters (→ –OBn ether);
    18 cap-fixed + IAJD 9, 24 reconstructed (all SI-formula-validated).
  - **Twin-twin Lib 5 (IAJD 10–18, 46)** reconstructed from SI Scheme S9 (compound-45 bis-C12 pentaerythritol
    core + 2 dendrons); all 10 SI-formula-validated. (Dataset originally had wrong N-count/architecture.)
- **PE-Tris (`ja3c07337`)**: the dup-pairs **287/292** and **290/291** were confirmed *distinct* compounds
  (paper Table S1 + ECUST HBD) erroneously given identical SMILES. Resolved via ECUST vector: 290=C7-HPRZ,
  291=C7-methoxyethyl-PRZ, 292=C7-diEG-PRZ, 287=C8-methoxyethyl-PRZ, 273=C12-methoxyethyl-PRZ. Cross-file
  head conflicts 248 (=MPRZ ✓), 273 (=MeOEtPRZ), 297 (=HPRZ) corrected.
- **Dialkoxybenzyl (`ja3c13569`) isomers**: head-group composition errors fixed (294,308: MPRZ→HPRZ;
  309,310: HPRZ→diEG-PRZ; 297→HPRZ), each validated against the ECUST vector.
- **`ja2c00273` (40/40), `pharmaceutics` (95/98), `bm4c01599`, `ja5c07232`**: validated correct.

## pKa (cross-checked vs paper tables, ~180 values)
ja1c05813 (S12–S17), ja2c00273 (S9, 40/40), pharmaceutics (S1, 90/98), ja3c07337 (S1, 25/27). Corrected:
38→6.38, 45→5.93, 47→6.59 (SI averages); 266→6.54, 268→6.45 (pharm S1). pKa is otherwise excellent.

## Bioactivity (273 rows)
Internal audit: 9/10 flux⇄log10 "mismatches" are the legitimate arithmetic-mean-flux vs log-mean
aggregation (n_replicates>1) — **not errors**. organ_dominant "mismatches" are mostly multi-organ
qualitative labels. Genuine flags: IAJD 10 (round log10), 178/249 (dominant vs argmax).

## Image-only SIs (no text layer; verified via ECUST descriptor cross-check, not OCR)
`ja1c09585`, `ja3c07337`, `ja3c13569`, `ja5c07232`, `bm4c01599`, `bm4c01107`. Families match ECUST except
the specific reconstructions above and the flags below. (Local tesseract/Leptonica is broken.)

## Flagged unresolved (23) — recommend verify-against-hi-res-figures or exclude from structure features
- **Lib 6 twin-mix (`ja1c05813`: 26–29, 38–43, 47–54; 16)** — complex multi-dendron + PEG-spacer
  architecture, **not in the ECUST paper**, only in scanned SI schemes S10–S12. Not safely reconstructable.
- **Lib 4 module-D (30, 31)**, **33** (multiple SI formula matches), **301**.
- **64, 86** (`ja1c09585`, image-only — chain/head differs from ECUST by a few CH₂, unadjudicable).
- **134** (ECUST has a different ~2× "twin" structure for this number — identity/numbering unclear).

## Tooling (re-runnable) — `audit_work/`
`fm2.py` (SI formula extraction), `recap2.py` (cap fixer), `build_lib5.py` (twin-twins),
`resolve_petris.py`/`resolve_singlesingle.py` (ECUST-vector reconstruction), `descriptor_audit.py`
+`fill_descriptors.py` (RDKit descriptor recompute), `cross_10118_full.py` (independent cross-check),
`apply_*.py` (write corrected copies + ledger).
