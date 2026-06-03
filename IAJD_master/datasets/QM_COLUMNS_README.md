# QM descriptor columns — data dictionary (added 2026-06-03)

Real **GFN2-xTB** electronic-structure descriptors were computed for every corrected IAJD and merged
into the dataset files as `qm_*` columns. This documents what they are, how they were made, and which
rows have them. Companion analysis: `docs/QM_REGEN_FINDINGS_2026-06-02.md`.

## Files that carry these columns
`IAJD_Bioact_v13_clean.xlsx`, `IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx`,
`IAJD_pKa_v21_final.xlsx`, `IAJD_pKa_v21_final.AUDIT_FIXED.xlsx` (sheet `Sheet1`).
(These also already carry the RDKit 2D/3D features and the `sp_` sample-prep features; QM is additive.)

## The 9 columns
| Column | Meaning | Units | median [p5, p95]¹ |
|---|---|---|---|
| `qm_q_ionizableN` | GFN2 partial charge on the most-basic (protonatable) N — the Henderson-Hasselbalch input | e | −0.19 [−0.20, −0.17] |
| `qm_dipole_D` | Total molecular dipole magnitude | Debye | 4.38 [1.89, 11.99] |
| `qm_polarizability` | Isotropic molecular polarizability α(0) | atomic units | 539 [429, 1286] |
| `qm_homo_lumo_eV` | HOMO–LUMO gap | eV | 2.75 [1.46, 3.25] |
| `qm_dGsolv_kJmol` | ALPB(water) solvation free energy of the whole molecule | kJ/mol | −53.9 [−151, −36] |
| `qm_dGsolv_head` | ΔGsolv of the head fragment (xtb re-run on head only) | kJ/mol | −44.5 [−67, −22] |
| `qm_dGsolv_tail` | ΔGsolv of the largest tail fragment | kJ/mol | −25.2 [−158, −19] |
| `qm_Ehedup` | E(protonated) − E(neutral), electronic ΔE | kJ/mol | −492 [−529, −456] |
| `qm_source` | provenance flag: `xtb` = real QM present; **empty** = not computed | — | — |

¹ ranges from the bioactivity file (273 rows).

## Method / provenance
- Engine: **GFN2-xTB** (local env `~/micromamba/envs/xtb_env`). Per-molecule pipeline (`qm_descriptors.py`):
  GFN-FF pre-opt → GFN2 geometry opt with **ALPB(water)** → GFN2 single-point with `--dipole --polar`
  on the lowest-G Boltzmann conformer set.
- Driver: `precompute_qm.py --resume --checkpoint-every 1` → cache `IAJD_master/bundles_caches/physics/qm_cache.csv`
  (keyed by **canonical SMILES**). Merged into the datasets by `audit_work/append_qm_to_datasets.py`
  (joins on `canonical(SMILES_canonical)` for bioact, `canonical(SMILES)` for pKa).
- These are the SAME values feeding the model "Block D'" physics features (via `physics_cache_io.py`).

## Coverage (2026-06-03)
- **Bioactivity: 273/273 rows have real QM (`qm_source=xtb`).**
- **pKa: 262/278 rows.** The 16 blanks are rows whose SMILES are still **flagged UNRESOLVED** in
  `audit_status` — QM was deliberately **not** computed from a known-wrong structure, and the cells are
  **left empty (never proxied/imputed)**. They will get real QM once those SMILES are resolved.
- No-proxy invariant: an empty `qm_*` cell means "not computed," never a default or family-median.

## Refresh / verify
- Re-merge from the cache (idempotent): `python audit_work/append_qm_to_datasets.py`
- Verify a file: `python -c "import pandas as pd; d=pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx'); print((d['qm_source']=='xtb').sum(),'real QM of',len(d))"`
- Recompute QM for any new/unresolved SMILES: `python precompute_qm.py --resume --checkpoint-every 1` (xtb on PATH), then re-merge.

## Notes & caveats
- Computed on the **audit-corrected** structures. The corrections shifted QM shape/solvation substantially
  (mean |Δ| vs the old wrong SMILES: dipole ~3 D, ΔGsolv ~14 kJ/mol, polarizability ~19) but barely moved
  `qm_q_ionizableN` (~0.04) — see the findings report.
- 3 compounds initially failed to a scratch-cleanup race (`qm_scratch_janitor` deleting live scratch of
  slow >30-min compounds; fixed in commit `ed23a8e`); all were retried successfully — **0 net QM lost**.
- `dG_escape_helfrich` (a derived Block-D' feature) is computed downstream in the model pipeline, not stored here.
