# Dataset-Correction Regen — Execution Log (2026-06-01)

Executes `docs/DATASET_CORRECTION_AND_REGEN_GUIDE.md` end-to-end: regenerate the stale
structure-derived caches for the **corrected** dataset (least→most compute, **MD excluded
per instruction**) and retrain every model (XGBoost / ADMET / AGILE / LiON / pKa) on the
corrected data. Companion to the audit report + correction guide + `DATASET_REFERENCE_SPEC.md`.

## Headline result
- bioact v14 LOO pooled **MAE 0.4111 (old, n=247) → 0.4098 (new, n=236)**, R²=0.626 — retrained on
  corrected SMILES with the 11 audit-flagged bioact rows correctly excluded.
- pKa v92 blend LOO **MAE 0.1183** (n=255; weights analog 0.00 / xgb 0.62 / **MolGpKa 0.38**).
- End-to-end validation (`audit_work/regen_validate.py`): **ALL CHECKS PASSED**.

## Root-cause fix (was silently defeating the whole correction)
The audit corrected the `SMILES` column but left **`SMILES_canonical` stale for 36 bioact rows**
(old wrong structure). That column is the join key for `precompute_qm`, `agile_embeddings`,
`physics_cache_io`, and `bioact_v14_pipeline` — so the corrections would never have reached any
structure-keyed cache/model. Fixed: `SMILES_canonical = canonical(SMILES)` for all rows
(`audit_work/regen_step2_fix_canonical.py`), verified 0 residual mismatches.

## What ran (least → most compute; MD skipped)
| Engine | How | Result |
|---|---|---|
| ADMET | local `admet_env` via `extend_caches.predict_admet_for_smiles` | 236/236 covered, 0 degenerate |
| AGILE | local `agile_embeddings.py` (frozen 60k encoder) | bioact (273,512)+pKa (278,512), 0 NaN; train array (236,512) |
| LiON | local `lion_env` (chemprop 1.6.1, `all_random_split_for_paper` CV) | 236/236 covered, 16 OOD |
| MolGpKa | local live GCN (`compute_molgpka_live.py`) | 255 (flagged-excluded), aligned to pKa bundle |
| **QM (xTB)** | local `precompute_qm.py --resume` (xtb_env) | **launched overnight** (~21 missing; twin-twins ~200 atoms) |
| MD | — | **NOT run (excluded by instruction)**; Block-D' MD cols stay NaN→emulator (pre-existing 65% NaN) |

> The guide's "LiON/ADMET are cloud-only" was outdated — `admet_env/`, `lion_env/` (+ real LiON
> checkpoints) and `extend_caches.py` already run them **locally**.

## Models retrained on corrected data
`bioact_v14_bundle` (236×106), `bioact_loo_components.npz`, `agile/cpp/qmmd _v14_train` arrays
(all realigned to the new 236-row bundle — fixes a latent 335-vs-247 misalignment),
`bioact_stacker_bundle` (**adaptive 6-head: direct/analog/LiON/ADMET/AGILE/CPP/qmmd**, MAE 0.4218 vs
static 0.4485), `bioact_ensemble`, `bioact_per_organ`, `bioact_binary`, `qmmd_head.joblib`, `pka_v92_bundle`.

## Flagged-row exclusion (audit §4d)
Rows with `audit_status` containing `UNRESOLVED|FLAG` (11 bioact w/ flux, 23 pKa) are excluded from
all structure-based feature/training paths: added to `bioact_v14_pipeline.load_v13`, `precompute_qm`,
`iajd_pka_v52.build_bundle`, `compute_molgpka_live`, `train_per_organ`. (`run_iajd_panel`,
`nn/train_transfer` already had it.)

## Workflow bugs fixed (were broken on this machine, RDKit 2022.09.5)
- `predict_v14_real.py`: dead `/mnt/user-data/...` paths → repo-relative; `AllChem.GetMorganGenerator`
  (absent in this RDKit) → `rdFingerprintGenerator` fallback; `BLOCK_SLICES['B'][0]:[1]` (slices, not
  tuples) → direct slice index. Smoke test now passes.
- `adaptive_stacker.py`: used unbound `fpgen` in the `except` branch → use `_fp`.
- `iajd_bioact_v14.py`: `/mnt` bundle fallback → repo path.
- `agile_embeddings.py`, `compute_molgpka_live.py`, `iajd_pka_v52.build_bundle`: pKa sheet `'Dataset'`
  → first sheet (post-audit files use `Sheet1`).
- `extend_caches.py`: LiON/ADMET subprocess timeouts made env-configurable (LiON default 1800s).

## Pod / git state
Corrected datasets are committed + pushed (`35e861f` + autosave) to `origin/physics-overnight`;
canonical xlsx LFS oid == AUDIT_FIXED oid and `diff HEAD` is empty → the committed+pushed datasets
include the SMILES_canonical fix. `nn/iajd_transfer_input.csv` refreshed + committed. So a pod
`git clone + git lfs pull` gets corrected data.

## Overnight + follow-ups
- `precompute_qm.py --resume` runs overnight → `finalize_after_qm.sh` (launched) waits for it, then
  rebuilds the **full bioact stack with REAL QM** in Block D' (marker: `physics_logs/finalize_after_qm.DONE`).
- `auto_retrain_watcher.py` was **stopped** for a clean rebuild and left stopped (finalize replaces it
  for this regen). Restart with `nohup .venv/bin/python auto_retrain_watcher.py &` after finalize if you
  want ongoing cache-change monitoring.
- After finalize completes overnight, re-commit the updated bundles (real-QM Block D').
- Backups: `audit_work/pre_regen_backup_20260601/` (caches/bundles), `*.PRE_AUDIT.xlsx` (datasets).
