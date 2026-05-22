# IAJD Master Folder

Curated set for the IAJD pKa + bioactivity tandem workflow. Built May 21, 2026
from `Files-2.zip` + `/mnt/project/` contents.

Total: ~45 MB across 6 directories (down from ~150 MB / 199 files in the raw zip).

---

## Directory layout

| Folder | What's in it | When you need it |
|---|---|---|
| `code/` | All Python modules required to run the tandem | Always |
| `bundles_caches/` | Trained model bundles + MolGpKa/LION/ADMET caches | Always |
| `datasets/` | Canonical pKa (v21) and bioact (v13) training data, plus EXP264 worked example | Always |
| `docs/` | READMEs, model cards, session prompts, honest metrics, specs | Reference / handoff |
| `optional/` | v13 fallback bundle, chemprop install notes, EXP264 deliverables | Situational |
| `source_papers/` | SI PDFs for SMILES reconstruction provenance | When adding/validating IAJDs |

---

## Quickstart

```python
import sys
sys.path.insert(0, '<path-to>/IAJD_master/code')
from iajd_tandem_final import load_tandem_bundle, predict_iajd

b = load_tandem_bundle()
r = predict_iajd('<SMILES>', b, family_hint='PE-Tris')
print(r['combined_summary']['pka'])
print(r['combined_summary']['bioact'])
```

`family_hint` is REQUIRED for everything except PE-Tris (Bug-2 guard).
Allowed values: `sSS-Nonsym`, `PE-Tris`, `GA-Tris`, `PE-Gallic`,
`Dialkoxybenzyl`, `G1-Janus-Dendrimer`, `HTM-Dendrimer`, `TT-Dendrimer`.

---

## Honest performance (cite these numbers)

| Stage | Honest MAE | n | Source |
|---|---|---|---|
| v9.1 pKa (5 trained families) | **0.067** pKa units, R²=0.738 | 246 | `docs/pka_v91_loo_report.json` |
| v9.1 pKa (3 bioact-only families) | unvalidated | 0 | pooled debias, PI=0.60 |
| v14.0 bioact (LION+ADMET on for all) | **~0.41** log10-flux | 335 | `docs/honest_summary_v140_REAL.json` |

---

## ⚠️ Gaps to fill before the tandem will import

Two files are referenced by `code/iajd_tandem_final.py` and
`code/predict_v14_real.py` but were not present in the original zip or in
`/mnt/project/`. They lived in the ephemeral `/mnt/user-data/outputs/bioact_v14/`
directory from a prior session.

- `bioact_v14_pipeline.py` — exports `assemble_X`, `BLOCK_SLICES`
- `iajd_bioact_v14.py` — exports `_detect_family`, `load_bundle`, `predict_bioactivity`

**Action required:** recover these from the chat session that produced
`bioact_v14_bundle.pkl`, or rebuild them from `MODEL_CARD_v14.md` + the bundle's
`feature_names` / `block_slices`. Drop both into `code/` once recovered.

---

## File-by-file reference

### `code/`

| File | Role |
|---|---|
| `iajd_tandem_final.py` | **Production API.** `load_tandem_bundle()` + `predict_iajd()` |
| `iajd_pka_v91.py` | v9.1 pKa model — imports v52 and v71 |
| `iajd_pka_v52.py` | Required by v91: `compute_features`, `compute_3d_features_from_mol`, `tokens_from_mol`, `build_bundle` |
| `iajd_pka_v71.py` | Required by v91: `_try_live_molgpka` (MolGpKa live-call wrapper; falls back gracefully) |
| `iajd_pka_v72.py` | Included for completeness — verify whether v91 should be switched to this newer variant |
| `predict_v14_real.py` | Bioactivity-only inference with real LION/ADMET auto-fetch (requires the two missing files above) |
| `extend_caches.py` | Adds new SMILES to `lion_cache_v13.json` / `admet_cache_v13.json` |

### `bundles_caches/`

| File | Role |
|---|---|
| `bioact_v14_bundle.pkl` | **Production v14.0 bundle.** 335 rows, 7 families, honest MAE 0.4032 |
| `bioact_v14_3_bundle.pkl` | Research variant (v14.3); honest MAE 0.4251 — research only |
| `molgpka_preds.npy` | Cached MolGpKa predictions for 246 IAJDs (used because live MolGpKa is broken in sandbox) |
| `molgpka_debias_models.joblib` | Per-family LinearRegression debias models for feature 30 |
| `molgpka_rich_features.npz` | MolGpKa rich-feature cache |
| `molgpka_debiased_loo.npy` | LOO-debiased MolGpKa values |
| `lion_cache_v13.json` | Real LION (chemprop 1.6.1 5-CV ensemble) predictions for 335 training SMILES |
| `admet_cache_v13.json` | Real ADMET-AI 2.0.1 predictions for 335 training SMILES |
| `lion_train_fps.pkl` | Morgan fingerprints of LION training set (for novelty checks) |

### `datasets/`

| File | Role |
|---|---|
| `IAJD_pKa_v21_final.xlsx` | Canonical pKa dataset — 246 IAJDs, 5 families. Sheet: "Dataset" |
| `IAJD_Bioact_v13_clean.xlsx` | Canonical bioact dataset — 335 rows / 274 unique IAJDs, 7 families |
| `IAJD_Bioact_v13_clean.pkl` | Pickled version of same |
| `EXP264_Predictions_v1_2.xlsx` | Worked example: predictions for the 5 EXP264 GA-Tris IAJDs |

### `docs/`

Read in this order for full context:

1. `tandem_README.md` — production walkthrough
2. `v14_README.md` — bioactivity release notes
3. `tandem_MODEL_CARD.md` — tandem architecture details
4. `v14_MODEL_CARD.md` — bioactivity model card (REAL LION+ADMET edition)
5. `v14_HONEST_SUMMARY.md` — why 0.4032 is the right number to cite
6. `SESSION_PROMPT_v15.md` — ready-to-paste prompt to start a v15 session
7. `v13_dataset_HANDOFF.md` — dataset-construction provenance
8. `V11_SESSION_SUMMARY.md`, `V10_SESSION_SUMMARY.md` — earlier dataset expansion notes
9. `IAJD_v91_certification_report.docx` — pKa v9.1 validation writeup
10. `IAJD_pKa_Prediction_System_Spec_v4_0.docx` — pKa system spec
11. `IAJD_Bioactivity_Prediction_System_Spec_v2_0.md` — bioact system spec
12. `IAJD_Dataset_Construction_Guide_v3.docx` — process for adding new IAJDs

Metrics JSONs: `pka_v91_loo_report.json`, `tandem_smoke_results.json`,
`honest_summary_v140_REAL.json`, `honest_summary_v143_REAL.json`.

### `optional/`

| File | When to use |
|---|---|
| `iajd_bioact_v13_production_bundle.pkl` | v13 pre-LION fallback if v14 inference is broken |
| `CHEMPROP_INSTALL.md` | Setup notes for the `lion_env` venv (needed for novel-SMILES LION inference) |
| `EXP264/` | All EXP264 deliverables: cdxml source, structures PNGs, prediction logs, methodology + predictions pptx |

### `source_papers/`

SI PDFs from the Percec lab papers used to construct the v13 dataset. Keep
these together so any IAJD's MALDI cross-validation can be re-verified
without hunting through the project file list.

| File | Paper |
|---|---|
| `ja1c05813_si_001.pdf` | Percec et al. JACS 2021 |
| `ja1c09585_si_001.pdf` | Percec et al. JACS 2022 |
| `ja2c00273_si_001.pdf` | Percec et al. JACS 2022 |
| `ja3c07337_si_003.pdf` | Percec et al. JACS 2023 |
| `ja3c13569_si_001.pdf` | Percec et al. JACS 2024 |
| `ja5c07232_si_001.pdf` | Percec et al. JACS 2025 |
| `bm4c01107_si_001.pdf` | Percec et al. Biomacromolecules 2024 |
| `bm4c01599_si_001.pdf` | Percec et al. Biomacromolecules 2024 |
| `pharmaceutics-2390918-supplementary.pdf` | Pharmaceutics |
| `sciadv_adv1554_sm.pdf` | Science Advances |
| `10118_2026_3563_MOESM1_ESM.pdf` | (verify journal) |

---

## What was dropped from the original zip and why

- All `(1)`-suffixed duplicates at root — byte-identical to the originals
- `files-15/` — older copy of `files-16/`, every file is same or older
- `pKa prediction workflow/pKa prediction workflow #1/iajd_pka_v9.py` and related v9.0 artifacts — superseded by v9.1
- `Bioactivity Update/` (v09-LION) — entirely superseded by v13/v14
- `Bioactivity Update #2/` — kept only `HANDOFF_v13.md` and the v13 bundle
- `MOST UPDATED Dataset/` intermediate build scripts (`build_v11_v2.py`, `build_v12.py`, etc.) — work captured in the final v13 xlsx
- `MOST UPDATED Dataset/v13_full_workdir.tar.gz` (36 MB) and `v10_session.tar.gz` (34 MB) — scratch sessions
- `BIOACTIVITY WORKFLOW/3_external_models/LNP_ML_essentials/checkpoints/` (33 MB) — chemprop checkpoints are re-downloadable from the LNP_ML repo
- `BIOACTIVITY WORKFLOW/2_full_pipeline/` — earlier v09 pipeline, superseded
- `__pycache__/` directories
- `BIOACTIVITY WORKFLOW/IMG_4770.HEIC` — orphan photo
- Various intermediate `.pkl` files (`sss_smiles_built.pkl`, `new_iajd_3d.pkl`, `deferred_iajds.pkl`, `pharm1572_combined.xlsx`, `g1janus_3d.pkl`, etc.)
