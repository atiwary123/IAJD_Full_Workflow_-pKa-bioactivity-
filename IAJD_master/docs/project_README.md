# IAJD Bioactivity v2.0-MIN-v09-LION — Project Files

This is the minimum-viable set of files to (a) load the trained bioactivity
predictor in a future session, (b) make new predictions, and (c) retrain from
scratch if the dataset is updated.

**Total size:** 8.9 MB (21 files)

**Definitive write-up:** `MODEL_CARD_v2_0_v09_LION.md` — read this first.

---

## What's here

### Inference (use these to make predictions on a new SMILES)
- `iajd_bioact_v2.py` — main API (`load_bundle`, `predict_bioactivity`)
- `feature_assembler_v2.py` — 88-d feature vector assembly
- `block_b_lion.py` — LiON pretrained inference wrapper (5-checkpoint ensemble)
- `block_c_admet.py` — ADMET-AI wrapper (currently RDKit fallback)
- `block_d.py` — geometric / corona heuristics (Peterca radius, packing parameter, etc.)
- `pka_surrogate.py` — pKa GPR predictor (consumes v1.0 baked-in surrogate)
- `features.py` — Block A feature computation (RDKit + 3D descriptors + pKa)
- `iajd_bioact_v2_v09_LION_bundle.pkl` — **the trained model** (7.5 MB)
- `lion_cache_v09.json` — cached LiON predictions for the 153 v09 SMILES
- `lion_train_fps.pkl` — LiON's 9,377 training-set Morgan FPs (for OOD distance)

### Training (use these to retrain from a new dataset)
- `train_v2.py` — generic training pipeline (Stage A + B, honest LOO)
- `train_v2_v09.py` — v09-specific wrapper
- `build_v2_dataset_v09.py` — assemble feature matrix from v09 source
- `v09_holdout_eval.py` — X1/X2/X3 evaluation harness

### Data
- `IAJD_Bioact_v09_clean.pkl` — the v09 source dataset (210 rows × 117 cols)
- `bioact_v2_v09_X.npy` — assembled feature matrix (210 × 88)
- `bioact_v2_v09_meta.pkl` — metadata aligned to X (holdout column, IAJD_id, family, etc.)
- `bioact_v2_v09_fps.pkl` — Morgan FPs aligned to X

### Documentation
- `MODEL_CARD_v2_0_v09_LION.md` — full report (numbers, caveats, roadmap)
- `v2_v09_metrics.json` — training-time metrics
- `v2_v09_holdout_metrics.json` — X1/X2/X3 held-out metrics

---

## What's NOT here, and why

- **The 5 LiON model.pt checkpoints (~32 MB).** These are reproducible from
  `github.com/jswitten/LNP_ML` (path: `data/crossval_splits/all_random_split_for_paper/cv_*/fold_0/model_0/model.pt`).
  Including them would push project files to ~40 MB. The README in the full
  release at `iajd_bioact_v2_0_v09_LION/` includes them.
- **Pre-LiON / older bundles, intermediate caches, ablation logs.** The full
  release at `iajd_bioact_v2_0_v09_LION/` has the lineage; this is the
  current production state only.
- **chemprop runtime patches.** Documented in the header of `block_b_lion.py`
  (numpy.VisibleDeprecationWarning + torch.load weights_only).

---

## Quickstart (when you load this in a new session)

```python
import sys, os
sys.path.insert(0, '/path/to/project_files')

# Tell block_b_lion where to find the LiON checkpoints
# (skip this if you don't need to recompute Block B — the cache covers all v09 SMILES)
os.environ['LION_REPO'] = '/path/to/LNP_ML'

from iajd_bioact_v2 import load_bundle, predict_bioactivity

bundle = load_bundle('iajd_bioact_v2_v09_LION_bundle.pkl')

result = predict_bioactivity(
    'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC',
    bundle,
)
print(result['log10_flux_total'], result['log10_flux_total_PI90'])
print(result['organ_dominant'], result['confidence_tier'])
```

If LiON checkpoints aren't available, the `block_b_lion.py` falls back to
zeros + `lion_active=False` — the bundle's metadata flag will reflect this
and downstream code handles it transparently.

---

## Headline numbers (carry these in your head)

- log10_flux_total LOO MAE: **0.430** (n=147 strict honest LOO)
- log10_flux_total X3 MAE:  **0.376** (n=16 held-out)
- log10_flux_liver X3 MAE:  **0.207** (best per-organ)
- 90% PI empirical coverage: **0.875** (in [0.85, 0.92] target ✓)
- Stage A spleen accuracy: **0.94** · liver: 0.81 · lung: 0.88 · LN: 0.94
- All 7 v2.0-MIN production gates pass

---

## Dependencies

- numpy, pandas, scikit-learn, xgboost, scipy, rdkit, joblib (core)
- chemprop==1.6.1 (only if recomputing Block B from scratch — cache covers existing compounds)
- mapie (only if regenerating PI half-widths from raw LOO residuals)

Tested with: numpy 2.4.4, pandas 3.0.2, sklearn 1.8.0, xgboost 3.2.0, rdkit 2026.03.1.
