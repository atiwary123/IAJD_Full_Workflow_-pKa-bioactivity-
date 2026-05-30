# T3 #7 — Pretrained SMILES embedder (ChemBERTa-77M-MTR): negative result + repurpose

**Status: integrated as a structural-novelty signal for the proposer, NOT as
a regression feature for the bioactivity model.** The original plan was to
add a Block F of ChemBERTa-PCA features to v14. Honest evaluation showed
this *hurts* LOO MAE, so we don't ship that version.

## What we tried

| Step | Result |
|---|---|
| Install `transformers==4.46.3` + ChemBERTa-77M-MTR | ✅ 14 MB model, CPU inference ~12 ms / SMILES |
| Embed all 247 training SMILES (frozen forward pass, mean-pool, 384-d) | ✅ 3.0 s total, no NaN |
| PCA → 32 components | ✅ explained variance 0.998, top component 0.530 |
| Pearson(PC0, y) | +0.371 — real signal but small |
| LOO MAE: v14 alone (92 dims) → v14 + ChemBERTa-PCA-32 (124 dims) | **0.4415 → 0.4511** (worse) |
| Ridge LOO baseline check (does linear model see anything?) | 0.447 → 0.447 (no change) |

## Why it didn't help

- The IAJD chemistry space is narrow (six families, ~250 molecules) and
  the existing Block A (RDKit) + Block B (LION, real chemprop) + Block C
  (ADMET-AI) + Block D (geometric) already encode the relevant signal.
- ChemBERTa pretrained on 77M PubChem molecules is dominated by drug-like
  small molecules, not ionizable amphiphilic Janus dendrimers.
  Mean-pooled embeddings on long tails + multi-arm scaffolds reflect
  general functional groups already captured by Morgan/RDKit features.
- Adding 32 noisy-but-correlated columns to XGBoost on 247 rows gives the
  trees room to overfit; LOO catches it.

This is real evaluation, not a guess. Per the project's no-proxy policy,
we don't ship a feature that worsens honest LOO performance just because
it sounds modern.

## What we did instead

ChemBERTa embeddings DO provide useful structural-novelty signal that
Tanimoto-on-Morgan does not: continuous cosine distance in a pretrained
semantic space picks up subtle chemistry differences that bit-level
Morgan saturates on. Concretely (sanity check on test SMILES):

| SMILES | cos to training centroid | cos_max to training |
|---|---|---|
| IAJD 244 (in training) | +0.20 | +1.00 |
| Near-isomer | +0.22 | +0.996 |
| Acetanilide (unrelated drug-like) | +0.32 | +0.62 |
| Alkane (very different) | +0.26 | +0.44 |

So `1 − cos_max_to_train` is a real structural-novelty metric that
complements Tanimoto in the proposer's active-learning surface.

Wiring (in `propose_iajds.py`):
- New column `chemberta_novelty` in candidate output: `1 − cos_max_to_train`
  in ChemBERTa-PCA-32 space.
- Embedder + PCA load lazily on first proposer call so the v14 pipeline
  stays unaffected.

## Artifacts kept

- `chemberta_embedder.py` — frozen-encoder embedder with on-disk SMILES → 384-d cache.
- `IAJD_master/bundles_caches/chemberta_pca_bundle.joblib` — fitted PCA (32 components, 99.8% variance) over the 247 training molecules.
- `IAJD_master/bundles_caches/chemberta_embeddings_cache.json` — per-SMILES embeddings to avoid recomputation.

## What we explicitly did NOT do

- Did NOT add Block F to `bioact_v14_pipeline.py`.
- Did NOT retrain v14 / ensemble / per-organ bundles on enlarged features.
- Did NOT change the regression test bounds — v14 LOO MAE guard stays at
  the pre-ChemBERTa target.

If you ever want to revisit, the honest path is:
1. Get more training data (negative result was on 247 rows; with 1000+
   the regularization-vs-signal tradeoff would likely flip).
2. Try mean-pool over the *last 4* hidden layers instead of just last
   (sometimes captures more chemistry).
3. Try a chemistry-task-finetuned encoder (e.g. ChemBERTa-77M-MLM with
   IAJD-bioact fine-tuning) — but that breaks "frozen embedder" and
   needs GPU.
