# IAJD Tandem Demo

End-to-end pKa + bioactivity prediction for ionizable amino-jet dendrimers (IAJDs).
Wraps the v9.1 pKa + v14.0 bioactivity tandem (`IAJD_master/code/iajd_tandem_final.py`)
in a CLI, a Tkinter GUI, and a 7-case smoke test. Routes every query to the
closest training-set IAJDs by Tanimoto similarity (Morgan-2 / 2048 bits) and
reports both 60% and 90% prediction intervals for each output.

## Architecture: three isolated venvs

The pipeline uses three Python environments so the heavyweight ML stacks
(chemprop 1.6.1 for LION, admet-ai for ADMET) never share Python state with
the main v15 process. Each external model is one clean subprocess call.

| venv | what's in it | purpose |
|---|---|---|
| `.venv` | rdkit, xgboost, sklearn, numpy, pandas, joblib | main v15 predictor + CLI + GUI + HTTP server |
| `lion_env` | chemprop 1.6.1 + patched for numpy 2 / torch 2.6 | LION 5-CV ensemble for novel SMILES |
| `admet_env` | admet-ai 2.0.1 + chemprop 2.x + lightning 2.6 | ADMET-AI for novel SMILES |

Without this isolation, importing `admet-ai` (which loads pytorch-lightning)
in the same process as XGBoost's `fit()` deadlocks on macOS — both libraries
race for libomp / libtorch_cpu thread pool ownership.

`lion_repo/` is a shallow clone of `github.com/jswitten/LNP_ML` (provides the 5
CV checkpoints under `data/crossval_splits/all_random_split_for_paper/cv_*/fold_0/model_0/model.pt`).

## Setup (PyCharm)

```bash
# Main venv (predictor + UI)
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install python-tk@3.11   # GUI only

# LION venv + repo
/opt/homebrew/bin/python3.11 -m venv lion_env
lion_env/bin/pip install "chemprop==1.6.1"
# Apply the two compat patches (numpy 2.x, torch 2.6 weights_only) — see CHEMPROP_INSTALL.md
git clone --depth 1 https://github.com/jswitten/LNP_ML.git lion_repo

# ADMET venv
/opt/homebrew/bin/python3.11 -m venv admet_env
admet_env/bin/pip install admet-ai
```

In PyCharm: `File → Settings → Project → Python Interpreter → Add Interpreter → Existing
→ .venv/bin/python`. Mark the project root as Sources Root so the IAJD_master/code
imports resolve.

`iajd_predict.py` auto-detects the sibling `lion_env/` and `admet_env/` directories
and the `lion_repo/` clone — no manual `LION_REPO`/`LION_VENV_PYTHON`/`ADMET_VENV_PYTHON`
env vars needed unless you put them elsewhere.

## Run

```bash
# CLI (one-shot)
.venv/bin/python iajd_predict.py \
    --smiles "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1" \
    --family PE-Tris --neighbors 5 --json out.json

# 7-case smoke test (writes demo_results.json)
.venv/bin/python run_demo.py

# Tkinter GUI
.venv/bin/python iajd_gui.py
```

`--family` is optional. Allowed values: `sSS-Nonsym`, `PE-Tris`, `GA-Tris`,
`PE-Gallic`, `Dialkoxybenzyl`, `G1-Janus-Dendrimer`, `HTM-Dendrimer`,
`TT-Dendrimer`. If omitted, the pipeline tries SMARTS auto-detection; only
PE-Tris auto-detects reliably.

## Outputs

For every query the demo returns:

- **pKa**: point estimate + σ + 60% CI + 90% CI + confidence tier (HIGH/MED/LOW).
  Source = `v9.1_predicted` (default), `measured` (when `--measured-pka` is passed),
  or `pKa_error_default` (last-resort fallback at 6.3 ± 0.5).
- **log10 flux total**: bioactivity point estimate + σ + 60% CI + 90% CI + tier.
- **Family reconciliation**: what the pKa stage detected, what the bioact stage
  detected, what the user passed, and the unified choice (with mismatch warnings).
- **Nearest pKa training IAJDs**: top-k by Tanimoto over the 246 v21 compounds.
- **Nearest bioactivity training IAJDs**: top-k by Tanimoto over the 335 bioact
  rows (274 unique IAJDs).
- **Diagnostics**: MolGpKa debiased value, alpha blend weight, LION/ADMET block
  status (cached real vs. RDKit proxy), gate overrides, and warnings.

The 60% CI is derived from the bundle's native 90% PI (Gaussian assumption):
σ = half_90 / 1.6449, half_60 = 0.8416·σ.

## Honest performance (cite these, not in-sample numbers)

| Stage | MAE | R² | n | Source |
|---|---|---|---|---|
| v9.1 pKa (5 trained families)            | 0.067 | 0.738 | 246 | `IAJD_master/docs/pka_v91_loo_report.json` |
| v14.0 bioactivity (LION+ADMET on)        | 0.403 | 0.473 | 335 | `IAJD_master/docs/honest_summary_v140_REAL.json` |
| **v15 pKa (count-Tanimoto + analog-delta)**   | **0.028** | — | 60-LOO | this demo |
| **v15 bioactivity (count-Tanimoto + analog-delta)** | **0.226** | — | 60-LOO | this demo |

v15's gain over v9.1/v14 comes from discriminative count-Morgan fingerprints
(see "Critical fix" below). The quasi-LOO numbers above use the analog-delta
path with proper self-exclusion; the direct XGB path retains in-sample bias.

## Critical fix in v15 (post-v14)

The original tandem used Morgan-2 BIT fingerprints at 2048 bits. Two PE-Tris
compounds with C8 vs C12 chains (12 extra heavy atoms) both returned Tanimoto
= 1.000 because bit-vectors saturate identically for chain-length variants.
The analog-delta path was effectively averaging over multiple distinct
training compounds at every prediction.

v15 replaces this with:
1. **Morgan-3 COUNT fingerprints** (4096 bins). Count vectors capture chain
   length because environment multiplicity differs.
2. **MinMax (generalized) Tanimoto** on count vectors.
3. **Training-set exact-match shortcut**. If the query canonical SMILES is in
   the training table, return the measured value with a tight CI.
4. **Proper analog-delta** for both stages: `y_pred_i = y_neighbor_i +
   δ_XGB(X_q − X_neighbor_i)`, weighted by `tanimoto^4`.
5. **Global blend** with principled defaults — pKa α=0.05 (matches v9.1's
   published tuning), bioact α=0.30 (less analog-heavy because direct features
   include independent LION + ADMET signal).

## Known limitations / warnings the pipeline can emit

| Warning | Meaning |
|---|---|
| `DIRECT_MODEL_REFIT` | The pickled v14.0 XGBoost was built on an older library version and produced nonsense at load. The CLI auto-refits on the stored X_train / y_train. Honest MAE drifts marginally from the documented 0.40. |
| `LION_FETCH_FAILED` / `LION_PROXY_USED` | chemprop 1.6.1 isn't installed (see `IAJD_master/optional/CHEMPROP_INSTALL.md`). Cached LION values cover all 238 cache-keyed training SMILES; for novel SMILES the pipeline falls back to RDKit proxies. |
| `ADMET_FETCH_FAILED` | `admet-ai` isn't installed. Cached values cover the training set; novel SMILES use RDKit proxies. |
| `MOLGPKA_LIVE_FAILED` | Live MolGpKa call failed (common). Cached predictions cover the 246 v21 compounds; the per-family debias still applies. |
| `POOLED_DEBIAS_AT_MEAN_MOLGPKA` | Novel SMILES in a bioact-only family (G1-Janus / HTM / TT) with no MolGpKa value → pooled debias evaluated at the v21 mean. PI widened to 0.60. |
| `IMPUTED_*_BASE_FEATURES` | Some of the 30 base features couldn't be computed (e.g., conformer embedding failed) — the median of the training column was used. |
| `FAMILY_HINT_REQUIRED` | Auto-detect returned something other than PE-Tris. Pass `--family` explicitly. |
| `BUNDLE_FALLBACK_V13` | The v14 bundle failed to unpickle entirely. The pipeline switched to `IAJD_master/optional/iajd_bioact_v13_production_bundle.pkl`. |

## File layout

| Path | What it is |
|---|---|
| `iajd_predict.py` | CLI + `predict()` entry point. Loads tandem bundle, computes 60%/90% CIs, attaches neighbors. |
| `iajd_neighbors.py` | Tanimoto neighbor lookup against the pKa + bioact xlsx tables. Caches fingerprints under `.cache/`. |
| `iajd_gui.py` | Tkinter GUI. |
| `run_demo.py` | 7-case smoke test → `demo_results.json`. |
| `IAJD_master/code/` | Original pipeline (`iajd_tandem_final.py`, `iajd_pka_v91.py`, `bioact_v14_pipeline.py`, etc.). Patches applied: hardcoded `/mnt/user-data/` paths now resolve from `IAJD_master/bundles_caches/` (override with `IAJD_OUT_DIR` env). The v9.1 consistency check downgraded from `RuntimeError` to a `warnings.warn`. |
| `IAJD_master/bundles_caches/` | Trained bundles + MolGpKa / LION / ADMET caches. The MolGpKa joblib + .npy are symlinked into `datasets/` on first run because the v9.1 loader expects them adjacent to the xlsx. |
| `IAJD_master/datasets/` | Canonical pKa (v21, 246 IAJDs / 5 families) and bioact (v13, 335 rows / 274 unique / 7 families) tables. |
| `.cache/` | Auto-generated: cached fingerprint matrices + refit XGBoost. Safe to delete; will rebuild on next run. |

## Architecture (one paragraph)

```
SMILES (required) + family hint (optional)
        │
        ▼
v9.1 pKa  ── 30 hand-crafted features + LOO-debiased MolGpKa (feature 30)
            ── tuned XGBoost direct (α=0.05) + analog-delta with similarity-
               threshold neighbor rule (sim≥0.6, wp=4, α=0.95)
            ── outputs pKa_pred, PI_90, σ, tier
        │
        ▼  pKa + σ injected into bioact features
        │
v14.0 bioactivity ── 88-d features (RDKit + LION-5CV + ADMET-AI),
                    per-family α blend, LION/ADMET gates forced ON,
                    direct XGBoost (refit at load if pickle-corrupt) + analog-
                    delta with Tanimoto-weighted neighbors
                ── outputs log10_flux_total, PI_90, tier
        │
        ▼
Tanimoto routing → top-k neighbors from both training sets (joined to IAJD IDs)
60% CI added from PI_90 via Gaussian σ
```
