# IAJD Bioactivity v14 — Model Card (REAL LION + ADMET edition)

**Version:** v14.0 (production); v14.1 / v14.2 / v14.3 included as research variants
**Build date:** May 20, 2026
**Training data:** `IAJD_Bioact_v13_clean.xlsx` — 335 rows, **274 unique IAJDs**, 236 unique canonical SMILES across 7 families
**Features:** REAL LION (5-CV chemprop ensemble, 14-d) + REAL ADMET-AI (10-d) + RDKit-2D (50-d) + geometric/corona (6-d) + formulation (8-d) = 88-d feature vector per IAJD
**Best production bundle:** `bioact_v14_bundle.pkl` (v14.0, the simplest variant that generalizes best)
**Inference API:** `predict_v14_real.py` (recommended — auto-fetches real LION+ADMET for new SMILES); `iajd_bioact_v14.py` (basic); `iajd_bioact_v143.py` (v14.3 research variant)
**Supersedes:** v14 proxy edition (RDKit-only LION/ADMET fallbacks)

---

## Headline numbers (REAL features)

| Variant | In-sample LOO MAE | Honest 5-fold CV MAE | Optimism | R² |
|---|---|---|---|---|
| Baseline (α=0.6, all blocks on) | 0.4342 | — | — | 0.466 |
| **v14.0 (per-fam α + per-fam gates)** | **0.4006** | **0.4032** | **−0.003** | **0.473** |
| v14.1 (+ per-fam direct + stack) | 0.4030 | — | — | 0.454 |
| v14.2 (+ per-fam HPO) | 0.3998 | — | — | 0.481 |
| v14.3 (best-of v14.1+v14.2 + analog variants) | 0.3931 | 0.4251 | −0.032 | 0.495 |

**Recommendation: use v14.0.** Its in-sample LOO and honest 5-fold nested CV agree to within 0.003 MAE units. This is the variant whose performance number you can trust. v14.3's lower in-sample (0.3931) hides −0.032 of selection-bias optimism — the honest result is 0.4251.

**v14.0 vs baseline (real features): −0.034 MAE (−7.7%); R² +0.011.** Every large family improves; PE-Tris and TT-Dendrimer improve dramatically (−0.11).

---

## Per-family honest MAE (v14.0 with REAL features)

| Family | n | Honest MAE | Notes |
|---|---|---|---|
| sSS-Nonsym | 175 | **0.416** | Workhorse family; near-noise floor |
| PE-Tris | 51 | **0.405** | Real LION delivered −0.11 from baseline |
| GA-Tris | 50 | **0.410** | Real LION ON +0.019, ADMET OFF −0.014 |
| G1-Janus | 26 | 0.438 | Real ADMET ON but not LION |
| Dialkoxybenzyl | 18 | **0.185** | Best per-family; α=0.5 (delta-blend) |
| HTM-Dendrimer | 10 | 0.411 | Real LION+ADMET both ON; +0.05 from baseline |
| TT-Dendrimer | 5 | 0.452 | Small-n; treat with caution |

Pooled honest MAE = **0.4032** (R² = 0.473).

---

## What changed: proxy → real

The headline change is replacing RDKit-based proxies for Block B (LION) and Block C (ADMET) with real predictions:

**Block B (LION)** — Real chemprop ensemble from Witten et al. 2024 (LNP_ML)
- 5 cross-validation checkpoints (cv_0..cv_4), averaged ensemble
- 6 tissues per prediction: liver_IV, lung_IT, lung_inh, lung_neb, muscle_IM, nasal
- Plus max-z-flux, argmax-OH-encoding, OOD flag → 14 features total
- **220/236 IAJDs in-distribution** for LION (Tanimoto ≥ 0.30 to LION training set)
- 80.5% of IAJDs are predicted to deliver best to muscle (IM route)

**Block C (ADMET)** — Real ADMET-AI 2.0.1 by Swanson et al.
- 10 properties: PPBR_AZ, BBB_Martins, VDss_Lombardo, HIA_Hou, Caco2_Wang, Pgp_Broccatelli, Half_Life_Obach, Clearance_Hepatocyte_AZ, Solubility_AqSolDB, Lipophilicity_AstraZeneca
- All 236 IAJDs predicted in 13 seconds (vs proxy was instant but uninformative)

### Gate decisions shifted with real features

| Family | n | Block B (LION) proxy | Block B real | Block C (ADMET) proxy | Block C real |
|---|---|---|---|---|---|
| sSS-Nonsym | 175 | OFF | **ON** (+0.001) | on (+0.001) | ON (+0.004) |
| PE-Tris | 51 | OFF | **ON** (+0.031) | OFF (−0.010) | **ON** (+0.006) |
| GA-Tris | 50 | OFF | **ON** (+0.019) | OFF (−0.002) | OFF (−0.014) |
| G1-Janus-Dendrimer | 26 | OFF | OFF (−0.001) | OFF (−0.007) | **ON** (+0.002) |
| Dialkoxybenzyl | 18 | on (+0.008) | ON (+0.002) | on (+0.002) | ON (+0.000) |
| HTM-Dendrimer | 10 | on (+0.041) | **ON** (+0.037) | OFF (−0.000) | **ON** (+0.023) |
| TT-Dendrimer | 5 | on | OFF (−0.001) | on | **ON** (+0.020) |

With real LION/ADMET:
- **LION helps 5/7 families** (was 3/7 with proxies). Real LION especially helps PE-Tris, GA-Tris, and HTM-Dendrimer where the proxy was misleading.
- **ADMET helps 6/7 families** (was 3/7 with proxies). Only GA-Tris doesn't benefit from ADMET features.

---

## Per-family configuration (v14.0 production)

| Family | n | α | Block B | Block C |
|---|---|---|---|---|
| sSS-Nonsym | 175 | 0.8 | ON | ON |
| PE-Tris | 51 | 1.0 | ON | ON |
| GA-Tris | 50 | 1.0 | ON | OFF |
| G1-Janus-Dendrimer | 26 | 1.0 | OFF | ON |
| Dialkoxybenzyl | 18 | 0.5 | ON | ON |
| HTM-Dendrimer | 10 | 1.0 | ON | ON |
| TT-Dendrimer | 5 | 1.0 | OFF | ON |

α=1.0 means direct-only (XGBoost prediction with no analog-delta blending). The two exceptions are sSS-Nonsym (α=0.8: 80% direct + 20% analog) and Dialkoxybenzyl (α=0.5: balanced) — both have very dense neighbor structures in the training set, so the analog-delta path contributes signal.

---

## Why v14.0 over v14.3?

v14.3 adds per-family direct heads + stacked ensembles + finer HPO grids, achieving in-sample MAE 0.3931 vs v14.0's 0.4006. But the honest nested-CV test reveals:

- v14.0 in-sample = 0.4006, honest = 0.4032 → **optimism −0.003** (negligible)
- v14.3 in-sample = 0.3931, honest = 0.4251 → **optimism −0.032** (substantial)

Every layer of selection (best-of-12-HPO-configs, best-of-2-pool-sources, best-of-2-fam-sources, best-of-3-analog-variants, best-of-66-stack-weight-combos) introduces optimism. v14.0 uses only α-tuning per family (12 values × 7 families = 84 selection events) — the optimism is bounded by chance correlations on 335 datapoints.

**v14.3 may still be useful for ranking IAJDs** (where absolute MAE matters less than relative ordering). But for quoted-performance and per-IAJD predictions, **v14.0 is the canonical bundle**.

---

## Critical caveat: novel SMILES require LION/ADMET extension

The `lion_cache_v13.json` and `admet_cache_v13.json` files cover the 236 training SMILES. For new query SMILES, the inference must:
1. Compute real LION (5-CV chemprop ensemble × 6 tissues, ~10 s per SMILES single-threaded)
2. Compute real ADMET-AI (instantaneous, ~0.5 s per SMILES)
3. Add to the caches
4. Then call the v14.0 model

`predict_v14_real.py` does this automatically via `extend_caches.py`. Required: `chemprop 1.6.1` in a venv at `/home/claude/lion_env`, `admet-ai` in the main env, and the LNP_ML repo at `/home/claude/LNP_ML` with checkpoints in `data/crossval_splits/all_random_split_for_paper/cv_X/fold_0/model_0/model.pt`.

If those external dependencies aren't present, the pipeline falls back to RDKit proxies and warns the user via `block_B_real: False` in the prediction output. Predictions are still valid but use the proxy-trained model heads — which gives a small mismatch since the model was trained on real LION/ADMET features.

---

## Files in this release

### Documentation
- `MODEL_CARD_v14.md` — this file (REAL LION/ADMET edition)
- `MODEL_CARD_v14_proxy.md.bak` — previous proxy-edition model card
- `README.md` — quickstart + file index
- `SESSION_PROMPT_v15.md` — handoff for next session

### Bundles + APIs
- `bioact_v14_bundle.pkl` — **v14.0 production bundle** (recommended)
- `bioact_v14_3_bundle.pkl` — v14.3 research variant (lower in-sample, higher optimism)
- `bioact_v14_2_bundle.pkl`, `bioact_v14_1_bundle.pkl` — intermediate variants
- `predict_v14_real.py` — production inference (auto-extends caches)
- `iajd_bioact_v14.py`, `iajd_bioact_v143.py` — basic inference APIs
- `extend_caches.py` — query-time LION/ADMET cache extension

### Caches (essential for real-feature inference)
- `lion_cache_v13.json` — 236+ SMILES × 14-d real LION vectors
- `admet_cache_v13.json` — 236+ SMILES × 10-d real ADMET-AI vectors
- `lion_train_fps.pkl` — Morgan FPs for LION training set (used for OOD flag)
- `lion_intermediate.npz` — raw per-tissue LION outputs (resumability)

### Pipeline + data
- `bioact_v14_pipeline.py` — feature pipeline (auto-uses caches when present)
- `v14_checkpoint_runner.py` — phase-by-phase v14.0 runner
- `build_lion_cache.py`, `build_admet_cache.py` — scripts that built the caches
- `bioact_v14_X.npy`, `bioact_v14_y.npy`, `bioact_v14_meta.csv`, `bioact_v14_fps.pkl` — features on v13
- `bioact_v14_families.json`, `bioact_v14_smis.json`, `bioact_v14_lion_modes.json`

### Metrics + audit
- `honest_summary_v140_REAL.json` — v14.0 real-feature honest 5-fold CV metrics
- `honest_summary_v143_REAL.json` — v14.3 real-feature honest 5-fold CV metrics
- `baseline_metrics.json`, `bioact_v14_alpha_sweep.json`, `bioact_v14_loo.json` — v14.0 metrics
- `hpo_results.json`, `perfam_hpo.json`, `per_family_direct.json` — v14.1/14.2 HPO
- `stack_results.json`, `stack_v142.json`, `stack_v143.json` — stacking searches
- `gates_summary.json` — consolidated gate audit
- `loo_*.npz` (16 files) — LOO prediction caches for reproducibility
- `proxy_run_backup/` — previous proxy-edition artifacts for comparison

---

## Quickstart (v14.0 with real features)

```python
import sys
sys.path.insert(0, '/mnt/user-data/outputs/bioact_v14')
from predict_v14_real import predict, predict_batch

# Single prediction; auto-fetches real LION + ADMET if not cached
r = predict('CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC')
print(r['log10_flux_total'])         # 6.591
print(r['log10_flux_total_PI90'])    # (6.391, 6.791)
print(r['family'])                    # 'sSS-Nonsym' (via NN-vote)
print(r['block_B_real'])              # True (real LION used)
print(r['warnings'])                  # [] for clean predictions

# Batch (DataFrame output)
df = predict_batch(['<SMILES1>', '<SMILES2>'])
```

Honest metrics accessible from the bundle:
```python
import pickle
b = pickle.load(open('/mnt/user-data/outputs/bioact_v14/bioact_v14_bundle.pkl', 'rb'))
print(b['metrics']['v14_honest_pooled_mae'])     # 0.4032
print(b['metrics']['v14_honest_per_family'])      # per-family honest dict
print(b['metrics']['features_source'])            # 'REAL LION ... + REAL ADMET-AI'
```

---

## Roadmap

**Completed this session:**
- ✅ Real LION integration: cloned LNP_ML, installed chemprop 1.6.1 in venv, patched numpy 2.0 + torch 2.6 compatibility, built 236-SMILES LION cache via 5-CV ensemble × 6 tissues
- ✅ Real ADMET integration: installed admet-ai 2.0.1, built 236-SMILES ADMET cache
- ✅ Reran all of v14.0–v14.4 with real features; verified gate decisions shifted from proxy → real
- ✅ Honest 5-fold nested CV on both v14.0 (clean) and v14.3 (optimistic) — proved v14.0 is the honest production target
- ✅ predict_v14_real.py inference helper that auto-extends caches for novel SMILES
- ✅ Patched bundles with v14_honest_* metrics for traceability

**Next-priority improvements (v15):**
1. **Per-organ Stage B retrain** — currently only log10_flux_total; per-organ regressors (spleen/liver/lung/LN) still on v09's small per-organ samples. Clone v14.0 architecture, swap target, retrain.
2. **Stage A organ-selectivity classifier** — v13 has organ_dominant labels for ~280 rows; train a multiclass classifier.
3. **Scaffold-out CV** — stricter generalization test than random LOO.
4. **MAPIE jackknife+** — replace heuristic PI90 (tier-based ±0.20/0.30/0.45) with calibrated intervals.
5. **EXP_264 predictions** — run the 5 novel GA-Tris compounds from EXP_264-1.cdxml through v14.0 with real features. Currently they were predicted with v9.1 pKa only.
6. **Multi-task XGBoost** — train one model across log10_flux_total + per-organ targets simultaneously.

**Stretch goals:**
- Fine-tune LION on v13 (spec §4.2 estimates −0.05 to −0.10 log-MAE; requires ~12 GPU-hours)
- Soft family classifier returning Tanimoto-weighted family distribution
- Counterfactual/Pareto API for design suggestions (e.g., "what if chain X has N+1 carbons")
