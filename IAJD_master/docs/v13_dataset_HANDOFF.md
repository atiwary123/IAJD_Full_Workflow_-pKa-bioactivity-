# v13 Production Bundle — Handoff

**Date:** May 9, 2026
**Final bundle:** `iajd_bioact_v13_production_bundle.pkl` (0.67 MB)
**Version string:** `v2.0-MIN-v13-task123`

---

## TL;DR

You'd been running v13 dataset improvements (pharm1572, G1-Janus, HTM/TT dendrimers, 
245 architecture corrections, 12 IUPAC verifications, 40 duplicate merges) in parallel 
to my model-side work on v09. The right move was to apply this session's tools 
(family classifier, Bayesian shrinkage α, scaffold-out CV) to the v13 data rather 
than continuing on v09. That's done.

| Metric | v09 (this session start) | v13 (now) | Δ |
|---|---|---|---|
| Unique IAJDs | 156 | **264** | +69% |
| Families | 5 | **7** (added G1-Janus, HTM, TT dendrimers) | new architectures |
| Pooled LOO MAE | 0.4297 | **0.4194** | −0.0103 |
| **Scaffold-out CV MAE** | 0.6898 | **0.5947** | **−0.0951 (~14%)** |

The big win is in **scaffold-out CV** — the honest novel-scaffold number that 
matters for the model card. v13 generalizes meaningfully better to new chemistry 
than v09 did. Pooled LOO is roughly flat because the 3 new dendrimer families 
(G1-Janus, HTM, TT) have intrinsically harder LOO MAE (0.55–0.65), pulling the 
average up — but the model can now reason about them at all, which it couldn't 
on v09.

---

## What's in the bundle

```python
import pickle
bundle = pickle.load(open('iajd_bioact_v13_production_bundle.pkl', 'rb'))

# Model
bundle['stage_b_direct']        # Final XGBRegressor on full v13 (n=196)
bundle['feature_names']          # 20 MIN_FEATURES (no ADMET — see below)

# Training data
bundle['train_X']                # (196, 20)
bundle['train_y']                # log10_flux_total per IAJD (replicates collapsed)
bundle['train_families']         # 7 families
bundle['train_smiles']           # canonical SMILES
bundle['train_meta']             # full per-IAJD metadata DataFrame

# Honest LOO predictions (pre-computed)
bundle['loo_direct']             # XGB direct LOO predictions
bundle['loo_delta']              # leakage-aware delta-pair LOO predictions  
bundle['loo_blend']              # final α-blended LOO predictions

# Task 1: Bayesian-shrunk per-family α (τ=64)
bundle['alpha_by_family']        # {family → α}, all 7 families
bundle['alpha_default']          # 0.175 (global optimum)
bundle['alpha_meta']             # τ, per-family in-sample optima

# Task 2: family audit (informational)
bundle['family_audit']           # 5 ambiguous compounds (Dialkoxy/sSS), no fixes needed

# Task 3: scaffold-out CV
bundle['scaffold_cv_mae']        # 0.5947 pooled
bundle['scaffold_cv_per_family'] # per-family scaffold-out MAE  
bundle['scaffold_cv_meta']       # method + 1.40× inflation factor

# Task 5: ADMET cache (kept available, not in production model)
bundle['admet_cache']            # 195 SMILES → 10-d ADMET vector
bundle['admet_active_in_model']  # False (see "decision" below)
bundle['admet_meta']             # decision rationale

# Headline metrics
bundle['loo_metrics']            # pooled LOO MAEs
bundle['per_family_metrics']     # per-family α and MAE
bundle['comparison_with_v09']    # the table above
```

---

## Per-family α and LOO MAE (the operational numbers)

| Family | n | α (shrunk) | LOO MAE |
|---|---|---|---|
| sSS-Nonsym | 135 | 0.072 | 0.4042 |
| G1-Janus-Dendrimer | 21 | 0.311 | 0.5495 |
| PE-Tris | 15 | 0.272 | 0.3835 |
| GA-Tris | 10 | 0.326 | 0.3447 |
| HTM-Dendrimer | 7 | 0.252 | 0.6537 |
| Dialkoxybenzyl | 5 | 0.209 | 0.1074 |
| TT-Dendrimer | 3 | 0.217 | 0.5943 |

The dendrimer families (G1-Janus, HTM, TT) drive most of the pooled MAE. The 
sSS-Nonsym family (n=135, MAE 0.40) is the workhorse and is well-predicted.

The Dialkoxybenzyl 0.107 is an LOO-leakage artifact (n=5 with high replication), 
NOT a real generalization estimate — scaffold-out CV gives 0.245 for that family, 
which is the more honest number.

---

## Decisions made along the way

### 1. Drop ADMET-AI from the production v13 model

On v09 with per-organ regressors (n=46–51), real ADMET-AI gave −0.020 spleen / 
−0.019 LN MAE wins. On v13's pooled total-flux model with n=196 and the existing 
20-feature MIN_FEATURES set (which already has MolLogP, TPSA, FractionCSP3, 
Hydrophobic_Index, etc.), adding 10 ADMET features INCREASES LOO MAE by +0.013. 

The MIN_FEATURES set already captures most of what ADMET-AI's per-target MLP heads 
can offer; adding 10 redundant features over-fits at n=196.

The ADMET cache is kept in the bundle (`bundle['admet_cache']`) for future use:
- Per-target retraining (when per-organ data grows past n~80–100)
- Feature-selection studies
- Larger-dataset retrain (v14+)

### 2. Family audit found 5 ambiguous compounds, no fixes applied

IAJDs 81, 86, 105, 106, 107 are labeled "Dialkoxybenzyl" in v13 but the v3 
detector calls them "sSS-Nonsym". Investigation: they share the 3,5-dialkoxybenzyl 
core with both families and differ only in chain symmetry (asymmetric C11/C18 
chains, etc.) — the labels are correct by Percec lab convention. NN-vote against 
training set would resolve them to "Dialkoxybenzyl" correctly. No data changes 
needed.

### 3. Architecture-correction status preserved

v13's `v16_status` column tracks every correction applied since v10:
- 132 rows: features_refreshed,arch_corrected
- 96 rows: features_refreshed (clean from v10)
- 37 rows: merged duplicates IAJDs [97, 300]
- 31 rows: ceh_fixed,features_refreshed
- 12 rows: iupac_verified_iupac_mismatch,features_refreshed
- 3 rows: merged duplicates IAJDs [89, 267]

These are all **upstream of the bundle** — by the time the data hits 
`build_v13_production_final.py`, they're already applied.

---

## Files in `/mnt/user-data/outputs/`

```
iajd_bioact_v13_production_bundle.pkl  ★ FINAL bundle, ship this
v13_admet_cache.json                     ADMET features (kept for future use)
v13_loo_results.json                     Honest LOO results + τ sweep
v13_scaffold_cv_results.json             Honest scaffold-out CV results
v13_family_audit.json                    5 ambiguous Dialkoxy/sSS compounds (informational)
v13_loo_test.py                          Re-runnable LOO benchmark
v13_scaffold_cv.py                       Re-runnable scaffold-CV
build_v13_production_final.py            Bundle assembler (re-runnable)

(also still in outputs from v09 session — these are now superseded:)
iajd_bioact_v2_v09_LION_task12345_bundle.pkl    [SUPERSEDED]
HANDOFF.md (v09 version)                          [SUPERSEDED]
admet_cache_v09.json                              [SUPERSEDED — use v13_admet_cache.json]
iajd_bioact_v2.py                                 [v09 inference module — keep for reference]
```

---

## What this session NOT did

1. **No per-target retrain on v13.** The per-organ subsets in v13 (spleen n=77, 
   liver n=57, lung n=67, LN n=73) are now big enough to justify per-target Stage B 
   regressors with their own α and feature subsets. That work is out of scope here. 
   Recommendation: run `build_v13_production_final.py`-style work for each per-organ 
   target separately, with per-target α tuning from nested LOO.

2. **No deployment plumbing.** The v13 bundle has `train_meta`, `loo_blend`, etc., 
   but does not yet have a wrapped `predict()` function akin to the v09 
   `iajd_bioact_v2.py`. To deploy:
   - Adapt the v09 `predict_bioactivity()` interface (in 
     `/mnt/user-data/outputs/iajd_bioact_v2.py` — still relevant) to read from this 
     bundle's keys.
   - The Task 2 v3 family detector + `_count_aromatic_rings` helper from v09 
     `iajd_bioact_v2.py` are still correct and applicable.

3. **No PI half-width recalibration.** The v13 LOO residuals should be re-fit 
   for conformal half-widths (the v09 task3_pi_widening_proposal still applies — 
   for OOD queries with max_tanimoto < 0.50, scale half-widths by 
   `scaffold_cv_mae / loo_mae` ≈ 1.40× for v13).

4. **No retrain after applying per-target Stage B.** Per-organ predictions still 
   use the v09 task12345 bundle for spleen/LN. Should be redone on v13.

---

## Recommended next session

1. **Plumb v13 bundle into a deploy-ready API.** Replace v09 internals with v13 
   data, keep the same public surface (`predict_bioactivity(smiles, bundle)`).

2. **Per-organ Stage B on v13.** With spleen n=77, lung n=67, liver n=57, LN n=73, 
   do per-organ regressors with their own α and feature selection. This is where 
   ADMET-AI may help (per-target models on small subsets).

3. **Recalibrate conformal PI half-widths on v13 LOO + scaffold-CV residuals.**

4. **Update model card** with the v13 numbers — quote both LOO 0.4194 and 
   scaffold-CV 0.5947 per the established quoting policy.
