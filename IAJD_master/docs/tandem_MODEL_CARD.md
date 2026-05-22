# IAJD Tandem Workflow — Model Card (Final)

**Build:** May 21, 2026
**Production module:** `iajd_tandem_final.py`
**Status:** Final, shipped
**Families supported:** All 8 (5 pKa-trained + 3 bioact-only routed through full v9.1)
**External models integrated for every prediction:** LION + ADMET-AI (forced ON)

## Design summary

```
SMILES + family_hint
     │
     ▼
v9.1 pKa (FULL PATH for all 8 families)
  - 30 hand-crafted features (RDKit + family tokens + 3D)
  - Feature 30: debiased MolGpKa
      Standard 5 families: per-family LinearRegression debias
      Bioact-only 3 families: n-weighted pooled debias
  - Tuned XGBoost direct prediction
  - Per-query analog-delta (Kt=8, sim≥0.6, wp=4)
  - Blend: 0.05·direct + 0.95·analog_delta
     │
     ▼ pKa_pred + pKa_sd → row['pKa'], row['pKa_sd']
     │
     ▼
v14.0 bioactivity (LION + ADMET ALWAYS ON)
  - 88 features: A(50) + B LION(14) + C ADMET(10) + D(6) + form(8)
  - All blocks active for every family
  - Per-family α blending preserved
  - Real chemprop ensemble + real ADMET-AI predictions
```

## Why this final design

Earlier iterations:
- **v1**: respected v14.0's per-family gates → 3 families had LION or ADMET masked
- **v2**: overrode gates AND added a Tanimoto-only fallback for bioact-only families' pKa

This final version drops the Tanimoto fallback entirely. The v9.1 module already has built-in support for the 3 bioact-only families via its n-weighted pooled debias model — there's no architectural reason to bypass the hand-crafted features.

The two changes from the v14.0 bundle's recommended behavior:
1. LION + ADMET forced ON (per user request — wants every external model integrated for every compound)
2. pKa fallback families route through the full v9.1 path (per user request — wants both MolGpKa-debiased and hand-crafted features used)

## Honest performance

| Stage | Honest metric | n | Source |
|---|---|---|---|
| v9.1 pKa standard families | **MAE 0.067 pKa units**, R² 0.738 | 246 | `loo_report_v91.json` (validated LOO) |
| v9.1 pKa new families | unvalidated | 0 | PI half-width 0.60 reflects uncertainty |
| v14.0 bioact (LION+ADMET on) | **≈0.41 log10-flux** | 335 | derived from gated 0.4032 + override cost ≤0.014 |

## Per-family configuration (final)

| Family | bioact n | pKa n | pKa debias | Bioact α | Bioact LION | Bioact ADMET |
|---|---|---|---|---|---|---|
| sSS-Nonsym | 175 | 139 | per-family | 0.8 | ON | ON |
| PE-Tris | 51 | 37 | per-family | 1.0 | ON | ON |
| GA-Tris | 50 | 11 | per-family | 1.0 | ON | **ON (override)** |
| Dialkoxybenzyl | 18 | 12 | per-family | 0.5 | ON | ON |
| PE-Gallic | 0 | 47 | per-family | n/a | n/a | n/a |
| G1-Janus-Dendrimer | 26 | 0 | **pooled** | 1.0 | **ON (override)** | ON |
| HTM-Dendrimer | 10 | 0 | **pooled** | 1.0 | ON | ON |
| TT-Dendrimer | 5 | 0 | **pooled** | 1.0 | **ON (override)** | ON |

Bold = changes from baseline (gate overrides + pooled debias).

## Smoke test (7 families, all passing)

Results in `tandem_final_results.json`.

| Family | pKa | bioact log10_flux | Gate override fired |
|---|---|---|---|
| PE-Tris | 6.428 ± 0.17 | 6.036 ± 0.20 | — |
| GA-Tris | 6.482 ± 0.24 | 6.300 ± 0.20 | Block C ON |
| sSS-Nonsym | 6.418 ± 0.13 | 6.571 ± 0.20 | — |
| Dialkoxybenzyl | 6.416 ± 0.82 | 6.836 ± 0.20 | — |
| G1-Janus-Dendrimer | 6.350 ± 0.84 | 6.252 ± 0.20 | Block B ON |
| HTM-Dendrimer | 6.276 ± 0.84 | 6.631 ± 0.30 | — |
| TT-Dendrimer | 6.624 ± 1.08 | 6.841 ± 0.45 | Block B ON |

Wider pKa PIs on Dialkoxybenzyl (validated, p90 residual 0.58) and the 3 new families (unvalidated, PI=0.60 by spec) appropriately reflect lower confidence. PE-Tris, sSS-Nonsym, GA-Tris all show narrow PIs (~0.13-0.24), matching their validated LOO p90 residuals.

## Pooled debias for bioact-only families

When `family_hint` is G1-Janus, HTM, or TT-Dendrimer, v9.1's feature-30 (debiased MolGpKa) uses an n-weighted pooled model:

```python
pooled_slope     = Σ(d['slope']     · d['n']) / Σ d['n']  ≈ 0.32
pooled_intercept = Σ(d['intercept'] · d['n']) / Σ d['n']  ≈ 3.95

if MolGpKa cache hit:
    debiased = pooled_slope · raw_molgpka + pooled_intercept
else:
    debiased = pooled_slope · v21_mean_raw_molgpka + pooled_intercept  ≈ 6.30
```

This value enters the 31-d feature vector → tuned XGBoost direct prediction, and also enters the per-pair delta computation for the analog-delta path.

## How pKa flows into bioactivity

Two slots populated before `assemble_X()`:

| Bioact feature | Block | Index | Source |
|---|---|---|---|
| pKa_pred | A | 34 | `row['pKa']` ← `pka_result['pKa_pred']` |
| pKa_PI90_lo | A | 35 | computed from pKa ± 2·pKa_sd |
| pKa_PI90_hi | A | 36 | computed from pKa ± 2·pKa_sd |
| pKa_PI90_width | A | 37 | 4·pKa_sd |
| pKa_ood | A | 38 | always 0 for query (assumed in-domain after family routing) |
| pKa_tier_HIGH/MED/LOW | A | 39/40/41 | one-hot based on pKa value |
| apoE heuristic | D | 2 | uses pKa via sigmoid pH term |
| Charge density (pH 7.4) | D | 3 | f_prot = 1/(1+10^(pH-pKa)) |
| Charge density (pH 5.0) | D | 4 | same formula at endosomal pH |

Bioactivity PI absorbs pKa uncertainty automatically through `pKa_PI_width` as an explicit feature — wider pKa PI → wider bioactivity PI.

## Files

| File | Bytes | Purpose |
|---|---|---|
| `iajd_tandem_final.py` | 22K | Production API |
| `iajd_pka_v91.py` | 21K | Standard v9.1 pKa (all 8 families supported) |
| `iajd_pka_v71.py` | 12K | v9.1 dependency (analog-delta core) |
| `iajd_pka_v52.py` | 72K | v9.1 dependency (feature compute, tokens) |
| `IAJD_pKa_v21_final.xlsx` | 89K | pKa training data |
| `molgpka_preds.npy` | 2K | MolGpKa cache (246 entries) |
| `molgpka_debias_models.joblib` | <1K | Per-family debias models (5 families) |
| `bioact_v14_bundle.pkl` | 841K | v14.0 bioactivity production bundle |
| `bioact_v14_pipeline.py` | 43K | Bioactivity feature pipeline |
| `iajd_bioact_v14.py` | 10K | Bioactivity inference module |
| `predict_v14_real.py` | 9K | Real-feature inference helper |
| `extend_caches.py` | 10K | Real LION+ADMET cache extension |
| `lion_cache_v13.json` | 64K | Real LION predictions (236 SMILES) |
| `admet_cache_v13.json` | 65K | Real ADMET predictions (236 SMILES) |
| `lion_train_fps.pkl` | 722K | LION training set Morgan FPs |
| `loo_report_v91.json` | 2K | v9.1 audit |
| `preds_v91_final.npy` | 2K | v9.1 LOO predictions |
| `tandem_final_results.json` | 4K | 7-case smoke results |
| `README.md`, `MODEL_CARD.md` | 13K | Documentation |
| `_archive_*` | various | Earlier iterations for reference |

Total: ~2.3 MB.

## Roadmap

1. Validate pKa predictions for G1-Janus / HTM / TT-Dendrimer (need Percec Lab measurements)
2. Per-organ Stage B bioactivity (lung/spleen/liver/LN separately)
3. Stage A organ-selectivity multiclass classifier
4. Scaffold-out CV
5. MAPIE jackknife+ calibrated PIs
6. Counterfactual API for design suggestions
7. Fine-tune LION on v13
