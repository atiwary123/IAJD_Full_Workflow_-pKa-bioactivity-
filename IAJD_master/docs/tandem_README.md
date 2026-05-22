# IAJD Tandem Workflow — Final

**Full-stack IAJD prediction.** Single function call: SMILES + family_hint → pKa + log10-flux + PIs + diagnostics. All 8 families supported. All compounds use the full v9.1 pKa architecture (30 hand-crafted features + debiased MolGpKa + XGBoost analog-delta) and the full v14.0 bioactivity feature set (LION + ADMET-AI always ON).

## Production module: `iajd_tandem_final.py`

```python
import sys
sys.path.insert(0, '/mnt/user-data/outputs/iajd_tandem')
from iajd_tandem_final import load_tandem_bundle, predict_iajd

b = load_tandem_bundle()
r = predict_iajd('<SMILES>', b, family_hint='PE-Tris')
print(r['combined_summary']['pka'])
print(r['combined_summary']['bioact'])
print(r['combined_summary']['family'])
```

## Architectural choices

### Choice 1: All 8 families use the full v9.1 pKa path
Every family — including G1-Janus-Dendrimer, HTM-Dendrimer, TT-Dendrimer that have no v21 training compounds — gets the full hand-crafted-feature + MolGpKa + analog-delta treatment:

- 30 base features (MolWt, LogP, TPSA, HBA/HBD, RB, Aromatic rings, Bertz CT, χ indices, LabuteASA, Gasteiger charges, family tokens, 3D pct buried volume / asphericity, …)
- Feature 30 = debiased MolGpKa:
  - Standard 5 families: per-family LinearRegression debias model
  - Bioact-only 3 families: **n-weighted pooled debias** across all 5 v21 families
- Tuned XGBoost direct prediction
- Per-query analog-delta (Kt=8, sim≥0.6, wp=4)
- Final = 0.05 × direct + 0.95 × analog_delta

No Tanimoto-only fallback. No special case for any family. Same code path for all 8.

For G1-Janus / HTM / TT-Dendrimer queries, the pooled debias slope (≈0.32) × the v21 mean raw MolGpKa (7.884) + pooled intercept gives a debiased MolGpKa feature value of ≈6.30, which feeds into both the direct XGBoost and the analog-delta calls.

### Choice 2: LION + ADMET always ON for bioactivity
The v14.0 bundle's per-family LION/ADMET gates are overridden — every prediction uses all 88 features. 3 gates flip ON:

| Family | Bundle gate | Override | Honest MAE cost |
|---|---|---|---|
| GA-Tris Block C (ADMET) | OFF | **ON** | +0.014 |
| G1-Janus Block B (LION) | OFF | **ON** | +0.001 |
| TT-Dendrimer Block B (LION) | OFF | **ON** | +0.001 |

The `gate_overrides_applied` field documents when an override fired. Per-family α blending (sSS=0.8, Dialkoxybenzyl=0.5, others=1.0) is unchanged.

## Honest performance

| Stage | Honest MAE | n | Source |
|---|---|---|---|
| v9.1 pKa (5 trained families) | **0.067** pKa units, R²=0.738 | 246 | `loo_report_v91.json` (validated LOO) |
| v9.1 pKa (3 bioact-only families) | unvalidated | 0 | pooled debias model used; PI widths set to 0.60 to reflect uncertainty |
| v14.0 bioact (LION+ADMET on for all) | **≈0.41** log10-flux | 335 | derived from gated honest 0.4032 + worst-case override cost ≤0.014 |

## Smoke test (7 cases, all families)

Results saved to `tandem_final_results.json`. Sample:

| Family | pKa | log10_flux | LION/ADMET active | Override |
|---|---|---|---|---|
| PE-Tris | 6.428 | 6.036 | both ON | none |
| GA-Tris | 6.482 | 6.300 | both ON | ADMET (Block C) |
| sSS-Nonsym | 6.418 | 6.571 | both ON | none |
| Dialkoxybenzyl | 6.416 | 6.836 | both ON | none |
| G1-Janus-Dendrimer | 6.350 | 6.252 | both ON | LION (Block B) |
| HTM-Dendrimer | 6.276 | 6.631 | both ON | none |
| TT-Dendrimer | 6.624 | 6.841 | both ON | LION (Block B) |

All bioact-only families went through the full v9.1 path with the pooled debias model — same architecture as the 5 trained families.

## How pKa flows into bioactivity

The tandem injects `pKa_pred` and `pKa_sd` into the bioactivity row before `assemble_X()`:

```python
row['pKa']     = pka_result['pKa_pred']
row['pKa_sd']  = pka_result['pKa_sd']
```

Bioact Block A indices 34–41 (pKa propagation: pred, PI bounds, width, OOD flag, tier one-hots) and Block D indices 2–4 (apoE heuristic, charge density at pH 7.4 / 5.0) pick up these values automatically. The bioactivity PI absorbs pKa uncertainty through `pKa_PI_width` as an explicit feature.

## Returned dict structure

```
{
  'smiles', 'canonical_smiles', 'version',
  'pka': {                                       # full v9.1 output
    'pKa_pred', 'pKa_PI_90', 'pKa_sd',
    'family_assigned', 'confidence_tier', 'ood_flag',
    'source',                                    # 'v9.1_predicted' | 'measured'
    'max_tanimoto_to_training', 'nearest_neighbors',
    'direct_xgb_pred', 'analog_delta_pred', 'debiased_molgpka_feature',
    'routing', 'warnings'
  },
  'bioactivity': {
    'log10_flux_total', 'log10_flux_total_PI90', 'confidence_tier',
    'family_detected', 'family_used', 'family_method',
    'alpha_used',
    'block_B_active': True,                      # always True
    'block_C_active': True,                      # always True
    'block_B_real',                              # True if real LION cache hit
    'gate_overrides_applied',                    # list of override notes
    'pred_direct', 'pred_analog', 'max_tanimoto',
    'nearest_neighbors', 'warnings',
    'pka_used_in_features', 'pka_sd_used_in_features'
  },
  'family_reconciliation': {...},
  'flow': {...},
  'combined_summary': {...}
}
```

## Files

| File | Purpose |
|---|---|
| `iajd_tandem_final.py` | **Production API — use this** |
| `iajd_pka_v91.py` | v9.1 wrapper (supports all 8 families) |
| `iajd_pka_v71.py`, `iajd_pka_v52.py` | v9.1 dependencies |
| `IAJD_pKa_v21_final.xlsx` | pKa training data (246 IAJDs, 5 families) |
| `molgpka_preds.npy`, `molgpka_debias_models.joblib` | MolGpKa caches |
| `bioact_v14_bundle.pkl` | v14.0 bioactivity bundle |
| `bioact_v14_pipeline.py`, `iajd_bioact_v14.py`, `predict_v14_real.py`, `extend_caches.py` | Bioactivity inference |
| `lion_cache_v13.json`, `admet_cache_v13.json`, `lion_train_fps.pkl` | Real LION + ADMET caches |
| `loo_report_v91.json`, `preds_v91_final.npy` | v9.1 verification |
| `tandem_final_results.json` | 7-case smoke results |
| `README.md`, `MODEL_CARD.md` | Documentation |
| `_archive_*` | Earlier iterations (v1 gated, v2 Tanimoto fallback) |

## Known caveats

1. **`family_hint` is REQUIRED** unless auto-detected as PE-Tris (v9.1's Bug-2 guard from v8.2). Allowed values: `sSS-Nonsym`, `PE-Tris`, `GA-Tris`, `PE-Gallic`, `Dialkoxybenzyl`, `G1-Janus-Dendrimer`, `HTM-Dendrimer`, `TT-Dendrimer`.

2. **MolGpKa live calls disabled** in this environment (torch_scatter ABI mismatch). For SMILES not in the 246-compound cache, the debias feature uses the family median (standard families) or pooled debias evaluated at the v21 mean raw MolGpKa value (≈6.30 for bioact-only families). Warnings: `MOLGPKA_LIVE_FAILED`, `DEBIAS_FALLBACK`, `POOLED_DEBIAS_AT_MEAN_MOLGPKA`. The analog-delta path is unaffected and provides the dominant signal (α=0.95).

3. **pKa for bioact-only families is unvalidated** — no held-out pKa data exists for G1-Janus, HTM, or TT-Dendrimer. The v9.1 PI table assigns 0.60 base half-width for these families, reflecting that uncertainty.

4. **Honest bioactivity MAE is ~0.005-0.014 worse** than the gated v14.0 (0.4032 → ~0.41-0.42). This is the documented cost of forcing LION+ADMET ON for every family.

5. **Novel bioactivity SMILES** auto-fetch real LION (~10s, requires chemprop 1.6.1 venv) and real ADMET (~0.5s, requires admet-ai). Falls back to RDKit proxies with `block_B_real: False` warning if those aren't available.

6. **PE-Gallic compounds** have pKa training data (47 compounds) but no bioactivity training data — bioactivity will route to `family_used='Unknown'` for them.

## Roadmap

1. Validate pKa predictions for G1-Janus / HTM / TT-Dendrimer (requires Percec Lab measurements)
2. Per-organ Stage B bioactivity (currently log10_flux_total only)
3. Stage A organ-selectivity classifier
4. Scaffold-out CV stricter test than random LOO
5. MAPIE jackknife+ calibrated PIs
6. Counterfactual API for design suggestions
7. Fine-tune LION on v13 (−0.05 to −0.10 log10-flux MAE expected)
