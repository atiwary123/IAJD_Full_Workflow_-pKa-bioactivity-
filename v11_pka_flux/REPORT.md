# pKa-Dominant Flux Model — v11 Experiment Report

**Hypothesis.** Predicted molecular pKa, central to endosomal escape, can carry
most of the signal for ionizable-amine-driven log10 flux. Compared head-to-head
with the v14 cascade (3D + LION + ADMET + Family-α blend) on the same strict-LOO
protocol family-stratified k=5 fold (the prompt's sanctioned fallback to true nested LOO).

**Setup.**
- Bioact table: 247 rows with `log10_flux_total` and `predicted_pKa`.
- Predicted pKa: v9.1 LOO on 261 rows present in v21 (pooled MAE on those = 0.1556, expected ≈0.067), v9.1 full-bundle on 12 remaining rows.
- Family-stratified k=5; per-fold full refit of M0/M1/M2 (XGB depth-4) and the M3 v14 direct head + analog-delta blend on n−fold rows.
- Replicate noise floor (bioact SI): 0.28 log-units.

## Models
- **M0**: predicted_pKa only (1 col). Plus a quadratic OLS sanity check.
- **M1**: M0 + family one-hot + (pKa × family) interaction.
- **M2**: M1 + tail/shape descriptors: MolLogP, FractionCSP3, RotatableBonds, NumAromaticRings, TPSA, LabuteASA, HeavyAtomCount, linker_carbons.
- **M3**: v14 baseline (88-feature pipeline, direct XGB + analog-delta blend with per-family α and B/C gates), refit per fold on the bundle's stored `X_train/y_train/fps_train`.

## Pooled results
| Model | MAE | RMSE | R2 | Spearman | n |
|---|---|---|---|---|---|
| M0 | 0.679 | 0.926 | -0.178 | 0.248 | 247 |
| M0_quad | 0.654 | 0.835 | 0.043 | 0.169 | 247 |
| M1 | 0.608 | 0.806 | 0.108 | 0.338 | 247 |
| M2 | 0.535 | 0.710 | 0.309 | 0.495 | 247 |
| M3 | 0.479 | 0.624 | 0.434 | 0.562 | 239 |

## Per-family MAE
| Family | n | M0-MAE | M1-MAE | M2-MAE | M3-MAE |
|---|---|---|---|---|---|
| Dialkoxybenzyl | 10 | 0.386 | 0.279 | 0.301 | 0.290 |
| G1-Janus-Dendrimer | 21 | 0.905 | 0.825 | 0.742 | 0.751 |
| GA-Tris | 20 | 0.887 | 0.754 | 0.456 | 0.559 |
| PE-Gallic | 11 | 1.343 | 0.964 | 1.025 | 0.731 |
| PE-Tris | 41 | 0.914 | 0.820 | 0.602 | 0.494 |
| sSS-Nonsym | 144 | 0.520 | 0.492 | 0.475 | 0.423 |

## Family pKa optima (inverted-U fits on held-out predictions)
| Family | n | pKa_opt | curvature b | rmse | note |
|---|---|---|---|---|---|
| Dialkoxybenzyl | 10 | 3.000 | 0.003 | 0.277 | optimum_at_boundary_no_clear_peak |
| G1-Janus-Dendrimer | 21 | 7.038 | 3.396 | 0.727 | — |
| GA-Tris | 20 | 6.740 | 6.588 | 0.547 | — |
| PE-Gallic | 11 | 6.273 | 13.134 | 0.933 | — |
| PE-Tris | 41 | 3.000 | 0.754 | 1.174 | optimum_at_boundary_no_clear_peak |
| sSS-Nonsym | 144 | 6.452 | 28.130 | 0.523 | — |

## SHAP top features for M2 (mean |SHAP value|, full-table refit)
| Rank | Feature | mean |SHAP\| |
|---|---|---|
| 1 | `predicted_pKa` | 0.1767 |
| 2 | `TPSA` | 0.1425 |
| 3 | `MolLogP` | 0.1307 |
| 4 | `RotatableBonds` | 0.1062 |
| 5 | `pKa_x_fam_PE-Tris` | 0.0976 |
| 6 | `pKa_x_fam_sSS-Nonsym` | 0.0902 |
| 7 | `NumAromaticRings` | 0.0730 |
| 8 | `LabuteASA` | 0.0600 |
| 9 | `FractionCSP3` | 0.0452 |
| 10 | `fam_PE-Tris` | 0.0321 |

## Caveats
- 21 of the v21-LOO rows had a family-table label that disagreed with the v9.1 detector. They are routed through the table's family for M1/M2 features (consistent with how the bioact pipeline would run them).
- M3 baseline was refit per fold using the v14 bundle's stored `X_train/y_train/fps_train`. 8 bioact rows have a canonical SMILES that does not match any v14 training row and were skipped in the M3 column (M3 n=239 vs n=247). They remain in M0/M1/M2 columns.
- `linker_carbons` is mostly NaN in the bioact table (342/361) and gets median-imputed in M2; treat any importance from that column as proxy for family rather than chain length.

Files emitted alongside this report:
- `predictions.csv` — per-fold, per-row predictions and residuals.
- `metrics.json` — full pooled + per-family metric table.
- `shap_M2.png` — SHAP summary for M2 (refit on full table) showing whether predicted_pKa is the dominant feature in practice.
- `calibration.png` — 2×2 panel of predicted vs measured for M0/M1/M2/M3.
- `family_optima.csv` — per-family fitted pKa_opt and curvature.
- `predicted_pka_cache.csv` — pKa values per row (cached for reuse).

## Honest interpretation
- The single-feature M0 establishes the raw pKa→flux signal floor.
- M1 quantifies how much of the variance is *family-specific pKa response* vs *family-level intercept differences*.
- M2 adds the minimum tail-descriptor set the literature argues survives pKa-control — if SHAP shows tail descriptors out-importance predicted_pKa, the "pKa-dominant" framing is too strong.
- M3 is the reference. Beating M3 with M2 would suggest the v14 3D-descriptor stack is overfitting; tying within noise would suggest the pKa-centric representation is a strictly cheaper and more interpretable choice for similar accuracy; underperforming by more than ~0.05 log-units says the 3D/LION/ADMET features carry orthogonal signal.

## Verdict
**pKa-dominant model is a useful baseline but underperforms v1.1 by 0.055 log units** (M2 MAE=0.535, M3 MAE=0.479).
