# pKa-Dominant Flux Model — v11 Experiment Report

**Hypothesis.** Predicted molecular pKa, central to endosomal escape, can carry
most of the signal for ionizable-amine-driven log10 flux. Compared head-to-head
with the v14 cascade (3D + LION + ADMET + Family-α blend) on the same strict-LOO
protocol family-stratified k=5 fold (the prompt's sanctioned fallback to true nested LOO).

**Setup.**
- Bioact table: 344 rows with `log10_flux_total` and `predicted_pKa`.
- Predicted pKa: v9.1 LOO on 356 rows present in v21 (pooled MAE on those = 0.1522, expected ≈0.067), v9.1 full-bundle on 13 remaining rows.
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
| M0 | 0.580 | 0.834 | -0.039 | 0.322 | 344 |
| M0_quad | 0.640 | 0.819 | -0.003 | 0.004 | 344 |
| M1 | 0.543 | 0.788 | 0.073 | 0.369 | 344 |
| M2 | 0.475 | 0.707 | 0.254 | 0.515 | 344 |
| M3 | 0.430 | 0.630 | 0.386 | 0.598 | 335 |

## Per-family MAE
| Family | n | M0-MAE | M1-MAE | M2-MAE | M3-MAE |
|---|---|---|---|---|---|
| Dialkoxybenzyl | 18 | 0.198 | 0.155 | 0.125 | 0.151 |
| G1-Janus-Dendrimer | 27 | 0.824 | 0.700 | 0.737 | 0.713 |
| GA-Tris | 58 | 0.677 | 0.645 | 0.540 | 0.645 |
| HTM-Dendrimer | 10 | 0.724 | 0.409 | 0.342 | 0.393 |
| PE-Tris | 51 | 0.784 | 0.740 | 0.496 | 0.385 |
| TT-Dendrimer | 5 | 0.824 | 0.752 | 0.660 | 0.806 |
| sSS-Nonsym | 175 | 0.475 | 0.470 | 0.446 | 0.359 |

## Family pKa optima (inverted-U fits on held-out predictions)
| Family | n | pKa_opt | curvature b | rmse | note |
|---|---|---|---|---|---|
| Dialkoxybenzyl | 18 | 12.000 | 0.013 | 0.232 | optimum_at_boundary_no_clear_peak |
| G1-Janus-Dendrimer | 27 | 12.000 | 0.061 | 0.836 | optimum_at_boundary_no_clear_peak |
| GA-Tris | 58 | 6.905 | 4.096 | 0.636 | — |
| HTM-Dendrimer | 10 | 3.000 | 0.623 | 0.581 | optimum_at_boundary_no_clear_peak |
| PE-Tris | 51 | 3.000 | 0.680 | 1.161 | optimum_at_boundary_no_clear_peak |
| TT-Dendrimer | 5 | — | — | — | too_few_rows |
| sSS-Nonsym | 175 | 6.449 | 23.134 | 0.570 | — |

## SHAP top features for M2 (mean |SHAP value|, full-table refit)
| Rank | Feature | mean |SHAP\| |
|---|---|---|
| 1 | `predicted_pKa` | 0.1577 |
| 2 | `MolLogP` | 0.1352 |
| 3 | `TPSA` | 0.1202 |
| 4 | `RotatableBonds` | 0.1006 |
| 5 | `pKa_x_fam_sSS-Nonsym` | 0.0736 |
| 6 | `LabuteASA` | 0.0726 |
| 7 | `pKa_x_fam_PE-Tris` | 0.0522 |
| 8 | `FractionCSP3` | 0.0498 |
| 9 | `fam_PE-Tris` | 0.0462 |
| 10 | `NumAromaticRings` | 0.0420 |

## Caveats
- 56 of the v21-LOO rows had a family-table label that disagreed with the v9.1 detector. They are routed through the table's family for M1/M2 features (consistent with how the bioact pipeline would run them).
- M3 baseline was refit per fold using the v14 bundle's stored `X_train/y_train/fps_train`. 9 bioact rows have a canonical SMILES that does not match any v14 training row and were skipped in the M3 column (M3 n=335 vs n=344). They remain in M0/M1/M2 columns.
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
**pKa-dominant model is a useful baseline but underperforms v1.1 by 0.046 log units** (M2 MAE=0.475, M3 MAE=0.430).
