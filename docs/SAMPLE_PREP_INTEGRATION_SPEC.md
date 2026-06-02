# SPEC — Integrating sample-prep parameters into the IAJD models

**Status:** IMPLEMENTED (flag-gated, default-off) + validated by eval — **no production model
re-trained yet** (a weighted retrain is a user-triggered run; see §11). · **Written/updated:** 2026-06-02
**Depends on:** `docs/SAMPLE_PREP_PARAMETERS.md`, `build_sample_prep_features.py`, the `sp_*` block now in all live datasets.
**Audience:** whoever runs the next bioactivity/pKa training pass.

---

## 0. Goal, non-goals, and the honest prior

**Goal.** Use the newly-extracted `sp_*` sample-prep block to (a) reduce label-noise impact via
**provenance/variance-aware sample weighting**, and (b) test whether the one genuinely-varying
formulation covariate (`sp_assembly_pH`) adds predictive signal — under a CV design that does not
fool us with the paper confound.

**Non-goals.** Do not retrain or ship a model under this spec. Do not silently edit any label.
Do not add the constant `sp_*` columns as features (they carry zero variance in this corpus).

**Honest prior (set expectations before touching code).** From `docs/SAMPLE_PREP_PARAMETERS.md`:
the Percec corpus is highly standardized, so prep variation is small and mostly **confounded with
`paper`/`family`**. We therefore expect:
- **Weighting (sp_confidence + replicate variance): the likely real win** — better calibration and a
  cleaner fit by down-weighting the 8 uncited `novel_2026` rows, the inferred `bm4c01599` rows, and
  `n_mice = 1` measurements.
- **`sp_assembly_pH`: a within-corpus covariate, not a cross-family generalization lever** — every
  pH-5.2 row is `ja1c05813`, so leave-one-family-out CV mostly cannot exploit it. Treat any LOFO gain
  with suspicion and test it under the controls in §5.
- Net measured-R² gains will be **modest** — label noise caps the measured score (see
  `docs/SMALL_NOISY_DATA_METHODS.md`); the win is honesty/calibration, not a headline number.

---

## 1. Inputs — which `sp_*` columns are in scope

Use **only the columns that vary** (the rest are protocol documentation, zero-variance here):

| column | role in modeling | bioact variance |
|---|---|---|
| `sp_assembly_pH` | **candidate feature** (formulation covariate) | 4.0 / 5.2 |
| `sp_imaging_time_h` | **candidate feature** (= `T_hours`; assay covariate) | 4–7 h, 10 levels |
| `sp_confidence` | **sample weight input** (NOT a feature) | 1.0 / 0.85 / 0.65 / 0.35 / 0.25 |
| `sp_dialyzed`, `sp_mRNA_conc_final_mg_mL`, `sp_IAJD_mRNA_mass_ratio` | **excluded** (constant in bioact; the only varying instance, IAJD 9, is pKa-only) | — |
| all other `sp_*` | **excluded** (constant) — keep for documentation/future non-Percec data | — |

`sp_provenance` is the categorical source of `sp_confidence`; keep it for filtering/auditing, not as a feature.

For the **pKa model**, the relevant axis is `sp_pKa_method` (`_code` 1 vs the TNS option for ja5c07232)
and `sp_confidence`; `sp_assembly_pH`/`sp_imaging_time_h` are irrelevant to a molecular pKa and must
**not** be fed to the pKa model.

---

## 2. Feature contract (if §5 shows the covariates earn their place)

- `sp_assembly_pH` → **binary indicator** `is_pH52 = (sp_assembly_pH == 5.2)`. Do not treat as continuous
  (only two levels; 3.0 is pKa-only). Median-fill is not needed (full coverage in bioact).
- `sp_imaging_time_h` → use directly **or** prefer the existing `T_hours` (identical where present).
  Optional centering at 4 h. Beware: for `ja1c09585`, values >7 h are data-quality-suspect (§4) — these
  should be down-weighted, not trusted as a clean covariate.
- Register as a **separate feature set** in the benchmark (do not silently fold into Morgan/RDKit), so
  the ablation in §8 can attribute any change to prep.

---

## 3. `sp_confidence` → sample weighting (the primary mechanism)

No current trainer uses `sample_weight` (verified: zero hits across `train_*.py`). Add it.

**Combine three independent reliability signals into one per-row weight** `w_i`:
1. **Provenance:** `sp_confidence` ∈ {1.0, 0.85, 0.65, 0.35, 0.25}.
2. **Replicate noise:** `y_std` (already computed in `benchmark_features_gp_lofo.py::load_labels`) — the
   per-compound replicate spread of `log10_flux_total`.
3. **Cohort size:** `n_mice` (and/or `n_replicates`) — `n_mice == 1` rows are high-variance.

**Recommended forms (pick per model, pre-register before running):**
- **GP (primary model):** pass per-point noise via the `alpha` vector instead of a scalar:
  `alpha_i = σ0² · (y_std_i² + ε) / sp_confidence_i`. This is the principled heteroscedastic route —
  low-confidence / high-spread points get more slack, not deletion. (`WhiteKernel` stays for the
  irreducible floor.)
- **RF / Ridge / WLS baselines:** `sample_weight = w_i`, with
  `w_i = sp_confidence_i · n_eff_i`, where `n_eff_i = n_mice_i` (fallback `n_replicates_i`, else 1).
- **Robust sensitivity:** also run a **hard filter** variant (`sp_confidence ≥ 0.50`, i.e. drop the 8
  `novel_2026` rows) and report both — weighting and filtering should agree in direction.

**Constraint:** weighting changes the *fit*, not the held-out *truth*. In CV, apply weights to **training**
folds only; held-out metrics stay unweighted (or report both unweighted and weight-normalized).

---

## 4. Data-quality handling (surfaced by this audit; do not fabricate fixes)

1. **`n_mice == 1`** (many `pharmaceutics1501572` rows): high per-row variance. Handle via the weight in
   §3 (do not drop). Add a boolean `dq_low_n` for the ablation.
2. **`ja1c09585` `T_hours` > 7 h** (e.g. ~10.25): the SI reports imaging only 4–6 h, so these are
   **suspected data-entry/averaging artifacts**. Policy: (a) **flag** them (`dq_time_suspect`), (b)
   down-weight via §3, (c) **do not overwrite** the value — instead open a follow-up to re-derive
   `T_hours` for `ja1c09585` from the source table. Record the decision; never silently "correct" a label.
3. Keep all of the above as **flags + weights**, never as silent label edits (honors the project's
   no-fabrication / honesty rules).

---

## 5. The confound, and the CV design that controls for it

**The confound:** `sp_assembly_pH == 5.2` ⊂ `ja1c05813` ⊂ {PE-Gallic, sSS} families; and within
`ja1c05813` the pH-4.0 subset were partly the better compounds re-run at pH 4.0. So a naive gain could
be the model learning "ja1c05813-ish" rather than a pH effect.

**Required tests (run all three; report all three honestly):**
1. **Leave-one-family-out (existing default, `--split lofo`):** the headline generalization test. Expect
   `sp_assembly_pH` to give ~0 here (held-out families are all pH 4.0). If it *helps* LOFO, be suspicious
   and explain why.
2. **Within-`ja1c05813` split** (new): restrict to the 48 `ja1c05813` rows, random/scaffold 5-fold,
   compare residual variance & calibration **with vs without** `is_pH52`. This is where a real pH effect
   would show up. A drop in residual variance here = legitimate within-corpus covariate.
3. **Paper-as-group control:** add a `paper`/`family` indicator to the baseline; then test whether
   `is_pH52` adds anything **beyond** paper identity (nested-model ΔLL / partial-R², or permutation
   importance with `paper` already in). If `is_pH52` ≈ 0 once `paper` is in the model, report it as
   "subsumed by paper" — that is a valid, honest finding.

Always print against the **mean-predictor floor** and the **noise ceiling** (`y_std` mean) the harness
already reports — a feature that doesn't beat the floor is not adopted.

---

## 6. Integration points (exact, staged; each step independently testable)

**Stage A — measurement harness first (no production impact).** `benchmark_features_gp_lofo.py`:
- In `load_labels()`: also carry `sp_confidence`, `sp_assembly_pH`, `n_mice` per compound (group-first,
  like `family`). Add `y_std` is already there.
- Add `f_prep(df)` feature builder → `[is_pH52, imaging_time_centered]` registered as feature set
  `"prep"`, plus combo sets `"morgan+prep"`, `"rdkit+prep"`, `"physics+prep"`.
- In `models()`/`evaluate()`: thread an optional `weights` arg → GP `alpha` vector and RF/Ridge
  `sample_weight` per §3 (train folds only).
- Add `--weight {none,conf,conf_var,filter}` and `--within ja1c05813` flags to drive §5 tests.

**Stage B — production fit (only if Stage A passes §7).** `train_v15_physics_ml.py`:
- Read the same canonical bioact xlsx (already has `sp_*`). Extend `PHYSICS_COLS` (or a new
  `COVARIATE_COLS`) with the §2 features **only if** Stage A showed a within-corpus gain.
- Pass `sample_weight` per §3 into the Ridge/ensemble fit.
- Keep the change behind a flag (`--use-prep`, `--weight`) so the baseline remains reproducible.

**Stage C — pKa model (separate, smaller).** `train_pka_v92.py` / `pka_primary_model.py`:
- Only `sp_confidence` weighting + optional `sp_pKa_method_code`. Do **not** add formulation covariates.

**Stage D — NN transfer (optional).** `nn/train_transfer.py`: `sp_confidence` as loss weight on the IAJD
fine-tune head; `iajd_transfer_input.csv` already carries the `sp_*` block.

Touch order: **A → (gate) → B → C → D.** Do not start B until A’s gate (§7) is met.

---

## 7. Acceptance gates (pre-register before running; adopt only if met)

For each change, compare to the **current baseline** (same feature sets, no prep, no weights) on the
**same splits/seeds**:

- **G1 (do no harm):** weighted/with-prep model’s LOFO pooled R² ≥ baseline − 0.01 **and** MAE not worse
  beyond noise. (We will not regress generalization to add prep.)
- **G2 (weighting win):** under §3 weighting, **calibration improves** (e.g. held-out NLL / interval
  coverage) OR pooled MAE improves by ≥ 1 SE (bootstrap, `bootstrap_ci.py`), with weighting and
  filtering agreeing in sign.
- **G3 (covariate earns its place):** `is_pH52` adopted **only if** §5 test 2 shows a within-`ja1c05813`
  residual-variance/calibration drop **and** §5 test 3 shows it is not fully subsumed by `paper`.
  Otherwise document "subsumed by paper / no marginal signal" and **drop the feature.**
- **G4 (honesty):** every reported delta carries the mean-floor and `y_std` noise-ceiling context and a
  bootstrap CI. No cherry-picking a split.

If only G1+G2 pass → ship **weighting**, not the pH feature. That is the expected outcome.

---

## 8. Ablation matrix (what to report)

Rows = feature sets {Morgan, RDKit, Physics, +prep each}; Cols = weighting {none, conf, conf+var, filter≥0.5}.
Splits = {LOFO, within-ja1c05813, cluster}. Metrics = pooled R², MAE, Spearman, (GP) NLL/coverage, all
vs mean-floor, with bootstrap CIs. Deliverable: one table + a one-paragraph honest read
(“weighting helped X; assembly_pH was/ wasn’t marginal after paper; here’s the noise ceiling”).

---

## 9. Risks / do-not

- **Don’t** add the 28 constant `sp_*` columns as features (zero variance → no signal, only dilution).
- **Don’t** let `sp_confidence` leak as a *feature* (it’s meta-label about reliability, not chemistry).
- **Don’t** apply test-fold weights or filter the **held-out** set — that inflates scores.
- **Don’t** overwrite `ja1c09585` `T_hours` or any label; flag + down-weight + open a follow-up.
- **Don’t** interpret a LOFO gain from `is_pH52` as causal — it’s confounded with paper/selection (§5).
- **Don’t** feed formulation covariates to the pKa model.

---

## 10. Rollback / provenance

- Datasets: restore `*.PRE_SAMPLEPREP.*`; rebuild with `python build_sample_prep_features.py`.
- All weighting/feature changes stay behind flags (`--use-prep`, `--weight`, `--within`) so the
  pre-spec baseline is always reproducible by omitting them.
- Source-of-truth for every prep value + provenance: `sample_prep_protocols.json`,
  `docs/SAMPLE_PREP_PARAMETERS.md`.

---

### Appendix — one-line summary for the impatient
Add `sample_weight = sp_confidence × n_mice` (and GP per-point `alpha` from `y_std`), test
`is_pH52` **within ja1c05813 only** with `paper` already in the model, gate on “do no harm to LOFO +
improve calibration,” and expect to ship the **weighting**, not the pH feature.

---

## 11. Implementation log (2026-06-02)

**Done (flag-gated, default-off — verified no-op when `IAJD_USE_PREP_WEIGHTS` unset):**
- `sample_prep_weights.py` — shared helper: `row_reliability_weight`, `gp_alpha`,
  `prep_feature_matrix`, `dq_flags`, `within_paper_mask`, and the bundle/SMILES aligners
  `bundle_weight`/`maybe_bundle_weight` (bioact, by `smis_train`) and
  `smiles_weight`/`maybe_pka_weight` (pKa-appropriate: `sp_confidence` + `pKa_sd`, no `n_mice`).
- Bioactivity trainers wired with `maybe_bundle_weight(b14, BIOACT)` → `sample_weight` on every
  `.fit` (LOO + bootstrap + final): `train_bioact_ensemble.py`, `train_per_organ.py`,
  `train_binary_classifier.py`.
- pKa trainer `train_pka_v92.py`: XGB head LOO + final fit weighted via `maybe_pka_weight`
  (analog & MolGpKa heads and the blend optimizer left unweighted by design — not standard
  sample-weight fits).
- `nn/prepare_iajd.py`: now carries the `sp_*` block durably (so retrains don't drop it).
- Verified: all edited files parse + import; with the flag off, hooks return `None`
  → `fit(..., sample_weight=None)` ≡ pre-change behavior (so the in-flight `finalize_after_qm.sh`
  is unaffected).

**Eval (`eval/sample_prep_eval.py` → `eval/sample_prep_eval.json`, 247 compounds, LOFO):**
- **Q1 weighting — positive.** RandomForest LOFO improves with `conf` weighting:
  R² 0.072→0.185, MAE 0.630→0.604, Spearman 0.298→0.356, MAE@trusted(conf≥0.85) 0.615→0.575
  (−6.5%); beats the mean floor (R²=−0.090). Ridge stays negative (too linear; ignore). `conf` and
  `conf+var` near-identical (few rows carry replicate spread). Point estimates — confirm with the
  GP+bootstrap variant (`--gp`) before publishing a number.
- **Q2 is_pH52 within ja1c05813 — underpowered** (24 compounds after replicate-grouping; skipped).
- **Q3 is_pH52 vs paper — marginal.** ΔR²≈0.010, 95% boot ≈[0, 0.045] → essentially subsumed by
  `paper`. **Do not add is_pH52 as a production feature** (consistent with the prior).

**Outcome:** shipped the **weighting mechanism** (flag-gated); did **not** add the pH covariate.

**To run a weighted retrain (user-triggered, when the box is free of the QM/MD/finalize load):**
```bash
IAJD_USE_PREP_WEIGHTS=1 .venv/bin/python train_bioact_ensemble.py   # + per_organ / binary / pka_v92
```
NOTE: that overwrites the production bundles, so run it when `finalize_after_qm.sh` is not mid-rebuild.
Leave the flag unset for the normal (unweighted) pipeline.
