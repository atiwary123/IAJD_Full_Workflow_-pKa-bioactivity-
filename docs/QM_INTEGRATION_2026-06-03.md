# QM Descriptor Integration — Decision & Execution (2026-06-03)

Should the 8 xTB QM descriptors be features in (1) the **pKa** model and (2) the **bioact**
(`log10_flux_total`) model? Decided by an 8-agent verification workflow (eval + 3 adversarial
skeptics + synthesis), then executed. Companion: `docs/QM_REGEN_FINDINGS_2026-06-02.md`,
`IAJD_master/datasets/QM_COLUMNS_README.md`.

## TL;DR

| Model | Integrate QM? | Result |
|---|---|---|
| **pKa** | **NO** | QM *worsens* honest LOO MAE (+0.0045); 3/3 skeptics refuted at high confidence; the orthogonality script's "helps" flag was a labeling bug (a partial-correlation gate, not the MAE). pKa left untouched. |
| **bioact** | **YES — by activating QM that was silently dead** | The live bundle had been trained on a **100%-NaN Block D'** (QM importance == 0). Fixed the bug, retrained: QM is now live (importance 0→0.0096; Block D' 0.097). v14 pooled LOO 0.4309→0.4273; **v15 production flux 0.2588→0.2526, better on 5/6 families.** |

---

## The bug that mattered: silent QM-blind Block D' (`sys.path`/ImportError)

The premise "bioact already integrates QM via Block D'" was **false as built**, twice over (the
overnight finalize bundle *and* the first retrain attempt). Both were trained with Block D'
(`X_train[:,92:106]`) **100% NaN**, and `direct_model_full` assigned **exactly 0.0** importance to
every `qm_` column. The model had literally never seen a QM value.

**Root cause (verified, not guessed):** `bioact_v14_pipeline.py` lives in `IAJD_master/code/`.
Run as a script (`python IAJD_master/code/bioact_v14_pipeline.py`), `sys.path[0]` is the script's
own directory — repo root is **not** on the path. So `compute_block_dprime`'s
`from physics_cache_io import load_physics, BLOCK_DPRIME_KEYS` raised `ImportError`, and its
`except ImportError: return np.full(..., np.nan)` branch **silently returned an all-NaN Block D'**.
The QM cache was complete and correct the whole time — it was simply never read, because the module
that reads it failed to import. (The qmmd-features path looked fine because `adaptive_stacker.py`
lives at repo root, so *its* import succeeded — which is why `qmmd_features_v14_train.npy` had real
QM while the trained bundle did not. That mismatch is what exposed the bug.)

This was **not** a timing/cache-finalization artifact (the workflow's initial hypothesis). It was a
launch-context import failure that degraded silently.

**Fix** (`IAJD_master/code/bioact_v14_pipeline.py`):
1. Put repo root on `sys.path` at module load (`_REPO_ROOT = WORK.parent.parent`). This fixes both
   the **script-launch** (training) and the **import-launch** path — `predict_v14_real.py` does
   `from bioact_v14_pipeline import assemble_X`, so the same module-level fix makes **inference**
   resolve `physics_cache_io` too. Verified: the predict smoke test now reports `Block D' QM
   coverage: 100.0% finite (REAL QM live)`.
2. Made the `except ImportError` branch **loud** — a silent all-NaN here trained two QM-blind
   bundles believed to carry QM; it must never pass unnoticed again.
3. Added a `Block D' QM coverage` print to `assemble_X` so every run states coverage explicitly.

A hard gate in `retrain_qm_integration.sh` now aborts before the (expensive) downstream chain if
the QM cols are not live after the pipeline rebuild.

---

## Decision 1 — pKa: DO NOT integrate

- **Dedicated LOO MAE A/B** (`eval/qm_pka_eval.py`, n=262, XGB+LOO, native NaN): structural+molgpka
  Δ=**+0.0045** (0.1478→0.1524); all-numeric+molgpka Δ=**+0.0027**. QM makes pKa MAE slightly WORSE.
- **3/3 adversarial verdicts REFUTED** "QM materially improves pKa LOO MAE" at high confidence.
- The orthogonality script's `qm_helps=True` is a **labeling bug** (`eval/qm_orthogonality_pka.py`
  set it from a `|partial ρ|≥0.4` gate, not the MAE). The gate fires only in tiny family strata
  (GA-Tris n=12, PE-Gallic n=29) with sign-flipping ρ; none survive Bonferroni.
- Shuffled-QM (+0.0115) and random-column (+0.0100) controls are indistinguishable from real QM
  (+0.0073) — small-n overfit, not signal.
- Production pKa **v92** already uses MolGpKa + RDKit and ingests no QM; nothing here justifies the
  8-col block. **pKa code/models unchanged.**

## Decision 2 — bioact: INTEGRATE (executed)

The fair, identical-row A/B (`eval/skeptic_bioact_qm_controls.json`, n=219, all-8-QM-finite) showed
robust lift on a **structural-only baseline**: ΔMAE −0.047, bootstrap CI [−0.075,−0.018], beats
shuffled-QM 100/100, negative in every family, leave-one-family-out −0.042 to −0.059. That motivated
activation. The **production** gain is smaller and honestly reported below.

### Before → after (production stack, only Block-D' QM toggled NaN→real)

| Artifact | BEFORE (QM-blind) | AFTER (real QM) | Δ |
|---|---|---|---|
| v14 direct bundle, pooled LOO MAE | 0.4309 | **0.4273** | −0.0036 |
| v14 pooled R² | 0.587 | 0.594 | +0.007 |
| QM feature-importance sum | **0.0000** | **0.0096** | now used |
| Block D' importance sum | 0.000 | 0.097 | now used |
| **v15 hybrid (production flux), pooled LOO MAE** | **0.2588** | **0.2526** | **−0.0062** |

v15 per-family (before→after): Dialkoxybenzyl 0.177→0.167, G1-Janus 0.340→**0.307**, GA-Tris
0.152→0.172, PE-Gallic 0.293→0.274, PE-Tris 0.222→0.202, sSS 0.275→0.274. **Better on 5/6;** only
GA-Tris (n=20) worse. v14 per-family is mixed (PE-Gallic & sSS better; PE-Tris/GA-Tris/Dialkoxybenzyl
slightly worse) — expected small-n noise from adding 4 features to n=10–41 families.

### Why the production gain (−0.004 to −0.006) is far below the eval's −0.047 — honest read

1. **The eval's baseline was deliberately weak** (42-col structural-only, MAE 0.523). The production
   stack is already strong (0.43) with LiON + ADMET + analog-delta capturing signal that **overlaps**
   QM (desolvation/size). QM's *marginal* contribution on top of an already-good model is small.
2. **Block D' carries only 4 of the 8 QM descriptors** (`qm_q_ionizableN, qm_dipole_D,
   qm_dGsolv_kJmol, qm_homo_lumo_eV`). The fragment-solvation terms (`qm_dGsolv_head/tail`),
   `qm_polarizability`, and `qm_Ehedup` are computed and in the dataset/cache but **not** in the
   model's feature block.
3. The v15 **physics-only head still gets 0 blend weight** (ml=1.0) — Block-D' physics doesn't beat
   the descriptor/LiON/ADMET stack on in-distribution LOO. QM helps **indirectly**: the ML/stacker
   head improved because its underlying v14 features now include real QM.

**Bottom line:** QM is now genuinely integrated (was literally unused), and the production flux
predictor is modestly but consistently better (−0.0062 pooled, 5/6 families). It is *not* the −0.047
headline; that figure was against a weak baseline and with all 8 descriptors.

### Headroom (documented, not done)
Adding the 4 unused QM descriptors — especially the fragment-solvation terms `qm_dGsolv_head/tail`
(the orthogonality analysis flagged tail as the one descriptor with genuine non-redundant content, and
ΔGsolv is the recurring workhorse for both pKa and flux per the findings report) — to Block D' could
recover more of the eval's gain. That changes the 106-feature contract (BLOCK_SLICES, predict path),
so it's a deliberate follow-up, not folded into this plumbing fix.

---

## What was retrained (full downstream chain, `retrain_qm_integration.sh`)
bioact_v14_pipeline (bundle w/ real Block-D' QM) → **[gate: QM live]** → regen_step10_train_arrays →
loo_bioact_components → adaptive_stacker --build-qmmd-block → train_qmmd_head_only →
train_bioact_ensemble → train_per_organ → train_binary_classifier → adaptive_stacker (production) →
train_v15_physics_ml. **pKa side: not touched.**

Validation (`audit_work/regen_validate.py`): ALL CHECKS PASSED — datasets consistent, caches real
(no proxy), all train arrays aligned (247×106; agile/cpp/qmmd/loo aligned), QM real coverage 247/247,
and the **predict_v14_real smoke test confirms inference uses real QM (100% Block-D' coverage)**.

## Caveats
- Within-distribution LOO only (single target `log10_flux_total`); not leave-one-family-out
  generalization to a new chemotype.
- Small-n families (n=10–41) — per-family deltas are directional.
- **MD half of Block D' (8 `md_*` + `dG_escape_helfrich`) remains 100% NaN** (`md_cache.csv` = 4
  rows; MD deliberately not run). This is a *QM* win, not a "physics block" win.
- ~30% of training SMILES are 2026-06 reconstructions; both MAEs are bounded by those structures.

## Artifacts
`eval/bioact_BEFORE_qm_integration.json`, `eval/bioact_AFTER_qm_integration.json`,
`eval/skeptic_bioact_qm_controls.json`, `eval/skeptic_pka_qm_controls.json`,
`eval/qm_orthogonality_pka.py` (line 128 = labeling bug), `retrain_qm_integration.sh`,
`physics_logs/retrain_qm_integration.log`, `IAJD_master/bundles_caches/bioact_v14_bundle.pkl`
(Block-D' QM now finite), `v15_hybrid_bundle.joblib`, `v15_loo_report.json`.
