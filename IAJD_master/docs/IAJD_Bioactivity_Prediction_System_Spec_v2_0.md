# IAJD Bioactivity Prediction System — Technical Specification v2.0
### Optimized Architecture: Three-Stage Cascade with External-Knowledge Injection

**Percec Laboratory · University of Pennsylvania**
**CHAS Research Fellowship — Machine Learning for Ionizable Lipid Design**

**Document version:** v2.0 (Apr 2026)
**Supersedes:** Bioactivity Spec v1.0 (Mar 2026), Bioact v0 Prototype Spec (Apr 2026)
**Companion to:** IAJD pKa Prediction System v4.0 / v7.x

---

## Document Position

This specification merges three prior documents into a single optimized design:

1. **Bioactivity Spec v1.0** — the publication-grade three-stage cascade (organ classifier → magnitude regressor → formulation residual) with the five-level analog hierarchy, conformal prediction intervals, and full ablation/baseline protocol. **Retained nearly verbatim as the architectural backbone of v2.0.**
2. **Hydrophobic Tail Build-Ready Blueprint** — the multi-output GP + MAP4/ECFC4 + composite-Tanimoto-Matérn kernel design, the SAR-grounded counterfactual API, and the explicit mechanistic-unit-test design. **Retained as the v2.0-stretch GP-variant of Stage B and as the source of the counterfactual / Pareto API.**
3. **Bioact v0 Prototype Spec** — the inference-only external-knowledge injection paths (LiON pretrained Chemprop, ADMET-AI, Peterca/Percec geometric model, ApoE corona heuristic) and the no-GPU build path. **Promoted in v2.0 from a "v0 hack" to a first-class feature block (Block B+C+D), making v0 the deployable shipped state of v2.0.**

The unified architecture is what each prior document was reaching toward. v1.0 had the right cascade but no external-knowledge injection; the blueprint had the right GP and counterfactual ambitions but was too compute-heavy for the n we have; v0 had the right pragmatism but was scoped down to the point where its Stage A and Stage B did not match the v1.0 spec's ambitions. v2.0 keeps the v1.0 cascade as the default code path, integrates v0's external blocks as additive feature inputs, and treats the blueprint's GP+ICM as a documented stretch target that activates only when n grows past 350 or when GPU time becomes available.

The deliverable end state is `iajd_bioact_v2.py` and `iajd_bioact_v2_bundle.pkl`, mirroring the existing `iajd_pka_predict.py` API. Two named build targets are defined explicitly:

- **v2.0-MIN** — the no-GPU, single-session shippable subset. ~80 features, three-stage cascade with XGBoost throughout, conformal PIs, Block B+C+D inference-only external injection. Build budget ≤ 3 hours of dialog turns. **This is the v0 spec promoted to first-class status.**
- **v2.0-FULL** — the full publication-grade target. v2.0-MIN + GPU fine-tuned LiON + multi-output GP with rank-2 ICM + counterfactual / Pareto API + CG-MD descriptors. Build budget ~16 weeks (matches v1.0's original timeline).

Every section below specifies behavior for both targets. When v2.0-MIN behavior diverges from v2.0-FULL, the divergence is called out in a `[MIN vs FULL]` block.

---

## Section 0. Executive Summary

The v2.0 IAJD bioactivity system predicts in vivo Luc-mRNA delivery activity (BALB/c, retro-orbital IV, 4 h, 5–10 µg) for any IAJD given its SMILES and optional DNP characterization. The output is a calibrated probability distribution over the five candidate delivery organs (whole-body, spleen, liver, lung, lymph node), a per-organ log10-flux point estimate with a 90% prediction interval, a 3-bin magnitude posterior (low / mid / high), a structured similarity hierarchy level (1–5) explaining how the prediction was made, and a confidence tier (HIGH / MEDIUM / LOW). The model is trained on ~220–280 unique (compound × formulation × time-point) rows extracted from twelve Percec-lab SI tables, intersected with the v21 pKa dataset on IAJD number to inherit ~50 molecular features per compound.

The architecture is the v1.0 three-stage cascade with two structural upgrades:

1. **Four-block feature design.** The 60-feature in-house vector from v1.0 (Block A) is augmented with three additive blocks: Block B (LiON pretrained inference, 14 features capturing ~9,000-LNP latent delivery competence), Block C (ADMET-AI pretrained inference, 10 drug-like distribution endpoints used as a domain-shift null baseline), and Block D (six geometric / corona heuristics: Peterca/Percec dendrimersome radius, Israelachvili packing parameter, ApoE-binding heuristic, surface charge density at pH 5 vs 7.4). Total feature dimensionality is 88 in v2.0-MIN and ~110 in v2.0-FULL (which adds CG-MD descriptors).

2. **Calibrated multi-stage uncertainty.** Stage A outputs are calibrated via isotonic regression on a held-out fold; Stage B produces 80%, 90%, and 95% prediction intervals via MAPIE jackknife+ conformal prediction (replacing v1.0's split-conformal); the five-level similarity hierarchy from v1.0 §5.5 is retained verbatim as a structured-reasoning layer that determines anchor selection and per-level uncertainty multipliers. The three uncertainty signals (conformal PI, hierarchy level, Tanimoto OOD flag) compose by min-tier rule into the final HIGH/MEDIUM/LOW confidence label.

The three-stage cascade itself is unchanged from v1.0:
- **Stage A:** Multi-label one-vs-rest XGBoost classifier over {whole_body_active, spleen_strong, liver_strong, lung_strong, LN_strong, heart_active}, with isotonic calibration.
- **Stage B:** Six per-organ XGBoost regressors with 60/40 blend against analog-anchored delta predictor (v7.1 pKa pattern), wrapped in MAPIE for jackknife+ conformal PIs. GA-Tris routes 100% to analog-delta. **v2.0-FULL also fits a multi-output GP with rank-2 ICM + composite Tanimoto+Matérn kernel as an alternative inference path; both paths emit predictions and the bundle stores both for comparison.**
- **Stage C:** Optional GradientBoostingRegressor predicting per-formulation residual correction in [-0.5, +0.5] log-units when (size, PDI, EE) are all supplied at inference. Skipped when formulation incomplete. **v2.0-MIN ships Stage C disabled by default and re-enables it when ≥75% of training rows have complete formulation triples.**

**Performance targets.** v2.0-MIN: pooled LOO MAE on log10(flux_total) ≤ 0.55, organ-dominance accuracy ≥ 70% on 5-fold stratified CV, conformal 90% PI coverage in [0.85, 0.92]. v2.0-FULL: MAE ≤ 0.45 (matching v1.0's acceptance bar), accuracy ≥ 80%, coverage in [0.88, 0.91]. Stretch (FULL-only): MAE ≤ 0.30 on best families, accuracy ≥ 90%. The noise floor on the underlying experimental measurements is ~0.28 log-units (Percec-corpus replicate animal-to-animal variance), which establishes the irreducible lower bound.

**External-knowledge contributions, ranked by expected per-LOO-MAE improvement on n=200:**

| Source | Expected ΔMAE | Cost | Status in v2.0-MIN |
|---|---|---|---|
| LiON pretrained inference (Block B) | -0.03 to -0.08 | 10 min setup + 30 sec/batch inference | Included |
| Block D geometric / corona features | -0.02 to -0.05 | 50 lines of code | Included |
| ADMET-AI inference (Block C) | 0 to -0.02 | 5 min setup | Included |
| 3D conformer features (D6 from pKa project) | -0.015 to -0.020 | already partially computed in v15 cache | Included where available |
| MAP4 fingerprints | -0.01 to -0.03 | 1–2 hr install fight | **Excluded from MIN** (FULL-only) |
| GPU fine-tune LiON on IAJD data | -0.05 to -0.10 | ~12 GPU-hours | **Excluded from MIN** (FULL-only) |
| MOGP with rank-2 ICM (vs independent XGBs) | -0.02 to -0.05 | ~5 hours of fitting + debug | **Excluded from MIN** (FULL-only) |
| CG-MD-derived membrane descriptors | -0.03 to -0.07 | ~1 month of HPC | **Excluded from MIN** (FULL-only) |

**Total expected v2.0-MIN improvement over v1.0-without-Block-B/C/D:** approximately -0.08 to -0.15 log-MAE, which is the difference between "inadequate for production" (v1.0 without external blocks at this n would land at ~0.55–0.65 MAE pooled) and "deployable with caveats" (v2.0-MIN target ~0.45–0.50 MAE pooled).

---

## Section 1. Design Philosophy

### 1.1 Three principles inherited from v1.0

The v1.0 spec established three design principles that v2.0 retains without modification:

1. **Decomposition over end-to-end.** Organ selectivity and magnitude are governed by different feature subsets (discrete architectural tokens vs continuous hydrophobic descriptors), use different training-set sizes (qualitative labels yield ~280+ rows; numerical fluxes yield ~200 rows), and support different downstream decisions. A single end-to-end model balances these poorly; the three-stage cascade lets each stage use the inductive bias best suited to its sub-problem. v1.0 §5.1 explicates this in detail; the argument is unchanged for v2.0.

2. **Analog-anchored reasoning over black-box regression.** When a query IAJD has near-neighbors in the training set, predicting the nearest neighbor's value plus a small delta is more accurate AND more interpretable than predicting from scratch. The five-level similarity hierarchy formalizes this by routing every query through Levels 1 (exact match) → 2 (SMILES match, formulation different) → 3 (same architecture, different chains) → 4 (same family, one token changed) → 5 (cross-family or OOD). The pKa v7.1 system uses this pattern with a pooled LOO MAE of 0.12 against a per-compound noise floor of 0.05; the bioactivity adaptation extends it to multi-output and adds a formulation-pair matching layer.

3. **Honest uncertainty over point estimates.** Every prediction emits both a point estimate and a 90% prediction interval whose empirical coverage on held-out data must lie in [85%, 92%]. The downstream lab decision ("should we synthesize this IAJD?") is made on the interval, not on the point. A model that reports 90% PIs with empirical coverage of 60% is worse than useless; v2.0 retains v1.0's conformal calibration as a non-negotiable production gate.

### 1.2 Three additions in v2.0

Beyond v1.0, v2.0 adds three principles motivated by the v0 prototyping work and the LNP-survey research:

4. **Pretrained-model-as-feature-extractor.** No external lipid-bioactivity model has been trained on IAJDs, but several public models (LiON, AGILE, COMET, ADMET-AI) have been trained on related chemistry and ship as runnable artifacts. The v2.0 design uses these as feature extractors: each is invoked once per IAJD SMILES at training time, the predictions are cached, and the cached predictions become inputs to v2.0's own cascade. **This injects ~9,000 LNP measurements of latent training signal into v2.0 without any of the cost of fine-tuning, retraining, or domain-adaptation.** The price is that the external predictions are domain-shifted (LiON has no spleen / LN endpoint, ADMET-AI is drug-like-trained), so the external features are auxiliary signals rather than primary predictors. XGBoost handles this gracefully — features that hurt validation performance get near-zero feature importance.

5. **Mechanism-as-prior.** No public model captures the apolipoprotein-corona-mediated tropism that Siegwart's SORT lipids exploit, the Peterca/Percec deterministic dendrimersome geometry, or the Henderson-Hasselbalch pH-dependent surface charge that explains the pKa~6 endosomal-escape window. We can't train these mechanisms from data at n≈200, but we can encode them as hand-built features. Block D in v2.0 is exactly this: six features, each grounded in a specific Percec-lab or LNP-literature mechanism, computed deterministically from SMILES + (optional) DLS. They are not as good as a learned corona model; they are better than nothing; and they are the only place v2.0 gets to inject biophysics that no LNP-trained model has access to.

6. **Two-target build (MIN and FULL).** v1.0 specified a single target (16-week timeline, GPU-trained, MAP4 fingerprints, MOGP-ICM stretch). v2.0 separates this into v2.0-MIN (deployable in a single session with no GPU, ships as the operational system the lab uses today) and v2.0-FULL (the publication-grade target the lab works toward). The MIN target exists because shipping a deployed v0.5 today is worth more than shipping a v1 in 16 weeks; the FULL target exists because the operational v2.0-MIN must continuously be measured against a stretch goal so that incremental improvements have a defensible ceiling.

### 1.3 Decision framework for what's IN vs OUT in MIN vs FULL

| Component | v2.0-MIN | v2.0-FULL | Decision rule |
|---|---|---|---|
| Three-stage cascade (A → B → C) | ✅ | ✅ | Architectural backbone; non-negotiable |
| Five-level similarity hierarchy | ✅ | ✅ | Same |
| XGBoost direct + analog-delta blend (Stage B) | ✅ | ✅ | Same |
| Conformal 90% PIs (MAPIE jackknife+) | ✅ | ✅ | Calibration is non-negotiable |
| Block A (in-house ~50 features) | ✅ | ✅ | The workhorse; reuses existing `features.py` |
| Block B (LiON pretrained inference) | ✅ | ✅ | Highest external ΔMAE per setup hour |
| Block C (ADMET-AI inference) | ✅ | ✅ | Cheap; XGBoost ignores if useless |
| Block D (geometric / corona heuristics) | ✅ | ✅ | Mechanism-as-prior, no public alternative |
| 3D conformer features (D6 cache) | ✅ where available | ✅ | Inherit from existing pKa project work |
| MAP4 fingerprints | ❌ | ✅ | Install friction; defer until env supports it |
| Multi-output GP with rank-2 ICM (Stage B variant) | ❌ | ✅ | ~5 hr of fitting + debug; FULL-only |
| GPU fine-tune LiON on IAJD | ❌ | ✅ | ~12 GPU-hours; FULL-only |
| GPU fine-tune AGILE/COMET/LipidBERT | ❌ | ✅ | Decide after LiON fine-tune evaluated |
| CG-MD-derived membrane descriptors | ❌ | ✅ | ~1 month HPC |
| Counterfactual / Pareto API | ❌ | ✅ | Requires Stage B MAE < 0.35 |
| Bioactivity Periodic Table deliverable | ❌ | ✅ | Requires Stage B sufficiently calibrated |
| TDC submission | ❌ | ✅ | Final publication step |

The MIN/FULL split is a contract: anything in the MIN column must work end-to-end with the existing project files and no GPU; anything in the FULL column requires explicit infrastructure or time commitments and is documented but not built in the v2.0-MIN session.

---

## Section 2. Problem Framing

This section is retained from v1.0 §2 with minor edits. The core argument — that bioactivity is harder than pKa for four orthogonal reasons — is unchanged. Summarized:

**2.1 The Dynamic Range Problem.** pKa spans ~3 units with 0.05-unit measurement noise (S/N ≈ 60:1). Total flux spans 3+ decades with replicate-to-replicate animal noise of ~0.28 log-units (S/N ≈ 11:1). Achievable MAE on bioactivity is therefore intrinsically worse than on pKa.

**2.2 The Multi-Cause Problem.** pKa is a single-mechanism endpoint (acid-base equilibrium of one ionizable amine). Bioactivity is governed by at least seven coupled mechanisms (DNP self-assembly, biodistribution, endosomal uptake, endosomal escape, mRNA release, ribosomal translation, luciferase folding). Any single feature can only partially capture the cause, so the model must combine many features each carrying weak signal.

**2.3 The Reproducibility Ceiling.** IVIS bioluminescence has an animal-to-animal variance of ~0.28 log-units even on identical formulations (Percec-corpus replicate analysis). The IAJD97 stability time-course (Sci Adv 2026) shows batch-to-batch variation of similar magnitude. No model can do substantially better than this floor.

**2.4 The Cell/Mouse Divergence.** In vitro HEK293T luminescence and in vivo IVIS flux correlate poorly (r ≈ 0.3 in the Percec corpus). v2.0 treats in vitro as a secondary endpoint; no in vitro feature is used for in vivo prediction.

**2.5 Realistic Performance Targets.** See Executive Summary; v2.0-MIN aims for log-MAE ≤ 0.55, v2.0-FULL aims for ≤ 0.45 (matching v1.0).

---

## Section 3. Dataset

### 3.1 Source corpus (unchanged from v1.0 §3.1)

Twelve Percec-lab papers + supporting reviews provide the bioactivity corpus. Tier-1 papers (clean per-organ tabular flux data) are ja2c00273, ja1c05813, ja1c09585, bm4c01599. Tier-2 are ja3c13569 (X2 holdout), ja5c07232, ja3c07337. Tier-3 are pharm 15:1572, sciadv.adv1554 (X1 holdout), and the IAJD97 scale-up paper. Inclusion/exclusion criteria are unchanged from v1.0 §3.1.1–3.1.2.

### 3.2 Master schema (extended from v1.0 §3.2)

The v2.0 master schema is a strict superset of the v1.0 schema. New columns added in v2.0:

```
# BLOCK B — LiON-derived (cached at training time, hot at inference)
lion_zflux_liver_IV       : float         # raw LiON output for liver-IV-mRNA tissue
lion_zflux_lung_IT        : float         # lung intratracheal
lion_zflux_lung_inh       : float         # lung inhalation
lion_zflux_lung_neb       : float         # lung nebulization
lion_zflux_muscle_IM      : float         # muscle intramuscular
lion_zflux_nasal          : float         # nasal mucosa
lion_max_zflux            : float         # max over 6 tissues
lion_argmax_tissue        : str           # which tissue won
lion_argmax_OH_<tissue>   : 6 binary cols # one-hot of the argmax
lion_train_max_tanimoto   : float         # max Tanimoto to LiON training set
lion_OOD_flag             : int           # 1 if lion_train_max_tanimoto < 0.30

# BLOCK C — ADMET-AI inference (drug-like null baseline)
admet_VDss_Lombardo       : float
admet_PPBR_AZ             : float
admet_BBB_Martins         : float
admet_Clearance_Hep       : float
admet_HalfLife_Obach      : float
admet_HIA_Hou             : float
admet_Caco2_Wang          : float
admet_Pgp_Broccatelli     : float
admet_Solubility_AqSolDB  : float
admet_Lipophilicity_AZ    : float

# BLOCK D — Geometric / corona (mechanism-as-prior)
peterca_radius_proxy_nm   : float         # JACS 2011 dendrimersome model
packing_parameter_CPP     : float         # Israelachvili
apoE_binding_heuristic    : float         # PC-DB-motivated Gaussian product
charge_density_pH7        : float         # H-H equation × headgroup area
charge_density_pH5        : float         # H-H equation at endosomal pH
curvature_proxy_inv_nm    : float         # 1 / peterca_radius_proxy_nm
```

All v1.0 columns (compound identification, inherited v21 features, hydrophobic-part features, formulation features, target columns, censored flags) are retained verbatim. Total v2.0 column count: ~135.

### 3.3 Cleaning tracks B1–B7 (unchanged from v1.0 §3.4)

Track sequence is identical to v1.0. v2.0 adds two checks during B6 (feature computation):
- **B6.1**: After computing Block B, verify that `lion_max_zflux` has variance > 0.5 across the dataset; a flat-output indicates LiON has effectively collapsed and Block B should be flagged for downweighting.
- **B6.2**: After computing Block C, verify that `admet_PPBR_AZ` predictions span > 30 percentage points across the dataset; a clipped-to-near-100% output indicates ADMET-AI has saturated on the IAJDs (likely due to extreme MolLogP) and Block C is partially uninformative.

### 3.4 Train/holdout split (unchanged from v1.0 §6.1)

X1 (IAJD97 stability), X2 (14 constitutional isomers from ja3c13569), X3 (10% stratified random) are reserved before any training. The 90% working set is used for 10-fold family-stratified CV (primary) and LOO (for analog-delta lookup table construction).

---

## Section 4. Feature Engineering

### 4.1 Block A — In-house features (~50 features) [unchanged from v1.0 §4.1–4.3]

The 34 inherited features from v21 (RDKit descriptors, electronic features, architectural tokens, 3D conformer descriptors), the 8 new hydrophobic-part features (chain_ratio, parity, asymmetry, branched_count, total_carbons, bilayer_thickness, total_strength, interdigitation flag), and the 8 pKa-prediction features (pKa_v7_pred, PI bounds, OOD flag, tier one-hot) are retained verbatim from v1.0. The implementation is the existing `features.py` — `compute_bioact_features()` returns a 62-dim vector that includes Block A plus the 12-dim formulation block.

**v2.0 changes to Block A:** None. The existing pKa-pipeline-inherited code is correct and is reused without modification.

### 4.2 Block B — LiON pretrained inference (14 features) [new in v2.0]

The Witten et al. LiON model (Anderson lab, *Nat. Biotechnol.* 43:1790, 2025; MIT-licensed at github.com/jswitten/LNP_ML) is a Chemprop D-MPNN trained on ~9,000 LNP activity measurements aggregated from 20 prior screens, covering liver-IV, lung-IT, lung-inhaled, lung-nebulized, muscle-IM, and nasal mucosa. It accepts ionizable-lipid SMILES + auxiliary metadata (cargo, target tissue, formulation ratios) and outputs a z-scored log-flux per (lipid, tissue) tuple.

**Why LiON is the right pretrained model to inject.** Of the four candidate LNP-bioactivity models in the open-source ecosystem (LiON, AGILE, COMET, LUMI-lab), LiON has (i) the largest and most diverse training corpus, (ii) the cleanest standalone inference path via the chemprop CLI, (iii) released checkpoints under a permissive license, and (iv) a multi-tissue output that matches v2.0's per-organ design even though no LiON tissue overlaps the IAJD-relevant {spleen, LN}. AGILE is single-readout per cell line and would require fine-tuning to be useful; COMET requires a full multi-component formulation as input which is not the IAJD form factor; LUMI-lab is single-task on lung HBE potency.

**The "wrong organ" caveat (must appear on the model card).** LiON has no spleen or LN endpoint. The five LiON tissues are all "wrong organs" for the IAJD use case. v2.0 uses LiON anyway because the predictions are correlated with each other and with general lipid-delivery competence — that latent "delivery competence" signal is real and structurally informative even when the absolute organ specificity is wrong. Specifically: high `lion_zflux_liver_IV` likely indicates *any* successful endosomal escape and protein expression, and we expect that to correlate with high IAJD `flux_total` even when the IAJD's actual dominant organ is spleen.

**Setup procedure.** One-time, ~10 minutes:

```bash
git clone https://github.com/jswitten/LNP_ML.git /home/claude/LNP_ML
pip install chemprop==1.7.0 --break-system-packages
# Download pretrained checkpoints from the Zenodo/Figshare DOI in repo README
# v2.0 needs: liver-IV-mRNA, lung-IT-mRNA, lung-inh-mRNA, lung-neb-mRNA, muscle-IM-mRNA, nasal-mRNA
```

**Inference and caching.**

```python
LION_TISSUES = ['liver_IV', 'lung_IT', 'lung_inh', 'lion_neb', 'muscle_IM', 'nasal']

def lion_features_batch(smiles_list, ckpt_dir='/home/claude/LNP_ML/checkpoints'):
    """Run LiON inference once per tissue; return dict of arrays."""
    out = {}
    for tissue in LION_TISSUES:
        # Write smiles to a chemprop-format CSV, invoke `chemprop_predict`, parse output
        out[f'lion_zflux_{tissue}'] = _run_chemprop_predict(smiles_list, f'{ckpt_dir}/{tissue}')
    return out

# Cache to /home/claude/lion_cache.parquet keyed by canonical SMILES.
# At training time: invoke once on all unique SMILES, ~30 sec/100 compounds on CPU.
# At inference time: lookup from cache; if SMILES is new, invoke LiON for just that one.
```

**Derived features.** From the 6 raw `lion_zflux_<tissue>` outputs:
- `lion_max_zflux`: max over tissues. "Is this lipid predicted to be active anywhere LiON has seen?"
- `lion_argmax_tissue` one-hot: 6 binary cols indicating which tissue won.
- `lion_train_max_tanimoto`: max Tanimoto similarity of this IAJD's Morgan FP to LiON's training-set FPs (LiON ships training SMILES as a CSV in the repo; v2.0 loads them once at bundle build time).
- `lion_OOD_flag`: `1` if `lion_train_max_tanimoto < 0.30`, else `0`. Conservative threshold because IAJD scaffolds are essentially absent from LiON's training space.

[MIN vs FULL] In v2.0-MIN, LiON is used inference-only: weights are frozen and never updated. In v2.0-FULL, the LiON checkpoints are fine-tuned on the IAJD bioactivity dataset for ~3 epochs on a single GPU (~12 GPU-hours), with a small learning rate (1e-5) to preserve the pretrained representation. The fine-tuned model's per-tissue predictions replace the frozen-weight predictions in Block B. Expected improvement: -0.05 to -0.10 log-MAE on Stage B.

### 4.3 Block C — ADMET-AI pretrained inference (10 features) [new in v2.0]

ADMET-AI (Stanford; github.com/swansonk14/admet_ai; Bioinformatics 2024) is a Chemprop+RDKit GNN ensemble trained on 41 TDC ADMET datasets, all small-molecule drug-like. It is included in v2.0 as a **null baseline / domain-shift indicator**: tree models can use Block C if it helps, ignore it if it doesn't, and the question of whether IAJDs benefit from drug-like distribution endpoints becomes empirical rather than dogmatic.

The ten endpoints retained in Block C are the most distribution-relevant of the 41 (VDss, PPB, BBB, hepatic clearance, half-life, HIA, Caco-2 permeability, Pgp substrate, aqueous solubility, lipophilicity). The full 41-endpoint vector is recomputed at training time and cached; only the 10 most useful are exposed to the model, but the full cache is preserved for ablation studies and for v2.0-FULL extensions.

**Why we keep these features even though IAJDs are far OOD for ADMET-AI.** Three reasons. First, tree ensembles can downweight features freely; the cost of including them is near zero. Second, the small head-group fragment + ionizable amine + linker portion of the IAJD does overlap drug-like chemical space, so the ADMET predictions are not totally meaningless for that subspace. Third, PPB in particular may correlate with apolipoprotein-corona competency in a directionally useful way — proteins that bind drug-like molecules and proteins that bind LNP surfaces share thermodynamic features (hydrophobic patches, surface charge complementarity). This is speculative and is exactly what the ablation study (ABL-16, see §6.3) will test.

**Setup and inference:**

```bash
pip install admet-ai --break-system-packages
```

```python
from admet_ai import ADMETModel
admet = ADMETModel()  # loads all 41 pretrained checkpoints from package
df_admet = admet.predict(smiles=df['SMILES'].tolist())
# ~50 ms per compound on CPU, ~10 sec for n=200 batch
```

Cached to `/home/claude/admet_cache.parquet` keyed by canonical SMILES.

### 4.4 Block D — Geometric and corona heuristics (6 features) [new in v2.0]

These are mechanism-as-prior features: hand-built, deterministic, computed from SMILES + (optional) DLS. They encode physics that no public model has access to.

**D1. `peterca_radius_proxy_nm`** — From the Peterca/Percec deterministic geometric model (JACS 2011 133:20507) for predicting dendrimersome size from lamellar bulk structure:

```python
def peterca_radius_proxy(chain_min, chain_max, linker_length):
    # Linker contributes ~0.13 nm/atom (extended); chain ~0.125 nm/C
    lamellar_d = 0.13 * linker_length + 0.125 * (chain_min + chain_max) / 2
    return 30.0 + 5.0 * lamellar_d  # nm; empirical scaling from Percec corpus
```

When measured `DNP_size_nm` is available, it replaces the proxy and a flag is set.

**D2. `packing_parameter_CPP`** — Israelachvili critical packing parameter:

```python
def packing_parameter(chain_min, chain_max, head_group):
    v = (chain_min + chain_max) * 27.4         # Å³, vol per CH2
    l = 0.125 * 10 * (chain_min + chain_max) / 2  # Å, extended chain length
    a_lookup = {'HPRZ': 47, 'MPRZ': 35, 'DMA': 25, 'PIP': 30, 'DMBA': 55}
    a = a_lookup.get(head_group, 40)
    return v / (a * l)
```

CPP < 1/3 → micelles; 1/3–1/2 → cylinders; 1/2–1 → bilayers/vesicles (the IAJD regime); >1 → inverse phases. Deviations from 0.5–1.0 flag OOD assembly behavior.

**D3. `apoE_binding_heuristic`** — From PC-DB observations that ApoE/ApoB-100 corona binding correlates with sub-100 nm size, near-neutral zeta at pH 7.4, intermediate hydrophobicity, and protonation state at serum pH. A Gaussian-product score:

```python
def apoE_heuristic(size_nm, zeta_mV, MolLogP, pKa_pred):
    size_term = np.exp(-((size_nm - 100) / 50)**2)
    zeta_term = np.exp(-(zeta_mV / 10)**2)
    logp_term = np.exp(-((MolLogP - 11) / 4)**2)
    pH_term   = 1.0 / (1.0 + np.exp(-(pKa_pred - 5.5) * 2))
    return size_term * zeta_term * logp_term * pH_term
```

Higher score → higher predicted ApoE recruitment → higher liver tropism expected. v2.0 uses this as a continuous feature; v2.0-FULL replaces it with a learned ApoE-binding model trained on PC-DB + Ban 2020 + lab corona data.

**D4 + D5. `charge_density_pH7` and `charge_density_pH5`** — Henderson-Hasselbalch protonation fraction × inverse headgroup area:

```python
def charge_density(pKa, pH, head_group):
    f_protonated = 1.0 / (1.0 + 10**(pH - pKa))
    a = a_lookup[head_group]  # Å²
    return f_protonated / a
```

The pH 5 vs 7.4 differential is the mechanistic basis of the ~6.0–6.5 pKa optimum window: at pH 7.4 the IAJD is mostly neutral (low protein-corona binding, long circulation), at pH 5 it is mostly protonated (membrane disruption, endosomal escape). Block D explicitly exposes both endpoints.

**D6. `curvature_proxy_inv_nm`** — `1 / peterca_radius_proxy_nm`. Membrane curvature correlates with cellular uptake mechanism: high curvature → caveolar / clathrin-independent; low curvature → clathrin-mediated.

[MIN vs FULL] In v2.0-MIN, Block D is the six features above. In v2.0-FULL, Block D extends with CG-MD-derived descriptors (MARTINI 3 simulations of dendrimersome self-assembly): bilayer thickness from atom densities, area-per-lipid, S_CD order parameters, cohesive energy density, and tilt angle. These add ~5 features and are estimated to improve MAE by -0.03 to -0.07 but require ~1 month of HPC time per dataset rebuild. They're documented here for FULL but are not in the MIN build path.

### 4.5 Full v2.0 feature vector (88 features in MIN, ~110 in FULL)

```
[Block A: 50] | [Block B: 14] | [Block C: 10] | [Block D: 6] | [Formulation: 8] = 88
```

In v2.0-FULL: + MAP4 fingerprints (concatenated as separate channel for kernel methods, not engineered features) + CG-MD descriptors (~5 added to Block D) + fine-tuned LiON predictions (replacing frozen Block B) → ~110 total.

**Stage A feature subset (25 features):** All of Block D (6), head-group one-hot from A (8), core one-hot from A (4), linkage one-hot from A (3), pKa_v7_pred + PI width + OOD flag from A (3), the 6 LiON zfluxes from B, and 1 lion_max_zflux. Total = 31 (revised up from v1.0's 25 to incorporate Block D and Block B).

**Stage B feature subset:** Full 88.

**Stage C feature subset (10 features):** DNP_size_nm + log(DNP_size_nm) + DNP_PDI + DNP_PDI_high_flag + DNP_EE_pct + DNP_zeta_mV + buffer_pH + dose_mRNA_ug + family_id one-hot (5) → 12. **v2.0-MIN: Stage C disabled by default.**

---

## Section 5. Modeling Architecture — The Three-Stage Cascade

The cascade is unchanged from v1.0 §5: Stage A → Stage B → Stage C, executed in strict serial order at inference, with Stage C optional and conditional on formulation completeness. v2.0 retains v1.0's rationale (§5.1) for decomposition over end-to-end and adds two implementation upgrades: (i) the GP-variant of Stage B is promoted from "documented but not built" to "built in v2.0-FULL, retained as fallback in v2.0-MIN," and (ii) the conformal prediction layer uses MAPIE jackknife+ instead of v1.0's split conformal, which improves coverage stability at small n.

### 5.1 Rationale for decomposition (unchanged from v1.0 §5.1)

The four arguments for the three-stage decomposition are preserved verbatim from v1.0:
1. Discrete features dominate organ selectivity, continuous features dominate magnitude.
2. Qualitative organ labels are available for ~280+ rows; numerical fluxes only for ~200.
3. Downstream decisions weight Stage A and Stage B differently.
4. Formulation data is sparse (~50% coverage), so a separate stage that can be skipped is preferable to a unified model that imputes missing values.

### 5.2 Stage A — Organ-selectivity classifier

**Targets (multi-label, 6 outputs):** `whole_body_active`, `spleen_strong`, `liver_strong`, `lung_strong`, `LN_strong`, `heart_active`. Each is binary, derived per v1.0 §5.2.1: 1 if paper reports "strong" OR `flux_<organ>_log10 ≥ 7.0`, else 0. The whole_body_active label is `OR` over the others. Targets are not mutually exclusive.

**Model class:** Six independent XGBoost binary classifiers (one-vs-rest) on the 31-feature Stage A subset, with isotonic calibration on a held-out fold.

**Hyperparameters (fixed, no tuning in MIN):**
```python
STAGE_A_HP = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.1,
    objective='binary:logistic', eval_metric='logloss',
    early_stopping_rounds=30, random_state=42,
)
```

[MIN vs FULL] v2.0-MIN uses these fixed HPs across all six classifiers. v2.0-FULL runs a 30-iteration random search over `{n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree}` per organ on 10-fold CV.

**Calibration.** Each per-organ XGBoost output is wrapped in `CalibratedClassifierCV(method='isotonic', cv='prefit')` using the X3 holdout fold. The calibration step is non-negotiable — uncalibrated XGBoost probabilities are systematically pushed toward 0/1, with raw 0.95 corresponding to ~70% empirical TPR.

**Decision thresholds.** The "dominant organ" call is computed per v1.0 §5.2.4: argmax over `{spleen_strong, liver_strong, lung_strong}` of calibrated probabilities, with overrides:
- If `max(p_spleen, p_liver, p_lung) < 0.4`, output `organ_dominant = 'unclear'` and emit warning.
- If top-2 probabilities differ by < 0.10, output `organ_dominant = 'tied'` with the second organ's name in `organ_dominant_tied_with`.

LN and heart are reported only as multi-label probabilities, never as the dominant call.

### 5.3 Stage B — Magnitude regressor

**Targets:** Six log10-flux regression outputs: total, spleen, liver, lung, LN, heart. Masked loss handles missing per-organ values cleanly via XGBoost's `sample_weight=0` for masked rows.

**Two implementation paths run in parallel:**

#### 5.3.1 Path B-XGB — Family-stratified XGBoost + analog blend [v1.0 default; v2.0-MIN default]

For non-GA-Tris compounds, Stage B output is a 60/40 blend (`ALPHA_BLEND_BIOACT = 0.60`) of:

- **Direct XGBoost** on the full 88-dim feature vector. Per-organ regressor with hyperparameter search:
  ```python
  STAGE_B_HP_SEARCH = dict(
      n_estimators=[200, 400, 600],
      max_depth=[3, 4, 5],
      learning_rate=[0.03, 0.05, 0.08],
      min_child_weight=[3, 5, 8],
      subsample=[0.8, 1.0],
      colsample_bytree=[0.7, 0.85, 1.0],
      reg_lambda=[1.0, 3.0],
  )
  # Random search, 30 iterations, MAE on 10-fold CV
  ```
- **Analog-anchored delta** mirroring v7.1 pKa: K=8 nearest neighbors by Tanimoto on Morgan-2-2048, similarity-weighted (w_k = sim²) anchor flux + delta-model-predicted residual. Delta model is a small XGBoost trained on all LOO pairs.

For GA-Tris compounds (n=11 in v21), the direct XGBoost is skipped (insufficient training data) and the blend goes 100% analog. This exactly mirrors v7.1's GA-Tris routing and is the architectural fix for the small-family problem.

[MIN vs FULL] v2.0-MIN uses fixed `STAGE_B_HP` per organ chosen as the median of the v1.0 search ranges (`n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=3, subsample=0.8, colsample_bytree=0.85, reg_lambda=1.0`); no per-organ hyperparameter tuning. v2.0-FULL runs the full 30-iter search per organ.

#### 5.3.2 Path B-GP — Multi-output GP with rank-2 ICM [blueprint default; v2.0-FULL only]

The blueprint's recommended primary architecture, retained verbatim as the v2.0-FULL Stage B variant. Composite kernel:

```python
# 50% Matérn-3/2 ARD on standardized engineered features
# 50% Tanimoto on Morgan-2-2048 fingerprints
# + WhiteKernel(noise_level=0.09) for replicate noise
# + IndexKernel rank-2 over 6 organ outputs → ICM coregionalization
```

GPyTorch implementation. ICM rank-2 means the 6 organ outputs are modeled as linear combinations of 2 latent task functions plus per-organ residuals — captures the correlation structure (compounds that deliver well to spleen tend to also deliver well to LN; lung-targeted compounds are usually liver-spared) without modeling all 6×6 covariances independently.

[MIN vs FULL] In v2.0-MIN, Path B-GP is **not built**. The GP-variant adds ~5 hours of fitting + debugging time and modest expected ΔMAE (-0.02 to -0.05); not worth the budget at MIN. In v2.0-FULL, Path B-XGB and Path B-GP are both fit, both stored in the bundle, and the inference function emits both predictions side-by-side. The default `predict_bioactivity` returns Path B-XGB; `predict_bioactivity_gp` returns Path B-GP. This dual-path design is the v2.0 answer to "should we use trees or GPs?" — ship both and let the empirical result decide.

#### 5.3.3 Conformal prediction intervals (MAPIE jackknife+)

v1.0 specified split-conformal on the X3 fold. v2.0 upgrades to MAPIE jackknife+ (cross-conformal+) on the working set, which gives tighter intervals at small n with the same coverage guarantee.

```python
from mapie.regression import MapieRegressor
mapie = MapieRegressor(estimator=xgb_per_organ, method='plus', cv=10)
mapie.fit(X_train, y_train)
preds, intervals = mapie.predict(X_test, alpha=[0.20, 0.10, 0.05])
# returns 80%, 90%, 95% PIs
```

For Path B-GP, the GP posterior already produces calibrated PIs natively, but they are still passed through MAPIE for conformal recalibration (Romano et al. 2019 "Conformalized Quantile Regression" pattern). This is empirically more robust than relying on the GP posterior alone, especially in regions of low training density.

#### 5.3.4 Three-bin posterior over total flux

Computed for downstream usability. Bins are LOW (`<7.0`), MID (`7.0–8.0`), HIGH (`≥8.0`). Probabilities derived from a Gaussian assumption: `sigma = (PI90_hi - PI90_lo) / (2 × 1.645)`, then `P(low) = Φ((7.0 - μ) / σ)`, etc. The 3-bin output maps directly to the lab decision: "weak / typical / elite" delivery agent.

### 5.4 Stage C — Formulation-aware residual correction

**[v2.0-MIN: disabled by default. Re-enable when ≥75% of training rows have complete formulation triples.]**

When all of `(DNP_size_nm, DNP_PDI, DNP_EE_pct)` are supplied at inference, Stage C produces an additive correction in `[-0.5, +0.5]` log-units to the Stage B output. v1.0 §5.4 spec is retained verbatim:

- **Training target:** Stage B residuals (LOO-computed) = `y_true − y_stage_B_LOO`.
- **Model:** sklearn `GradientBoostingRegressor`, shallow (`max_depth=2`), highly regularized, Huber loss (robust to residual outliers).
- **Features (12-dim):** `(DNP_size_nm, log(DNP_size_nm), DNP_PDI, DNP_PDI_high_flag, DNP_EE_pct, DNP_zeta_mV, buffer_pH, is_acetate, dose_mRNA_ug, family_id one-hot × 5)`.
- **Safety rail:** Output passed through `tanh × 0.5` to clamp into `[-0.5, +0.5]`.

When formulation is missing or incomplete, Stage C is bypassed entirely and Stage B output is returned unchanged.

[MIN vs FULL] In v2.0-MIN, Stage C is implemented but **trained only if** the available training set has ≥75% formulation completeness. Below that threshold (the current expected state), Stage C training is skipped, the bundle includes a `stage_c_trained=False` flag, and `predict_bioactivity` warns the user that formulation correction is unavailable in this bundle. In v2.0-FULL, Stage C is always trained (the lab fills in missing DLS via re-measurement before the v2.0-FULL build).

### 5.5 Five-level analog hierarchy [unchanged from v1.0 §5.5]

The five-level similarity hierarchy is retained verbatim from v1.0. Every prediction passes through the hierarchy in fixed order; the first level whose activation criterion fires is selected:

| Level | Activation | Anchor | Prediction | σ_total |
|---|---|---|---|---|
| 1 | SMILES exact match + formulation match (±15%, ±0.05, ±5%) | matched training compound | training log-flux directly | max(σ_meas, 0.28) |
| 2 | SMILES match, formulation different | formulation-similarity weighted mean | weighted mean of training fluxes | 0.35 |
| 3 | Same arch_coarse, different chain pair | arch_mean | 2D Manhattan-kNN interpolation across chain-pair grid, per organ | family-conditional |
| 4 | Same family, one architectural token changed | nearest analog | analog flux + per-token delta from delta table | RSS(σ_B, δ_SD) |
| 5 | Cross-family or max Tanimoto < 0.50 | none | Stage B output unchanged | family-worst-case PI doubled |

The bioactivity delta table at Level 4 is multi-output (one delta per organ flux target). A delta entry is `RELIABLE` only when `n_pairs ≥ 5 AND per-pair_delta_SD ≤ 0.40`. Unreliable deltas are still used at inference but with RSS-inflated uncertainty.

[MIN vs FULL] Same hierarchy in both targets. The hierarchy is not a model; it's a structured-reasoning layer that determines anchor selection and PI inflation. It runs before Stage B in the inference pipeline.

### 5.6 Uncertainty quantification and confidence tier composition

v2.0 produces three independent uncertainty signals per prediction, composed by min-tier rule:

1. **Conformal PI half-width** (Stage B output) → tier from absolute width: HIGH if `PI90_half_width ≤ 0.45`, MEDIUM if `≤ 0.65`, LOW otherwise.
2. **Hierarchy level** (§5.5) → tier from level: HIGH if Level 1–2, MEDIUM if Level 3–4, LOW if Level 5.
3. **Tanimoto OOD flags**: in-house max-Tanimoto-to-training (HIGH ≥ 0.75 / MEDIUM ≥ 0.50 / LOW < 0.50), AND `lion_OOD_flag` from Block B (LOW if flag=1).

The reported `confidence_tier` is the **minimum** of the three. This is conservative by design: a prediction needs to look good on all three independent axes to earn HIGH confidence. The min-tier rule is the same composition pattern used in v7.1 pKa.

### 5.7 Inference path (full pipeline)

For a query SMILES + (optional) formulation:

```
1. Canonicalize SMILES (RDKit). Compute Morgan FP.
2. Family classification → assign family, arch_coarse, head_group, linkage tokens.
3. Compute Block A features (Block A reuses features.py from project).
4. Lookup or compute Block B (LiON inference, cached or live).
5. Lookup or compute Block C (ADMET-AI inference, cached or live).
6. Compute Block D (Peterca proxy, CPP, ApoE heuristic, charge densities, curvature).
7. Compute pKa via predict_pka_v71(); add to Block A.
8. Five-level hierarchy → determine activation level + anchor.
9. Stage A → calibrated probs per organ, dominant organ call.
10. Stage B → per-organ log-flux + 80/90/95% PIs (Path B-XGB; FULL also runs Path B-GP).
11. If formulation supplied AND Stage C trained: Stage C residual correction.
12. Compose confidence tier from {PI width, hierarchy level, Tanimoto OOD}.
13. Return dict per Section 7 schema.
```

Inference latency target: < 2 seconds single-threaded CPU for all 13 steps. v2.0-MIN typically lands around 0.5–1 sec because Block B/C are cache lookups for known training compounds and only ~0.2 sec for Block D and Stage A/B XGBoost prediction.

---

## Section 6. Training and Validation Protocol

The training protocol is a nested CV design unchanged from v1.0 §6:

- **Outer:** X3 (10% stratified random) — touched twice across project lifetime (conformal calibration + final test reporting).
- **Middle:** 10-fold family-stratified CV on the 90% working set — used for hyperparameter selection and headline pooled metrics.
- **Inner:** LOO on the 90% working set — used for analog-delta lookup table construction, conformal nonconformity scores, and per-family PI calibration.

X1 (IAJD97 stability, 6–8 rows) and X2 (14 constitutional isomers) are mechanistic holdouts touched once at the end.

### 6.1 Critical LOO-CV refit requirements

The v4 pKa spec §9 specifies five objects that must be refit per LOO fold. v2.0 inherits and extends this to seven:

1. **arch_mean tables** — per arch_coarse group.
2. **Chain-pair lookup tables** — 2D grid per arch_coarse.
3. **Delta coefficient tables** — every pair containing the held-out compound is removed; n_pairs < 3 reflags the entry as UNRELIABLE.
4. **Stage A XGBoost classifiers (×6)** — refit per fold.
5. **Stage A isotonic calibrators (×6)** — refit per fold on the inner-held-out set.
6. **Stage B XGBoost regressors (×6)** — refit per fold.
7. **Stage B MAPIE jackknife+ wrappers (×6)** — refit per fold (MAPIE handles this internally with `cv='prefit'` or `cv=k`).
8. **[v2.0 new]** **Block B/C/D feature scalers and any feature-selection masks** — refit per fold to prevent test-set leakage through normalization statistics.

The Block B (LiON) and Block C (ADMET) caches are NOT refit per fold because they are inference-only outputs of frozen pretrained models; their values for any given SMILES are the same regardless of which training fold the SMILES belongs to. This is a real efficiency advantage of inference-only external blocks.

### 6.2 Metrics and acceptance criteria

The full metric panel from v1.0 §6.2 is retained:

| Metric | Scope | Notes |
|---|---|---|
| MAE | per-target per-family + pooled | log10 units |
| RMSE | per-target per-family + pooled | log10 units |
| R² | per-target per-family + pooled | |
| Bias | per-target per-family + pooled | must be near zero |
| Empirical 90% PI coverage | per-target per-family + pooled | target [85%, 92%] |
| 90% PI mean half-width | per-target per-family + pooled | log10 units |
| 3-bin accuracy | total + per-organ | low / mid / high |
| 3-bin macro-F1 | total + per-organ | class-balanced |
| Organ dominance accuracy | Stage A | argmax {spleen, liver, lung} |
| Multi-label Hamming F1 | Stage A | averaged over organs |
| X1 stability variance | Stage B + Stage C | within-compound across formulation |
| X2 isomer rank-order | Stage A + B combined | constitutional isomer test |

#### 6.2.1 Acceptance criteria — split by build target

**v2.0-MIN acceptance criteria (production deployment to lab users):**
1. `iajd_bioact_v2.py` imports cleanly; bundle pickle loads in fresh Python.
2. Stage A organ dominance accuracy on 5-fold stratified CV ≥ 70%.
3. Stage A multi-label Hamming F1 on 5-fold stratified CV ≥ 0.60.
4. Stage B pooled MAE on log10(flux_total) on 5-fold stratified CV ≤ 0.55.
5. Stage B 90% PI empirical coverage on X3 in [0.85, 0.92].
6. X1 stability: predicted total-flux variance across IAJD97 rows < 0.20 log-units.
7. Inference latency < 2 sec single-threaded CPU.

**v2.0-FULL acceptance criteria (publication-grade, matching v1.0):**
1. All MIN criteria hold.
2. Stage A organ dominance accuracy on X3 held-out ≥ 80%.
3. Stage A multi-label Hamming F1 on X3 ≥ 0.65.
4. Stage B pooled LOO MAE on log10(flux_total) ≤ 0.45.
5. Stage B pooled LOO MAE on log10(flux_spleen) ≤ 0.50.
6. X2 rank-order: ≥ 9 of 14 constitutional isomers correctly ranked.
7. X1 with Stage C: predicted variance within 0.15 log-units of observed.
8. No family-level PI coverage below 80%.
9. Pooled signed bias |MAE| ≤ 0.08.

**Stretch (v2.0-FULL stretch, unchanged from v1.0 stretch):**
- Stage B pooled LOO MAE ≤ 0.30.
- Organ dominance accuracy ≥ 90%.
- Per-organ Hamming F1 ≥ 0.82.

### 6.3 Ablation study plan

v1.0 specified 15 ablations (ABL-1 through ABL-15). v2.0 adds five external-block ablations:

| ID | Description | Hypothesis |
|---|---|---|
| ABL-16 | Remove Block B (LiON pretrained) | LiON adds ≥ 0.03 ΔMAE on Stage B |
| ABL-17 | Remove Block C (ADMET-AI) | ADMET adds 0–0.02 ΔMAE; may be drop-in-zero |
| ABL-18 | Remove Block D (geometric / corona) | Block D adds ≥ 0.02 ΔMAE; ApoE heuristic carries most signal |
| ABL-19 | Remove only `apoE_binding_heuristic` from D | Test whether ApoE heuristic alone explains Block D's contribution |
| ABL-20 | Replace LiON with random Gaussian features (same shape) | Tests whether LiON's information is real vs whether XGBoost likes any extra columns |

ABL-16 through ABL-20 are run for v2.0-MIN. ABL-21 (v2.0-FULL only) tests whether GPU-fine-tuned LiON improves over frozen LiON.

The original 15 ablations from v1.0 are retained in their entirety:
- ABL-1: Remove predicted pKa features.
- ABL-2: Replace XGBoost with Ridge regression.
- ABL-3: Remove analog-delta blend (direct XGBoost only).
- ABL-4: Remove direct XGBoost (analog-delta only).
- ABL-5: Remove Stage A; use unified regressor.
- ABL-6: Remove Stage C.
- ABL-7: Replace hydrophobic-new features with `chain_diff` only.
- ABL-8: Remove 3D descriptors.
- ABL-9: Family-pooled training (ignore family labels).
- ABL-10: Replace XGBoost with Random Forest.
- ABL-11: K=3 instead of K=8 in analog-delta.
- ABL-12: 40/60 or 50/50 instead of 60/40 blend.
- ABL-13: Natural log target.
- ABL-14: Drop formulation features.
- ABL-15: GP-variant of Stage B (Path B-GP).

Each ablation reports the full §6.2 metric panel and a signed delta vs full v2.0.

### 6.4 Baseline models for reference

The four baselines from v1.0 §6.4 are retained:

- **BL-1 Family-mean baseline:** predict log-flux as the family mean per organ.
- **BL-2 Single-feature baselines:** XGBoost with only `(pKa_v7_pred, chain_sum, chain_diff, head_group_one_hot)` — total 12 features.
- **BL-3 Random-forest pooled:** RF on the full 88-feature vector with no decomposition (compare against full cascade to justify §5.1).
- **BL-4 LiON-only baseline:** Use Block B alone as the prediction (no Stage A, no in-house features). Tests whether IAJD bioactivity can be predicted from external models alone (the answer should be "partially, but not well enough").

v2.0 adds:
- **BL-5 ADMET-AI-only:** Same as BL-4 but with Block C alone. Confirms IAJDs are too OOD for drug-like ADMET predictions to substitute for in-house features.
- **BL-6 Block-D-only:** Predict from the six geometric/corona heuristics alone. Tests how much signal mechanism-as-prior carries vs the rest.

---

## Section 7. Deployment and Prediction API

### 7.1 `predict_bioactivity()` function specification

```python
def predict_bioactivity(
    smiles: str,
    formulation: dict | None = None,         # {size_nm, PDI, EE_pct, zeta_mV, buffer_pH, dose_ug}
    return_diagnostics: bool = False,
    return_neighbors: int = 5,
    use_gp_variant: bool = False,            # v2.0-FULL only; False in MIN
) -> dict:
    """
    Returns the v2.0 bioactivity prediction dict.
    """
```

### 7.2 Output schema (v2.0)

The return dict is a strict superset of v1.0's schema. New keys in v2.0 marked [v2.0]:

```python
{
    # IDENTIFIER
    'smiles', 'canonical_smiles', 'version': 'v2.0-MIN' or 'v2.0-FULL',
    'family_assigned', 'arch_coarse_assigned',
    
    # CONFIDENCE COMPOSITION [v2.0]
    'confidence_tier': 'HIGH' | 'MEDIUM' | 'LOW',
    'confidence_components': {
        'pi_width_tier': str,
        'hierarchy_level_tier': str,
        'tanimoto_tier': str,
        'lion_ood_tier': str,                  # [v2.0]
    },
    
    # FIVE-LEVEL HIERARCHY (unchanged from v1.0)
    'similarity_level': int,                   # 1-5
    'anchor_IAJD': str | None,
    'anchor_tanimoto': float,
    'delta_applied': bool,
    'delta_dimension': str | None,
    'delta_value': float | None,
    'max_tanimoto_to_training': float,
    
    # PROPAGATED pKa
    'pKa_pred': float,
    'pKa_PI90': [lo, hi],
    'pKa_tier': str,
    
    # STAGE A
    'organ_probs': {
        'whole_body_active', 'spleen_strong', 'liver_strong',
        'lung_strong', 'LN_strong', 'heart_active',  # all calibrated
    },
    'organ_dominant': str,                     # 'spleen' | 'liver' | 'lung' | 'unclear' | 'tied'
    'organ_dominant_tied_with': str | None,
    
    # STAGE B (per organ × 6)
    'log10_flux_<organ>': float,
    'log10_flux_<organ>_PI80': [lo, hi],       # [v2.0; v1.0 had only 90/95]
    'log10_flux_<organ>_PI90': [lo, hi],
    'log10_flux_<organ>_PI95': [lo, hi],
    'flux_<organ>_p_s': float,
    'flux_<organ>_PI90_p_s': [lo, hi],
    
    # 3-BIN POSTERIOR (total flux)
    'log10_flux_total_bin': 'low' | 'mid' | 'high',
    'log10_flux_total_bin_probs': {'low': p, 'mid': p, 'high': p},
    
    # STAGE C
    'stage_c_applied': bool,
    'stage_c_residual': float | None,
    
    # GP VARIANT [v2.0-FULL only]
    'gp_log10_flux_<organ>': float,
    'gp_log10_flux_<organ>_PI90': [lo, hi],
    
    # EXTERNAL BLOCK DIAGNOSTICS [v2.0]
    'lion_predictions': {tissue: zflux for tissue in LION_TISSUES},
    'lion_max_zflux': float,
    'lion_OOD_flag': int,
    'admet_predictions': {endpoint: value for endpoint in ADMET_KEEP},
    'block_d_features': {feature: value for feature in BLOCK_D_NAMES},
    
    # NEIGHBORS
    'nearest_neighbors': [
        {'IAJD_id', 'tanimoto', 'family', 'log10_flux_total', ...},
        ...  # K=5 by default
    ],
    
    # WARNINGS
    'warnings': list[str],
    'domain_flag': str,                        # EXACT_MATCH, LEVEL2_*, ..., OUT_OF_DOMAIN
}
```

Warning codes inherit from v4 pKa + v1.0 bioactivity, with v2.0 additions: `LION_OOD`, `ADMET_SATURATED`, `BLOCK_D_OUT_OF_RANGE`, `STAGE_C_DISABLED_NO_DLS_DATA`.

### 7.3 Bundle structure

```python
@dataclass
class BioActV2Bundle:
    version: str                                       # 'v2.0-MIN' or 'v2.0-FULL'
    train_df: pd.DataFrame                             # full master table
    train_fps: list                                    # Morgan FPs parallel to train_df
    feature_names: list[str]                           # 88 (MIN) or ~110 (FULL)
    
    # Five-level hierarchy state
    arch_mean_tables: dict[str, dict]                  # per arch_coarse → mean log-fluxes per organ
    chain_pair_grids: dict[str, dict]                  # per arch_coarse → 2D pair lookup
    delta_tables: dict[str, dict]                      # per family → token-delta per organ
    
    # Stage A
    stage_a_xgb: dict[str, XGBClassifier]              # 6 organs
    stage_a_isotonic: dict[str, IsotonicRegression]    # 6
    stage_a_feature_indices: list[int]                 # 31-feature Stage A subset
    
    # Stage B — Path B-XGB
    stage_b_xgb_direct: dict[str, XGBRegressor]        # 6 organs
    stage_b_mapie: dict[str, MapieRegressor]           # 6 organs, jackknife+
    stage_b_analog_delta: dict                         # K=8 NN + per-pair delta XGB
    stage_b_blend_alpha: float = 0.60                  # for non-GA-Tris
    stage_b_ga_tris_path: str = 'analog_only'
    
    # Stage B — Path B-GP [v2.0-FULL only]
    stage_b_gp: object | None                          # gpytorch MOGP model
    stage_b_gp_likelihood: object | None
    
    # Stage C
    stage_c_trained: bool
    stage_c_gbr: GradientBoostingRegressor | None
    stage_c_clamp: float = 0.5
    
    # External block caches
    lion_cache: dict[str, np.ndarray]                  # SMILES → 6 z-fluxes
    admet_cache: dict[str, np.ndarray]                 # SMILES → 41 endpoints (10 used)
    
    # pKa propagation
    pka_v7_bundle: object                              # existing v7.1 bundle
    
    # Domain
    williams_h_star: float                             # AD threshold
    family_pi_widths: dict[str, dict[str, float]]     # per family per organ PI half-widths
    
    # Metrics
    cv_metrics: dict                                   # full §6.2 panel per CV strategy
    holdout_metrics: dict                              # X1, X2, X3
    ablation_metrics: dict                             # ABL-1 through ABL-20
    baseline_metrics: dict                             # BL-1 through BL-6
```

Bundle size: ~5–10 MB for v2.0-MIN (model + training data). ~80–150 MB for v2.0-FULL (adds GP weights + fine-tuned LiON checkpoints if cached locally).

---

## Section 8. Analysis and Interpretability

v1.0 §8 is retained verbatim. The three deliverables are:

1. **Per-prediction SHAP attribution** — TreeSHAP for Stage A and Stage B XGBoost models. Computed per call when `return_diagnostics=True`. Aggregated over training set into `bioact_v2_feature_importance.xlsx` with one sheet per (stage, target).

2. **Partial dependence and counterfactual reasoning** — PDPs for top-8 features per (stage, target, family). v2.0-FULL adds the counterfactual `suggest_modifications()` API from the blueprint:

```python
def suggest_modifications(
    parent_smiles: str,
    objective: str,                          # 'max_spleen' | 'min_liver' | 'pareto:spleen-liver'
    top_k: int = 10,
    constraints: dict | None = None,        # {'liver': ('<', 7.0, 0.90)}
    fix_head: bool = True,                  # vary tails only
    n_candidates: int = 312,                # enumerated tail space
) -> list[ModificationProposal]:
    """v2.0-FULL only. Stage B MAE must be < 0.35 for this API to be enabled."""
```

3. **The Bioactivity Periodic Table deliverable** — analog of the Percec-lab pKa periodic table. Heatmap rendering of predicted log-flux per (chain pair × head group × core × linker) cell, with measured-cells highlighted and predicted-cells colored by confidence tier. Generated as an HTML artifact in v2.0-FULL.

---

## Section 9. Risk Register and Failure Modes

The v1.0 §9.3 risk register is retained and extended with v0/v2 risks:

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R-01 | SI tables for some papers lack numerical per-organ fluxes | High | Med | Use qualitative for Stage A; mask Stage B regressor row |
| R-02 | Same IAJD inconsistent fluxes across papers | Med | Med | Retain as separate rows; flag discrepancies ≥ 0.5 log |
| R-03 | DLS data missing for ~30% of rows | High | Low | Stage C disabled in MIN; bypassed at inference if missing |
| R-04 | Dataset n ≤ 290 insufficient for 88-feature model | Med | High | Strong regularization + family stratification + analog blend |
| R-05 | Stage B MAE > 0.55 (miss MIN bar) | Med | High | Drop per-organ targets except total; ship Stage A only |
| R-06 | X2 isomer test fails (< 9 of 14 correct) | Med | Med | Increase 3,4 vs 3,4,5 vs 3,5 token weights; retrain |
| R-07 | pKa v7.1 inference too slow for batch API | Low | Low | Cache pKa for training set |
| R-08 | Stage C learns unphysical correction (clamp active >5%) | Low | Med | Tanh clamp already applied |
| R-09 | Lab users need in-vitro prediction | Med | Low | v2.1 adds in-vitro head |
| R-10 | Novel query IAJD with no family match | Med | Med | Level 5 + LiON_OOD + low-Tanimoto → triple-flagged LOW conf |
| R-11 | New paper mid-project adds important data | Low | Low | v2.0.x release on intake |
| R-12 | RDKit ETKDG fails on some IAJDs | Low | Low | 2D fallback; v21 already handles for 127 molecules |
| R-13 | MALDI threshold too strict | Low | Med | Hand-review excluded compounds |
| R-14 | Researcher unavailable | Low | High | Spec is sufficient for handoff |
| **R-V0-1** | LiON checkpoint download fails | Med | High | Block B → 0 features; A+C+D still trains; bundle metadata flag |
| **R-V0-2** | ADMET-AI install conflict (chemprop version) | Med | Low | Block C → 0 features |
| **R-V0-3** | LiON v1.7.0 chemprop conflicts with admet-ai's pin | High | Med | Run LiON in separate venv subprocess; cache outputs |
| **R-V0-4** | Tier-1 SI tables figure-only and OCR fails | Med | High | Drop affected paper from Stage B; keep Stage A qualitative |
| **R-V0-5** | n_train < 100 after extraction | Low | High | Defer per-organ targets; ship total-flux + Stage A only |
| **R-V0-6** | LiON output collapses to constant on OOD IAJDs | Med | Med | Block B → near-zero feature importance; XGBoost ignores |
| **R-V0-7** | Predicted-pKa feature is correlated with itself in training | Low | High | Use measured pKa for training rows; predicted only for inference |
| **R-V0-8** | Heart organ has near-zero positives; classifier degenerate | High | Low | Fix predicted prob to base rate when n_pos < 5 |
| **R-V0-9** | Some Tier-1 IAJDs not in v21 | Med | Low | `_compute_inherited_from_mol` fallback in `features.py` |
| **R-V2-1** | MAPIE jackknife+ slower than expected on n>200 | Low | Low | Fall back to split-conformal; document in model card |
| **R-V2-2** | GP-variant fitting fails (numerical issues, hyperparameter instability) | Med | Low (FULL only) | Disabled in MIN; FULL falls back to Path B-XGB |
| **R-V2-3** | Block D features uncorrelated with target (heuristic too crude) | Med | Low | Ablation ABL-18/19 detects; Block D drops from FULL |

---

## Section 10. Build Order

### 10.1 v2.0-MIN build order (single session, ~3 hours)

| Phase | Time | Deliverable |
|---|---|---|
| **Phase 1 — Environment + dataset** | 45 min | `bioact_v2_master.csv` (~155 rows), Tier-1 SIs extracted, MALDI-validated |
| **Phase 2 — Block A computation** | 15 min | 50-dim Block A per row (reuses `features.py`) |
| **Phase 3 — Block B (LiON inference)** | 15 min | `lion_cache.parquet`, 14 features per row |
| **Phase 4 — Block C (ADMET-AI inference)** | 10 min | `admet_cache.parquet`, 10 features per row |
| **Phase 5 — Block D (heuristics)** | 10 min | 6 features per row |
| **Phase 6 — Train/holdout split** | 10 min | X1, X2, X3 reserved; train.parquet ready |
| **Phase 7 — Stage A training + calibration** | 20 min | 6 calibrated XGBoost classifiers; 5-fold CV metrics |
| **Phase 8 — Stage B training + MAPIE** | 30 min | 6 XGBoost regressors + jackknife+ PIs + analog-delta tables |
| **Phase 9 — Five-level hierarchy assembly** | 15 min | `arch_mean`, `chain_pair_grids`, `delta_tables` |
| **Phase 10 — Bundle + smoke test** | 20 min | `iajd_bioact_v2_bundle.pkl` + `iajd_bioact_v2.py` + 4-compound smoke test |
| **Phase 11 — Holdout evaluation** | 15 min | X1, X2, X3 metrics in bundle.holdout_metrics |
| **Phase 12 — Ablations 16–20 + baselines 4–6** | 30 min | External-block ablations only; full §6.3 deferred to FULL |
| **Phase 13 — Model card + documentation** | 15 min | One-page MD with all metrics, caveats, limitations |

Total: ~250 minutes (~4 hours of dialog turns; achievable in two consecutive sessions if needed).

### 10.2 v2.0-FULL build order (16 weeks, matches v1.0 timeline)

Weeks 1–3: v2.0-MIN built first (it's the seed for FULL).
Weeks 4–5: Full hyperparameter search per stage; full §6.3 ablation (ABL-1 through ABL-20).
Week 6: GPU fine-tune LiON on IAJD bioactivity data; integrate fine-tuned weights into Block B.
Weeks 7–8: Multi-output GP with rank-2 ICM (Path B-GP); GPyTorch implementation; convergence debugging.
Week 9: MAP4 fingerprint integration (resolve `tmap` install or use docker container).
Weeks 10–11: CG-MD descriptors via MARTINI 3 on dendrimersome assembly (HPC; lab does this).
Week 12: Stage C re-enable once formulation completeness ≥ 75%.
Week 13: Counterfactual / Pareto API (`suggest_modifications`).
Week 14: Bioactivity Periodic Table HTML deliverable.
Week 15: TDC submission preparation.
Week 16: Documentation, technical report, v1.0-FULL release tag.

### 10.3 Decision tree for "should we build MIN or FULL today"

The MIN target is what gets built today, in this chat session, with no GPU and no external infrastructure. It is the operational system the lab uses while FULL is being built. The decision flow:

1. Is the dataset n ≥ 100 with usable per-organ flux numbers? **If no, build a Stage-A-only MIN** (skip Stage B targets except total) and document.
2. Are the LiON checkpoints downloadable? **If yes, build Block B; if no, skip Block B and document in model card.** Block A+C+D alone is still defensible.
3. Are Tier-1 SI tables OCR-extractable? **If yes, full MIN; if 1+ paper fails, drop to Tier-1-minus-one and document.**
4. Does the env support `chemprop==1.7.0` AND `admet-ai`? **If they conflict, use subprocess isolation per Risk R-V0-3.**

The MIN build is the v0 spec promoted to first-class status. The v2.0 design treats it as the deployable shipped state, with FULL as the documented stretch target.

---

## Section 11. Acceptance Criteria Summary

### v2.0-MIN production gate (deployable today)

1. End-to-end inference works: `predict_bioactivity('SMILES')` returns the §7.2 dict.
2. Pooled 5-fold MAE on log10(flux_total) ≤ 0.55.
3. Stage A organ-dominance accuracy ≥ 70%.
4. 90% PI empirical coverage ∈ [0.85, 0.92].
5. X1 stability variance < 0.20 log-units.
6. Inference latency < 2 sec.
7. Model card written and committed.

### v2.0-FULL release gate (publication-grade)

1. All MIN criteria hold.
2. Pooled LOO MAE on log10(flux_total) ≤ 0.45.
3. Pooled LOO MAE on log10(flux_spleen) ≤ 0.50.
4. Organ-dominance accuracy on X3 ≥ 80%.
5. X2 rank-order ≥ 9 of 14.
6. All 20 ablations (ABL-1 through ABL-20) executed with full §6.2 metric panel.
7. All 6 baselines (BL-1 through BL-6) reported.
8. Counterfactual API works for ≥ 80% of test compounds.
9. Bioactivity Periodic Table deliverable rendered.
10. Technical report committed; v2.0-FULL tagged.

---

## Section 12. Appendices

### A. Master Extraction Checklist per Paper [retained verbatim from v1.0 Appendix A]

### B. SMARTS Patterns for Hydrophobic-Part Features [retained from v1.0 Appendix B; live in `features.py`]

### C. Reference Implementation Sketch (`iajd_bioact_v2.py`) [updated from v1.0 Appendix C]

```python
# iajd_bioact_v2.py — IAJD Bioactivity v2.0 Inference

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from features import compute_bioact_features, STAGE_A_FEATURE_INDICES, STAGE_C_FEATURE_INDICES
from iajd_pka_predict import load_bundle as load_pka_bundle, predict_pka

BUNDLE_VERSION = 'v2.0-MIN'   # or 'v2.0-FULL'
LION_TISSUES = ['liver_IV', 'lung_IT', 'lung_inh', 'lung_neb', 'muscle_IM', 'nasal']
ADMET_KEEP = [
    'VDss_Lombardo', 'PPBR_AZ', 'BBB_Martins', 'Clearance_Hepatocyte_AZ',
    'Half_Life_Obach', 'HIA_Hou', 'Caco2_Wang', 'Pgp_Broccatelli',
    'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca',
]
BLOCK_D_NAMES = [
    'peterca_radius_proxy_nm', 'packing_parameter_CPP', 'apoE_binding_heuristic',
    'charge_density_pH7', 'charge_density_pH5', 'curvature_proxy_inv_nm',
]


@dataclass
class BioActV2Bundle:
    version: str
    train_df: pd.DataFrame
    train_fps: List
    feature_names: List[str]
    arch_mean_tables: Dict
    chain_pair_grids: Dict
    delta_tables: Dict
    stage_a_xgb: Dict
    stage_a_isotonic: Dict
    stage_a_feature_indices: List[int]
    stage_b_xgb_direct: Dict
    stage_b_mapie: Dict
    stage_b_analog_delta: Dict
    stage_c_trained: bool
    stage_c_gbr: object
    lion_cache: Dict
    admet_cache: Dict
    pka_v7_bundle: object
    williams_h_star: float
    family_pi_widths: Dict
    cv_metrics: Dict
    holdout_metrics: Dict


def load_bundle(path='iajd_bioact_v2_bundle.pkl') -> BioActV2Bundle:
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_bioactivity(
    smiles: str,
    bundle: BioActV2Bundle,
    formulation: Optional[Dict] = None,
    return_diagnostics: bool = False,
    return_neighbors: int = 5,
) -> Dict:
    """Full v2.0 inference. See spec §5.7 for the 13-step pipeline."""
    # Step 1: canonicalize + FP
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    
    # Step 2: token assignment (family, head, core, linker)
    tokens = assign_tokens(mol)  # uses features.py SMARTS patterns
    
    # Step 3: Block A (50 features)
    pka_pred = predict_pka(canonical, bundle.pka_v7_bundle)
    block_a = compute_bioact_features(mol, tokens, pka_bundle=bundle.pka_v7_bundle)[:50]
    
    # Step 4: Block B (LiON, 14 features) — cache-first
    block_b = lion_features_or_cache(canonical, bundle.lion_cache)
    
    # Step 5: Block C (ADMET-AI, 10 features) — cache-first
    block_c = admet_features_or_cache(canonical, bundle.admet_cache)
    
    # Step 6: Block D (geometric / corona, 6 features)
    block_d = compute_block_d(tokens, pka_pred['pKa_pred'], formulation)
    
    # Step 7: Formulation block
    block_form = compute_formulation_vector(formulation) if formulation else default_formulation()
    
    # Concat full feature vector
    X = np.concatenate([block_a, block_b, block_c, block_d, block_form])
    assert X.shape == (88,), f'Expected 88 features, got {X.shape}'
    
    # Step 8: Five-level hierarchy
    sims = np.array([TanimotoSimilarity(fp, f) for f in bundle.train_fps])
    max_sim = float(sims.max())
    hierarchy_level, anchor_info = determine_hierarchy_level(
        canonical, tokens, sims, bundle, formulation
    )
    
    # Step 9: Stage A
    X_a = X[bundle.stage_a_feature_indices]
    organ_probs = {}
    for organ, model in bundle.stage_a_xgb.items():
        raw = float(model.predict_proba(X_a.reshape(1, -1))[0, 1])
        calibrated = float(bundle.stage_a_isotonic[organ].predict([raw])[0])
        organ_probs[f'{organ}'] = calibrated
    organ_dominant, tied_with = compute_dominant_organ(organ_probs)
    
    # Step 10: Stage B (Path B-XGB)
    stage_b_out = {}
    for organ, model in bundle.stage_b_xgb_direct.items():
        direct_pred = float(model.predict(X.reshape(1, -1))[0])
        # Analog-delta blend
        delta_pred = analog_delta_predict(X, fp, organ, bundle, K=8)
        # GA-Tris routing
        if tokens.family == 'GA-Tris':
            blended = delta_pred  # 100% analog
        else:
            blended = 0.60 * direct_pred + 0.40 * delta_pred
        # Conformal PI
        pi80, pi90, pi95 = mapie_pis(bundle.stage_b_mapie[organ], X)
        stage_b_out[organ] = dict(
            point=blended, direct=direct_pred, delta=delta_pred,
            PI80=pi80, PI90=pi90, PI95=pi95,
        )
    
    # Step 11: Stage C (if enabled and formulation supplied)
    stage_c_applied = False
    if bundle.stage_c_trained and formulation and formulation_complete(formulation):
        for organ in stage_b_out:
            X_c = build_stage_c_features(formulation, tokens.family)
            residual = float(bundle.stage_c_gbr.predict(X_c.reshape(1, -1))[0])
            residual_clamped = 0.5 * np.tanh(residual / 0.5)
            stage_b_out[organ]['point'] += residual_clamped
        stage_c_applied = True
    
    # Step 12: Confidence tier composition
    pi_tier = tier_from_pi_width(stage_b_out['log10_flux_total']['PI90'])
    hierarchy_tier = tier_from_level(hierarchy_level)
    tanimoto_tier = tier_from_tanimoto(max_sim)
    lion_ood_tier = 'LOW' if block_b[12] == 1 else 'HIGH'  # lion_OOD_flag at index 12
    confidence_tier = min_tier([pi_tier, hierarchy_tier, tanimoto_tier, lion_ood_tier])
    
    # Step 13: Build return dict
    result = build_return_dict(
        canonical=canonical, version=bundle.version, family=tokens.family,
        confidence_tier=confidence_tier, hierarchy_level=hierarchy_level,
        organ_probs=organ_probs, organ_dominant=organ_dominant,
        stage_b_out=stage_b_out, stage_c_applied=stage_c_applied,
        block_b=block_b, block_c=block_c, block_d=block_d,
        nearest_neighbors=top_k_neighbors(sims, bundle.train_df, k=return_neighbors),
        pka_pred=pka_pred, max_tanimoto=max_sim, anchor_info=anchor_info,
    )
    
    return result
```

### D. Validation Set Design [retained from v1.0 Appendix D]

X1 = IAJD97 stability time-course (Sci Adv 2026 Table S4).
X2 = 14 constitutional isomers from ja3c13569 (IAJDs 87, 294, 311, 297, 93, 249, 97, 300, 253, 301, 178, 308, 310, 309).
X3 = 10% stratified random by family × organ_dominant.

### E. Known Edge Cases and Failure Modes [extended from v1.0 Appendix E]

- **MAP4 fingerprint install fails** → use Morgan-2-2048 instead; document.
- **LiON checkpoint not available** → Block B reverts to zeros; bundle metadata flag `lion_active=False`.
- **chemprop and admet-ai version conflict** → run LiON in subprocess venv; cache outputs.
- **GA-Tris query with no in-family training neighbor** → fall through to Level 5; doubled PI.
- **Novel head group (e.g. new piperazine variant)** → SMARTS doesn't match; emit `HEAD_GROUP_UNCLASSIFIABLE`; Level 5 routing.
- **DLS supplied but PDI > 0.5** → Stage C clamp likely to fire; emit `DLS_OUT_OF_RANGE` warning.
- **`lion_max_zflux` > 99th percentile of training** → unusual: LiON thinks this lipid is extraordinarily active; emit `LION_HIGH_PREDICTION` warning so user knows the prediction is partly LiON-derived.

### F. Glossary [retained from v1.0 Appendix F]

---

**End of v2.0 specification.**

This document supersedes Bioactivity v1.0, the Hydrophobic Tail Blueprint, and the Bioact v0 Prototype Spec. The MIN target is buildable in a single session and ships the operational system; the FULL target is the publication-grade stretch goal documented for the v2.0-MIN → v2.0-FULL upgrade path. Every architectural decision in v2.0 is traceable to a specific document section in v1.0, the blueprint, or v0, and every divergence from those documents is explicitly justified in §1 or in the relevant section preamble.
