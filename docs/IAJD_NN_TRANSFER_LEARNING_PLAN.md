# IAJD Neural-Net Transfer-Learning Plan (pretrain → fine-tune → frozen-encoder + GP)

> Goal: test whether a deep model, **pretrained on a domain-matched lipid dataset and
> transferred to the ~273 IAJDs**, beats the current XGBoost/GP baseline for predicting
> organ flux (and the physics levers). Written 2026-06-01. This is a SIDE-TRACK to the
> physics-design panel — slot it AFTER the panel so it doesn't fork the overnight run.

## 0. Honest framing — when does a NN help here at all?

With only ~273 IAJDs, a net trained from scratch **will overfit** and lose to the GP/GAM
(see [[feedback-design-optima-gam-gpbo]]). A NN earns its place ONLY via **transfer**:
learn representations on a large, *related* dataset, then adapt to the IAJDs. So the whole
question reduces to: **is there a pretraining source close enough to IAJDs that its learned
features transfer?** The answer is a qualified yes — and the field already did the
experiment (AGILE).

## 1. Pretraining-source comparison (UPDATED 2026-06-01 after a dedicated lit search)

A thorough literature search (verified, repos opened) found something **better than AGILE as
a dataset**: **LNPDB** — a public MIT-licensed superset that *contains* AGILE's data and adds
**in-vivo organ-delivery labels** (our actual endpoint). Ranked:

| Source | What it is | Match to IAJDs | Verdict |
|---|---|---|---|
| **LNPDB** (Collins et al., Nat. Commun. 2026; `github.com/evancollins1/LNPDB`, MIT) | **12,845 unique ionizable-lipid SMILES** from 42 papers; **2,388 in-vivo rows / 1,513 lipids** across 18 in-vivo papers with **per-organ labels (liver 905, spleen 258, lung 96, muscle 486, …)**; head/linker/tail-decomposed SMILES; ships ready AGILE + LiON/Chemprop fine-tune harnesses | **In-vivo organ-flux endpoint — matches our spleen/liver/lung target directly.** Ionizable-amphiphile prior. Still 0 dendritic/≥3-tail rows. | **PRIMARY pretraining corpus** — public, in-vivo, large, superset of AGILE |
| **AGILE** (Xu et al., Nat. Commun. 2024; `bowang-lab/AGILE`, MIT) | MolCLR GNN **self-sup on 60k virtual ionizable lipids** (pretrained `model.pth` shipped) + 1,200 in-vitro labels | Closest published **contrastive encoder** to our amphiphile space; labels are in-vitro only | **Use the 60k pretrained ENCODER as init** (not its labels) |
| **LANTERN** (arXiv 2507.03209; `AsalMehradfar/LANTERN`) | re-audited AGILE labels: **235/1,200 were wrong**; ships a cleaned **1,100-lipid** set | corrected in-vitro labels | **Use INSTEAD of AGILE's raw labels** if adding the in-vitro task |
| LipidAI/Ouyang (Nat. Commun. 2024, figshare) · LNP_ML (`jswitten/LNP_ML`) | ~370 in-vivo lipids · Chemprop LNP in-vivo codebase | extra in-vivo signal | **in-vivo supplements** to LNPDB |
| Siegwart iPhos / SORT (5A2-SC8 **dendrimer-lipid**) | closest *dendritic* chemistry + in-vivo organ targeting | **best scaffold match** BUT data **not public** (SMILES not tabulated, "on request") | **want it, can't get it** — note the gap |

**Bottom line (honest):** the ideal set — thousands of *dendritic, in-vivo* ionizable
amphiphiles — **does not exist publicly** (the entire Percec IAJD literature is ~hundreds of
molecules in figures, never released as SMILES+activity; our ~273 IAJDs essentially *are* the
world's IAJD dataset). So pretrain on the **in-vivo ionizable-amphiphile** domain and transfer.
**Use LNPDB (in-vivo rows) as the supervised pretraining corpus + AGILE's 60k MolCLR weights as
the encoder init**, and be explicit that the prior is "ionizable amphiphile," NOT "Janus
dendrimer" — that residual is exactly what the IAJD fine-tune + the physics features (A/B/C)
must carry.

## 2. The recipe (3 stages, UPDATED)

1. **Encoder init = AGILE's 60k MolCLR weights** (self-supervised on 60k virtual ionizable
   lipids; the closest published contrastive encoder to our amphiphile space). Optionally
   continue self-sup contrastive pretraining on an **IAJD-like virtual library** (enumerate
   Janus-dendrimer head/linker/tail combos from `iajd_grammar`) to nudge the representation
   toward the dendritic scaffold the public data lacks.
2. **Supervised pretrain on LNPDB IN-VIVO rows** (filter `Model_type==in_vivo`, our organs:
   spleen/liver/lung/LN; within-study normalization — their LiON pipeline does this). This is
   the upgrade over the old plan: the pretraining endpoint is now *in-vivo organ delivery*,
   matching the IAJD target, not in-vitro transfection. (Add LANTERN's cleaned 1,100 in-vitro
   set + the LipidAI in-vivo rows as auxiliary tasks if helpful.)
3. **Freeze the encoder → GP head on the 273 IAJDs.** Embed the IAJDs, put the existing
   **GP / Bayesian-opt** ([[feedback-design-optima-gam-gpbo]]) on the embeddings (+ the physics
   A/B/C descriptors) to predict organ flux. Keeps the **calibrated uncertainty the design loop
   needs**; full fine-tune on 273 points would overfit. (Deep-kernel GP = the principled form.)

The physics descriptors (Module A pKa, Module B c₀, Module C H_II) concatenate onto the
embedding as extra GP features — mechanism + learned representation together.

## 3. Honest caveats (what will limit the lift)

- **Transfection cliffs.** AGILE and TransLNP both report that tiny structural changes cause
  large activity jumps — the dominant prediction error. IAJDs have the same activity cliffs
  (the SAR prior [[project-informed-mutation-sar]] is exactly such steep local structure→
  activity). A smooth GNN+GP will underperform precisely at the cliffs that matter for design.
- **Scaffold domain gap.** AGILE lipids are 1–2-tail Ugi lipids; IAJDs are multi-tail Janus
  dendrimers — out-of-distribution for AGILE's encoder. Transfer helps head/tail chemistry,
  NOT the dendritic topology. The added IAJD-like pretraining library (stage 1) is meant to
  patch this, but it's unlabeled, so it only shapes the representation, not the endpoint.
- **Endpoint mismatch (the big one).** AGILE = *in-vitro* HeLa/macrophage transfection; IAJD
  flux = *in-vivo* organ-targeted (spleen) luciferase. In-vitro→in-vivo is a real, documented
  gap; expect *directional* transfer, not calibrated. lion's in-vivo data (if usable) is the
  bridge. This is also why the GP uncertainty matters — don't trust point predictions.
- **Data-truth.** Same master confound as everywhere here: AGILE/lion labels carry their own
  measurement noise; weight by reliability, don't treat transfection numbers as ground truth.

## 4. Validation protocol — the NN must EARN its use

Non-negotiable gate (honesty): the transfer model is used for design **only if it beats the
current XGBoost/GP baseline on held-out IAJDs**, by the SAME leave-one-out / leave-one-family-
out protocol already in the repo (`loo_bioact_components.py`, `auto_retrain_log.csv`). Report:
- LOO R²/Spearman vs baseline, with honest CIs (n=273 → wide).
- **Leave-one-FAMILY-out** especially — the real test of whether transfer generalizes to a
  held-out architecture (the actual design use-case), not just interpolation.
- Calibration of the GP uncertainty (coverage of the 90% interval).
If it doesn't beat the baseline, we say so and keep the GP/GAM. A pretrained net is not
automatically better — it has to prove it on our data.

## 5. Concrete next steps (after the physics panel)

1. **Audit lion data**: read `lion_repo/data/libraries/` + the chemprop args — what molecules,
   what endpoint (in-vitro/in-vivo), how many, label units. Decide if it's an in-vivo bridge.
2. **Fetch AGILE**: GitHub (Bowen Li lab) for code + the 1,200-lipid data + MolCLR weights;
   confirm license. bioRxiv 2023.06.01.543345 / Nat. Commun. s41467-024-50619-z.
3. **Featurize IAJDs** consistently (SMILES_canonical → the AGILE graph featurizer).
4. Stage 1–3 as above; **train on the rented 3090** (minutes–hours; ~$1–10 total).
5. **Gate vs baseline** (§4). Only then wire into the design loop as an extra scorer.

## 6. Cost
Trivial: GNN pretrain+finetune on a 3090 is minutes-to-a-few-hours (~$0.50–10 on RunPod).
**Compute is not the constraint — data relevance + the cliffs/endpoint gaps are.**

## Sources
- AGILE — Nat. Commun. 2024: https://www.nature.com/articles/s41467-024-50619-z ; bioRxiv: https://www.biorxiv.org/content/10.1101/2023.06.01.543345v1 ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11282250/
- Data-balanced transformer (TransLNP) — Brief. Bioinform. 2024: https://academic.oup.com/bib/article/25/3/bbae186/7658017
- TransMA — arXiv 2407.05736: https://arxiv.org/html/2407.05736v1
- LANTERN — arXiv 2507.03209: https://arxiv.org/html/2507.03209v1
- ML for ionizable lipid ID — bioRxiv 2023.11.09.565872: https://www.biorxiv.org/content/10.1101/2023.11.09.565872.full.pdf
