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

## 1. Pretraining-source comparison (the key decision)

| Source | What it is | Match to IAJDs | Verdict |
|---|---|---|---|
| **AGILE** (Xu et al., Nat. Commun. 2024) | MolCLR-style GNN **self-supervised on 60k virtual ionizable lipids**, **fine-tuned on 1,200 Ugi-3CR ionizable lipids × HeLa/macrophage transfection** | Ionizable-amphiphile chemistry shared; scaffold differs (2-tail lipids vs Janus dendrimers); endpoint = in-vitro transfection | **RECOMMENDED primary** — domain-matched on the *ionizable* axis, SOTA, code+data public |
| **lion_repo** (in this repo) | chemprop GNN pipeline + LNP delivery data the lab already curated | Possibly closer on the *delivery in-vivo* endpoint; need to audit its actual molecules/labels | **Use as 2nd source / in-vivo bridge** — audit `lion_repo/data/libraries/` to see how close |
| TransLNP / TransMA / LANTERN / data-balanced transformer (2024–25) | transformer/multimodal ionizable-LNP screeners, pretrain+finetune | same ionizable-lipid space, different architectures + extra datasets | **Mine for additional labeled data**, not as the backbone |
| Generic (MolCLR/ChemBERTa on ZINC/PubChem) | self-supervised on millions of drug-like molecules | weak — IAJDs are far OOD amphiphiles | **backbone init only**, not the domain signal |

**Bottom line:** AGILE is strictly better-matched than generic chemistry for the *ionizable*
half of an IAJD, and its 60k-lipid self-supervised encoder is the single most reusable
asset. lion may add an *in-vivo delivery* signal AGILE lacks. Use **AGILE primary, lion as
the in-vivo bridge.**

## 2. The recipe (3 stages)

1. **Self-supervised pretrain** a GNN encoder (MolCLR contrastive, as AGILE does) on a large
   UNLABELED amphiphile library — reuse AGILE's 60k virtual lipids and, to close the scaffold
   gap, **add an IAJD-like virtual library** (enumerate Janus-dendrimer head/linker/tail
   combos from `iajd_grammar`). Learns amphiphile structure representations.
2. **Supervised fine-tune** the encoder on AGILE's **1,200-lipid transfection** data (+ lion's
   in-vivo labels if compatible). Learns *delivery-relevant* features, not just chemistry.
3. **Freeze the encoder → GP head on the IAJDs.** Embed the 273 IAJDs, then put the existing
   **GP / Bayesian-opt** ([[feedback-design-optima-gam-gpbo]]) on the embeddings to predict
   flux. This keeps the **calibrated uncertainty the design loop needs** — fine-tuning all
   weights on 273 points would overfit. (Deep-kernel GP = the principled version.)

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
