# Methods for Small, Noisy Chemical Datasets — a cited, honest reference

**Scope:** building predictive/design models on **tens–few-hundred** compounds with
**noisy** measured labels (e.g. ~270 IAJDs with noisy mRNA-transfection flux). Distilled
from a verified literature search (27 primary sources; adversarial verification killed 7
over-claims, recorded in §7). Read it with the optimization protocol
(`docs/DESIGN_OPTIMIZATION_PROTOCOL.md`) — this covers *breaking the noise ceiling*, that
covers *finding optima + proposing candidates*.

---

## 0. The one finding to internalize

**Label noise caps your *measured* correlation, not the model's *true* accuracy.**
Across 5 algorithms × 8 datasets, true-test RMSE stayed ~flat while noisy-test RMSE rose
as training noise grew — "QSAR models can make predictions more accurate than their
training data" (Kolmar & Grulke, *J. Cheminform.* 2021, 13:92).

**Consequences:**
- Weak physics-vs-target correlations are **consistent with a noise ceiling, not absent
  physics.** Don't conclude the physics is dead from a weak correlation.
- **No fancier algorithm will lift the *measured* ceiling.** Switching models to "rescue
  the correlation" is futile. (A stronger claim — that noise hard-caps *true* accuracy —
  was **refuted**; it only caps what you can *measure*.)
- So the job is: **reduce the noise, use the features well, validate honestly** — and
  accept that the validation number will look modest even when the model is good.
  *(Caveat: this clean result assumes Gaussian-random, not systematic, error.)*

---

## 1. The biggest *validated* lever: descriptors as FEATURES (not raw correlations)

Adding explicit chemical descriptors (count Morgan fingerprints) to the AGILE lipid model
raised random-split **R² 0.27 → 0.71** (LANTERN, arXiv:2507.03209), and supervised-
pretraining gains **erode once good hand-crafted features are present** (Sun, Dai & Yu,
*Does GNN Pretraining Help Molecular Representation?*, NeurIPS 2022).

➡️ **Physics/RDKit/fingerprint features fed into a simple model (GP/RF) is the most
reliable documented gain** — and means your QM/MD physics pays off *as features*, not as
standalone correlations. This is the opposite of "the physics was useless."

---

## 2. Domain transfer learning: real, validated — with one genuine risk for you

- **The recipe works.** Domain lipid-ML on a few-hundred-compound screen generalized
  *prospectively*: 584 ionizable lipids → ML → 40,000-lipid virtual screen → synthesize
  top 16 → hit **119-23** beat benchmark lipids in muscle/immune tissues (Li, Langer,
  Anderson et al., *Nat. Mater.* 2024, 23:1002). **AGILE** = GIN encoder warm-started from
  MolCLR (>10M molecules) → contrastive pretrain on 60k virtual lipids → supervised
  fine-tune on ~1,200 transfection measurements (Xu/Wang/Li et al., *Nat. Commun.* 2024,
  s41467-024-50619-z; code `bowang-lab/AGILE`). **Supervised** pretraining with labels
  aligned to the task is the *favored* regime (NeurIPS 2022).
- **⚠️ The risk for IAJDs:** they are amphiphilic **Janus dendrimers**, structurally
  distinct from the simple ionizable lipids AGILE was trained on — i.e. **out-of-
  distribution**, where **negative transfer is a documented, real failure mode** (Hu et
  al., *ICLR* 2020: naive pretraining was *worse than none* on 2/8 molecular datasets;
  the combined node+graph strategy is what avoids it). AGILE's own stated weakness is poor
  extrapolation beyond its training distribution.
- ➡️ **Treat AGILE transfer as a *benchmark to test*, not an assumed win** — always
  compared against a physics-features-only GP baseline.

---

## 3. What is OVERSOLD (skip or down-weight)

- **Generic self-supervised pretraining / molecular "foundation models":** in the largest
  benchmark to date (25 pretrained embeddings × 25 datasets, hierarchical Bayesian test),
  **nearly none beat a plain ECFP fingerprint**; only CLAMP (itself fingerprint-based) won
  (arXiv:2508.06199). Don't chase generic foundation models. *(Refuted over-reads: this
  does NOT prove GNN embeddings are categorically worse, nor that domain transfer like
  AGILE is futile.)*
- **Heteroscedastic GPs at n≈270:** the documented recipe exists — Most-Likely
  Heteroscedastic GP (fit G1 to the function, G2 to log-noise; Kersting et al. 2007, via
  Griffiths et al. arXiv:1910.07779) — **but it helps mainly when noise is large and needs
  enough samples to fit the noise function**, so it's likely **under-powered at 270**. A
  **homoscedastic GP on replicate-averaged labels** is the safer default. *(A claim that
  supplying an explicit per-point-uncertainty α-vector substantially improves true-value
  prediction was **refuted**.)*
- **Pooling external/literature labels without curation:** itself a major noise source —
  merging IC50/Ki across assays, **65% of duplicates differ >0.3 log, 27% >1 log**; heavy
  curation cuts noise (Kendall τ 0.51→0.71) but discards ~99% of data (Landrum & Riniker,
  *JCIM* 2024, 64:1560). Curation trades noise for sample size — choose deliberately.

---

## 4. Validation: family/cluster-aware, never random

Random-split metrics are **over-optimistic and reverse model rankings**. Under Murcko
scaffold splitting, best MLP **R² 0.82 → 0.53**, simple kNN became top, AGILE collapsed to
~0.006 (LANTERN). Use **leave-one-family-out** (your families) and a cluster/UMAP split as
a cross-check — scaffold splits themselves can be optimistic vs cluster splits on small
data (arXiv:2406.00873). **Trust no single split blindly.**

---

## 5. Prioritized recipe (what to actually do)

1. **Average replicates** per compound; record the per-compound spread (a real noise
   estimate, and your honest error bars).
2. **Physics + RDKit + Morgan features → a homoscedastic Gaussian Process** (replicate-
   averaged y). This is the highest-confidence lever (§1).
3. **AGILE transfer (frozen encoder + GP head) as a benchmark** vs #2 — expect possible
   OOD negative transfer for dendrimers; only adopt if it *beats* the physics-features GP
   under honest validation.
4. **Leave-one-family-out + cluster split** (§4). Report both; trust neither alone.
5. **Set expectations:** the LOFO number will look modest even for a good model — that's
   the noise ceiling on *measurement*, not on the model (§0).
6. Feed the validated model into the **optimum/GP-BO design loop**
   (`DESIGN_OPTIMIZATION_PROTOCOL.md`) — find the peak, propose the next candidate.

---

## 6. Honest limits & open questions (state these)

- **No source measures AGILE-transfer on noisy-flux *regression* at n≈270 under LOFO** —
  it's an empirical question to settle on your own data, vs the physics-features baseline.
- **The dendrimer-vs-lipid OOD gap is the key unknown** for transfer (negative-transfer risk).
- **You need the replicate structure / per-point noise** of the 270 flux values to decide
  het-GP vs homoscedastic, and whether the noise is Gaussian-random (where "true accuracy
  preserved" applies) or systematic.
- Effect sizes above are mostly from **classification benchmarks**, not your exact noisy-
  flux regression — directional, not guaranteed.

---

## 7. Refuted over-claims (do not repeat)

- Noise does **not** impose a hard ceiling on *true* accuracy (only on *measured*).
- An explicit GP measurement-uncertainty α-vector was **not** shown to substantially
  improve true-value prediction on the cited QSAR data.
- GNN embeddings are **not** categorically worse than fingerprints.
- "A simple MLP beats AGILE" (random split) — refuted as standalone; the robust version is
  the within-AGILE Morgan-fingerprint improvement.

---

## 8. Primary citations

- Kolmar & Grulke, *J. Cheminform.* 2021, 13:92 — noise caps measured, not true accuracy.
- Xu/Wang/Li et al. (AGILE), *Nat. Commun.* 2024, s41467-024-50619-z; `bowang-lab/AGILE`.
- Li, Langer, Anderson et al., *Nat. Mater.* 2024, 23:1002 — 584-lipid → prospective hit 119-23.
- Hu et al., *Strategies for Pre-training GNNs*, ICLR 2020 (arXiv:1905.12265) — negative transfer.
- Sun, Dai & Yu, *Does GNN Pretraining Help?*, NeurIPS 2022 (arXiv:2207.06010).
- molecular-embedding benchmark, arXiv:2508.06199 — foundation models rarely beat ECFP.
- Griffiths et al., arXiv:1910.07779; Kersting et al., ICML 2007 — heteroscedastic GP (MLHGP).
- Landrum & Riniker, *JCIM* 2024, 64:1560 — multi-source label noise.
- LANTERN, arXiv:2507.03209 (preprint) — Morgan-on-AGILE 0.27→0.71; scaffold-split collapse.
- *Scaffold Splits Overestimate VS Performance*, arXiv:2406.00873.
