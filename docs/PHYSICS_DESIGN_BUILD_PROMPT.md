# BUILD PROMPT — Physics-Design Compass for IAJD Endosomal Escape

**Status:** specification + executable prompt. Hand this to a capable coding/simulation
agent (or execute directly). Every instruction is grounded in the verified
literature synthesis (see §9 Citations) and must obey the **no-proxy** and
**CPU-only** constraints in §1.

**Author context:** Percec-type Ionizable Amphiphilic Janus Dendrimers (IAJDs)
for mRNA delivery. Reference molecule = **IAJD-369**, the highest measured fluxer
and **spleen-tropic**:
`CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC`
(GA-Tris family · gallic-acid core · 3 branched 2-ethylhexyl-type tails ·
benzyl-ester linker · hydroxyethyl-piperazine "H2EPRZ" ionizable head ·
log10 flux 9.39, spleen-dominant).

---

## 1. MISSION & HARD CONSTRAINTS

**Mission.** Build a CPU-runnable, mechanistically-grounded pipeline that places any
IAJD in a small, physically-meaningful coordinate space tied to *endosomal escape*,
and uses it to **propose a structural analog of 369 with qualitatively higher
escape propensity** — by chasing the mechanistic optimum, NOT by ML flux
correlation. Exact flux prediction is explicitly out of scope; **qualitative,
direction-correct improvement is the success criterion.**

**Hard constraints (non-negotiable):**
1. **No proxies.** Every reported quantity is either a real computed value or
   `NaN` with an audit reason. No family medians, no constants, no "representative"
   substitutions. (Project policy, reaffirmed throughout.)
2. **CPU-only, no cluster.** Must run on a 10-core Apple-Silicon laptop. GROMACS
   is CPU-only here (no CUDA). Time is not a constraint, but a single run must not
   require a GPU or HPC node. **Thermal:** the machine is a *fanless* M4 Air —
   cap sustained load to ≤ ~4–6 cores, `nice` long jobs, and checkpoint.
3. **Validate before trusting.** No determinant is used for design until its
   protocol reproduces a published benchmark lipid to stated tolerance (§4).
4. **Honest scope.** The three caveats in §8 (spleen mechanism, hypothesis-status
   of H_II, Janus-dendrimer mapping gap) must be surfaced in every report; do not
   present qualitative rankings as exact predictions.

**Reuse, don't reinvent.** The repo already has: `martini/` (Martini-3
self-assembly: `build_cg.py`, `run_selfassembly.py`, `analyze_md.py`),
`qm_descriptors.py` (GFN2-xTB), `physics_cache_io.py`, `precompute_md.py`,
caches in `IAJD_master/bundles_caches/physics/`. Extend these; match their style
and the cache/audit pattern.

---

## 2. THE PHYSICS (what to compute and why it correlates with delivery)

Three determinants quantitatively track measured escape/transfection. Build one
module per determinant.

| Det. | Quantity | Why it correlates | Key refs |
|---|---|---|---|
| **A** | **Apparent pKa** + bilayer-binding ΔG(pH) | Near-neutral at blood pH 7.4, maximally charged in pH-5.5 endosome; *the* best-validated correlate (Jayaraman optimum) | Jayaraman 2012; Semple 2010 |
| **B** | **Packing parameter CPP** & **monolayer spontaneous curvature c₀** | Small-head/wide-tail "wedge" (CPP>1) → negative c₀ → drives Lα→H_II | Hafez–Cullis 2001; Israelachvili |
| **C** | **H_II / non-bilayer propensity** in the ionizable + *anionic* endosomal-lipid mixture | Lα→H_II transition on LNP-endosome fusion ruptures the membrane; DOPE potentiates, DOPC/PEG inhibit | Hafez–Cullis 2001; Ramezanpour–Tieleman 2022 |

**Coupling law (must enforce):** apparent pKa and conicity are **not independent**
— altering tails to change c₀ shifts apparent pKa by up to several units
(PNAS 2024). Any analog proposal MUST recompute A *and* B/C together; never report
a curvature change without its pKa consequence.

---

## 3. MODULES TO BUILD

Each module spec = {inputs · method · protocol · observables · validation ·
CPU cost · no-proxy failure mode · cache keys}. Implement as standalone, resumable,
cached drivers mirroring `precompute_md.py`.

### Module E (do FIRST) — Coarse-grained mapping of the IAJD architecture
*This is the pioneering step; no published Janus-dendrimer Martini model exists.*
- **Inputs:** SMILES → atomistic 3D (RDKit ETKDGv3, already available).
- **Method:** Map to **Martini 3** beads: gallic-acid aromatic core → TC5/TC4 ring
  beads; ester/ether linkers → standard Martini-3 linker beads; branched
  2-ethylhexyl tails → C1/C2 with the correct branch topology; **H2EPRZ
  hydroxyethyl-piperazine head → titratable bead set from the Martini-3 ionizable-
  lipid library** (`github.com/Martini-Force-Field-Initiative/M3_Ionizable_Lipids`,
  JCTC 2026). Prefer `martinize2`/the library's building blocks over hand mapping.
- **Observables:** the `.itp` topology + a mapping document.
- **Validation (mandatory before any A/B/C use):** run a short atomistic reference
  (CHARMM36 or GAFF, ~50–100 ns, single molecule in water + small bilayer patch)
  and require the CG model to reproduce: (i) radius of gyration ±15%, (ii)
  water–bilayer partitioning depth qualitatively, (iii) head-group solvation
  ordering. Log all three. If the CG model fails, **flag and stop** — do not paper
  over a bad mapping.
- **CPU:** AA reference is the costliest single step here (~100 ns × small system,
  hours–1 day on 6 cores). One-time per architecture (reuse across the GA-Tris family).
- **No-proxy failure:** mapping failure → the molecule's A/B/C are `NaN`,
  audit `cg_mapping_failed`.

### Module A — Apparent pKa & ΔG(pH) (titratable-MARTINI constant-pH MD)
- **Method:** Titratable Martini 3 (Grünewald 2020) — explicit proton beads,
  Grotthuss-like hopping, pH as external variable. Embed ONE CG IAJD in a model
  bilayer patch (start neutral POPC; then repeat in an **endosomal-mimic** patch:
  POPC/POPE/anionic BMP(LBPA)-or-POPS/cholesterol).
- **Protocol:** constant-pH MD scanning **pH 3.0→8.0 in 0.5 steps**; ≥ a few
  hundred ns per pH window (Martini effective time); compute fraction protonated
  ⟨q⟩(pH); fit Henderson–Hasselbalch → **apparent pKa = pH at ⟨q⟩ = 0.5**. Also
  extract partitioning/binding ΔG at pH 7.4 and pH 5.0 (depth + PMF-lite or
  insertion free energy from biased pulling if needed).
- **Observables:** `phys_apparent_pKa`, `phys_dGbind_pH74`, `phys_dGbind_pH50`,
  `phys_charge_pH74`, `phys_charge_pH50`, titration curve.
- **Validation:** reproduce **DLin-MC3-DMA apparent pKa ≈ 6.4** (and ideally
  DODAP ≈ 6.6, ALC-0315) to **±0.5 unit**. Tolerance for the piperazine head is
  the well-reproduced class (not the strong-primary-amine ~1.0-unit class).
- **CPU:** cheap–moderate (single molecule + bilayer patch, CG). Days, feasible.
- **Optional rigor escalation:** atomistic **CpHMD** (CHARMM36, ~µs, Böckmann
  2024 protocol) for a final-candidate cross-check — expensive but CPU-feasible
  given no time limit.
- **No-proxy failure:** non-convergent titration → `NaN`, audit `cpH_nonconverged`.
- **Note on existing xTB:** GFN2-xTB gives the *intrinsic* (gas/implicit-solvent)
  protonation energetics already (`qm_Ehedup`, `qm_q_ionizableN`, ΔGsolv). Keep
  it as a cheap first-pass / sanity anchor, but the **membrane environment shifts
  intrinsic→apparent by up to 3.5 units**, so the apparent pKa MUST come from the
  membrane-embedded CG/CpHMD run, not xTB alone. (Do NOT apply any fixed additive
  intrinsic→apparent correction — that was explicitly refuted.)

### Module B — Packing parameter & spontaneous curvature
- **CPP:** CPP = v/(a·l). v (tail volume) and l (tail length) from Tanford
  (already in `precompute_md._tanford_volume_length`); **a (head area)** from the
  Martini-3 bilayer area-per-lipid or the existing 3D head-area. Report CPP and its
  components.
- **c₀ (the rigorous one):** monolayer spontaneous curvature via the **first
  moment of the lateral pressure profile** of a flat, tensionless bilayer of the
  (protonated) IAJD or IAJD:helper mix — c₀ = −(1/κ)∫ z·[p_L(z)−p_N(z)] dz over a
  monolayer. Compute the lateral pressure profile with a local-stress build
  (GROMACS-LS or MDStress) on a tensionless Martini bilayer. Alternative/confirmation:
  build curved geometries with **BUMPy** (Boyd 2018; vesicle/cylinder/buckle,
  populated at the **pivotal plane**) and find the zero-bending-stress curvature.
- **Observables:** `phys_CPP`, `phys_c0_pressureprofile`, `phys_a_head`,
  `phys_v_tail`, `phys_l_tail`.
- **Validation:** reproduce sign/order — **DOPE c₀ strongly negative
  (~ −1/3 nm⁻¹), DOPC ≈ 0** — before trusting IAJD values.
- **CPU:** cheap (one flat bilayer + pressure profile). Hours.
- **No-proxy failure:** unstable bilayer / non-tensionless → `NaN`,
  audit `c0_pp_failed`.

### Module C — H_II / non-bilayer propensity
- **Method:** Martini-3 self-assembly (reuse `run_selfassembly.py`) of a mixture:
  **protonated IAJD + anionic endosomal lipid** (BMP/LBPA or POPS or POPG) ±
  cholesterol ± DOPE, at endosomal pH (head protonated). Long unbiased run; detect
  whether the system adopts **lamellar vs inverted (H_II / inverted-micellar /
  cubic)** topology.
- **Observables:** `phys_HII_score` (e.g., fraction of water in inverted channels /
  presence of a continuous inverted topology / negative-Gaussian-curvature
  metric), `phys_nonbilayer` (bool), aggregation/order parameters.
- **Validation:** the **DOPE-containing** mix should go non-bilayer; the
  **DOPC/PEG-PE** mix should stay lamellar (Hafez–Cullis correlation). Reproduce
  this contrast before scoring IAJDs.
- **CPU:** moderate (self-assembly box; the existing infra handles it, with the
  trajectory-cleanup janitor for disk).
- **Optional rigor escalation:** atomistic CHARMM36 H_II characterization
  (~352 lipids × ~300 ns; Ramezanpour–Tieleman 2022; observables R_w water-core
  radius, d_hex lattice spacing) for a final candidate — expensive, CPU-feasible.
- **No-proxy failure:** non-equilibrated → `NaN`, audit `HII_nonequilibrated`.

### Module D (optional) — Insertion / pore PMF (umbrella sampling)
Only if A–C don't discriminate. Transbilayer or insertion PMF of the IAJD in a
model endosomal bilayer via umbrella sampling + WHAM. Most expensive; defer.

---

## 4. VALIDATION & CALIBRATION (the part that makes it "winning, not hand-wavy")

**4a. Per-module benchmark gate.** No module's output enters design until it passes
its §3 validation (DLin-MC3-DMA pKa ±0.5; DOPE/DOPC curvature sign; DOPE/DOPC H_II
contrast). Store a `validation_report.json` with the benchmark numbers. A failing
module's outputs stay `NaN`.

**4b. Data-grounded calibration to 369 (critical — spleen optimum is unknown).**
The famous apparent-pKa optimum (6.2–6.5) is **liver/ApoE/siRNA-specific and does
NOT transfer to spleen-tropic 369.** Therefore DO NOT import a target pKa. Instead:
1. Compute A+B+C for a **stratified panel of the GA-Tris family** spanning the
   measured flux range (top fluxers incl. 369, mid, low — pull from the bioactivity
   dataset, e.g. ~12–20 compounds, reuse the QM cache where present).
2. Find which physics axis (apparent pKa, c₀, CPP, H_II score) **discriminates high
   from low flux *in this dataset*** (simple correlation/visualization; n is small,
   so report effect sizes + honest CIs, not p-hacked significance).
3. That data-derived axis — not a literature number — defines the **direction to
   push.** If, say, more-negative c₀ tracks higher spleen flux in-family, design
   toward more-negative c₀.

This turns the spleen-mechanism unknown into an *empirical, in-domain* target.

---

## 5. THE DESIGN LOOP (how to beat 369) — find the OPTIMUM, don't extrapolate the extreme

> Full, generalizable methodology: **`docs/DESIGN_OPTIMIZATION_PROTOCOL.md`**.
> The mechanistic levers are **optima, not monotones** (apparent pKa ~6–6.5; c₀/CPP
> have sweet spots), so the question is *"where is the peak and is 369 on the right
> side of it,"* never *"push X past 369."* Do **not** use a high-capacity model
> (XGBoost/NN) to locate optima — at this n it overfits, gives no calibrated
> uncertainty, and step-functions can't resolve a smooth peak.

1. **Baseline.** Compute 369's coordinates: (apparent pKa, ΔG(pH), CPP, c₀, H_II score).
2. **Family-stratified map.** Place the GA-Tris panel (and other families) in physics
   space vs measured flux, with **family as a stratum/covariate** so each physics lever
   is isolated from architecture confounds.
3. **Find the optima — GAM / response curves.** Fit a Generalized Additive Model
   `flux ~ s(pKa) + s(c₀) + s(CPP) + s(H_II) + family` (smooth per-variable terms);
   read each variable's response curve, its **peak**, and its confidence band. **Cross-
   check every data-peak against the mechanistic prior** (e.g. pKa peak should sit
   ~6–6.5; spleen may differ) — and **trust the mechanism over a data-peak that
   contradicts it** (your "369 flux may be mismeasured" failure mode). This is exactly
   how the pKa optimum was found experimentally (Jayaraman 2012).
4. **Surrogate + propose — Gaussian Process + Bayesian optimization.** Fit a GP
   (Matérn/RBF, ARD length-scales to reveal which variables matter, explicit noise term
   from replicates, family as covariate / multi-task) — small-n-robust with **calibrated
   uncertainty**, so you know whether a predicted optimum is signal or noise near 369.
   Enumerate synthetically-plausible in-family moves on 369 (tail branch/length/saturation
   → v, c₀, pKa; head H2EPRZ variants → head area/hydration → c₀, pKa; linker ester/amide)
   as the candidate pool, then **rank by an acquisition function (Expected Improvement)**
   that balances *near the predicted optimum* against *where the GP is most uncertain* —
   that tells you which analog to compute/make next. Honour the coupling law (§2):
   recompute pKa whenever a tail move changes c₀.
5. **Score the top picks with full physics.** For the acquisition-ranked shortlist,
   recompute A+B+C from scratch (no surrogate shortcut) and re-rank.
6. **Confirm the winner** with rigor-escalation runs (atomistic CpHMD pKa + CHARMM36 H_II).
7. **Deliverable:** a ranked shortlist; for each — the computed physics, **where it sits
   vs each variable's optimum**, the GP mean ± uncertainty, the mechanistic rationale, and
   the honest caveat (small n; the surrogate *refines* the physics prior, never overrides it).

**The hierarchy (never invert it):** mechanism (prior) → GAM/curves (does the data agree?)
→ GP/BO (quantify + propose next) → full physics evaluation (confirm). A model optimum
that contradicts the mechanism is a flag for *measurement error*, not a discovery.

---

## 6. ENGINEERING / INFRASTRUCTURE

- **New deps (CPU, conda-forge / pip):** `martinize2` (vermouth), the M3 ionizable-
  lipid library (git), BUMPy (pip/git), a local-stress build for pressure profiles
  (GROMACS-LS or MDStress), titratable-Martini parameters + scripts (cgmartini.nl).
  GROMACS 2026 + xTB already installed. Add these to `setup_physics_env.sh` and
  `requirements-offline.txt`; pin versions.
- **Drivers:** one per module (`compute_apparent_pka.py`, `compute_curvature.py`,
  `compute_hii.py`, `cg_map_iajd.py`), each resumable, cached to
  `bundles_caches/physics/design/`, audited, with the trajectory **janitor** pattern
  to bound disk.
- **Orchestrator:** `physics_design.py` — runs E→A→B→C for a SMILES, assembles the
  coordinate vector, writes `design_cache.csv`.
- **Thermal/CPU:** reuse the `nice`/worker-cap/`caffeinate`/janitor scaffolding from
  the QM run; never exceed the fanless-Air thermal budget; checkpoint every run.
- **No-proxy contract:** identical to the existing pipeline — real value or `NaN` +
  audit; downstream consumers handle `NaN`.

---

## 7. DELIVERABLES (definition of done)

1. Validated CG mapping of the GA-Tris/H2EPRZ architecture (+ AA-reference report).
2. Three validated modules (A/B/C) each passing its benchmark gate, with
   `validation_report.json`.
3. **369 fully characterized** in physics space.
4. GA-Tris family map + the identified escape-discriminating axis (calibration).
5. A **ranked shortlist of analogs predicted to exceed 369**, each with full
   computed physics + mechanistic rationale + honest uncertainty.
6. Final-candidate confirmation via atomistic escalation runs.
7. A written report tying every number to a benchmark and a citation, with §8
   caveats stated.

---

## 8. RISKS & HONEST GAPS (state these in every report)

1. **Spleen mechanism is unsettled.** Quantitative pKa/H_II evidence is
   liver/ApoE/siRNA. 369 is spleen-tropic (ApoE-independent). Mitigation: §4b
   in-domain calibration, not imported optima.
2. **H_II escape is the leading hypothesis, not law.** Competing models
   (wedge-disruption, vesicle budding-and-collapse, proton-sponge, fusion-pore);
   matched-pKa LNPs can differ in efficacy. Treat H_II score as a strong but
   non-exclusive correlate; that's why we compute three axes, not one.
3. **No published Janus-dendrimer membrane CG model.** Module E is genuinely novel
   and is the largest validity risk; its AA-reference gate is mandatory.
4. **Accuracy ceiling.** Titratable-Martini pKa ~0.5 unit; CG curvature/H_II is
   qualitative-to-semiquantitative. These RANK and reveal mechanism; they are not
   exact-pKa or exact-flux predictors — consistent with the qualitative-improvement
   goal.

---

## 9. CITATIONS (verified, primary)

- Jayaraman et al., *Angew. Chem. Int. Ed.* 2012, 51:8529 — apparent-pKa optimum 6.2–6.5 (PMID 22782619).
- Semple et al., *Nat. Biotechnol.* 2010, 28:172 — mechanism-guided lipid design DLinDMA→KC2 (PMID 20081866).
- Hafez & Cullis, *Adv. Drug Deliv. Rev.* 2001, 47:139 — shape/CPP, H_II–transfection correlation (PMID 11311989).
- Ermilova & Swenson / Tesei et al., *PNAS* 2024, 121:e2311700120 — shape–pKa coupling; CG reactive-MC ionization (PMC10786277).
- Grünewald et al., *J. Chem. Phys.* 2020, 153:024118 — titratable Martini 3 (DOI 10.1063/5.0014258).
- Trollmann, Rossetti & Böckmann, bioRxiv 2024 / 2026 — membrane CpHMD apparent pKa of aminolipids.
- Kjølbye, Bruininks, Souza, Marrink et al., *JCTC* 2026, 22:1069 — Martini-3 ionizable-lipid library (PMID 41407294; github M3_Ionizable_Lipids).
- Boyd & Swanson, *JCTC* 2018, 14:6642 — BUMPy curved-membrane builder; pivotal plane.
- Ramezanpour & Tieleman, *Langmuir* 2022, 38:7462 (+ 2020, 36:6668) — atomistic H_II of ionizable lipid mixtures vs SAXS/NMR (PMC9220946).

---

## 10. SUGGESTED EXECUTION ORDER (CPU-budget-aware)

1. Install deps; build + **validate** Module B (curvature) on DOPE/DOPC — cheapest,
   proves the pressure-profile machinery.
2. Build + **validate** Module A (apparent pKa) on DLin-MC3-DMA — the highest-value
   determinant.
3. Build + **validate** Module C (H_II) on DOPE-vs-DOPC mixes.
4. Build + validate Module E (CG map) on 369 vs an AA reference.
5. Characterize 369 (A+B+C). 6. Family-map calibration (§4b). 7. Generate/score
   analogs. 8. Escalation confirm the winner. 9. Report.

*Build B and A first: they gate everything and are the cheapest to validate.*
