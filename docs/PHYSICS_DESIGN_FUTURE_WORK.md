# Physics-Design — Downstream / Future Work

> Notes for work to do AFTER the IAJD-369 pipeline is fully finished. Do **not** start
> these until the 369 run is complete (Modules A/B/C + CG-map + characterize-369 +
> family calibration §4b + analog generation/scoring + escalation-confirm).

---

## FW-1 — Replicate the whole pipeline on a higher-powered target family

**Ask (user, 2026-05-31):** once the 369 experiment is done, run the *exact same*
physics-design pipeline for the **highest-flux candidate** in a **larger family that has
the highest apparent-pKa ↔ flux correlation** (ideally larger n).

**Why this is worth doing (and complementary to 369).** IAJD-369 is GA-Tris and
**spleen-tropic**, where the mechanism is unsettled and the famous liver/ApoE apparent-pKa
optimum (6.2–6.5) does *not* transfer — so the apparent-pKa axis may be a *weak*
discriminator in-family and the §4b calibration is statistically thin. A second target
chosen for **statistical power (largest n)** and the **strongest in-family pKa↔flux
correlation** gives a cleaner, better-powered demonstration that the physics-design
compass actually works: the §4b in-domain calibration is robust, and the apparent-pKa
lever (Module A — the best-validated escape correlate) is unambiguous, so the design loop
has a clear, data-backed direction to push.

**Procedure (downstream — no new code, just new inputs):**
1. From the bioactivity dataset (`IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx`), for
   **each family** compute (a) n, and (b) the in-family correlation between apparent pKa
   (Module A predicted, or the existing pKa feature as a first pass) and log10 flux. Report
   effect sizes + honest CIs (small n → no p-hacking).
2. **Select the family** maximizing strong |correlation| × adequate n (prefer larger n).
   Candidate families to score include GA-Tris, PE-Tris, PE-Gallic, and any others in the
   dataset's `architecture`/family vocabulary; the SAR prior
   ([[project-informed-mutation-sar]]) is a starting hint, not the answer.
3. **Target** = the highest-flux candidate in that family.
4. Run the SAME pipeline, parameterized by SMILES/family (no rebuild):
   - Module A apparent pKa (`compute_apparent_pka.py` + `build_membrane_pka.py`),
   - Module B c₀/CPP (`compute_curvature.py`),
   - Module C H_II (`compute_hii.py`, once built),
   - CG-map validation (Module E),
   - characterize the target in physics space,
   - §4b family calibration on the **larger** panel,
   - generate + score synthetically-plausible analogs of the target,
   - escalation-confirm the winner.
5. Because the pKa↔flux correlation is strong here, **apparent pKa (Module A) is likely the
   primary design lever** — push the target's analogs toward the in-family optimal apparent
   pKa (data-derived, NOT the imported liver 6.2–6.5), honoring the pKa↔curvature coupling.

**Reuse:** every driver is parameterized by SMILES/family, so this is new *inputs*, not new
*code*. The membrane builder currently hard-codes MC3 (`build_membrane_pka.build`,
`ionizable='MC3'`) and a POPC host — generalize it to take an arbitrary IAJD SMILES (CG-map
via Module E) and the endosomal-mimic host mix when extending.

**Status:** DOWNSTREAM. Blocked on the 369 pipeline finishing. No work started.

---

## FW-3 — panel ordering: compute the flux EXTREMES first (369 → lowest-flux GA-Tris)

**Decision (user, 2026-05-31):** after 369 (the HIGHEST-flux GA-Tris member), the *second*
IAJD to compute is the **lowest-flux GA-Tris member**, then compare the shifted coordinates
and sanity-check whether c₀/H_II/CPP moved in the mechanistically-expected direction.

**Why this is right:** it's the cheapest max-contrast **falsification** — same logic as the
DOPE-vs-DOPC extremes that validated Module B. Two flux-extreme points give the *sign of the
gradient* (does c₀ get more negative / H_II higher from low→high flux?) for ~2 molecules of
compute, before committing weeks to the full panel. Especially valuable for spleen-tropic
GA-Tris, where the mechanism is unsettled and the calibrator already found **no pKa lever**:
if even the flux extremes don't separate on c₀/H_II, that's an honest "not designable on
these axes" verdict we want for 2 molecules, not 20. It's also good GP/BO active-learning
ordering (most-informative points first).

**Third point = family MEDIAN-flux member (user, 2026-05-31).** High + low = a *line* (sign
of the gradient). Adding the middle tests the **SHAPE** — does the mid-flux point lie ON the
high–low line (monotone) or OFF it (curvature → a window/optimum)? This is the single most
important question per the optimization protocol ("levers are optima, NOT monotones — find
the peak, not the edge"), and 3 points is the minimum to fit a quadratic / see non-monotonicity.
Use the member nearest the **median** flux (robust to outliers) and reliably measured.

**Then quantile space-filling, then GP/BO (user, 2026-05-31).** After high/low/median it
doesn't much matter which order — a sensible default is **bisection by flux quantile**:
100th → 0th → 50th → 75th → 25th → 87.5th → 12.5th → … (progressively refines coverage of the
flux range, exactly what the GAM wants for even x-coverage). Once ~5–7 points are in, hand off
to the **GP/BO acquisition** (EI/UCB) to pick the most-informative next IAJD instead of the
fixed sweep. (Minor refinement: quantiles on flux are a fine first proxy; ideally you fill the
*descriptor* axis evenly — but you don't know c₀/H_II until you compute them, so flux-quantile
first, then GP/BO fills descriptor gaps.)

**Honest caveats / refinements:**
- n=2 gives *direction*, n=3 gives a *hint of curvature* — NEITHER is the calibration (locating
  a window precisely needs the panel + GAM). These are SANITY/SCREEN steps, not §4b.
- **Window detection needs BOTH shoulders.** If the lever is an optimum, the high fluxer sits
  near the peak and low fluxers are on *either* side of the physics axis. A single low fluxer +
  a median may both fall on one shoulder → you'd see a "line" and miss the window. So if the
  3-point screen hints at curvature, the 4th point should target the *opposite* physics-axis
  shoulder (a low fluxer on the other side), not just more flux-spread.
- With noisy flux, the mid-point being "off the line" can be measurement noise, not real
  curvature — weight by reliability and don't over-read 3 points.
- Pick the lowest-flux member that is **reliably measured** (low SEM, good replicate count —
  Step 0 reliability weighting), not just the numerically smallest flux, to avoid a
  detection-floor/measurement artifact masquerading as biophysics.
- The extremes likely differ in *several* structural ways at once → multiple axes may shift
  together; you can't isolate the single lever from 2 points (that's fine for a sanity check).
  A near-analog-of-369 low fluxer would give cleaner attribution but less contrast — do the
  max-contrast pair FIRST (see any shift at all), refine with near-analogs after.

## FW-4 — IAJD c₀ via the HOST METHOD (pure IAJD bilayer collapses) — found 2026-06-01

**Finding (first real IAJD Module B run, IAJD 369 neutral):** the pipeline runs end-to-end
with the de-overlap fix (no crash; force recompute **exact, 0.0002 %** vs GROMACS), BUT the
**pure-369 flat bilayer collapses** under the tensionless semiisotropic barostat: APL →
**0.265 nm²** (impossibly dense — one tail alone is ~0.2 nm²), thickness → **8.3 nm** (a
bilayer is ~4–5), `bilayer_intact=False`, γ=−9.6 mN/m, water not flat → `c0_physics_converged
=False`. The reported c₀=+1.55 nm⁻¹ is a collapsed-aggregate artifact and is correctly
flagged untrusted (NOT handed back as a real number).

**Interpretation (honest, and itself a signal):** 369 is the highest-flux GA-Tris member, a
3-tail dendritic amphiphile — i.e. strongly **non-bilayer-prone**. A pure flat bilayer of it
is not even metastable (unlike DOPE, which holds flat for ~100 ns and so passed Module B).
That non-lamellar propensity is *exactly* the membrane-disrupting property that drives
endosomal escape, so "the pure bilayer won't stay flat" is information, not just a failure.

**The fix — HOST METHOD (standard for H_II-formers):** measure 369's **spontaneous-curvature
contribution** by embedding it dilutely in a stable POPC host bilayer and using the
first-moment difference (or the host-area / Δc₀-per-mole-fraction extrapolation, as the field
does for DOPE/PE). Reuse `build_membrane_pka.py` (it already swaps ONE molecule into a POPC
bilayer) — drop the titratable conversion, keep the host bilayer, add 1–few IAJDs, run the
Module B pressure profile, attribute the curvature shift to the IAJD. This gives a TRUSTED
(modulo Module E) c₀ for non-bilayer IAJDs.

**Panel implication:** the `c0_physics_converged` flag honestly separates bilayer-stable
members (likely lower-flux → pure-bilayer c₀ measurable) from collapsing ones (likely
higher-flux → need the host method); the *bilayer-stability threshold itself* may track flux.
For collapsing members, **Module C (H_II / self-assembly)** is the complementary, more natural
characterization.

**Diagnostic result (start-APL sweep, 2026-06-01):** start APL 1.2 → final 0.265, thick 8.3,
bilayer collapsed, c₀=+1.55; start APL 0.65 → final 0.353, thick 5.9, **bilayer intact**,
c₀=+0.73. So the *total* collapse from 1.2 was largely a **barostat overshoot from a too-loose
start**, not purely fundamental. BUT even the intact 0.65 run does **not converge** (final APL
0.35 nm² is implausibly small for a 3-tail molecule — likely tail over-cohesion or
interdigitation — and γ is still off), and c₀ stays **positive**. Positive c₀ is either a
compression artifact OR a real inverted-cone geometry of 369's CG model (bulky dendritic head
+ short GA-Tris tails) — **a non-converged pure bilayer cannot distinguish these.** Verdict:
pure-IAJD-bilayer Module B is too initial-condition-sensitive and non-convergent to trust for
369; **do NOT chase it with more start-APL runs** — build the host method (and/or get a proper
equilibrium APL from a longer, area-relaxed or surface-tension-scanned run). The host method
also sidesteps the "what is the right area" question entirely (the host sets it).

## FW-2 (implied) — generalize the membrane-pKa builder to arbitrary IAJDs

`build_membrane_pka.py` currently swaps one MC3 into a POPC bilayer. For 369 and the FW-1
target, it needs: (a) the CG-mapped IAJD topology (Module E, the GA-Tris/H2EPRZ mapping)
with its ionizable head as a titratable bead, and (b) the **endosomal-mimic host** (POPC/
POPE/anionic BMP-or-POPS/cholesterol) rather than pure POPC. Same titration machinery.
