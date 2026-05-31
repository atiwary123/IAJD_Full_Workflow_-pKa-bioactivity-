# Physics-Design Protocol — mechanism ∩ data corroboration

**Canonical design/calibration protocol** for the IAJD physics-design compass. Supersedes
the looser §4b/§5 of `PHYSICS_DESIGN_BUILD_PROMPT.md` and governs BOTH the 369 run and the
downstream FW-1 family (`PHYSICS_DESIGN_FUTURE_WORK.md`). Implemented in `physics_calibrate.py`
(Steps 0–4 runnable now) and the design loop (Steps 5–7).

**The optimization *engine* is `docs/DESIGN_OPTIMIZATION_PROTOCOL.md`** (mechanism → **GAM**
response-curves to locate optima → **GP + Bayesian optimization** with calibrated
uncertainty to propose → **confirm**; NOT XGBoost/NN at this n). This document is the
IAJD-specific *instantiation* of that engine: it adds (a) the candidate axes + their
mechanistic priors, (b) explicit **flux reliability weighting** (data-truth is the master
confound), and (c) the **corroboration verdict** (act only where mechanism ∩ reliability-
weighted data agree). Where the two docs overlap, the optimization doc governs the method,
this doc governs the IAJD specifics. `physics_calibrate.py` implements: Step 0 reliability +
Step 1 GAM peaks + Step 4 composite direction + Step 2 (engine numbering) GP/BO surrogate.

## Core principle
We have **two unreliable oracles**: the *mechanism* (contested for spleen-tropic 369 — H_II
is a hypothesis, competing wedge/proton-sponge/fusion models) and the *data* (noisy, small-n,
data-truth issues). The protocol therefore **acts only where mechanism and data corroborate,
and scales confidence to the reliability of each** — beating both "trust mechanism &
extrapolate" (over-commits to one contested mechanism) and "fit the data" (fits noise).

## Steps
**0. Truth-weight the data FIRST.** Quantify per-compound flux reliability: replicate
counts/variance + a structural-twin self-consistency check (near-identical IAJDs that
disagree on flux define the noise floor). Output a reliability weight per point. Reliability
is the master variable — a "clean" family is one whose flux is trustworthy enough to test
*any* axis (a high pKa↔flux correlation is partly just a clean-data signal).

**1. Full coordinate vector, no cherry-picking.** For reference + a stratified panel
(~12–20, spanning the flux range), compute *all* candidate axes — apparent pKa (Module A),
c₀/CPP (Module B), H_II (Module C) — each with honest CG error bars. Each axis is a
competing hypothesis; never pre-select.

**2. Mechanism = a SET of falsifiable directional priors.** From biophysics, not the
dataset: pKa near-neutral-blood/charged-endosome window; more-negative c₀ / CPP>1; higher
H_II → more escape. These signs are what we test.

**3. Adjudicate leniently, reliability-weighted.** Per axis, test in-family sign-consistency
with its prior via reliability-weighted effect size + honest CI — **lenient** (sign +
non-trivial magnitude, NOT p<0.05; noisy data hides real mechanisms). Three verdicts:
mechanism∩clean-data agree → operative (high weight); mechanism holds but data too noisy →
keep at *reduced* weight (don't discard a true mechanism for lack of noisy support);
mechanism vs clean-data disagree → diagnose (low-reliability points vs wrong mechanism).

**4. Confidence-weighted composite direction, no single winner.** Weight each axis by
(prior strength × data support × data reliability). Where clean data + mechanism agree, that
axis dominates the design vector; elsewhere it contributes less but nonzero. This is the
explicit mechanism-vs-data resolution.

**5. Design in the COUPLED feasible region.** Enumerate synthetically-plausible moves on the
target, but recompute the *entire* vector per candidate (pKa↔curvature coupling: one move
shifts several axes); reject moves that leave a window on any supported axis. The "perfect
IAJD" maximizes the composite direction *inside* the coupled feasible region — a joint
optimum, not per-variable sweeps. Some axes are windowed (pKa: too-high and too-low bad),
others monotonic-until-a-limit (more-negative c₀).

**6. Rank by ROBUSTNESS (agreement), not a point score.** Confidence = agreement across
independent axes AND stability under κ/sampling/CG uncertainty. Favorable on several axes +
robust = high; favorable on one noisy axis = low. Ship a ranked shortlist with per-analog
uncertainty.

**7. Falsify before believing; earn trust on clean data.** (a) Retrodiction on held-out
data: can it rank known high-vs-low fluxers it didn't see? If not, don't trust forward
predictions. (b) Validate the whole machine on the FW-1 clean-data family *first* (where data
can adjudicate), THEN apply to noisy 369. (c) Escalate the top analog with an orthogonal
method (atomistic CpHMD / CHARMM36 H_II); independent agreement = final confidence.

## The honest failure mode (a feature, not a bug)
If even the clean family shows NO axis surviving Step 3, the escape determinant isn't in our
three coordinates (or the data can't support inference), and the correct output is **"we
cannot direct-design this yet"** — a ranked shortlist must NOT be manufactured. The protocol
is built to *return that verdict*. (Honesty is the top project value — see memory.)
