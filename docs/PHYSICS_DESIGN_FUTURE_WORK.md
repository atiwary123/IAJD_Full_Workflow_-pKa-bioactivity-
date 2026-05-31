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

## FW-2 (implied) — generalize the membrane-pKa builder to arbitrary IAJDs

`build_membrane_pka.py` currently swaps one MC3 into a POPC bilayer. For 369 and the FW-1
target, it needs: (a) the CG-mapped IAJD topology (Module E, the GA-Tris/H2EPRZ mapping)
with its ionizable head as a titratable bead, and (b) the **endosomal-mimic host** (POPC/
POPE/anionic BMP-or-POPS/cholesterol) rather than pure POPC. Same titration machinery.
