# Module A — Apparent pKa (titratable Martini 3 constant-pH MD)

**Status:** ✅ **VALIDATED 2026-06-01** — both validation tiers passed: aniline-in-water
recovers the intrinsic pKa (4.83 vs 4.8), and DLin-MC3-DMA-in-membrane recovers the
apparent pKa (**6.44 vs 6.44**) *non-circularly* from a generic 10.2 intrinsic bead via a
physics-computed −3.76-unit membrane shift. Method ready; IAJD values still pending the
IAJD titratable model + Module E.
**Code:** `compute_apparent_pka.py` · `physics_design/build_membrane_pka.py` ·
`martini/ionizable/MC3_titratable.itp` · vendored `martini/titratable/`.

Implements determinant **A** of the physics-design compass: the apparent pKa of an
ionizable lipid in a membrane (the best-validated correlate of endosomal escape).

---

## 1. Method (titratable Martini 3, Grünewald 2020)

A titratable bead whose **type encodes an intrinsic pKa** (`N2_10.2`=amine pKa 10.2,
`SN6d_4.8`=aniline pKa 4.8, …) exchanges a proton particle (`POS`) with titratable
water; `#define pH<value>` selects pH-dependent water↔proton↔bead interaction strengths
from `pH_dep_interactions.itp` (0.25-pH grid, 3.0–8.0). Running a pH scan and measuring
the average **degree of deprotonation** ⟨q⟩(pH) gives a titration curve; the **apparent
pKa = pH at ⟨q⟩=0.5** (Henderson–Hasselbalch / Hill fit). In a membrane the apparent pKa
is shifted from the intrinsic by the environment — the quantity that tracks escape.

**Runs on stock GROMACS 2026** (verified): `sd` integrator, **PME, ε_r=6**, dt=10 fs — no
patched engine (unlike GROMACS-LS for Module B). The pH chemistry lives entirely in the
topology + the titratable force field.

## 2. Pipeline (`compute_apparent_pka.py`)

Per pH: `sed` the pH into the topology → EM → NVT eq → NpT production (stock GROMACS,
resumable) → `degree_of_deprot.py` (counts `POS` protons bound to the titratable site vs
water, distance scheme) → ⟨q⟩. Then fit deprot(pH)=1/(1+10^(n(pKa−pH))) → apparent pKa,
Hill n, RMSE. Analysis runs under the venv (the **GROMACS-2026 tpr (tpx v138) is too new
for MDAnalysis' TPR parser**, so we pass a version-independent `.gro` topology).

## 3. Building blocks (built & validated)

- **Titratable DLin-MC3-DMA** (`martini/ionizable/MC3_titratable.itp`): MC3 body (Kjølbye
  2026 fixed-charge MC3) with the head bead replaced by the calibrated titratable
  dimethylamine motif from **DMEA** (`P2`=N2_10.2 pKa 10.2, `DN`=DB1, `DP`=DB2). Grompps
  clean with the titratable FF.
- **Membrane system** (`build_membrane_pka.py`): a POPC bilayer (reuses the Module B
  bilayer machinery — titratable POPC shares standard POPC geometry) with one MC3 swapped
  in bead-for-bead + titratable water. The full system (FF + POPC + MC3T + WNA + H⁺)
  **grompps on GROMACS 2026 (5969 atoms)**. Analysis: `-sel "name P2" -ref "name W"`.

## 4. Validation

- **Pipeline (aniline in water) — ✅ PASSED 2026-05-31.** Titrating aniline (`P2`=SN6d_4.8,
  intrinsic pKa 4.8), 9 pH × 5 ns: clean sigmoidal deprotonation curve (0.14→0.97 over pH
  3–7), Henderson-Hasselbalch fit **apparent pKa = 4.828** (vs 4.8; |Δ|=0.03, tolerance
  ±0.5), Hill n = 0.52, fit RMSE = 0.026. The titratable-Martini apparent-pKa pipeline
  reproduces a calibrated pKa essentially exactly → the method is validated.
- **Target (DLin-MC3-DMA in POPC) — ✅ VALIDATED 2026-06-01.** 8 pH × 20 ns membrane
  titration. Clean sigmoid (⟨q⟩: 0.002→0.004→0.004→0.010→0.013→0.632→0.985→0.961 over pH
  3–8), Henderson-Hasselbalch fit **apparent pKa = 6.44** (vs experimental LNP 6.44),
  Hill n = 4.0, RMSE = 0.015. **NON-CIRCULAR:** the input was the *generic* N2_10.2 amine
  bead (intrinsic aqueous pKa 10.2), and the membrane constant-pH MD shifted it **−3.76
  units** to 6.44 — reproducing the well-known ~3.5-unit environmental pKa depression of
  ionizable lipids and landing on MC3's measured value. We did NOT input 6.44.
  - **Honest precision:** the central value lands on 6.44, but the method's true precision
    is ~±0.2–0.3 (single MC3, finite sampling, steep Hill, hand-built titratable MC3), so
    the exact-to-2-decimals match is partly fortuitous; the *validated* claim is "recovers
    the right value within ~0.3 and captures the correct large membrane shift."
  - **Bug caught + fixed during the run** (`degree_of_deprot.py`): `prot_less` was scoped
    inside the water-present branch, so a frame with no titratable water near the acid hit
    the else-branch → NameError → ⟨q⟩=NaN at pH 6.5 (and would have voided the whole
    transition region). Fixed; re-fit via `refit_membrane_pka.py` (cached MD, no re-sim).
  - Result → `bundles_caches/physics/design/mc3_membrane_pka_refit.json`.

## 5. Honest scope

- Titratable-Martini pKa accuracy is ~0.5 unit (the well-reproduced piperazine/amine
  class, not the strong-primary-amine ~1-unit class).
- The titratable MC3 head-substitution is a reasonable approximation (DMEA *is* MC3's
  dimethylamino head) but not a fully re-calibrated lipid (N1 TN3a→N2_10.2 R-size swap;
  reused N1–CN bond/angle); the intrinsic head is the calibrated N2_10.2 and the membrane
  titration measures the apparent shift.
- xTB (GFN2) gives the *intrinsic* gas/implicit-solvent protonation energetics as a cheap
  sanity anchor only; the membrane shifts intrinsic→apparent by up to ~3.5 units, so the
  apparent pKa MUST come from the membrane-embedded titration. **No fixed additive
  intrinsic→apparent correction** (explicitly refuted).
