# Module A — Apparent pKa (titratable Martini 3 constant-pH MD)

**Status:** method verified on GROMACS 2026; driver + building blocks built & validated;
pipeline validation (aniline) running; DLin-MC3-DMA membrane titration system built,
validated (grompps), and launched (multi-day).
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
- **Target (DLin-MC3-DMA in POPC):** apparent pKa ≈ **6.44 ± 0.5** (build prompt §3;
  the membrane shifts the ~10 intrinsic down). Titration launched (8 pH × 20 ns,
  multi-day, chained after aniline). Result → `bundles_caches/physics/design/pka/`.

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
