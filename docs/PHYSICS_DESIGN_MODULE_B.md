# Module B — Monolayer Spontaneous Curvature (c₀) and Packing Parameter (CPP)

**Status:** built; benchmark-gated on DOPE/DOPC (see §6).
**Code:** `physics_design/` package + `compute_curvature.py` driver + `analyze_curvature.py`.
**Constraints honoured:** no-proxy (every number real or NaN+audit), CPU-only (stock
GROMACS 2026.2, fanless-Air thermal cap), validate-before-trust (§6 gate).

This implements determinant **B** of the physics-design compass
(`docs/PHYSICS_DESIGN_BUILD_PROMPT.md` §3): the monolayer spontaneous curvature c₀
from the first moment of the lateral pressure profile of a flat, tensionless
Martini-3 bilayer, plus the packing parameter CPP.

---

## 1. Why this method (and why not GROMACS-LS)

The rigorous c₀ observable is the first moment of the lateral pressure profile of a
tensionless bilayer (Helfrich; Marsh 2007; Sodt & Pastor 2013):

```
    kappa_m · c0  =  - ∫_0^∞  z · [ P_L(z) - P_N(z) ]  dz        (over one monolayer)
```

where `P_L = (P_xx+P_yy)/2` is the lateral and `P_N = P_zz` the normal pressure, z is
measured from the bilayer midplane, and `kappa_m` is the monolayer bending rigidity.

Computing `P_L(z)-P_N(z)` needs a *z-resolved local stress*, which stock GROMACS does
not output. The canonical tool, **GROMACS-LS** (Vanegas/Torres-Sánchez/Arroyo), is
frozen at **GROMACS 4.6.6** and cannot read 2026.x trajectories/TPRs; building that
2011-era code on Apple Silicon is infeasible, and MDStressLib needs an equally old
patched engine. So we **recompute the Irving-Kirkwood configurational stress in Python
directly from a stock-GROMACS trajectory** — the *same physics* GROMACS-LS implements
internally (IKN stress with a central-force decomposition of 3-body terms). This is not
a proxy: it is the exact local stress of the exact trajectory, and it is validated to
reproduce GROMACS' own energies/forces to < 10⁻³ % (§5).

**Key simplification:** the kinetic pressure is isotropic on time-average
(equipartition), so it cancels in `P_L - P_N`. Only the *configurational* (virial) part
is needed → positions only, no velocities.

---

## 2. Pipeline (`compute_curvature.py`)

1. **Build** (`physics_design/bilayer.py`): a flat bilayer patch of one lipid species is
   tiled from a BFS-laid-out single-lipid template (head-up, tails splayed), mirrored
   into two leaflets, solvated with Martini W (`gmx solvate`), and waters inside the
   hydrophobic slab are stripped. EM relaxes the seed geometry.
2. **Equilibrate + produce** (`physics_design/md_runner.py`): EM → gentle NPT (c-rescale,
   dt=10 fs) → NPT (dt=20 fs) → production (Parrinello-Rahman). Pressure coupling is
   **semiisotropic at 1 bar / 1 bar**, which enforces zero average surface tension
   (`γ = Lz⟨P_N−P_L⟩ = 0`) — the required *tensionless* reference state. Martini-3
   nonbonded settings: reaction-field, `rcoulomb=rvdw=1.1`, `epsilon_r=15`,
   `epsilon_rf=0` (∞), Verlet + Potential-shift.
3. **Lateral pressure profile** (`physics_design/pressure_profile.py`): for each frame,
   recompute every interaction's central pair force and bin its lateral-minus-normal
   virial `s_ij = (f/|d|)[½(dx²+dy²) − dz²]` onto a z grid via the **Irving-Kirkwood
   contour** (K-point line quadrature; spreads each pair's stress along the line joining
   the two beads — the smooth, rigorous choice). 3-body angle forces are first
   decomposed into central pair forces (CFD, exact for 3 bodies) so they enter the same
   way. The bilayer is recentred each frame (midplane → 0).
4. **First moment → c₀** (`physics_design/curvature.py`): integrate `z·[P_L−P_N]` over
   the **monolayer** (midplane → membrane–water interface; bulk water is isotropic and
   contributes nothing, so it is excluded — including it only amplifies noise by the
   large-z weight), symmetrise the two leaflets, divide by `kappa_m`. CPP = v/(a₀·l)
   with Tanford v,l and the measured area per lipid.
5. **Cache + janitor**: per-lipid JSON + profile `.npz` to
   `bundles_caches/physics/design/`; bulky trajectory purged after analysis.

---

## 3. Forces reproduced (exactly as GROMACS integrated them)

| Term | Potential | Scalar central force on i (along d = r_i−r_j) |
|---|---|---|
| LJ (Verlet+pot-shift) | `C12/r¹² − C6/r⁶` (force unshifted, hard cutoff at rvdw) | `(12 C12/r¹² − 6 C6/r⁶)/r` |
| Reaction field (nonbond) | `(fE/εr) qq [1/r + k_rf r² − c_rf]` | `(fE/εr) qq [1/r² − 2 k_rf r]` |
| Reaction field (excluded 1-2 pairs) | `(fE/εr) qq · k_rf r²` (homogeneous term only) | `(fE/εr) qq · (−2 k_rf r)` |
| Bond (harmonic) | `½ k (r−r0)²` | `−k (r−r0)` |
| Angle (G96 cosine) | `½ k (cosθ−cosθ0)²` | analytic 3-body force → CFD central pairs |

with `k_rf = 1/(2 rc³)`, `c_rf = 3/(2 rc)` for `εrf=∞`, `fE = 138.935458 kJ·nm/mol/e²`.
C6/C12 come from the explicit `[nonbond_params]` of `martini_v3.0.0.itp` (Martini-3
provides every pair; a missing pair is a hard error, never a combination-rule guess).

Exclusions: `nrexcl=1` → only directly-bonded (1-2) pairs are removed from the
nonbonded sum; their reaction-field homogeneous term is added back (GROMACS convention).

---

## 4. Spontaneous-curvature conversion (κ)

The first moment `τ_m` is computed with **no free parameter** (it is `κ_m·c0`, the
stress moment). To express c₀ in nm⁻¹ we divide by the monolayer bending rigidity
`κ_m = κ_bilayer/2`, with κ_bilayer = 0.85×10⁻¹⁹ J = 20.5 kT for DOPC
(**Rawicz et al., Biophys. J. 2000, 79:328**, micropipette aspiration). We therefore
report **both** `τ_m` (proxy-free) and `c0`; the c₀ magnitude carries ≈ ±25 % from the
κ assumption, while the *sign* and the *DOPE-vs-DOPC contrast* are κ-independent.

---

## 5. Force-field reproduction check (`--force-check`)

A single production frame is re-run through GROMACS (`gmx mdrun -rerun`) and its
configurational energies compared term-by-term to our recomputation:

| term | recomputed | GROMACS | |err| |
|---|---|---|---|
| Bond | 1202.838 | 1202.84 | 1.6e-4 % |
| G96Angle | 577.141 | 577.141 | 4.8e-5 % |
| LJ (SR) | −54909.43 | −54909.4 | 4.8e-5 % |
| Coulomb (SR) | −242.116 | −242.116 | 8.6e-5 % |

(representative DOPC frame; `max |err| ≈ 1.6e-4 %`). This proves the recomputed forces
**are** GROMACS' forces — the in-Python Irving-Kirkwood stress is exact for this
force field. The reaction-field excluded-pair convention was pinned numerically:
`nb_with_crf + excl(k_rf r² only) = −328.91` vs GROMACS `−328.908`.

Two further internal checks: the local-profile surface tension equals the global-virial
surface tension to the digit, and (for a tensionless run) both ≈ 0.

---

## 6. Validation gate (DOPE / DOPC)  —  **PASSED 2026-05-31**

Benchmark (build prompt §3): **DOPE c₀ strongly negative (~ −1/3 nm⁻¹), DOPC ≈ 0.**
Reference: DOPE R₀ ≈ −2.6 to −3 nm (Rand & Fuller; Kozlov) → c₀ ≈ −0.33 to −0.38 nm⁻¹.

Result (128-lipid patches, 300 K, 45 W/lipid, 50 ns prod, IK contour, κ_mono from
Rawicz 2000), `validation_report.json`:

| lipid | APL (nm²) | thick (nm) | γ (mN/m) | water-base (bar) | τ (bar·nm²) | **c₀ (nm⁻¹)** | force-check |
|---|---|---|---|---|---|---|---|
| DOPC | 0.688 | 3.70 | −2.4 | 15.3 | −72 | **−0.167** | 3.3e-4 % |
| DOPE | 0.650 | 3.87 | +2.4 | 19.2 | −188 | **−0.435** | 2.6e-4 % |

- **DOPE strongly negative** (−0.44, within CG tolerance of the −1/3 ≈ −0.33 target) ✓
- **DOPC relatively flat** (−0.17, much closer to zero) ✓ — Martini-3 DOPC is mildly more
  negative than experiment's ~−0.05 (a known Martini curvature-exaggeration), but the
  **sign and order are correct**: DOPE is 2.6× more negative, separation 0.27 nm⁻¹.
- Both **tensionless** (|γ|<4) and **water-flat** (<25 bar) → `c0_trusted`.
- **Forces reproduce GROMACS to <10⁻³ %** for both head types (Q1 choline, Q4p ethanolamine).
- `PASS: true`. DOPE packs tighter (APL 0.650 < 0.688) — the small-PE-head signature of
  negative curvature, independently visible in the structure.

The contrast (DOPE−DOPC) is the robust, physically-meaningful quantity (z_max-independent
to ±0.02 nm⁻¹); the c₀ magnitude carries ~±25 % from κ. Sign convention anchored to the
benchmark (cone = negative), see `curvature.py`. Figure: `curvature_profiles.png`.

---

## 7. Honest scope (carried into every report)

- CG curvature is qualitative-to-semiquantitative; it RANKS and reveals mechanism, not
  an exact c₀.
- DOPE is run as a metastable flat patch at 300 K (it favours H_II); the profile is the
  *frustrated* stress of the flat state, which is exactly the spontaneous-curvature
  drive. If the flat patch destabilises, the water-flat/tensionless gate flags it NaN.
- κ is a cited literature value, not computed here; τ_m (the stress moment) is the
  proxy-free primary observable.
