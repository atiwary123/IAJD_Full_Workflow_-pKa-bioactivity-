# T4 #10 — MARTINI coarse-grained MD: deliberately skipped

**Status: not implemented. This document explains *why* and *what would be
needed* if we revisit it.**

## What it would have been

The original deferred-improvements list included a "MARTINI MD" feature
block: for each IAJD candidate, build a coarse-grained molecular-dynamics
simulation of the dendrimer's interaction with a lipid bilayer (using the
MARTINI force field), extract real conformational ensembles (mean head-area,
distribution of bilayer-insertion depths, escape free-energy estimate from
umbrella-sampling along the membrane normal), and use those as Block F
features in the bioactivity v14 model.

The hypothesis: real conformational ensembles would replace the single
ETKDGv3+MMFF94 conformer head-area we currently use (`head_area_3d.py`),
yielding a more honest physics signal for amphiphile packing and
endosomal escape — especially for novel/OOD candidates where one
conformer is misleading.

## Why it's skipped

This pipeline is **strict no-proxy** (project policy, reaffirmed
2026-05-29). For MARTINI MD to live in the codebase honestly, every
candidate scored must run a real MD trajectory. There is no faithful
shortcut — pulling fewer frames or shorter trajectories silently moves us
back into proxy territory.

A real MARTINI per-candidate workflow needs:

| Requirement | What we have | Gap |
|---|---|---|
| GROMACS 2024+ install | not installed; not in the venv | install GROMACS + plumed |
| MARTINI 3 topology generator | none in this codebase | install `martinize2` |
| Pre-equilibrated bilayer template | none | build POPC/POPG bilayer once, store as .gro |
| Atomistic → CG mapping for each candidate | depends on per-fragment martinize rules; some IAJD heads (HPRZ, H2EPRZ, DEHPRZ) aren't in the standard MARTINI library | author custom CG topologies |
| Compute time | n_candidates × (~2–6 h per trajectory on CPU; ~20 min on a single GPU) | needs a real cluster or persistent GPU |
| Convergence checks per candidate | none | implement RMSD plateau + insertion-depth distribution sanity checks |
| Storage for trajectories | none | ~100 MB per candidate × thousands |

Skipping the work-and-pretending path (random feature values from a 1 ns
trajectory) is the only honest option.

## What's there instead

The pipeline already has a **real** 3D-derived head area
(`head_area_3d.head_area_for_group`): ETKDGv3 + MMFF94 conformer
generation, principal-axis-perpendicular plane projection, van der Waals
disk convex hull, in nm². This is a real per-molecule single-conformer
quantity. It is the right physics for the *minimum* projection area at
the bilayer interface; it does *not* capture ensemble effects.

`compute_cpp.py` and `physics_features.py` use this 3D area in the
geometric CPP formula. The result, `cpp_geometric`, is what
`_physics_quality_quick` (called by the proposer) consumes.

The honest characterization is that the physics block is a single-conformer
proxy *for the conformational ensemble*, but every input scalar in it (head
area, tail volume, tail length, head pKa, charge density) is real
per-molecule computation — none are family medians or constants.

## What it would take to revisit

A reasonable phase-in for MARTINI MD without violating no-proxy:

1. Install GROMACS 2024 + `martinize2` + a templated bilayer system.
2. Author CG topologies for the IAJD head/linker/tail fragments missing
   from the standard MARTINI 3 library (HPRZ, H2EPRZ, DMA-Tris, etc.).
3. Build a `martini_runner.py` (analogous to `molgpka_runner.py`):
   one-shot subprocess that takes a SMILES, builds CG topology, runs
   1 µs (or shorter, justified by convergence diagnostics) of unbiased
   MD against the bilayer template, writes per-frame insertion depth
   and head-area distributions to JSON.
4. Wire it into `_extend_caches_live` alongside LION + ADMET so novel
   SMILES get their MD features computed before XGBoost scoring. Cache
   keys = canonical SMILES; cache values = real ensemble statistics.
5. Add Block F (n×k MD features) to `bioact_v14_pipeline.py` and retrain
   the v14 + ensemble + per-organ bundles.
6. Regression-test that MD features for known IAJDs match published
   coarse-grained results (e.g. Webb/Klein/Schmid IAJD CG papers, if
   any have been reported).

Until step 1 has a real GROMACS install in the runtime environment, this
feature stays out of the production codebase. The honest version is too
infrastructure-heavy to ship as a backend addition; the dishonest version
(a fake MD trajectory) violates the no-proxy rule.
