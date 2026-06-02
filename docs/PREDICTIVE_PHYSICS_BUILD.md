# Predictive-Physics (Block D′) — Build, Recovery & Operations

**Last updated:** 2026-05-31
**Owner:** atiwary1@sas.upenn.edu
**Policy:** strict no-proxy. Every feature is either a *real* computed value or
`NaN`. Never a constant, family-median, or fabricated number. `NaN` is honest —
XGBoost routes it through its default branch.

---

## 0. TL;DR — rebuild after a reboot

A macOS reboot wipes `/tmp`. The original engines lived there, so they vanish on
restart. To restore everything:

```bash- 
cd ~/Downloads/IAJD_FULL_WORKFLOW_CONDENSED-3
bash setup_physics_env.sh          # reinstall xtb + gromacs (persistent now)
./.venv/bin/python physics_status.py   # confirm green
bash run_physics_overnight.sh      # resume the build where the caches left off
```

That is the entire recovery. The caches in `IAJD_master/bundles_caches/physics/`
are the source of truth and are never lost on reboot (they live in the repo, not
`/tmp`); only the *engines* need reinstalling.

---

## 1. What Block D′ is

A first-principles physics feature block for the bioactivity-v14 model. For each
IAJD it produces, from real simulation:

- **QM (GFN2-xTB)** — 8 electronic-structure descriptors: partial charge on the
  ionizable N, dipole, polarizability, HOMO–LUMO gap, ALPB(water) solvation free
  energies (whole / head / tail), and the protonation electronic ΔE.
  → `qm_descriptors.py`, keys `QM_KEYS`.
- **CG-MD (MARTINI 3 self-assembly)** — ~13 ensemble observables from real
  GROMACS trajectories in both protonation states: whether it assembles, aggregate
  number, head areas (neutral/prot/Δ), bilayer thickness, order parameter, radius
  of gyration, water penetration, CPPs, spontaneous curvature.
  → `martini/`, `precompute_md.py`, keys `MD_KEYS`.
- **Helfrich escape** — endosomal-escape ΔG term derived in
  `physics_cache_io.load_physics` from the MD curvature/area observables.

These flow into the model as the `qmmd` head (`adaptive_stacker.py`) and as
Block D′ in `IAJD_master/code/bioact_v14_pipeline.py`. The rationale for it being
a real ensemble rather than a single-conformer proxy is in
`docs/T4_10_MARTINI_MD_skip.md` (the original "skip" note — now superseded by
this real build).

---

## 2. Why it broke on 2026-05-31 (root cause)

The original build installed the engines with micromamba **under `/tmp`**:

```
DEFAULT_XTB_CMD = "/tmp/bin/micromamba run -n xtb_env --root-prefix /tmp/mamba_root xtb"
```
(`qm_descriptors.py`, and mirrored in `physics_pka_v1/run_physics_pipeline.py`).

macOS clears `/tmp` on reboot. After the restart, `xtb` and `gmx` were gone, so
`_xtb_available()` / `gromacs_available()` returned `False` and the pipeline
silently degraded to NaN. **Nothing computed was lost** — only the binaries.

**The fix (this rebuild):** both engines now live in persistent locations and the
code points at them. `/tmp` is no longer in the path.

---

## 3. The environment

| Component | Where it lives now (persistent) | Resolved in code by |
|---|---|---|
| `xtb` 6.7.1 | micromamba env `xtb_env` at `~/micromamba` (conda-forge) | `qm_descriptors.DEFAULT_XTB_CMD` |
| `gromacs` 2026.2 (`gmx`) | Homebrew `/opt/homebrew/bin/gmx` | `run_selfassembly._resolve_gmx_cmd` (via `which gmx`) |
| `micromamba` 2.6.2 | Homebrew `/opt/homebrew/bin/micromamba` | — |
| Python stack | `./.venv` (py 3.11): rdkit, MDAnalysis, xgboost, pyarrow, sklearn, scipy | — |

Overrides: set `XTB_CMD` and/or `GMX_CMD` env vars to point elsewhere without
editing code.

### Reconstructed install recipe (`setup_physics_env.sh` automates this)

```bash
brew install gromacs micromamba
micromamba create -y -r ~/micromamba -n xtb_env -c conda-forge xtb
# verify:
micromamba run -n xtb_env -r ~/micromamba xtb --version   # -> xtb version 6.7.1
gmx --version                                              # -> GROMACS 2026.2
```

**Honest gap:** the *exact* one-liner originally typed to build the `/tmp` env was
never logged (the setup log only captured the pKa run, not the conda install).
The recipe above is a functionally identical reconstruction, verified to produce
working `xtb` 6.7.1 + `gromacs`. The full original conda spec (which also listed
`tblite`, `crest`, `xtb-python`, `vermouth`) is preserved in
`requirements-offline.txt`; the actual precompute drivers only shell out to
`xtb` and `gmx`, so those two are sufficient. `crest`/`tblite` are not invoked by
the current code path (RDKit ETKDGv3 does the conformer search).

---

## 4. Pipeline & data flow

```
IAJD_Bioact_v13_clean.xlsx  (271 unique canonical SMILES)
        │
        ├─ precompute_qm.py ──────► qm_cache.csv/.parquet     (8 QM descriptors / SMILES)
        │     uses qm_descriptors.compute_qm_descriptors (real xtb)
        │
        ├─ run_md_continuous.py ──► md_cache.csv/.parquet      (~13 MD observables / SMILES)
        │     uses precompute_md.run_one_compound
        │       → martini.build_cg.build         (CG topology)
        │       → martini.run_selfassembly.run   (real gmx: insert→solvate→EM→equil→2µs prod)
        │       → martini.analyze_md             (observables from trajectory)
        │       → trajectory deleted after extraction (disk-bounded; see §6)
        │
        ▼
physics_cache_io.load_physics(smiles, head_group, pka)
        cache hit → real values ; miss → qm_emulator/md_emulator ; both gone → NaN
        + computes Helfrich ΔG_escape
        │
        ├─ adaptive_stacker.py --build-qmmd-block ─► qmmd_features_v14_train.npy
        ├─ train_qmmd_head_only.py ───────────────► qmmd_head.joblib  (logs CV-MAE)
        ├─ train_qm_emulator.py  (cache ≥ 30) ────► qm_emulator.joblib
        └─ train_md_emulator.py  (cache ≥ 30) ────► md_emulator.joblib
        │
        ▼
bioact_v14_pipeline.py  Block D′  →  v14 model + ensemble + per-organ bundles
```

`auto_retrain_watcher.py` watches the two cache row-counts and re-runs the four
training steps whenever either grows by `--threshold` rows (default 3), logging
each event to `auto_retrain_log.csv`.

---

## 5. Running it

| Goal | Command |
|---|---|
| **Full overnight build (detached)** | `bash run_physics_overnight.sh` |
| Smoke-scale overnight | `QUICK=1 bash run_physics_overnight.sh` |
| QM only, resume (sequential) | `./.venv/bin/python precompute_qm.py --resume --skip-correlations` |
| **QM parallel (recommended, ~3× faster)** | `QM_THREADS=2 ./.venv/bin/python precompute_qm_parallel.py --workers 4` |
| MD only, production | `./.venv/bin/python run_md_continuous.py` |
| MD only, smoke | `./.venv/bin/python run_md_continuous.py --quick --limit 1 --keep-trajectories` |
| Retrain watcher | `./.venv/bin/python auto_retrain_watcher.py` |
| Health check | `./.venv/bin/python physics_status.py` |
| Stop everything | `pkill -f 'precompute_qm.py\|run_md_continuous.py\|auto_retrain_watcher.py\|physics_autosave_loop'` |

`run_physics_overnight.sh` launches QM + MD + watcher + autosave detached under
`caffeinate`, logging to `physics_logs/`. It needs **AC power + lid open**
(software can't keep an unplugged or clamshell-closed Mac awake).

---

## 6. State, resume & disk

- **Caches are the source of truth.** Both drivers skip SMILES already present,
  so killing and restarting never loses or repeats work.
- **Disk is bounded.** `precompute_md.run_one_compound` deletes each trajectory
  right after observables are extracted (`keep_workdir=False` default;
  `_purge_md_bulk`). Peak usage stays ~one trajectory (~1–2 GB) instead of
  accumulating hundreds of GB. Pass `--keep-trajectories` only for debugging.
  `physics_cache/` is git-ignored.
- **Coverage today:** QM 16/271, MD 3/271 (pre-overnight baseline).

---

## 7. Cost expectations

- **QM:** ~15–40 min per compound (measured; varies a lot with size / number of
  protomers; ETKDGv3 + GFN-FF + GFN2/ALPB opt on whole + head + tail, both
  states). Sequential ≈ 3–4 days for the ~255 remaining; **`precompute_qm_parallel.py`
  (4 workers × 2 threads) ≈ 1–1.5 days** by using all 10 cores. Fully resumable.
  Do NOT drop `n_confs` to go faster — that's a fidelity cut.
- **MD:** production is 256 molecules × 2 µs per state × 2 states. This is hours
  per compound on CPU; full coverage of 268 remaining compounds is a multi-day
  background effort. The watcher improves the model incrementally as rows land,
  so partial coverage is already useful. **We do not shorten trajectories to go
  faster** — that would be a proxy (see `docs/T4_10_MARTINI_MD_skip.md`).

(Replace these with measured per-compound times from `physics_logs/` once the
first few complete.)

---

## 8. Backup / autosave

`physics_autosave_loop.sh` commits a curated path list (caches, models, audits,
code, this doc — never `lion_repo/`, `.venv/`, or trajectories) every hour and
pushes to `origin`. Branch: `physics-overnight`. So an unexpected restart costs
at most ~1 h of compute. Push auth is confirmed working (HTTPS credential helper).

---

## 9. No-proxy contract (do not weaken)

- Missing engine, non-convergence, parse error, decomposition failure →
  the failing key is `NaN`, never a default.
- Emulators (`qm_emulator`, `md_emulator`) are a *fallback for cache misses only*
  and are themselves trained on real computed rows — they are not used to
  overwrite real values.
- Shorter/cheaper trajectories, fewer conformers "to save time", or borrowing a
  family member's value are all proxies and are forbidden.

---

## 10. Known issues / follow-ups

- `qm_emulator_report.json` / `md_emulator_report.json` were 2 bytes (`{}`) after
  the last run — emulator training emitted empty reports. Investigate whether the
  emulators trained on too few rows (cache was 16/3) or the report write is a
  no-op below a row threshold.
- When `$HOME` ≠ `/Users/aryamantiwary`, update the hardcoded root-prefix in
  `qm_descriptors.py:DEFAULT_XTB_CMD` (or export `XTB_CMD`).
