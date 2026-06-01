# SPEC — Ensure ALL compute (local + rented pod) references the CORRECTED datasets

**Audience:** the next Claude session.
**Goal (one line):** make every CPU/GPU computation — on this laptop AND on a rented RunPod box —
read the **audit-corrected** IAJD data (corrected SMILES/pKa), never the old false SMILES.
**Status when this spec was written (2026-06-01):** the local swap is applied but **NOT yet
committed/pushed**, so the pod would still get the OLD data. Closing that is the #1 task below.

---

## 0. TL;DR — what the next session MUST do (in order)

1. **Verify the swap is in the working tree** (canonical xlsx == corrected): §5 verification.
2. **Commit + LFS-push the corrected canonical xlsx + the current AUDIT_FIXED xlsx** so the pod's
   `git clone`+`git lfs pull` gets corrected data. **This is the critical gap** — see §3 + §4.
3. **Finish the §4d UNRESOLVED filter** on `run_iajd_panel.py` (and confirm it on `nn/train_transfer.py`,
   already added) so no compute runs on a known-wrong SMILES. §6.
4. **Re-run `nn/prepare_iajd.py`** to refresh `nn/iajd_transfer_input.csv` from the corrected data, commit it.
5. **Verify on the pod side** that `git lfs pull` yields a REAL corrected xlsx (not a 131-byte pointer). §4.
6. Only then tell the user "ready to run forward."

---

## 1. Background — the problem being solved

The IAJD datasets contained **systematically wrong SMILES** (chiefly the PE-Gallic family, paper
`ja1c05813`: free-OH vs OCH₃ methyl-ether, phenylacetate-ester vs OBn-ether, plus twin-twin/twin-mix
reconstructions). A deep audit corrected them. Because **features are computed FRESH from SMILES**
(Morgan fingerprints, AGILE GNN embeddings, the CG model for c₀, QM/MD precompute), a wrong SMILES
silently produces wrong features/predictions. So every compute entry point must read the corrected SMILES.

Full audit findings: **`docs/DATASET_AUDIT_REPORT_2026-06-01.md`**
Correction + heavy-feature-regen guide: **`docs/DATASET_CORRECTION_AND_REGEN_GUIDE.md`**
Per-change ledger (70 entries): **`audit_work/AUDIT_corrections_master.csv`**

---

## 2. The corrected-data assets (exact paths)

| File | Role |
|---|---|
| `IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx` | corrected bioactivity table (273 rows; ~51 corrected/regen, ~20 UNRESOLVED) |
| `IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx` | corrected pKa table (278 rows; ~57 corrected, ~23 UNRESOLVED) |
| `IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx` (canonical) | **after §4a swap == the corrected content** |
| `IAJD_master/datasets/IAJD_pKa_v21_final.xlsx` (canonical) | **after §4a swap == the corrected content** |
| `IAJD_master/datasets/*.PRE_AUDIT.xlsx` | backups of the pre-swap (old/false) canonical — local only |

**The `audit_status` column** (present in both corrected files) tags every change:
- contains `corrected` / `reconstruct` / `rebuilt` / `REGEN` → a fix was applied (trust the SMILES).
- contains **`UNRESOLVED`** or **`FLAG_10118`** → the SMILES is STILL wrong/unverifiable
  (~20–23 rows, mostly PE-Gallic Library-6 twin-mix IAJD 26–54 + image-only specials). **These must be
  EXCLUDED from any structure-based compute/training** (audit §4d).

---

## 3. The two-layer mechanism (how corrected data is referenced)

There are TWO independent layers; **both** now point at corrected data — keep them consistent.

### Layer A — the §4a SWAP (covers ~46 canonical-reading scripts)
`docs/...REGEN_GUIDE.md §4a` says: when the overnight run is idle, copy AUDIT_FIXED → canonical
(with a `.PRE_AUDIT` backup). **This was applied locally this session** (the cp commands ran; backups
exist). Because ~46 scripts hard-read `IAJD_Bioact_v13_clean.xlsx` / `IAJD_pKa_v21_final.xlsx`
(canonical), the swap makes ALL of them use corrected data with no per-script edits. The full list of
canonical-reading compute scripts includes: `precompute_qm.py`, `precompute_qm_parallel.py`,
`precompute_md.py`, `agile_embeddings.py` (the user's, distinct from `nn/agile_embed.py`),
`martini/build_cg.py`, `iajd_grammar.py`, `train_v15_physics_ml.py`, `physics_calibrate.py`,
`compute_cpp.py`, `validate_cg_mapping.py`, `IAJD_master/code/bioact_v14_pipeline.py`, `iajd_predict.py`,
`propose_iajds.py`, and ~30 more (regenerate the list with the grep in §5).

### Layer B — AUDIT_FIXED-preferring (4 launch scripts, already committed)
These were edited this session to prefer `*.AUDIT_FIXED.xlsx` when present, else canonical:
- `run_iajd_panel.py` (the c₀ panel family loader)
- `analyze_panel.py` (the 24h-decision signal tool)
- `nn/prepare_iajd.py` (the NN target table)
- `physics_design/iajd_cg.py` (the CG model SMILES source for the c₀ panel)

After the swap, canonical == AUDIT_FIXED content, so Layer A and Layer B read the same corrected data.
**Consistency rule:** if the user updates `*.AUDIT_FIXED.xlsx` again, re-run §4a (swap) so canonical
stays in sync — otherwise Layer A (canonical) and Layer B (AUDIT_FIXED) diverge.

---

## 4. The CRITICAL chain: how corrected data reaches the RENTED POD

The rented box does **`git clone` → `git lfs pull`** of this repo (see `RUNPOD_LAUNCH.md` §0 +
`cloud_setup.sh` step 4). Two facts make this fragile:

1. **`*.xlsx` is git-LFS-tracked** (`.gitattributes`: `*.xlsx filter=lfs ...`). A plain clone yields
   **131-byte POINTER stubs**, not the real xlsx → `read_excel` fails. `cloud_setup.sh` already installs
   git-lfs and runs `git lfs pull --include="IAJD_master/datasets/*.xlsx"` to fetch the REAL files. Keep this.
2. **The corrected canonical must be COMMITTED + the LFS blob PUSHED to GitHub**, or the pod pulls the
   OLD blob. As of this spec, **the swap modified the working tree but was NOT committed** — so the
   pod would still get the old SMILES. **The next session must:**
   ```bash
   git add IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx \
           IAJD_master/datasets/IAJD_pKa_v21_final.xlsx \
           IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx \
           IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx
   git commit -m "Apply AUDIT_FIXED -> canonical (corrected SMILES) for pod + local compute"
   git push origin physics-overnight
   git lfs push origin physics-overnight        # guarantee the LFS blobs are on the remote
   ```
   (Do NOT commit `*.PRE_AUDIT.xlsx` — keep them local, or `.gitignore` them; the old data is also
   recoverable from git history.)

**Pod-side proof it worked** (run after cloud_setup on the pod, or simulate with a fresh
`GIT_LFS_SKIP_SMUDGE=1 git clone` + targeted `git lfs pull`):
```bash
python -c "import pandas as pd; d=pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx'); \
print('audit_status' in d.columns, (d['audit_status'].astype(str).str.contains('corrected',case=False,na=False)).sum())"
# expect: True  ~51   (if False/0 or the file is ~131 bytes -> the pod has OLD/pointer data)
```

---

## 5. Verification protocol (confirm corrected data flows everywhere)

```bash
# (a) the swap is in the working tree (canonical == corrected)
python -c "import pandas as pd; \
[print(n, 'audit_status' in pd.read_excel(f'IAJD_master/datasets/{n}.xlsx').columns) \
 for n in ('IAJD_Bioact_v13_clean','IAJD_pKa_v21_final')]"   # both True

# (b) regenerate the list of scripts still reading the dataset (to spot any new canonical readers)
grep -rln "IAJD_Bioact_v13_clean\|IAJD_pKa_v21_final" --include="*.py" . \
  | grep -vE '\.venv|__pycache__|agile_repo|lion_repo|audit_work'

# (c) the launch entry points use corrected + filter UNRESOLVED
python -c "from nn.prepare_iajd import BIOACT; print(BIOACT.name)"     # *.AUDIT_FIXED.xlsx
python nn/prepare_iajd.py                                              # refresh the NN table
head -1 nn/iajd_transfer_input.csv                                     # has 'audit_status' column

# (d) a corrected example actually changed (spot-check IAJD 6 = PE-Gallic phenylacetate->OBn)
python -c "import pandas as pd; d=pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx'); \
r=d[d['IAJD_num']==6].iloc[0]; print(r['audit_status']); print(r[[c for c in d.columns if 'smiles' in c.lower()][0]])"
```

---

## 6. The §4d UNRESOLVED filter (don't compute on known-wrong SMILES)

The ~20 rows with `audit_status` containing `UNRESOLVED` or `FLAG_10118` still have wrong SMILES.
Exclude them from **structure-based** compute (Morgan/AGILE features, the c₀ panel, any structure-CV).
They keep correct pKa/flux LABELS (usable for label-only analyses).

- **Done this session:** `nn/train_transfer.py` drops them after the validity filter (search
  `audit §4d` in that file). **Re-verify it's committed.**
- **TODO next session:** add the same filter to `run_iajd_panel.py::load_family` (so a PE-Gallic c₀
  panel skips them; GA-Tris has none, so GA-Tris is unaffected). Pattern:
  ```python
  if "audit_status" in sub.columns:
      sub = sub[~sub["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG_10118", na=False)]
  ```
- **Precompute (QM/MD):** `precompute_qm.py --resume` only computes SMILES not already cached; the
  UNRESOLVED SMILES were not corrected (unchanged) so they are already cached → not recomputed (no
  wasted compute). The exclusion therefore matters at TRAIN/feature-assembly time, not at precompute.

---

## 7. Heavy-cache regen (separate workstream — the v14/v15 PREDICTOR, not the c₀ launch)

The host-method **c₀ panel + the Morgan/AGILE NN compute features FRESH from SMILES — they do NOT use
the QM/MD caches**, so they are correct the moment the corrected SMILES are in place (Layers A+B). The
QM/MD/LiON/ADMET caches matter only for the separate v14/v15 predictor pipeline. Regen per
`docs/...REGEN_GUIDE.md §3`:
- QM (xTB, local): `python precompute_qm.py --resume` (xtb on PATH from `~/micromamba/envs/xtb_env`);
  36 corrected SMILES, twin-twins ~200 atoms → budget overnight. Writes `…/physics/qm_cache.{csv,parquet}`.
- MD (GROMACS, local): `python precompute_md.py --resume` (only 4/270 done project-wide — large job).
- LiON / ADMET: cloud-only (no local checkpoints) — guide §3c/§3d.
- AGILE: the user's `agile_embeddings.py` (canonical-reading → corrected after swap), or the launch
  uses `nn/agile_embed.py` (frozen 60k MolCLR encoder; reads SMILES from the corrected NN table).
- Then rebuild `qmmd_features_v14_train.npy` + retrain + re-run honest LOO/scaffold-CV (guide §3f).

---

## 8. Risks / caveats to honor

- **Don't claim the pod is corrected until §4 (commit + LFS push + pod-side proof) is done.** The local
  swap alone does NOT reach the pod.
- **The swap orphaned the QM/MD caches for the 38 corrected rows** (caches are keyed by SMILES; the new
  SMILES are cache-misses). Expected — regen per §7. The v14/v15 predictor is in a "corrected SMILES,
  stale heavy caches" state until then; the c₀ panel + Morgan/AGILE NN are unaffected.
- **`auto_retrain_watcher.py` may be running** (it refits emulators as caches change). It reads the
  caches/labels; the swap changes 5 pKa labels → a refit will pick those up. Harmless, but if you want a
  clean retrain, stop it (`pkill -f auto_retrain_watcher`) before the regen and restart after.
- **Two AGILE embedders exist:** the user's root `agile_embeddings.py` (their pipeline) and this
  session's `nn/agile_embed.py` (the launch's frozen-encoder transfer). Don't conflate them.

---

## 9. Reference index (everything touched)

- Corrected data: `IAJD_master/datasets/IAJD_{Bioact_v13_clean,pKa_v21_final}{.AUDIT_FIXED,}.xlsx`,
  `*.PRE_AUDIT.xlsx`.
- Audit docs: `docs/DATASET_AUDIT_REPORT_2026-06-01.md`, `docs/DATASET_CORRECTION_AND_REGEN_GUIDE.md`,
  `audit_work/AUDIT_corrections_master.csv`.
- Layer-B scripts (AUDIT_FIXED-preferring): `run_iajd_panel.py`, `analyze_panel.py`, `nn/prepare_iajd.py`,
  `physics_design/iajd_cg.py`.
- LFS / pod: `.gitattributes` (`*.xlsx` LFS), `cloud_setup.sh` (git-lfs install + `git lfs pull`),
  `RUNPOD_LAUNCH.md` (launch sequence), `nn/fetch_data.sh` (LNPDB/AGILE fetch).
- Launch compute: `compute_curvature.py` (`--host`), `run_iajd_panel.py`, `analyze_panel.py`,
  `physics_design/build_mixed_bilayer.py`, `nn/{prepare_iajd,train_transfer,agile_embed}.py`.
- Memory: `project-dataset-smiles-audit`, `project-runpod-launch-ready`, `reference-agile-nn-pretraining`.
