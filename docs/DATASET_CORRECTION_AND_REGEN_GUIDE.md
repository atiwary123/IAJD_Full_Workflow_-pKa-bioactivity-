# IAJD Dataset — Correction & Feature-Regeneration Guide (2026-06-01)

Companion to `docs/DATASET_AUDIT_REPORT_2026-06-01.md`. This document records **exactly what was
corrected**, **what was not**, and gives **step-by-step instructions to regenerate the
compute-heavy features** (QM / MD / LiON / ADMET / AGILE) for the corrected rows.

Corrected data lives in (canonical files left untouched — overnight physics run is active):
- `IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx`
- `IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx`
- Change ledger: `audit_work/AUDIT_corrections_master.csv`  ·  per-row notes: `audit_status` column.

---

## 1. CORRECTED this session

### 1a. SMILES (38 rows fixed/reconstructed, all validated)
| Group | IAJDs | Fix | Validation |
|---|---|---|---|
| PE-Gallic single-single cap errors | 1,2,3,4,6,7,8,19,20,21,22,23,25,34,35,36,44,45 | free-OH→OCH₃ / phenylacetate→OBn ether | SI formula (Scheme S5) + 10118 FractionCSP3 |
| PE-Gallic reconstructions | 9, 24 | full rebuild (right linkage/caps) | SI formula |
| Twin-twin Library 5 | 10,11,12,13,14,15,16,17,18,46 | rebuilt from SI Scheme S9 (cmpd-45 core + 2 dendrons) | SI formula (all 10) |
| PE-Tris (ja3c07337) | 273,287,291,297 | dup/conflict → distinct heads (MeOEtPRZ/HPRZ/pentanoate) | 10118 descriptor vector |
| Dialkoxybenzyl isomers (ja3c13569) | 294,308,309,310 | head-group composition (MPRZ→HPRZ, HPRZ→diEG) | 10118 vector |

### 1b. pKa (5 rows) — to the paper table value
38→6.38, 45→5.93, 47→6.59 (ja1c05813 SI avg); 266→6.54, 268→6.45 (pharmaceutics Table S1).

### 1c. Metadata
- `source` attributed: 92,100,101 → ja3c07337 (in its Table S1).
- Cross-file head conflicts reconciled: 248 (MPRZ), 273 (MeOEtPRZ), 297 (HPRZ).
- head_group label errors noted (44=PIP, 45=MPRZ, 83/89=HPRZ) — see audit report.

### 1d. Features REGENERATED (real computation, cheap) — `audit_work/regen_features.py`
For all 38 SMILES-changed rows, recomputed with the project's OWN functions
(`expand_datasets.compute_rdkit_features` + `compute_3d_features`) — NOT proxies:
- **2D + custom:** ExactMolWt, MolFormula, HeavyAtomCount, NumNitrogens, MolLogP, TPSA, LabuteASA,
  FractionCSP3, RotatableBonds, BertzCT, Chi0v, Chi1v, HallKierAlpha, NumAromaticRings, NumHDonors,
  NumHAcceptors, NumEsters/Amides/Ethers/TertiaryAmines, HasPiperazine, NumAmines_total, GasteigerN_min/max,
  Hydrophobic_Index, Polar_Surface_Ratio, Inductive_Effect_Strength, Gasteiger_Charge_N/Calpha,
  N_ED_Groups_5A, HBD_HBA_Ratio, Desolvation_Proxy.
- **3D (ETKDGv3 + MMFF94, 5 confs):** Rg_3D, Asphericity_3D, Pct_V_Bur_max/mean, E_min_3D,
  N_LowE_Conformers, N_basic_N_3D, embed_method_3D, ff_3D, n_confs_valid_3D.
- **Metadata:** Linker_Length/Taft_Steric_Sum preserved (set 273/287/291 → pentanoate=5, Taft=−6.2).
- Verified: 0 unparseable SMILES, 0 formula≠SMILES, 0 duplicate structures, FractionCSP3 216/224 ≈ 10118.

### 1e. Missing features FILLED (real) — `audit_work/regen_missing.py`
Beyond the 38 corrected rows, ~90 more rows (44 pKa + 46 bioact) that previously had **blank** custom-2D
/ 3D features (originally-missing PE-Tris, G1-Janus, etc. — valid SMILES) were computed with the same
project functions. **Final coverage: pKa 275/278 and bioact 272/273 rows now have the COMPLETE real
RDKit 2D+custom+3D feature set.** The only rows still without full features are the flagged-wrong-SMILES
ones (§2a) — deliberately skipped so features are never computed from a known-wrong structure.

---

## 2. NOT corrected (and why)

### 2a. SMILES still flagged unresolved (23) — `audit_status` = UNRESOLVED/FLAG_*
- **Library 6 twin-mix** (ja1c05813: 26,27,28,29,38,39,40,41,42,43,47,48,49,50,51,52,53,54): complex
  multi-dendron + PEG-spacer (n=3,4,8,45) architecture; **not in the 10118 paper**; only in scanned SI
  Schemes S10–S12 (no clean machine-readable structure). Not safely reconstructable from available data.
- **Lib-4 module-D** (30,31), **33** (multiple SI formula matches), **301** (complex).
- **64, 86** (ja1c09585, image-only SI): chain/head differs from 10118 by a few CH₂ — unadjudicable.
- **134** (10118 lists a different ~2× "twin" structure for this number; identity unclear).
→ Recommendation: **exclude these from structure-based features/training** until resolved (§4d).

### 2b. Compute-heavy feature caches now STALE for the 38 corrected SMILES
These are keyed by **canonical SMILES** in separate caches (not the xlsx). The corrected SMILES are new
keys, so their entries are missing/orphaned:
| Cache | File | Engine | Missing for corrected | Local engine? |
|---|---|---|---|---|
| QM (xTB) | `IAJD_master/bundles_caches/physics/qm_cache.csv` | xtb | 36 / 38 | ✅ `~/micromamba/envs/xtb_env/bin/xtb` |
| MD (MARTINI) | `…/physics/md_cache.csv` | GROMACS | 38 / 38 (only 4/270 done project-wide) | ✅ `/opt/homebrew/bin/gmx` |
| LiON | `IAJD_master/bundles_caches/lion_cache_v13.json` | chemprop 1.7.0 GNN | all 38 | ❌ cloud (no local checkpoint) |
| ADMET | `…/admet_cache_v13.json` | admet-ai | all 38 | ❌ cloud (no local install) |
| AGILE emb. | `agile_embeddings_*.npy` | torch_geometric + agile_model | all 38 | ⚠️ local if torch_geometric installed |
| Assembled | `qmmd_features_v14_train.npy` | (rebuild from caches) | — | rebuild step |

---

## 3. THOROUGH instructions — regenerate the heavy features

> Run from repo root. First decide whether to point the scripts at the corrected file (recommended:
> apply AUDIT_FIXED → canonical first, §4a) so the precompute scripts read the corrected SMILES.

### 3a. QM descriptors (xTB) — local, slow (~hours for the 36 large twin SMILES)
```bash
source .venv/bin/activate
export PATH="$HOME/micromamba/envs/xtb_env/bin:$PATH"   # xtb on PATH
python precompute_qm.py --resume        # skips the 234 already-cached SMILES, computes the 36 new
# parallel variant if cores allow:
python precompute_qm_parallel.py --resume
```
- Writes `IAJD_master/bundles_caches/physics/qm_cache.{csv,parquet}` (key = `smiles_canonical`).
- The twin-twins (C133–C167) are the slow ones (xTB conformer opt on ~200 heavy atoms). Budget overnight.
- Old (wrong-SMILES) rows become orphaned in the cache — harmless; the feature builder joins on the
  current canonical SMILES.

### 3b. MD observables (GROMACS + MARTINI) — local, slow; project-wide incomplete
```bash
export PATH="/opt/homebrew/bin:$PATH"    # gmx
python precompute_md.py --resume         # neutral+protonated, seeds 42/43, per SMILES
```
- Writes `…/physics/md_cache.csv`. NOTE: only 4/270 rows are currently populated project-wide (a
  pre-existing GROMACS/throughput gap, not caused by the corrections) — a full MD pass is a larger job.
- If `gmx` setup is unavailable, rows are written with NaN observables + `error="gmx_unavailable"`
  (cache stays present; the MD emulator `physics_cache/md_emulator.joblib` can stand in).

### 3c. LiON predictions (chemprop GNN) — CLOUD (no local checkpoint)
On a machine with the LiON checkpoints (see repo README Zenodo/Figshare DOI):
```bash
python -m venv lion_env && source lion_env/bin/activate
pip install chemprop==1.7.0
# write the corrected canonical SMILES to a column-CSV, then run the LiON predict script:
python lion_repo/scripts/predict.py --test_path corrected_smiles.csv \
       --checkpoint_dir <lion_checkpoints> --preds_path lion_preds.csv
```
Then merge into `IAJD_master/bundles_caches/lion_cache_v13.json` (key = canonical SMILES). The v14
pipeline (`IAJD_master/code/predict_v14_real.py`) auto-reads this cache; until then it uses an RDKit proxy.

### 3d. ADMET-AI properties — CLOUD (separate env)
```bash
python -m venv admet_env && source admet_env/bin/activate
pip install admet-ai
python -c "from admet_ai import ADMETModel; import json,pandas as pd; \
m=ADMETModel(); df=m.predict(smiles=open('corrected_smiles.txt').read().split()); \
df.to_json('admet_new.json')"
```
Merge into `IAJD_master/bundles_caches/admet_cache_v13.json` (10-d vector / canonical SMILES).

### 3e. AGILE encoder embeddings — local if `torch_geometric` is installed (fast: frozen forward pass)
```bash
source .venv/bin/activate
pip install torch torch_geometric    # if missing
python agile_embeddings.py           # loads agile_model/model.pth, embeds all SMILES -> agile_embeddings_*.npy
```

### 3f. Rebuild assembled feature matrix + retrain
```bash
# after qm_cache + md_cache (+ lion/admet) are extended:
python -c "import precompute_qm"     # qmmd join is in the v14/v15 build scripts:
python retrain_with_physics.py       # rebuilds qmmd_features_v14_train.npy and refits
python IAJD_master/code/bioact_v14_pipeline.py   # rebuild bioact bundle on corrected features
python train_pka_v92.py              # refit pKa model on corrected features
```
Always re-run the LOO/scaffold-CV reports afterward (the corrected PE-Gallic structures change the
feature distribution for ~38 rows; expect the honest metrics to shift).

---

## 4. Operational notes

### 4a. Apply corrected copies → canonical (when overnight run is idle)
```bash
cp IAJD_master/datasets/IAJD_pKa_v21_final.xlsx       IAJD_master/datasets/IAJD_pKa_v21_final.PRE_AUDIT.xlsx
cp IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx    IAJD_master/datasets/IAJD_Bioact_v13_clean.PRE_AUDIT.xlsx
cp IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx    IAJD_master/datasets/IAJD_pKa_v21_final.xlsx
cp IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx
```
Then run §3 to regenerate heavy caches, then §3f to retrain.

### 4b. Order of operations
1. Apply AUDIT_FIXED → canonical (4a).  2. QM (3a) + MD (3b).  3. LiON/ADMET/AGILE (3c–3e).
4. Rebuild + retrain (3f).  5. Re-run honest LOO/scaffold-CV.

### 4c. RDKit features are already real — do NOT re-proxy
The 38 corrected rows already have real RDKit 2D+3D features (this session). Only the engine-backed
caches in §2b remain. Custom features (Hydrophobic_Index, Taft, etc.) were regenerated with the
project's own code — do not overwrite with proxies.

### 4d. The 23 unresolved SMILES
Resolve by reading the high-resolution SI structure figures (Library 6: ja1c05813 Schemes S10–S12;
isomers 301 / dialkoxy 64/86/134: the respective image-only SIs) or by requesting machine-readable
structures from the authors. Until then, filter rows where `audit_status` contains `UNRESOLVED` or
`FLAG_10118` before structure-based training. They keep correct pKa/bioactivity labels (usable for
label-only analyses).
```python
import pandas as pd
df = pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx')
clean = df[~df['audit_status'].astype(str).str.contains('UNRESOLVED|FLAG_10118', na=False)]
```
