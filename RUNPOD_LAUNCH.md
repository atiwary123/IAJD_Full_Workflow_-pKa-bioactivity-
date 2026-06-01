# RunPod Launch Manifest — IAJD physics panel + NN transfer model

Everything below is **de-risked locally** (builds, EM, short MD, mixed-system force check,
NN pipeline all run on the laptop) so the pod is **pure compute**. Copy-paste top to bottom.

Box: any RunPod CUDA pod (e.g. RTX 3090/4090 + ≥16 vCPU). The physics is CPU-parallel; the
GPU helps the (optional) GNN encoder + GROMACS GPU offload.

---

## 0. One-time setup (~10 min)
```bash
cd /workspace && curl -fsSL https://raw.githubusercontent.com/atiwary123/IAJD_Full_Workflow_-pKa-bioactivity-/physics-overnight/cloud_setup.sh -o cloud_setup.sh && bash cloud_setup.sh
source /workspace/IAJD/cloud_env.sh && cd /workspace/IAJD
```
(If the repo is private: prepend `GIT_URL="https://<token>@github.com/atiwary123/IAJD_Full_Workflow_-pKa-bioactivity-.git"`.)

---

## 1. PHYSICS — host-method c₀ panel (the main compute)

The pure-IAJD bilayer collapses for membrane-active IAJDs (FW-4), so the panel uses the
**host method** (IAJDs dilute in a stable POPC bilayer; c₀ from leaflet asymmetry). Verified
locally on 369: bilayer stays intact, mixed-system forces exact to 0.00008%.

```bash
# whole GA-Tris family, FW-3 order, host method, parallel across cores (CPU)
python run_iajd_panel.py --family GA-Tris --host --n-iajd 8 \
       --jobs 6 --threads 4 --gpu-jobs 0 --prod-ns 80 --force-check
```
- `--jobs N` concurrent IAJDs, `--threads K` cores each → size to `N*K ≈ vCPUs`.
- **GPU:** conda-forge GROMACS is **CPU-only**, so keep `--gpu-jobs 0` UNLESS you ran
  `BUILD_GPU_GMX=1 bash cloud_setup.sh` and `source /workspace/gromacs-gpu/bin/GMXRC` first
  (only then does `--gpu-jobs 1` / `-nb gpu` work). The CG panel is CPU-parallel-bound anyway,
  so CPU is the right default; the GPU mainly helps the (future) atomistic Module E.
- Output: `IAJD_master/bundles_caches/physics/design/IAJD<n>_host_T300_curvature.json`
  (per IAJD: `c0_nm_inv`, `tau_upper/lower`, `c0_physics_converged`, `force_check`) +
  `iajd_panel_GA-Tris.json` (summary, FW-3 order, high/low Δc₀).
- **Trust rule:** `c0_physics_converged` must be true (tensionless + intact); all c₀ remain
  PROVISIONAL until Module E validates the CG mapping. Honestly read the converged flag.
- Scale to other families: `--family PE-Tris` etc., or `--iajds 369,360,...`.

**Cost:** ~80 ns × ~150 IAJD-runs, parallel → roughly $30–80 over a few days on a 4090 pod.

### (optional) Module A apparent-pKa on IAJDs
Needs a titratable IAJD model first (per-IAJD, not yet built — see FW-2). The METHOD is
validated (MC3 → 6.44). Not part of this launch.

---

## 2. NN — transfer-learning model (LNPDB → frozen encoder + GP head)

Pretraining source (chosen after a dedicated lit search, see IAJD_NN_TRANSFER_LEARNING_PLAN.md):
**LNPDB** (12,845 ionizable lipids, 2,388 in-vivo organ rows — matches our endpoint) +
**AGILE**'s 60k MolCLR encoder as init. All MIT-licensed.

```bash
# fetch corpora + build the in-vivo subset
bash nn/fetch_data.sh
# IAJD target table (SMILES + flux + physics features) — already builds locally, refresh on pod:
python nn/prepare_iajd.py
# transfer vs baseline, leave-one-FAMILY-out gate (Morgan encoder = runs now)
python nn/train_transfer.py --lnpdb nn/lnpdb_invivo.csv --target log10_flux_spleen
python nn/train_transfer.py --lnpdb nn/lnpdb_invivo.csv --target log10_flux_total
```
- Output: `nn/transfer_results.json` — leave-one-family-out Spearman/R² for **baseline vs
  transfer**, and the VERDICT (transfer used only if it beats baseline; local baseline
  Spearman ≈ +0.04 is the bar).
- **Upgrade (GPU):** swap the featurizer for AGILE's frozen 60k-MolCLR embedding —
  `--encoder agile` (wire the checkpoint at `extern/AGILE/ckpt/.../model.pth`; the GP head +
  gate are unchanged). The Morgan path runs immediately; the GNN path is the higher-fidelity
  version.

**Cost:** minutes–an hour (~$1–10).

---

## 3. Honesty — what's proven vs what needs pod iteration
- **Proven locally:** host-method build/EM/MD + exact mixed forces; FW-3 panel runner;
  NN data prep + GP head + leave-one-family-out gate (baseline arm).
- **Needs the pod (not fully de-riskable here):** full tensionless host runs (the c₀ VALUES,
  vs just the machinery); the LNPDB fetch + transfer arm; the AGILE-GNN-encoder integration
  (external repo + checkpoint loading). Expect light iteration on the GNN path.
- **Irreducible caveat:** no public dataset has dendritic/multi-tail chemistry — the NN prior
  is "ionizable amphiphile," not "Janus dendrimer"; physics features carry that residual.

---

## 4. Pull results back
```bash
git add -A && git commit -m "pod: host-method panel + NN transfer results" && git push
```
(or `scp` the `design/*.json` + `nn/transfer_results.json`).
