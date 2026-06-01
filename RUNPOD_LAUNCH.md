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
# EFFICIENCY: size --jobs to your cores. Each IAJD ≈ 28 min (60 ns, 4 threads). So
#   throughput ≈ jobs × (60 IAJD/28min) ; e.g. 32 vCPU→jobs 8→~17/h ; 64 vCPU→jobs 16→~34/h.
# n-per-leaflet 48 + n-iajd 8 => x=0.167, the validated config (369 converged: c0=+1.27,
# tensionless, exact forces). prod-ns 60 is the validated minimum; don't go lower.
python run_iajd_panel.py --family GA-Tris --host --n-iajd 8 --n-per-leaflet 48 \
       --jobs $(( $(nproc) / 4 )) --threads 4 --gpu-jobs 0 --prod-ns 60 --force-check
```
- `--jobs N` concurrent IAJDs, `--threads K` cores each → keep `N*K ≈ vCPUs` (4 threads/job
  is the sweet spot; more threads/job scale poorly on these small systems).
- **Run GA-Tris first (369's family, the design target), then `--family PE-Tris`** (n=42, the
  family with the strongest existing pKa↔flux signal). FW-3 ordering means the EARLY runs span
  the flux range, so the correlation is visible after ~8–10 IAJDs (~1 h), not at the end.
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

**Cost/time:** ~28 min/IAJD ÷ parallelism. On a 32-vCPU box (~17/h) the whole 273-IAJD set is
~16 h; on 64 vCPU (~34/h), ~8 h. So **a 24 h CPU run covers the entire dataset** — well within
your decision window. ~$15–60 depending on box.

## 1b. THE 24-HOUR DECISION: is a physics→flux signal emerging?

Run this **anytime as results land** (no need to wait for the panel to finish):
```bash
python analyze_panel.py --target log10_flux_spleen   # or --target log10_flux_total
python analyze_panel.py --target log10_flux_spleen --family GA-Tris
```
It reports, with honest small-n bootstrap CIs: **c₀↔flux** (converged points), **pKa↔flux**
(the existing-feature baseline, available *immediately*), per-family breakdowns, a GAM response
curve once n≥8, and a conservative **VERDICT** (signal needs |Spearman|≥0.35, 90% CI excluding
0, n≥8).
- **You already have a read before any physics:** pKa↔flux_spleen is weak pooled (−0.22) but
  **PE-Tris is moderate (−0.35, n=25)** — so PE-Tris is the most promising family to test whether
  c₀ adds signal.
- **Decision:** if after a family completes (~1–2 h) c₀↔flux is flat AND pKa↔flux is flat, the
  physics axis likely isn't the lever *for that family* → don't burn more compute there; pivot
  family or stop. If a family shows |Spearman|≥0.35 with a CI off 0, keep going + widen the panel.

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
