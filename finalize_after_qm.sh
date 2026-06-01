#!/bin/bash
# finalize_after_qm.sh — overnight closure for the 2026-06-01 dataset-correction regen.
#
# The cheap regen + retrain already ran on the corrected dataset; Block D' (physics)
# used the QM *emulator* fallback for the ~35 corrected compounds whose real xTB QM
# was still computing. Once precompute_qm.py finishes, this script rebuilds the full
# bioactivity stack so Block D' uses REAL QM for every compound. (pKa side is
# independent of QM and is already final.)
#
# Launched in the background alongside precompute_qm.py. Safe to re-run manually.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
# single-thread BLAS/OMP avoids the macOS libomp/torch hang in the AGILE step
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
LOG=physics_logs/finalize_after_qm.log
mkdir -p physics_logs
echo "[finalize] started $(date)" | tee -a "$LOG"

# 1. Wait for the overnight xTB run to finish (process gone) — 16h safety cap.
for i in $(seq 1 1920); do
  if ! pgrep -f "precompute_qm.py" >/dev/null 2>&1; then
    echo "[finalize] precompute_qm not running (iter $i) $(date)" | tee -a "$LOG"; break
  fi
  sleep 30
done

# 2. Report QM coverage of the 236 training SMILES (informational).
$PY - <<'PYCOV' 2>&1 | tee -a "$LOG"
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
def canon(s):
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    return Chem.MolToSmiles(m) if m else None
b=pd.read_excel("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
flag=b["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG",na=False)
b=b[~flag & b["log10_flux_total"].notna()]
need={canon(s) for s in b["SMILES_canonical"] if canon(s)}
qm=set(pd.read_csv("IAJD_master/bundles_caches/physics/qm_cache.csv")["smiles_canonical"].astype(str))
miss=need-qm
print(f"[finalize] QM coverage of training SMILES: {len(need)-len(miss)}/{len(need)}  (still emulator-filled: {len(miss)})")
PYCOV

# 3. Full rebuild chain (Block D' now real where QM landed; emulator only for any QM failures).
echo "[finalize] rebuild: bioact_v14_pipeline" | tee -a "$LOG"
$PY -u IAJD_master/code/bioact_v14_pipeline.py            >> "$LOG" 2>&1
echo "[finalize] rebuild: train arrays (agile+cpp aligned)" | tee -a "$LOG"
$PY -u audit_work/regen_step10_train_arrays.py            >> "$LOG" 2>&1
echo "[finalize] rebuild: loo components" | tee -a "$LOG"
$PY -u loo_bioact_components.py                           >> "$LOG" 2>&1
echo "[finalize] rebuild: qmmd block + head" | tee -a "$LOG"
$PY -u adaptive_stacker.py --build-qmmd-block             >> "$LOG" 2>&1
$PY -u train_qmmd_head_only.py                            >> "$LOG" 2>&1
echo "[finalize] rebuild: ensemble / per-organ / binary" | tee -a "$LOG"
$PY -u train_bioact_ensemble.py                           >> "$LOG" 2>&1
$PY -u train_per_organ.py                                 >> "$LOG" 2>&1
$PY -u train_binary_classifier.py                         >> "$LOG" 2>&1
echo "[finalize] rebuild: adaptive stacker (production, last)" | tee -a "$LOG"
$PY -u adaptive_stacker.py                                >> "$LOG" 2>&1

echo "[finalize] DONE $(date)" | tee -a "$LOG"
touch physics_logs/finalize_after_qm.DONE
