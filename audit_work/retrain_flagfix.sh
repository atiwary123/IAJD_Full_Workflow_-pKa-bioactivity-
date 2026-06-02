#!/bin/bash
# retrain_flagfix.sh — propagate the 2026-06-02 flag-fix (corrected SMILES + MED rows now included)
# into the models. NON-QM/MD path: QM/MD feature block uses the existing emulator fallback for the
# changed/new rows (no xTB/GROMACS run, per user). v15 physics-hybrid is DEFERRED (it depends on real QM).
# Assumes bioact_v14_pipeline.py already rebuilt the foundation bundle.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
LOG=physics_logs/retrain_flagfix.log
mkdir -p physics_logs
echo "[retrain] start $(date)" | tee "$LOG"
step(){ echo "[retrain] >>> $1 $(date +%H:%M:%S)" | tee -a "$LOG"; }

step "train arrays (agile+cpp+qmmd, aligned to new bundle)"
$PY -u audit_work/regen_step10_train_arrays.py >> "$LOG" 2>&1 || echo "[retrain] train_arrays FAILED" | tee -a "$LOG"
step "loo bioact components"
$PY -u loo_bioact_components.py                 >> "$LOG" 2>&1 || echo "[retrain] loo_components FAILED" | tee -a "$LOG"
step "qmmd block (emulator-QM fallback) + head"
$PY -u adaptive_stacker.py --build-qmmd-block   >> "$LOG" 2>&1 || echo "[retrain] qmmd-block FAILED" | tee -a "$LOG"
$PY -u train_qmmd_head_only.py                  >> "$LOG" 2>&1 || echo "[retrain] qmmd_head FAILED" | tee -a "$LOG"
step "bioact ensemble / per-organ / binary"
$PY -u train_bioact_ensemble.py                 >> "$LOG" 2>&1 || echo "[retrain] ensemble FAILED" | tee -a "$LOG"
$PY -u train_per_organ.py                       >> "$LOG" 2>&1 || echo "[retrain] per_organ FAILED" | tee -a "$LOG"
$PY -u train_binary_classifier.py               >> "$LOG" 2>&1 || echo "[retrain] binary FAILED" | tee -a "$LOG"
step "adaptive stacker (production)"
$PY -u adaptive_stacker.py                      >> "$LOG" 2>&1 || echo "[retrain] stacker FAILED" | tee -a "$LOG"
step "pKa v92 retrain (QM-independent)"
$PY -u train_pka_v92.py                         >> "$LOG" 2>&1 || echo "[retrain] pka_v92 FAILED" | tee -a "$LOG"
step "refresh NN transfer input CSV"
$PY -u nn/prepare_iajd.py                       >> "$LOG" 2>&1 || echo "[retrain] nn_prepare FAILED" | tee -a "$LOG"

echo "[retrain] DONE $(date)" | tee -a "$LOG"
touch physics_logs/retrain_flagfix.DONE
echo "[retrain] === FAILED steps (if any) ==="; grep -c FAILED "$LOG" | tee -a "$LOG"
