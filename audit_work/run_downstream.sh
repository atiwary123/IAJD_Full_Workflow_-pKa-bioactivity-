#!/bin/bash
# run_downstream.sh — Task #10 downstream retrain chain on the corrected bundle.
# Builder-independent models run first (concurrent with the CPP train-array builder),
# then the adaptive stacker once the agile/cpp train arrays are ready.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
LOG=physics_logs/regen_downstream.log
BUILDER_PID="${1:-}"
echo "[downstream] start $(date)  builder_pid=$BUILDER_PID" | tee "$LOG"

step () { echo "[downstream] >>> $1 $(date +%H:%M:%S)" | tee -a "$LOG"; }

# --- builder-independent (need only bioact_v14_bundle.pkl) ---
step "adaptive_stacker --build-qmmd-block (qmmd_features_v14_train, 236x14)"
$PY -u adaptive_stacker.py --build-qmmd-block >> "$LOG" 2>&1 || echo "[downstream] build-qmmd-block FAILED" | tee -a "$LOG"
step "train_qmmd_head_only (physics/qmmd_head.joblib)"
$PY -u train_qmmd_head_only.py >> "$LOG" 2>&1 || echo "[downstream] qmmd_head FAILED" | tee -a "$LOG"
step "train_bioact_ensemble"
$PY -u train_bioact_ensemble.py >> "$LOG" 2>&1 || echo "[downstream] ensemble FAILED" | tee -a "$LOG"
step "train_per_organ"
$PY -u train_per_organ.py >> "$LOG" 2>&1 || echo "[downstream] per_organ FAILED" | tee -a "$LOG"
step "train_binary_classifier"
$PY -u train_binary_classifier.py >> "$LOG" 2>&1 || echo "[downstream] binary FAILED" | tee -a "$LOG"

# --- wait for the train-array builder (adaptive_stacker needs agile/cpp v14_train) ---
if [ -n "$BUILDER_PID" ]; then
  step "waiting for train-array builder PID $BUILDER_PID"
  while kill -0 "$BUILDER_PID" 2>/dev/null; do sleep 10; done
fi
# sanity: arrays must exist and be 236-aligned before the stacker
step "adaptive_stacker (production stacker — AGILE+CPP+physics, LAST)"
$PY -u adaptive_stacker.py >> "$LOG" 2>&1 || echo "[downstream] adaptive_stacker FAILED" | tee -a "$LOG"

echo "[downstream] DONE $(date)" | tee -a "$LOG"
touch physics_logs/regen_downstream.DONE
