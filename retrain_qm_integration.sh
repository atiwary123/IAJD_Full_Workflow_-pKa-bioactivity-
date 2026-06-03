#!/bin/bash
# retrain_qm_integration.sh — ACTIVATE QM in the bioact stack and retrain.
#
# Decision basis (workflow w4lddtiz3, 2026-06-03):
#   * bioact  : INTEGRATE QM. The live bioact_v14_bundle.pkl was fit on a 100%-NaN
#               Block D' (QM importance == 0); it never saw a QM value, even though
#               qm_cache.csv is now complete and load_physics resolves every training
#               SMILES. This re-runs the pipeline so Block D' QM cols (X[:,100:104])
#               go live, then retrains the whole downstream stack on the real-QM bundle.
#   * pKa     : NO CHANGE. QM worsens pKa LOO MAE (+0.0045); 3/3 skeptics refuted.
#
# The earlier all-NaN was a build-ORDER timing artifact (the pipeline ran before the
# QM cache finalized). It is NOT a code bug — verified: load_physics on the bundle's
# own smis_train returns 30/30 finite QM right now. A clean re-run fixes it.
#
# Hard gate: if Block-D' QM cols are still NaN after the rebuild, ABORT before the
# (expensive) downstream chain so we never train a second QM-blind stack.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
LOG=physics_logs/retrain_qm_integration.log
mkdir -p physics_logs eval
rm -f physics_logs/retrain_qm_integration.DONE physics_logs/retrain_qm_integration.FAIL
echo "=== retrain_qm_integration START $(date) ===" | tee "$LOG"

# 1. Rebuild the bioact bundle — compute_block_dprime re-pulls the now-complete QM cache.
echo "[1/9] bioact_v14_pipeline (rebuild bundle w/ real Block-D' QM)" | tee -a "$LOG"
$PY -u IAJD_master/code/bioact_v14_pipeline.py >> "$LOG" 2>&1

# 2. HARD GATE — verify the 4 QM cols (X[:,100:104]) are now real before spending more compute.
echo "[gate] verifying Block-D' QM is live..." | tee -a "$LOG"
$PY - <<'PYGATE' >> "$LOG" 2>&1
import pickle, numpy as np, sys
with open('IAJD_master/bundles_caches/bioact_v14_bundle.pkl','rb') as f:
    B = pickle.load(f)
X = B['X_train']
qm_frac = float(np.isfinite(X[:,100:104]).mean())   # the 4 QM cols
bd_frac = float(np.isfinite(X[:,92:106]).mean())     # whole Block D' (MD stays NaN by design)
print(f"[gate] QM-cols finite frac = {qm_frac:.3f}  (expect ~1.0)")
print(f"[gate] Block-D' finite frac = {bd_frac:.3f} (MD cols stay NaN — MD not run)")
sys.exit(0 if qm_frac > 0.9 else 7)
PYGATE
if [ $? -ne 0 ]; then
  echo "[gate] FAILED — QM cols still NaN after rebuild. Aborting downstream." | tee -a "$LOG"
  touch physics_logs/retrain_qm_integration.FAIL
  exit 7
fi
echo "[gate] PASS — Block-D' QM is live. Proceeding with full retrain." | tee -a "$LOG"

# 3-9. Retrain the full downstream stack on the real-QM bundle (same chain as finalize).
echo "[2/9] regen_step10_train_arrays (agile+cpp aligned)" | tee -a "$LOG"
$PY -u audit_work/regen_step10_train_arrays.py >> "$LOG" 2>&1 || echo "  (regen_step10 non-fatal err)" | tee -a "$LOG"
echo "[3/9] loo_bioact_components" | tee -a "$LOG"
$PY -u loo_bioact_components.py >> "$LOG" 2>&1
echo "[4/9] adaptive_stacker --build-qmmd-block" | tee -a "$LOG"
$PY -u adaptive_stacker.py --build-qmmd-block >> "$LOG" 2>&1
echo "[5/9] train_qmmd_head_only" | tee -a "$LOG"
$PY -u train_qmmd_head_only.py >> "$LOG" 2>&1 || echo "  (qmmd_head non-fatal err)" | tee -a "$LOG"
echo "[6/9] train_bioact_ensemble" | tee -a "$LOG"
$PY -u train_bioact_ensemble.py >> "$LOG" 2>&1 || echo "  (ensemble non-fatal err)" | tee -a "$LOG"
echo "[7/9] train_per_organ" | tee -a "$LOG"
$PY -u train_per_organ.py >> "$LOG" 2>&1 || echo "  (per_organ non-fatal err)" | tee -a "$LOG"
echo "[8/9] train_binary_classifier" | tee -a "$LOG"
$PY -u train_binary_classifier.py >> "$LOG" 2>&1 || echo "  (binary non-fatal err)" | tee -a "$LOG"
echo "[9/9] adaptive_stacker (production) + train_v15 (hybrid, last)" | tee -a "$LOG"
$PY -u adaptive_stacker.py >> "$LOG" 2>&1
$PY -u train_v15_physics_ml.py >> "$LOG" 2>&1

# 10. AFTER snapshot + before/after report.
echo "[report] capturing AFTER state + before/after delta" | tee -a "$LOG"
$PY - <<'PYAFTER' >> "$LOG" 2>&1
import pickle, numpy as np, json
with open('IAJD_master/bundles_caches/bioact_v14_bundle.pkl','rb') as f:
    B = pickle.load(f)
m = B.get('metrics', {}); X = B['X_train']; names = B['feature_names']
mdl = B.get('direct_model_full'); imp = getattr(mdl,'feature_importances_',None)
qm_idx = [i for i,n in enumerate(names) if str(n).startswith('qm_')]
after = {
  'v14_pooled_mae': m.get('v14_pooled_mae'),
  'baseline_pooled_mae': m.get('baseline_pooled_mae'),
  'v14_pooled_r2': m.get('v14_pooled_r2'),
  'qm_cols_finite_frac': float(np.isfinite(X[:,100:104]).mean()),
  'blockD_finite_frac': float(np.isfinite(X[:,92:106]).mean()),
  'qm_importance_sum_AFTER': (float(sum(imp[i] for i in qm_idx)) if imp is not None else None),
  'blockD_importance_sum_AFTER': (float(sum(imp[92:106])) if imp is not None else None),
  'per_family': m.get('per_family'),
}
try:
    before = json.load(open('eval/bioact_BEFORE_qm_integration.json'))
except Exception:
    before = {}
after['v14_pooled_mae_BEFORE'] = before.get('v14_pooled_mae')
after['delta_pooled_mae'] = (after['v14_pooled_mae'] - before['v14_pooled_mae']) if before.get('v14_pooled_mae') else None
json.dump(after, open('eval/bioact_AFTER_qm_integration.json','w'), indent=2, default=str)
print(json.dumps(after, indent=2, default=str))
PYAFTER

echo "=== retrain_qm_integration DONE $(date) ===" | tee -a "$LOG"
touch physics_logs/retrain_qm_integration.DONE
