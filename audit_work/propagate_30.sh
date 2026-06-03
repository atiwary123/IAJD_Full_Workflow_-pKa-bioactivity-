#!/bin/bash
# propagate_30.sh — propagate the SI-verified IAJD 30 (now HIGH, included) into caches + models.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
LOG=physics_logs/propagate_30.log; mkdir -p physics_logs
echo "[prop30] start $(date)" | tee "$LOG"

echo "[prop30] >>> extend ADMET+LiON for IAJD 30 new SMILES" | tee -a "$LOG"
$PY - >> "$LOG" 2>&1 <<'PYEOF'
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'IAJD_master/code')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from extend_caches import predict_admet_for_smiles, predict_lion_for_smiles
bi=pd.read_excel("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
s=Chem.MolToSmiles(Chem.MolFromSmiles(bi[bi['IAJD_num']==30].iloc[0]['SMILES']))
predict_admet_for_smiles([s]); predict_lion_for_smiles([s])
print("extended caches for 30:", s)
PYEOF

echo "[prop30] >>> MolGpKa live (all, now incl 30)" | tee -a "$LOG"
$PY -u compute_molgpka_live.py >> "$LOG" 2>&1 || echo "[prop30] molgpka FAILED" | tee -a "$LOG"
echo "[prop30] >>> AGILE embeddings (all, now incl 30)" | tee -a "$LOG"
$PY -u agile_embeddings.py >> "$LOG" 2>&1 || echo "[prop30] agile FAILED" | tee -a "$LOG"
echo "[prop30] >>> bioact_v14 foundation rebuild (now incl 30)" | tee -a "$LOG"
$PY -u IAJD_master/code/bioact_v14_pipeline.py >> "$LOG" 2>&1 || echo "[prop30] foundation FAILED" | tee -a "$LOG"

echo "[prop30] >>> retrain chain" | tee -a "$LOG"
bash audit_work/retrain_flagfix.sh >> "$LOG" 2>&1 || echo "[prop30] retrain chain FAILED" | tee -a "$LOG"

echo "[prop30] DONE $(date)" | tee -a "$LOG"
touch physics_logs/propagate_30.DONE
