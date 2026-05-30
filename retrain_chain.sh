#!/bin/bash
# retrain_chain.sh — runs the full retrain pipeline in order:
#   v14 (with Block E) → LOO components → stacker → binary → ensemble → per-organ
set -e
cd /Users/aryamantiwary/Downloads/IAJD_FULL_WORKFLOW_CONDENSED-3

echo "===== [1/6] v14 pipeline (Block A..E, 92 features) ====="
.venv/bin/python -u IAJD_master/code/bioact_v14_pipeline.py 2>&1 | tail -10

echo ""
echo "===== [2/6] LOO components for stacker ====="
.venv/bin/python -u loo_bioact_components.py 2>&1 | tail -10

echo ""
echo "===== [3/6] Adaptive stacker ====="
.venv/bin/python -u train_bioact_stacker.py 2>&1 | tail -6

echo ""
echo "===== [4/6] Binary classifier ====="
.venv/bin/python -u train_binary_classifier.py 2>&1 | tail -10

echo ""
echo "===== [5/6] Deep ensemble (M=7) — for per-candidate σ ====="
.venv/bin/python -u train_bioact_ensemble.py 2>&1 | tail -15

echo ""
echo "===== [6/6] Per-organ predictors ====="
.venv/bin/python -u train_per_organ.py 2>&1 | tail -20

echo ""
echo "===== CHAIN COMPLETE ====="
