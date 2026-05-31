#!/usr/bin/env bash
# physics_autosave_loop.sh — every hour, commit the predictive-physics caches,
# models, logs and code, and push them to GitHub. So if the machine restarts
# mid-run, at most ~1h of computed physics is lost and the rest is recoverable.
#
# Commits ONLY a curated path list (never lion_repo/, .venv/, or trajectories)
# to whatever branch is checked out, and pushes to origin. Idempotent: commits
# only when something actually changed.
#
#   bash physics_autosave_loop.sh              # loop forever, 1h cadence
#   INTERVAL=1800 bash physics_autosave_loop.sh   # custom cadence (seconds)
set -uo pipefail
cd "$(dirname "$0")"
INTERVAL="${INTERVAL:-3600}"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo physics-overnight)"
mkdir -p physics_logs

# Curated — caches/models first (the irreplaceable compute), then the code that
# produced them and the recovery docs. Missing paths are skipped each cycle.
PATHS=(
  IAJD_master/bundles_caches/physics
  qm_audit.json md_audit.json auto_retrain_log.csv
  qmmd_features_v14_train.npy
  docs/PREDICTIVE_PHYSICS_BUILD.md
  setup_physics_env.sh run_physics_overnight.sh physics_autosave_loop.sh stop_qm.sh qm_scratch_janitor.sh
  precompute_qm.py precompute_qm_parallel.py precompute_md.py run_md_continuous.py
  qm_descriptors.py auto_retrain_watcher.py
  physics_features.py physics_cache_io.py physics_status.py
  train_qm_emulator.py train_md_emulator.py train_qmmd_head_only.py
  retrain_with_physics.py train_v15_physics_ml.py adaptive_stacker.py
  head_area_3d.py head_area_ensemble.py extend_caches.py
  martini requirements-offline.txt
)

echo "[autosave] loop start $(date)  branch=$BRANCH  interval=${INTERVAL}s"
while true; do
  # Only stage paths that currently exist (qmmd_features appears after 1st retrain).
  existing=()
  for p in "${PATHS[@]}"; do [ -e "$p" ] && existing+=("$p"); done
  git add -- "${existing[@]}" 2>/dev/null || true

  if git diff --cached --quiet 2>/dev/null; then
    echo "[autosave] no changes at $(date)"
  else
    nqm=$(( $(wc -l < IAJD_master/bundles_caches/physics/qm_cache.csv 2>/dev/null || echo 1) - 1 ))
    nmd=$(( $(wc -l < IAJD_master/bundles_caches/physics/md_cache.csv 2>/dev/null || echo 1) - 1 ))
    git commit -q \
      -m "physics autosave $(date '+%Y-%m-%d %H:%M')  (qm=$nqm md=$nmd)" \
      -m "Automated checkpoint of predictive-physics caches/models/code.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" || true
    if git push -q -u origin "$BRANCH" 2>>physics_logs/autosave_push.err; then
      echo "[autosave] committed+pushed qm=$nqm md=$nmd at $(date)"
    else
      echo "[autosave] committed locally; push FAILED at $(date) (retry next cycle, see physics_logs/autosave_push.err)"
    fi
  fi
  sleep "$INTERVAL"
done
