#!/usr/bin/env bash
# run_physics_overnight.sh — launch the full predictive-physics build DETACHED,
# so it survives this terminal/session closing and runs uninterrupted overnight.
#
# Starts long-lived workers under caffeinate (idle-sleep guard):
#   1. precompute_qm.py        GFN2-xTB descriptors for every IAJD   -> qm_cache
#   2. run_md_continuous.py    MARTINI 3 self-assembly MD            -> md_cache
#   3. auto_retrain_watcher.py refits emulators + qmmd head as caches grow
#   4. physics_autosave_loop   hourly git commit+push of the caches (GitHub backup)
#
# All REAL computation, no proxies. Resumable: re-running skips finished work
# (caches are the source of truth). Logs go to physics_logs/.
#
#   bash run_physics_overnight.sh           # production  (256 mols x 2 us MD)
#   QUICK=1 bash run_physics_overnight.sh   # smoke scale (16 mols x 20 ns MD)
#
# Stop everything:
#   pkill -f 'precompute_qm.py|run_md_continuous.py|auto_retrain_watcher.py|physics_autosave_loop'
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/physics_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

is_running() { pgrep -f "$1" >/dev/null 2>&1; }

# --- power / sleep guard -------------------------------------------------
# caffeinate blocks idle/display/disk/system sleep WHILE it runs. It cannot
# keep an unplugged Mac alive, nor a clamshell-closed MacBook without an
# external display — keep the machine plugged in with the lid open.
if ! is_running "caffeinate -dimsu"; then
  nohup caffeinate -dimsu >/dev/null 2>&1 </dev/null &
  echo "[start] caffeinate (sleep guard)  pid=$!"
else
  echo "[skip]  caffeinate already running"
fi

if ! pmset -g batt 2>/dev/null | grep -q "AC Power"; then
  echo "  !! WARNING: running on BATTERY. Plug in or this stops when it drains."
fi

# --- worker launcher -----------------------------------------------------
launch() {  # match-pattern  logfile  command...
  local pat="$1"; shift
  local log="$1"; shift
  if is_running "$pat"; then
    echo "[skip]  $pat already running"
    return
  fi
  nohup "$@" >>"$log" 2>&1 </dev/null &
  echo "[start] $pat  pid=$!  ->  ${log#$ROOT/}"
}

MD_ARGS=""
[ "${QUICK:-0}" = "1" ] && MD_ARGS="--quick"

launch "precompute_qm_parallel.py" "$LOGDIR/qm_$STAMP.log"     env QM_THREADS="${QM_THREADS:-2}" "$PY" precompute_qm_parallel.py --workers "${QM_WORKERS:-4}"
if [ "${SKIP_MD:-0}" = "1" ]; then
  echo "[skip]  run_md_continuous.py (SKIP_MD=1 — MD deferred by request)"
else
  launch "run_md_continuous.py"  "$LOGDIR/md_$STAMP.log"       "$PY" run_md_continuous.py $MD_ARGS
fi
launch "auto_retrain_watcher.py" "$LOGDIR/watcher_$STAMP.log"  "$PY" auto_retrain_watcher.py
launch "physics_autosave_loop"   "$LOGDIR/autosave_$STAMP.log" bash physics_autosave_loop.sh

cat <<EOF

All workers launched detached (survive terminal close).
  logs:    $LOGDIR/
  status:  $PY physics_status.py
  live QM: tail -f $LOGDIR/qm_$STAMP.log
  live MD: tail -f $LOGDIR/md_$STAMP.log
EOF
