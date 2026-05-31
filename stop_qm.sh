#!/usr/bin/env bash
# stop_qm.sh — cleanly stop the parallel QM run.
#
# WHY THIS EXISTS: precompute_qm_parallel.py uses a ProcessPoolExecutor whose
# worker processes have a generic `multiprocessing.spawn` command line — they do
# NOT match `precompute_qm_parallel`, so `pkill -f precompute_qm_parallel` leaves
# them alive and they keep spawning micromamba→xtb (orphan storm). This kills the
# whole lineage in the right order: spawners (driver + pool workers + micromamba)
# BEFORE xtb, so nothing respawns.
set -uo pipefail
echo "[stop_qm] stopping parallel QM at $(date)…"
pkill -9 -f "precompute_qm_parallel"            2>/dev/null
pkill -9 -f "multiprocessing.spawn"             2>/dev/null
pkill -9 -f "multiprocessing.resource_tracker"  2>/dev/null
# Kill any still-live micromamba's parent (a pool worker), then micromamba, then xtb.
for mm in $(pgrep -f "micromamba run -n xtb_env" 2>/dev/null); do
  kill -9 "$(ps -o ppid= -p "$mm" 2>/dev/null | tr -d ' ')" 2>/dev/null
done
pkill -9 -f "micromamba run -n xtb_env" 2>/dev/null
pkill -9 -x xtb 2>/dev/null
echo "[stop_qm] now: xtb=$(pgrep -x xtb|wc -l|tr -d ' ') " \
     "micromamba=$(pgrep -f 'micromamba run'|wc -l|tr -d ' ') " \
     "poolworkers=$(pgrep -f multiprocessing.spawn|wc -l|tr -d ' ')"
echo "[stop_qm] (watcher/autosave are left running)"
