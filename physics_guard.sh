#!/usr/bin/env bash
# physics_guard.sh — laptop guardian for the physics-design runs on a fanless M-series
# Air. Monitors thermal throttling + disk + load, logs them, auto-cleans my transient
# artifacts, and (only if macOS reports CPU thermal throttling) briefly pauses the heavy
# compute to shed heat, ALWAYS resuming it (trap on exit). Non-destructive to live runs.
#
#   bash physics_guard.sh            # default 60s cadence, 45s cooldown, 6 GB disk floor
#   INTERVAL=90 COOL=60 bash physics_guard.sh
#
# Safe by construction: a trap resumes every paused process on any exit, so a crash of
# this script can never leave GROMACS/xtb stopped.
set -u
cd "$(dirname "$0")"
INTERVAL="${INTERVAL:-60}"
COOL="${COOL:-45}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-6}"
LOG=physics_guard.log

resume_all() { pkill -CONT -f "gmx mdrun" 2>/dev/null; pkill -CONT -x xtb 2>/dev/null; }
trap 'echo "[guard] exit $(date +%H:%M:%S) — resuming all" >>"$LOG"; resume_all; exit' EXIT INT TERM

free_gb() { df -g . | awk 'NR==2{print $4}'; }

clean_tmp() {
  # my own transient dumps only — never touch the QM run's /tmp/claude-501/qm_* scratch
  find /tmp -maxdepth 1 -type f \( -name 'fcheck.*' -o -name 'dope_*.gro' \
       -o -name 'smoke_*.log' -o -name '*_chk.log' \) -mmin +10 -delete 2>/dev/null
}

purge_finished_bulk() {
  # drop prod trajectories of curvature runs that already produced a design JSON
  for d in physics_cache/curvature_runs/*/; do
    lip=$(basename "$d"); lip=${lip%%_*}
    if ls IAJD_master/bundles_caches/physics/design/${lip}_*_curvature.json >/dev/null 2>&1; then
      find "$d" \( -name '*.xtc' -o -name '*.trr' -o -name '*.edr' -o -name '*.cpt' \
           -o -name '*.tpr' \) -mmin +2 -delete 2>/dev/null
    fi
  done
}

echo "[guard] start $(date)  interval=${INTERVAL}s cooldown=${COOL}s disk_floor=${DISK_FLOOR_GB}GB" >>"$LOG"
while true; do
  resume_all   # idempotent: ensure nothing is left paused from a previous cycle

  # --- thermal: macOS CPU speed limit (100 = no throttle; <100 = throttling) ---
  therm=$(pmset -g therm 2>/dev/null)
  limit=$(echo "$therm" | grep -oE 'CPU_Speed_Limit[ ]*=[ ]*[0-9]+' | grep -oE '[0-9]+$')
  [ -z "$limit" ] && limit=100
  load=$(uptime | sed -E 's/.*load averages?: //')
  disk=$(free_gb)
  cache=$(du -sm physics_cache 2>/dev/null | awk '{print $1}')
  ts=$(date +%H:%M:%S)

  # --- disk safety ---
  clean_tmp
  if [ "${disk:-99}" -lt "$DISK_FLOOR_GB" ]; then
    echo "[guard] $ts LOW DISK ${disk}GB < ${DISK_FLOOR_GB}GB -> purging finished-run bulk" >>"$LOG"
    purge_finished_bulk
    disk=$(free_gb)
  fi

  # --- thermal mitigation ---
  if [ "$limit" -lt 100 ]; then
    echo "[guard] $ts THROTTLE CPU_Speed_Limit=${limit}% -> ${COOL}s cooldown (pausing gmx+xtb)" >>"$LOG"
    pkill -STOP -f "gmx mdrun" 2>/dev/null; pkill -STOP -x xtb 2>/dev/null
    sleep "$COOL"
    resume_all
    echo "[guard] $ts resumed after cooldown" >>"$LOG"
  else
    echo "[guard] $ts ok  speed=${limit}%  load=[${load}]  disk=${disk}GB  physics_cache=${cache}MB" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
