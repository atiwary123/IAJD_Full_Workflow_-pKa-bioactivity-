#!/usr/bin/env bash
# qm_regen_watch.sh — overnight babysitter for the dataset-correction QM regen.
#
# Companion to the existing combo: physics_guard.sh (thermal STOP/CONT + disk) and
# qm_scratch_janitor.sh (xtb /tmp scratch sweep). Those guard heat + disk; THIS one
# watches that the regen actually makes PROGRESS and auto-corrects crashes:
#   - keeps the guard + janitor alive (relaunch if either dies)
#   - relaunches precompute_qm --resume if it dies before completing
#   - restarts a wedged precompute_qm (no progress + no xtb child for 2 cycles)
#   - (re)launches finalize_after_qm.sh once QM truly completes, exits when it's done
#
# Cadence: 30-min checks for the first PHASE1_HRS hours (QM most active), then hourly.
# Non-destructive: only ever (re)launches; never kills live, progressing work. A STOPped
# (thermally paused) process still shows in pgrep, so it is never mistaken for a crash.
#
#   bash qm_regen_watch.sh                       # 30min x2h then hourly, 24h cap
#   PHASE1_HRS=3 MAX_HRS=18 bash qm_regen_watch.sh
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
XTB_BIN="$HOME/micromamba/envs/xtb_env/bin"
LOG=physics_logs/qm_regen_watch.log
QM_LOG=physics_logs/regen_qm_overnight.log
FIN_DONE=physics_logs/finalize_after_qm.DONE
QM_CACHE=IAJD_master/bundles_caches/physics/qm_cache.csv

PHASE1_HRS="${PHASE1_HRS:-2}"   # 30-min cadence for the first 2h, then hourly
INT1=1800                       # 30 min
INT2=3600                       # 1 hour
MAX_HRS="${MAX_HRS:-24}"        # safety stop

start=$(date +%s)
log()      { echo "[qm-watch] $(date '+%m-%d %H:%M:%S') $*" >>"$LOG"; }
alive()    { pgrep -f "$1" >/dev/null 2>&1; }                 # STOPped procs still count as alive
rows()     { [ -f "$QM_CACHE" ] && echo $(( $(wc -l < "$QM_CACHE") - 1 )) || echo 0; }
progress() { grep -cE "^[[:space:]]*\[[0-9]+/[0-9]+\]" "$QM_LOG" 2>/dev/null || echo 0; }  # per-compound lines

relaunch_qm() {
  log "CORRECT: precompute_qm not running before completion -> relaunch --resume"
  nohup env PATH="$XTB_BIN:$PATH" OMP_NUM_THREADS=4 \
    "$PY" -u precompute_qm.py --resume >>"$QM_LOG" 2>&1 &
}
relaunch_guard() {
  log "CORRECT: physics_guard.sh died -> relaunch (thermal protection)"
  nohup bash physics_guard.sh >>physics_logs/guard_restart.log 2>&1 &
}
relaunch_janitor() {
  log "CORRECT: qm_scratch_janitor.sh died -> relaunch (disk protection)"
  nohup bash qm_scratch_janitor.sh >>physics_logs/janitor.log 2>&1 &
}
relaunch_finalize() {
  log "CORRECT: finalize_after_qm.sh not running, QM complete, not DONE -> relaunch"
  nohup bash finalize_after_qm.sh >>physics_logs/finalize_after_qm_outer.log 2>&1 &
}

log "start (phase1=${PHASE1_HRS}h @30min then @1h; max ${MAX_HRS}h)  qm_cache=$(rows) compounds_logged=$(progress)"
prev=$(progress); stall=0; cyc=0
while true; do
  cyc=$((cyc+1))
  now=$(date +%s); eh=$(( (now-start)/3600 ))
  cur=$(progress); cache=$(rows); disk=$(df -g . | awk 'NR==2{print $4}')
  qm=no;  alive "precompute_qm.py"      && qm=yes
  xtb=no; pgrep -x xtb >/dev/null 2>&1  && xtb=yes
  grd=no; alive "physics_guard.sh"      && grd=yes
  jan=no; alive "qm_scratch_janitor.sh" && jan=yes
  fin=no; alive "finalize_after_qm.sh"  && fin=yes
  qm_done=no; grep -q "Done in" "$QM_LOG" 2>/dev/null && qm_done=yes
  fin_done=no; [ -f "$FIN_DONE" ] && fin_done=yes

  # 1) keep the protective combo alive
  [ "$grd" = no ] && relaunch_guard
  [ "$jan" = no ] && relaunch_janitor

  # 2) terminal: finalize finished -> everything done
  if [ "$fin_done" = yes ]; then
    log "COMPLETE: finalize done. compounds_logged=${cur} qm_cache=${cache} rows. watcher exiting."
    exit 0
  fi

  # 3) QM lifecycle
  if [ "$qm" = no ] && [ "$qm_done" = no ]; then
    relaunch_qm                                   # crashed before finishing
  elif [ "$qm" = no ] && [ "$qm_done" = yes ] && [ "$fin" = no ]; then
    relaunch_finalize                             # QM done, finalize not up -> (re)start it
  fi

  # 4) wedge detection (alive but no progress and no xtb worker)
  if [ "$qm" = yes ]; then
    if [ "$cur" -le "$prev" ] && [ "$xtb" = no ]; then
      stall=$((stall+1))
      if [ "$stall" -ge 2 ]; then
        log "CORRECT: wedged (no new compound 2 cycles, no xtb child) -> restart precompute_qm"
        pkill -f precompute_qm.py 2>/dev/null; sleep 3; relaunch_qm; stall=0
      fi
    else
      stall=0
    fi
  fi
  prev=$cur

  log "cyc${cyc} phase$([ $eh -lt $PHASE1_HRS ] && echo 1 || echo 2) qm=${qm} xtb=${xtb} guard=${grd} janitor=${jan} finalize=${fin}(done=${fin_done}) qm_done=${qm_done} compounds=${cur} cache=${cache}rows disk=${disk}GB stall=${stall}"

  [ "$eh" -ge "$MAX_HRS" ] && { log "max ${MAX_HRS}h reached -> exit"; exit 0; }
  if [ "$eh" -lt "$PHASE1_HRS" ]; then sleep "$INT1"; else sleep "$INT2"; fi
done
