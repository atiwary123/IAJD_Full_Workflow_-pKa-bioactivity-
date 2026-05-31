#!/usr/bin/env bash
# qm_scratch_janitor.sh — keep /tmp from filling during long QM runs.
#
# qm_descriptors.py leaves ~150 MB of xtb scratch per compound (mkdtemp dirs
# named qm_*). Over ~254 compounds that's ~38 GB — it would fill the disk and
# crash the run partway. This sweeps scratch dirs untouched for >30 min every
# 10 min. Active compounds write to their dir every <15 min (per-xtb-call cap is
# 900 s), so a 30-min-idle dir is always a finished/abandoned one — safe to drop.
#
# Runs detached alongside the QM workers. Logs a one-line reclaim summary.
set -uo pipefail
echo "[janitor] start $(date)  sweeping qm_* scratch >30min idle, every 10min"
while true; do
  before=$(du -shc "${TMPDIR:-/tmp}"/qm_* /tmp/claude-501/qm_* 2>/dev/null | tail -1 | awk '{print $1}')
  find "${TMPDIR:-/tmp}" /tmp/claude-501 -maxdepth 1 -type d -name 'qm_*' -mmin +30 \
       -exec rm -rf {} + 2>/dev/null
  after=$(du -shc "${TMPDIR:-/tmp}"/qm_* /tmp/claude-501/qm_* 2>/dev/null | tail -1 | awk '{print $1}')
  echo "[janitor] $(date '+%H:%M')  scratch ${before:-0} -> ${after:-0}"
  sleep 600
done
