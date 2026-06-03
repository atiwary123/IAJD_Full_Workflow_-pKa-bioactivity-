#!/usr/bin/env bash
# qm_scratch_janitor.sh — keep /tmp from filling during long QM runs.
#
# qm_descriptors.py leaves ~150 MB of xtb scratch per compound (mkdtemp dirs
# named qm_*). Over ~254 compounds that's ~38 GB — it would fill the disk and
# crash the run partway. This sweeps scratch dirs every 10 min.
#
# FIX 2026-06-02: the old `-maxdepth 1 -mmin +30` checked only the PARENT dir's
# mtime. Huge twin/G1-Janus molecules run a single xtb call >30 min writing only
# to SUBdirs, so the parent looked "idle" and got deleted mid-run (FileNotFound,
# 3 lost compounds). Now we delete a qm_* dir only if NOTHING anywhere inside it
# was modified in the last 60 min — protects live slow compounds, still reclaims
# finished/abandoned scratch.
set -uo pipefail
echo "[janitor] start $(date)  sweeping qm_* scratch (recursive 60min-idle), every 10min"
while true; do
  before=$(du -shc "${TMPDIR:-/tmp}"/qm_* /tmp/claude-501/qm_* 2>/dev/null | tail -1 | awk '{print $1}')
  for d in "${TMPDIR:-/tmp}"/qm_* /tmp/claude-501/qm_*; do
    [ -d "$d" ] || continue
    if [ -z "$(find "$d" -mmin -60 -print -quit 2>/dev/null)" ]; then
      rm -rf "$d" 2>/dev/null
    fi
  done
  after=$(du -shc "${TMPDIR:-/tmp}"/qm_* /tmp/claude-501/qm_* 2>/dev/null | tail -1 | awk '{print $1}')
  echo "[janitor] $(date '+%H:%M')  scratch ${before:-0} -> ${after:-0}"
  sleep 600
done
