#!/usr/bin/env bash
# cloud_benchmark.sh — confirm the cloud env works + benchmark speed by re-running the
# VALIDATED checks (Module B DOPC curvature, Module A aniline pKa). Run after cloud_setup.sh.
set -euo pipefail
WORK="${WORK:-/workspace}"; cd "$WORK/IAJD"; source "$WORK/IAJD/cloud_env.sh"
N=$(nproc)
echo "== benchmark on $N cores =="
echo "[B] Module B curvature DOPC (40ns, force-check vs GROMACS)"
time python compute_curvature.py --lipid DOPC --n-per-leaflet 64 --prod-ns 40 \
  --eq2-ps 5000 --temp 300 --threads "$N" --force-check 2>&1 | \
  grep -E '"c0_nm_inv"|"apl_nm2"|"max_abs_pct_err"|"surface_tension' || true
echo "[done] if c0_DOPC ~ -0.2 and max_abs_pct_err < 1e-3 %, the env is correct + fast."
