#!/usr/bin/env bash
# cloud_setup.sh — set up the IAJD physics-design pipeline on a fresh Linux box
# (RunPod / Vast / Lambda / any Ubuntu+CUDA host). One command, idempotent.
#
# Run it on the POD's web terminal or over SSH:
#   cd /workspace && \
#   curl -fsSL https://raw.githubusercontent.com/atiwary123/IAJD_Full_Workflow_-pKa-bioactivity-/physics-overnight/cloud_setup.sh -o cloud_setup.sh && \
#   bash cloud_setup.sh
#
# Optional env vars:
#   WORK=/workspace        # persistent dir (mount your network volume here for persistence)
#   GIT_URL=...            # if the repo is PRIVATE, pass an authed URL (PAT): https://<token>@github.com/...
#   BUILD_GPU_GMX=1        # also build CUDA GROMACS for the atomistic Module E (needs nvcc)
set -euo pipefail

WORK="${WORK:-/workspace}"
GIT_URL="${GIT_URL:-https://github.com/atiwary123/IAJD_Full_Workflow_-pKa-bioactivity-.git}"
BRANCH="${BRANCH:-physics-overnight}"
mkdir -p "$WORK"; cd "$WORK"
echo "== IAJD cloud setup ==  WORK=$WORK  $(nproc) cores  $(free -g | awk '/Mem/{print $2}')GB RAM"

# 1) micromamba (self-contained; no root needed)
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
if [ ! -x "$WORK/bin/micromamba" ]; then
  echo "[1] installing micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORK" bin/micromamba
fi
MM="$WORK/bin/micromamba"

# 2) conda env: CPU GROMACS + xTB + AmberTools + the scientific stack
if ! "$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd gmx --version >/dev/null 2>&1; then
  echo "[2] creating conda env 'iajd' (gromacs, xtb, ambertools, rdkit, mdanalysis, ...)"
  "$MM" create -y -r "$MAMBA_ROOT_PREFIX" -n iajd -c conda-forge \
    python=3.11 "gromacs=2024.*" xtb ambertools \
    "numpy<2" scipy pandas mdanalysis rdkit scikit-learn xgboost pyarrow \
    matplotlib openpyxl joblib git
fi
# 3) pip-only deps
echo "[3] pip deps (pygam, scikit-optimize, pymbar, vermouth, acpype, tqdm)"
"$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd pip install -q pygam scikit-optimize pymbar tqdm vermouth acpype

# 4) clone the repo (code + vendored Martini/titratable FFs + dataset)
if [ ! -d "$WORK/IAJD/.git" ]; then
  echo "[4] cloning repo ($BRANCH)"
  git clone -b "$BRANCH" "$GIT_URL" "$WORK/IAJD"
else
  echo "[4] repo present; pulling latest"; git -C "$WORK/IAJD" pull --ff-only || true
fi

# 5) point the IAJD code at the cloud engines (gmx/xtb live in the conda env)
GMXBIN="$MAMBA_ROOT_PREFIX/envs/iajd/bin"
cat > "$WORK/IAJD/cloud_env.sh" <<EOF
# source this before running anything: source $WORK/IAJD/cloud_env.sh
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
export PATH="$GMXBIN:\$PATH"
export GMX_CMD="$GMXBIN/gmx"
export PYTHONPATH="$WORK/IAJD:\$PYTHONPATH"
alias py="$GMXBIN/python"
EOF

# 6) optional: build CUDA GROMACS for the atomistic Module E (the one GPU win)
if [ "${BUILD_GPU_GMX:-0}" = "1" ]; then
  echo "[6] building CUDA GROMACS (atomistic Module E) — ~15 min on $(nproc) cores"
  "$MM" install -y -r "$MAMBA_ROOT_PREFIX" -n iajd -c conda-forge cmake cuda-toolkit gcc_linux-64 gxx_linux-64 || true
  cd "$WORK"; [ -f gromacs-2024.4.tar.gz ] || curl -fsSLO https://ftp.gromacs.org/gromacs/gromacs-2024.4.tar.gz
  tar xf gromacs-2024.4.tar.gz; cd gromacs-2024.4; mkdir -p build; cd build
  "$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd cmake .. -DGMX_GPU=CUDA -DGMX_BUILD_OWN_FFTW=ON \
    -DCMAKE_INSTALL_PREFIX="$WORK/gromacs-gpu" && \
  "$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd make -j"$(nproc)" && \
  "$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd make install && \
  echo "  GPU gmx -> source $WORK/gromacs-gpu/bin/GMXRC (use for atomistic Module E)" || \
  echo "  [warn] GPU GROMACS build failed; CPU gmx still works for the CG panel"
fi

# 7) optional: NN transfer deps for the AGILE GNN-encoder UPGRADE (chemprop + torch-geometric).
#    The runnable Morgan+GP transfer path (nn/train_transfer.py --encoder morgan) needs NOTHING
#    extra — rdkit + scikit-learn are already in the iajd env.
if [ "${BUILD_NN:-0}" = "1" ]; then
  echo "[7] installing NN GNN-encoder deps (chemprop, torch-geometric)"
  "$MM" run -r "$MAMBA_ROOT_PREFIX" -n iajd pip install -q chemprop torch torch-geometric || \
    echo "  [warn] GNN deps failed; the Morgan+GP transfer path still runs without them"
fi

cat <<EOF

================ SETUP DONE ================
Activate + sanity check:
  source $WORK/IAJD/cloud_env.sh
  cd $WORK/IAJD
  gmx --version | head -3
  python -c "import MDAnalysis, rdkit, pygam, skopt, pymbar; print('python stack ok')"

Benchmark the box (validated Module B/A on this hardware):
  bash cloud_benchmark.sh

Engines: gmx (CPU, $(nproc) cores), xtb (CPU), antechamber/acpype (Module E). The $(nproc)-core
EPYC runs the parallel CG-MD + QM panel; build_gpu (BUILD_GPU_GMX=1) accelerates atomistic Module E.
===========================================
EOF
