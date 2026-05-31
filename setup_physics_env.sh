#!/usr/bin/env bash
# setup_physics_env.sh — rebuild the predictive-physics engines after a reboot.
#
# WHY THIS EXISTS
#   The original build installed xtb + gromacs via micromamba under
#   /tmp/mamba_root. macOS wipes /tmp on reboot, which deleted both binaries and
#   silently degraded the no-proxy pipeline to all-NaN (cache misses route to
#   the emulator, and with no engines even fresh compounds can't be computed).
#   This script reinstalls them into PERSISTENT locations so a restart can never
#   do that again.  See docs/PREDICTIVE_PHYSICS_BUILD.md for the full story.
#
# WHAT IT INSTALLS (idempotent — safe to re-run)
#   xtb 6.7.1   ->  micromamba env "xtb_env" rooted at ~/micromamba (conda-forge)
#   gromacs     ->  Homebrew  /opt/homebrew/bin/gmx
#   (Python deps already live in ./.venv — rdkit, MDAnalysis, xgboost, pyarrow.)
#
# AFTER THIS, verify with:  ./.venv/bin/python physics_status.py
set -euo pipefail

MAMBA_ROOT="${MAMBA_ROOT:-$HOME/micromamba}"
MM="/opt/homebrew/bin/micromamba"

echo "== predictive-physics engine setup =="
echo "   mamba root: $MAMBA_ROOT"

# 1) Homebrew engines: gromacs (gmx) + micromamba.
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew not found. Install it from https://brew.sh first." >&2
  exit 1
fi
for f in gromacs micromamba; do
  if brew list --formula "$f" >/dev/null 2>&1; then
    echo "  [ok]  $f already installed"
  else
    echo "  [..]  brew install $f"
    brew install "$f"
  fi
done

# 2) Persistent xtb env (conda-forge xtb).
if "$MM" run -n xtb_env -r "$MAMBA_ROOT" xtb --version >/dev/null 2>&1; then
  echo "  [ok]  xtb_env already provides xtb"
else
  echo "  [..]  creating xtb_env (conda-forge xtb) at $MAMBA_ROOT"
  "$MM" create -y -r "$MAMBA_ROOT" -n xtb_env -c conda-forge xtb
fi

# 3b) Physics-design assets: Martini-3 lipid topologies + martinize2 (Module B/E).
LIPIDOME="$(dirname "$0")/martini/lipidome"
mkdir -p "$LIPIDOME"
fetch() {  # url dest
  if [ -s "$2" ]; then echo "  [ok]  $(basename "$2")"; else
    echo "  [..]  fetch $(basename "$2")"
    curl -fsSL "$1" -o "$2" || echo "  [WARN] could not fetch $1 (offline?)"
  fi
}
fetch "https://raw.githubusercontent.com/marrink-lab/TS2CG1.1/master/Tutorials/files/itp/martini3/martini_v3.0_phospholipids.itp" \
      "$LIPIDOME/martini_v3.0_phospholipids.itp"
fetch "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters/main/ITPs/martini_v3.0.0_phospholipids_PE_v2.itp" \
      "$LIPIDOME/martini_v3.0.0_phospholipids_PE_v2.itp"
# martinize2 (vermouth) into the project .venv for Module E CG mapping.
VENV_PY="$(dirname "$0")/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
  if "$VENV_PY" -c "import vermouth" >/dev/null 2>&1; then
    echo "  [ok]  vermouth (martinize2) present in .venv"
  else
    echo "  [..]  pip install vermouth into .venv"
    "$VENV_PY" -m pip install --quiet vermouth || echo "  [WARN] vermouth install failed"
  fi
fi

# 3) Verify both engines actually run.
echo "== verify =="
"$MM" run -n xtb_env -r "$MAMBA_ROOT" xtb --version 2>&1 | grep -i "xtb version" \
  || { echo "xtb FAILED to run" >&2; exit 1; }
/opt/homebrew/bin/gmx --version 2>&1 | grep -i "GROMACS version" \
  || { echo "gmx FAILED to run" >&2; exit 1; }

cat <<EOF

Engines ready. The code resolves them via:
  qm_descriptors.DEFAULT_XTB_CMD = $MM run -n xtb_env --root-prefix $MAMBA_ROOT xtb
  run_selfassembly.DEFAULT_GMX   = /opt/homebrew/bin/gmx  (found via 'which gmx')

NOTE: qm_descriptors.py hardcodes the root-prefix path. If \$HOME is not
/Users/aryamantiwary, also update DEFAULT_XTB_CMD there (or export XTB_CMD).

Next:  bash run_physics_overnight.sh        # resume the full build
   or  ./.venv/bin/python physics_status.py # one-glance health check
EOF

# 4) Module E atomistic-reference toolchain: AmberTools (antechamber/GAFF) + acpype.
if "$MM" run -n ambertools -r "$MAMBA_ROOT" which antechamber >/dev/null 2>&1; then
  echo "  [ok]  ambertools env present (antechamber/acpype)"
else
  echo "  [..]  creating ambertools env (conda-forge ambertools + acpype)"
  "$MM" create -y -r "$MAMBA_ROOT" -n ambertools -c conda-forge ambertools acpype
fi
