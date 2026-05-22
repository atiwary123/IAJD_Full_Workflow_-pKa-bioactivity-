# Installing chemprop 1.6.1 for the IAJD bioactivity workflow

The LiON Block B inference uses chemprop 1.6.1 (an older version, pinned to match
the pretrained LNP_ML checkpoints). Installing it on a modern system requires
two compatibility patches because chemprop 1.6.1 predates numpy 2.x and PyTorch
2.6.

You only need this if you'll be predicting on novel SMILES (not already in
`lion_cache_v09.json`). For all 153 v09 SMILES + the 5 EXP264 compounds, the
cache is already populated and predictions don't need chemprop.

---

## Quick install

```bash
pip install chemprop==1.6.1
```

This installs chemprop and its dependencies (~150 MB, no torch CUDA pulled in
if you already have CPU torch).

## Apply the two compatibility patches

Save this as `patch_chemprop.py` and run it once after install:

```python
# patch_chemprop.py
"""Patch chemprop 1.6.1 for numpy 2.x + PyTorch 2.6+ compatibility."""
import os, chemprop

cp_dir = os.path.dirname(chemprop.__file__)
print(f"Patching chemprop at: {cp_dir}")

# --- Patch 1: numpy.VisibleDeprecationWarning was removed in numpy 2.0 ---
fp = os.path.join(cp_dir, 'train', 'run_training.py')
with open(fp) as f: txt = f.read()
old = 'warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)'
new = 'warnings.filterwarnings("ignore", category=getattr(np, "VisibleDeprecationWarning", DeprecationWarning))'
if old in txt:
    with open(fp, 'w') as f: f.write(txt.replace(old, new))
    print("  Patch 1 applied")
elif new in txt:
    print("  Patch 1 already applied")
else:
    print("  WARNING: Patch 1 source pattern not found")

# --- Patch 2: PyTorch 2.6 default weights_only=True breaks checkpoint loading ---
fp = os.path.join(cp_dir, 'utils.py')
with open(fp) as f: txt = f.read()
old = 'torch.load(path, map_location=lambda storage, loc: storage)'
new = 'torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)'
n = txt.count(old)
if n > 0:
    with open(fp, 'w') as f: f.write(txt.replace(old, new))
    print(f"  Patch 2 applied to {n} occurrences")
elif new in txt:
    print("  Patch 2 already applied")
else:
    print("  WARNING: Patch 2 source pattern not found")

# Verify
try:
    from chemprop.train.make_predictions import make_predictions
    print("OK — chemprop imports correctly")
except Exception as e:
    print(f"FAIL: {e}")
```

Run it:

```bash
python patch_chemprop.py
```

You should see "OK — chemprop imports correctly" at the end.

## Set the LION_REPO environment variable

The block_b_lion module finds the LiON checkpoints via the LION_REPO env var.
Two valid layouts:

**Option A — using this bundle's slim checkpoints folder:**
```bash
export LION_REPO=/path/to/SESSION_BUNDLE/3_external_models/LNP_ML_essentials
# Expected: $LION_REPO/checkpoints/cv_0/fold_0/model_0/model.pt etc.
```

**Option B — using the full LNP_ML repo (if you have it cloned somewhere):**
```bash
git clone https://github.com/jswitten/LNP_ML.git ~/LNP_ML
export LION_REPO=$HOME/LNP_ML
# Expected: $LION_REPO/data/crossval_splits/all_random_split_for_paper/cv_0/fold_0/model_0/model.pt
```

The block_b_lion.py module checks both layouts and uses whichever it finds.

To make permanent: add the `export` line to `~/.bashrc` (Linux) or `~/.zshrc` (Mac).

## Verify it works

```python
import os
os.environ['LION_REPO'] = '/path/to/your/checkpoints'

import sys; sys.path.insert(0, '/path/to/SESSION_BUNDLE/1_production')
from block_b_lion import is_lion_active, _LION_CKPTS

print(is_lion_active())   # MUST print True
print(len(_LION_CKPTS))   # MUST print 5
```

## Now run a real prediction

```python
from iajd_bioact_v2 import load_bundle, predict_bioactivity
bundle = load_bundle('/path/to/SESSION_BUNDLE/1_production/iajd_bioact_v2_v09_LION_bundle.pkl')

# A novel SMILES not in any existing cache
result = predict_bioactivity(
    'YOUR_NEW_SMILES_HERE',
    bundle,
)
print(f"flux_total = {result['log10_flux_total']}, PI90 = {result['log10_flux_total_PI90']}")
print(f"lion_active = {result['lion_active']}")  # MUST be True
```

The first prediction on a new SMILES takes ~10-15 seconds (5-checkpoint × 6-tissue
LiON ensemble). Subsequent predictions on the same SMILES are instant (cached).

---

## Failure modes

**`No module named chemprop`** after install
→ pip and python don't match. Run `which pip` and `which python` to verify.
  In conda envs, use `python -m pip install chemprop==1.6.1` instead.

**`AttributeError: module 'numpy' has no attribute 'VisibleDeprecationWarning'`**
→ Patch 1 didn't take. Re-run patch_chemprop.py.

**`UnpicklingError: Weights only load failed`** when loading checkpoints
→ Patch 2 didn't take. Re-run patch_chemprop.py.

**`is_lion_active()` returns False even after install**
→ LION_REPO env var not set, or pointing at the wrong directory. Verify with
  `echo $LION_REPO` (Mac/Linux) or `echo %LION_REPO%` (Windows).

**Predictions slow on first run**
→ Normal. The 5-checkpoint ensemble loads each model into memory once, then runs
  all 6 tissue contexts. Subsequent runs in the same Python session are faster.

---

## Disk footprint after install

- chemprop + dependencies: ~150 MB
- LiON checkpoints (in this bundle): ~32 MB
- Per-prediction cache growth: ~300 bytes per new compound

Total: ~200 MB for the LiON workflow on top of whatever you already have for
the bundle's basic dependencies (rdkit, xgboost, sklearn, etc.).
