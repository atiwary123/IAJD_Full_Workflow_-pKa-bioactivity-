# IAJD Bioactivity v14 — Release Files (REAL LION + ADMET edition)

**Status:** v14.0 production bundle uses REAL LION (chemprop 1.6.1 + 5-CV ensemble) + REAL ADMET-AI 2.0.1 predictions for all training SMILES. Proxy-edition has been superseded.
**Training data:** v13_clean (335 rows, **274 unique IAJDs**, 7 families).

**Headline (in-sample LOO):** v14.0 pooled MAE 0.4006 (baseline 0.4342, −7.7%).
**Headline (HONEST nested CV):** v14.0 pooled MAE **0.4032** (optimism only −0.003).

The in-sample LOO and honest CV agree to within 0.003 MAE units — **v14.0 numbers are reliable predictors of held-out performance**. v14.3 has lower in-sample (0.3931) but −0.032 of selection-bias optimism (honest = 0.4251). **For the LSM application and production predictions, cite 0.4032.**

## Read first (in order)
- `MODEL_CARD_v14.md` — full writeup, real-feature gate decisions, v14.0 → v14.3 progression
- `honest_summary_v140_REAL.json` — v14.0 real-feature honest CV metrics
- `honest_summary_v143_REAL.json` — v14.3 real-feature honest CV metrics
- `bioact_v14_loo.json` — v14.0 LOO breakdown per family
- `gate_*.json` — per-(family, block) real-feature gate decisions

## Inference (use v14.0 with real features)

```python
import sys
sys.path.insert(0, '/mnt/user-data/outputs/bioact_v14')
from predict_v14_real import predict, predict_batch

# Auto-fetches real LION + ADMET for new SMILES (~10s each for LION)
r = predict('<SMILES>')
print(r['log10_flux_total'], r['log10_flux_total_PI90'])
print(r['family'], r['confidence_tier'], r['block_B_real'])
print(r['warnings'])

# Batch
df = predict_batch(['<SMILES1>', '<SMILES2>'])
```

Or for faster inference using proxies (slight accuracy loss):
```python
from iajd_bioact_v14 import load_bundle, predict_bioactivity
b = load_bundle()
r = predict_bioactivity('<SMILES>', b)  # uses RDKit proxies if cache misses
```

Honest metrics accessible from the bundle:
```python
import pickle
b = pickle.load(open('bioact_v14_bundle.pkl', 'rb'))
print(b['metrics']['v14_honest_pooled_mae'])     # 0.4032
print(b['metrics']['v14_honest_per_family'])      # per-family
```

## Real-feature dependencies

For LION inference:
- LNP_ML repo cloned to `/home/claude/LNP_ML`
- chemprop 1.6.1 installed in `/home/claude/lion_env` (with numpy 2.0 + torch 2.6 patches applied)
- Checkpoints at `LNP_ML/data/crossval_splits/all_random_split_for_paper/cv_X/fold_0/model_0/model.pt`

For ADMET inference:
- `pip install --break-system-packages admet-ai` (pulls chemprop 2.x + torch 2.x + lightning)

If those aren't present, the pipeline silently falls back to RDKit proxies and flags `block_B_real: False` in prediction output.

## Retrain from scratch with REAL features

```bash
# Build caches (1 + 13 minutes)
python3 build_admet_cache.py                                # 13 seconds
python3 build_lion_cache.py                                  # ~1 min  (uses lion_env venv)

# Assemble features (real caches auto-picked up)
python3 v14_checkpoint_runner.py --phase assemble --force    # ~1 min
python3 v14_checkpoint_runner.py --phase base_loo            # ~2 min
python3 v14_checkpoint_runner.py --phase alpha_sweep         # instant

# Per-family gates
for FAM in sSS-Nonsym PE-Tris GA-Tris G1-Janus-Dendrimer Dialkoxybenzyl HTM-Dendrimer TT-Dendrimer; do
  for BLK in B C; do
    python3 v14_checkpoint_runner.py --phase gate --family "$FAM" --block $BLK   # ~2 min each
  done
done

python3 v14_checkpoint_runner.py --phase final_loo            # ~2 min
python3 v14_checkpoint_runner.py --phase bundle               # instant
```

## Retrain research variants (v14.1/14.2/14.3)

```bash
python3 code/bioact_v14_1_maximal.py     # HPO + per-fam direct + stack → v14.1
python3 code/bioact_v14_2_perfam_hpo.py  # finer HPO → v14.2
python3 code/bioact_v14_3_best.py        # best-of-1+2 per family → v14.3
python3 code/bioact_v14_4_honest_check.py # 5-fold nested CV honest MAE for v14.3
python3 code/honest_check_v140.py        # 5-fold nested CV honest MAE for v14.0
```

## Variant guide

| Bundle | In-sample MAE | Honest MAE | When to use |
|---|---|---|---|
| `bioact_v14_bundle.pkl` (v14.0) | 0.4006 | **0.4032** | **best for production** |
| `bioact_v14_1_bundle.pkl` | 0.4030 | (not measured) | per-family stack, similar to v14.0 |
| `bioact_v14_2_bundle.pkl` | 0.3998 | (not measured) | + per-family HPO |
| `bioact_v14_3_bundle.pkl` | 0.3931 | 0.4251 | research-only; high optimism |

## Quickref of v14.0 production architecture per family

```
Family             n    α    Block B  Block C  honest MAE
sSS-Nonsym         175  0.8  ON       ON       0.416
PE-Tris             51  1.0  ON       ON       0.405
GA-Tris             50  1.0  ON       OFF      0.410
G1-Janus            26  1.0  OFF      ON       0.438
Dialkoxybenzyl      18  0.5  ON       ON       0.185
HTM-Dendrimer       10  1.0  ON       ON       0.411
TT-Dendrimer         5  1.0  OFF      ON       0.452
```

LION (Block B) helps 5/7 families with real predictions (was 3/7 with proxies).
ADMET (Block C) helps 6/7 families with real predictions (was 3/7 with proxies).
