# v14 → v15 Session Prompt

Pick this up cleanly in a new chat by pasting the prompt below and attaching:
- `bioact_v14_bundle.pkl` (production bundle, v14.0)
- `bioact_v14_3_bundle.pkl` (research variant)
- `predict_v14_real.py`, `extend_caches.py`, `iajd_bioact_v14.py`, `bioact_v14_pipeline.py`
- `MODEL_CARD_v14.md`, `README.md`, `HONEST_SUMMARY.md`
- `honest_summary_v140_REAL.json`, `honest_summary_v143_REAL.json`
- `lion_cache_v13.json`, `admet_cache_v13.json`
- `gates_summary.json`

---

## Prompt

> Continuing IAJD bioactivity from v14.0 (REAL LION + ADMET edition). Attached are the production files.
>
> v14.0 was retrained with real LION (5-CV chemprop ensemble + 6 tissues) + real ADMET-AI (10 features) instead of RDKit proxies. Key numbers on v13 (335 rows, 274 unique IAJDs):
>
> - Baseline (α=0.6 uniform, all blocks on): LOO MAE 0.4342
> - **v14.0 production (per-family α + per-family gates): LOO MAE 0.4006, honest 5-fold CV MAE 0.4032** (optimism -0.003, essentially zero)
> - v14.3 research variant (full stack + per-fam HPO): in-sample 0.3931, honest 0.4251 (high optimism)
>
> Real LION/ADMET integration confirmed the model card's prediction: LION helps 5/7 families with real predictions (was 3/7 with proxies). ADMET helps 6/7 families (was 3/7). Biggest gate flips:
> - GA-Tris Block B: OFF → ON (real LION helps by Δ=+0.019 MAE)
> - PE-Tris Block B: OFF → ON (real LION helps by Δ=+0.031 MAE)
> - HTM-Dendrimer Block C: OFF → ON (real ADMET helps by Δ=+0.023 MAE)
>
> Per-family honest MAE (v14.0):
>   sSS-Nonsym (n=175): 0.416
>   PE-Tris (n=51):     0.405
>   GA-Tris (n=50):     0.410
>   G1-Janus (n=26):    0.438
>   Dialkoxybenzyl (n=18): 0.185 (best)
>   HTM-Dendrimer (n=10): 0.411
>   TT-Dendrimer (n=5): 0.452
>
> Inference: `predict_v14_real.predict(smiles)` auto-fetches real LION+ADMET for new SMILES (~10s/SMILES for LION).
>
> Today's priority is one of:
>
> 1. **EXP_264 predictions with v14.0 real-features.** Run the 5 novel GA-Tris compounds from EXP_264-1.cdxml (IAJD 300, 365, 366, 367, 369; from v9.1 pKa output) through v14.0. We have their pKa predictions already; need flux predictions + 90% PIs + neighbor surfacing.
>
> 2. **Per-organ Stage B retrain.** Currently we only predict log10_flux_total. v13 has per-organ log10 fluxes for spleen, liver, lung, LN (~50-77 samples per organ — much larger than v09's per-organ samples). Train 4 per-organ regressors using the v14.0 architecture, run honest CV per organ.
>
> 3. **Stage A organ-selectivity classifier.** v13 has organ_dominant labels for ~280 rows. Train a multiclass classifier returning P(spleen) + P(liver) + P(lung) + P(LN) for any new IAJD.
>
> 4. **Scaffold-out CV.** Replace random LOO with Bemis-Murcko scaffold-out splits. Stricter generalization test — expected honest MAE to rise to 0.45-0.50 if scaffolds are well-distributed.
>
> 5. **MAPIE jackknife+ PIs.** Replace heuristic tier-based PIs (±0.20/0.30/0.45) with calibrated jackknife+ intervals using the per-fold LOO residuals already cached.
>
> 6. **Fine-tune LION on v13.** Spec §4.2 estimates -0.05 to -0.10 log-MAE gain from fine-tuning LION's chemprop weights on the 220 in-distribution IAJDs. Requires ~12 GPU hours (CPU-only would take ~5 days).
>
> 7. **Counterfactual API.** "What if I extend chain X by N carbons" — propose modifications and predict their bioactivity to help guide synthesis priorities.
>
> Which priority do you want to tackle first?

---

## Reproducing v14.0 from scratch (~30 min total)

```bash
# Caches (one-time, 14 min)
cd /mnt/user-data/outputs/bioact_v14
python3 build_admet_cache.py
/home/claude/lion_env/bin/python3 build_lion_cache.py

# v14.0 pipeline (~15 min, with checkpoints)
python3 v14_checkpoint_runner.py --phase assemble --force
python3 v14_checkpoint_runner.py --phase base_loo
python3 v14_checkpoint_runner.py --phase alpha_sweep
for FAM in sSS-Nonsym PE-Tris GA-Tris G1-Janus-Dendrimer Dialkoxybenzyl HTM-Dendrimer TT-Dendrimer; do
  for BLK in B C; do
    python3 v14_checkpoint_runner.py --phase gate --family "$FAM" --block $BLK
  done
done
python3 v14_checkpoint_runner.py --phase final_loo
python3 v14_checkpoint_runner.py --phase bundle

# Honest CV
python3 code/honest_check_v140.py
```

## Real-feature dependencies (pre-installed in this session)

- `/home/claude/LNP_ML/` (cloned from github.com/jswitten/LNP_ML, 386 MB)
- `/home/claude/lion_env/` (venv with chemprop 1.6.1 patched for numpy 2.0 + torch 2.6)
- `admet-ai 2.0.1` (in main pip env)
- `/home/claude/lion_env/lib/.../chemprop/train/run_training.py` line 8 patched (numpy.VisibleDeprecationWarning removed)
- `/home/claude/lion_env/lib/.../chemprop/utils.py` 4 torch.load calls patched (weights_only=False)
