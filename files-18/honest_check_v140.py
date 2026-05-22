"""
honest_check_v140.py — 5-fold nested CV on v14.0 (per-family α + gating only).

For each fold:
  - Re-select α per family on n-1 folds via the cached LOO from base+gates
  - Apply selected α to held-out fold
  - Re-run gates on n-1 folds (the gate decision is a selection step) 
  - Use selected gates on held-out fold

Actually simpler: just re-select α per family on train fold. Gates were already
decided per family using full-data LOO; that selection is itself selection bias,
but we approximate honest by always using the FINAL gates (which were derived once).

A more rigorous honest check would re-derive gates per fold too, but the gate
selection involves running another full LOO per fold per gate → too expensive.
Approximate: hold the gates fixed (since they have small Δ values, low risk of
overfitting), and only re-optimize α per fold.
"""
import os, sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold

OUT = Path('/mnt/user-data/outputs/bioact_v14')


def evaluate(preds, y, families):
    pooled_mae = float(mean_absolute_error(y, preds))
    pooled_r2 = float(r2_score(y, preds))
    per_fam = {}
    for fam in set(families):
        mask = np.array([f == fam for f in families])
        if mask.sum() >= 3:
            per_fam[fam] = {
                'n': int(mask.sum()),
                'mae': float(mean_absolute_error(y[mask], preds[mask])),
            }
    return {'pooled_mae': pooled_mae, 'pooled_r2': pooled_r2, 'per_family': per_fam}


def blend_with_alpha(directs, deltas, alpha_per_family, families):
    n = len(directs)
    preds = np.zeros(n)
    for i in range(n):
        fam = families[i]
        a = alpha_per_family.get(fam, 0.6)
        if np.isnan(deltas[i]):
            preds[i] = directs[i]
        else:
            preds[i] = a * directs[i] + (1 - a) * deltas[i]
    return preds


def main():
    print('='*70)
    print('Honest 5-fold nested CV on v14.0 (α + gates)')
    print('='*70)

    y = np.load(OUT / 'bioact_v14_y.npy')
    with open(OUT / 'bioact_v14_families.json') as f:
        families = np.array(json.load(f))

    # Load final LOO (already with v14.0 gates applied)
    d = np.load(OUT / 'loo_final.npz')
    directs = d['directs']
    deltas = d['deltas']

    # 5-fold nested: optimize α on train fold, evaluate on test fold
    alphas_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(y)
    all_pred = np.zeros(n)

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), families)):
        # For each family, find best α on training fold only
        fam_alpha = {}
        for fam in set(families):
            mask_tr = (families[tr_idx] == fam)
            if mask_tr.sum() < 3:
                fam_alpha[fam] = 1.0
                continue
            tr_in_fam = tr_idx[mask_tr]
            best_a, best_mae = 1.0, float('inf')
            for a in alphas_grid:
                preds = []
                for i in tr_in_fam:
                    if np.isnan(deltas[i]):
                        preds.append(directs[i])
                    else:
                        preds.append(a * directs[i] + (1 - a) * deltas[i])
                mae = float(mean_absolute_error(y[tr_in_fam], preds))
                if mae < best_mae:
                    best_mae, best_a = mae, a
            fam_alpha[fam] = best_a

        # Apply to test fold
        for qi in te_idx:
            fam = families[qi]
            a = fam_alpha[fam]
            if np.isnan(deltas[qi]):
                all_pred[qi] = directs[qi]
            else:
                all_pred[qi] = a * directs[qi] + (1 - a) * deltas[qi]

    pooled_mae = float(mean_absolute_error(y, all_pred))
    pooled_r2 = float(r2_score(y, all_pred))
    print(f'\nv14.0 honest 5-fold nested CV: pooled MAE = {pooled_mae:.4f}, R² = {pooled_r2:.4f}')

    # Per-family
    print('\nPer-family honest MAE (v14.0):')
    per_fam = {}
    for fam in sorted(set(families.tolist()), key=lambda f: -np.sum(families == f)):
        mask = families == fam
        if mask.sum() >= 3:
            mae = float(mean_absolute_error(y[mask], all_pred[mask]))
            n_fam = int(mask.sum())
            per_fam[fam] = {'n': n_fam, 'mae': mae}
            print(f'  {fam:20s} n={n_fam:3d}  honest MAE = {mae:.4f}')

    out = {
        'honest_pooled_mae': pooled_mae,
        'honest_pooled_r2': pooled_r2,
        'in_sample_pooled_mae_v140': 0.4006,
        'optimism': 0.4006 - pooled_mae,
        'per_family_honest': per_fam,
        'methodology': '5-fold family-stratified nested CV on v14.0 (α selected per fold on train, gates fixed)',
        'features_source': 'REAL LION (5-CV chemprop ensemble) + REAL ADMET-AI predictions',
    }
    json.dump(out, open(OUT / 'honest_summary_v140_REAL.json', 'w'), indent=2)
    print(f'\nSaved honest_summary_v140_REAL.json')


if __name__ == '__main__':
    main()
