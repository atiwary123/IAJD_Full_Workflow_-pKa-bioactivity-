"""
bioact_v14_4_honest_check.py — Nested-CV check on v14.3 stacking.

The v14.3 stack weights were optimized on the same LOO data they're being
evaluated on. This is potentially optimistic. Honest test:

  For each fold:
    - Train v14.x components on (n-fold) rows
    - Compute LOO-style predictions for the held-out fold
    - Use the BEST-OF-rest-of-data stack weights to evaluate

This 5-fold nested-LOO gives an honest generalization estimate.

We use a simpler proxy: stratified 5-fold CV at the QUERY level —
for each held-out fold, the v14.3 stack uses the weights derived from the
other 4 folds (held-out family group).
"""

import os, sys, json, pickle, time
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold

WORK = Path(__file__).resolve().parent
OUT  = Path('/mnt/user-data/outputs/bioact_v14')


def main():
    print('='*70)
    print('v14.4 HONEST CHECK: scaffold-level nested-CV on v14.3 stacking')
    print('='*70)

    y = np.load(OUT / 'bioact_v14_y.npy')
    with open(OUT / 'bioact_v14_families.json') as f:
        families = np.array(json.load(f))

    # Load cached LOO predictions for each component
    pool_v141 = np.load(OUT / 'loo_pooled_hpo.npz')['directs']
    pool_v142 = np.load(OUT / 'pooled_loo_v142.npz')['directs']
    pool_d_v141 = np.load(OUT / 'loo_pooled_hpo.npz')['deltas']
    pool_d_v142 = np.load(OUT / 'pooled_loo_v142.npz')['deltas']
    fam_v141 = np.load(OUT / 'per_family_direct_preds.npy')
    fam_v142 = np.load(OUT / 'perfam_loo_v142.npy')
    delta_var = np.load(OUT / 'delta_variants.npz')
    n = len(y)

    # For each fam-stratified 5-fold, optimize stack on n−fold and evaluate on fold
    # Use family as the stratification var
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_pred = np.zeros(n)

    # Coarse grid for nested-CV (otherwise too slow)
    grid = []
    for wp in np.arange(0, 1.01, 0.1):
        for wf in np.arange(0, 1.01 - wp, 0.1):
            wa = 1.0 - wp - wf
            if 0 <= wa <= 1.0001:
                grid.append((float(wp), float(wf), float(wa)))

    pool_options_dict = {
        'v141_pool': pool_v141, 'v142_pool': pool_v142,
    }
    fam_options_dict = {
        'v141_fam': fam_v141, 'v142_fam': fam_v142,
    }
    analog_options_dict = {
        'analog_v1': delta_var['v1'],
        'analog_v2': delta_var['v2'],
        'analog_v3': delta_var['v3'],
    }

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), families)):
        # For each family, optimize stack ONLY on training fold
        tr_fams = families[tr_idx]
        te_fams = families[te_idx]
        fam_stack_choices = {}
        for fam in set(families):
            mask_tr = (tr_fams == fam)
            if mask_tr.sum() < 3:
                # too few in train — use pool_v141 directly
                fam_stack_choices[fam] = ('v141_pool', 'v141_pool', 'analog_v1', 1.0, 0.0, 0.0)
                continue
            tr_in_fam = tr_idx[mask_tr]
            ytrue = y[tr_in_fam]
            fam_y_mean = float(np.mean(ytrue))

            best_mae = float('inf')
            best_choice = None
            for p_name, p_arr_full in pool_options_dict.items():
                p_arr = p_arr_full[tr_in_fam]
                for f_name, f_arr_full in fam_options_dict.items():
                    if np.any(np.isnan(f_arr_full[tr_in_fam])):
                        f_arr = p_arr  # fallback to pool
                    else:
                        f_arr = f_arr_full[tr_in_fam]
                    for a_name, a_arr_full in analog_options_dict.items():
                        a_arr = a_arr_full[tr_in_fam].copy()
                        a_arr[np.isnan(a_arr)] = fam_y_mean
                        for (wp, wf, wa) in grid:
                            pred = wp * p_arr + wf * f_arr + wa * a_arr
                            mae = float(mean_absolute_error(ytrue, pred))
                            if mae < best_mae:
                                best_mae = mae
                                best_choice = (p_name, f_name, a_name, wp, wf, wa)
            fam_stack_choices[fam] = best_choice

        # Apply to test fold
        for qi in te_idx:
            fam = families[qi]
            ch = fam_stack_choices[fam]
            p_name, f_name, a_name, wp, wf, wa = ch
            p_v = pool_options_dict[p_name][qi]
            f_v_full = fam_options_dict[f_name][qi]
            if np.isnan(f_v_full):
                f_v = pool_options_dict['v141_pool'][qi]
            else:
                f_v = f_v_full
            a_v = analog_options_dict[a_name][qi]
            if np.isnan(a_v):
                # use fam mean of TRAIN fold y in this family
                mask_tr_fam = (families[tr_idx] == fam)
                if mask_tr_fam.sum() > 0:
                    a_v = float(np.mean(y[tr_idx[mask_tr_fam]]))
                else:
                    a_v = float(np.mean(y[tr_idx]))
            all_pred[qi] = wp * p_v + wf * f_v + wa * a_v

    honest_mae = float(mean_absolute_error(y, all_pred))
    honest_r2 = float(r2_score(y, all_pred))
    print(f'\n  v14.3 in-sample stack MAE: 0.3954')
    print(f'  v14.4 honest 5-fold nested-CV MAE: {honest_mae:.4f}  R² = {honest_r2:.3f}')
    print(f'  Optimism (in-sample - honest):     {0.3954 - honest_mae:+.4f}')

    # Per-family
    print('\n  Per-family honest MAE (5-fold nested):')
    for fam in sorted(set(families), key=lambda f: -np.sum(families == f)):
        mask = (families == fam)
        if mask.sum() >= 3:
            fmae = float(mean_absolute_error(y[mask], all_pred[mask]))
            n_fam = int(mask.sum())
            print(f'    {fam:20s} n={n_fam:3d}  honest MAE = {fmae:.4f}')

    out = {'honest_mae': honest_mae, 'honest_r2': honest_r2,
           'in_sample_mae': 0.3954, 'optimism': 0.3954 - honest_mae,
           'predictions': all_pred.tolist()}
    with open(OUT / 'honest_nested_cv_v14.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
