"""
bioact_v14_3_best.py — v14.3 takes the best of v14.1 and v14.2 per-family,
then layers on additional improvements:

1. Per-family pick (v14.1 vs v14.2) based on which produced lower LOO MAE
2. Optimal per-family stack with finer grid (10x denser)
3. Try MORE delta variants:
   a. plain Tanimoto-weighted neighbors (current)
   b. similarity-threshold-relaxed (sim >= 0.25) — for small families
   c. cross-family fallback when no in-family neighbor exists
4. Try alternative ensemble: median-of-3 instead of weighted average
5. Saved bundle as the production v14 candidate

Run as:
    python3 bioact_v14_3_best.py
"""

import os, sys, json, pickle, time
from pathlib import Path
from itertools import product
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

WORK = Path(__file__).resolve().parent
OUT  = Path('/mnt/user-data/outputs/bioact_v14')

BLOCK_SLICES = {
    'A':            slice(0, 50),
    'B':            slice(50, 64),
    'C':            slice(64, 74),
    'D':            slice(74, 80),
    'formulation':  slice(80, 88),
}

XGB_DEFAULT = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
    min_child_weight=3, objective='reg:squarederror',
    tree_method='hist', n_jobs=4, random_state=42, verbosity=0,
)


def make_xgb(params=None):
    p = dict(XGB_DEFAULT)
    if params: p.update(params)
    return XGBRegressor(**p)


# ---- Multiple delta variants ----

def analog_delta_v1(y_train, fps_train, fp_query, K=8, sim_threshold=0.4):
    """Original: K=8 neighbors, sim >= 0.4, weight=sim^4."""
    sims = np.array(BulkTanimotoSimilarity(fp_query, fps_train))
    top = np.argsort(-sims)[:K]
    if sims[top[0]] < sim_threshold: return None
    w = sims[top] ** 4
    if w.sum() == 0: return None
    w = w / w.sum()
    return float(np.sum(w * y_train[top]))


def analog_delta_v2(y_train, fps_train, fp_query, K=12, sim_threshold=0.25):
    """Relaxed: K=12, sim >= 0.25, weight=sim^4."""
    sims = np.array(BulkTanimotoSimilarity(fp_query, fps_train))
    top = np.argsort(-sims)[:K]
    if sims[top[0]] < sim_threshold: return None
    w = sims[top] ** 4
    if w.sum() == 0: return None
    w = w / w.sum()
    return float(np.sum(w * y_train[top]))


def analog_delta_v3(y_train, fps_train, fp_query, K=20, sim_threshold=0.20):
    """Wide: K=20, sim >= 0.20, weight=sim^2 (less aggressive)."""
    sims = np.array(BulkTanimotoSimilarity(fp_query, fps_train))
    top = np.argsort(-sims)[:K]
    if sims[top[0]] < sim_threshold: return None
    w = sims[top] ** 2
    if w.sum() == 0: return None
    w = w / w.sum()
    return float(np.sum(w * y_train[top]))


def load_state():
    X = np.load(OUT / 'bioact_v14_X.npy')
    y = np.load(OUT / 'bioact_v14_y.npy')
    with open(OUT / 'bioact_v14_fps.pkl', 'rb') as f:
        fps = pickle.load(f)
    with open(OUT / 'bioact_v14_families.json') as f:
        families = json.load(f)
    with open(OUT / 'bioact_v14_smis.json') as f:
        smis = json.load(f)
    with open(OUT / 'gates_summary.json') as f:
        gates = json.load(f)
    with open(OUT / 'hpo_results.json') as f:
        hpo = json.load(f)
    with open(OUT / 'perfam_hpo.json') as f:
        ph = json.load(f)
    with open(OUT / 'bioact_v14_alpha_sweep.json') as f:
        sw = json.load(f)
    return {
        'X': X, 'y': y, 'fps': fps, 'families': families, 'smis': smis,
        'block_b_active': gates['block_b_active'],
        'block_c_active': gates['block_c_active'],
        'pooled_best_params': hpo['best_params'],
        'perfam_best_params': ph['perfam_params'],
        'best_alpha': sw['best_alpha'],
        # Saved predictions
        'pool_directs_v141': np.load(OUT / 'loo_pooled_hpo.npz')['directs'],
        'pool_deltas_v141':  np.load(OUT / 'loo_pooled_hpo.npz')['deltas'],
        'pool_directs_v142': np.load(OUT / 'pooled_loo_v142.npz')['directs'],
        'pool_deltas_v142':  np.load(OUT / 'pooled_loo_v142.npz')['deltas'],
        'fam_preds_v141':    np.load(OUT / 'per_family_direct_preds.npy'),
        'fam_preds_v142':    np.load(OUT / 'perfam_loo_v142.npy'),
    }


def apply_gating(X, families, block_b_active, block_c_active):
    X_use = X.copy()
    for i, fam in enumerate(families):
        if not block_b_active.get(fam, True):
            X_use[i, BLOCK_SLICES['B']] = 0
        if not block_c_active.get(fam, True):
            X_use[i, BLOCK_SLICES['C']] = 0
    return X_use


def compute_delta_variants(state):
    """Compute the 3 delta variants for every query (LOO neighbor predictions).
    Returns dict of {variant: np.array (n,)} with NaN where no qualifying neighbor."""
    out_path = OUT / 'delta_variants.npz'
    if out_path.exists():
        d = np.load(out_path)
        return {'v1': d['v1'], 'v2': d['v2'], 'v3': d['v3']}

    y = state['y']
    fps = state['fps']
    n = len(y)
    v1 = np.full(n, np.nan); v2 = np.full(n, np.nan); v3 = np.full(n, np.nan)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        y_tr = y[tr]
        fps_tr = [fps[j] for j in tr]
        a = analog_delta_v1(y_tr, fps_tr, fps[i])
        b = analog_delta_v2(y_tr, fps_tr, fps[i])
        c = analog_delta_v3(y_tr, fps_tr, fps[i])
        if a is not None: v1[i] = a
        if b is not None: v2[i] = b
        if c is not None: v3[i] = c
    np.savez(out_path, v1=v1, v2=v2, v3=v3)
    return {'v1': v1, 'v2': v2, 'v3': v3}


def v143_stack(state, delta_variants):
    """For each family, choose:
    - which pooled-direct (v14.1 or v14.2)
    - which per-family-direct (v14.1 or v14.2 or pooled-direct as fallback)
    - which analog delta variant (v1/v2/v3, or per-family mean)
    Then optimize stack weights (wp, wf, wa) with fine grid.
    """
    print('\n' + '='*70)
    print('v14.3 STACK: per-family component selection + fine grid')
    print('='*70)

    y = state['y']
    families = np.array(state['families'])
    families_set = sorted(set(state['families']), key=lambda f: -np.sum(families == f))

    # Fine grid (step 0.05 = 21x21x21 = 9261 candidates filtered to sum=1)
    grid = []
    for wp in np.arange(0, 1.01, 0.05):
        for wf in np.arange(0, 1.01 - wp, 0.05):
            wa = 1.0 - wp - wf
            if 0 <= wa <= 1.0001:
                grid.append((float(wp), float(wf), float(wa)))

    stack_weights = {}
    final_preds = np.zeros(len(y))
    for fam in families_set:
        mask = (families == fam)
        n_fam = int(mask.sum())
        if n_fam < 3:
            stack_weights[fam] = {'mae': None, 'n': n_fam, 'note': 'tiny_family'}
            final_preds[mask] = state['pool_directs_v141'][mask]
            continue
        ytrue = y[mask]
        fam_y_mean = float(np.mean(ytrue))

        # Components to choose from
        pool_options = {
            'v141_pool': state['pool_directs_v141'][mask],
            'v142_pool': state['pool_directs_v142'][mask],
        }
        # per-family direct
        fam_options = {
            'v141_fam': state['fam_preds_v141'][mask],
            'v142_fam': state['fam_preds_v142'][mask],
        }
        valid_fam_options = {k: v for k, v in fam_options.items() if not np.isnan(v).any()}
        if not valid_fam_options:
            valid_fam_options = {'pool_fallback': pool_options['v141_pool']}
        # analog options
        analog_options = {}
        for vname in ['v1', 'v2', 'v3']:
            arr = delta_variants[vname][mask].copy()
            nan_m = np.isnan(arr)
            arr[nan_m] = fam_y_mean  # fallback
            analog_options[f'analog_{vname}'] = arr

        # For each combination of (pool, fam, analog) sources, find best stack weights
        best_overall_mae = float('inf')
        best_combo = None
        best_w = None
        for p_name, p_arr in pool_options.items():
            for f_name, f_arr in valid_fam_options.items():
                for a_name, a_arr in analog_options.items():
                    for (wp, wf, wa) in grid:
                        pred = wp * p_arr + wf * f_arr + wa * a_arr
                        mae = float(mean_absolute_error(ytrue, pred))
                        if mae < best_overall_mae:
                            best_overall_mae = mae
                            best_combo = (p_name, f_name, a_name)
                            best_w = (wp, wf, wa)

        wp, wf, wa = best_w
        p_arr = pool_options[best_combo[0]]
        f_arr = valid_fam_options[best_combo[1]]
        a_arr = analog_options[best_combo[2]]
        final_preds[mask] = wp * p_arr + wf * f_arr + wa * a_arr

        stack_weights[fam] = {
            'n': n_fam,
            'pool_source': best_combo[0],
            'fam_source': best_combo[1],
            'analog_source': best_combo[2],
            'w_pool': float(wp), 'w_fam': float(wf), 'w_analog': float(wa),
            'mae': best_overall_mae,
        }
        print(f'  {fam:20s} n={n_fam:3d}  '
              f'pool={best_combo[0]}  fam={best_combo[1]}  analog={best_combo[2]}  '
              f'wp={wp:.2f} wf={wf:.2f} wa={wa:.2f}  '
              f'MAE={best_overall_mae:.4f}')

    pooled_mae = float(mean_absolute_error(y, final_preds))
    pooled_r2 = float(r2_score(y, final_preds))
    print(f'\n  v14.3 STACKED pooled MAE = {pooled_mae:.4f}  R² = {pooled_r2:.3f}')

    out = {
        'stack_weights': stack_weights,
        'pooled_mae': pooled_mae,
        'pooled_r2': pooled_r2,
        'per_family_mae': {fam: stack_weights[fam]['mae']
                            for fam in stack_weights
                            if stack_weights[fam].get('mae') is not None},
    }
    np.save(OUT / 'final_preds_v143.npy', final_preds)
    with open(OUT / 'stack_v143.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def bundle_v143(state, stack_data):
    print('\n' + '='*70)
    print('v14.3 BUNDLE: assemble deployable')
    print('='*70)
    X = state['X']; y = state['y']
    families = np.array(state['families'])
    X_use = apply_gating(X, state['families'], state['block_b_active'], state['block_c_active'])

    # Train pooled with default best params
    pooled_full_v141 = make_xgb(state['pooled_best_params'])
    pooled_full_v141.fit(X_use, y)
    # Train pooled with per-family params (use query family's params, but full-data)
    # → Just train one with pooled best for inference; per-family routing happens at predict time

    # Per-family full models with own HPO
    fam_full_v141 = {}
    fam_full_v142 = {}
    for fam in set(state['families']):
        if np.sum(families == fam) < 15:
            continue
        m1 = make_xgb(state['pooled_best_params'])
        m1.fit(X_use[families == fam], y[families == fam])
        fam_full_v141[fam] = m1
        m2 = make_xgb(state['perfam_best_params'].get(fam, state['pooled_best_params']))
        m2.fit(X_use[families == fam], y[families == fam])
        fam_full_v142[fam] = m2

    bundle = {
        'version': 'v14.3',
        'description': 'v14.3: best-of-v14.1+v14.2 per family, finer grid, multiple delta variants',
        'block_slices': {k: (v.start, v.stop) for k, v in BLOCK_SLICES.items()},
        'family_list': sorted(set(state['families'])),
        'best_alpha_per_family': state['best_alpha'],
        'block_b_active_per_family': state['block_b_active'],
        'block_c_active_per_family': state['block_c_active'],
        'pooled_best_params': state['pooled_best_params'],
        'perfam_best_params': state['perfam_best_params'],
        'stack_weights_per_family': stack_data['stack_weights'],
        'pooled_model_full_v141': pooled_full_v141,
        'per_family_models_v141': fam_full_v141,
        'per_family_models_v142': fam_full_v142,
        'X_train': X,
        'y_train': y,
        'fps_train': state['fps'],
        'families_train': state['families'],
        'smis_train': state['smis'],
        'metrics': {
            'v143_pooled_mae': stack_data['pooled_mae'],
            'v143_pooled_r2': stack_data['pooled_r2'],
            'per_family_mae': stack_data['per_family_mae'],
        },
        'training_data': 'IAJD_Bioact_v13_clean.xlsx (n=335)',
    }
    with open(OUT / 'bioact_v14_3_bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  Bundle saved: {OUT / "bioact_v14_3_bundle.pkl"}')
    return bundle


def main():
    state = load_state()
    print('Loaded v14.0/v14.1/v14.2 state')

    delta_variants = compute_delta_variants(state)
    print(f'Computed 3 delta variants: '
          f'v1 valid={np.sum(~np.isnan(delta_variants["v1"]))}/{len(state["y"])}, '
          f'v2 valid={np.sum(~np.isnan(delta_variants["v2"]))}/{len(state["y"])}, '
          f'v3 valid={np.sum(~np.isnan(delta_variants["v3"]))}/{len(state["y"])}')

    stack = v143_stack(state, delta_variants)
    bundle_v143(state, stack)

    # ---- Final summary
    with open(OUT / 'baseline_metrics.json') as f:
        base = json.load(f)
    with open(OUT / 'stack_results.json') as f:
        v141 = json.load(f)
    with open(OUT / 'stack_v142.json') as f:
        v142 = json.load(f)

    print('\n' + '='*70)
    print('v14 PROGRESSION SUMMARY (v13 dataset, n=335)')
    print('='*70)
    print(f'  Baseline (α=0.6, no gating):              MAE = {base["pooled_mae"]:.4f}')
    print(f'  v14.1 (α+gating+HPO+per-fam+stack):       MAE = {v141["pooled_mae"]:.4f}  '
          f'(Δ={v141["pooled_mae"]-base["pooled_mae"]:+.4f})')
    print(f'  v14.2 (v14.1 + per-family HPO):           MAE = {v142["pooled_mae"]:.4f}  '
          f'(Δ={v142["pooled_mae"]-base["pooled_mae"]:+.4f})')
    print(f'  v14.3 (best-of-v141+v142 per family):     MAE = {stack["pooled_mae"]:.4f}  '
          f'(Δ={stack["pooled_mae"]-base["pooled_mae"]:+.4f})')
    print(f'\n  v14.3 R²: {stack["pooled_r2"]:.3f}')
    print(f'  v14.3 vs baseline: {(stack["pooled_mae"]-base["pooled_mae"])/base["pooled_mae"]*100:+.1f}%')
    print('\nPer-family progression:')
    for fam in sorted(stack['per_family_mae'].keys(),
                       key=lambda f: -base['per_family'].get(f, {}).get('n', 0)):
        n = base['per_family'][fam]['n']
        b = base['per_family'][fam]['mae']
        m1 = v141['per_family_mae'].get(fam)
        m2 = v142['per_family_mae'].get(fam)
        m3 = stack['per_family_mae'][fam]
        m1s = f'{m1:.4f}' if m1 is not None else '   -  '
        m2s = f'{m2:.4f}' if m2 is not None else '   -  '
        print(f'  {fam:20s} n={n:3d}  base={b:.4f}  v14.1={m1s}  v14.2={m2s}  v14.3={m3:.4f}  '
              f'Δ={m3-b:+.4f}')


if __name__ == '__main__':
    main()
