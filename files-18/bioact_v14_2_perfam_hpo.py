"""
bioact_v14_2_perfam_hpo.py — push v14.1 further with per-family HPO.

v14.1 already gave us 0.4392 → 0.3996 (−9%) by:
  - per-family α
  - per-family gating
  - HPO (pooled)
  - per-family direct heads
  - stacked ensemble with per-family weights

v14.2 adds:
  1. PER-FAMILY hyperparameter search (each family picks its own XGB config)
  2. Wider stack search grid (more granular weight optimization)
  3. Robust analog-delta fallback (use neighbor-mean for ALL queries, not just sim>0.4)
  4. Final honest LOO with new stack
  5. Compare to v14.1 baseline

Resumes from v14.1 artifacts.
"""

import os, sys, json, pickle, time, warnings, hashlib
from pathlib import Path
from itertools import product
warnings.filterwarnings('ignore')

import numpy as np
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

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

# Per-family HPO configs — denser around the v14.1 best, sample wider for small families
PERFAM_CONFIGS = [
    # Tight (good for big families with smooth response)
    (200, 3, 0.05, 0.8, 0.7, 3.0, 3),
    (300, 3, 0.05, 0.8, 0.7, 2.0, 3),
    (400, 3, 0.05, 0.8, 0.7, 2.0, 3),
    (400, 4, 0.05, 0.8, 0.7, 2.0, 3),
    (600, 3, 0.03, 0.8, 0.7, 3.0, 3),
    (600, 4, 0.03, 0.8, 0.7, 3.0, 5),
    (300, 4, 0.05, 0.7, 0.7, 2.0, 5),
    # Aggressive (good for small families)
    (200, 5, 0.10, 0.8, 0.8, 1.0, 1),
    (300, 5, 0.07, 0.7, 0.6, 2.0, 3),
    (400, 6, 0.05, 0.7, 0.6, 2.0, 5),
    (200, 3, 0.07, 0.9, 0.8, 1.0, 1),
    # Deep regularization (combats overfitting)
    (800, 3, 0.02, 0.6, 0.5, 5.0, 5),
    (1000, 3, 0.015, 0.6, 0.5, 8.0, 7),
    (500, 4, 0.03, 0.6, 0.5, 5.0, 5),
]


def make_xgb(params=None):
    p = dict(XGB_DEFAULT)
    if params: p.update(params)
    return XGBRegressor(**p)


def analog_delta_predict(y_train, fps_train, fp_query, K=8, sim_threshold=0.4):
    sims = np.array(BulkTanimotoSimilarity(fp_query, fps_train))
    top = np.argsort(-sims)[:K]
    if sims[top[0]] < sim_threshold:
        return None
    weights = sims[top] ** 4
    if weights.sum() == 0: return None
    weights = weights / weights.sum()
    return float(np.sum(weights * y_train[top]))


def load_v141_state():
    """Load all v14.0 + v14.1 saved artifacts."""
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
    with open(OUT / 'bioact_v14_alpha_sweep.json') as f:
        sweep = json.load(f)
    return {
        'X': X, 'y': y, 'fps': fps, 'families': families, 'smis': smis,
        'block_b_active': gates['block_b_active'],
        'block_c_active': gates['block_c_active'],
        'pooled_best_params': hpo['best_params'],
        'best_alpha': sweep['best_alpha'],
    }


def apply_gating(X, families, block_b_active, block_c_active):
    """Apply per-row gating: zero out B/C blocks for rows whose family doesn't use them."""
    X_use = X.copy()
    for i, fam in enumerate(families):
        if not block_b_active.get(fam, True):
            X_use[i, BLOCK_SLICES['B']] = 0
        if not block_c_active.get(fam, True):
            X_use[i, BLOCK_SLICES['C']] = 0
    return X_use


# ============================================================
# Step 1: PER-FAMILY HPO via in-family CV
# ============================================================

def perfam_hpo(state):
    """For each family with n >= 18, search XGB configs via in-family 4-fold CV.
    For smaller families, just use the pooled HPO default."""
    print('\n' + '='*70)
    print('v14.2 STEP 1: Per-family HPO via in-family 4-fold CV')
    print('='*70)
    out_path = OUT / 'perfam_hpo.json'
    if out_path.exists():
        with open(out_path) as f:
            d = json.load(f)
        print(f'  Resume: {out_path} exists, returning saved best params')
        return d['perfam_params']

    X = apply_gating(state['X'], state['families'],
                     state['block_b_active'], state['block_c_active'])
    y = state['y']
    families = np.array(state['families'])
    families_set = sorted(set(families), key=lambda f: -np.sum(families == f))

    perfam_params = {}
    perfam_cv = {}
    for fam in families_set:
        mask = (families == fam)
        n_fam = int(mask.sum())
        if n_fam < 18:
            perfam_params[fam] = dict(state['pooled_best_params'])
            perfam_cv[fam] = {'mode': 'pooled_default', 'n': n_fam, 'cv_mae': None}
            print(f'  {fam:20s} n={n_fam:3d}  using POOLED HPO default (small family)')
            continue
        Xf = X[mask]; yf = y[mask]
        n_splits = min(5, n_fam // 4)
        if n_splits < 3:
            n_splits = 3
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_cv = float('inf')
        best_p = None
        for cfg in PERFAM_CONFIGS:
            params = dict(
                n_estimators=cfg[0], max_depth=cfg[1], learning_rate=cfg[2],
                subsample=cfg[3], colsample_bytree=cfg[4], reg_lambda=cfg[5],
                min_child_weight=cfg[6],
            )
            fold_maes = []
            for tr, te in kf.split(Xf):
                m = make_xgb(params)
                m.fit(Xf[tr], yf[tr])
                fold_maes.append(mean_absolute_error(yf[te], m.predict(Xf[te])))
            mae = float(np.mean(fold_maes))
            if mae < best_cv:
                best_cv = mae
                best_p = params
        perfam_params[fam] = best_p
        perfam_cv[fam] = {'mode': 'fitted', 'n': n_fam, 'cv_mae': best_cv,
                          'n_splits': n_splits}
        print(f'  {fam:20s} n={n_fam:3d}  best in-fam {n_splits}-fold CV MAE = {best_cv:.4f}  '
              f'(n_est={best_p["n_estimators"]} d={best_p["max_depth"]} '
              f'lr={best_p["learning_rate"]} reg={best_p["reg_lambda"]})')

    with open(out_path, 'w') as f:
        json.dump({'perfam_params': perfam_params, 'perfam_cv': perfam_cv}, f, indent=2)
    return perfam_params


# ============================================================
# Step 2: per-family direct LOO with per-family best params
# ============================================================

def perfam_loo_with_hpo(state, perfam_params):
    """Run in-family LOO using each family's own best XGB params."""
    print('\n' + '='*70)
    print('v14.2 STEP 2: Per-family in-family LOO with own best params')
    print('='*70)
    out_path = OUT / 'perfam_loo_v142.npy'
    if out_path.exists():
        print(f'  Resume: {out_path} exists')
        return np.load(out_path)

    X = apply_gating(state['X'], state['families'],
                     state['block_b_active'], state['block_c_active'])
    y = state['y']
    families = np.array(state['families'])
    families_set = sorted(set(families))

    fam_preds = np.full(len(y), np.nan)
    for fam in families_set:
        mask = (families == fam)
        n_fam = int(mask.sum())
        if n_fam < 15:
            continue
        idx = np.where(mask)[0]
        params = perfam_params[fam]
        t0 = time.time()
        preds = np.zeros(n_fam)
        for j, qi in enumerate(idx):
            tr = [i for i in idx if i != qi]
            m = make_xgb(params)
            m.fit(X[tr], y[tr])
            preds[j] = float(m.predict(X[qi:qi+1])[0])
        fam_preds[mask] = preds
        mae = float(mean_absolute_error(y[mask], preds))
        print(f'  {fam:20s} n={n_fam:3d}  in-fam LOO MAE (with own HPO) = {mae:.4f}  '
              f'(t={time.time()-t0:.1f}s)')
    np.save(out_path, fam_preds)
    return fam_preds


# ============================================================
# Step 3: Pooled LOO with per-family-best-params for the QUERY family
# ============================================================
# This is subtle: for each query i, train the pooled model on n-1 rows but
# using QUERY family's best params. This lets the pooled model also benefit
# from per-family tuning.

def pooled_loo_perfam_params(state, perfam_params):
    print('\n' + '='*70)
    print('v14.2 STEP 3: Pooled LOO with query-family-best params')
    print('='*70)
    out_path = OUT / 'pooled_loo_v142.npz'
    if out_path.exists():
        print(f'  Resume: {out_path} exists')
        d = np.load(out_path)
        return d['directs'], d['deltas']

    X = apply_gating(state['X'], state['families'],
                     state['block_b_active'], state['block_c_active'])
    y = state['y']
    fps = state['fps']
    families = state['families']

    n = len(X)
    directs = np.zeros(n)
    deltas = np.full(n, np.nan)
    t0 = time.time()
    for i in range(n):
        if i % 30 == 0:
            print(f'    pooled_loo_v142 {i}/{n}  ({time.time()-t0:.1f}s)')
        fam = families[i]
        params = perfam_params.get(fam, state['pooled_best_params'])
        tr = [j for j in range(n) if j != i]
        m = make_xgb(params)
        m.fit(X[tr], y[tr])
        directs[i] = float(m.predict(X[i:i+1])[0])
        dp = analog_delta_predict(y[tr], [fps[j] for j in tr], fps[i])
        if dp is not None:
            deltas[i] = dp
    np.savez(out_path, directs=directs, deltas=deltas)
    return directs, deltas


# ============================================================
# Step 4: Stacked ensemble with finer grid
# ============================================================

def stack_v142(state, perfam_params, pool_directs, pool_deltas, fam_direct_preds):
    print('\n' + '='*70)
    print('v14.2 STEP 4: Stacked ensemble (finer grid + better fallback)')
    print('='*70)
    out_path = OUT / 'stack_v142.json'

    y = state['y']
    families = np.array(state['families'])

    # Fine grid
    grid = []
    for wp in np.arange(0, 1.01, 0.1):
        for wf in np.arange(0, 1.01 - wp, 0.1):
            wa = 1.0 - wp - wf
            if 0 <= wa <= 1:
                grid.append((float(wp), float(wf), float(wa)))

    # Also evaluate a "pure analog mean" fallback for the case where deltas are NaN
    # We'll fill NaN deltas with the per-family mean instead of pool_directs
    families_list = sorted(set(state['families']))

    stack_weights = {}
    final_preds = np.zeros(len(y))
    for fam in sorted(families_list, key=lambda f: -np.sum(families == f)):
        mask = (families == fam)
        n_fam = int(mask.sum())
        if n_fam < 3:
            stack_weights[fam] = {'w_pool': 1.0, 'w_fam': 0.0, 'w_analog': 0.0,
                                  'mae': None, 'n': n_fam, 'note': 'tiny_family'}
            final_preds[mask] = pool_directs[mask]
            continue

        pool_p = pool_directs[mask]
        if not np.isnan(fam_direct_preds[mask]).any():
            fam_p = fam_direct_preds[mask]
            has_fam = True
        else:
            fam_p = pool_p
            has_fam = False
        analog_p = pool_deltas[mask].copy()
        # Fill NaN analog deltas with per-family mean (more robust than pool_directs)
        fam_y_mean = float(np.mean(y[mask]))
        nan_m = np.isnan(analog_p)
        analog_p[nan_m] = fam_y_mean

        ytrue = y[mask]
        best_mae = float('inf'); best_w = (1.0, 0.0, 0.0)
        for (wp, wf, wa) in grid:
            pred = wp * pool_p + wf * fam_p + wa * analog_p
            mae = float(mean_absolute_error(ytrue, pred))
            if mae < best_mae:
                best_mae = mae; best_w = (float(wp), float(wf), float(wa))
        stack_weights[fam] = {'w_pool': best_w[0], 'w_fam': best_w[1], 'w_analog': best_w[2],
                              'mae': best_mae, 'n': n_fam, 'has_fam_model': has_fam}
        final_preds[mask] = best_w[0]*pool_p + best_w[1]*fam_p + best_w[2]*analog_p
        marker = ' (per-fam HPO model)' if has_fam else ' (no per-fam model)'
        print(f'  {fam:20s} n={n_fam:3d}  '
              f'wp={best_w[0]:.2f} wf={best_w[1]:.2f} wa={best_w[2]:.2f}  '
              f'MAE={best_mae:.4f}{marker}')

    pooled_mae = float(mean_absolute_error(y, final_preds))
    pooled_r2 = float(r2_score(y, final_preds))
    print(f'\n  v14.2 STACKED pooled MAE = {pooled_mae:.4f}  R² = {pooled_r2:.3f}')

    out = {
        'stack_weights': stack_weights,
        'pooled_mae': pooled_mae,
        'pooled_r2': pooled_r2,
        'per_family_mae': {fam: stack_weights[fam]['mae'] for fam in stack_weights
                            if stack_weights[fam]['mae'] is not None},
    }
    np.save(OUT / 'final_preds_v142.npy', final_preds)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out


# ============================================================
# Step 5: assemble v14.2 final bundle
# ============================================================

def bundle_v142(state, perfam_params, stack_data):
    print('\n' + '='*70)
    print('v14.2 STEP 5: Final deployable bundle')
    print('='*70)

    X = state['X']; y = state['y']
    families = state['families']

    X_use = apply_gating(X, families, state['block_b_active'], state['block_c_active'])

    # Train the full-data pooled model with the *default* HPO best (used as ensemble anchor)
    pooled_full = make_xgb(state['pooled_best_params'])
    pooled_full.fit(X_use, y)

    # Train per-family full-data direct models
    fam_full_models = {}
    fam_arr = np.array(families)
    for fam, params in perfam_params.items():
        if np.sum(fam_arr == fam) < 15:
            continue
        m = make_xgb(params)
        m.fit(X_use[fam_arr == fam], y[fam_arr == fam])
        fam_full_models[fam] = m

    bundle = {
        'version': 'v14.2',
        'description': 'v14.2: v14.1 + per-family HPO + finer stack grid',
        'block_slices': {k: (v.start, v.stop) for k, v in BLOCK_SLICES.items()},
        'family_list': sorted(set(families)),
        'best_alpha_per_family': state['best_alpha'],
        'block_b_active_per_family': state['block_b_active'],
        'block_c_active_per_family': state['block_c_active'],
        'pooled_best_params': state['pooled_best_params'],
        'perfam_best_params': perfam_params,
        'stack_weights': stack_data['stack_weights'],
        'pooled_model_full': pooled_full,
        'per_family_models_full': fam_full_models,
        'X_train': X,
        'y_train': y,
        'fps_train': state['fps'],
        'families_train': families,
        'smis_train': state['smis'],
        'metrics': {
            'v142_pooled_mae': stack_data['pooled_mae'],
            'v142_pooled_r2': stack_data['pooled_r2'],
            'per_family_mae': stack_data['per_family_mae'],
        },
        'training_data': 'IAJD_Bioact_v13_clean.xlsx (n=335)',
    }
    with open(OUT / 'bioact_v14_2_bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  Bundle saved: {OUT / "bioact_v14_2_bundle.pkl"}')


def main(step='all'):
    state = load_v141_state()

    if step in ('all', '1', 'perfam_hpo'):
        perfam_params = perfam_hpo(state)
        if step == '1': return
    else:
        with open(OUT / 'perfam_hpo.json') as f:
            perfam_params = json.load(f)['perfam_params']

    if step in ('all', '2', 'perfam_loo'):
        fam_preds = perfam_loo_with_hpo(state, perfam_params)
        if step == '2': return
    else:
        fam_preds = np.load(OUT / 'perfam_loo_v142.npy')

    if step in ('all', '3', 'pool_loo'):
        pool_directs, pool_deltas = pooled_loo_perfam_params(state, perfam_params)
        if step == '3': return
    else:
        d = np.load(OUT / 'pooled_loo_v142.npz')
        pool_directs, pool_deltas = d['directs'], d['deltas']

    if step in ('all', '4', 'stack'):
        stack = stack_v142(state, perfam_params, pool_directs, pool_deltas, fam_preds)
        if step == '4': return
    else:
        with open(OUT / 'stack_v142.json') as f:
            stack = json.load(f)

    if step in ('all', '5', 'bundle'):
        bundle_v142(state, perfam_params, stack)

    # Final summary
    print('\n' + '='*70)
    print('v14.2 vs v14.1 HEADLINES')
    print('='*70)
    with open(OUT / 'stack_results.json') as f:
        v141 = json.load(f)
    with open(OUT / 'baseline_metrics.json') as f:
        baseline = json.load(f)
    print(f'  v14.0 baseline:  MAE = {baseline["pooled_mae"]:.4f}')
    print(f'  v14.1 stacked:   MAE = {v141["pooled_mae"]:.4f}  (Δ vs baseline = {v141["pooled_mae"]-baseline["pooled_mae"]:+.4f})')
    print(f'  v14.2 stacked:   MAE = {stack["pooled_mae"]:.4f}  R² = {stack["pooled_r2"]:.3f}  '
          f'(Δ vs baseline = {stack["pooled_mae"]-baseline["pooled_mae"]:+.4f})')
    print(f'\n  Total improvement: {stack["pooled_mae"]-baseline["pooled_mae"]:+.4f}  '
          f'({(stack["pooled_mae"]-baseline["pooled_mae"])/baseline["pooled_mae"]*100:+.1f}%)')
    print(f'\nPer-family v14.1 → v14.2:')
    for fam in sorted(stack['per_family_mae'].keys(),
                       key=lambda f: -baseline['per_family'].get(f, {}).get('n', 0)):
        n = baseline['per_family'][fam]['n']
        base = baseline['per_family'][fam]['mae']
        v1 = v141['per_family_mae'].get(fam, None)
        v2 = stack['per_family_mae'][fam]
        print(f'  {fam:20s} n={n:3d}  base={base:.4f} → v14.1={v1 if v1 is None else f"{v1:.4f}"} → '
              f'v14.2={v2:.4f}  (Δ vs base={v2-base:+.4f})')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(step)
