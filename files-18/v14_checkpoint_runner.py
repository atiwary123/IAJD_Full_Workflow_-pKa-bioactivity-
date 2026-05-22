"""
v14_checkpoint_runner.py — Run the v14 pipeline phase-by-phase with hard saves.

Each phase saves its outputs to /mnt/user-data/outputs/bioact_v14/ before exiting,
so the next phase can resume from disk even after a session reset.

Phases (run with --phase <name>):
    assemble      — Build X, y, fps, meta. (~1 min)
    base_loo      — LOO with all blocks ON. Saves directs_full, deltas_full. (~2 min)
    alpha_sweep   — Reuses base_loo cache. (~5 sec)
    gate_B_<fam>  — Gate decision for one family on Block B.  (~1.5 min each)
    gate_C_<fam>  — Gate decision for one family on Block C.  (~1.5 min each)
    final_loo     — LOO with per-family α + per-family gates. (~2 min)
    bundle        — Build deployable bundle + model card. (instant)

Resilience: each phase loads its predecessors from disk. If a phase has
already run and produced output, calling it again is a no-op (unless --force).
"""

import os, sys, json, pickle, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
OUT  = Path('/mnt/user-data/outputs/bioact_v14')
OUT.mkdir(parents=True, exist_ok=True)

# Import pipeline machinery
sys.path.insert(0, str(WORK))
from bioact_v14_pipeline import (
    load_v13, assemble_X, train_direct, analog_delta_predict,
    blend_with_alpha, evaluate_predictions,
    ALL_NAMES_V14, BLOCK_SLICES,
)
from sklearn.metrics import mean_absolute_error, r2_score


def loo_cache_with_save(X, y, fps, families, gate_b, gate_c, save_path=None,
                         verbose=True):
    """LOO that saves directs/deltas to disk after every 50 rows so progress
    isn't lost on timeout. Can resume from a partial save."""
    n = len(X)
    directs = np.full(n, np.nan)
    deltas = np.full(n, np.nan)
    # Resume if partial save exists
    start_i = 0
    if save_path and Path(save_path).exists():
        with np.load(save_path) as d:
            saved_directs = d['directs']
            saved_deltas = d['deltas']
            saved_complete = d['complete'] if 'complete' in d else None
        if len(saved_directs) == n:
            directs = saved_directs.copy()
            deltas = saved_deltas.copy()
            # Find first NaN to resume
            done_mask = ~np.isnan(directs)
            if done_mask.all():
                if verbose:
                    print(f'    [{save_path.name}] already complete, loading')
                return directs, deltas
            start_i = int(np.argmin(done_mask))
            if verbose:
                print(f'    [{save_path.name}] resuming from row {start_i}')

    t0 = time.time()
    for i in range(start_i, n):
        fam = families[i]
        use_b = gate_b.get(fam, True)
        use_c = gate_c.get(fam, True)
        X_gated = X.copy()
        if not use_b:
            X_gated[:, BLOCK_SLICES['B']] = 0
        if not use_c:
            X_gated[:, BLOCK_SLICES['C']] = 0
        train_idx = [j for j in range(n) if j != i]
        m = train_direct(X_gated[train_idx], y[train_idx])
        directs[i] = float(m.predict(X_gated[i:i+1])[0])
        dp = analog_delta_predict(X_gated[train_idx], y[train_idx],
                                    [fps[j] for j in train_idx], fps[i], X_gated[i],
                                    K=8, sim_threshold=0.4)
        if dp is not None:
            deltas[i] = dp
        # Save every 50 rows
        if save_path and (i + 1) % 50 == 0:
            np.savez(save_path, directs=directs, deltas=deltas)
            if verbose:
                elapsed = time.time() - t0
                rate = (i - start_i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f'    {i+1}/{n}  ({rate:.1f} rows/s, ETA {eta:.0f}s)')
    if save_path:
        np.savez(save_path, directs=directs, deltas=deltas)
    return directs, deltas


def phase_assemble(args):
    print('Phase: assemble')
    if (OUT / 'bioact_v14_X.npy').exists() and not args.force:
        print('  already done; use --force to rebuild')
        return
    df, mols, fps = load_v13()
    smis = df['canonical_smi'].tolist()
    y = df['log10_flux_total'].values.astype(float)
    families = df['family'].fillna('Unknown').tolist()
    # check for caches
    lion_cache = WORK / 'lion_cache_v13.json'
    admet_cache = WORK / 'admet_cache_v13.json'
    lion_train_fps_path = WORK / 'lion_train_fps.pkl'
    X, lion_modes = assemble_X(
        df, mols, fps, smis,
        lion_cache_path=str(lion_cache) if lion_cache.exists() else None,
        admet_cache_path=str(admet_cache) if admet_cache.exists() else None,
        lion_train_fps_path=str(lion_train_fps_path) if lion_train_fps_path.exists() else None,
    )
    np.save(OUT / 'bioact_v14_X.npy', X)
    np.save(OUT / 'bioact_v14_y.npy', y)
    df[['IAJD_id','family','log10_flux_total','SMILES_canonical','holdout']].to_csv(
        OUT / 'bioact_v14_meta.csv', index=False)
    with open(OUT / 'bioact_v14_fps.pkl', 'wb') as f:
        pickle.dump([fp for fp in fps], f)
    with open(OUT / 'bioact_v14_families.json', 'w') as f:
        json.dump(families, f)
    with open(OUT / 'bioact_v14_smis.json', 'w') as f:
        json.dump(smis, f)
    with open(OUT / 'bioact_v14_lion_modes.json', 'w') as f:
        json.dump(lion_modes, f)
    print(f'  X shape: {X.shape}, unique IAJDs: {len(set(df["IAJD_id"]))}')


def _load_assembly():
    X = np.load(OUT / 'bioact_v14_X.npy')
    y = np.load(OUT / 'bioact_v14_y.npy')
    with open(OUT / 'bioact_v14_fps.pkl', 'rb') as f:
        fps = pickle.load(f)
    with open(OUT / 'bioact_v14_families.json') as f:
        families = json.load(f)
    return X, y, fps, families


def phase_base_loo(args):
    print('Phase: base_loo (all blocks ON, computes directs_full + deltas_full)')
    X, y, fps, families = _load_assembly()
    fam_list = set(families)
    gate_all = {f: True for f in fam_list}
    save_path = OUT / 'loo_base.npz'
    directs, deltas = loo_cache_with_save(X, y, fps, families, gate_all, gate_all,
                                            save_path=save_path)
    # Evaluate at α=0.6 uniform
    preds, _ = blend_with_alpha(directs, deltas, {f: 0.6 for f in fam_list}, families)
    base_mae, base_r2, base_perfam = evaluate_predictions(preds, y, families)
    with open(OUT / 'baseline_metrics.json', 'w') as f:
        json.dump({
            'pooled_mae': base_mae, 'pooled_r2': base_r2,
            'per_family': base_perfam, 'config': 'α=0.6 uniform, all blocks on',
        }, f, indent=2)
    print(f'  Baseline pooled MAE: {base_mae:.4f}, R²: {base_r2:.3f}')
    for f, d in sorted(base_perfam.items(), key=lambda kv: -kv[1]['n']):
        print(f"    {f:18s} n={d['n']:3d}  MAE={d['mae']:.4f}")


def phase_alpha_sweep(args):
    print('Phase: alpha_sweep (re-blend cached predictions)')
    X, y, fps, families = _load_assembly()
    fam_list = set(families)
    with np.load(OUT / 'loo_base.npz') as d:
        directs = d['directs']; deltas = d['deltas']
    alphas_to_try = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_res = {}
    for a in alphas_to_try:
        preds, _ = blend_with_alpha(directs, deltas, {f: a for f in fam_list}, families)
        mae, r2, perfam = evaluate_predictions(preds, y, families)
        sweep_res[str(a)] = {
            'pooled_mae': mae, 'pooled_r2': r2,
            'per_family_mae': {f: d['mae'] for f, d in perfam.items()},
            'per_family_n': {f: d['n'] for f, d in perfam.items()},
        }
        print(f'  α={a:.2f}: pooled MAE = {mae:.4f}, R² = {r2:.3f}')
    best_alpha = {}
    for f in fam_list:
        candidates = [(a, sweep_res[str(a)]['per_family_mae'].get(f, 999)) for a in alphas_to_try]
        candidates.sort(key=lambda x: x[1])
        best_alpha[f] = candidates[0][0]
    print(f'\n  Best α per family:')
    for f, a in sorted(best_alpha.items()):
        print(f'    {f:18s}  α = {a}')
    with open(OUT / 'bioact_v14_alpha_sweep.json', 'w') as f:
        json.dump({'sweep': sweep_res, 'best_alpha': best_alpha}, f, indent=2)


def phase_gate(args):
    """One LOO pass for one (family, block) combo. Family + block specified by args."""
    fam = args.family
    block = args.block.upper()
    print(f'Phase: gate_{block}_{fam}')
    X, y, fps, families = _load_assembly()
    fam_list = set(families)
    if fam not in fam_list:
        print(f'  ERROR: family {fam} not in dataset')
        return
    n_fam = sum(1 for x in families if x == fam)
    if n_fam < 5:
        print(f'  skipping: only n={n_fam} in this family')
        with open(OUT / f'gate_{block}_{fam}.json', 'w') as f:
            json.dump({'family': fam, 'block': block, 'use': True,
                       'note': f'small_family_n={n_fam}'}, f)
        return
    gate_b = {f: True for f in fam_list}
    gate_c = {f: True for f in fam_list}
    if block == 'B':
        gate_b[fam] = False
    elif block == 'C':
        gate_c[fam] = False
    save_path = OUT / f'loo_gate_{block}_{fam}.npz'
    directs, deltas = loo_cache_with_save(X, y, fps, families, gate_b, gate_c,
                                            save_path=save_path)
    # Compare to baseline only on the affected family
    with np.load(OUT / 'loo_base.npz') as d:
        directs_full = d['directs']; deltas_full = d['deltas']
    # Use best α per fam if available
    if (OUT / 'bioact_v14_alpha_sweep.json').exists():
        with open(OUT / 'bioact_v14_alpha_sweep.json') as f:
            best_alpha = json.load(f)['best_alpha']
    else:
        best_alpha = {f: 0.6 for f in fam_list}
    # MAE comparison for this family only
    mask = np.array([f == fam for f in families])
    preds_full, _ = blend_with_alpha(directs_full, deltas_full, best_alpha, families)
    preds_off, _ = blend_with_alpha(directs, deltas, best_alpha, families)
    mae_full = float(mean_absolute_error(y[mask], preds_full[mask]))
    mae_off = float(mean_absolute_error(y[mask], preds_off[mask]))
    use_block = bool(mae_off >= mae_full)
    delta = mae_off - mae_full
    print(f'  {fam} (n={n_fam}) Block {block}: '
          f'mae_full={mae_full:.4f} mae_OFF={mae_off:.4f} Δ={delta:+.4f}  '
          f'→ use_{block}={use_block}')
    with open(OUT / f'gate_{block}_{fam}.json', 'w') as f:
        json.dump({
            'family': fam, 'block': block,
            'mae_full': mae_full, 'mae_off': mae_off, 'delta': delta,
            'use': use_block, 'n': n_fam,
        }, f, indent=2)


def phase_final_loo(args):
    print('Phase: final_loo (per-family α + per-family gates)')
    X, y, fps, families = _load_assembly()
    fam_list = set(families)
    # Load α
    with open(OUT / 'bioact_v14_alpha_sweep.json') as f:
        best_alpha = json.load(f)['best_alpha']
    # Load gates
    gate_b = {}; gate_c = {}
    for fam in fam_list:
        gb_path = OUT / f'gate_B_{fam}.json'
        gc_path = OUT / f'gate_C_{fam}.json'
        if gb_path.exists():
            with open(gb_path) as f: gate_b[fam] = json.load(f).get('use', True)
        else:
            gate_b[fam] = True
        if gc_path.exists():
            with open(gc_path) as f: gate_c[fam] = json.load(f).get('use', True)
        else:
            gate_c[fam] = True
    print(f'  Gates per family:')
    for fam in sorted(fam_list):
        print(f'    {fam:18s} α={best_alpha[fam]}  B={"on" if gate_b[fam] else "OFF"}  C={"on" if gate_c[fam] else "OFF"}')
    save_path = OUT / 'loo_final.npz'
    directs, deltas = loo_cache_with_save(X, y, fps, families, gate_b, gate_c,
                                            save_path=save_path)
    preds, _ = blend_with_alpha(directs, deltas, best_alpha, families)
    final_mae, final_r2, final_perfam = evaluate_predictions(preds, y, families)
    # Per-family delta vs baseline
    with open(OUT / 'baseline_metrics.json') as f:
        baseline = json.load(f)
    print(f'\n  v14 FINAL pooled MAE: {final_mae:.4f}, R²: {final_r2:.3f}')
    print(f'  Per-family (baseline → v14):')
    out = {
        'pooled_mae': final_mae, 'pooled_r2': final_r2,
        'per_family': final_perfam,
        'best_alpha': best_alpha,
        'gate_b': gate_b, 'gate_c': gate_c,
        'baseline_pooled_mae': baseline['pooled_mae'],
        'pooled_mae_improvement': baseline['pooled_mae'] - final_mae,
    }
    for fam in sorted(fam_list, key=lambda f: -final_perfam.get(f, {}).get('n', 0)):
        if fam not in final_perfam: continue
        bmae = baseline['per_family'][fam]['mae']
        vmae = final_perfam[fam]['mae']
        n = final_perfam[fam]['n']
        delta = vmae - bmae
        print(f"    {fam:18s} n={n:3d}  base={bmae:.4f} → v14={vmae:.4f}  Δ={delta:+.4f}")
    with open(OUT / 'bioact_v14_loo.json', 'w') as f:
        json.dump(out, f, indent=2)


def phase_bundle(args):
    print('Phase: bundle (build deployable bundle)')
    X, y, fps, families = _load_assembly()
    fam_list = set(families)
    with open(OUT / 'bioact_v14_alpha_sweep.json') as f:
        best_alpha = json.load(f)['best_alpha']
    with open(OUT / 'bioact_v14_loo.json') as f:
        loo = json.load(f)
    with open(OUT / 'bioact_v14_smis.json') as f:
        smis = json.load(f)
    with open(OUT / 'bioact_v14_lion_modes.json') as f:
        lion_modes = json.load(f)
    # Collect gates
    gates = {}
    for fam in fam_list:
        for blk in ('B', 'C'):
            gp = OUT / f'gate_{blk}_{fam}.json'
            if gp.exists():
                with open(gp) as f: d = json.load(f)
                gates.setdefault(fam, {})[blk] = d
    # Train full model on all data
    full_model = train_direct(X, y)
    bundle = {
        'version': 'v14.0',
        'training_data': 'IAJD_Bioact_v13_clean.xlsx',
        'n_rows': int(len(X)),
        'n_unique_smiles': len(set(smis)),
        'feature_names': ALL_NAMES_V14,
        'block_slices': {k: (s.start, s.stop) for k, s in BLOCK_SLICES.items()},
        'family_list': sorted(fam_list),
        'best_alpha_per_family': best_alpha,
        'block_b_active_per_family': loo['gate_b'],
        'block_c_active_per_family': loo['gate_c'],
        'direct_model_full': full_model,
        'X_train': X,
        'y_train': y,
        'fps_train': fps,
        'families_train': families,
        'smis_train': smis,
        'metrics': {
            'baseline_pooled_mae': loo['baseline_pooled_mae'],
            'v14_pooled_mae': loo['pooled_mae'],
            'v14_pooled_r2': loo['pooled_r2'],
            'per_family': loo['per_family'],
        },
        'gate_decisions': gates,
        'lion_modes': lion_modes,
    }
    with open(OUT / 'bioact_v14_bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  Saved bundle: {OUT / "bioact_v14_bundle.pkl"}')
    print(f'  Headline metrics:')
    print(f'    n_rows: {len(X)}, n_unique_smiles: {len(set(smis))}')
    print(f'    Baseline MAE: {loo["baseline_pooled_mae"]:.4f}')
    print(f'    v14 MAE:      {loo["pooled_mae"]:.4f}')
    print(f'    Improvement:  {loo["baseline_pooled_mae"] - loo["pooled_mae"]:+.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=[
        'assemble', 'base_loo', 'alpha_sweep',
        'gate', 'final_loo', 'bundle',
    ])
    p.add_argument('--family', default=None)
    p.add_argument('--block', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.phase == 'assemble':
        phase_assemble(args)
    elif args.phase == 'base_loo':
        phase_base_loo(args)
    elif args.phase == 'alpha_sweep':
        phase_alpha_sweep(args)
    elif args.phase == 'gate':
        if not args.family or not args.block:
            print('--family and --block required for gate phase')
            sys.exit(1)
        phase_gate(args)
    elif args.phase == 'final_loo':
        phase_final_loo(args)
    elif args.phase == 'bundle':
        phase_bundle(args)
