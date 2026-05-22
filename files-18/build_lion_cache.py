#!/usr/bin/env /home/claude/lion_env/bin/python3
"""
build_lion_cache.py — Run real LION inference on all v13 unique SMILES.

For each of 236 unique SMILES × 6 tissues × 5 CV models (cv_0..cv_4):
    - run chemprop predict
    - average across CV models
This produces a 14-dim LION feature vector per SMILES (6 z-flux + max + 6 argmax-OH + OOD flag).

Saves lion_cache_v13.json: {canonical_smi: [14-d feature vector]}

Strategy: All 236 SMILES go through each (tissue, cv_model) call as one batch.
That's 30 batched chemprop calls total (6 tissues × 5 CVs), each on 236 SMILES.
Empirically a single batch of 236 takes ~3-5s, so total wall-clock ≈ 2-3 min.
"""

import os, sys, json, time, tempfile, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

OUT = Path('/mnt/user-data/outputs/bioact_v14')
LION_REPO = Path('/home/claude/LNP_ML')
CACHE_PATH = OUT / 'lion_cache_v13.json'

LION_TISSUES = ['liver_IV', 'lung_IT', 'lung_inh', 'lung_neb', 'muscle_IM', 'nasal']

# Feature column order (must match LION training)
LION_FEAT_COLS = [
    'Cationic_Lipid_to_mRNA_weight_ratio', 'Cationic_Lipid_Mol_Ratio',
    'Phospholipid_Mol_Ratio', 'Cholesterol_Mol_Ratio', 'PEG_Lipid_Mol_Ratio',
    'Delivery_target_dendritic_cell', 'Delivery_target_generic_cell',
    'Delivery_target_liver', 'Delivery_target_lung',
    'Delivery_target_lung_epithelium', 'Delivery_target_macrophage',
    'Delivery_target_muscle', 'Delivery_target_spleen',
    'Helper_lipid_ID_DOPE', 'Helper_lipid_ID_DOTAP', 'Helper_lipid_ID_DSPC',
    'Helper_lipid_ID_MDOA', 'Helper_lipid_ID_None',
    'Route_of_administration_in_vitro', 'Route_of_administration_intramuscular',
    'Route_of_administration_intratracheal', 'Route_of_administration_intravenous',
    'Batch_or_individual_or_barcoded_Barcoded',
    'Batch_or_individual_or_barcoded_Individual',
    'Cargo_type_mRNA', 'Cargo_type_pDNA', 'Cargo_type_siRNA',
    'Model_type_A549', 'Model_type_BDMC', 'Model_type_BMDM',
    'Model_type_HBEC_ALI', 'Model_type_HEK293T', 'Model_type_HeLa',
    'Model_type_IGROV1', 'Model_type_Mouse', 'Model_type_RAW264p7',
]
assert len(LION_FEAT_COLS) == 36

_TISSUE_CONTEXT = {
    'liver_IV':  ('Delivery_target_liver',           'Route_of_administration_intravenous'),
    'lung_IT':   ('Delivery_target_lung',            'Route_of_administration_intratracheal'),
    'lung_inh':  ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
    'lung_neb':  ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
    'muscle_IM': ('Delivery_target_muscle',          'Route_of_administration_intramuscular'),
    'nasal':     ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
}


def build_context(tissue):
    row = {c: 0 for c in LION_FEAT_COLS}
    row['Cationic_Lipid_to_mRNA_weight_ratio'] = 10.0
    row['Cationic_Lipid_Mol_Ratio'] = 35.0
    row['Phospholipid_Mol_Ratio'] = 16.0
    row['Cholesterol_Mol_Ratio']  = 46.5
    row['PEG_Lipid_Mol_Ratio']    = 2.5
    target_col, route_col = _TISSUE_CONTEXT[tissue]
    row[target_col] = 1
    row[route_col]  = 1
    row['Helper_lipid_ID_DOPE'] = 1
    row['Cargo_type_mRNA'] = 1
    row['Model_type_Mouse'] = 1
    row['Batch_or_individual_or_barcoded_Individual'] = 1
    return [row[c] for c in LION_FEAT_COLS]


def find_checkpoints():
    paths = []
    for cv in range(5):
        p = LION_REPO / f'data/crossval_splits/all_random_split_for_paper/cv_{cv}/fold_0/model_0/model.pt'
        if p.exists():
            paths.append(str(p))
    return paths


def predict_tissue_one_ckpt(smiles_list, tissue, ckpt_path):
    """One chemprop call: all SMILES × one tissue × one checkpoint → array shape (n,)."""
    from chemprop.train.make_predictions import make_predictions
    from chemprop.args import PredictArgs
    n = len(smiles_list)
    with tempfile.TemporaryDirectory() as tmpdir:
        smi_path = os.path.join(tmpdir, 'smi.csv')
        feat_path = os.path.join(tmpdir, 'feats.csv')
        out_path = os.path.join(tmpdir, 'preds.csv')
        with open(smi_path, 'w') as f:
            f.write('smiles\n')
            for s in smiles_list: f.write(s + '\n')
        feat_row = build_context(tissue)
        with open(feat_path, 'w') as f:
            f.write(','.join(LION_FEAT_COLS) + '\n')
            row_str = ','.join(str(v) for v in feat_row)
            for _ in smiles_list:
                f.write(row_str + '\n')
        args = PredictArgs().parse_args([
            '--test_path', smi_path,
            '--features_path', feat_path,
            '--checkpoint_path', ckpt_path,
            '--preds_path', out_path,
            '--num_workers', '0',
        ])
        preds = make_predictions(args=args)
        return np.array([p[0] if (p and p[0] is not None) else 0.0 for p in preds])


def main():
    print('='*70)
    print('Building real LION cache for v13 unique SMILES')
    print('='*70)

    # Load unique SMILES
    with open(OUT / 'bioact_v14_smis.json') as f:
        smis = json.load(f)
    unique_smis = list(dict.fromkeys(smis))
    print(f'Total rows: {len(smis)}, unique SMILES: {len(unique_smis)}')

    ckpts = find_checkpoints()
    print(f'Found {len(ckpts)} LION checkpoints')
    if not ckpts:
        print('No checkpoints found — aborting')
        return

    # For each tissue, predict with each CV model, average → shape (n_smiles, 6)
    intermediate_path = OUT / 'lion_intermediate.npz'
    z_per_tissue = {}

    # Resume support
    if intermediate_path.exists():
        d = np.load(intermediate_path, allow_pickle=True)
        z_per_tissue = {k: d[k] for k in d.files if k in LION_TISSUES}
        print(f'Resumed: {list(z_per_tissue.keys())}')

    t0 = time.time()
    for tissue in LION_TISSUES:
        if tissue in z_per_tissue:
            print(f'  [{tissue}] already done, skipping')
            continue
        cv_preds = []
        for ck in ckpts:
            print(f'  [{tissue}] cv={os.path.basename(os.path.dirname(os.path.dirname(ck)))}')
            t_one = time.time()
            preds = predict_tissue_one_ckpt(unique_smis, tissue, ck)
            print(f'    {len(unique_smis)} preds in {time.time()-t_one:.1f}s')
            cv_preds.append(preds)
        z_per_tissue[tissue] = np.mean(np.stack(cv_preds), axis=0)
        # Checkpoint after each tissue (saves work on interruption)
        np.savez(intermediate_path, **z_per_tissue)
        print(f'  [{tissue}] saved checkpoint, elapsed total {(time.time()-t0)/60:.1f} min')

    # Build OOD flags using LION training set Morgan FPs
    print('\nComputing OOD flags (Tanimoto to LION training set)...')
    fps_path = OUT / 'lion_train_fps.pkl'
    if fps_path.exists():
        import pickle
        with open(fps_path, 'rb') as f:
            train_fps = pickle.load(f)
    else:
        csv_path = LION_REPO / 'data/all_data.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=['smiles'])
            mfp = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            train_smis = df['smiles'].dropna().unique().tolist()
            print(f'  Computing Morgan FPs for {len(train_smis)} LION train SMILES...')
            train_fps = []
            for s in train_smis:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    train_fps.append(mfp.GetFingerprint(m))
            import pickle
            with open(fps_path, 'wb') as f:
                pickle.dump(train_fps, f)
        else:
            train_fps = []

    mfp = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    ood_flags = []
    for s in unique_smis:
        m = Chem.MolFromSmiles(s)
        if m is None or not train_fps:
            ood_flags.append(1)
            continue
        fp = mfp.GetFingerprint(m)
        max_sim = max((TanimotoSimilarity(fp, tf) for tf in train_fps), default=0.0)
        ood_flags.append(int(max_sim < 0.30))
    print(f'  OOD count: {sum(ood_flags)}/{len(ood_flags)}')

    # Build 14-d cache vectors
    cache = {}
    for i, smi in enumerate(unique_smis):
        z = np.array([z_per_tissue[t][i] for t in LION_TISSUES])
        if np.allclose(z, 0):
            max_z = 0.0
            argmax_idx = -1
        else:
            argmax_idx = int(np.argmax(z))
            max_z = float(z[argmax_idx])
        one_hot = np.zeros(len(LION_TISSUES))
        if argmax_idx >= 0:
            one_hot[argmax_idx] = 1.0
        vec = np.concatenate([z, [max_z], one_hot, [ood_flags[i]]])
        cache[smi] = vec.tolist()

    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    print(f'\nDone. Saved {len(cache)} cached LION vectors to {CACHE_PATH}')
    print(f'Total elapsed: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
