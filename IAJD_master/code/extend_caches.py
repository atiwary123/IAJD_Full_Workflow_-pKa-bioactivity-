"""
extend_caches.py — Extend lion_cache_v13.json and admet_cache_v13.json with new SMILES.

Use this when you have new query SMILES that aren't in the training set, to compute
their real LION and ADMET predictions before running v14 inference. Otherwise the
inference falls back to RDKit-based proxies which significantly under-represent
the real LION/ADMET signal.

Usage from Python:
    from extend_caches import predict_admet_for_smiles, predict_lion_for_smiles
    predict_admet_for_smiles(['CCO', '...'])  # adds to admet_cache_v13.json
    predict_lion_for_smiles(['CCO', '...'])   # adds to lion_cache_v13.json
"""
import os, sys, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

OUT = Path(os.environ.get(
    'IAJD_OUT_DIR',
    str(Path(__file__).resolve().parent.parent / 'bundles_caches'),
))
_APP_ROOT = Path(__file__).resolve().parent.parent.parent
LION_REPO = Path(os.environ.get('LION_REPO', str(_APP_ROOT / 'lion_repo')))
LION_VENV_PYTHON = os.environ.get('LION_VENV_PYTHON', str(_APP_ROOT / 'lion_env' / 'bin' / 'python3'))
ADMET_VENV_PYTHON = os.environ.get('ADMET_VENV_PYTHON', str(_APP_ROOT / 'admet_env' / 'bin' / 'python3'))

ADMET_COLUMNS = [
    'PPBR_AZ', 'BBB_Martins', 'VDss_Lombardo', 'HIA_Hou', 'Caco2_Wang',
    'Pgp_Broccatelli', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ',
    'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca',
]


def _canonical(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


ADMET_RUNNER_SCRIPT = '''
import sys, json, warnings
warnings.filterwarnings("ignore")
from admet_ai import ADMETModel

ADMET_COLUMNS = {ADMET_COLUMNS}
todo = {TODO}

m = ADMETModel()
df = m.predict(smiles=todo)
out = {{}}
for s in todo:
    if s in df.index:
        row = df.loc[s]
        if hasattr(row, "iloc") and hasattr(row, "shape") and len(row.shape) > 1:
            row = df.loc[[s]].iloc[0]
        out[s] = [float(row.get(c, float("nan"))) for c in ADMET_COLUMNS]
    else:
        out[s] = [0.0] * len(ADMET_COLUMNS)

print("OUTPUT_START")
print(json.dumps(out))
print("OUTPUT_END")
'''


def predict_admet_for_smiles(smiles_list, cache_path=None, verbose=True):
    """Run real ADMET-AI for new SMILES via a fully isolated subprocess
    (admet_env). Updates cache; returns dict {canonical_smi: 10-d vec}.

    Isolating in a subprocess avoids the libomp / libtorch_cpu deadlock that
    occurs when admet-ai's lightning is loaded in the same process as XGBoost
    training (the v15 direct-model refit) on macOS.
    """
    if cache_path is None:
        cache_path = OUT / 'admet_cache_v13.json'
    cache_path = Path(cache_path)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}

    canonicals = [_canonical(s) for s in smiles_list]
    todo = [c for c in canonicals if c is not None and c not in cache]
    if not todo:
        if verbose: print('All SMILES already cached.')
        return {c: cache[c] for c in canonicals if c is not None and c in cache}

    if verbose: print(f'Computing real ADMET for {len(todo)} new SMILES (uses {ADMET_VENV_PYTHON})...')

    script = ADMET_RUNNER_SCRIPT.format(
        ADMET_COLUMNS=json.dumps(ADMET_COLUMNS),
        TODO=json.dumps(todo),
    )
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    # Clean env so parent's torch state can't leak into the child
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("DYLD_", "LD_LIBRARY_PATH", "PYTHON"))
    }
    clean_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    clean_env["OMP_NUM_THREADS"] = "1"

    try:
        result = subprocess.run(
            [ADMET_VENV_PYTHON, script_path],
            capture_output=True, text=True,
            timeout=int(os.environ.get('ADMET_TIMEOUT_S', '600')),
            env=clean_env, start_new_session=True,
        )
        out_text = result.stdout
        if 'OUTPUT_START' not in out_text:
            if verbose:
                print(f'ADMET subprocess failed (rc={result.returncode}):\n{result.stderr[:500]}')
            return {}
        json_str = out_text.split('OUTPUT_START')[1].split('OUTPUT_END')[0].strip()
        new_results = json.loads(json_str)
    finally:
        os.unlink(script_path)

    cache.update(new_results)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    if verbose: print(f'Updated {cache_path} → {len(cache)} entries.')
    return {c: cache[c] for c in canonicals if c is not None}


# LION inference must be run in a separate venv (chemprop 1.6.1 vs admet-ai's chemprop 2.x)
LION_RUNNER_SCRIPT = '''
import os, sys, json, tempfile
import numpy as np
import torch
_orig_load = torch.load
def _safe_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _safe_load
LION_REPO = "{LION_REPO}"
OUT = "{OUT}"
todo = {TODO}

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
TISSUES = ['liver_IV','lung_IT','lung_inh','lung_neb','muscle_IM','nasal']
TISSUE_CTX = {{
    'liver_IV':  ('Delivery_target_liver',           'Route_of_administration_intravenous'),
    'lung_IT':   ('Delivery_target_lung',            'Route_of_administration_intratracheal'),
    'lung_inh':  ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
    'lung_neb':  ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
    'muscle_IM': ('Delivery_target_muscle',          'Route_of_administration_intramuscular'),
    'nasal':     ('Delivery_target_lung_epithelium', 'Route_of_administration_intratracheal'),
}}

def context_row(tissue):
    row = {{c: 0 for c in LION_FEAT_COLS}}
    row.update({{'Cationic_Lipid_to_mRNA_weight_ratio': 10.0,
                'Cationic_Lipid_Mol_Ratio': 35.0, 'Phospholipid_Mol_Ratio': 16.0,
                'Cholesterol_Mol_Ratio': 46.5, 'PEG_Lipid_Mol_Ratio': 2.5,
                'Helper_lipid_ID_DOPE': 1, 'Cargo_type_mRNA': 1, 'Model_type_Mouse': 1,
                'Batch_or_individual_or_barcoded_Individual': 1}})
    t_col, r_col = TISSUE_CTX[tissue]
    row[t_col] = 1
    row[r_col] = 1
    return [row[c] for c in LION_FEAT_COLS]

import warnings; warnings.filterwarnings('ignore')
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs

ckpts = [f'{{LION_REPO}}/data/crossval_splits/all_random_split_for_paper/cv_{{i}}/fold_0/model_0/model.pt' for i in range(5)]
z_per_tissue = {{}}

for tissue in TISSUES:
    cv_preds = []
    for ck in ckpts:
        with tempfile.TemporaryDirectory() as td:
            smi_p = os.path.join(td, 'smi.csv')
            feat_p = os.path.join(td, 'feats.csv')
            out_p = os.path.join(td, 'preds.csv')
            with open(smi_p, 'w') as f:
                f.write('smiles\\n')
                for s in todo: f.write(s + '\\n')
            ctx = context_row(tissue)
            with open(feat_p, 'w') as f:
                f.write(','.join(LION_FEAT_COLS) + '\\n')
                rs = ','.join(str(v) for v in ctx)
                for _ in todo: f.write(rs + '\\n')
            args = PredictArgs().parse_args([
                '--test_path', smi_p, '--features_path', feat_p,
                '--checkpoint_path', ck, '--preds_path', out_p, '--num_workers', '0',
            ])
            preds = make_predictions(args=args)
            cv_preds.append(np.array([p[0] if p and p[0] is not None else 0.0 for p in preds]))
    z_per_tissue[tissue] = np.mean(np.stack(cv_preds), axis=0).tolist()

print('OUTPUT_START')
print(json.dumps(z_per_tissue))
print('OUTPUT_END')
'''


def predict_lion_for_smiles(smiles_list, cache_path=None, verbose=True):
    """Run real LION on new SMILES; update cache; return dict {canonical_smi: 14-d vec}."""
    if cache_path is None:
        cache_path = OUT / 'lion_cache_v13.json'
    cache_path = Path(cache_path)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}

    canonicals = [_canonical(s) for s in smiles_list]
    todo = [c for c in canonicals if c is not None and c not in cache]
    if not todo:
        if verbose: print('All SMILES already cached.')
        return {c: cache[c] for c in canonicals if c is not None and c in cache}

    if verbose: print(f'Computing real LION for {len(todo)} new SMILES (uses {LION_VENV_PYTHON})...')

    script = LION_RUNNER_SCRIPT.format(
        LION_REPO=str(LION_REPO), OUT=str(OUT), TODO=json.dumps(todo)
    )
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    # Use a clean env so the parent's torch/lightning libraries can't leak into
    # the child via DYLD_*/LD_LIBRARY_PATH. The child is a separate venv with
    # its own pinned chemprop 1.6.1 + torch 2.12 stack.
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("DYLD_", "LD_LIBRARY_PATH", "PYTHON"))
    }
    clean_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    clean_env["OMP_NUM_THREADS"] = "1"
    try:
        result = subprocess.run(
            [LION_VENV_PYTHON, script_path],
            capture_output=True, text=True,
            timeout=int(os.environ.get('LION_TIMEOUT_S', '1800')),
            env=clean_env, start_new_session=True,
        )
        out = result.stdout
        if 'OUTPUT_START' not in out:
            err_msg = result.stderr[:500] if result.stderr else '(no stderr)'
            out_msg = result.stdout[:500] if result.stdout else '(no stdout)'
            if verbose:
                print(f'LION inference failed (returncode={result.returncode}):')
                print(f'  stderr: {err_msg}')
                print(f'  stdout: {out_msg}')
            raise RuntimeError(f'LION chemprop failed (rc={result.returncode}): stderr={err_msg} stdout={out_msg}')
        json_str = out.split('OUTPUT_START')[1].split('OUTPUT_END')[0].strip()
        z_per_tissue = json.loads(json_str)
    finally:
        os.unlink(script_path)

    # Compute OOD flags
    fps_path = OUT / 'lion_train_fps.pkl'
    train_fps = []
    if fps_path.exists():
        import pickle
        with open(fps_path, 'rb') as f:
            train_fps = pickle.load(f)
    try:
        mfp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        def _fp(mol):
            return mfp_gen.GetFingerprint(mol)
    except AttributeError:
        def _fp(mol):
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    tissues = ['liver_IV','lung_IT','lung_inh','lung_neb','muscle_IM','nasal']
    for i, c in enumerate(todo):
        z = np.array([z_per_tissue[t][i] for t in tissues])
        if np.allclose(z, 0):
            max_z, argmax_idx = 0.0, -1
        else:
            argmax_idx = int(np.argmax(z))
            max_z = float(z[argmax_idx])
        one_hot = np.zeros(len(tissues))
        if argmax_idx >= 0:
            one_hot[argmax_idx] = 1.0
        # OOD
        m = Chem.MolFromSmiles(c)
        if m and train_fps:
            fp = _fp(m)
            max_sim = max((TanimotoSimilarity(fp, tf) for tf in train_fps), default=0.0)
            ood = int(max_sim < 0.30)
        else:
            ood = 1
        vec = np.concatenate([z, [max_z], one_hot, [ood]]).tolist()
        cache[c] = vec

    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    if verbose: print(f'Updated {cache_path} → {len(cache)} entries.')
    return {c: cache[c] for c in canonicals if c is not None}


def predict_for_smiles(smiles_list, verbose=True):
    """Compute both LION and ADMET for new SMILES; update both caches."""
    lion = predict_lion_for_smiles(smiles_list, verbose=verbose)
    admet = predict_admet_for_smiles(smiles_list, verbose=verbose)
    return {'lion': lion, 'admet': admet}


if __name__ == '__main__':
    test_smis = sys.argv[1:] if len(sys.argv) > 1 else [
        'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC'
    ]
    out = predict_for_smiles(test_smis)
    print('\nResults:')
    for s in test_smis:
        c = _canonical(s)
        print(f'  {s[:50]}...')
        if c in out['lion']:
            print(f'    LION (14-d): {[round(x, 3) for x in out["lion"][c]]}')
        if c in out['admet']:
            print(f'    ADMET (10-d): {[round(x, 3) for x in out["admet"][c]]}')
