"""
build_admet_cache.py — Run real ADMET-AI on all v13 unique SMILES.
Saves admet_cache_v13.json: {canonical_smi: [10-d feature vector]} matching BLOCK_C_NAMES.
Saves incrementally every 25 SMILES so progress isn't lost on interruption.
"""

import os, sys, json, time
from pathlib import Path
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

import pandas as pd
import numpy as np
from admet_ai import ADMETModel

# Match BLOCK_C_NAMES order from bioact_v14_pipeline.py
ADMET_COLUMNS = [
    'PPBR_AZ', 'BBB_Martins', 'VDss_Lombardo', 'HIA_Hou', 'Caco2_Wang',
    'Pgp_Broccatelli', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ',
    'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca',
]

OUT = Path('/mnt/user-data/outputs/bioact_v14')
CACHE_PATH = OUT / 'admet_cache_v13.json'

def main():
    # Load unique SMILES from v13 dataset (use bundled smis list as canonical source)
    with open(OUT / 'bioact_v14_smis.json') as f:
        smis = json.load(f)
    unique_smis = list(dict.fromkeys(smis))  # preserve order, dedupe
    print(f'Total rows: {len(smis)}, unique SMILES: {len(unique_smis)}')

    # Resume from existing cache if any
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f'  Resuming from {len(cache)} cached SMILES')

    todo = [s for s in unique_smis if s not in cache]
    print(f'  To predict: {len(todo)} SMILES')
    if not todo:
        print('  Cache complete.')
        return

    print('Initializing ADMETModel (loads ensemble weights)...')
    model = ADMETModel()
    print('Predicting...')

    # Batch in chunks of 25
    chunk_size = 25
    t0 = time.time()
    for i in range(0, len(todo), chunk_size):
        batch = todo[i:i+chunk_size]
        try:
            df = model.predict(smiles=batch)
        except Exception as e:
            print(f'  ERROR on batch {i}: {e}')
            # Save partial progress and continue
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)
            continue

        # df indexed by SMILES, with 104 columns; extract our 10
        for s in batch:
            if s in df.index:
                row = df.loc[s]
                vec = [float(row.get(col, np.nan)) for col in ADMET_COLUMNS]
                # If admet-ai returns the same SMILES twice (unlikely), take first
                if isinstance(vec[0], pd.Series):
                    vec = [float(v.iloc[0]) for v in vec]
                cache[s] = vec
            else:
                print(f'  WARNING: {s[:50]}... not in output, filling with zeros')
                cache[s] = [0.0] * 10

        # Checkpoint every batch
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f)
        elapsed = time.time() - t0
        done = i + len(batch)
        rate = done / elapsed
        eta = (len(todo) - done) / rate if rate > 0 else 0
        print(f'  {done}/{len(todo)} done ({rate:.1f} smi/s, ETA {eta:.0f}s)')

    print(f'\nDone. Cached {len(cache)} SMILES to {CACHE_PATH}')

if __name__ == '__main__':
    main()
