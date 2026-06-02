#!/usr/bin/env python3
"""Extend ADMET + LiON caches with the canonical SMILES of the flag-fix rows that are now
included (changed structures 33/64/86 + the 17 MED rows) but missing from cache.
Excludes IAJD 30 (training-excluded, LOW-confidence connectivity). Skips QM/MD by design."""
import warnings, json, sys; warnings.filterwarnings('ignore')
sys.path.insert(0, 'IAJD_master/code')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from extend_caches import predict_admet_for_smiles, predict_lion_for_smiles

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

EXCLUDE = {30}  # LOW-confidence connectivity; stays out
adm = set(json.load(open('IAJD_master/bundles_caches/admet_cache_v13.json')).keys())
lion = set(json.load(open('IAJD_master/bundles_caches/lion_cache_v13.json')).keys())

want = set()
for path, idc in [('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx','IAJD'),
                  ('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx','IAJD_num')]:
    df = pd.read_excel(path)
    # included = not flagged/unresolved (so excludes 30) AND has a valid SMILES
    inc = df[~df['audit_status'].astype(str).str.contains('UNRESOLVED|FLAG', na=False)]
    for _, r in inc.iterrows():
        if int(r[idc]) in EXCLUDE: continue
        c = canon(r['SMILES'])
        if c: want.add(c)

need_adm = sorted(want - adm)
need_lion = sorted(want - lion)
print(f"included canonical SMILES: {len(want)} | missing ADMET: {len(need_adm)} | missing LiON: {len(need_lion)}")

if need_adm:
    print('computing ADMET (admet_env)...', flush=True)
    predict_admet_for_smiles(need_adm)
if need_lion:
    print('computing LiON (lion_env)...', flush=True)
    predict_lion_for_smiles(need_lion)

# verify
adm2 = set(json.load(open('IAJD_master/bundles_caches/admet_cache_v13.json')).keys())
lion2 = set(json.load(open('IAJD_master/bundles_caches/lion_cache_v13.json')).keys())
print(f"after: ADMET {len(adm)}->{len(adm2)} (still missing {len(want-adm2)}), LiON {len(lion)}->{len(lion2)} (still missing {len(want-lion2)})")
