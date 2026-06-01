#!/usr/bin/env python3
"""Regenerate ALL SMILES-derived features (2D custom + 3D) for the SMILES-changed rows,
reusing the project's own compute_rdkit_features + compute_3d_features (no proxies).
Preserves structural metadata (Linker_Length/Taft from architecture; set pentanoate rows).
QM/MD features are NOT touched here (heavy — see regen doc)."""
import warnings,sys,time; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from expand_datasets import compute_rdkit_features, compute_3d_features

orig=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx').set_index('IAJD')
def canon(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else str(s)
# pentaerythritol pentanoate-linker rows -> linker length 5
PENTANOATE={273,287,291}
KEEP={'Linker_Length','Taft_Steric_Sum'}  # metadata, set explicitly

def regen(path, idc):
    df=pd.read_excel(path)
    changed=[]
    for i,r in df.iterrows():
        iid=r[idc]
        if iid not in orig.index: continue
        if canon(orig.loc[iid,'SMILES'])==canon(r['SMILES']): continue
        changed.append(i)
    print(f'{path}: {len(changed)} changed rows to regen', flush=True)
    for n,i in enumerate(changed):
        smi=df.at[i,'SMILES']; iid=df.at[i,idc]; t=time.time()
        d2=compute_rdkit_features(smi); d3=compute_3d_features(smi)
        feats={**d2,**d3}
        if 'HBD_HBA_Ratio' not in feats and feats.get('NumHAcceptors'):
            feats['HBD_HBA_Ratio']=feats['NumHDonors']/feats['NumHAcceptors'] if feats['NumHAcceptors'] else 0
        for k,v in feats.items():
            if k in df.columns and k not in KEEP: df.at[i,k]=v
        # Linker_Length / Taft (metadata): pentanoate=5 for the MeOEtPRZ PE-Tris, else keep existing (butanoate)
        if iid in PENTANOATE:
            df.at[i,'Linker_Length']=5.0; df.at[i,'linker_length']=5.0; df.at[i,'Taft_Steric_Sum']=-1.24*5
        else:
            ll=df.at[i,'Linker_Length']
            if pd.isna(ll): ll=df.at[i,'linker_length']
            if pd.notna(ll): df.at[i,'Taft_Steric_Sum']=-1.24*float(ll)
        cur=str(df.at[i,'audit_status']); cur='' if cur=='nan' else cur
        cur=cur.replace('FEATURES_NEED_REGEN','features_REGENERATED_RDKit2D+3D')
        if 'REGENERATED' not in cur: cur=(cur+'; features_REGENERATED_RDKit2D+3D').strip('; ')
        df.at[i,'audit_status']=cur[:250]
        print(f'  [{n+1}/{len(changed)}] IAJD {iid}: {time.time()-t:.1f}s Rg={feats.get("Rg_3D")} nconf={feats.get("n_confs_valid_3D")}', flush=True)
    df.to_excel(path,index=False)
    print(f'  saved {path}', flush=True)

regen('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx','IAJD')
regen('IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx','IAJD_num')
print('FEATURE REGEN DONE', flush=True)
