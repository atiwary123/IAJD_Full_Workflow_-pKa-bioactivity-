#!/usr/bin/env python3
"""Fill MISSING custom-2D + 3D features for rows whose SMILES are valid & not flagged-wrong
(originally-blank PE-Tris/G1-Janus etc.). Uses the project's own functions. 'Missing -> real'."""
import warnings,sys,time; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from expand_datasets import compute_rdkit_features, compute_3d_features
KEEP={'Linker_Length','Taft_Steric_Sum'}
CHECK=['Hydrophobic_Index','Inductive_Effect_Strength','Gasteiger_Charge_Calpha','Desolvation_Proxy',
       'HBD_HBA_Ratio','Rg_3D','Asphericity_3D','E_min_3D','Pct_V_Bur_max','n_confs_valid_3D']
def go(path, idc):
    df=pd.read_excel(path); cols=[c for c in CHECK if c in df.columns]
    todo=[]
    for i,r in df.iterrows():
        st=str(r.get('audit_status'))
        if 'UNRESOLVED' in st or 'FLAG_10118' in st: continue   # don't compute from flagged-wrong SMILES
        if df.loc[i,cols].isna().any() and Chem.MolFromSmiles(str(r['SMILES'])): todo.append(i)
    print(f'{path}: {len(todo)} rows missing custom/3D features to fill', flush=True)
    for n,i in enumerate(todo):
        smi=df.at[i,'SMILES']; t=time.time()
        feats={**compute_rdkit_features(smi),**compute_3d_features(smi)}
        if 'HBD_HBA_Ratio' not in feats and feats.get('NumHAcceptors'):
            feats['HBD_HBA_Ratio']=feats['NumHDonors']/feats['NumHAcceptors'] if feats['NumHAcceptors'] else 0
        for k,v in feats.items():
            if k in df.columns and k not in KEEP: df.at[i,k]=v
        # Taft from existing Linker_Length if present
        ll=df.at[i,'Linker_Length'] if pd.notna(df.at[i,'Linker_Length']) else df.at[i,'linker_length']
        if pd.notna(ll) and pd.isna(df.at[i,'Taft_Steric_Sum']): df.at[i,'Taft_Steric_Sum']=-1.24*float(ll)
        cur=str(df.at[i,'audit_status']); cur='' if cur=='nan' else cur
        df.at[i,'audit_status']=(cur+'; features_FILLED_RDKit2D+3D').strip('; ')[:250]
        if (n+1)%10==0 or n==len(todo)-1: print(f'  [{n+1}/{len(todo)}] {df.at[i,idc]} {time.time()-t:.1f}s', flush=True)
    df.to_excel(path,index=False); print(f'  saved {path}', flush=True)
go('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx','IAJD')
go('IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx','IAJD_num')
print('FILL MISSING DONE', flush=True)
