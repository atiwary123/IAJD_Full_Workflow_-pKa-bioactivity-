#!/usr/bin/env python3
"""Incremental validated fixes/flags on the AUDIT_FIXED copies (round 2)."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
PKA='IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx'
BIO='IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx'

pk=pd.read_excel(PKA); bi=pd.read_excel(BIO)
log=[]
def flag(df,idcol,iid,note):
    m=df[idcol]==iid
    if m.any():
        cur=str(df.loc[m,'audit_status'].iloc[0]); cur='' if cur=='nan' else cur
        df.loc[m,'audit_status']=(cur+'; '+note).strip('; ')

# 1. pharmaceutics pKa fixes (verified vs Table S1)
for iid,newp in {266:6.54, 268:6.45}.items():
    m=pk['IAJD']==iid
    if m.any():
        old=float(pk.loc[m,'pKa'].iloc[0]); pk.loc[m,'pKa']=newp; flag(pk,'IAJD',iid,'pKa_corrected_vs_pharm_TableS1')
        log.append((iid,'pKa',old,newp,'pharmaceutics Table S1'))

# 2. source attribution for NaN-source rows confirmed in ja3c07337 Table S1
for iid in [92,100,101]:
    m=pk['IAJD']==iid
    if m.any() and pd.isna(pk.loc[m,'source'].iloc[0]):
        pk.loc[m,'source']='ja3c07337'; flag(pk,'IAJD',iid,'source_attributed_ja3c07337')
        log.append((iid,'source','NaN','ja3c07337','ja3c07337 Table S1 lists this IAJD'))

# 3. flag dup-pair SMILES (distinct compounds in paper, identical SMILES in dataset)
for iid in [287,290,291,292]:
    flag(pk,'IAJD',iid,'SMILES_ERROR_distinct_paper_compound_collapsed_to_shared_SMILES_NEEDS_DIFFERENTIATION')
    log.append((iid,'SMILES_flag','identical within pair','distinct in ja3c07337 Table S1','pKa differs: 287=6.30/292=6.48; 290=6.50/291=6.42'))

# 4. cross-file head-group conflict: align bioact SMILES to the grid-consistent pKa-file SMILES + flag
for iid in [248,273,297]:
    mp=pk['IAJD']==iid; mb=bi['IAJD_num']==iid
    if mp.any() and mb.any():
        good=pk.loc[mp,'SMILES'].iloc[0]; oldb=bi.loc[mb,'SMILES'].iloc[0]
        if str(good)!=str(oldb):
            bi.loc[mb,'SMILES']=good; flag(bi,'IAJD_num',iid,'SMILES_aligned_to_pKa_file_grid_consistent_CONFIRM_VS_PAPER')
            flag(pk,'IAJD',iid,'cross_file_head_conflict_resolved_to_this_value_CONFIRM_VS_PAPER')
            log.append((iid,'SMILES_bioact','HPRZ/ethoxyethyl(bioact)','MPRZ(pKa-file,grid-consistent)','self-consistency; confirm vs ja3c07337/ja3c13569'))

pk.to_excel(PKA,index=False); bi.to_excel(BIO,index=False)
L=pd.DataFrame(log,columns=['IAJD','field','old','new','evidence'])
# append to master ledger
master='audit_work/AUDIT_corrections_master.csv'
prev=pd.read_csv(master); prev['paper']=prev.get('paper','')
pd.concat([prev[['IAJD','field','old','new','evidence']],L]).to_csv(master,index=False)
print('Round-2 changes:'); print(L.to_string(index=False))
print('\nUpdated AUDIT_FIXED copies + master ledger.')
