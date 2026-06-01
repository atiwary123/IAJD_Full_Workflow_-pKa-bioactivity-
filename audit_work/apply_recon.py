#!/usr/bin/env python3
"""Apply SI-validated SMILES reconstructions (Library 5 twin-twins + IAJD 9) to AUDIT_FIXED copies."""
import warnings,json; warnings.filterwarnings('ignore')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, rdMolDescriptors
PKA='IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx'
BIO='IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx'

recon={}  # iid -> smiles (SI-formula-validated)
for iid,smi,f,ok in json.load(open('audit_work/lib5_built.json')):
    if ok: recon[iid]=smi
# IAJD 9 (Library1 48i: amide, 2 DMBA + 1 OBn) -> C75H125N3O17 (SI-validated)
i9=Chem.MolToSmiles(Chem.MolFromSmiles('CCCCCCCCCCCCOc1cc(CNC(=O)c2cc(OCCOCCOCCOC(=O)CCCN(C)C)c(OCCOCCOCCOCc3ccccc3)c(OCCOCCOCCOC(=O)CCCN(C)C)c2)cc(OCCCCCCCCCCCC)c1'))
recon[9]=i9

log=[]
def apply(df,idcol):
    for iid,newsmi in recon.items():
        m=df[idcol]==iid
        if not m.any(): continue
        old=df.loc[m,'SMILES'].iloc[0]
        mol=Chem.MolFromSmiles(newsmi)
        df.loc[m,'SMILES']=newsmi
        df.loc[m,'MolFormula']=rdMolDescriptors.CalcMolFormula(mol)
        df.loc[m,'ExactMolWt']=Descriptors.ExactMolWt(mol)
        cur=str(df.loc[m,'audit_status'].iloc[0]); cur='' if cur=='nan' else cur
        df.loc[m,'audit_status']=(cur+'; SMILES_RECONSTRUCTED_from_SI_validated').strip('; ')
        if idcol=='IAJD': log.append((iid, old, newsmi, rdMolDescriptors.CalcMolFormula(mol)))
pk=pd.read_excel(PKA); apply(pk,'IAJD'); pk.to_excel(PKA,index=False)
bi=pd.read_excel(BIO); apply(bi,'IAJD_num'); bi.to_excel(BIO,index=False)
print(f"Reconstructed & applied: {sorted(recon)} ({len(recon)} compounds)")
for iid,old,new,f in sorted(log): print(f"  IAJD {iid:>2}: -> {f}")
# append ledger
L=pd.DataFrame([(i,'SMILES_reconstruct',o,n,'SI formula validated (Scheme S5/S9)') for i,o,n,f in log],
               columns=['IAJD','field','old','new','evidence'])
prev=pd.read_csv('audit_work/AUDIT_corrections_master.csv')
pd.concat([prev,L]).to_csv('audit_work/AUDIT_corrections_master.csv',index=False)
print('ledger updated')
