#!/usr/bin/env python3
"""Apply validated reconstructions (24,273,287,291) + fix 297 head (10118 HBD=1 -> HPRZ) +
flag unresolved (Lib6 twin-mix, 30/31 Lib4, 33). Recompute descriptors for changed rows."""
import warnings,json; warnings.filterwarnings('ignore')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, rdMolDescriptors
PKA='IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx'
BIO='IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx'
R=json.load(open('audit_work/singlesingle_resolved.json'))
recon={int(k):v for k,v in R['smiles'].items()}; notes=R['notes']
# 297 -> HPRZ (dialkoxybenzyl, 10118 HBD=1). Use canonical.
i297=Chem.MolToSmiles(Chem.MolFromSmiles('CCCCC(CC)COc1ccc(C(=O)OCCCCN2CCN(CCO)CC2)cc1OCC(CC)CCCC'))
recon[297]=i297; notes['297']='head=HPRZ (10118 HBD=1 confirms bioact-file; reverts round-2 MPRZ over-alignment)'
FLAG_UNRESOLVED={**{i:'Lib6 twin-mix: complex architecture, not in 10118, scanned SI - RECOMMEND EXCLUDE from structure-features until reconstructed from hi-res figures' for i in [26,27,28,29,38,39,40,41,42,43,47,48,49,50,51,52,53,54]},
 30:'Lib4 module-D: glycol typo + scaffold unresolved vs SI - verify',31:'Lib4 module-D: unresolved vs SI - verify',
 33:'Lib3: multiple SI formula matches (ambiguous head/chain) - verify'}
RD={'ExactMolWt':lambda m:Descriptors.ExactMolWt(m),'MolFormula':lambda m:rdMolDescriptors.CalcMolFormula(m),
 'HeavyAtomCount':lambda m:m.GetNumHeavyAtoms(),'NumNitrogens':lambda m:sum(a.GetSymbol()=='N' for a in m.GetAtoms()),
 'MolLogP':lambda m:__import__('rdkit.Chem.Crippen',fromlist=['MolLogP']).MolLogP(m),'TPSA':lambda m:rdMolDescriptors.CalcTPSA(m),
 'FractionCSP3':lambda m:rdMolDescriptors.CalcFractionCSP3(m),'RotatableBonds':lambda m:rdMolDescriptors.CalcNumRotatableBonds(m),
 'NumAromaticRings':lambda m:rdMolDescriptors.CalcNumAromaticRings(m),'NumHDonors':lambda m:rdMolDescriptors.CalcNumHBD(m),
 'NumHAcceptors':lambda m:rdMolDescriptors.CalcNumHBA(m)}
log=[]
def go(df,idc):
    for iid,smi in recon.items():
        m=df[idc]==iid
        if not m.any(): continue
        old=df.loc[m,'SMILES'].iloc[0]; mol=Chem.MolFromSmiles(smi)
        df.loc[m,'SMILES']=smi
        for c,fn in RD.items():
            if c in df.columns: df.loc[m,c]=fn(mol)
        cur=str(df.loc[m,'audit_status'].iloc[0]); cur='' if cur=='nan' else cur
        df.loc[m,'audit_status']=(cur+'; SMILES_RECONSTRUCTED:'+notes.get(str(iid),'')).strip('; ')[:250]
        if idc=='IAJD': log.append((iid,'recon',old,smi))
    for iid,nt in FLAG_UNRESOLVED.items():
        m=df[idc]==iid
        if not m.any(): continue
        cur=str(df.loc[m,'audit_status'].iloc[0]); cur='' if cur=='nan' else cur
        df.loc[m,'audit_status']=(cur+'; UNRESOLVED:'+nt).strip('; ')[:250]
pk=pd.read_excel(PKA); go(pk,'IAJD'); pk.to_excel(PKA,index=False)
bi=pd.read_excel(BIO); go(bi,'IAJD_num'); bi.to_excel(BIO,index=False)
print('applied reconstructions:', sorted(recon))
print('flagged unresolved:', sorted(FLAG_UNRESOLVED))
L=pd.DataFrame([(i,'SMILES_reconstruct',o,n,notes.get(str(i),'')) for i,_,o,n in log],columns=['IAJD','field','old','new','evidence'])
prev=pd.read_csv('audit_work/AUDIT_corrections_master.csv'); pd.concat([prev,L]).to_csv('audit_work/AUDIT_corrections_master.csv',index=False)
