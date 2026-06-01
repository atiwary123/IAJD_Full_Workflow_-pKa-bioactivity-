#!/usr/bin/env python3
"""Comprehensive cross-check: recompute descriptors from user CORRECTED SMILES vs 10118 Table S1.
Validates structures (incl. image-only-paper families) and flags real disagreements."""
import warnings,json; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
RDLogger.DisableLog('rdApp.*')
P={int(k):v for k,v in json.load(open('audit_work/p10118_tableS1.json')).items()}
df=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx').set_index('IAJD')
src=df['source'].to_dict()
def desc(smi):
    m=Chem.MolFromSmiles(str(smi))
    if not m: return None
    return {'MolWt':Descriptors.MolWt(m),'FractionCSP3':rdMolDescriptors.CalcFractionCSP3(m),
            'HBA':rdMolDescriptors.CalcNumHBA(m),'HBD':rdMolDescriptors.CalcNumHBD(m),
            'RotB':rdMolDescriptors.CalcNumRotatableBonds(m),'Aromatic':rdMolDescriptors.CalcNumAromaticRings(m)}
TOL={'MolWt':1.0,'FractionCSP3':0.01,'HBA':0,'HBD':0,'RotB':1,'Aromatic':0}  # RotB tol 1 (minor def diff)
rows=[]
for iid in sorted(set(P)&set(df.index)):
    d=desc(df.loc[iid,'SMILES'])
    if not d: continue
    dis=[k for k in TOL if abs(d[k]-P[iid][k])>(TOL[k] if TOL[k]>0 else 0.5)]
    rows.append({'IAJD':iid,'source':str(src.get(iid))[:20],'n_disagree':len(dis),'disagree':','.join(dis),
                 'MolWt_user':round(d['MolWt'],1),'MolWt_10118':P[iid]['MolWt'],
                 'FCSP3_user':round(d['FractionCSP3'],3),'FCSP3_10118':round(P[iid]['FractionCSP3'],3),
                 'HBD_u':d['HBD'],'HBD_p':int(P[iid]['HBD']),'Arom_u':d['Aromatic'],'Arom_p':int(P[iid]['Aromatic'])})
R=pd.DataFrame(rows)
R.to_csv('audit_work/cross_10118_full.csv',index=False)
print(f'shared IAJDs: {len(R)}')
print('n_disagree distribution:'); print(R['n_disagree'].value_counts().sort_index().to_string())
print('\nBy source: mean #disagree + count fully-agree (n_disagree==0):')
g=R.groupby('source').agg(n=('IAJD','size'),agree=('n_disagree',lambda s:(s==0).sum()),meandis=('n_disagree','mean'))
print(g.to_string())
print('\n=== rows disagreeing on MolWt or FractionCSP3 (real structure differences) ===')
key=R[R['disagree'].str.contains('MolWt|FractionCSP3',na=False)].sort_values('source')
for _,r in key.iterrows():
    print(f"  IAJD {r['IAJD']:>3} [{r['source']:14}] dis={r['disagree']:30} MolWt {r['MolWt_user']} vs {r['MolWt_10118']}  FCSP3 {r['FCSP3_user']} vs {r['FCSP3_10118']}  HBD {r['HBD_u']}/{r['HBD_p']} Arom {r['Arom_u']}/{r['Arom_p']}")
