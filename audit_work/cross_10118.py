#!/usr/bin/env python3
"""Triangulate descriptors: 10118 Table S1 (independent IAJD ML paper) vs user ORIGINAL vs CORRECTED."""
import warnings,json; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
P=json.load(open('audit_work/p10118_tableS1.json'))   # {iid(str): {LogP,TPSA,HBA,HBD,RotB,Aromatic,FractionCSP3,MolWt}}
P={int(k):v for k,v in P.items()}
orig=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx').set_index('IAJD')
corr=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx').set_index('IAJD')
# map 10118 field -> user column + tolerance
M=[('FractionCSP3','FractionCSP3',0.01),('LogP','MolLogP',0.05),('TPSA','TPSA',0.5),
   ('HBA','NumHAcceptors',0),('HBD','NumHDonors',0),('RotB','RotatableBonds',0),('Aromatic','NumAromaticRings',0)]
def agree(a,b,tol):
    if pd.isna(a) or pd.isna(b): return None
    return abs(float(a)-float(b))<=(tol if tol>0 else 0.5)
shared=[i for i in P if i in orig.index]
print(f'IAJDs in both 10118 and dataset: {len(shared)}')
# overall: does 10118 match orig or corrected? (use FractionCSP3 as primary)
both_match=onlyorig=onlycorr=neither=0
neither_ids=[]
for i in shared:
    p=P[i]['FractionCSP3']; o=orig.loc[i,'FractionCSP3']; c=corr.loc[i,'FractionCSP3']
    ao=agree(p,o,0.01); ac=agree(p,c,0.01)
    if ao and ac: both_match+=1
    elif ao and not ac: onlyorig+=1
    elif ac and not ao: onlycorr+=1
    else: neither+=1; neither_ids.append(i)
print(f'FractionCSP3 vs 10118:  match-both(unchanged rows)={both_match}  match-ORIGINAL-only={onlyorig}  match-CORRECTED-only={onlycorr}  match-NEITHER={neither}')
print('  -> match-CORRECTED-only = my fixes that 10118 independently confirms')
print('  -> match-NEITHER (needs look):', sorted(neither_ids))
# detail: for corrected rows, did 10118 confirm?
print('\nCorrected-SMILES rows: does 10118 FractionCSP3 confirm the CORRECTION?')
changed=[i for i in shared if not agree(orig.loc[i,'FractionCSP3'],corr.loc[i,'FractionCSP3'],0.001)]
for i in sorted(changed):
    p=P[i]['FractionCSP3']; ov=float(orig.loc[i,'FractionCSP3']); cv=float(corr.loc[i,'FractionCSP3'])
    tag='CONFIRMS_CORRECTION' if agree(p,cv,0.01) else ('matches_original?!' if agree(p,ov,0.01) else 'matches_neither')
    print('  IAJD %3d: 10118=%.4f  orig=%.4f  corr=%.4f  -> %s'%(i,p,ov,cv,tag))
# full multi-descriptor mismatch scan vs CORRECTED (find any remaining disagreements on unchanged rows)
print('\nRows where 10118 disagrees with CORRECTED on >=2 descriptors (potential remaining issues):')
for i in sorted(shared):
    diffs=[name for (pf,uc,tol) in M if (a:=agree(P[i][pf], corr.loc[i,uc] if uc in corr.columns else np.nan, tol)) is False for name in [pf]]
    if len(diffs)>=2:
        print(f'  IAJD {i:>3}: disagree on {diffs}')
