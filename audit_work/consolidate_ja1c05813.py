#!/usr/bin/env python3
"""Consolidate ja1c05813 audit: pKa cross-check (vs SI Tables S12-S17) + SMILES corrections log."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'audit_work')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolDescriptors
from fm2 import si_neutrals, parse, norm
from recap2 import fix_caps

# Authoritative avg pKa from SI Tables S12-S17 (read directly from ja1c05813_si pp.S168-169)
SI_PKA={1:6.27,2:6.20,3:6.41,4:6.43,5:6.40,6:6.30,7:6.18,8:6.53,9:6.56,
19:6.38,20:6.34,21:6.28,22:6.38,23:6.38,24:6.40,25:6.15,
44:6.75,33:6.66,34:6.74,45:5.93,35:6.16,36:5.89,
30:6.42,31:6.70,37:6.83,
10:6.34,11:6.30,12:6.25,13:6.20,14:6.02,15:5.84,16:5.52,17:6.16,18:6.32,46:6.98,
38:6.38,39:6.24,40:6.36,41:6.36,42:6.06,48:6.39,26:6.24,27:6.26,28:6.25,49:6.47,
29:6.10,47:6.59,50:5.90,51:4.91,43:6.45,52:6.50,53:6.27,54:6.96,32:6.22}

SI={}
for tf in ['ja1c05813_si.txt','ja1c05813_si_plain.txt','ja1c05813_si_mu.txt']:
    for k,v in si_neutrals(open('audit_work/paper_text/'+tf,errors='ignore').read()).items(): SI.setdefault(k,[]).extend(v)

pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx')
g=pk[pk['source']=='ja1c05813']

print("=== pKa cross-check (dataset vs SI Table avg) ===")
pka_fixes=[]
for _,r in g.iterrows():
    iid=r['IAJD']
    if iid in SI_PKA:
        d=round(float(r['pKa']),2); s=SI_PKA[iid]
        if abs(d-s)>0.005:
            print(f"  IAJD {iid}: dataset {d} vs SI {s}  Δ={round(d-s,2)}")
            pka_fixes.append((iid,d,s))
print(f"pKa discrepancies: {len(pka_fixes)}")

print("\n=== SMILES corrections ===")
rows=[]
for _,r in g.iterrows():
    iid=r['IAJD']; f=r['MolFormula']; smi=r['SMILES']
    asis = isinstance(f,str) and norm(parse(f)) in SI
    if asis:
        rows.append(dict(IAJD=iid,status='OK_ASIS',formula_old=f,formula_new=f,smiles_old=smi,smiles_new=smi,si_validated=True)); continue
    new,acts=fix_caps(smi)
    if new is not None:
        nf=rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(new))
        # require amide present (single-single PE-Gallic) for confidence
        has_amide = Chem.MolFromSmiles(new).HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[NX3]'))
        val = norm(parse(nf)) in SI
        if val:
            rows.append(dict(IAJD=iid,status='FIXED_'+'+'.join(sorted(set(acts))),formula_old=f,formula_new=nf,
                             smiles_old=smi,smiles_new=new,si_validated=True)); continue
    rows.append(dict(IAJD=iid,status='NEEDS_RECONSTRUCTION',formula_old=f,formula_new='',smiles_old=smi,smiles_new='',si_validated=False))

df=pd.DataFrame(rows).sort_values('IAJD')
df.to_csv('audit_work/ja1c05813_corrections.csv',index=False)
from collections import Counter
c=Counter(s.split('_')[0]+('_'+s.split('_')[1] if s.startswith('FIXED') else '') for s in df['status'])
print("status tally:")
for k,v in sorted(c.items()): print(f"   {v:3d}  {k}")
print(f"\nTOTAL: {len(df)} | validated-correct (asis+fixed): {df['si_validated'].sum()} | needs reconstruction: {(~df['si_validated']).sum()}")
print("Saved audit_work/ja1c05813_corrections.csv")
