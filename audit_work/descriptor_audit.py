#!/usr/bin/env python3
"""
Descriptor audit (AGILE/RDKit cross-check). Recompute every standard RDKit descriptor that
exists as a dataset column, from the (corrected) SMILES, and compare to stored values.
FractionCSP3 is the flagship. Reports mismatches; can update stale values.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, GraphDescriptors
RDLogger.DisableLog('rdApp.*')

# dataset column -> (rdkit fn, float tolerance)
RDKIT={
 'ExactMolWt':(lambda m: Descriptors.ExactMolWt(m),0.02),
 'HeavyAtomCount':(lambda m: m.GetNumHeavyAtoms(),0),
 'NumNitrogens':(lambda m: sum(a.GetSymbol()=='N' for a in m.GetAtoms()),0),
 'MolLogP':(lambda m: Crippen.MolLogP(m),0.05),
 'TPSA':(lambda m: rdMolDescriptors.CalcTPSA(m),0.5),
 'LabuteASA':(lambda m: rdMolDescriptors.CalcLabuteASA(m),0.5),
 'FractionCSP3':(lambda m: rdMolDescriptors.CalcFractionCSP3(m),0.005),
 'RotatableBonds':(lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),0),
 'BertzCT':(lambda m: GraphDescriptors.BertzCT(m),1.0),
 'Chi0v':(lambda m: rdMolDescriptors.CalcChi0v(m),0.05),
 'Chi1v':(lambda m: rdMolDescriptors.CalcChi1v(m),0.05),
 'HallKierAlpha':(lambda m: rdMolDescriptors.CalcHallKierAlpha(m),0.05),
 'NumAromaticRings':(lambda m: rdMolDescriptors.CalcNumAromaticRings(m),0),
 'NumHDonors':(lambda m: rdMolDescriptors.CalcNumHBD(m),0),
 'NumHAcceptors':(lambda m: rdMolDescriptors.CalcNumHBA(m),0),
 'FractionCSP3':(lambda m: rdMolDescriptors.CalcFractionCSP3(m),0.005),
}
def audit(path, idcol, update=False):
    df=pd.read_excel(path); cols=[c for c in RDKIT if c in df.columns]
    print(f'\n### {path}  ({len(df)} rows) — checking {len(cols)} RDKit descriptor columns ###')
    mism={c:[] for c in cols}
    for i,r in df.iterrows():
        m=Chem.MolFromSmiles(str(r['SMILES']))
        if m is None: continue
        for c in cols:
            fn,tol=RDKIT[c]
            if pd.isna(r.get(c)): continue
            calc=fn(m); stored=float(r[c])
            if abs(stored-calc)> (tol if tol>0 else 0.5):
                mism[c].append((r[idcol],round(stored,4),round(float(calc),4)))
                if update: df.at[i,c]=calc
    for c in cols:
        print(f'  {c:18}: {len(mism[c])} mismatch'+(f'  e.g. {mism[c][:3]}' if mism[c] else ''))
    if update:
        df.to_excel(path,index=False); print('  -> updated stale descriptors in file')
    return mism

if __name__=='__main__':
    upd = '--update' in sys.argv
    print('=== ORIGINAL canonical files (descriptors vs ORIGINAL smiles) ===')
    audit('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx','IAJD',update=False)
    print('\n=== AUDIT_FIXED files (descriptors vs CORRECTED smiles) ===')
    audit('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx','IAJD',update=upd)
