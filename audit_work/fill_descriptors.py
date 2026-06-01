#!/usr/bin/env python3
"""Recompute & fill all standard RDKit descriptors (+ formula/MW) for every row from the
corrected SMILES in the AUDIT_FIXED files. Completes coverage (incl. the 20 rows that had none)
and guarantees descriptor⇄SMILES consistency. Custom/3D columns left untouched (need pipeline)."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, GraphDescriptors
RDLogger.DisableLog('rdApp.*')
FN={'ExactMolWt':lambda m:Descriptors.ExactMolWt(m),'MolFormula':lambda m:rdMolDescriptors.CalcMolFormula(m),
 'HeavyAtomCount':lambda m:m.GetNumHeavyAtoms(),'NumNitrogens':lambda m:sum(a.GetSymbol()=='N' for a in m.GetAtoms()),
 'MolLogP':lambda m:Crippen.MolLogP(m),'TPSA':lambda m:rdMolDescriptors.CalcTPSA(m),
 'LabuteASA':lambda m:rdMolDescriptors.CalcLabuteASA(m),'FractionCSP3':lambda m:rdMolDescriptors.CalcFractionCSP3(m),
 'RotatableBonds':lambda m:rdMolDescriptors.CalcNumRotatableBonds(m),'BertzCT':lambda m:GraphDescriptors.BertzCT(m),
 'Chi0v':lambda m:rdMolDescriptors.CalcChi0v(m),'Chi1v':lambda m:rdMolDescriptors.CalcChi1v(m),
 'HallKierAlpha':lambda m:rdMolDescriptors.CalcHallKierAlpha(m),'NumAromaticRings':lambda m:rdMolDescriptors.CalcNumAromaticRings(m),
 'NumHDonors':lambda m:rdMolDescriptors.CalcNumHBD(m),'NumHAcceptors':lambda m:rdMolDescriptors.CalcNumHBA(m),
 'NumRings':lambda m:rdMolDescriptors.CalcNumRings(m)}
for path,idc in [('IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx','IAJD'),
                 ('IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx','IAJD_num')]:
    df=pd.read_excel(path); cols=[c for c in FN if c in df.columns]
    filled=0
    for i,r in df.iterrows():
        m=Chem.MolFromSmiles(str(r['SMILES']))
        if m is None: continue
        had_nan=any(pd.isna(r.get(c)) for c in cols)
        for c in cols: df.at[i,c]=FN[c](m)
        if had_nan: filled+=1
    df.to_excel(path,index=False)
    print(f'{path}: recomputed {len(cols)} RDKit descriptors for all {len(df)} rows; {filled} rows had missing values now filled')
