#!/usr/bin/env python3
"""Fresh verification of every flagged/unresolved row against the 10118 Table S1
descriptor fingerprint, recomputed from the CURRENT dataset SMILES (not the stale
cross_10118_full.csv). Same descriptor defs + tolerances as cross_10118_full.py.

Prints, per flagged IAJD: in-10118?, n_disagree, which fields, user-vs-10118 values,
the architecture label, and the molecular formula computed from the current SMILES.
"""
import warnings, json; warnings.filterwarnings('ignore')
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')

P = {int(k): v for k, v in json.load(open('audit_work/p10118_tableS1.json')).items()}
TOL = {'MolWt':1.0,'FractionCSP3':0.01,'HBA':0,'HBD':0,'RotB':1,'Aromatic':0}

def desc(smi):
    m = Chem.MolFromSmiles(str(smi))
    if not m: return None
    return {'MolWt':Descriptors.MolWt(m),'FractionCSP3':rdMolDescriptors.CalcFractionCSP3(m),
            'HBA':rdMolDescriptors.CalcNumHBA(m),'HBD':rdMolDescriptors.CalcNumHBD(m),
            'RotB':rdMolDescriptors.CalcNumRotatableBonds(m),'Aromatic':rdMolDescriptors.CalcNumAromaticRings(m),
            'Formula':CalcMolFormula(m)}

def disagreements(d, ref):
    out=[]
    for k in TOL:
        tol = TOL[k] if TOL[k] > 0 else 0.5
        if abs(d[k]-ref[k]) > tol: out.append(k)
    return out

def is_flagged(s):
    s=str(s); return any(t in s for t in ('UNRESOLVED','FLAG','NEEDS_DIFFERENTIATION','CONFIRM_VS_PAPER'))

def run(path, idcol, label):
    df=pd.read_excel(path)
    fl=df[df['audit_status'].map(is_flagged)]
    print('='*120); print(f'{label}  ({len(fl)} flagged rows)'); print('='*120)
    for _,r in fl.iterrows():
        iid=int(r['IAJD_num']) if 'IAJD_num' in r and not pd.isna(r.get('IAJD_num')) else int(r[idcol])
        d=desc(r['SMILES'])
        arch=str(r.get('architecture'))
        if d is None:
            print(f"IAJD {iid:>3}  SMILES PARSE FAIL"); continue
        if iid in P:
            ref=P[iid]; dis=disagreements(d,ref)
            verdict='MATCH-10118' if not dis else 'DISAGREE:'+','.join(dis)
            print(f"IAJD {iid:>3} [{verdict:32}] MW {d['MolWt']:.1f} vs {ref['MolWt']:.1f} | "
                  f"HBA {d['HBA']}/{int(ref['HBA'])} HBD {d['HBD']}/{int(ref['HBD'])} "
                  f"Ar {d['Aromatic']}/{int(ref['Aromatic'])} RotB {d['RotB']}/{int(ref['RotB'])} "
                  f"FCsp3 {d['FractionCSP3']:.3f}/{ref['FractionCSP3']:.3f} | {d['Formula']:14} | {arch}")
        else:
            print(f"IAJD {iid:>3} [NOT-IN-10118 (self-consistency only)     ] MW {d['MolWt']:.1f} | "
                  f"HBA {d['HBA']} HBD {d['HBD']} Ar {d['Aromatic']} RotB {d['RotB']} "
                  f"FCsp3 {d['FractionCSP3']:.3f} | {d['Formula']:14} | {arch}")

run('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx','IAJD','pKa file')
print()
run('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx','IAJD_num','Bioact file')
