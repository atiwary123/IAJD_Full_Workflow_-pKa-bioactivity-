#!/usr/bin/env python3
"""Improved formula-corroboration: correct adduct->neutral, + methyl-cap hypothesis test."""
import re, sys, warnings
import pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, RWMol
RDLogger.DisableLog('rdApp.*')

ELEM=re.compile(r'([A-Z][a-z]?)(\d*)')
def parse(f):
    d={}
    for el,n in ELEM.findall(f):
        if el: d[el]=d.get(el,0)+(int(n) if n else 1)
    return d
def norm(d): return ''.join(f'{e}{d[e]}' for e in sorted(d) if d.get(e,0)>0)
def to_neutral(adduct, ionf):
    d=parse(ionf)
    if adduct=='Na' and d.get('Na'): d=dict(d); d['Na']-=1; d.pop('Na') if d.get('Na',0)<=0 else None
    elif adduct=='K' and d.get('K'): d=dict(d); d['K']-=1; d.pop('K') if d.get('K',0)<=0 else None
    elif adduct=='H': d=dict(d); d['H']=d.get('H',0)-1
    elif adduct=='NH4': d=dict(d); d['N']=d.get('N',0)-1; d['H']=d.get('H',0)-4
    elif adduct=='2H': d=dict(d); d['H']=d.get('H',0)-2
    return {k:v for k,v in d.items() if v>0}

CALC=re.compile(r"calc(?:ulated|d|'d)?\.?\s*(?:for)?\s+([A-Z][A-Za-z0-9]+)", re.I)
def si_neutrals(txt):
    """operate on whitespace-collapsed text so line-wrapped MALDI lines are caught"""
    flat=re.sub(r'\s+',' ',txt)
    neu={}
    for m in CALC.finditer(flat):
        ionf=m.group(1)
        if not re.match(r'^C\d', ionf): continue
        pre=flat[max(0,m.start()-50):m.start()]
        am=re.search(r'\[M\s*([+\-])\s*(2H|NH4|H|Na|K)\b', pre)
        adduct=am.group(2) if am else ('Na' if ionf.endswith('Na') else ('K' if ionf.endswith('K') else None))
        # adduct-tolerant: SI inconsistently quotes neutral vs [M+H]+ ion.
        cands=set()
        base=parse(ionf)
        if ionf.endswith('Na'): base=to_neutral('Na',ionf)
        elif ionf.endswith('K'): base=to_neutral('K',ionf)
        if base.get('C',0)>=20:
            cands.add(norm(base))                                   # as quoted (or metal-stripped)
            bh=dict(base); bh['H']=bh.get('H',0)-1; cands.add(norm({k:v for k,v in bh.items() if v>0}))  # minus H ([M+H]+ ion)
        for s in cands:
            neu.setdefault(s,[]).append((ionf, adduct))
    return neu

def methyl_cap(smi):
    """convert terminal glycol -OH to -OCH3; return new smiles + #caps"""
    m=Chem.MolFromSmiles(smi)
    if m is None: return None,0
    rw=RWMol(m); caps=0; to_methylate=[]
    for a in m.GetAtoms():
        if a.GetSymbol()=='O' and a.GetTotalNumHs()==1 and a.GetDegree()==1:
            nb=a.GetNeighbors()[0]
            # terminal hydroxyl on sp3 carbon (glycol/alcohol arm terminus) — not a carboxylic acid
            if nb.GetSymbol()=='C' and not nb.GetIsAromatic() and \
               not any(b.GetBondTypeAsDouble()==2 for b in nb.GetBonds()):
                to_methylate.append(a.GetIdx())
    for oidx in to_methylate:
        c=rw.AddAtom(Chem.Atom(6))
        rw.AddBond(oidx,c,Chem.BondType.SINGLE); caps+=1
    try:
        m2=rw.GetMol(); Chem.SanitizeMol(m2); return Chem.MolToSmiles(m2),caps
    except: return None,caps

if __name__=='__main__':
    txtfile, source = sys.argv[1], sys.argv[2]
    txt=open(txtfile,errors='ignore').read()
    SI=si_neutrals(txt)
    print(f"SI neutral formulas (C>=20): {len(SI)} unique")
    pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx')
    g=pk[pk['source'].astype(str).str.contains(source,na=False)]
    asis=capfix=dead=0
    deadlist=[]
    for _,r in g.iterrows():
        f=r['MolFormula']; smi=r['SMILES']
        s=norm(parse(f)) if isinstance(f,str) else None
        if s and s in SI: asis+=1; continue
        capped,nc=methyl_cap(smi)
        if capped:
            cf=rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(capped))
            if norm(parse(cf)) in SI:
                capfix+=1
                print(f"  IAJD {r['IAJD']:>3}: dataset {f} -> methyl-cap x{nc} -> {cf}  [MATCHES SI]")
                continue
        dead+=1; deadlist.append((r['IAJD'],f))
    print(f"\nSUMMARY {source}: match-as-is={asis}  fixed-by-methylcap={capfix}  still-unmatched={dead}  (n={len(g)})")
    if deadlist:
        print("STILL UNMATCHED:")
        for iid,f in deadlist: print(f"   IAJD {iid}: {f}")
