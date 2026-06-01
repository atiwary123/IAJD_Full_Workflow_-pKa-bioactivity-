#!/usr/bin/env python3
"""
Deterministic, formula-validated cap fixer for PE-Gallic (ja1c05813).
The dataset's WRONG cap encodes the INTENDED one:
  - free glycol -OH        -> methyl ether  -OCH3      (curator dropped the methyl)
  - phenylacetate ester    -> benzyl ether  -OCH2Ph    (curator added a spurious C=O)
    (-O-C(=O)-CH2-C6H5  ->  -O-CH2-C6H5)
  - p-methoxyphenylacetate -> PMB ether     -OCH2C6H4OMe
Each fixed compound's formula is then checked against the paper SI.
"""
import re
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, RWMol
RDLogger.DisableLog('rdApp.*')

def _branch(mol, start, blocked):
    seen=set(); st=[start]
    while st:
        i=st.pop()
        if i in seen or i==blocked: continue
        seen.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx()!=blocked: st.append(nb.GetIdx())
    return seen
def _has_basicN(mol, atoms):
    for i in atoms:
        a=mol.GetAtomWithIdx(i)
        if a.GetSymbol()=='N' and not a.GetIsAromatic():
            amide=any(nb.GetSymbol()=='C' and any(b.GetBondTypeAsDouble()==2 and b.GetOtherAtom(nb).GetSymbol()=='O' for b in nb.GetBonds()) for nb in a.GetNeighbors())
            if not amide: return True
    return False

def fix_caps(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None,[]
    methyls=[]      # O idx to methylate
    debenzoyl=[]    # (O idx, carbonylC idx, Ocarbonyl idx, Calpha idx, 'OBn'/'OPMB')
    actions=[]
    for O in m.GetAtoms():
        if O.GetSymbol()!='O': continue
        nb=O.GetNeighbors()
        if O.GetDegree()==1 and O.GetTotalNumHs()==1 and nb[0].GetSymbol()=='C' and not nb[0].GetIsAromatic():
            methyls.append(O.GetIdx()); actions.append('OH->OMe'); continue
        if O.GetDegree()==2:
            a,b=nb
            for chain,cap in ((a,b),(b,a)):
                if chain.GetSymbol()!='C' or chain.GetIsAromatic(): continue
                # ester cap: cap is carbonyl C
                if cap.GetSymbol()=='C':
                    dblO=[bd.GetOtherAtom(cap) for bd in cap.GetBonds() if bd.GetBondTypeAsDouble()==2 and bd.GetOtherAtom(cap).GetSymbol()=='O']
                    if dblO:
                        br=_branch(m, cap.GetIdx(), O.GetIdx())
                        if _has_basicN(m, br): break   # ionizable, keep
                        # alpha carbon = cap's neighbor that is C and not the =O
                        alpha=[x for x in cap.GetNeighbors() if x.GetSymbol()=='C']
                        if not alpha: break
                        ca=alpha[0]
                        # PMB vs Bn: does aromatic ring carry an OMe?
                        aroms=[i for i in br if m.GetAtomWithIdx(i).GetIsAromatic()]
                        pmb=any(any(x.GetSymbol()=='O' for x in m.GetAtomWithIdx(i).GetNeighbors()) for i in aroms)
                        debenzoyl.append((O.GetIdx(), cap.GetIdx(), dblO[0].GetIdx(), ca.GetIdx(), 'OPMB' if pmb else 'OBn'))
                        actions.append('phenylacetate->'+('OPMB' if pmb else 'OBn'))
                        break
    rw=RWMol(m); toremove=set()
    for oidx in methyls:
        c=rw.AddAtom(Chem.Atom(6)); rw.AddBond(oidx,c,Chem.BondType.SINGLE)
    for oidx,c1,oc,ca,_ in debenzoyl:
        rw.RemoveBond(oidx,c1); rw.AddBond(oidx,ca,Chem.BondType.SINGLE)
        toremove.add(c1); toremove.add(oc)
    for idx in sorted(toremove,reverse=True): rw.RemoveAtom(idx)
    try:
        m2=rw.GetMol(); Chem.SanitizeMol(m2); return Chem.MolToSmiles(m2), actions
    except Exception as e:
        return None, actions

if __name__=='__main__':
    import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'audit_work')
    import pandas as pd
    from fm2 import si_neutrals, parse, norm
    txtf, source = sys.argv[1], sys.argv[2]
    SI={}
    for tf in txtf.split(','):
        for k,v in si_neutrals(open(tf,errors='ignore').read()).items(): SI.setdefault(k,[]).extend(v)
    pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx')
    g=pk[pk['source'].astype(str).str.contains(source,na=False)]
    okv=fixv=dead=0; out={}
    for _,r in g.iterrows():
        iid=r['IAJD']; f=r['MolFormula']; smi=r['SMILES']
        if isinstance(f,str) and norm(parse(f)) in SI: okv+=1; out[iid]=('OK_ASIS',smi); continue
        new,acts=fix_caps(smi)
        if new is None: dead+=1; out[iid]=('FIX_FAIL '+str(acts),smi); continue
        nf=rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(new))
        val = norm(parse(nf)) in SI
        if val: fixv+=1; out[iid]=(f'FIXED {sorted(set(acts))} -> {nf} [SI✓]',new)
        else: dead+=1; out[iid]=(f'NOMATCH after {sorted(set(acts))} -> {nf} (was {f})',new)
    print(f"{source}: as-is={okv} fixed&validated={fixv} unresolved={dead} (n={len(g)})\n")
    for iid in sorted(out): print(f"  IAJD {iid:>3}: {out[iid][0]}")
