#!/usr/bin/env python3
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'audit_work')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolDescriptors
from fm2 import si_neutrals, parse, norm
from recap import reduce_to_sockets, solve_caps, apply_uniform_cap, CAP_FRAG

txtfile, source = sys.argv[1], sys.argv[2]
SI=si_neutrals(open(txtfile,errors='ignore').read())
pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx')
g=pk[pk['source'].astype(str).str.contains(source,na=False)]
print(f"SI formulas={len(SI)}  rows={len(g)}\n")
asis=fixed=ambig=dead=0
results=[]
for _,r in g.iterrows():
    iid=r['IAJD']; smi=r['SMILES']; f=r['MolFormula']
    s=norm(parse(f)) if isinstance(f,str) else None
    if s and s in SI:
        asis+=1; results.append((iid,'OK_ASIS',f,smi)); continue
    base, ncap = reduce_to_sockets(smi)
    if base is None:
        dead+=1; results.append((iid,'REDUCE_FAIL',f,smi)); continue
    basef=rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(base))
    # try every SI formula
    sols=[]
    for sif in SI:
        sol=solve_caps(basef, ncap, sif)
        if sol is not None: sols.append((sif,sol))
    if not sols:
        dead+=1; results.append((iid,f'NO_CAP_MATCH(ncap={ncap},base={basef})',f,smi)); continue
    # prefer uniform single-type solutions
    def uniform(sol): return sum(1 for k in ('OMe','OBn','OPMB','OH') if sol[k]>0)<=1
    uni=[(sif,sol) for sif,sol in sols if uniform(sol)]
    pick = uni[0] if len(uni)==1 else (sols[0] if len(sols)==1 else None)
    if pick is None:
        ambig+=1; results.append((iid,f'AMBIG({len(sols)} sols)',f,smi)); continue
    sif,sol=pick
    captype=[k for k in ('OMe','OBn','OPMB') if sol[k]>0]
    if not captype:  # all OH (no change needed but didn't match as-is -> odd)
        dead+=1; results.append((iid,f'ALLOH_NOMATCH',f,smi)); continue
    new=apply_uniform_cap(base, captype[0])
    if new is None:
        dead+=1; results.append((iid,'APPLY_FAIL',f,smi)); continue
    nf=rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(new))
    ok = norm(parse(nf))==sif
    fixed+= ok; (results.append((iid,f'FIX:{captype[0]}x{sol[captype[0]]} -> {nf}{"" if ok else " (MISMATCH)"}',f,new)) )
print(f"as-is OK={asis}  fixed={fixed}  ambig={ambig}  unresolved={dead}\n")
for iid,st,oldf,smi in results:
    print(f"  IAJD {iid:>3}: {st}")
