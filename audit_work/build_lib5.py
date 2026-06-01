#!/usr/bin/env python3
"""Reconstruct Library 5 twin-twin IAJDs (10-18,46): compound-45 core + 2 dendrons (SI Scheme S9)."""
import sys,warnings,json; warnings.filterwarnings('ignore'); sys.path.insert(0,'audit_work')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolDescriptors
from fm2 import si_neutrals, parse, norm
SI={}
for tf in ['ja1c05813_si.txt','ja1c05813_si_plain.txt','ja1c05813_si_mu.txt']:
    for k,v in si_neutrals(open('audit_work/paper_text/'+tf,errors='ignore').read()).items(): SI.setdefault(k,[]).extend(v)
C12='CCCCCCCCCCCC'
class R:  # unique ring-label allocator using %NN
    def __init__(s): s.n=10
    def __call__(s): s.n+=1; return f'%{s.n}'
def arm(kind,r):
    if kind=='DMBA': return 'OCCOCCOCCOC(=O)CCCN(C)C'
    if kind=='OMe':  return 'OCCOCCOCCOC'
    if kind=='OBn':  a=r(); return f'OCCOCCOCCOCc{a}ccccc{a}'
    if kind=='PIP':  a=r(); return f'OCCOCCOCCOC(=O)CCCN{a}CCCCC{a}'
def dendron(caps,r):
    g=r(); return f'C(=O)c{g}cc({arm(caps[0],r)})c({arm(caps[1],r)})c({arm(caps[2],r)})c{g}'
def bisC12(r):
    b=r(); return f'C(=O)c{b}cc(O{C12})cc(O{C12})c{b}'
def twin(caps):
    r=R(); return f'C(CO{bisC12(r)})(CO{bisC12(r)})(CO{dendron(caps,r)})CO{dendron(caps,r)}'
CAPS={10:['DMBA','OMe','OMe'],11:['OMe','DMBA','OMe'],12:['DMBA','DMBA','OMe'],
13:['DMBA','OMe','DMBA'],14:['DMBA','DMBA','DMBA'],15:['OBn','OBn','DMBA'],
16:['OBn','DMBA','OBn'],17:['OBn','DMBA','DMBA'],18:['DMBA','OBn','DMBA'],
46:['PIP','OBn','PIP']}
out=[]
for iid,caps in CAPS.items():
    smi=twin(caps); m=Chem.MolFromSmiles(smi)
    if not m: print(f'IAJD {iid}: BUILD FAIL  {smi[:80]}'); out.append([iid,'','',False]); continue
    f=rdMolDescriptors.CalcMolFormula(m); ok=norm(parse(f)) in SI
    print(f'IAJD {iid:>2} {caps}: {f}  SI={ok}')
    out.append([iid,Chem.MolToSmiles(m),f,ok])
json.dump(out,open('audit_work/lib5_built.json','w'))
print('validated:', sum(1 for x in out if x[3]),'/',len(out))
