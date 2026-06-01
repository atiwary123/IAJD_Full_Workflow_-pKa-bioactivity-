#!/usr/bin/env python3
"""Reconstruct remaining single-single specials (24 Lib2, 30/31 Lib4) vs primary SI formula;
build PE-Tris 273/287/291. Output validated SMILES dict."""
import warnings,json,sys,itertools; warnings.filterwarnings('ignore'); sys.path.insert(0,'audit_work')
import pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolDescriptors, Descriptors
from fm2 import si_neutrals, parse, norm
SI={}
for tf in ['ja1c05813_si.txt','ja1c05813_si_plain.txt','ja1c05813_si_mu.txt']:
    for k,v in si_neutrals(open('audit_work/paper_text/'+tf,errors='ignore').read()).items(): SI.setdefault(k,[]).extend(v)
def F(s):
    m=Chem.MolFromSmiles(s); return (rdMolDescriptors.CalcMolFormula(m), Chem.MolToSmiles(m)) if m else (None,None)
def sival(s): f,_=F(s); return f and norm(parse(f)) in SI
out={}; notes={}

# --- IAJD 24 (Lib2): dm8 chains, 2 DMBA + 1 OBn (SI-validated; trust SI over 10118) ---
DMBA='OCCOCCOCCOC(=O)CCCN(C)C'; OBn='OCCOCCOCCOCc3ccccc3'
i24=f'OCCC(C)CCCC(C)Cc1cc(CNC(=O)c2cc({DMBA})c({OBn})c({DMBA})c2)cc(OCCC(C)CCCC(C)C)c1'
f24,c24=F(i24)
print(f'IAJD 24 (dm8,2DMBA+1OBn): {f24} SI={sival(i24)}')
if sival(i24): out[24]=c24; notes[24]='SI-validated (dm8 chains, 2 DMBA + 1 OBn); 10118 differs (its known multi-DMBA offset)'

# --- IAJD 30, 31 (Lib4): fix acetal typo OCCOCCOCOC -> OCCOCCOCCOC, validate vs SI ---
pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx').set_index('IAJD')
for iid in [30,31]:
    old=pk.loc[iid,'SMILES']
    fixed=old.replace('OCCOCCOCOC','OCCOCCOCCOC').replace('%20','1')
    f,c=F(fixed)
    ok=sival(fixed) if c else False
    print(f'IAJD {iid}: typo-fix -> {f} SI={ok}')
    if ok: out[iid]=c; notes[iid]='Lib4 acetal-typo fixed (OCH2O->OCH2CH2O glycol); SI-validated'

# --- PE-Tris 273/287/291 (10118-resolved: MeOEtPRZ + pentanoate) ---
HEADS={'MeOEtPRZ':'N1CCN(CCOC)CC1'}
def petris(n): return f'C(CO{"C"*n})(CO{"C"*n})(CO{"C"*n})COC(=O)CCCC{HEADS["MeOEtPRZ"]}'
for iid,n in [(273,12),(287,8),(291,7)]:
    f,c=F(petris(n)); out[iid]=c; notes[iid]=f'10118-vector-resolved: C{n}+methoxyethyl-piperazine+pentanoate ({f}); image-SI not text-verified'
    print(f'IAJD {iid}: PE-Tris C{n}+MeOEtPRZ+pentanoate -> {f}')

json.dump({'smiles':out,'notes':notes}, open('audit_work/singlesingle_resolved.json','w'))
print('\nResolved:', sorted(out))
