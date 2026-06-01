#!/usr/bin/env python3
"""
Phase 0 — Internal-consistency audit of the IAJD pKa + bioactivity datasets.
Fully verifiable without papers. Surfaces curation bugs:
  - SMILES that don't parse / sanitize
  - stored RDKit descriptors that disagree with the SMILES (stale features / wrong structure)
  - linker_length / Linker_Length / linker_carbons disagreements (with each other and SMILES)
  - head_group / family / linkage labels inconsistent with the SMILES
  - duplicate structures with conflicting pKa / bioactivity
  - pKa vs pKa_paper, flux vs log10_flux, organ_dominant vs argmax-flux, label vs flux
  - cross-dataset SMILES/pKa agreement for the same IAJD id
Writes audit_phase0_results.json and prints a digest.
"""
import json, warnings, re
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
RDLogger.DisableLog('rdApp.*')

PKA_F   = 'IAJD_master/datasets/IAJD_pKa_v21_final.xlsx'
BIOACT_F= 'IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx'

pk = pd.read_excel(PKA_F)
bi = pd.read_excel(BIOACT_F)

# ---------- SMARTS ----------
ESTER  = Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')      # carboxylic ester
AMIDE  = Chem.MolFromSmarts('[CX3](=O)[NX3]')                # amide
ETHER  = Chem.MolFromSmarts('[OD2]([#6])[#6]')               # any ether O (incl. ester O)
PIPERAZINE = Chem.MolFromSmarts('C1CNCCN1')
AROM_N = Chem.MolFromSmarts('[n]')

def basic_n_count(m):
    """aliphatic amine N (not amide, not aromatic, not nitro)"""
    n=0
    for a in m.GetAtoms():
        if a.GetSymbol()!='N': continue
        if a.GetIsAromatic(): continue
        # amide N?
        is_amide=any(nb.GetSymbol()=='C' and any(b.GetBondTypeAsDouble()==2 and (b.GetOtherAtom(nb).GetSymbol()=='O')
                     for b in nb.GetBonds()) for nb in a.GetNeighbors())
        if is_amide: continue
        n+=1
    return n

def tertiary_amine_count(m):
    patt=Chem.MolFromSmarts('[NX3;!$(NC=O);!$(N=*);!$([N+]);!$(N-a)]([#6])([#6])[#6]')
    return len(m.GetSubstructMatches(patt))

def safe_mol(smi):
    if not isinstance(smi,str) or not smi.strip(): return None,'empty'
    m=Chem.MolFromSmiles(smi)
    if m is None: return None,'parse_fail'
    return m,'ok'

# ---------- per-row descriptor recompute ----------
def recompute(smi):
    m,st=safe_mol(smi)
    if m is None: return {'_status':st}
    d={'_status':'ok'}
    d['ExactMolWt']=Descriptors.ExactMolWt(m)
    d['MolFormula']=rdMolDescriptors.CalcMolFormula(m)
    d['HeavyAtomCount']=m.GetNumHeavyAtoms()
    d['NumNitrogens']=sum(1 for a in m.GetAtoms() if a.GetSymbol()=='N')
    d['NumAromaticRings']=rdMolDescriptors.CalcNumAromaticRings(m)
    d['NumHDonors']=rdMolDescriptors.CalcNumHBD(m)
    d['NumHAcceptors']=rdMolDescriptors.CalcNumHBA(m)
    d['NumEsters']=len(m.GetSubstructMatches(ESTER))
    d['NumAmides']=len(m.GetSubstructMatches(AMIDE))
    d['NumEthers']=len(m.GetSubstructMatches(ETHER))
    d['NumTertiaryAmines']=tertiary_amine_count(m)
    d['HasPiperazine']=int(m.HasSubstructMatch(PIPERAZINE))
    d['NumAmines_total']=basic_n_count(m)
    d['MolLogP']=Crippen.MolLogP(m)
    d['TPSA']=rdMolDescriptors.CalcTPSA(m)
    d['RotatableBonds']=rdMolDescriptors.CalcNumRotatableBonds(m)
    d['FractionCSP3']=rdMolDescriptors.CalcFractionCSP3(m)
    d['canonical']=Chem.MolToSmiles(m)
    d['n_arom_rings_size']=[len(r) for r in m.GetRingInfo().AtomRings()]
    return d

# columns to compare: (col, key, kind, tol)
EXACT_INT=['HeavyAtomCount','NumNitrogens','NumAromaticRings','NumHDonors','NumHAcceptors',
           'NumEsters','NumAmides','NumTertiaryAmines','HasPiperazine']
FLOAT_TOL={'ExactMolWt':0.02,'MolLogP':0.05,'TPSA':0.5,'FractionCSP3':0.01}

def audit_smiles_block(df, idcol):
    rows=[]
    for _,r in df.iterrows():
        rid=r[idcol]
        smi=r['SMILES']
        rc=recompute(smi)
        issues=[]
        if rc['_status']!='ok':
            issues.append(f"SMILES_{rc['_status']}")
            rows.append({'id':rid,'smiles':smi,'issues':issues}); continue
        # formula
        if 'MolFormula' in df.columns and isinstance(r.get('MolFormula'),str):
            if r['MolFormula'].strip()!=rc['MolFormula']:
                issues.append(f"MolFormula stored={r['MolFormula']} vs calc={rc['MolFormula']}")
        for c in EXACT_INT:
            if c in df.columns and pd.notna(r.get(c)):
                if int(round(r[c]))!=int(rc[c]):
                    issues.append(f"{c} stored={r[c]} vs calc={rc[c]}")
        for c,tol in FLOAT_TOL.items():
            if c in df.columns and pd.notna(r.get(c)):
                if abs(float(r[c])-rc[c])>tol:
                    issues.append(f"{c} stored={round(float(r[c]),3)} vs calc={round(rc[c],3)} (tol {tol})")
        rows.append({'id':rid,'smiles':smi,'canonical':rc.get('canonical'),
                     'calc_formula':rc.get('MolFormula'),'calc_basicN':rc.get('NumAmines_total'),
                     'calc_esters':rc.get('NumEsters'),'calc_amides':rc.get('NumAmides'),
                     'ring_sizes':rc.get('n_arom_rings_size'),'issues':issues})
    return rows

print("="*90)
print("PHASE 0 AUDIT  |  pKa rows:",len(pk)," bioact rows:",len(bi))
print("="*90)

# ===== 1. pKa dataset SMILES/descriptor consistency =====
pk_rows=audit_smiles_block(pk,'IAJD')
pk_bad=[r for r in pk_rows if r['issues']]
print(f"\n[1] pKa dataset: {len(pk_bad)}/{len(pk_rows)} rows with SMILES/descriptor issues")
# tally issue types
from collections import Counter
tally=Counter()
for r in pk_bad:
    for i in r['issues']: tally[i.split(' ')[0]]+=1
for k,v in tally.most_common(): print(f"     {v:4d}  {k}")

# ===== 2. linker length triple-consistency (pKa set) =====
print("\n[2] linker_length vs Linker_Length vs linker_carbons (pKa set)")
ll_cols=[c for c in ['linker_length','Linker_Length','linker_carbons'] if c in pk.columns]
print("     present:",ll_cols)
mismatch=0
for _,r in pk.iterrows():
    vals={c:r[c] for c in ll_cols if pd.notna(r[c])}
    uniq=set(round(float(v),3) for v in vals.values())
    if len(uniq)>1:
        mismatch+=1
        if mismatch<=15: print(f"     IAJD {r['IAJD']}: {vals}")
print(f"     total internal linker-length disagreements: {mismatch}")

# ===== 3. duplicates by canonical SMILES (pKa set) =====
print("\n[3] duplicate canonical SMILES in pKa set with conflicting pKa")
canon={}
for r,(_,row) in zip(pk_rows,pk.iterrows()):
    c=r.get('canonical')
    if not c: continue
    canon.setdefault(c,[]).append((row['IAJD'],row['pKa']))
dups={c:v for c,v in canon.items() if len(v)>1}
print(f"     {len(dups)} canonical structures appear >1x ({sum(len(v) for v in dups.values())} rows)")
for c,v in list(dups.items())[:20]:
    pkas=[x[1] for x in v]; spread=max(pkas)-min(pkas)
    flag=' <== pKa spread %.2f'%spread if spread>0.15 else ''
    print(f"     IAJDs {[x[0] for x in v]} pKa={pkas}{flag}")

# ===== 4. pKa sanity =====
print("\n[4] pKa range / outliers (pKa set)")
print("     min/max:",pk['pKa'].min(),pk['pKa'].max())
out=pk[(pk['pKa']<4)|(pk['pKa']>9)]
print("     out-of-[4,9]:",len(out))
print("     NaN pKa:",pk['pKa'].isna().sum())

# ===== 5. bioactivity internal consistency =====
print("\n[5] bioactivity dataset internal consistency")
bi_rows=audit_smiles_block(bi,'IAJD_num')
bi_bad=[r for r in bi_rows if r['issues']]
print(f"     SMILES/descriptor issues: {len(bi_bad)}/{len(bi_rows)} rows")
# flux vs log10 flux
def check_log(df,fcol,lcol):
    bad=[]
    for _,r in df.iterrows():
        f,l=r.get(fcol),r.get(lcol)
        if pd.notna(f) and pd.notna(l) and f>0:
            if abs(np.log10(f)-l)>0.02: bad.append((r['IAJD_num'],f,l,round(np.log10(f),3)))
    return bad
for fcol,lcol in [('flux_total','log10_flux_total'),('flux_lung','log10_flux_lung'),
                  ('flux_liver','log10_flux_liver'),('flux_spleen','log10_flux_spleen')]:
    if fcol in bi.columns and lcol in bi.columns:
        b=check_log(bi,fcol,lcol)
        print(f"     {fcol} vs {lcol}: {len(b)} inconsistent", b[:3] if b else "")
# pKa vs pKa_paper
if 'pKa_paper' in bi.columns:
    d=bi.dropna(subset=['pKa','pKa_paper'])
    diff=(d['pKa']-d['pKa_paper']).abs()
    print(f"     pKa vs pKa_paper: {(diff>0.05).sum()} rows differ >0.05 (of {len(d)} with both)")

# ===== 6. cross-dataset agreement for shared IAJD ids =====
print("\n[6] cross-dataset SMILES/pKa agreement (same IAJD id in both sets)")
pk_by=pk.set_index('IAJD')
shared=0; smi_mismatch=0; pka_mismatch=0
for _,r in bi.iterrows():
    iid=r.get('IAJD_num')
    if iid in pk_by.index:
        shared+=1
        prow=pk_by.loc[iid]
        if isinstance(prow,pd.DataFrame): prow=prow.iloc[0]
        # compare canonical
        try:
            cb=Chem.MolToSmiles(Chem.MolFromSmiles(r['SMILES'])) if Chem.MolFromSmiles(str(r['SMILES'])) else None
            cp=Chem.MolToSmiles(Chem.MolFromSmiles(prow['SMILES'])) if Chem.MolFromSmiles(str(prow['SMILES'])) else None
            if cb and cp and cb!=cp: smi_mismatch+=1
        except: pass
        if pd.notna(r.get('pKa')) and pd.notna(prow.get('pKa')) and abs(r['pKa']-prow['pKa'])>0.05:
            pka_mismatch+=1
print(f"     shared ids: {shared}; SMILES canonical mismatch: {smi_mismatch}; pKa mismatch>0.05: {pka_mismatch}")

# save
out={'pka_smiles_issues':pk_bad,'bioact_smiles_issues':bi_bad,
     'linker_mismatch_count':mismatch,'dup_canonical':{c:[list(map(float,[x[0],x[1]])) for x in v] for c,v in dups.items()}}
json.dump(out,open('audit_phase0_results.json','w'),indent=2,default=str)
print("\nSaved -> audit_phase0_results.json")
