#!/usr/bin/env python3
"""Enumerate family-appropriate building-block combinations for the genuinely-mismatched
flagged IAJDs (64, 33, 30) and keep only candidates matching the FULL 10118 descriptor vector
(MolWt, FractionCSP3, HBA, HBD, RotB, Aromatic). Same tolerances as resolve_final.py.

This is structure search constrained by an independent 6-number fingerprint — not free guessing.
Reports every passing candidate so non-uniqueness is visible (honest about ambiguity)."""
import warnings, json, itertools; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

P = {int(k): v for k, v in json.load(open('audit_work/p10118_tableS1.json')).items()}

def vec(smi):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    return dict(MolWt=round(Descriptors.MolWt(m),3),
                FractionCSP3=rdMolDescriptors.CalcFractionCSP3(m),
                HBA=rdMolDescriptors.CalcNumHBA(m), HBD=rdMolDescriptors.CalcNumHBD(m),
                RotB=rdMolDescriptors.CalcNumRotatableBonds(m),
                Aromatic=rdMolDescriptors.CalcNumAromaticRings(m),
                formula=rdMolDescriptors.CalcMolFormula(m), smiles=Chem.MolToSmiles(m))

def match(iid, v, rb_tol=2):
    p = P[iid]
    return (v and abs(v['MolWt']-p['MolWt']) < 1.2 and abs(v['FractionCSP3']-p['FractionCSP3']) < 0.008
            and v['HBA']==int(p['HBA']) and v['HBD']==int(p['HBD']) and v['Aromatic']==int(p['Aromatic'])
            and abs(v['RotB']-int(p['RotB'])) <= rb_tol)

# ---- alkyl chains ----
def alk(n): return 'C'*n
BRANCH = {'EH':'CC(CC)CCCC', 'dm8':'CCC(C)CCCC(C)C', '2EH':'CC(CC)CCCC'}
CHAINS = {f'C{n}': alk(n) for n in range(5,19)}
CHAINS.update(BRANCH)
# aryl ether connectivity:  front position = chain-O-ring (chain then O);
#                           internal substituent = ring-O-chain (O then chain)
def frontO(c): return CHAINS[c] + 'O'      # e.g. C12 -> 'CCCCCCCCCCCCO'  used as {frontO}c1...
def alkO(c):   return 'O' + CHAINS[c]       # e.g. C12 -> 'OCCCCCCCCCCCC'  used as c(...{alkO})

# ---- piperazine / amine heads ----
HEADS = {'MPRZ':'N1CCN(C)CC1','HPRZ':'N1CCN(CCO)CC1','PIP':'N1CCCCC1','PRZ':'N1CCNCC1',
         'DMA':'N(C)C','MeOEtPRZ':'N1CCN(CCOC)CC1','H2EPRZ':'N1CCN(CCOCCO)CC1'}

print('#'*70); print('# IAJD 64  target:', P[64]); print('#'*70)
# Family sSS / dialkoxybenzyl variants. Try: dialkoxybenzyl ESTER, monoalkoxy/monoalkyl mixed,
# benzoate (NoneC), amide; chains asym; linker 2/3/4 CH2; various heads.
def dialkoxybenzyl_ester(o1,o2,k,head):  # 3,5-dialkoxybenzyl ester of (CH2)k-head acid
    return f'{frontO(o1)}c1cc({alkO(o2)})cc(COC(=O){"C"*k}{HEADS[head]})c1'
def alkyl_alkoxy_benzyl(oalk, calk, k, head):  # one alkOXY + one alkYL (C-C) chain on ring
    return f'{frontO(oalk)}c1cc({CHAINS[calk]})cc(COC(=O){"C"*k}{HEADS[head]})c1'
def benzoate(o1,o2,k,head):  # 3,5-dialkoxybenzoate, ester to (CH2)k-head  (NoneC family)
    return f'{frontO(o1)}c1cc({alkO(o2)})cc(C(=O)O{"C"*k}{HEADS[head]})c1'

cands64=[]
chain_keys=list(CHAINS)
for o1,o2 in itertools.combinations_with_replacement(chain_keys,2):
    for k in (2,3,4):
        for head in HEADS:
            for fn,tag in [(dialkoxybenzyl_ester,'dialkoxybenzyl-ester'),
                           (benzoate,'dialkoxybenzoate-NoneC'),
                           (alkyl_alkoxy_benzyl,'alkyl+alkoxybenzyl-ester')]:
                try: s=fn(o1,o2,k,head)
                except Exception: continue
                v=vec(s)
                if match(64,v): cands64.append((tag,o1,o2,k,head,v['formula'],v['RotB'],v['smiles']))
seen=set()
for c in cands64:
    key=c[-1]
    if key in seen: continue
    seen.add(key)
    print(f"  [{c[0]:26}] {c[1]}/{c[2]} link{c[3]}C {c[4]:6} {c[5]:12} RotB{c[6]}  {c[7]}")
print(f"  total unique 64 candidates: {len(seen)}")

print('\n'+'#'*70); print('# IAJD 33  target:', P[33]); print('#'*70)
# PE-Gallic twin: 3,5-dialkoxybenzyl-amide -> gallic triester; 3 arms from {MPRZ,OBn,OMe,OH,DMBA}
ARM={'MPRZ':'OCCOCCOCCOC(=O)CCCN3CCN(C)CC3','DMBA':'OCCOCCOCCOC(=O)CCCN(C)C',
     'OBn':'OCCOCCOCCOCc3ccccc3','OMe':'OCCOCCOCCOC','OH':'OCCOCCOCCO','PIP':'OCCOCCOCCOC(=O)CCCN3CCCCC3'}
def gallic_amide(chain, arms): return f'{frontO(chain)}c1cc(CNC(=O)c2cc({ARM[arms[0]]})c({ARM[arms[1]]})c({ARM[arms[2]]})c2)cc({alkO(chain)})c1'
def gallic_ester(chain, arms): return f'{frontO(chain)}c1cc(COC(=O)c2cc({ARM[arms[0]]})c({ARM[arms[1]]})c({ARM[arms[2]]})c2)cc({alkO(chain)})c1'
cands33=[]
for chain in [f'C{n}' for n in range(9,14)]:
    for arms in itertools.product(ARM,repeat=3):
        for fn,tag in [(gallic_amide,'gallic-amide'),(gallic_ester,'gallic-ester')]:
            s=fn(chain,arms); v=vec(s)
            if match(33,v): cands33.append((tag,chain,arms,v['formula'],v['RotB'],v['smiles']))
seen=set()
for c in cands33:
    if c[-1] in seen: continue
    seen.add(c[-1])
    print(f"  [{c[0]:12}] {c[1]} arms={c[2]} {c[3]:12} RotB{c[4]}  {c[5]}")
print(f"  total unique 33 candidates: {len(seen)}")

print('\n'+'#'*70); print('# IAJD 30  target:', P[30], ' (Aromatic=0 -> fully aliphatic!)'); print('#'*70)
# 2-arm aliphatic Janus: bis-alkyl core (malonate/glycerol/pentaerythritol) + 2 DMBA-glycol arms.
# HBD0, Aromatic0, HBA11. Enumerate simple aliphatic 2-arm dendrons.
DMBAg='OCCOCCOCCOC(=O)CCCN(C)C'  # triEG-DMBA arm (HBA-rich, no OH)
def janus2(core):
    return core
# template A: 2,2-bis(alkyl)-propane-1,3-diyl bearing 2 DMBA-glycol arms via ester on a malonate
templates=[]
for n1 in range(8,16):
  for n2 in range(8,16):
    # glyceryl: CH(O-arm)(CH2-O-arm)(CH2-O-Cn) variants are many; try a symmetric bis-arm + dialkyl
    s=f'C(C{alk(n1)})(C{alk(n2)})(CO{DMBAg})CO{DMBAg}'   # neopentyl-like core, 2 alkyl + 2 DMBA arms (ether)
    templates.append(('neopentyl-2alkyl-2DMBAether',n1,n2,s))
cands30=[]
for tag,n1,n2,s in templates:
    v=vec(s)
    if match(30,v): cands30.append((tag,n1,n2,v['formula'],v['RotB'],v['smiles']))
for c in cands30[:20]:
    print(f"  [{c[0]}] C{c[1]}/C{c[2]} {c[3]} RotB{c[4]}  {c[5]}")
print(f"  total 30 candidates (this template): {len(cands30)}")
print("  NOTE: if 0, 30 needs the SI figure (aliphatic 2-arm scaffold underdetermined by vector alone).")
