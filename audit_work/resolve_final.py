#!/usr/bin/env python3
"""Build candidate structures for unresolved compounds, verify FULL 10118 descriptor vector
(MolWt,FractionCSP3,HBA,HBD,RotB,Aromatic) + (where text-paper) SI formula. Output validated SMILES."""
import warnings,json,sys; warnings.filterwarnings('ignore'); sys.path.insert(0,'audit_work')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, rdMolDescriptors
from fm2 import si_neutrals, parse, norm
P={int(k):v for k,v in json.load(open('audit_work/p10118_tableS1.json')).items()}
SI={}
for tf in ['ja1c05813_si.txt','ja1c05813_si_plain.txt','ja1c05813_si_mu.txt']:
    for k,v in si_neutrals(open('audit_work/paper_text/'+tf,errors='ignore').read()).items(): SI.setdefault(k,[]).extend(v)
def fullvec(s):
    m=Chem.MolFromSmiles(s)
    if not m: return None
    return dict(MolWt=round(Descriptors.MolWt(m),2),FractionCSP3=rdMolDescriptors.CalcFractionCSP3(m),
        HBA=rdMolDescriptors.CalcNumHBA(m),HBD=rdMolDescriptors.CalcNumHBD(m),
        RotB=rdMolDescriptors.CalcNumRotatableBonds(m),Aromatic=rdMolDescriptors.CalcNumAromaticRings(m),
        formula=rdMolDescriptors.CalcMolFormula(m),smiles=Chem.MolToSmiles(m))
def matches10118(iid, v, rb_tol=2):
    p=P[iid]
    return (abs(v['MolWt']-p['MolWt'])<1.2 and abs(v['FractionCSP3']-p['FractionCSP3'])<0.008
            and v['HBA']==int(p['HBA']) and v['HBD']==int(p['HBD']) and v['Aromatic']==int(p['Aromatic'])
            and abs(v['RotB']-int(p['RotB']))<=rb_tol)

HEADS={'MPRZ':'N1CCN(C)CC1','HPRZ':'N1CCN(CCO)CC1','diEGPRZ':'N1CCN(CCOCCO)CC1','PIP':'N1CCCCC1',
       'MeOEtPRZ':'N1CCN(CCOC)CC1'}
def petris(n,head,link): return f'C(CO{"C"*n})(CO{"C"*n})(CO{"C"*n})COC(=O){link}{HEADS[head]}'
PT={273:(12,'MeOEtPRZ','CCCC'),287:(8,'MeOEtPRZ','CCCC'),291:(7,'MeOEtPRZ','CCCC')}
print('=== PE-Tris 273/287/291 (10118-resolved) full-vector check ===')
out={}
for iid,(n,h,l) in PT.items():
    v=fullvec(petris(n,h,l)); ok=matches10118(iid,v)
    print(f'  IAJD {iid}: {v["formula"]} MW{v["MolWt"]} F{v["FractionCSP3"]:.3f} HBA{v["HBA"]} HBD{v["HBD"]} RotB{v["RotB"]} Ar{v["Aromatic"]}  10118match={ok}')
    if ok: out[iid]=v['smiles']

# PE-Gallic single specials: chain + 3 gallic arms (caps/ionizable), amide-linked
def pega(chain, arms):  # chain=alkyl SMILES (no leading O); arms=3 fragment list
    return f'{chain}c1cc(CNC(=O)c2cc({arms[0]})c({arms[1]})c({arms[2]})c2)cc({chain})c1'
ARM={'DMBA':'OCCOCCOCCOC(=O)CCCN(C)C','MPRZ':'OCCOCCOCCOC(=O)CCCN3CCN(C)CC3','PIP':'OCCOCCOCCOC(=O)CCCN3CCCCC3',
     'OMe':'OCCOCCOCCOC','OBn':'OCCOCCOCCOCc3ccccc3'}
import itertools
print('\n=== PE-Gallic specials 24, 33 (match 10118 full vector + SI formula) ===')
CHAINS={'C8':'OCCCCCCCC','C9':'OCCCCCCCCC','C10':'OCCCCCCCCCC','C11':'OCCCCCCCCCCC','C12':'OCCCCCCCCCCCC',
        '2EH':'OCC(CC)CCCC','dm8':'OCCC(C)CCCC(C)C'}
for iid,heads,chains in [(24,['DMBA','OBn'],['dm8','2EH','C10']),(33,['MPRZ','OMe','OBn'],['C11','C12'])]:
    found=[]
    for ch in chains:
        for combo in itertools.product(heads if iid==24 else ['MPRZ','OMe','OBn'],repeat=3):
            arms=[ARM[c] for c in combo]
            v=fullvec(pega(CHAINS[ch],arms))
            if v and matches10118(iid,v) and norm(parse(v['formula'])) in SI:
                found.append((ch,combo,v['formula']));
    uniq=set((c[2]) for c in found)
    print(f'  IAJD {iid}: {len(found)} candidate(s) match 10118+SI -> {found[:3]}')
    if len(found)>=1 and len(set(f[1] for f in found))==1:
        ch,combo,_=found[0]; out[iid]=fullvec(pega(CHAINS[ch],[ARM[c] for c in combo]))['smiles']
json.dump(out, open('audit_work/resolve_final.json','w'))
print('\nVALIDATED & saved:', sorted(out))
