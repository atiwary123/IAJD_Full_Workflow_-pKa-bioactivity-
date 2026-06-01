#!/usr/bin/env python3
"""Resolve PE-Tris (ja3c07337) structures by matching the 10118 descriptor vector (MolWt,HBA,HBD).
Scaffold: pentaerythritol(3 n-alkyl ethers) + butanoate ester + piperazine/piperidine head."""
import warnings,json; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, rdMolDescriptors
P={int(k):v for k,v in json.load(open('audit_work/p10118_tableS1.json')).items()}
HEADS={'MPRZ':'N1CCN(C)CC1','HPRZ':'N1CCN(CCO)CC1','diEGPRZ':'N1CCN(CCOCCO)CC1',
       'PIP':'N1CCCCC1','PRZ':'N1CCNCC1','HPIP':'N1CCC(O)CC1'}
def petris(n, head):
    ch='O'+'C'*n
    return f'C(C{ch})(C{ch})(C{ch})COC(=O)CCC{HEADS[head]}'
def vec(smi):
    m=Chem.MolFromSmiles(smi)
    return (round(Descriptors.MolWt(m),2), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m))
targets=[248,250,251,273,274,275,276,277,278,279,280,282,283,284,285,287,288,289,290,291,292,299]
print(f"{'IAJD':>5} {'10118 MolWt/HBA/HBD':>22} -> best match")
import itertools
res={}
for iid in targets:
    if iid not in P: continue
    tw,ta,td=P[iid]['MolWt'],int(P[iid]['HBA']),int(P[iid]['HBD'])
    best=None
    for n,head in itertools.product(range(5,15),HEADS):
        mw,ha,hd=vec(petris(n,head))
        if abs(mw-tw)<1.2 and ha==ta and hd==td:
            best=(n,head,mw); break
    res[iid]=best
    print(f'{iid:>5}   {tw:>8.2f}/{ta}/{td}    -> {best}')
json.dump({k:(v[0],v[1]) if v else None for k,v in res.items()}, open('audit_work/petris_resolved.json','w'))
