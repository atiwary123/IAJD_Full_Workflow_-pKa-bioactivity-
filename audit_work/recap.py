#!/usr/bin/env python3
"""
Formula-validated cap-fixer for Percec IAJD dendrimers.
Reduce every non-ionizable arm cap (free-OH / methyl-ether / benzyl-ether /
aroyl-ester) to a bare hydroxyl 'socket', then solve which uniform cap
{OH, OMe, OBn, OPMB} makes the formula match the paper SI. Returns corrected
SMILES + the matched SI formula, or a diagnosis if no cap assignment matches.
"""
import re
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

def _branch_atoms(mol, start_idx, blocked_idx):
    """atoms reachable from start without crossing blocked_idx"""
    seen=set(); stack=[start_idx]
    while stack:
        i=stack.pop()
        if i in seen or i==blocked_idx: continue
        seen.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx()!=blocked_idx: stack.append(nb.GetIdx())
    return seen

def _has_basic_N(mol, atomset):
    for i in atomset:
        a=mol.GetAtomWithIdx(i)
        if a.GetSymbol()=='N' and not a.GetIsAromatic():
            # exclude amide N
            amide=any(nb.GetSymbol()=='C' and any(b.GetBondTypeAsDouble()==2 and b.GetOtherAtom(nb).GetSymbol()=='O' for b in nb.GetBonds()) for nb in a.GetNeighbors())
            if not amide: return True
    return False

def find_caps(mol):
    """return list of (socket_O_idx, [atoms_to_delete]) for each non-ionizable cap."""
    caps=[]
    for O in mol.GetAtoms():
        if O.GetSymbol()!='O': continue
        nbrs=O.GetNeighbors()
        # free hydroxyl on sp3 C -> socket, nothing to delete
        if O.GetDegree()==1 and O.GetTotalNumHs()==1:
            c=nbrs[0]
            if c.GetSymbol()=='C' and not c.GetIsAromatic():
                caps.append((O.GetIdx(), [])); continue
        if O.GetDegree()!=2: continue
        a,b=nbrs[0],nbrs[1]
        # one side must be sp3 alkyl (the chain), other side the cap
        for chain,cap in ((a,b),(b,a)):
            if chain.GetIsAromatic(): continue
            if chain.GetSymbol()!='C': continue
            # cap = methyl (CH3 terminal)
            if cap.GetSymbol()=='C' and cap.GetDegree()==1 and cap.GetTotalNumHs()==3:
                caps.append((O.GetIdx(), [cap.GetIdx()])); break
            # cap = benzylic CH2 attached to aromatic (benzyl ether)
            if cap.GetSymbol()=='C' and cap.GetTotalNumHs()==2 and any(x.GetIsAromatic() for x in cap.GetNeighbors()):
                br=_branch_atoms(mol, cap.GetIdx(), O.GetIdx())
                if not _has_basic_N(mol, br): caps.append((O.GetIdx(), list(br))); break
            # cap = acyl carbon (ester) C(=O)-R  -> aroyl/phenylacetate cap if no basic N in branch
            if cap.GetSymbol()=='C' and any(bd.GetBondTypeAsDouble()==2 and bd.GetOtherAtom(cap).GetSymbol()=='O' for bd in cap.GetBonds()):
                br=_branch_atoms(mol, cap.GetIdx(), O.GetIdx())
                if not _has_basic_N(mol, br): caps.append((O.GetIdx(), list(br))); break
    return caps

def reduce_to_sockets(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None,0
    caps=find_caps(m)
    rw=RWMol(m)
    todel=set()
    for _,branch in caps: todel.update(branch)
    for idx in sorted(todel, reverse=True): rw.RemoveAtom(idx)
    try:
        m2=rw.GetMol(); Chem.SanitizeMol(m2)
        return Chem.MolToSmiles(m2), len(caps)
    except Exception as e:
        return None, len(caps)

# cap deltas relative to OH socket (replace the H on socket-O)
CAP_DELTA={'OH':{}, 'OMe':{'C':1,'H':2}, 'OBn':{'C':7,'H':6}, 'OPMB':{'C':8,'H':8,'O':1}}
CAP_FRAG ={'OMe':'C', 'OBn':'Cc1ccccc1', 'OPMB':'Cc1ccc(OC)cc1'}

def solve_caps(base_formula, n_sockets, target_formula):
    """find counts (a=OMe,b=OBn,c=OPMB, rest OH) so base+caps==target. return dict or None"""
    F0=parse(base_formula); T=parse(target_formula)
    # non-CHO elements must already match
    for el in set(F0)|set(T):
        if el not in ('C','H','O') and F0.get(el,0)!=T.get(el,0): return None
    dC=T.get('C',0)-F0.get('C',0); dH=T.get('H',0)-F0.get('H',0); dO=T.get('O',0)-F0.get('O',0)
    c=dO  # only OPMB adds O
    if c<0 or c>n_sockets: return None
    # remaining: a*1+b*7 = dC-8c ; a*2+b*6 = dH-8c
    rc=dC-8*c; rh=dH-8*c
    # solve: from a+? ; a=rc-7b ... use: a*2+b*6=rh and a+7b...
    # a = rc - 7b ; plug: 2(rc-7b)+6b = rh -> 2rc -14b +6b = rh -> -8b = rh-2rc -> b=(2rc-rh)/8
    if (2*rc-rh)%8!=0: return None
    b=(2*rc-rh)//8; a=rc-7*b
    if a<0 or b<0 or a+b+c>n_sockets: return None
    return {'OMe':a,'OBn':b,'OPMB':c,'OH':n_sockets-a-b-c}

def apply_uniform_cap(smi, captype):
    """add the same cap to every socket (free-OH/socket O). Returns SMILES."""
    m=Chem.MolFromSmiles(smi);
    # after reduce_to_sockets, all caps are free OH; methylate/benzylate each
    rw=RWMol(m); frag=CAP_FRAG[captype]
    sockets=[a.GetIdx() for a in m.GetAtoms() if a.GetSymbol()=='O' and a.GetDegree()==1 and a.GetTotalNumHs()==1
             and a.GetNeighbors()[0].GetSymbol()=='C' and not a.GetNeighbors()[0].GetIsAromatic()]
    for oidx in sockets:
        sub=Chem.MolFromSmiles(frag)
        amap={}
        for at in sub.GetAtoms():
            amap[at.GetIdx()]=rw.AddAtom(Chem.Atom(at.GetAtomicNum()))
        for bd in sub.GetBonds():
            rw.AddBond(amap[bd.GetBeginAtomIdx()],amap[bd.GetEndAtomIdx()],bd.GetBondType())
        # set aromatic flags
        for at in sub.GetAtoms():
            rw.GetAtomWithIdx(amap[at.GetIdx()]).SetIsAromatic(at.GetIsAromatic())
        rw.AddBond(oidx, amap[0], Chem.BondType.SINGLE)
    try:
        m2=rw.GetMol(); Chem.SanitizeMol(m2); return Chem.MolToSmiles(m2)
    except Exception as e:
        return None
