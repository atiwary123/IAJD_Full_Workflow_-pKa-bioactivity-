#!/usr/bin/env python3
"""
Reusable SMILES-corroboration engine.
Parse every MALDI/HRMS "calculated for <formula>" in a paper text, normalize each
to a NEUTRAL molecular formula (accounting for [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+,
[M+2H]2+, [M-H]-), and test whether each dataset row's formula (which Phase 0 proved
== its SMILES) is corroborated by the paper.

Usage: formula_match.py <paper_text.txt> <source_label>
"""
import re, sys, warnings
import pandas as pd
warnings.filterwarnings('ignore')

ELEM = re.compile(r'([A-Z][a-z]?)(\d*)')
def parse_formula(f):
    d={}
    for el,n in ELEM.findall(f):
        if not el: continue
        d[el]=d.get(el,0)+(int(n) if n else 1)
    return d
def norm(d):  # canonical string, Hill-ish
    return ''.join(f'{e}{d[e]}' for e in sorted(d) if d.get(e,0)>0)
def sub(d,el,k=1):
    d=dict(d); d[el]=d.get(el,0)-k
    if d[el]<=0: d.pop(el,None)
    return d

# adduct patterns: capture the formula token right after "calculated for"
ADDUCT_RX = re.compile(
    r'\[M\s*([+\-])\s*(\d?)\s*(H|Na|K|NH4|HCOO|Cl|2H)?\]\s*(\d?)\s*([+\-])?\s*'
    r'.{0,40}?calculated for\s+([A-Za-z0-9]+)', re.I)
# fallback: "calculated for <F>:" possibly preceded by adduct earlier on the line
SIMPLE_RX = re.compile(r'calculated for\s+([A-Za-z0-9]+)\s*[:= ]', re.I)

def neutral_from(adduct_sign, mult, addspec, ion_formula):
    d=parse_formula(ion_formula)
    m=int(mult) if mult else 1
    if addspec is None or addspec=='':
        return d
    addspec=addspec.replace('2H','HH')
    if adduct_sign=='+':
        # ion = M + adduct  -> M = ion - adduct
        if addspec in ('H','HH'):
            return sub(d,'H', 2 if addspec=='HH' else m)
        if addspec=='Na': return sub(d,'Na',m)
        if addspec=='K':  return sub(d,'K',m)
        if addspec=='NH4':
            return sub(sub(d,'N',1),'H',4)
    else:  # [M - H]-
        if addspec=='H':
            dd=dict(d); dd['H']=dd.get('H',0)+1; return dd
        if addspec=='Cl': return sub(d,'Cl',1)
    return d

def extract_si_neutrals(txt):
    """return set of neutral-formula strings and a list of (line, raw, neutral)."""
    neutrals=set(); rows=[]
    for ln in txt.splitlines():
        for m in re.finditer(r'(\[M[^\]]*\][\s\d+\-]*)?[^.]*?calculated for\s+([A-Za-z0-9]+)', ln):
            adduct_blob=m.group(1) or ''
            ionf=m.group(2)
            # determine adduct
            am=re.search(r'\[M\s*([+\-])\s*(\d?)\s*(2H|H|Na|K|NH4|Cl)?\]', adduct_blob)
            if am:
                neu=neutral_from(am.group(1), am.group(2), am.group(3), ionf)
            else:
                neu=parse_formula(ionf)
            # ignore tiny fragments
            if neu.get('C',0)>=20:
                s=norm(neu); neutrals.add(s); rows.append((ionf,s))
    return neutrals, rows

if __name__=='__main__':
    txtfile, source = sys.argv[1], sys.argv[2]
    txt=open(txtfile,encoding='utf-8',errors='ignore').read()
    si_neu, si_rows = extract_si_neutrals(txt)
    print(f"SI neutral formulas (C>=20) extracted: {len(si_neu)} unique")
    pk=pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx')
    g=pk[pk['source'].astype(str).str.contains(source, na=False)]
    print(f"dataset rows for source~='{source}': {len(g)}")
    miss=[]; ok=[]
    for _,r in g.iterrows():
        f=r['MolFormula']
        if not isinstance(f,str):
            miss.append((r['IAJD'],'<no formula stored>')); continue
        s=norm(parse_formula(f))
        if s in si_neu: ok.append(r['IAJD'])
        else: miss.append((r['IAJD'],f))
    print(f"\nCORROBORATED (formula found in SI): {len(ok)}/{len(g)}")
    print("ids ok:", ok)
    print(f"\nNOT FOUND in SI ({len(miss)}):")
    for iid,f in miss: print(f"   IAJD {iid}: {f}")
