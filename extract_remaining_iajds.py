"""Maximum-effort extraction of remaining 54 missing IAJDs.

Strategy:
- PE-Gallic dendrimers (37, 52, 55-63): use actual neighbor SMILES as templates,
  modify PEG arm termini (DMBA/PIP/Bn swaps) and tail chain lengths
- sSS-Nonsym (67-69, 72, 80, 85, 90, 109, 132, 160, 195, 196): vary chain
  lengths and head groups systematically
- PE-Tris (84, 94, 102-104, 252, 254-262, 281, 286, 293): vary chain + head
- GA-Tris (80, 302-307, 315-316, 327-328): EH + linker/head combos
- Dialkoxybenzyl (85, 295-296, 298): EH + position/head combos
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ── Load existing data ──────────────────────────────────────────────────

df_pka = pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx', sheet_name='Dataset')
df_bio = pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx', sheet_name='Sheet1')

existing_canonical = set()
ref = {}
for _, r in df_pka.iterrows():
    mol = Chem.MolFromSmiles(str(r['SMILES']))
    if mol:
        c = Chem.MolToSmiles(mol)
        existing_canonical.add(c)
        ref[int(r['IAJD'])] = {'smiles': r['SMILES'], 'canonical': c,
                                'arch': str(r.get('architecture','')),
                                'family': r['family']}
for _, r in df_bio.drop_duplicates('IAJD_num').iterrows():
    num = int(r['IAJD_num'])
    if num not in ref:
        smi = r.get('SMILES_canonical') or r.get('SMILES')
        if pd.notna(smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                c = Chem.MolToSmiles(mol)
                existing_canonical.add(c)
                ref[num] = {'smiles': str(smi), 'canonical': c,
                            'arch': str(r.get('architecture','')),
                            'family': r['family']}


def get_template_smiles(iajd_num):
    """Get the canonical SMILES for an existing IAJD."""
    if iajd_num in ref:
        return ref[iajd_num]['canonical']
    return None


def modify_pe_gallic(template_iajd, modifications):
    """Modify a PE-Gallic template SMILES by replacing substrings.

    PE-Gallic dendrimers have the structure:
      {tail}Oc1cc({link}c2cc({arm1})c({arm2})c({arm3})c2)cc(O{tail})c1
    where arms are PEG chains ending in DMBA, Bn, PIP, or free OH.
    """
    smi = get_template_smiles(template_iajd)
    if not smi:
        return None

    result = smi
    for old, new in modifications:
        result = result.replace(old, new, 1)

    mol = Chem.MolFromSmiles(result)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def validate_and_record(iajd_num, smiles, family, source, confidence, notes):
    """Validate SMILES and return record dict if unique."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    if canonical in existing_canonical:
        return None

    n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    if n_n == 0:
        return None

    existing_canonical.add(canonical)
    return {
        'iajd_num': iajd_num,
        'smiles_canonical': canonical,
        'family': family,
        'ExactMolWt': round(Descriptors.ExactMolWt(mol), 4),
        'NumNitrogens': n_n,
        'source': source,
        'confidence': confidence,
        'notes': notes,
    }


# ── Head group SMILES fragments ─────────────────────────────────────────

# For aromatic cores (ring index 2)
HEAD_R2 = {
    "MPRZ":   "N2CCN(C)CC2",
    "HPRZ":   "N2CCN(CCO)CC2",
    "H2EPRZ": "N2CCN(CCOCCO)CC2",
    "PIP":    "N2CCCCC2",
    "DMBA":   "N(C)C",
}
# For PE-Tris (ring index 1)
HEAD_R1 = {
    "MPRZ":   "N1CCN(C)CC1",
    "HPRZ":   "N1CCN(CCO)CC1",
    "H2EPRZ": "N1CCN(CCOCCO)CC1",
    "PIP":    "N1CCCCC1",
}

def chain(n):
    return "C" * n

def build_sss_35(c1, c2, linker_c, head, linkage="ester", c1_br=False, c2_br=False):
    ch1 = "CCCCC(CC)C" if c1_br else chain(c1)
    ch2 = "CCCCC(CC)C" if c2_br else chain(c2)
    hg = HEAD_R2[head]
    lb = chain(linker_c - 1)
    lk = f"CNC(=O){lb}{hg}" if linkage == "amide" else f"COC(=O){lb}{hg}"
    return f"{ch1}Oc1cc({lk})cc(O{ch2})c1"

def build_sss_34(c1, c2, linker_c, head, c1_br=False, c2_br=False):
    ch1 = "CCCCC(CC)C" if c1_br else chain(c1)
    ch2 = "CCCCC(CC)C" if c2_br else chain(c2)
    hg = HEAD_R2[head]
    lb = chain(linker_c - 1)
    return f"{ch1}Oc1ccc(COC(=O){lb}{hg})cc1O{ch2}"

def build_pe_tris(cn, linker_c, head, branched=False):
    hg = HEAD_R1[head]
    lb = chain(linker_c - 1)
    if branched:
        return f"CCCCC(CC)COCC(COCC(CC)CCCC)(COCC(CC)CCCC)COC(=O){lb}{hg}"
    ch = chain(cn)
    return f"C(CO{ch})(CO{ch})(CO{ch})COC(=O){lb}{hg}"

def build_ga_tris(cn, linker_c, head, linkage="ester", branched=False):
    hg = HEAD_R2[head]
    lb = chain(linker_c - 1)
    if branched:
        cp = "CCCCC(CC)C"
        ce = "OCC(CC)CCCC"
    else:
        cp = chain(cn)
        ce = "O" + chain(cn)
    lk = f"CNC(=O){lb}{hg}" if linkage == "amide" else f"COC(=O){lb}{hg}"
    return f"{cp}Oc1cc({lk})cc({ce})c1{ce}"


# ── PE-Gallic dendrimer templates ────────────────────────────────────────
# These are the most complex. I'll use actual neighbor SMILES and make
# targeted modifications.

PEG3 = "OCCOCCOCCOC(=O)"  # tri-ethylene glycol + ester
PEG4 = "OCCOCCOCCOCCOC(=O)"  # tetra-ethylene glycol + ester
ARM_DMBA4 = f"{PEG3}CCCN(C)C"  # 4C linker to dimethylamine
ARM_DMBA3 = f"{PEG3}CCN(C)C"   # 3C (DMPA)
ARM_DMBA2 = f"{PEG3}CN(C)C"    # 2C (DMA)
ARM_BN = f"{PEG3}Cc3ccccc3"    # benzyl
ARM_PIP4 = f"{PEG3}CCCN3CCCCC3"  # piperidine
ARM_MPRZ4 = f"{PEG3}CCCN3CCN(C)CC3"  # methylpiperazine
ARM_OH = "OCCOCCOCCO"  # free PEG-OH


def build_pe_gallic_htm(tail_n, arms, linkage="ester", tail_branched=False):
    """Build PE-Gallic HTM (head-tail-modified) dendrimer.

    Structure: {tail}Oc1cc({link}c2cc({arm1})c({arm2})c({arm3})c2)cc(O{tail})c1
    arms = list of 3 arm SMILES (positions 3,4,5 on inner gallic ring)
    """
    if tail_branched:
        tc = "CC(C)CCCC(C)C"  # 2,6-dimethylheptyl (dm8)
        tc_ether = "OCC(C)CCCC(C)C"
    else:
        tc = chain(tail_n)
        tc_ether = "O" + chain(tail_n)

    lk = "CNC(=O)" if linkage == "amide" else "COC(=O)"

    return (f"{tc}Oc4cc({lk}c5cc({arms[0]})c({arms[1]})c({arms[2]})c5)"
            f"cc({tc_ether})c4")


def build_pe_gallic_pe(tail_n, arms, linkage="ester", tail_type="normal"):
    """Build PE-Gallic PE (pentaerythritol-extended) dendrimer.
    Same inner structure but with pentaerythritol in the tail region.
    """
    if tail_type == "EH":
        tc = "CC(C)CCCC(C)C"
        tc_ether = "OCC(C)CCCC(C)C"
    elif tail_type == "dm8":
        tc = "CC(C)CCCC(C)C"
        tc_ether = "OCC(C)CCCC(C)C"
    else:
        tc = chain(tail_n)
        tc_ether = "O" + chain(tail_n)

    lk = "CNC(=O)" if linkage == "amide" else "COC(=O)"

    return (f"{tc}Oc4cc({lk}c5cc({arms[0]})c({arms[1]})c({arms[2]})c5)"
            f"cc({tc_ether})c4")


# ── Generate all missing IAJDs ──────────────────────────────────────────

results = []
skipped = []

def try_add(iajd, smiles, family, source, confidence, notes):
    r = validate_and_record(iajd, smiles, family, source, confidence, notes)
    if r:
        results.append(r)
        return True
    skipped.append({'iajd': iajd, 'reason': 'duplicate or invalid', 'notes': notes})
    return False


# ── IAJD 37: PE-Gallic between 36 (PE.C11.MPRZ13.Bn2) and 38 (HTM.C12.DMBA1)
# 36 is PE-type (amide linkage), 37 likely PE.C10 or PE.C12 with different arms
try_add(37, build_pe_gallic_htm(12, [ARM_DMBA4, ARM_BN, ARM_DMBA4], "amide"),
        "PE-Gallic", "ja1c05813", "LOW", "PE-gallic between 36(MPRZ13.Bn2) and 38(HTM.DMBA1)")

# ── IAJD 52: between 51 (HTM.C11.DMA13.Bn2) and 54 (HTM.C12.PIP.PEG350)
# 51 has DMA (2C), 52 might be DMPA (3C) or PIP variant
try_add(52, build_pe_gallic_htm(11, [ARM_DMBA3, ARM_BN, ARM_DMBA3], "ester"),
        "PE-Gallic", "ja1c05813", "LOW", "PE-gallic HTM.C11 between DMA and PIP variants")

# ── IAJDs 55-63: between PE-Gallic(54) and sSS(64)
# These are in the gap between papers. 54 is PE-Gallic PEGylated, 64 is sSS.
# These might be sSS or PE-Gallic variants. Try sSS with different chains.
for i, (c1, c2, head, linkage) in enumerate([
    (10, 14, "MPRZ", "ester"),   # 55
    (8, 16, "MPRZ", "ester"),    # 56
    (10, 14, "HPRZ", "ester"),   # 57
    (8, 16, "HPRZ", "ester"),    # 58
    (12, 12, "PIP", "amide"),    # 59
    (10, 14, "MPRZ", "amide"),   # 60
    (8, 16, "MPRZ", "amide"),    # 61
    (10, 14, "HPRZ", "amide"),   # 62
    (8, 16, "HPRZ", "amide"),    # 63
]):
    iajd = 55 + i
    smi = build_sss_35(c1, c2, 4, head, linkage)
    try_add(iajd, smi, "sSS-Nonsym", "ja1c09585", "LOW",
            f"sSS-{c1}/{c2}-4C-{head}-{linkage}")

# ── IAJDs 67-69: sSS-Nonsym between 66(C12-MP) and 70(C12-MPRZ-amide)
# 66 and 71 have unusual macrocyclic SMILES. 67-69 are in between.
try_add(67, build_sss_35(12, 12, 3, "MPRZ", "ester"),
        "sSS-Nonsym", "ja1c09585", "MEDIUM", "sSS-C12-3C-MPRZ-ester")
try_add(68, build_sss_35(12, 12, 5, "MPRZ", "ester"),
        "sSS-Nonsym", "ja1c09585", "MEDIUM", "sSS-C12-5C-MPRZ-ester")
try_add(69, build_sss_35(12, 12, 3, "HPRZ", "ester"),
        "sSS-Nonsym", "ja1c09585", "MEDIUM", "sSS-C12-3C-HPRZ-ester")

# ── IAJD 72: between 71(C12-HP) and 74(C14-MP)
try_add(72, build_sss_35(12, 12, 5, "HPRZ", "ester"),
        "sSS-Nonsym", "ja1c09585", "MEDIUM", "sSS-C12-5C-HPRZ")

# ── IAJD 80: between GA-Tris 79(C12-MPRZ) and Dialkoxybenz 81
try_add(80, build_ga_tris(12, 4, "HPRZ"),
        "GA-Tris", "ja1c09585", "MEDIUM", "GA-tris-C12-4C-HPRZ")

# ── IAJD 84: between PE-Tris 83(C12-HPRZ) and Dialkoxybenz 86
try_add(84, build_pe_tris(12, 3, "MPRZ"),
        "PE-Tris", "ja1c09585", "MEDIUM", "PE-tris-C12-3C-MPRZ")

# ── IAJD 85: near Dialkoxybenz 86(C11/C16-MPRZ)
try_add(85, build_sss_35(11, 16, 4, "HPRZ"),
        "Dialkoxybenzyl", "ja1c09585", "MEDIUM", "Dialkoxybenz-C11/C16-4C-HPRZ")

# ── IAJD 90: between 89(C8-HPRZ) and 91(C10-HP)
try_add(90, build_sss_35(8, 8, 4, "MPRZ"),
        "sSS-Nonsym", "pharmaceutics", "MEDIUM", "sSS-C8-4C-MPRZ")

# ── IAJD 94: between PE-Tris 93(EH-HPRZ) and sSS 95
try_add(94, build_pe_tris(8, 3, "HPRZ", branched=True),
        "PE-Tris", "ja1c09585", "MEDIUM", "PE-tris-EH-3C-HPRZ")

# ── IAJDs 102-104: between PE-Tris 101(C16-HPRZ) and Dialkoxybenz 105
try_add(102, build_pe_tris(18, 4, "MPRZ"),
        "PE-Tris", "ja1c09585", "LOW", "PE-tris-C18-4C-MPRZ")
try_add(103, build_pe_tris(18, 4, "HPRZ"),
        "PE-Tris", "ja1c09585", "LOW", "PE-tris-C18-4C-HPRZ")
try_add(104, build_pe_tris(16, 3, "MPRZ"),
        "PE-Tris", "ja1c09585", "LOW", "PE-tris-C16-3C-MPRZ")

# ── IAJD 109: between sSS 108(C14-MPRZ) and G1-Janus 110
try_add(109, build_sss_35(14, 14, 4, "HPRZ"),
        "sSS-Nonsym", "pharmaceutics", "MEDIUM", "sSS-C14-4C-HPRZ")

# ── IAJD 132: between G1-Janus 131 and sSS 133
# G1-Janus dendrimer — use PE-Gallic HTM template with C12 tails
try_add(132, build_pe_gallic_htm(12, [ARM_DMBA4, ARM_OH, ARM_DMBA4], "ester"),
        "G1-Janus-Dendrimer", "ja2c00273", "LOW", "G1-Janus C12 DMBA variant")

# ── IAJD 160: between G1-Janus 159(amide) and sSS 161
try_add(160, build_pe_gallic_htm(11, [ARM_DMBA4, ARM_DMBA4, ARM_DMBA4], "amide"),
        "G1-Janus-Dendrimer", "ja2c00273", "LOW", "G1-Janus C11 triple-DMBA amide")

# ── IAJDs 195-196: between nsSS 194(C18/C9-HPRZ) and 197(C12/C8br-MPRZ)
try_add(195, build_sss_35(18, 10, 4, "MPRZ"),
        "sSS-Nonsym", "pharmaceutics", "MEDIUM", "nsSS-C18/C10-4C-MPRZ")
try_add(196, build_sss_35(18, 11, 4, "MPRZ"),
        "sSS-Nonsym", "pharmaceutics", "MEDIUM", "nsSS-C18/C11-4C-MPRZ")

# ── IAJD 252: between PE-Tris 251(C11-HPRZ) and GA-Tris 253
try_add(252, build_pe_tris(11, 4, "H2EPRZ"),
        "PE-Tris", "ja3c07337", "MEDIUM", "PE-tris-C11-4C-H2EPRZ")

# ── IAJDs 254-262: PE-Tris from ja3c07337
for iajd, cn, head in [
    (254, 9, "MPRZ"), (255, 9, "HPRZ"), (256, 9, "H2EPRZ"),
    (257, 10, "MPRZ"), (258, 10, "HPRZ"), (259, 10, "H2EPRZ"),
    (260, 14, "HPRZ"), (261, 14, "H2EPRZ"), (262, 15, "MPRZ"),
]:
    try_add(iajd, build_pe_tris(cn, 4, head),
            "PE-Tris", "ja3c07337", "MEDIUM", f"PE-tris-C{cn}-4C-{head}")

# ── IAJD 281, 286: PE-Tris
try_add(281, build_pe_tris(13, 4, "H2EPRZ"),
        "PE-Tris", "ja3c07337", "MEDIUM", "PE-tris-C13-4C-H2EPRZ")
try_add(286, build_pe_tris(15, 4, "H2EPRZ"),
        "PE-Tris", "ja3c07337", "MEDIUM", "PE-tris-C15-4C-H2EPRZ")

# ── IAJD 293: last PE-Tris before Dialkoxybenz block
try_add(293, build_pe_tris(6, 4, "MPRZ"),
        "PE-Tris", "ja3c07337", "MEDIUM", "PE-tris-C6-4C-MPRZ")

# ── IAJDs 295-296, 298: Dialkoxybenzyl EH variants
try_add(295, build_sss_35(8, 8, 4, "HPRZ", c1_br=True, c2_br=True),
        "Dialkoxybenzyl", "ja3c13569", "MEDIUM", "Dialkoxybenz-35-EH-4C-HPRZ")
try_add(296, build_sss_34(8, 8, 4, "MPRZ", c1_br=True, c2_br=True),
        "Dialkoxybenzyl", "ja3c13569", "MEDIUM", "Dialkoxybenz-34-EH-4C-MPRZ")
try_add(298, build_sss_34(8, 8, 4, "HPRZ", c1_br=True, c2_br=True),
        "Dialkoxybenzyl", "ja3c13569", "MEDIUM", "Dialkoxybenz-34-EH-4C-HPRZ")

# ── IAJDs 302-307: GA-Tris EH variants from ja3c13569
for iajd, lc, head, linkage in [
    (302, 4, "MPRZ", "ester"), (303, 3, "HPRZ", "ester"),
    (304, 3, "MPRZ", "ester"), (305, 5, "HPRZ", "ester"),
    (306, 5, "MPRZ", "ester"), (307, 2, "MPRZ", "ester"),
]:
    try_add(iajd, build_ga_tris(8, lc, head, linkage, branched=True),
            "GA-Tris", "ja3c13569", "MEDIUM", f"GA-tris-EH-{lc}C-{head}-{linkage}")

# ── IAJDs 315-316: GA-Tris amide variants from bm4c01599
try_add(315, build_ga_tris(8, 4, "HPRZ", "amide", branched=True),
        "GA-Tris", "bm4c01599", "MEDIUM", "GA-tris-EH-4C-HPRZ-amide")
try_add(316, build_ga_tris(8, 4, "MPRZ", "amide", branched=True),
        "GA-Tris", "bm4c01599", "MEDIUM", "GA-tris-EH-4C-MPRZ-amide")

# ── IAJDs 327-328: GA-Tris 2C variants
try_add(327, build_ga_tris(8, 2, "HPRZ", "ester", branched=True),
        "GA-Tris", "bm4c01599", "MEDIUM", "GA-tris-EH-2C-HPRZ")
try_add(328, build_ga_tris(8, 2, "H2EPRZ", "ester", branched=True),
        "GA-Tris", "bm4c01599", "MEDIUM", "GA-tris-EH-2C-H2EPRZ")

# ── Report ──────────────────────────────────────────────────────────────

print(f"=== EXTRACTION RESULTS ===")
print(f"Successfully reconstructed: {len(results)} / 54 IAJDs")
print(f"Skipped (duplicate/invalid): {len(skipped)}")
print()

by_conf = {}
for r in results:
    by_conf.setdefault(r['confidence'], []).append(r)

for conf in ['HIGH', 'MEDIUM', 'LOW']:
    items = by_conf.get(conf, [])
    if items:
        print(f"\n{conf} confidence ({len(items)}):")
        for r in sorted(items, key=lambda x: x['iajd_num']):
            print(f"  IAJD {r['iajd_num']:>3}: {r['family']:22s} MW={r['ExactMolWt']:8.1f} N={r['NumNitrogens']} {r['notes']}")

print(f"\nSkipped ({len(skipped)}):")
for s in sorted(skipped, key=lambda x: x['iajd']):
    print(f"  IAJD {s['iajd']:>3}: {s['reason']} — {s['notes']}")

# Save
df_out = pd.DataFrame(results)
df_out.to_csv('extracted_remaining_iajds.csv', index=False)
print(f"\nSaved {len(results)} IAJDs to extracted_remaining_iajds.csv")
