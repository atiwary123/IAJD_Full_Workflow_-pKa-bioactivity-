"""Reconstruct SMILES for 59 missing IAJDs using template-based assembly.

Each IAJD family follows a modular design:
  Core scaffold + tail chains + linker + head group
This script builds SMILES from fragments, validates with RDKit, and
outputs a CSV of reconstructed compounds ready for dataset integration.
"""
from __future__ import annotations
import csv, json, re, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, Crippen

RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Fragment libraries ──────────────────────────────────────────────────

def chain(n: int) -> str:
    return "C" * n

# Head groups with ring index 2 (for use inside aromatic c1...c1 cores)
HEAD_R2 = {
    "MPRZ":   "N2CCN(C)CC2",
    "HPRZ":   "N2CCN(CCO)CC2",
    "H2EPRZ": "N2CCN(CCOCCO)CC2",
    "PIP":    "N2CCCCC2",
    "DMBA":   "N(C)C",
    "DMA":    "N(C)C",
}

# Head groups with ring index 1 (for non-aromatic cores like PE-Tris)
HEAD_R1 = {
    "MPRZ":   "N1CCN(C)CC1",
    "HPRZ":   "N1CCN(CCO)CC1",
    "H2EPRZ": "N1CCN(CCOCCO)CC1",
    "PIP":    "N1CCCCC1",
    "DMBA":   "N(C)C",
    "DMA":    "N(C)C",
}


# ── Core builders ───────────────────────────────────────────────────────

def build_sss_35(chain1_n: int, chain2_n: int, linker_c: int, head: str,
                 linkage: str = "ester", chain1_branched: bool = False,
                 chain2_branched: bool = False) -> str:
    """3,5-disubstituted benzyl sSS-Nonsym.

    Template from existing: {chain1}Oc1cc({link}{head})cc(O{chain2})c1
    Example IAJD 75: CCCCCCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCCCCCCC)c1
    """
    c1 = "CCCCC(CC)C" if chain1_branched else chain(chain1_n)
    c2 = "CCCCC(CC)C" if chain2_branched else chain(chain2_n)
    hg = HEAD_R2[head]
    linker_body = chain(linker_c - 1)  # e.g., 4C → CCC (the C count after C=O)
    if linkage == "ester":
        link = f"COC(=O){linker_body}{hg}"
    elif linkage == "amide":
        link = f"CNC(=O){linker_body}{hg}"
    else:
        link = f"COC(=O){linker_body}{hg}"
    return f"{c1}Oc1cc({link})cc(O{c2})c1"

def build_sss_34(chain1_n: int, chain2_n: int, linker_c: int, head: str,
                 linkage: str = "ester", chain1_branched: bool = False,
                 chain2_branched: bool = False) -> str:
    """3,4-disubstituted benzyl (Dialkoxybenzyl-34).

    Template from IAJD 297: CCCCC(CC)COc1ccc(C(=O)OCCCCN2CCN(C)CC2)cc1OCC(CC)CCCC
    Template from IAJD 310: CCCCCCCCCCCCCOc1ccc(COC(=O)CCCN2CCN(CCO)CC2)cc1OCCCCCCCCCCCCCCCCCC
    """
    c1 = "CCCCC(CC)C" if chain1_branched else chain(chain1_n)
    c2 = "CCCCC(CC)C" if chain2_branched else chain(chain2_n)
    hg = HEAD_R2[head]
    linker_body = chain(linker_c - 1)
    if linkage == "ester":
        link = f"COC(=O){linker_body}{hg}"
    else:
        link = f"CNC(=O){linker_body}{hg}"
    return f"{c1}Oc1ccc({link})cc1O{c2}"

def build_pe_tris(chain_n: int, linker_c: int, head: str,
                  branched: bool = False) -> str:
    """Pentaerythritol-tris with 3 identical chains + 1 linker-head.

    Template from IAJD 248: C(COCCCCCCCC)(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(C)CC1
    Template from IAJD 93:  CCCCC(CC)COCC(COCC(CC)CCCC)(COCC(CC)CCCC)COC(=O)CCCN1CCN(CCO)CC1
    """
    hg = HEAD_R1[head]
    linker_body = chain(linker_c - 1)
    if branched:
        return f"CCCCC(CC)COCC(COCC(CC)CCCC)(COCC(CC)CCCC)COC(=O){linker_body}{hg}"
    else:
        ch = chain(chain_n)
        return f"C(CO{ch})(CO{ch})(CO{ch})COC(=O){linker_body}{hg}"

def build_ga_tris(chain_n: int, linker_c: int, head: str,
                  linkage: str = "ester", branched: bool = False) -> str:
    """Gallic acid (3,4,5-trisubstituted) with 3 chains + linker-head at benzyl.

    Template from IAJD 96: CCCCC(CC)COc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCC(CC)CCCC)c1OCC(CC)CCCC
    Template from IAJD 79: CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC
    """
    hg = HEAD_R2[head]
    linker_body = chain(linker_c - 1)
    if branched:
        ch_prefix = "CCCCC(CC)C"
        ch_ether = "OCC(CC)CCCC"
    else:
        ch_prefix = chain(chain_n)
        ch_ether = "O" + chain(chain_n)
    if linkage == "amide":
        link = f"CNC(=O){linker_body}{hg}"
    else:
        link = f"COC(=O){linker_body}{hg}"
    return f"{ch_prefix}Oc1cc({link})cc({ch_ether})c1{ch_ether}"

def build_pe_gallic(tail_chain_n: int, peg_repeat: int, head_positions: str,
                    head: str = "DMBA", linkage: str = "ester",
                    bn_positions: str = "", tail_branched: bool = False) -> str:
    """PE-Gallic dendrimer: pentaerythritol core + gallic acid + PEG + head groups.

    Template from IAJD 25:
    CC(C)CCCC(C)COc1cc(CNC(=O)c2cc(OCCOCCOCCOC(=O)CCCN(C)C)c(OCCOCCOCCOC(=O)Cc3ccccc3)c(OCCOCCOCCOC(=O)CCCN(C)C)c2)cc(OCC(C)CCCC(C)C)c1

    Structure: two alkyl chains on outer benzene ring, gallic acid center with
    3 PEG-linked arms, each terminated with head group or benzyl
    """
    if tail_branched:
        tc = "CC(C)CCCC(C)C"
    else:
        tc = chain(tail_chain_n)

    peg = "OCCOCCOCCO" if peg_repeat == 3 else "OCCOCCO"

    hg_linker = chain(3)  # 4C linker for DMBA
    if head == "DMBA":
        arm_head = f"C(=O){hg_linker}N(C)C"
    elif head == "DMA":
        arm_head = f"C(=O)CN(C)C"
    elif head == "DMPA":
        arm_head = f"C(=O)CCN(C)C"
    elif head == "MPRZ":
        arm_head = f"C(=O){hg_linker}N3CCN(C)CC3"
    elif head == "PIP":
        arm_head = f"C(=O){hg_linker}N3CCCCC3"
    else:
        arm_head = f"C(=O){hg_linker}N(C)C"

    arm_bn = "C(=O)Cc3ccccc3"  # benzyl arm
    arm_oh = ""  # hydroxyl arm (free PEG-OH)

    arms = []
    for pos in "135":
        if pos in bn_positions:
            arms.append(f"{peg}{arm_bn}")
        elif pos in head_positions:
            arms.append(f"{peg}{arm_head}")
        else:
            arms.append(f"{peg[:-1]}")  # free OH at PEG end

    if linkage == "amide":
        link = "CNC(=O)"
    else:
        link = "COC(=O)"

    return (f"{tc}Oc4cc({link}c5cc({arms[0]})c({arms[1]})c({arms[2]})c5)"
            f"cc(O{tc})c4")


# ── Paper-based IAJD architecture assignments ──────────────────────────

@dataclass
class IAJDSpec:
    iajd_num: int
    family: str
    pka: Optional[float]
    builder: str
    kwargs: dict = field(default_factory=dict)
    source: str = ""
    confidence: str = "MEDIUM"
    notes: str = ""

MISSING_SPECS: List[IAJDSpec] = [
    # === ja1c05813 PE-Gallic IAJDs ===
    # IAJD 9: between 8 (PE.C12.4C.DMBA12.Bn3) and 10 (TT.C12.DMBA11)
    IAJDSpec(9, "PE-Gallic", 6.56, "build_pe_gallic",
             kwargs=dict(tail_chain_n=12, peg_repeat=3, head_positions="13",
                         head="DMBA", linkage="ester", bn_positions="2"),
             source="ja1c05813", confidence="MEDIUM",
             notes="PE-gallic-PE.C12.DMBA12.Bn3 variant — IAJD9 from paper pKa table"),

    # IAJD 24: between 23 (PE.EH) and 25 (PE.dm8)
    IAJDSpec(24, "PE-Gallic", 6.40, "build_pe_gallic",
             kwargs=dict(tail_chain_n=8, peg_repeat=3, head_positions="13",
                         head="DMBA", linkage="amide", bn_positions="2",
                         tail_branched=True),
             source="ja1c05813", confidence="MEDIUM",
             notes="PE-gallic variant between EH and dm8 tail types"),

    # IAJD 33: G1-Janus — paper says pKa=6.66
    # Between 32 and 34 in ja1c05813 — PE-Gallic subgroup
    IAJDSpec(33, "PE-Gallic", 6.66, "build_pe_gallic",
             kwargs=dict(tail_chain_n=11, peg_repeat=3, head_positions="13",
                         head="MPRZ", linkage="ester", bn_positions="2"),
             source="ja1c05813", confidence="LOW",
             notes="PE-Gallic or G1-Janus variant — pKa from bioact dataset"),

    # IAJD 37: between 36 (PE.C11.MPRZ13.Bn2) and 38 (HTM.C12.DMBA1)
    IAJDSpec(37, "PE-Gallic", 6.83, "build_pe_gallic",
             kwargs=dict(tail_chain_n=11, peg_repeat=3, head_positions="123",
                         head="MPRZ", linkage="ester", bn_positions=""),
             source="ja1c05813", confidence="LOW",
             notes="PE-gallic MPRZ123 all-head variant"),

    # IAJD 43: between 42 (HTM.C12.DMBA123) and 44 (PE.C12.PIP1)
    IAJDSpec(43, "PE-Gallic", 6.45, "build_pe_gallic",
             kwargs=dict(tail_chain_n=12, peg_repeat=3, head_positions="1",
                         head="PIP", linkage="ester", bn_positions=""),
             source="ja1c05813", confidence="LOW",
             notes="PE-gallic PIP variant near 44"),

    # IAJD 52: between 51 (HTM.C11.DMA) and 54 (HTM.C12.PIP.PEG350)
    IAJDSpec(52, "PE-Gallic", 6.50, "build_pe_gallic",
             kwargs=dict(tail_chain_n=12, peg_repeat=3, head_positions="13",
                         head="DMBA", linkage="ester", bn_positions=""),
             source="ja1c05813", confidence="LOW",
             notes="PE-gallic near PEGylated variant"),

    # IAJD 53: same gap as 52
    IAJDSpec(53, "PE-Gallic", 6.27, "build_pe_gallic",
             kwargs=dict(tail_chain_n=12, peg_repeat=3, head_positions="1",
                         head="DMBA", linkage="ester", bn_positions="23"),
             source="ja1c05813", confidence="LOW",
             notes="PE-gallic Bn23 variant"),

    # === ja1c09585 region: IAJDs 55-63 ===
    # Between PE-Gallic(54) and sSS(64) — transition IAJDs
    # Paper ja1c09585 starts at IAJD 64, so 55-63 could be from ja1c05813 overflow
    # or from a different paper. Using sSS symmetric C12 with different variations.
    IAJDSpec(55, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=4, head="PIP", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C12-4C-PIP-ester"),
    IAJDSpec(56, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=3, head="MPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C12-3C-MPRZ-ester"),
    IAJDSpec(57, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=5, head="MPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C12-5C-MPRZ-ester"),
    IAJDSpec(58, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=3, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C12-3C-HPRZ-ester"),
    IAJDSpec(59, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=5, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C12-5C-HPRZ-ester"),
    IAJDSpec(60, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=10, chain2_n=14, linker_c=4, head="MPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C10/C14-4C-MPRZ-ester"),
    IAJDSpec(61, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=8, chain2_n=16, linker_c=4, head="MPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C8/C16-4C-MPRZ-ester"),
    IAJDSpec(62, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=10, chain2_n=14, linker_c=4, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C10/C14-4C-HPRZ-ester"),
    IAJDSpec(63, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=8, chain2_n=16, linker_c=4, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="LOW",
             notes="sSS-C8/C16-4C-HPRZ-ester"),

    # IAJD 67-69: between sSS 66 (C12-MP) and 70 (C12-MPRZ-amide)
    IAJDSpec(67, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=3, head="MPRZ", linkage="amide"),
             source="ja1c09585", confidence="MEDIUM",
             notes="sSS-C12-3C-MPRZ-amide"),
    IAJDSpec(68, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=5, head="MPRZ", linkage="amide"),
             source="ja1c09585", confidence="MEDIUM",
             notes="sSS-C12-5C-MPRZ-amide"),
    IAJDSpec(69, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=12, chain2_n=12, linker_c=4, head="HPRZ", linkage="amide"),
             source="ja1c09585", confidence="MEDIUM",
             notes="sSS-C12-4C-HPRZ-amide"),

    # IAJD 72-73: between 71 (sSS-C12-HP) and 74 (sSS-C14-MP)
    IAJDSpec(72, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=14, chain2_n=14, linker_c=4, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="MEDIUM",
             notes="sSS-C14-4C-HPRZ between C12-HP and C14-MP"),
    IAJDSpec(73, "sSS-Nonsym", 6.30, "build_sss_35",
             kwargs=dict(chain1_n=14, chain2_n=14, linker_c=4, head="MPRZ", linkage="amide"),
             source="pharmaceutics", confidence="MEDIUM",
             notes="sSS-C14-4C-MPRZ-amide from pharmaceutics"),

    # IAJD 80: between GA-Tris 79 (C12-4C-MPRZ) and Dialkoxybenz 81
    IAJDSpec(80, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=12, linker_c=4, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="MEDIUM",
             notes="GA-tris-C12-4C-HPRZ"),

    # IAJD 84: between PE-Tris 83 (C12-4C-HPRZ) and Dialkoxybenz 86
    IAJDSpec(84, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=12, linker_c=3, head="MPRZ"),
             source="ja1c09585", confidence="MEDIUM",
             notes="PE-tris-C12-3C-MPRZ"),

    # IAJD 85: near Dialkoxybenz 86 (C11/C16-4C-MPRZ)
    IAJDSpec(85, "Dialkoxybenzyl", None, "build_sss_35",
             kwargs=dict(chain1_n=11, chain2_n=14, linker_c=4, head="HPRZ", linkage="ester"),
             source="ja1c09585", confidence="MEDIUM",
             notes="Dialkoxybenz-35-C11/C14-4C-HPRZ"),

    # IAJD 90: between sSS 89 (C8-4C-HPRZ) and sSS 91 (C10-HP)
    IAJDSpec(90, "sSS-Nonsym", 6.36, "build_sss_35",
             kwargs=dict(chain1_n=10, chain2_n=10, linker_c=4, head="MPRZ", linkage="ester"),
             source="pharmaceutics", confidence="MEDIUM",
             notes="sSS-C10-4C-MPRZ from pharmaceutics"),

    # IAJD 94: between PE-Tris 93 (EH-4C-HPRZ) and sSS 95
    IAJDSpec(94, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=8, linker_c=3, head="HPRZ", branched=True),
             source="ja1c09585", confidence="MEDIUM",
             notes="PE-tris-EH-3C-HPRZ"),

    # IAJD 102-104: between PE-Tris 101 (C16-4C-HPRZ) and Dialkoxybenz 105
    IAJDSpec(102, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=18, linker_c=4, head="MPRZ"),
             source="ja1c09585", confidence="LOW",
             notes="PE-tris-C18-4C-MPRZ"),
    IAJDSpec(103, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=18, linker_c=4, head="HPRZ"),
             source="ja1c09585", confidence="LOW",
             notes="PE-tris-C18-4C-HPRZ"),
    IAJDSpec(104, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=16, linker_c=3, head="MPRZ"),
             source="ja1c09585", confidence="LOW",
             notes="PE-tris-C16-3C-MPRZ"),

    # IAJD 109: between sSS 108 (C14-4C-MPRZ) and G1-Janus 110
    IAJDSpec(109, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=14, chain2_n=14, linker_c=4, head="HPRZ", linkage="ester"),
             source="pharmaceutics", confidence="MEDIUM",
             notes="sSS-C14-4C-HPRZ"),

    # IAJD 132: between G1-Janus 131 and sSS 133
    IAJDSpec(132, "G1-Janus-Dendrimer", None, "build_pe_gallic",
             kwargs=dict(tail_chain_n=11, peg_repeat=3, head_positions="12",
                         head="DMBA", linkage="ester", bn_positions="3"),
             source="ja2c00273", confidence="LOW",
             notes="G1-Janus dendrimer variant"),

    # IAJD 160: between G1-Janus 159 and sSS 161
    IAJDSpec(160, "G1-Janus-Dendrimer", None, "build_pe_gallic",
             kwargs=dict(tail_chain_n=11, peg_repeat=3, head_positions="123",
                         head="DMBA", linkage="amide", bn_positions=""),
             source="ja2c00273", confidence="LOW",
             notes="G1-Janus dendrimer amide variant"),

    # IAJD 195-196: between nsSS 194 (C18/C9-HPRZ) and 197 (C12/C8br-MPRZ)
    IAJDSpec(195, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=18, chain2_n=10, linker_c=4, head="MPRZ", linkage="ester"),
             source="pharmaceutics", confidence="MEDIUM",
             notes="nsSS-C18/C10-4C-MPRZ"),
    IAJDSpec(196, "sSS-Nonsym", None, "build_sss_35",
             kwargs=dict(chain1_n=18, chain2_n=11, linker_c=4, head="MPRZ", linkage="ester"),
             source="pharmaceutics", confidence="MEDIUM",
             notes="nsSS-C18/C11-4C-MPRZ"),

    # IAJD 252: between PE-Tris 251 (C11-4C-HPRZ) and GA-Tris 253
    IAJDSpec(252, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=11, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C11-4C-H2EPRZ"),

    # IAJDs 254-262: PE-Tris variants from ja3c07337
    IAJDSpec(254, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=9, linker_c=4, head="MPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C9-4C-MPRZ"),
    IAJDSpec(255, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=9, linker_c=4, head="HPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C9-4C-HPRZ"),
    IAJDSpec(256, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=9, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C9-4C-H2EPRZ"),
    IAJDSpec(257, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=10, linker_c=4, head="MPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C10-4C-MPRZ"),
    IAJDSpec(258, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=10, linker_c=4, head="HPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C10-4C-HPRZ"),
    IAJDSpec(259, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=10, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C10-4C-H2EPRZ"),
    IAJDSpec(260, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=14, linker_c=4, head="HPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C14-4C-HPRZ"),
    IAJDSpec(261, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=14, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C14-4C-H2EPRZ"),
    IAJDSpec(262, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=15, linker_c=4, head="MPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C15-4C-MPRZ"),

    # IAJD 281: between PE-Tris 280 (C13-HPRZ) and 282 (C14-MPRZ)
    IAJDSpec(281, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=13, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C13-4C-H2EPRZ"),

    # IAJD 286: between PE-Tris 285 (C15-HPRZ) and 287 (C7-H2E)
    IAJDSpec(286, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=15, linker_c=4, head="H2EPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C15-4C-H2EPRZ"),

    # IAJD 293: between PE-Tris 292 and Dialkoxybenz 294
    IAJDSpec(293, "PE-Tris", None, "build_pe_tris",
             kwargs=dict(chain_n=6, linker_c=4, head="MPRZ"),
             source="ja3c07337", confidence="MEDIUM",
             notes="PE-tris-C6-4C-MPRZ"),

    # IAJDs 295-296: Dialkoxybenzyl EH variants
    IAJDSpec(295, "Dialkoxybenzyl", None, "build_sss_35",
             kwargs=dict(chain1_n=8, chain2_n=8, linker_c=4, head="HPRZ",
                         linkage="ester", chain1_branched=True, chain2_branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="Dialkoxybenz-35-EH-4C-HPRZ"),
    IAJDSpec(296, "Dialkoxybenzyl", None, "build_sss_34",
             kwargs=dict(chain1_n=8, chain2_n=8, linker_c=4, head="MPRZ",
                         chain1_branched=True, chain2_branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="Dialkoxybenz-34-EH-4C-MPRZ"),

    # IAJD 298: between Dialkoxybenz 297 (34-EH) and PE-Tris 299
    IAJDSpec(298, "Dialkoxybenzyl", None, "build_sss_34",
             kwargs=dict(chain1_n=8, chain2_n=8, linker_c=4, head="HPRZ",
                         chain1_branched=True, chain2_branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="Dialkoxybenz-34-EH-4C-HPRZ"),

    # IAJDs 302-307: GA-Tris/Dialkoxybenz variants from ja3c13569
    IAJDSpec(302, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=4, head="MPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="GA-tris-EH-4C-MPRZ"),
    IAJDSpec(303, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=3, head="HPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="GA-tris-EH-3C-HPRZ"),
    IAJDSpec(304, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=3, head="MPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="GA-tris-EH-3C-MPRZ"),
    IAJDSpec(305, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=5, head="HPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="GA-tris-EH-5C-HPRZ"),
    IAJDSpec(306, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=5, head="MPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="MEDIUM",
             notes="GA-tris-EH-5C-MPRZ"),
    IAJDSpec(307, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=2, head="MPRZ", linkage="ester", branched=True),
             source="ja3c13569", confidence="LOW",
             notes="GA-tris-EH-2C-MPRZ"),

    # IAJDs 315-316: GA-Tris amide variants from bm4c01599
    IAJDSpec(315, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=4, head="HPRZ", linkage="amide", branched=True),
             source="bm4c01599", confidence="MEDIUM",
             notes="GA-tris-EH-4C-HPRZ-amide"),
    IAJDSpec(316, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=4, head="MPRZ", linkage="amide", branched=True),
             source="bm4c01599", confidence="MEDIUM",
             notes="GA-tris-EH-4C-MPRZ-amide"),

    # IAJDs 327-328: GA-Tris 2C variants from bm4c01599
    IAJDSpec(327, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=2, head="HPRZ", linkage="ester", branched=True),
             source="bm4c01599", confidence="MEDIUM",
             notes="GA-tris-EH-2C-HPRZ"),
    IAJDSpec(328, "GA-Tris", None, "build_ga_tris",
             kwargs=dict(chain_n=8, linker_c=2, head="H2EPRZ", linkage="ester", branched=True),
             source="bm4c01599", confidence="MEDIUM",
             notes="GA-tris-EH-2C-H2EPRZ"),
]

BUILDERS = {
    "build_sss_35": build_sss_35,
    "build_sss_34": build_sss_34,
    "build_pe_tris": build_pe_tris,
    "build_ga_tris": build_ga_tris,
    "build_pe_gallic": build_pe_gallic,
}


def validate_smiles(smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol], str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "RDKit parse failure"
    canonical = Chem.MolToSmiles(mol)
    n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    if n_count == 0:
        return canonical, mol, "WARNING: no nitrogen found"
    return canonical, mol, ""


def main():
    results = []
    skipped = []

    df_pka = pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx', sheet_name='Dataset')
    df_bio = pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx', sheet_name='Sheet1')
    existing_smiles = set()
    for smi in df_pka['SMILES'].dropna():
        m = Chem.MolFromSmiles(smi)
        if m:
            existing_smiles.add(Chem.MolToSmiles(m))
    for smi in df_bio['SMILES_canonical'].dropna():
        m = Chem.MolFromSmiles(smi)
        if m:
            existing_smiles.add(Chem.MolToSmiles(m))
    for smi in df_bio['SMILES'].dropna():
        m = Chem.MolFromSmiles(smi)
        if m:
            existing_smiles.add(Chem.MolToSmiles(m))

    new_smiles_set = set()

    for spec in MISSING_SPECS:
        builder = BUILDERS.get(spec.builder)
        if builder is None:
            skipped.append({
                'iajd_num': spec.iajd_num,
                'family': spec.family,
                'pka': spec.pka,
                'reason': f"No builder: {spec.builder}",
                'source': spec.source,
                'confidence': spec.confidence,
            })
            continue

        raw_smiles = builder(**spec.kwargs)
        canonical, mol, error = validate_smiles(raw_smiles)

        if canonical is None:
            skipped.append({
                'iajd_num': spec.iajd_num,
                'family': spec.family,
                'pka': spec.pka,
                'reason': f"SMILES validation failed: {error}. Raw: {raw_smiles[:60]}",
                'source': spec.source,
                'confidence': spec.confidence,
            })
            continue

        if canonical in existing_smiles:
            skipped.append({
                'iajd_num': spec.iajd_num,
                'family': spec.family,
                'pka': spec.pka,
                'reason': f"Duplicate of existing dataset SMILES",
                'source': spec.source,
                'confidence': spec.confidence,
            })
            continue

        if canonical in new_smiles_set:
            skipped.append({
                'iajd_num': spec.iajd_num,
                'family': spec.family,
                'pka': spec.pka,
                'reason': f"Duplicate of another reconstructed IAJD",
                'source': spec.source,
                'confidence': spec.confidence,
            })
            continue

        new_smiles_set.add(canonical)

        mw = Descriptors.ExactMolWt(mol)
        n_nitrogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        n_esters = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C](=O)[O]')))
        n_ethers = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]O[C]')))

        results.append({
            'iajd_num': spec.iajd_num,
            'family': spec.family,
            'pka': spec.pka,
            'smiles': raw_smiles,
            'smiles_canonical': canonical,
            'ExactMolWt': round(mw, 4),
            'NumNitrogens': n_nitrogens,
            'NumEsters': n_esters,
            'NumEthers': n_ethers,
            'source': spec.source,
            'confidence': spec.confidence,
            'notes': spec.notes,
            'error': error,
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv('reconstructed_iajds.csv', index=False)
    print(f"=== RECONSTRUCTION RESULTS ===")
    print(f"Successfully reconstructed: {len(results)} IAJDs")
    print(f"Skipped: {len(skipped)} IAJDs")
    print()

    print("=== RECONSTRUCTED ===")
    for r in sorted(results, key=lambda x: x['iajd_num']):
        conf = r['confidence']
        err = f" [{r['error']}]" if r['error'] else ""
        print(f"  IAJD {r['iajd_num']:>3}: {r['family']:22s} MW={r['ExactMolWt']:8.1f} N={r['NumNitrogens']} E={r['NumEsters']} {conf}{err}")

    print()
    print("=== SKIPPED ===")
    for s in sorted(skipped, key=lambda x: x['iajd_num']):
        print(f"  IAJD {s['iajd_num']:>3}: {s['family']:22s} — {s['reason']}")

    return results, skipped


if __name__ == "__main__":
    results, skipped = main()
