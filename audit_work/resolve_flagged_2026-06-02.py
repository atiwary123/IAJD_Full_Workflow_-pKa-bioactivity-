#!/usr/bin/env python3
"""Resolve ALL flagged/unresolved rows (2026-06-02) with the most rigorous inference available,
recording an HONEST confidence tier + evidence per row. Writes both canonical and AUDIT_FIXED
copies of the pKa and bioact datasets (kept byte-identical), regenerates RDKit 2D+3D features for
structurally-changed rows, refreshes SMILES_canonical, and emits a full resolution ledger.

Confidence tiers:
  HIGH   - structure matches the independent 10118 Table-S1 descriptor vector (and/or SI MALDI
           formula). For CONFIRM rows the existing SMILES already matches; for FIX rows the new
           SMILES is the unique vector-consistent structure.
  MED    - not in 10118 and formula not in extractable SI text; structure is internally +
           homologous-series + architecture-label consistent (inference-grade).
  LOW    - molecular FORMULA is certain (10118 + SI), but connectivity is underdetermined by the
           available descriptors and needs the SI structural figure; best-guess SMILES provided
           and the row is kept excluded from structure-feature training.

Nothing here is fabricated: every SMILES either matches an independent fingerprint or is the
prior reconstruction shown to be self/series/architecture consistent. Originals are backed up.
"""
import warnings, json, shutil, sys, os; warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'audit_work')
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
RDLogger.DisableLog('rdApp.*')
from expand_datasets import compute_rdkit_features, compute_3d_features

DS = 'IAJD_master/datasets'
PKA = f'{DS}/IAJD_pKa_v21_final.xlsx'
BIO = f'{DS}/IAJD_Bioact_v13_clean.xlsx'
PKA_AF = f'{DS}/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx'
BIO_AF = f'{DS}/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx'

def canon(s): return Chem.MolToSmiles(Chem.MolFromSmiles(s))

# ---- the 30 best-guess (formula C58H114N2O9 certain; connectivity LOW) ----
S30 = 'CCCCCCCCCCCCCCCCCCCOCC(COCCOC(=O)CCCN(C)C)(COCCOC(=O)CCCN(C)C)COC(=O)CCCCCCCCCCCCCCCCC'
S33 = 'CCCCCCCCCCCCOc1cc(CNC(=O)c2cc(OCCOCCOCCOC(=O)CCCN3CCCCC3)c(OCCOCCOCCOCc3ccccc3)c(OCCOCCOCCOC(=O)CCCN3CCCCC3)c2)cc(OCCCCCCCCCCCC)c1'
S64 = 'CCCCCCCCCCCCOc1cc(COC(=O)CCCN(C)C)cc(OCCCCCCCCCCCC)c1'
S86 = 'CCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCC)c1'

GROUP_C = [26,27,28,29,38,39,40,41,42,43,47,48,49,50,51,53,54]

# resolution table: iajd -> dict(new_smiles, status, conf, evidence, labels, exclude)
R = {}
def add(iid, status, conf, evidence, new_smiles=None, labels=None, exclude=False):
    R[iid] = dict(new=new_smiles, status=status, conf=conf, ev=evidence, labels=labels or {}, exclude=exclude)

# --- HIGH: already match 10118 (confirm; fix mislabels where 10118 determines head identity) ---
add(31,  'CONFIRMED_10118', 'HIGH', 'full 6-descriptor vector matches 10118 #31 (MW1464.07, HBA21, Ar2)')
add(248, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #248 exactly; HBA7/HBD0 => head is MPRZ (N-methylpiperazine), not HPRZ',
    labels={'architecture':'PE-tris-C8-4C-MPRZ','head_group':'MPRZ','head_amine_label':'MPRZ'})
add(273, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #273 (MW867.44, HBA8/HBD0); head = methoxyethyl-piperazine (HBD0 rules out HPRZ/EEPRZ)',
    labels={'architecture':'PE-tris-C12-5C-MeOEtPRZ','head_group':'MeOEtPRZ','head_amine_label':'MeOEtPRZ'})
add(287, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #287 on 5/6 (MW/HBA/HBD/Ar/FCsp3 exact; RotB 37 vs 34 = known rotatable-bond definition diff); C8-tris+pentanoate+MeOEtPRZ',
    labels={'architecture':'PE-tris-C8-5C-MeOEtPRZ','head_group':'MeOEtPRZ','head_amine_label':'MeOEtPRZ'})
add(290, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #290 (MW629.0, HBA8/HBD1); head = HPRZ (hydroxyethyl-piperazine), butanoate')
add(291, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #291 (MW657.0, HBA8/HBD0); HBD0 => head is methoxyethyl-piperazine, not HPRZ; pentanoate',
    labels={'architecture':'PE-tris-C7-5C-MeOEtPRZ','head_group':'MeOEtPRZ','head_amine_label':'MeOEtPRZ'})
add(292, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #292 (MW673.0, HBA9/HBD1); head = hydroxyethoxyethyl-piperazine (H2EPRZ)')
add(297, 'CONFIRMED_10118', 'HIGH', 'matches 10118 #297 (MW562.84, HBA7/HBD1); HBD1 confirms HPRZ head; 3,4-bis(2-ethylhexyloxy)benzoate')

# --- HIGH: SMILES changed to the unique 10118-vector-consistent structure ---
add(64, 'RESOLVED_10118_FIX', 'HIGH',
    'TPSA48.0 = 2 ethers+1 ester+ONE tert-amine => head is DMA (dimethylamino), not MPRZ; bis-C12 dialkoxybenzyl 4-(dimethylamino)butanoate = C37H67NO4 = exact 10118 #64 (MW589.946,HBA5,HBD0,Ar1,RotB30). MPRZ->DMA.',
    new_smiles=S64, labels={'architecture':'Dialkoxybenz-35-C12-4C-DMA-ester','head_group':'DMA','head_amine_label':'DMA'})
add(86, 'RESOLVED_10118_FIX', 'HIGH',
    'bis-C11 (symmetric) = C38H68N2O4 = exact 10118 #86 (MW616.972,HBA6,RotB28); prior C11/C16 was +70 (5xCH2) too heavy. C16->C11.',
    new_smiles=S86, labels={'architecture':'Dialkoxybenz-35-C11-4C-MPRZ-ester','head_amine_label':'MPRZ'})
add(33, 'RESOLVED_10118+SI', 'HIGH',
    'bis-C12 gallic-AMIDE with 2x piperidine(PIP) + 1x benzyl(OBn) = C81H133N3O17 = exact 10118 #33 vector (MW1420.96,HBA19,HBD1,Ar3) AND SI MALDI calcd C81H134N3O17 = [M+H]+. Heads MPRZ->PIP (10118 needs 3 N not 5).',
    new_smiles=S33, labels={'architecture':'PE-gallic-amide-C12.PIP13.OBn2','head_group':'PIP','head_amine_label':'PIP'})

# --- LOW connectivity (formula HIGH): keep excluded from structure-training ---
add(30, 'FORMULA_RESOLVED_CONN_LOW', 'LOW',
    'formula C58H114N2O9 CERTAIN (10118 #30 MW983.555 + SI MALDI calcd C58H115N2O9=[M+H]+); DoU=3 => acyclic, 3 esters, 2 DMBA, Aromatic=0. Composition forces long ether/ester alkyl tails. Connectivity (tail split / core) underdetermined by descriptors -> needs SI figure. Best-guess pentaerythritol Janus provided; EXCLUDE from structure-features.',
    new_smiles=S30, labels={'architecture':'G1-Janus-aliphatic-PE.2arm.DMBA2 (connectivity unverified)'}, exclude=True)

# --- MED-HIGH: 10118 numbering collision; validated against own source paper ---
add(134, 'RESOLVED_NUMBERING_COLLISION', 'MED-HIGH',
    '10118 #134 is an unrelated ~2x-mass compound (MW1205.7,HBA17,Ar2) - numbering collision. This row = bis-C11 3,5-dialkoxybenzyl-amide-piperidine (C38H68N2O3) from pharmaceutics1501572 Fig11; internally + architecture consistent. Validated vs source paper, NOT vs 10118.')

# --- MED: not in 10118, formula not in SI text; series+arch+formula self-consistent ---
for iid in GROUP_C:
    add(iid, 'RESOLVED_SERIES_ARCH', 'MED',
        'PE-Gallic twin (Lib6/Lib4); not in 10118 library and formula not in extractable SI text. SMILES is internally consistent, fits the homologous chain-length series, and matches its architecture-label decomposition (head/benzyl/chain census). Inference-grade; SI hi-res figure would upgrade to HIGH.')

# ---------------- feature recompute for changed-SMILES rows ----------------
RD2 = {  # additional dataset columns computed in apply_round3 (kept for parity)
 'ExactMolWt':lambda m:Descriptors.ExactMolWt(m),'MolFormula':lambda m:rdMolDescriptors.CalcMolFormula(m),
 'HeavyAtomCount':lambda m:m.GetNumHeavyAtoms(),'NumNitrogens':lambda m:sum(a.GetSymbol()=='N' for a in m.GetAtoms()),
 'MolLogP':lambda m:Crippen.MolLogP(m),'TPSA':lambda m:rdMolDescriptors.CalcTPSA(m),
 'FractionCSP3':lambda m:rdMolDescriptors.CalcFractionCSP3(m),'RotatableBonds':lambda m:rdMolDescriptors.CalcNumRotatableBonds(m),
 'NumAromaticRings':lambda m:rdMolDescriptors.CalcNumAromaticRings(m),'NumHDonors':lambda m:rdMolDescriptors.CalcNumHBD(m),
 'NumHAcceptors':lambda m:rdMolDescriptors.CalcNumHBA(m)}

def full_features(smi, do3d=True):
    feats = {}
    try: feats.update(compute_rdkit_features(smi))
    except Exception as e: print(f'   [warn] 2D failed: {e}')
    if do3d:
        try: feats.update(compute_3d_features(smi))
        except Exception as e: print(f'   [warn] 3D failed: {e}')
    m = Chem.MolFromSmiles(smi)
    for c, fn in RD2.items(): feats[c] = fn(m)
    return feats

CHANGED = {30:S30, 33:S33, 64:S64, 86:S86}
print('Precomputing features for changed-SMILES rows (incl. 3D embedding)...')
FEAT = {}
for iid, smi in CHANGED.items():
    print(f'  IAJD {iid} ({rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))}) ...', flush=True)
    FEAT[iid] = full_features(smi, do3d=True)
    print(f'    -> {len(FEAT[iid])} feature cols')

# ---------------- apply ----------------
ledger = []
def new_status(conf, status, ev):
    return f'RESOLVED_2026-06-02 [{status}] confidence={conf}: {ev}'[:480]

def apply_file(path, idcol, tag):
    df = pd.read_excel(path)
    has_canon = 'SMILES_canonical' in df.columns
    for iid, res in R.items():
        mask = df[idcol] == iid
        if not mask.any(): continue
        old_smi = df.loc[mask, 'SMILES'].iloc[0]
        old_status = df.loc[mask, 'audit_status'].iloc[0]
        # SMILES + features
        if res['new'] is not None:
            cs = canon(res['new'])
            df.loc[mask, 'SMILES'] = cs
            if has_canon: df.loc[mask, 'SMILES_canonical'] = cs
            for c, v in FEAT[iid].items():
                if c in df.columns: df.loc[mask, c] = v
        elif has_canon:
            # confirmed-unchanged: ensure canonical consistent
            df.loc[mask, 'SMILES_canonical'] = canon(old_smi)
        # label fixes
        for c, v in res['labels'].items():
            if c in df.columns: df.loc[mask, c] = v
        # status (REPLACE the old UNRESOLVED/FLAG text)
        df.loc[mask, 'audit_status'] = new_status(res['conf'], res['status'], res['ev'])
        if tag == 'pKa':
            ledger.append(dict(IAJD=iid, file=tag, conf=res['conf'], status=res['status'],
                               smiles_changed=res['new'] is not None,
                               old_smiles=old_smi, new_smiles=df.loc[mask,'SMILES'].iloc[0],
                               old_status=str(old_status)[:120], evidence=res['ev'], exclude=res['exclude']))
    df.to_excel(path, index=False)
    print(f'  wrote {path}  ({mask.sum() if False else len(R)} resolutions attempted)')

# backups (whole-file) before writing
os.makedirs('audit_work/pre_flagfix_backup_20260602', exist_ok=True)
for p in [PKA, BIO, PKA_AF, BIO_AF]:
    shutil.copy2(p, 'audit_work/pre_flagfix_backup_20260602/' + os.path.basename(p))
print('backed up 4 dataset files to audit_work/pre_flagfix_backup_20260602/')

for path, idc in [(PKA,'IAJD'), (PKA_AF,'IAJD')]:
    apply_file(path, idc, 'pKa' if path==PKA else 'pKa_af')
for path, idc in [(BIO,'IAJD_num'), (BIO_AF,'IAJD_num')]:
    apply_file(path, idc, 'bioact' if path==BIO else 'bioact_af')

# ledger out
led = pd.DataFrame(ledger).sort_values('IAJD')
led.to_csv('audit_work/flagged_resolution_2026-06-02.csv', index=False)
json.dump({int(k): {kk: vv for kk, vv in v.items() if kk != 'labels'} | {'labels': v['labels']}
           for k, v in R.items()}, open('audit_work/flagged_resolution_2026-06-02.json','w'), indent=1)
print(f'\nLEDGER: audit_work/flagged_resolution_2026-06-02.csv  ({len(led)} rows)')
print('confidence breakdown:', dict(led['conf'].value_counts()))
print('SMILES changed for:', sorted(led[led.smiles_changed].IAJD.tolist()))
print('kept excluded (LOW conn):', sorted(led[led.exclude].IAJD.tolist()))
