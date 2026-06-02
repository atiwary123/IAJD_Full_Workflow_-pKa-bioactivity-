"""reconstruct_g1janus.py — store the image-reconstructed (formula-validated) SMILES for the
8 G1-Janus dendrimers from pharmaceutics-15-01572 Fig 10/11, and regenerate their structural
features with the SAME pipeline used for the rest of the dataset (expand_datasets).

Reconstruction basis (best-guess from images, per user request 2026-06-02):
- family architecture from the clean G1-Janus templates (IAJD34 class): outer benzene (2 alkyl
  tails) - CH2-[ester|amide] - inner benzene [1 benzyl-OEG cap + 2 triethylene-glycol-ester-
  PIPERIDINE heads]. Piperidine = the clean lung-series head (IAJD34).
- tails read from Fig 11A schematics / paper text (111 = 11+11 "equal alkyl groups").
- EVERY reconstruction's molecular formula EXACTLY matches the dataset's recorded formula.
- IAJD131 tail split (14+15) is one of 4 formula-equivalent splits -> flagged lower-confidence.
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
sys.path.insert(0, "audit_work")
from expand_datasets import compute_rdkit_features, compute_3d_features
from infer_metadata import infer_row

DEND = "OCCOCCOCCOCc3ccccc3"
PIP = "OCCOCCOCCOC(=O)CCCN3CCCCC3"
def g1(t1, t2, link):
    L = "CNC(=O)" if link == "amide" else "COC(=O)"
    return f"{t1}Oc1cc({L}c2cc({DEND})c({PIP})c({PIP})c2)cc(O{t2})c1"
C = lambda n: "C" * n
EH = "CC(CC)CCCC"

TAG = "SMILES_IMAGE_RECONSTRUCTED_2026-06-02_formula_validated_pharmaceutics1501572_Fig10-11"
RECON = {  # iajd_num -> (smiles, extra_note)
    110: (g1(C(12), C(12), "ester"), ""),
    111: (g1(C(11), C(11), "ester"), ""),
    155: (g1(C(11), EH,    "ester"), ""),
    156: (g1(C(11), C(14), "ester"), ""),
    157: (g1(C(11), C(15), "ester"), ""),
    158: (g1(C(11), C(17), "ester"), ""),
    159: (g1(C(11), C(17), "amide"), ""),
    131: (g1(C(14), C(15), "ester"), "; tail_split_best_guess(14+15 of 4 formula-equivalent)"),
}
DS = Path("IAJD_master/datasets")
FILES = ["IAJD_Bioact_v13_clean", "IAJD_pKa_v21_final",
         "IAJD_Bioact_v13_clean.AUDIT_FIXED", "IAJD_pKa_v21_final.AUDIT_FIXED"]


def feature_row(smi):
    feats = {}
    try:
        feats.update(compute_rdkit_features(smi))
    except Exception as e:
        print(f"   [warn] 2D features failed: {e}")
    try:
        feats.update(compute_3d_features(smi))
    except Exception as e:
        print(f"   [warn] 3D features failed: {e}")
    inf = infer_row(smi)
    for k, v in (("head_group", inf["head_group"]), ("head_amine_label", inf["head_group"]),
                 ("linker_length", inf["linker_length"]), ("linker_carbons", inf["linker_length"]),
                 ("Linker_Length", inf["linker_length"]), ("linkage", inf["linkage"])):
        if v is not None:
            feats[k] = v
    return feats


def main():
    # compute once per IAJD (3D is the slow part)
    print("computing features for 8 reconstructions (3D embedding ~big molecules)…")
    computed = {}
    for n, (smi, note) in RECON.items():
        cano = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        feats = feature_row(smi)
        computed[n] = (smi, cano, feats, TAG + note)
        print(f"  IAJD{n}: {len(feats)} features computed")
    for name in FILES:
        f = DS / f"{name}.xlsx"
        if not f.exists():
            continue
        df = pd.read_excel(f)
        bak = DS / f"{name}.PRE_RECON.xlsx"
        if not bak.exists():
            shutil.copy(f, bak)
        idcol = "IAJD_num" if "IAJD_num" in df.columns else ("IAJD" if "IAJD" in df.columns else None)
        if idcol is None:
            print(f"{f.name}: no IAJD id column — skipped"); continue
        nset = 0
        for n, (smi, cano, feats, tag) in computed.items():
            mask = df[idcol] == n
            if not mask.any():
                continue
            i = df.index[mask][0]
            df.at[i, "SMILES"] = smi
            if "SMILES_canonical" in df:
                df.at[i, "SMILES_canonical"] = cano
            df.at[i, "audit_status"] = tag
            for k, v in feats.items():
                if k in df.columns:
                    df.at[i, k] = v
            nset += 1
        df.to_excel(f, index=False)
        print(f"{f.name}: updated {nset} rows")


if __name__ == "__main__":
    main()
