"""finalize_dataset.py — finish the metadata fill with the improved inference, flag the
8 malformed-SMILES G1-Janus rows, fill derivable architecture columns. Honest: never
fabricates experimental labels; flags bad SMILES rather than deriving garbage from them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "audit_work")
import importlib, infer_metadata
importlib.reload(infer_metadata)
from infer_metadata import infer_row

DS = Path("IAJD_master/datasets")
MALFORMED = {110, 111, 131, 155, 156, 157, 158, 159}   # G1-Janus polyene-macrocycle bad reconstructions
FILES = ["IAJD_Bioact_v13_clean", "IAJD_pKa_v21_final",
         "IAJD_Bioact_v13_clean.AUDIT_FIXED", "IAJD_pKa_v21_final.AUDIT_FIXED"]


def finalize(f: Path):
    df = pd.read_excel(f)
    smicol = "SMILES_canonical" if "SMILES_canonical" in df else "SMILES"
    filled = 0
    for i, r in df.iterrows():
        num = int(r["IAJD_num"]) if pd.notna(r.get("IAJD_num")) else None
        # flag malformed rows
        if num in MALFORMED and "audit_status" in df:
            a = str(r.get("audit_status") or "")
            if "MALFORMED" not in a:
                df.at[i, "audit_status"] = (a + "; FLAG_MALFORMED_SMILES_G1Janus_polyene_needs_reconstruct").lstrip("; ")
            continue
        inf = infer_row(r.get(smicol) or r.get("SMILES"))
        for col, k in (("head_group", "head_group"), ("head_amine_label", "head_group"),
                       ("linker_length", "linker_length"), ("Linker_Length", "linker_length"),
                       ("linker_carbons", "linker_length"), ("linkage", "linkage")):
            if col in df and pd.isna(r.get(col)) and inf[k] is not None:
                df.at[i, col] = (float(inf[k]) if "linker" in col.lower() else inf[k]); filled += 1
        # derivable architecture columns from family
        fam = r.get("family")
        if pd.notna(fam):
            for col in ("architecture", "architecture_coarse", "architecture_v13", "family_original"):
                if col in df and pd.isna(r.get(col)):
                    df.at[i, col] = fam
    df.to_excel(f, index=False)
    return filled


def main():
    for name in FILES:
        f = DS / f"{name}.xlsx"
        if f.exists():
            n = finalize(f)
            df = pd.read_excel(f)
            bad = df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False) if "audit_status" in df else pd.Series(False, index=df.index)
            cln = df[~bad]
            nan3 = {c: int(cln[c].isna().sum()) for c in ("head_group", "linker_length", "linkage") if c in cln}
            print(f"{f.name}: filled {n} cells | NaN in clean rows now: {nan3}")


if __name__ == "__main__":
    main()
