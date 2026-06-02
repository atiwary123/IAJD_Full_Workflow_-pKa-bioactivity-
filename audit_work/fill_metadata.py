"""fill_metadata.py — fill the STRUCTURE-DERIVABLE metadata blanks (head_group,
linker_length, linkage, linker_carbons, head_amine_label) from SMILES, in BOTH the
canonical and AUDIT_FIXED xlsx. Never overwrites an existing value; never touches
experimental columns. Backs up first. Reports decompose coverage before/after.
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, "audit_work")
from infer_metadata import infer_row
sys.path.insert(0, ".")
from iajd_grammar import decompose_row

DS = Path("IAJD_master/datasets")
FILES = ["IAJD_Bioact_v13_clean", "IAJD_pKa_v21_final"]


def _decompose_ok_count(df):
    bad = df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False) if "audit_status" in df else pd.Series(False, index=df.index)
    sub = df[~bad]
    ok = 0
    for _, r in sub.iterrows():
        try:
            ok += decompose_row(r) is not None
        except Exception:
            pass
    return ok, len(sub)


def fill_df(df):
    smicol = "SMILES_canonical" if "SMILES_canonical" in df else "SMILES"
    filled = {"head_group": 0, "linker_length": 0, "linkage": 0, "linker_carbons": 0, "head_amine_label": 0}
    for i, r in df.iterrows():
        inf = infer_row(r.get(smicol) or r.get("SMILES"))
        # head_group
        if "head_group" in df and pd.isna(r.get("head_group")) and inf["head_group"]:
            df.at[i, "head_group"] = inf["head_group"]; filled["head_group"] += 1
        if "head_amine_label" in df and pd.isna(r.get("head_amine_label")) and inf["head_group"]:
            df.at[i, "head_amine_label"] = inf["head_group"]; filled["head_amine_label"] += 1
        # linker_length (+ alt encodings)
        if inf["linker_length"] is not None:
            for col in ("linker_length", "Linker_Length", "linker_carbons"):
                if col in df and pd.isna(r.get(col)):
                    df.at[i, col] = float(inf["linker_length"]); filled["linker_length" if col != "linker_carbons" else "linker_carbons"] += 1
        # linkage (chemically-correct ester/amide; only where blank)
        if "linkage" in df and pd.isna(r.get("linkage")) and inf["linkage"]:
            df.at[i, "linkage"] = inf["linkage"]; filled["linkage"] += 1
    return filled


def main():
    for name in FILES:
        for suffix in ("", ".AUDIT_FIXED"):
            f = DS / f"{name}{suffix}.xlsx"
            if not f.exists():
                continue
            df = pd.read_excel(f)
            ok0, ntot = _decompose_ok_count(df)
            shutil.copy(f, DS / f"{name}{suffix}.PRE_METAFILL.xlsx")
            filled = fill_df(df)
            df.to_excel(f, index=False)
            ok1, _ = _decompose_ok_count(df)
            print(f"{f.name}: filled {filled}")
            print(f"   decompose-OK (non-flagged): {ok0} -> {ok1}  (of {ntot})")


if __name__ == "__main__":
    main()
