"""regen_step2_fix_canonical.py — recompute SMILES_canonical = canonical(SMILES).

ROOT CAUSE FIX (2026-06-01 regen): the audit corrected the `SMILES` column but
left `SMILES_canonical` (the join key used by precompute_qm, agile_embeddings,
physics_cache_io, bioact_v14_pipeline) holding the OLD wrong structure for 35/36
corrected rows. This script re-derives SMILES_canonical from the corrected SMILES
so the corrections actually propagate to every structure-keyed cache/model.

Operates in place on the *.AUDIT_FIXED.xlsx files (already backed up to
audit_work/pre_regen_backup_20260601/). Idempotent.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
BK = ROOT / "audit_work" / "pre_regen_backup_20260601"
FILES = [
    ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx",
    ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx",
]


def canon(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def main():
    BK.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        print("=" * 70)
        print(path.name)
        # back up the AUDIT_FIXED file itself before editing
        shutil.copy2(path, BK / path.name)
        xl = pd.ExcelFile(path)
        sheet = xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
        n = len(df)
        if "SMILES_canonical" not in df.columns:
            # Confirm all SMILES parse; consumers canonicalize from SMILES live.
            bad = [i for i, s in enumerate(df["SMILES"]) if canon(s) is None]
            print(f"  no SMILES_canonical column (consumers canonicalize live). "
                  f"unparseable SMILES: {len(bad)}/{n}")
            if bad:
                print("    rows with unparseable SMILES:", bad[:20])
            continue
        new_canon = []
        n_changed = n_unparse = n_same = 0
        for _, r in df.iterrows():
            c = canon(r["SMILES"])
            if c is None:
                # keep existing value, never blank a row out; report it
                new_canon.append(r["SMILES_canonical"])
                n_unparse += 1
                continue
            old = r["SMILES_canonical"]
            # compare structurally (re-canonicalize the old value too)
            old_c = canon(old) if isinstance(old, str) else None
            if old_c != c:
                n_changed += 1
            else:
                n_same += 1
            new_canon.append(c)
        df["SMILES_canonical"] = new_canon
        df.to_excel(path, sheet_name=sheet, index=False)
        print(f"  rows={n}  SMILES_canonical updated: changed={n_changed} "
              f"unchanged={n_same} unparseable_kept={n_unparse}")
        # verify: 0 structural mismatches now
        mism = 0
        for _, r in df.iterrows():
            c = canon(r["SMILES"]); oc = canon(r["SMILES_canonical"])
            if c is not None and oc is not None and c != oc:
                mism += 1
        print(f"  VERIFY post-fix structural mismatches: {mism}")


if __name__ == "__main__":
    main()
