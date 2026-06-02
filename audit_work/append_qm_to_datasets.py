"""append_qm_to_datasets.py — merge the real xTB QM descriptors into the fixed datasets.

After the overnight precompute_qm run, this joins qm_cache.csv (keyed by canonical SMILES)
onto each corrected dataset and writes the 8 QM descriptor columns alongside the existing
features. Rows whose canonical SMILES has no real QM entry (audit-flagged, or QM not yet
computed / failed) get blank QM cells — never a proxy. Idempotent: re-running refreshes the
qm_* columns. The QM values are flagged in a `qm_source` column ('xtb' vs '' = none).

Usage: python audit_work/append_qm_to_datasets.py [--dry-run]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
QM_CACHE = ROOT / "IAJD_master/bundles_caches/physics/qm_cache.csv"
QM_COLS = ["qm_q_ionizableN", "qm_dipole_D", "qm_polarizability", "qm_homo_lumo_eV",
           "qm_dGsolv_kJmol", "qm_dGsolv_head", "qm_dGsolv_tail", "qm_Ehedup"]
# (file, smiles_column_to_join_on)
TARGETS = [
    ("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx", "SMILES_canonical"),
    ("IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx", "SMILES_canonical"),
    ("IAJD_master/datasets/IAJD_pKa_v21_final.xlsx", "SMILES"),
    ("IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx", "SMILES"),
]
DRY = "--dry-run" in sys.argv


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    qm = pd.read_csv(QM_CACHE)
    # dict: canonical SMILES -> {qm_col: value} keeping only finite (real) values
    qmap = {}
    for _, r in qm.iterrows():
        vals = {c: float(r[c]) for c in QM_COLS if c in qm.columns}
        if any(np.isfinite(v) for v in vals.values()):
            qmap[str(r["smiles_canonical"])] = vals
    print(f"qm_cache: {len(qm)} rows, {len(qmap)} with >=1 finite QM descriptor")

    for path, smi_col in TARGETS:
        p = ROOT / path
        if not p.exists():
            print(f"  SKIP missing {path}"); continue
        xl = pd.ExcelFile(p); sheet = xl.sheet_names[0]
        df = pd.read_excel(p, sheet_name=sheet)
        keys = df[smi_col].map(canon)
        n_real = 0
        cols = {c: [] for c in QM_COLS}; src = []
        for k in keys:
            rec = qmap.get(k) if k else None
            if rec is not None:
                n_real += 1; src.append("xtb")
                for c in QM_COLS:
                    cols[c].append(rec.get(c, np.nan))
            else:
                src.append("")
                for c in QM_COLS:
                    cols[c].append(np.nan)
        print(f"  {p.name:42s} rows={len(df):3d}  real QM={n_real:3d}  ({100*n_real/len(df):.0f}%)")
        if DRY:
            continue
        for c in QM_COLS:
            df[c] = cols[c]
        df["qm_source"] = src
        df.to_excel(p, sheet_name=sheet, index=False)
    print("DRY-RUN (no writes)" if DRY else "Wrote QM columns into all target datasets.")


if __name__ == "__main__":
    main()
