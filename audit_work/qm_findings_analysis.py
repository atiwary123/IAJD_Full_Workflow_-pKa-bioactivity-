"""qm_findings_analysis.py — surface 'anything interesting' from the QM regen.

Computes (prints + dumps JSON):
  1. QM coverage (real xTB) per dataset and per family.
  2. QM-descriptor vs target correlations (Spearman): bioact log10_flux_total + pKa.
  3. OLD->NEW QM delta for audit-corrected compounds: the corrected SMILES' QM vs the
     OLD (wrong) SMILES' QM (still orphaned in qm_cache) — quantifies how much fixing the
     structure (e.g. PE-Gallic OH->OMe / phenylacetate->OBn) changed the physics.

Run at completion. Output: audit_work/qm_findings.json (+ stdout summary).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
QM = pd.read_csv(ROOT / "IAJD_master/bundles_caches/physics/qm_cache.csv").set_index("smiles_canonical")
QM_COLS = ["qm_q_ionizableN", "qm_dipole_D", "qm_polarizability", "qm_homo_lumo_eV",
           "qm_dGsolv_kJmol", "qm_dGsolv_head", "qm_dGsolv_tail", "qm_Ehedup"]
LEDGER = ROOT / "audit_work/AUDIT_corrections_master.csv"
out = {}


def canon(s):
    m = Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s.strip() else None
    return Chem.MolToSmiles(m) if m else None


def qm_of(smi):
    k = canon(smi)
    if k in QM.index:
        row = QM.loc[k]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        return {c: float(row[c]) for c in QM_COLS if c in QM.columns}
    return None


# --- 1+2: coverage + correlations vs target ---
for name, path, smicol, tgt in [
    ("bioact", "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx", "SMILES_canonical", "log10_flux_total"),
    ("pKa", "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx", "SMILES", "pKa")]:
    df = pd.read_excel(ROOT / path, sheet_name=0)
    if "audit_status" in df.columns:
        df = df[~df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)]
    df = df[df[tgt].notna()]
    recs = [qm_of(s) for s in df[smicol]]
    have = [r is not None for r in recs]
    cov = {"n": int(len(df)), "real_qm": int(sum(have))}
    y = df[tgt].astype(float).to_numpy()
    corr = {}
    for c in QM_COLS:
        v = np.array([r[c] if r else np.nan for r in recs], float)
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() >= 15:
            rho, p = spearmanr(v[m], y[m])
            corr[c] = {"spearman": round(float(rho), 3), "p": float(p), "n": int(m.sum())}
    out[name] = {"coverage": cov, "corr_vs_" + tgt: corr}
    print(f"[{name}] real QM {cov['real_qm']}/{cov['n']}; top |rho| vs {tgt}:",
          ", ".join(f"{k}={d['spearman']:+.2f}" for k, d in
                    sorted(corr.items(), key=lambda kv: -abs(kv[1]['spearman']))[:4]))

# --- 3: OLD -> NEW QM delta for audit-corrected compounds ---
# Use the PRE_AUDIT dataset's SMILES_canonical (the structure actually QM-computed
# before the fix) vs the corrected one, matched by IAJD_num — far more reliable than
# the ledger, whose 'old' PE-Gallic SMILES are the malformed %20 strings.
deltas = []
PRE = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.PRE_AUDIT.xlsx"
CUR = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
if PRE.exists():
    pre = pd.read_excel(PRE, sheet_name=0)
    cur = pd.read_excel(CUR, sheet_name=0)
    pre_smi = {r["IAJD_num"]: r.get("SMILES_canonical") for _, r in pre.iterrows() if "IAJD_num" in pre.columns}
    for _, r in cur.iterrows():
        iid = r.get("IAJD_num")
        old_s, new_s = pre_smi.get(iid), r.get("SMILES_canonical")
        if not isinstance(old_s, str) or not isinstance(new_s, str):
            continue
        if canon(old_s) == canon(new_s):
            continue  # structure unchanged by the audit
        q_old, q_new = qm_of(old_s), qm_of(new_s)
        if q_old and q_new:
            d = {"IAJD": int(iid) if pd.notna(iid) else None}
            for c in QM_COLS:
                if np.isfinite(q_old.get(c, np.nan)) and np.isfinite(q_new.get(c, np.nan)):
                    d[c] = round(q_new[c] - q_old[c], 3)
            deltas.append(d)
out["old_to_new_qm_delta"] = {"n_compounds": len(deltas), "per_compound": deltas}
if deltas:
    print(f"\n[old->new QM] {len(deltas)} corrected compounds have BOTH old+new QM. Mean |Δ| per descriptor:")
    for c in QM_COLS:
        vals = [abs(d[c]) for d in deltas if c in d]
        if vals:
            print(f"  {c:20s} mean|Δ|={np.mean(vals):.3f}  max|Δ|={np.max(vals):.3f}  (n={len(vals)})")

(ROOT / "audit_work/qm_findings.json").write_text(json.dumps(out, indent=2, default=str))
print(f"\nWrote audit_work/qm_findings.json")
