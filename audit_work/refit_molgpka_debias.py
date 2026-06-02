"""refit_molgpka_debias.py — refresh molgpka_debias_models.joblib on the CORRECTED data.

train_pka_v92 refits the per-family MolGpKa debias per-fold for its LOO blend (so the
LOO metric is already honest), but it LOADS this standalone .joblib and EMBEDS it in
pka_v92_bundle for INFERENCE. The shipped file was fit pre-audit (05-30) on the old
molgpka/SMILES, so inference applied a stale per-family slope/intercept. This re-fits it
on the fresh molgpka_preds + corrected pKa (audit-flagged rows excluded, same 255-row
subset/order as iajd_pka_v52.build_bundle), so re-running train_pka_v92 embeds a fresh one.

Structure matches what train_pka_v92.head_molgpka_loo expects: {family: {slope, intercept, n}}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LinearRegression
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
PKA_XLSX = ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
MOLGPKA = ROOT / "IAJD_master/bundles_caches/molgpka_preds.npy"
OUT = ROOT / "IAJD_master/bundles_caches/molgpka_debias_models.joblib"

df = pd.read_excel(PKA_XLSX, sheet_name=0)
if "audit_status" in df.columns:                       # identical filter to build_bundle
    df = df[~df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)].reset_index(drop=True)
mp = np.load(MOLGPKA)
assert len(mp) == len(df), f"molgpka ({len(mp)}) != filtered pKa rows ({len(df)})"

y = df["pKa"].astype(float).to_numpy()
fam = df["family"].astype(str).to_numpy()
models = {}
for f in sorted(set(fam)):
    m = (fam == f) & np.isfinite(mp) & np.isfinite(y)
    if m.sum() >= 4:
        lr = LinearRegression().fit(mp[m].reshape(-1, 1), y[m])
        models[f] = {"slope": float(lr.coef_[0]), "intercept": float(lr.intercept_), "n": int(m.sum())}
        print(f"  {f:22s} n={int(m.sum()):3d}  slope={lr.coef_[0]:+.3f} intercept={lr.intercept_:+.3f}")
    else:
        print(f"  {f:22s} n={int(m.sum()):3d}  (skipped, <4)")
joblib.dump(models, OUT)
print(f"Saved {OUT.name}: {len(models)} families")
