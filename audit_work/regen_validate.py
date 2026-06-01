"""regen_validate.py — end-to-end validation of the 2026-06-01 correction regen.

Checks dataset correctness, cache coverage, model artifact alignment, before/after
metrics, and runs a live predict_v14_real smoke test. Prints a structured report;
exit code 0 if all hard invariants hold.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "IAJD_master/bundles_caches"
BK = ROOT / "audit_work/pre_regen_backup_20260601"
fails = []


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'XX '}] {msg}")
    if not cond:
        fails.append(msg)


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


print("=" * 72); print("1. DATASETS"); print("=" * 72)
bio = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
pka = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx")
mism = sum(1 for _, r in bio.iterrows()
           if canon(r["SMILES"]) and canon(r["SMILES_canonical"])
           and canon(r["SMILES"]) != canon(r["SMILES_canonical"]))
ok(mism == 0, f"bioact SMILES_canonical consistent with SMILES (mismatches={mism})")
ok("audit_status" in bio.columns and "audit_status" in pka.columns, "audit_status present in both")
bflag = int(bio["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False).sum())
pflag = int(pka["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False).sum())
print(f"      bioact rows={len(bio)} flagged={bflag} | pKa rows={len(pka)} flagged={pflag}")
for f in ["IAJD_pKa_v21_final.PRE_AUDIT.xlsx", "IAJD_Bioact_v13_clean.PRE_AUDIT.xlsx"]:
    ok((ROOT / "IAJD_master/datasets" / f).exists(), f"pre-audit backup exists: {f}")

# training SMILES (non-flagged, has flux)
bf = bio[~bio["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)]
bf = bf[bf["log10_flux_total"].notna()]
train_smis = []
seen = set()
for s in bf["SMILES_canonical"]:
    c = canon(s)
    if c and c not in seen:
        seen.add(c); train_smis.append(c)
print(f"      training canonical SMILES (non-flagged, has flux): {len(train_smis)}")

print("=" * 72); print("2. EXTERNAL CACHES (real, no proxy)"); print("=" * 72)
admet = json.load(open(C / "admet_cache_v13.json"))
lion = json.load(open(C / "lion_cache_v13.json"))
a_miss = [s for s in train_smis if s not in admet]
a_zero = [s for s in train_smis if s in admet and np.allclose(np.array(admet[s], float), 0)]
l_miss = [s for s in train_smis if s not in lion]
l_zero = [s for s in train_smis if s in lion and np.allclose(np.array(lion[s], float)[:6], 0)]
ok(len(a_miss) == 0, f"ADMET covers all training SMILES (missing={len(a_miss)})")
ok(len(a_zero) == 0, f"ADMET no degenerate all-zero vectors (zero={len(a_zero)})")
ok(len(l_miss) == 0, f"LiON covers all training SMILES (missing={len(l_miss)})")
ok(len(l_zero) == 0, f"LiON no degenerate tissue-zero vectors (zero={len(l_zero)})")

print("=" * 72); print("3. BIOACT BUNDLE + TRAIN ARRAYS (alignment)"); print("=" * 72)
b = pickle.load(open(C / "bioact_v14_bundle.pkl", "rb"))
n = len(b["smis_train"])
print(f"      new bundle: version={b.get('version')} n={n} n_features={getattr(b.get('direct_model_full'),'n_features_in_','?')}")
ok(n == len(train_smis), f"bundle n ({n}) == training SMILES ({len(train_smis)})")
for name, exp_cols in [("agile_embeddings_v14_train.npy", 512), ("cpp_features_v14_train.npy", 23),
                        ("qmmd_features_v14_train.npy", 14)]:
    p = ROOT / name
    if p.exists():
        a = np.load(p)
        ok(a.shape == (n, exp_cols), f"{name} shape {a.shape} == ({n},{exp_cols})")
    else:
        ok(False, f"{name} MISSING")
loo = np.load(ROOT / "bioact_loo_components.npz", allow_pickle=True)
ok(len(loo["direct"]) == n, f"loo_components aligned ({len(loo['direct'])}=={n})")

# before/after metrics
try:
    old = pickle.load(open(BK / "bioact_v14_bundle.pkl", "rb"))
    om = old["metrics"].get("v14_pooled_mae"); nm = b["metrics"].get("v14_pooled_mae")
    print(f"      bioact v14 pooled MAE:  old(n={len(old['smis_train'])})={om:.4f}  ->  new(n={n})={nm:.4f}")
except Exception as e:
    print(f"      (old bundle compare skipped: {e})")

print("=" * 72); print("4. pKa MODEL"); print("=" * 72)
import joblib
pkb = joblib.load(C / "pka_v92_bundle.joblib")
w = pkb["weights"]; met = pkb["metrics"]
print(f"      weights: {', '.join(f'{k}={v:.3f}' for k,v in w.items())}")
print(f"      blend LOO MAE: {met['loo_mae_blend']:.4f}  (n_train={met['n_train']})")
mp = np.load(C / "molgpka_preds.npy")
ok(len(mp) == met["n_train"], f"molgpka_preds aligned with pKa bundle ({len(mp)}=={met['n_train']})")

print("=" * 72); print("5. QM CACHE COVERAGE (overnight)"); print("=" * 72)
qm = set(pd.read_csv(C / "physics/qm_cache.csv")["smiles_canonical"].astype(str))
qmiss = [s for s in train_smis if s not in qm]
print(f"      QM real coverage of training SMILES: {len(train_smis)-len(qmiss)}/{len(train_smis)} "
      f"(emulator-filled until overnight QM lands: {len(qmiss)})")

print("=" * 72); print("6. predict_v14_real SMOKE TEST"); print("=" * 72)
try:
    sys.path.insert(0, str(ROOT))
    from importlib import import_module
    pv = import_module("IAJD_master.code.predict_v14_real") if False else None
    sys.path.insert(0, str(ROOT / "IAJD_master" / "code"))
    import predict_v14_real as PV
    r = PV.predict("CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC",
                   auto_compute=True, return_neighbors=2)
    ok("log10_flux_total" in r and r["log10_flux_total"] is not None,
       f"predict returns flux={r.get('log10_flux_total'):.3f} family={r.get('family')} "
       f"block_B_real={r.get('block_B_real')}")
except Exception as e:
    ok(False, f"predict_v14_real smoke test raised: {type(e).__name__}: {e}")

print("=" * 72)
print(f"VALIDATION: {'ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILURE(S)'}")
for f in fails:
    print(f"  FAIL: {f}")
print("=" * 72)
sys.exit(1 if fails else 0)
