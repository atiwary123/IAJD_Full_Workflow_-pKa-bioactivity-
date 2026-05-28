"""
regen_pka_cache_robust.py — rebuild v11_pka_flux/predicted_pka_cache.csv with:
  - LOO predictions (from preds_v91_final.npy) for SMILES in the pKa training table
  - Live v9.1 calls (with timeout + subprocess isolation) for SMILES NOT in pKa table

For training-set compounds we KNOW the LOO prediction is the honest model output
(no proxy). For new SMILES we do a real v9.1 call which internally runs live
MolGpKa + per-family debias. If a particular compound's live call exceeds the
timeout (e.g. ETKDGv3 conformer generation gets stuck), we record NaN and move
on — the bioact pipeline downstream will then fall back to its own live v9.1
path or skip the row.
"""
from __future__ import annotations
import os, sys, json, signal, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

PKA_XLSX = ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
PRED_LOO = ROOT / "preds_v91_final.npy"
OUT_CACHE = ROOT / "v11_pka_flux/predicted_pka_cache.csv"

CALL_TIMEOUT_SEC = 60   # per-compound live v9.1 timeout


class TimeoutError_(Exception): pass


def _alarm(signum, frame):
    raise TimeoutError_("timeout")


def _canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else ""


def main():
    print(f"[regen] loading tables …", flush=True)
    bio = pd.read_excel(BIO_XLSX)
    pka = pd.read_excel(PKA_XLSX, sheet_name="Dataset")
    loo = np.load(PRED_LOO)
    print(f"  bio={bio.shape}  pka={pka.shape}  loo={loo.shape}", flush=True)
    assert len(loo) == len(pka), f"LOO ({len(loo)}) must match pKa table ({len(pka)})"

    pka_canon = pka["SMILES"].map(_canon)
    loo_map = {}
    for i, canon in enumerate(pka_canon):
        if canon:
            loo_map[canon] = (float(loo[i]),
                              float(pka.iloc[i]["pKa"]) if pd.notna(pka.iloc[i]["pKa"]) else np.nan,
                              str(pka.iloc[i]["family"]))
    print(f"  loo_map: {len(loo_map)} canonical SMILES", flush=True)

    # Load v9.1 once for the live path
    print(f"[regen] loading v9.1 bundle (one-time, ~30-60s)…", flush=True)
    os.chdir(ROOT / "IAJD_master/datasets")
    from iajd_pka_v91 import load_v91_bundle, predict_pka_v91
    bundle = load_v91_bundle(
        xlsx_path="IAJD_pKa_v21_final.xlsx",
        debias_path="molgpka_debias_models.joblib",
        molgpka_npy="molgpka_preds.npy",
    )
    os.chdir(ROOT)
    print(f"  v9.1 bundle loaded", flush=True)

    out = []
    n_loo = n_full = n_timeout = 0
    t0 = time.time()
    for idx, row in bio.iterrows():
        smi = row.get("SMILES_canonical") or row.get("SMILES")
        canon = _canon(smi)
        fam_table = row.get("family")
        rec = {"row_id": row.get("row_id"), "IAJD_id": row.get("IAJD_id"),
                "family_table": fam_table, "canonical_smiles": canon}
        if canon in loo_map:
            pred, meas, fam_v21 = loo_map[canon]
            rec.update({"predicted_pKa": pred, "measured_pKa_v21": meas,
                        "family_v21": fam_v21, "pka_source": "v21_loo"})
            n_loo += 1
        else:
            # Live v9.1 inference for cache-miss SMILES
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(CALL_TIMEOUT_SEC)
            try:
                r = predict_pka_v91(canon, bundle, family_hint=str(fam_table),
                                     return_diagnostics=False)
                signal.alarm(0)
                if "error" in r:
                    rec.update({"predicted_pKa": np.nan,
                                "pka_source": f"error:{r['error'][:40]}"})
                else:
                    rec.update({"predicted_pKa": float(r["pKa_pred"]),
                                "measured_pKa_v21": np.nan,
                                "family_v21": r.get("family_assigned"),
                                "pka_source": "v91_full_bundle_live"})
                n_full += 1
                print(f"  [{idx+1}] LIVE pKa={rec.get('predicted_pKa')}  fam={fam_table}  smi=...{canon[-30:]}", flush=True)
            except TimeoutError_:
                signal.alarm(0)
                rec.update({"predicted_pKa": np.nan,
                            "pka_source": f"timeout_{CALL_TIMEOUT_SEC}s"})
                n_timeout += 1
                print(f"  [{idx+1}] TIMEOUT after {CALL_TIMEOUT_SEC}s  smi=...{canon[-30:]}", flush=True)
            except Exception as exc:
                signal.alarm(0)
                rec.update({"predicted_pKa": np.nan,
                            "pka_source": f"exception:{type(exc).__name__}:{str(exc)[:40]}"})
                print(f"  [{idx+1}] EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        out.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"  [progress] {idx+1}/{len(bio)} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(OUT_CACHE, index=False)
    print(f"\n[regen] saved {OUT_CACHE.name}", flush=True)
    print(f"  {n_loo} LOO + {n_full} live + {n_timeout} timeout = {len(df)} rows", flush=True)
    print(f"  total time: {time.time()-t0:.1f}s", flush=True)

    nov = df[df["IAJD_id"].isin([f"IAJD_{n}" for n in [347,348,365,366,367,369,372,373]])]
    print(f"\nNovels:\n{nov[['IAJD_id','family_table','predicted_pKa','pka_source']].to_string(index=False)}",
           flush=True)


if __name__ == "__main__":
    main()
