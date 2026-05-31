"""
precompute_qm_parallel.py — PARALLEL QM precompute.

Runs the IDENTICAL real computation as precompute_qm.py
(qm_descriptors.compute_qm_descriptors → GFN2-xTB, n_confs=3, timeout 900 s, no
proxy) but across a pool of worker processes, so the 10-core machine isn't left
half-idle behind one sequential worker.

No-clobber design: the MAIN process is the SOLE writer of qm_cache.csv and does
an atomic replace (write .tmp → os.replace); the workers only compute and return
their record. So concurrent readers (the retrain watcher, autosave) never see a
half-written cache, and there is no write race.

Same canonical-SMILES dedup, same cache schema (smiles_canonical + 8 QM_KEYS,
error → audit only) as precompute_qm.py. Resumable: skips SMILES already cached.

  QM_THREADS=2 python precompute_qm_parallel.py --workers 4
"""
from __future__ import annotations
import os

# Cap xtb's OpenMP threads PER worker so N workers × T threads fits the machine.
# Must be set before numpy / any xtb subprocess inherits the env; children
# spawned by ProcessPoolExecutor inherit this too.
_THREADS = os.environ.get("QM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from qm_descriptors import compute_qm_descriptors, QM_KEYS, _xtb_available

BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
CACHE_DIR = ROOT / "IAJD_master/bundles_caches/physics"
CACHE_CSV = CACHE_DIR / "qm_cache.csv"
CACHE_PARQUET = CACHE_DIR / "qm_cache.parquet"
AUDIT_JSON = ROOT / "qm_audit.json"

N_CONFS = 3
TIMEOUT_S = 900


def _canonical_smi(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else ""


def _work(smi: str):
    """Worker process: the real QM computation. Returns (smi, record, elapsed, error)."""
    t = time.time()
    try:
        res = compute_qm_descriptors(smi, n_confs=N_CONFS, timeout_s=TIMEOUT_S)
    except Exception as exc:  # keep the pool alive on any single-compound failure
        res = {k: float("nan") for k in QM_KEYS}
        res["qm_error"] = f"exception:{type(exc).__name__}:{exc}"
    rec = {"smiles_canonical": smi, **{k: res.get(k, float("nan")) for k in QM_KEYS}}
    return smi, rec, round(time.time() - t, 1), res.get("qm_error")


def _atomic_write(records: list, audit: list) -> None:
    """Sole-writer, atomic cache + audit write (tmp → os.replace)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    tmp = CACHE_CSV.with_name(CACHE_CSV.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, CACHE_CSV)
    try:
        tmpp = CACHE_PARQUET.with_name(CACHE_PARQUET.name + ".tmp")
        df.to_parquet(tmpp, index=False)
        os.replace(tmpp, CACHE_PARQUET)
    except (ImportError, ValueError):
        pass
    tmpa = AUDIT_JSON.with_name(AUDIT_JSON.name + ".tmp")
    tmpa.write_text(json.dumps(audit, indent=2, default=str))
    os.replace(tmpa, AUDIT_JSON)


def _family_balanced(todo, bioact_xlsx):
    """Reorder `todo` round-robin across families so QM coverage is DIVERSE
    early (option C) — makes the correlation/ablation decisive after ~30-40
    compounds instead of needing full coverage. Falls back to input order if
    family info is unavailable."""
    try:
        df = pd.read_excel(bioact_xlsx)
        fam = {}
        for _, r in df.iterrows():
            c = _canonical_smi(r.get("SMILES_canonical", ""))
            if c:
                fam.setdefault(c, str(r.get("family", "other")))
    except Exception:
        return todo
    buckets = {}
    for s in todo:
        buckets.setdefault(fam.get(s, "other"), []).append(s)
    out = []
    maxlen = max((len(v) for v in buckets.values()), default=0)
    for i in range(maxlen):
        for f in sorted(buckets):
            if i < len(buckets[f]):
                out.append(buckets[f][i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel xtb worker processes.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not _xtb_available():
        print("ERROR: xtb not available (check XTB_CMD / xtb_env).", file=sys.stderr)
        return 2

    df = pd.read_excel(BIOACT_XLSX)
    df["SMILES_canonical"] = df["SMILES_canonical"].fillna("").map(_canonical_smi)
    df = df[df["SMILES_canonical"].str.len() > 0]
    unique = list(dict.fromkeys(df["SMILES_canonical"].tolist()))  # order-preserving dedup

    done = {}
    if CACHE_CSV.exists():
        for _, r in pd.read_csv(CACHE_CSV).iterrows():
            done[r["smiles_canonical"]] = dict(r)
    todo = [s for s in unique if s not in done]
    todo = _family_balanced(todo, BIOACT_XLSX)   # option C: diversity-first order
    if args.limit:
        todo = todo[:args.limit]

    print(f"[parallel-qm] {len(unique)} unique | {len(done)} cached | "
          f"{len(todo)} to compute | workers={args.workers} "
          f"threads/worker={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    if not todo:
        print("[parallel-qm] nothing to do.")
        return 0

    records = list(done.values())
    audit = []
    if AUDIT_JSON.exists():
        try:
            audit = json.loads(AUDIT_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            audit = []

    t0 = time.time()
    n = 0
    nfail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, s): s for s in todo}
        for fut in as_completed(futs):
            smi, rec, elapsed, err = fut.result()
            records.append(rec)
            audit.append({"smiles_canonical": smi, "elapsed_s": elapsed, "error": err})
            n += 1
            if err and err not in (None, "ok"):
                nfail += 1
            nfin = sum(1 for k in QM_KEYS
                       if isinstance(rec.get(k), (int, float)) and np.isfinite(rec.get(k)))
            _atomic_write(records, audit)  # serial, atomic — main is sole writer
            rate = n / (time.time() - t0) * 3600.0
            eta_h = (len(todo) - n) / max(rate, 1e-9)
            print(f"[parallel-qm] {n}/{len(todo)} | {smi[:38]}… "
                  f"{elapsed:.0f}s fin={nfin}/8 err={err} | "
                  f"{rate:.1f}/h ETA~{eta_h:.1f}h fails={nfail}", flush=True)

    print(f"[parallel-qm] COMPLETE: {n} computed, {nfail} failed, "
          f"{(time.time() - t0) / 60:.0f} min total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
