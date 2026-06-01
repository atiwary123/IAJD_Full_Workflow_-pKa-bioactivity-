"""
precompute_qm.py — offline driver that runs qm_descriptors.compute_qm_descriptors
on every canonical SMILES in the bioactivity training set and writes a
shippable cache.

Outputs:
  IAJD_master/bundles_caches/physics/qm_cache.csv      (key = SMILES_canonical)
  IAJD_master/bundles_caches/physics/qm_cache.parquet  (parquet if pyarrow available)
  qm_audit.json                                         (per-compound convergence + errors)

Run plan:
  source .venv/bin/activate
  python precompute_qm.py                # all rows
  python precompute_qm.py --limit 20     # first 20 (sanity check)
  python precompute_qm.py --resume       # skip rows already in cache

Mirrors compute_molgpka_live.py:
  - checkpoints every N compounds
  - per-row try/except so one xtb failure never stops the whole run
  - audit log records every non-OK outcome
  - prints Pearson/Spearman of each QM descriptor vs log10_flux_total so you
    immediately see which features carry signal.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

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


def _canonical_smi(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else ""


def _load_existing_cache() -> dict:
    if not CACHE_CSV.exists():
        return {}
    df = pd.read_csv(CACHE_CSV)
    return {row["smiles_canonical"]: dict(row) for _, row in df.iterrows()}


def _save_cache(records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(CACHE_CSV, index=False)
    try:
        df.to_parquet(CACHE_PARQUET, index=False)
    except (ImportError, ValueError):
        pass


def _save_audit(audit: list[dict]) -> None:
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))


def _print_correlations(df_cache: pd.DataFrame, df_bio: pd.DataFrame) -> None:
    """Print Pearson/Spearman of each QM descriptor vs log10_flux_total."""
    from scipy.stats import pearsonr, spearmanr
    merged = df_bio.merge(df_cache, left_on="SMILES_canonical",
                           right_on="smiles_canonical", how="inner")
    flux = merged["log10_flux_total"].values
    print("\nCorrelation of QM descriptors with log10_flux_total:")
    print(f"{'descriptor':24s} {'r_pearson':>10s} {'rho_spearman':>14s} {'n':>5s}")
    for k in QM_KEYS:
        if k not in merged.columns:
            continue
        v = merged[k].values
        mask = np.isfinite(v) & np.isfinite(flux)
        if mask.sum() < 10:
            print(f"  {k:22s} insufficient data (n={mask.sum()})")
            continue
        r, p = pearsonr(v[mask], flux[mask])
        rho, p2 = spearmanr(v[mask], flux[mask])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
        print(f"  {k:22s} {r:>+10.3f} {rho:>+14.3f} {mask.sum():>5d}  {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for sanity checks).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip SMILES already present in qm_cache.csv.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--n-confs", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--skip-correlations", action="store_true")
    args = parser.parse_args()

    if not _xtb_available():
        print("ERROR: xtb binary not available. "
              "Set XTB_CMD or check the offline conda env.", file=sys.stderr)
        return 2

    print(f"Loading {BIOACT_XLSX.name}…")
    df_bio = pd.read_excel(BIOACT_XLSX)
    # Skip audit-flagged rows (suspect/unresolved SMILES) so xTB never spends
    # hours optimizing a known-wrong structure (dataset audit §4d, 2026-06-01).
    if "audit_status" in df_bio.columns:
        flagged = df_bio["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
        if int(flagged.sum()):
            print(f"  skipping {int(flagged.sum())} audit-flagged rows")
        df_bio = df_bio[~flagged].reset_index(drop=True)
    df_bio["SMILES_canonical"] = df_bio["SMILES_canonical"].fillna("").map(_canonical_smi)
    df_bio = df_bio[df_bio["SMILES_canonical"].str.len() > 0].reset_index(drop=True)
    print(f"  {len(df_bio)} rows with parseable SMILES")

    # De-duplicate by canonical SMILES — the cache is keyed on chemistry,
    # not on row_id (multiple bioactivity rows can share the same compound).
    unique_smis = df_bio["SMILES_canonical"].drop_duplicates().tolist()
    print(f"  {len(unique_smis)} unique canonical SMILES")

    if args.limit:
        unique_smis = unique_smis[:args.limit]
        print(f"  --limit {args.limit}: processing first {len(unique_smis)}")

    existing = _load_existing_cache() if args.resume else {}
    if existing:
        print(f"  loaded {len(existing)} cached records (--resume)")

    records: list[dict] = list(existing.values())
    audit: list[dict] = []
    if (ROOT / "qm_audit.json").exists():
        try:
            audit = json.loads((ROOT / "qm_audit.json").read_text())
        except (json.JSONDecodeError, OSError):
            audit = []

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0
    for i, smi in enumerate(unique_smis):
        if args.resume and smi in existing:
            n_skipped += 1
            continue
        t_start = time.time()
        try:
            res = compute_qm_descriptors(
                smi, n_confs=args.n_confs, timeout_s=args.timeout_s,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            res = {k: float("nan") for k in QM_KEYS}
            res["qm_error"] = f"exception:{type(exc).__name__}:{exc}"
        elapsed = time.time() - t_start

        record = {"smiles_canonical": smi, **{k: res.get(k, float("nan")) for k in QM_KEYS}}
        records.append(record)
        audit.append({
            "smiles_canonical": smi,
            "elapsed_s": round(elapsed, 1),
            "error": res.get("qm_error"),
        })
        if res.get("qm_error") and res["qm_error"] not in (None, "ok"):
            n_failed += 1
        n_done += 1

        if n_done % args.checkpoint_every == 0:
            _save_cache(records)
            _save_audit(audit)
            total_elapsed = time.time() - t0
            rate = n_done / total_elapsed if total_elapsed > 0 else 0
            remaining = (len(unique_smis) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(unique_smis)}] last={elapsed:.0f}s "
                  f"rate={rate*60:.1f}/min  fails={n_failed}  "
                  f"ETA={remaining/60:.0f}min", flush=True)
        else:
            print(f"  [{i+1}/{len(unique_smis)}] {smi[:60]}… {elapsed:.0f}s "
                  f"err={res.get('qm_error')}", flush=True)

    _save_cache(records)
    _save_audit(audit)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min: "
          f"{n_done} processed, {n_skipped} skipped, {n_failed} failed")
    print(f"  cache: {CACHE_CSV}")
    print(f"  audit: {AUDIT_JSON}")

    if not args.skip_correlations:
        df_cache = pd.read_csv(CACHE_CSV)
        _print_correlations(df_cache, df_bio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
