"""
run_md_continuous.py — runs MARTINI 3 self-assembly MD on a series of IAJDs,
one at a time, until interrupted or until every unique SMILES in the bioact
training set has a md_cache row.

Designed to be left running in the background. Picks the next compound in a
deterministic order (family-balanced) so we get diversity in the cache early.

Honest no-proxy: every run uses real GROMACS, real MARTINI 3 FF, real genion
for charged states. Failures are logged with a non-NaN audit but the cache
entry stays NaN — XGBoost handles via default branch.

Usage:
  python run_md_continuous.py                        # production: 256 mols × 2 µs
  python run_md_continuous.py --quick                # smoke runs: 16 mols × 20 ns
  python run_md_continuous.py --limit 5              # stop after 5 new entries
  python run_md_continuous.py --families GA-Tris PE-Tris  # filter
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
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from iajd_grammar import decompose_row
from precompute_md import run_one_compound, _canonical_smi, _save, MD_KEYS
from martini.run_selfassembly import gromacs_available

BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
CACHE_DIR = ROOT / "IAJD_master/bundles_caches/physics"
CACHE_CSV = CACHE_DIR / "md_cache.csv"
AUDIT_JSON = ROOT / "md_audit.json"
PHYSICS_WORK = ROOT / "physics_cache" / "md_runs"


def family_balanced_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort so we sample one row per family in round-robin order — gets
    diversity into the cache early instead of finishing one family first."""
    df = df.copy()
    df["_family_clean"] = df["family"].fillna("Unknown")
    out = []
    by_fam = {f: g.reset_index(drop=True) for f, g in df.groupby("_family_clean")}
    maxlen = max(len(g) for g in by_fam.values())
    for i in range(maxlen):
        for f in sorted(by_fam):
            if i < len(by_fam[f]):
                out.append(by_fam[f].iloc[i])
    return pd.DataFrame(out).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-scale: 16 mols × 20 ns. Default is "
                              "production: 256 mols × 2 µs.")
    parser.add_argument("--n-molecules", type=int, default=None)
    parser.add_argument("--prod-ns", type=float, default=None)
    parser.add_argument("--timeout-s", type=int, default=86400)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--seed-neutral", type=int, default=42)
    parser.add_argument("--seed-prot", type=int, default=43)
    parser.add_argument("--retrain-every", type=int, default=5,
                        help="After every N new cache rows, retrain qmmd head.")
    parser.add_argument("--keep-trajectories", action="store_true",
                        help="Keep raw MD trajectories (default: delete after "
                             "observables extracted, to bound disk use).")
    args = parser.parse_args()

    if args.quick:
        n_molecules = args.n_molecules or 16
        prod_ns = args.prod_ns or 20.0
    else:
        n_molecules = args.n_molecules or 256
        prod_ns = args.prod_ns or 2000.0

    if not gromacs_available():
        print("ERROR: gmx not found.", file=sys.stderr)
        return 2

    print(f"[run_md_continuous] starting at {time.ctime()}")
    print(f"  config: n_molecules={n_molecules}, prod_ns={prod_ns}")
    print(f"  workdir: {PHYSICS_WORK}")

    df = pd.read_excel(BIOACT_XLSX)
    df["SMILES_canonical"] = df["SMILES_canonical"].fillna("").map(_canonical_smi)
    df = df[df["SMILES_canonical"].str.len() > 0].reset_index(drop=True)
    df = (df.sort_values("row_id")
            .drop_duplicates("SMILES_canonical", keep="first")
            .reset_index(drop=True))
    if args.families:
        df = df[df["family"].isin(args.families)].reset_index(drop=True)
    df = family_balanced_order(df)
    print(f"  {len(df)} unique canonical SMILES queued (family-balanced order)")

    # Skip already cached.
    existing: dict = {}
    if CACHE_CSV.exists():
        try:
            cached_df = pd.read_csv(CACHE_CSV)
            for _, row in cached_df.iterrows():
                existing[row["smiles_canonical"]] = dict(row)
        except Exception:
            pass
    print(f"  {len(existing)} entries already in md_cache")

    audit: list = []
    if AUDIT_JSON.exists():
        try:
            audit = json.loads(AUDIT_JSON.read_text())
        except json.JSONDecodeError:
            audit = []

    new_count = 0

    def _save_safe(new_records: list):
        """Re-read disk cache before writing so concurrent writers don't
        clobber each other's entries."""
        from precompute_md import _save as _md_save
        disk: dict = {}
        if CACHE_CSV.exists():
            try:
                for _, r in pd.read_csv(CACHE_CSV).iterrows():
                    disk[r["smiles_canonical"]] = dict(r)
            except Exception:
                pass
        # Merge in our new records (overrides disk for matching SMILES so
        # we capture the post-quality-gate values).
        for rec in new_records:
            disk[rec["smiles_canonical"]] = rec
        _md_save(list(disk.values()), audit)
        return disk

    for _, row in df.iterrows():
        smi = row["SMILES_canonical"]
        if smi in existing:
            continue
        # Decompose check
        seed = decompose_row(row.to_dict())
        if seed is None:
            audit.append({"smiles_canonical": smi, "error": "decompose_failed_skipped"})
            continue
        # Skip compounds with ACTIVELY running MD (avoid colliding with
        # other precompute_md processes started in parallel). Stale dirs
        # from killed runs get cleaned + retried.
        cmpd_dir = PHYSICS_WORK / smi.replace("/", "_")[:64]
        if cmpd_dir.exists():
            in_progress_state = None
            for state in ("neutral", "prot"):
                state_dir = cmpd_dir / state
                if not (state_dir / "prod.tpr").exists():
                    continue
                if (state_dir / "prod.gro").exists():
                    continue
                # prod.tpr without prod.gro — was it touched recently?
                log_p = state_dir / "prod.log"
                fresh = False
                if log_p.exists():
                    age = time.time() - log_p.stat().st_mtime
                    if age < 120:  # log written in the last 2 minutes → active
                        fresh = True
                if fresh:
                    in_progress_state = state
                    break
            if in_progress_state is not None:
                print(f"  skip {smi[:30]}: {in_progress_state} state is "
                      f"actively running elsewhere")
                continue
            # Otherwise — stale dir from a killed run. Wipe and retry.
            if any((cmpd_dir / state / "prod.tpr").exists()
                    and not (cmpd_dir / state / "prod.gro").exists()
                    for state in ("neutral", "prot")):
                import shutil as _sh
                print(f"  cleaning stale dir: {cmpd_dir.name[:40]}")
                _sh.rmtree(cmpd_dir, ignore_errors=True)
        print(f"\n[{new_count + 1}] {row.get('IAJD_id', smi[:30])} "
              f"({row.get('family', '?')})")
        t0 = time.time()
        try:
            rec = run_one_compound(
                smi, row.to_dict(),
                workroot=PHYSICS_WORK,
                n_molecules=n_molecules,
                prod_ns=prod_ns,
                seed_neutral=args.seed_neutral,
                seed_prot=args.seed_prot,
                timeout_s=args.timeout_s,
                keep_workdir=args.keep_trajectories,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            rec = {"smiles_canonical": smi,
                   **{k: float("nan") for k in MD_KEYS},
                   "md_error": f"exception:{type(exc).__name__}:{exc}"}
        elapsed = time.time() - t0
        rec_audit = rec.pop("_audit", None) if isinstance(rec, dict) else None
        audit.append({"smiles_canonical": smi, "elapsed_s": round(elapsed, 1),
                       "error": rec.get("md_error"), "audit": rec_audit})
        new_count += 1
        existing[smi] = rec
        existing = _save_safe([rec])
        print(f"  done in {elapsed/60:.1f} min  err={rec.get('md_error')}")
        # Auto-retrain qmmd head every N new entries.
        if args.retrain_every and new_count % args.retrain_every == 0:
            print("  → auto-retraining qmmd head…")
            try:
                import subprocess
                subprocess.run([sys.executable, "adaptive_stacker.py",
                                  "--build-qmmd-block"],
                                 cwd=str(ROOT), timeout=600)
                subprocess.run([sys.executable, "train_qmmd_head_only.py"],
                                 cwd=str(ROOT), timeout=600)
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"  retrain skipped: {exc}")
        if args.limit and new_count >= args.limit:
            break

    print(f"\n[run_md_continuous] done. {new_count} new MD records produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
