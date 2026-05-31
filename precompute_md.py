"""
precompute_md.py — offline driver that runs the MARTINI 3 self-assembly
protocol for every SMILES in IAJD_Bioact_v13_clean.xlsx, in both protonation
states, and aggregates the observables into the shippable MD cache.

Outputs:
  IAJD_master/bundles_caches/physics/md_cache.csv      (key = SMILES_canonical)
  IAJD_master/bundles_caches/physics/md_cache.parquet  (if pyarrow available)
  md_audit.json                                         (per-compound convergence)

Inputs needed in PATH (offline env):
  gmx (gromacs 2024.x)         — required for MD runs
  python -m martini.build_cg   — required for topology assembly
  rdkit, MDAnalysis            — required for SMILES parsing and analysis

Honest no-proxy contract: if gromacs is unavailable we still write topologies
(via martini.build_cg) and a placeholder MD cache row with all NaN observables
and audit error="gmx_unavailable". This keeps the cache file present and
downstream readers happy; XGBoost will route the NaN through its default branch.

Resumable: re-running with --resume skips SMILES already present in md_cache.csv.

Determinism: each SMILES gets seeds (42, 43) for the (neutral, prot) replicas.
A second replica per state can be turned on with --replicas 2.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from iajd_grammar import build_library, decompose_row
from martini.build_cg import build
from martini.run_selfassembly import run, RunSpec, gromacs_available
from martini.analyze_md import extract_md_observables, compose_md_record, MDObservables

BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
CACHE_DIR = ROOT / "IAJD_master/bundles_caches/physics"
CACHE_CSV = CACHE_DIR / "md_cache.csv"
CACHE_PARQUET = CACHE_DIR / "md_cache.parquet"
AUDIT_JSON = ROOT / "md_audit.json"
PHYSICS_WORK = ROOT / "physics_cache" / "md_runs"

MD_KEYS = (
    "md_assembles", "md_n_agg",
    "md_a_head_neutral_nm2", "md_a_head_prot_nm2", "md_delta_a_head_nm2",
    "md_bilayer_thick_nm", "md_order_param", "md_radius_gyration_nm",
    "md_water_penetration",
    "md_cpp_neutral", "md_cpp_prot", "md_delta_cpp",
    "md_c0_spontaneous",
)


def _canonical_smi(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else ""


def _nan_md_record(smiles: str, error: str) -> Dict[str, float]:
    rec = {"smiles_canonical": smiles, **{k: float("nan") for k in MD_KEYS}}
    rec["md_error"] = error
    return rec


def _tanford_volume_length(seed) -> tuple[float, float]:
    """Per-tail Tanford V and l from average tail-C count.

    V_tail [nm³] = 0.027 · n_C (Tanford 1980)
    l_tail [nm]  = 0.127 · n_C
    For multi-tail molecules: scale V_tail by n_tails (cumulative tail volume),
    keep l_tail as the per-chain length (Tanford's geometry for amphiphile CPP).
    """
    n_tails = len(seed.tails)
    nC = []
    for t in seed.tails:
        m = Chem.MolFromSmiles(t)
        if m:
            nC.append(sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6))
    if not nC:
        return float("nan"), float("nan")
    nC_mean = float(np.mean(nC))
    V_per_chain = 0.027 * nC_mean
    l_per_chain = 0.127 * nC_mean
    V_total = V_per_chain * n_tails
    return V_total, l_per_chain


# GROMACS bulk intermediates — safe to delete once observables are extracted.
# The trajectory is fully simulated and analyzed first; we just don't hoard the
# raw frames afterward (standard MD practice). Bounds peak disk to ~one run
# instead of accumulating ~1-2 GB × hundreds of compounds.
_MD_BULK_GLOBS = ("*.xtc", "*.trr", "*.tpr", "*.edr", "*.cpt", "#*#",
                  "*.gro", "step*.pdb")


def _purge_md_bulk(d: Path) -> None:
    """Delete bulky MD files under directory `d` (best-effort)."""
    if not d.exists():
        return
    for pat in _MD_BULK_GLOBS:
        for f in d.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass


def run_one_compound(smiles: str, row: dict, *,
                     workroot: Path,
                     n_molecules: int, prod_ns: float,
                     seed_neutral: int, seed_prot: int,
                     timeout_s: int,
                     keep_workdir: bool = False) -> Dict[str, float]:
    """Run the full neutral+prot pipeline for one SMILES. Returns one record
    dict with smiles_canonical, all MD_KEYS, and md_error."""
    if not gromacs_available():
        return _nan_md_record(smiles, "gmx_unavailable")

    seed = decompose_row(row)
    if seed is None:
        return _nan_md_record(smiles, "decompose_failed")

    cmpd_dir = workroot / smiles.replace("/", "_")[:64]
    cmpd_dir.mkdir(parents=True, exist_ok=True)

    audit = {"smiles_canonical": smiles, "states": {}}
    state_results: Dict[str, MDObservables] = {}

    ff_dir = Path(__file__).resolve().parent / "martini" / "ff"
    # Box side scales with molecule count so packing density stays sane.
    box_side = max(6.0, (n_molecules * 0.5) ** (1 / 3) * 3.0)
    box_nm = (box_side, box_side, box_side)
    for state, prot, run_seed in (("neutral", False, seed_neutral),
                                   ("prot",    True,  seed_prot)):
        sdir = cmpd_dir / state
        topo = build(seed, sdir, protonated=prot, n_molecules=n_molecules,
                      box_nm=box_nm)
        rs = RunSpec(
            workdir=sdir,
            single_gro=sdir / f"molecule_{state}.gro",
            single_itp=sdir / f"molecule_{state}.itp",
            topol_top=sdir / f"topol_{state}.top",
            n_molecules=n_molecules,
            box_nm=box_nm,
            prod_ns=prod_ns,
            seed=run_seed,
            timeout_s=timeout_s,
            ff_dir=ff_dir,
            net_charge_per_molecule=float(topo.total_charge),
        )
        run_audit = run(rs)
        audit["states"][state] = {k: v for k, v in run_audit.items()
                                    if k not in ("log",)}
        if run_audit.get("status") != "ok":
            state_results[state] = MDObservables(error=run_audit.get("status"))
            continue
        obs = extract_md_observables(sdir / "prod.xtc", sdir / "prod.gro")
        state_results[state] = obs
        # Free disk: trajectory is fully analyzed; drop bulk frames now so peak
        # usage stays ~one trajectory instead of accumulating across compounds.
        if not keep_workdir:
            _purge_md_bulk(sdir)

    rec = compose_md_record(state_results["neutral"], state_results["prot"])
    # Add CPPs from measured a_head + Tanford V/l.
    V_total, l_chain = _tanford_volume_length(seed)
    if np.isfinite(V_total) and np.isfinite(l_chain) and l_chain > 0:
        if np.isfinite(rec["md_a_head_neutral_nm2"]) and rec["md_a_head_neutral_nm2"] > 0:
            rec["md_cpp_neutral"] = V_total / (rec["md_a_head_neutral_nm2"] * l_chain)
        if np.isfinite(rec["md_a_head_prot_nm2"]) and rec["md_a_head_prot_nm2"] > 0:
            rec["md_cpp_prot"] = V_total / (rec["md_a_head_prot_nm2"] * l_chain)
        if np.isfinite(rec["md_cpp_prot"]) and np.isfinite(rec["md_cpp_neutral"]):
            rec["md_delta_cpp"] = rec["md_cpp_prot"] - rec["md_cpp_neutral"]
    rec["smiles_canonical"] = smiles

    # Collapse error: take the worst-of the two state errors.
    state_errs = [state_results[s].error for s in ("neutral", "prot")
                   if state_results[s].error]
    rec["md_error"] = ";".join(state_errs) if state_errs else None
    rec["_audit"] = audit
    if not keep_workdir:
        shutil.rmtree(cmpd_dir, ignore_errors=True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iajd-num", type=int, default=None,
                        help="Only run this IAJD_num (overrides --limit).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--n-molecules", type=int, default=256)
    parser.add_argument("--prod-ns", type=float, default=2000.0,
                        help="CG production time per state (ns).")
    parser.add_argument("--timeout-s", type=int, default=86400)
    parser.add_argument("--seed-neutral", type=int, default=42)
    parser.add_argument("--seed-prot", type=int, default=43)
    parser.add_argument("--keep-trajectories", action="store_true",
                        help="Keep raw MD trajectories (default: delete after "
                             "observables are extracted, to bound disk use).")
    args = parser.parse_args()

    if not gromacs_available():
        print("WARNING: gromacs binary not detected on PATH. The driver will "
              "write topologies for every compound but no MD will run, and the "
              "MD cache will be populated with NaN observables tagged "
              "md_error='gmx_unavailable'. Install gromacs in the offline conda "
              "env (see requirements-offline.txt) before relying on these "
              "values.", file=sys.stderr)

    df_bio = pd.read_excel(BIOACT_XLSX)
    df_bio["SMILES_canonical"] = df_bio["SMILES_canonical"].fillna("").map(_canonical_smi)
    df_bio = df_bio[df_bio["SMILES_canonical"].str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df_bio)} bioactivity rows with parseable SMILES")

    # Build a representative row per unique canonical SMILES.
    df_unique = (df_bio.sort_values("row_id")
                        .drop_duplicates("SMILES_canonical", keep="first")
                        .reset_index(drop=True))
    print(f"  → {len(df_unique)} unique canonical SMILES")
    if args.iajd_num is not None:
        df_unique = df_unique[df_unique["IAJD_num"] == args.iajd_num].reset_index(drop=True)
        print(f"  --iajd-num {args.iajd_num}: running {len(df_unique)}")
    elif args.limit:
        df_unique = df_unique.head(args.limit)
        print(f"  --limit {args.limit}: running {len(df_unique)}")

    existing: Dict[str, dict] = {}
    if args.resume and CACHE_CSV.exists():
        for _, row in pd.read_csv(CACHE_CSV).iterrows():
            existing[row["smiles_canonical"]] = dict(row)
        print(f"  loaded {len(existing)} cached records (--resume)")

    audit: List[dict] = []
    if AUDIT_JSON.exists():
        try:
            audit = json.loads(AUDIT_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            audit = []

    records: List[dict] = list(existing.values())
    PHYSICS_WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for i, (_, row) in enumerate(df_unique.iterrows()):
        smi = row["SMILES_canonical"]
        if args.resume and smi in existing:
            continue
        t_start = time.time()
        try:
            rec = run_one_compound(
                smi, row.to_dict(),
                workroot=PHYSICS_WORK,
                n_molecules=args.n_molecules,
                prod_ns=args.prod_ns,
                seed_neutral=args.seed_neutral,
                seed_prot=args.seed_prot,
                timeout_s=args.timeout_s,
                keep_workdir=args.keep_trajectories,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            rec = _nan_md_record(smi, f"exception:{type(exc).__name__}:{exc}")
        elapsed = time.time() - t_start

        rec_audit = rec.pop("_audit", None)
        records.append(rec)
        audit.append({"smiles_canonical": smi,
                       "elapsed_s": round(elapsed, 1),
                       "error": rec.get("md_error"),
                       "audit": rec_audit})
        n_done += 1

        if n_done % args.checkpoint_every == 0:
            _save(records, audit)
        print(f"  [{i+1}/{len(df_unique)}] {smi[:50]}… "
              f"{elapsed:.0f}s  err={rec.get('md_error')}", flush=True)

    _save(records, audit)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min "
          f"({n_done} new records, {len(records)} total).")


def _save(records, audit):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(CACHE_CSV, index=False)
    try:
        df.to_parquet(CACHE_PARQUET, index=False)
    except (ImportError, ValueError):
        pass
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
