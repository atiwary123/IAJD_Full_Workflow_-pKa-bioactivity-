"""run_pka_panel.py — 32-CPU parallel apparent-pKa (Module A) panel over the IAJD library.

For each IAJD (corrected AUDIT_FIXED dataset, §4d-filtered to drop UNRESOLVED|FLAG rows):
  1. build a titratable constant-pH system — 1 IAJD embedded head-up in a POPC host bilayer,
     ionizable head -> titratable P2/DN/DP motif, water -> WNA+H+ (build_membrane_pka.build_iajd
     + the merged martini_titratable_full.itp force field),
  2. run a constant-pH pH scan (compute_apparent_pka.run_one_pH: EM -> NVT eq -> NpT prod at
     each pH), measure the degree of deprotonation <q>(pH),
  3. fit the apparent pKa = midpoint of the Henderson-Hasselbalch titration curve.

PARALLELISM (all 32 cores): the unit of work is one (IAJD, pH) job — an independent
EM->NVT->NpT chain. We enumerate every (IAJD x pH) job and run --jobs of them concurrently,
each using --threads OpenMP threads with -pin off so concurrent mdruns don't fight over the
same physical cores (jobs * threads ~= ncores). This load-balances far better than running
IAJDs one-at-a-time (which would idle cores whenever an IAJD has < jobs pH points left).

ROBUSTNESS: resumable at every level (run_one_pH skips completed pH dirs; --resume skips
IAJDs already fit); fault-isolated (a broad except per job so one bad IAJD/pH can never abort
the batch); disk-hygienic (purges per-pH min/eq scratch after production, keeping only the
trajectory + final frame the analysis needs). No proxies: every pH point is a real
constant-pH MD run; a non-converged titration -> NaN apparent pKa + audit, never a fabricated
value. Caches per IAJD to IAJD_master/bundles_caches/physics/design/pka/IAJD<n>_pka.json.

HONEST SCOPE (see docs): the titratable method is VALIDATED for MC3-in-POPC (apparent pKa
6.44 recovered non-circularly). For IAJDs it inherits (a) the provisional, Module-E-unvalidated
CG mapping (build_cg), and (b) a consistent generic 10.2 intrinsic amine bead — so the
per-IAJD number is the MEMBRANE-SHIFTED apparent pKa relative to that intrinsic; the
design-relevant signal is the shift / the relative ordering across IAJDs, not the absolute.

Usage on the pod (32 vCPU):
  python run_pka_panel.py --jobs 8 --threads 4 --prod-ns 20 --eq-ns 2
  python run_pka_panel.py --family GA-Tris --jobs 8 --threads 4         # one family
  python run_pka_panel.py --n-iajd 16 --order flux --jobs 8 --threads 4 # flux extremes first
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compute_apparent_pka import (run_one_pH, analyze_pH, henderson_hasselbalch, _save,
                                  DEFAULT_PH, CACHE_DIR, _gmx)
from physics_design.build_membrane_pka import build_iajd
from physics_design.build_titratable_ff import build as build_titratable_ff, OUT as FULL_FF

_DS = ROOT / "IAJD_master" / "datasets"
BIOACT = (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx") if \
    (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx").exists() else \
    (_DS / "IAJD_Bioact_v13_clean.xlsx")


def load_iajds(family: Optional[str], n_limit: Optional[int], order: str) -> List[int]:
    """IAJD_nums to run: corrected dataset, §4d-filtered, optionally one family, ordered."""
    d = pd.read_excel(BIOACT)
    bad = (d["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
           if "audit_status" in d else pd.Series(False, index=d.index))
    d = d[~bad].copy()
    if family:
        fam_col = "family" if "family" in d else "family_original"
        d = d[d[fam_col].astype(str).str.strip().str.lower() == family.strip().lower()]
    d = d.dropna(subset=["IAJD_num"]).drop_duplicates(subset=["IAJD_num"])
    if order == "flux":
        # flux EXTREMES first (FW-3): the most design-informative pKa points.
        fcol = next((c for c in ("flux_liver", "flux_spleen", "flux_lung", "log10_flux")
                     if c in d.columns and d[c].notna().any()), None)
        if fcol is not None:
            d["_rank"] = (d[fcol] - d[fcol].median()).abs()
            d = d.sort_values("_rank", ascending=False)
        else:
            d = d.sort_values("IAJD_num")
    else:
        d = d.sort_values("IAJD_num")
    nums = [int(x) for x in d["IAJD_num"].tolist()]
    return nums[:n_limit] if n_limit else nums


def _is_done(name: str) -> bool:
    f = CACHE_DIR / f"{name}_pka.json"
    if not f.exists():
        return False
    try:
        rec = json.loads(f.read_text())
        return np.isfinite(float(rec.get("apparent_pKa", float("nan"))))
    except Exception:
        return False


def build_one(iajd: int, workroot: Path, n_per_leaflet: int, gmx: List[str]) -> Optional[Dict]:
    """Build the titratable system for one IAJD (resumable: reuse existing start.gro+top)."""
    wd = workroot / f"IAJD{iajd}"
    start, top = wd / "start.gro", wd / "system.top"
    if start.exists() and top.exists():
        return {"iajd": iajd, "workdir": str(wd), "start_gro": str(start), "top": str(top),
                "sel": "name P2", "ref": "name WN", "status": "cached"}
    try:
        info = build_iajd(iajd, wd, gmx, n_per_leaflet=n_per_leaflet)
        if info.get("status") != "ok":
            return {"iajd": iajd, "status": f"build_failed:{info.get('status')}",
                    "log": info.get("log", "")[:400]}
        return {"iajd": iajd, "workdir": str(wd), "start_gro": info["start_gro"],
                "top": info["top"], "sel": info["analysis_sel"], "ref": info["analysis_ref"],
                "status": "built"}
    except Exception as e:
        return {"iajd": iajd, "status": "build_exception", "log": f"{type(e).__name__}: {e}"}


def _purge_scratch(pH_dir: Path) -> None:
    """Drop per-pH min/ + eq/ scratch + prod tpr after production; keep traj + confout + dop."""
    for sub in ("min", "eq"):
        shutil.rmtree(pH_dir / sub, ignore_errors=True)
    for f in (pH_dir / "prod").glob("*.tpr"):
        try:
            f.unlink()
        except OSError:
            pass


def run_pH_job(job: Tuple[Dict, float], *, eq_steps: int, prod_steps: int, threads: int,
               gmx: List[str], purge: bool) -> Dict:
    """One (IAJD, pH) constant-pH MD job. Fault-isolated."""
    sysinfo, pH = job
    iajd = sysinfo["iajd"]
    try:
        r = run_one_pH(Path(sysinfo["workdir"]), pH, Path(sysinfo["start_gro"]),
                       Path(sysinfo["top"]), gmx, eq_steps=eq_steps, prod_steps=prod_steps,
                       threads=threads, pin="off")
        if purge and r.get("status") in ("ok", "cached"):
            _purge_scratch(Path(sysinfo["workdir"]) / f"pH_{pH}")
        return {"iajd": iajd, "pH": pH, "status": r["status"], "dir": r.get("dir")}
    except Exception as e:
        return {"iajd": iajd, "pH": pH, "status": "job_exception",
                "log": f"{type(e).__name__}: {str(e)[:200]}"}


def finalize_iajd(sysinfo: Dict, pH_values: List[float], gmx: List[str]) -> Dict:
    """Analyze every pH for one IAJD + Henderson-Hasselbalch fit -> apparent pKa. Cache."""
    iajd = sysinfo["iajd"]
    name = f"IAJD{iajd}"
    wd = Path(sysinfo["workdir"])
    rec: Dict = {"name": name, "sel": sysinfo["sel"], "ref": sysinfo["ref"],
                 "method": "titratable-Martini-3 constant-pH MD (Module A), generic N2_10.2 "
                           "intrinsic; apparent = membrane-shifted", "pH_values": pH_values,
                 "points": [], "apparent_pKa": float("nan")}
    for pH in pH_values:
        d = wd / f"pH_{pH}"
        q = float("nan")
        if (d / "prod" / "traj_comp.xtc").exists() and (d / "prod" / "confout.gro").exists():
            try:
                q = analyze_pH(d, sysinfo["sel"], sysinfo["ref"], gmx)
            except Exception:
                q = float("nan")
        rec["points"].append({"pH": pH, "q": q})
    pHs = np.array([p["pH"] for p in rec["points"]], float)
    qs = np.array([p["q"] for p in rec["points"]], float)
    pKa, hill, rmse = henderson_hasselbalch(pHs, qs)
    rec.update({"apparent_pKa": pKa, "hill_n": hill, "fit_rmse": rmse,
                "n_pH_ok": int(np.isfinite(qs).sum())})
    _save(rec)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default=None, help="restrict to one family (e.g. GA-Tris)")
    p.add_argument("--n-iajd", type=int, default=None, help="limit number of IAJDs")
    p.add_argument("--order", choices=["num", "flux"], default="num",
                   help="'flux' = flux extremes first (most design-informative)")
    p.add_argument("--jobs", type=int, default=0, help="concurrent MD jobs (0=auto=ncpu//threads)")
    p.add_argument("--threads", type=int, default=4, help="OpenMP threads per job")
    p.add_argument("--prod-ns", type=float, default=20.0)
    p.add_argument("--eq-ns", type=float, default=2.0)
    p.add_argument("--dt-ps", type=float, default=0.01, help="10 fs (stiffened CG bonds)")
    p.add_argument("--pH", type=float, nargs="*", default=None, help="pH grid (default 3.0-8.0)")
    p.add_argument("--n-per-leaflet", type=int, default=32)
    p.add_argument("--resume", action="store_true", help="skip IAJDs already fit")
    p.add_argument("--no-purge", action="store_true", help="keep per-pH min/eq scratch")
    p.add_argument("--workroot", default=str(ROOT / "physics_cache" / "pka_panel"))
    args = p.parse_args()

    gmx = _gmx()
    threads = max(1, args.threads)
    jobs = args.jobs or max(1, (os.cpu_count() or 8) // threads)
    pH_values = args.pH or DEFAULT_PH
    eq_steps = int(args.eq_ns * 1000 / args.dt_ps)
    prod_steps = int(args.prod_ns * 1000 / args.dt_ps)
    workroot = Path(args.workroot)
    workroot.mkdir(parents=True, exist_ok=True)
    purge = not args.no_purge

    if not FULL_FF.exists():
        print("[ff] building merged titratable FF (martini_titratable_full.itp)…", flush=True)
        build_titratable_ff(verbose=False)

    nums = load_iajds(args.family, args.n_iajd, args.order)
    if args.resume:
        nums = [n for n in nums if not _is_done(f"IAJD{n}")]
    print(f"[panel] {len(nums)} IAJDs | {len(pH_values)} pH points | "
          f"jobs={jobs} x threads={threads} | prod={args.prod_ns}ns eq={args.eq_ns}ns | "
          f"gmx={' '.join(gmx)}", flush=True)
    if not nums:
        print("[panel] nothing to do (all done / none match).")
        return 0
    t0 = time.time()

    # ---- Phase 1: build all systems (parallel; fast — solvate + convert) ----
    systems: Dict[int, Dict] = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for info in ex.map(lambda n: build_one(n, workroot, args.n_per_leaflet, gmx), nums):
            if info and info.get("status") in ("built", "cached"):
                systems[info["iajd"]] = info
            else:
                print(f"  [build] IAJD{info.get('iajd')} SKIPPED: {info.get('status')} "
                      f"{info.get('log','')[:120]}", flush=True)
    print(f"[panel] built {len(systems)}/{len(nums)} systems "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- Phase 2: run every (IAJD, pH) job concurrently ----
    panel_jobs = [(systems[n], pH) for n in systems for pH in pH_values]
    done = 0
    total = len(panel_jobs)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_pH_job, j, eq_steps=eq_steps, prod_steps=prod_steps,
                          threads=threads, gmx=gmx, purge=purge): j for j in panel_jobs}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r["status"] not in ("ok", "cached"):
                print(f"  [pH] IAJD{r['iajd']} pH{r['pH']}: {r['status']} "
                      f"{r.get('log','')[:120]}", flush=True)
            if done % 20 == 0 or done == total:
                el = (time.time() - t0) / 60
                print(f"[panel] {done}/{total} pH-jobs done ({el:.1f} min)", flush=True)

    # ---- Phase 3: analyze + Henderson-Hasselbalch fit per IAJD (parallel) ----
    fits = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for rec in ex.map(lambda n: finalize_iajd(systems[n], pH_values, gmx), list(systems)):
            fits.append(rec)
            print(f"  [fit] {rec['name']}: apparent_pKa={rec['apparent_pKa']:.2f} "
                  f"hill_n={rec.get('hill_n', float('nan')):.2f} "
                  f"rmse={rec.get('fit_rmse', float('nan')):.3f} "
                  f"({rec['n_pH_ok']}/{len(pH_values)} pH ok)", flush=True)

    ok = [r for r in fits if np.isfinite(r["apparent_pKa"])]
    summ = CACHE_DIR / "_pka_panel_summary.json"
    summ.write_text(json.dumps(
        {"n_iajd": len(systems), "n_fit": len(ok), "elapsed_min": round((time.time()-t0)/60, 1),
         "results": sorted(({"iajd": r["name"], "apparent_pKa": r["apparent_pKa"],
                             "n_pH_ok": r["n_pH_ok"]} for r in fits),
                           key=lambda x: x["iajd"])}, indent=2))
    print(f"\n[panel] DONE: {len(ok)}/{len(systems)} apparent pKa fit, "
          f"{(time.time()-t0)/60:.1f} min. Summary -> {summ}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
