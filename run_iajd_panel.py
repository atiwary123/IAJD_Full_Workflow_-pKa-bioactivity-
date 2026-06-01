"""
run_iajd_panel.py — launch the Module B (c0) curvature panel over an IAJD family in
parallel, in the FW-3 active-learning order. Built for a many-core rented box
(RunPod/Vast/Lambda): runs `--jobs` IAJDs concurrently, each compute_curvature.py
subprocess pinned to `--threads` cores, and keeps the GPU busy by offloading up to
`--gpu-jobs` of the concurrent runs (gmx -nb gpu). Resumable (skips finished JSONs).

PANEL ORDER (docs/PHYSICS_DESIGN_FUTURE_WORK.md FW-3):
  highest-flux (369) -> lowest-flux (reliable) -> median-flux -> quantile bisection
  (farthest-point in flux). Flux extremes falsify cheapest; the median tests
  monotone-vs-window; then space-fill. After ~5-7 points, hand to GP/BO.

HONESTY: every c0 here is PROVISIONAL. A tensionless, converged MD (c0_physics_
converged) does NOT make the number trusted — Module E CG-mapping validation does
(c0_trusted). The panel's value is the ORDER + the relative shifts, read with that
caveat. Non-bilayer-preferring IAJDs may not hold a flat bilayer on this timescale;
that is reported (bilayer_intact=False), not hidden.

Usage (on the pod, after cloud_setup.sh + `source cloud_env.sh`):
  python run_iajd_panel.py --family GA-Tris --jobs 6 --threads 4 --gpu-jobs 1 \
         --prod-ns 80 --force-check
  python run_iajd_panel.py --iajds 369,360,352 --jobs 3 --threads 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BIOACT = ROOT / "IAJD_master" / "datasets" / "IAJD_Bioact_v13_clean.xlsx"
DESIGN_DIR = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design"
PYBIN = sys.executable


def _col(df: pd.DataFrame, *cands: str):
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_family(family: str, flux_col: str):
    df = pd.read_excel(BIOACT)
    fam_col = _col(df, "family", "Family", "architecture", "Architecture")
    if fam_col and family.lower() != "all":
        sub = df[df[fam_col].astype(str).str.contains(family, case=False, na=False)].copy()
    else:
        sub = df.copy()
    sub = sub.dropna(subset=[flux_col, "IAJD_num"])
    return sub


def fw3_order(sub: pd.DataFrame, flux_col: str):
    """FW-3 order: high, low, median, then farthest-point (quantile bisection) on flux.
    Returns (ordered_iajd_nums, {iajd: flux}, {iajd: reliability_note})."""
    s = sub.sort_values(flux_col).reset_index(drop=True)
    nums = s["IAJD_num"].astype(int).tolist()
    flux = np.asarray(s[flux_col], float)
    if not nums:
        return [], {}, {}
    # reliability columns (for an honest "lowest RELIABLE" pick downstream / reporting)
    sem_col = _col(s, "flux_total_SEM", "log10_flux_total_SEM")
    nrep_col = _col(s, "n_replicates", "n_mice")
    rel = {}
    for i, n in enumerate(nums):
        note = []
        if sem_col is not None and pd.notna(s.iloc[i][sem_col]):
            note.append(f"SEM={float(s.iloc[i][sem_col]):.2f}")
        if nrep_col is not None and pd.notna(s.iloc[i][nrep_col]):
            note.append(f"n={int(s.iloc[i][nrep_col])}")
        rel[n] = ",".join(note) or "n/a"

    seed = []
    for idx in (len(nums) - 1, 0, len(nums) // 2):   # high, low, median
        if idx not in seed:
            seed.append(idx)
    placed = list(seed)
    while len(placed) < len(nums):
        best, bestd = -1, -1.0
        for i in range(len(nums)):
            if i in placed:
                continue
            d = min(abs(flux[i] - flux[p]) for p in placed)
            if d > bestd:
                bestd, best = d, i
        placed.append(best)
    order = [nums[i] for i in placed]
    fmap = {nums[i]: float(flux[i]) for i in range(len(nums))}
    return order, fmap, rel


def run_one(iajd: int, args, gpu: bool):
    tag = "host" if args.host else "neutral"
    out = DESIGN_DIR / f"IAJD{iajd}_{tag}_T{int(args.temp)}_curvature.json"
    if out.exists() and not args.force:
        try:
            r = json.loads(out.read_text())
            if r.get("md_status") == "ok" or r.get("error"):
                return iajd, f"cached:{r.get('md_status') or r.get('error')}", \
                    r.get("c0_nm_inv"), r.get("c0_physics_converged")
        except (json.JSONDecodeError, OSError):
            pass
    cmd = [PYBIN, str(ROOT / "compute_curvature.py"), "--iajd", str(iajd),
           "--prod-ns", str(args.prod_ns), "--eq2-ps", str(args.eq2_ps),
           "--temp", str(args.temp), "--threads", str(args.threads),
           "--n-per-leaflet", str(args.n_per_leaflet)]
    if args.host:
        cmd += ["--host", "--n-iajd", str(args.n_iajd)]   # FW-4: stable POPC host
    else:
        cmd += ["--apl", str(args.apl)]
    if args.force_check:
        cmd.append("--force-check")
    if gpu:
        cmd.append("--gpu")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = (time.time() - t0) / 60.0
    try:
        rr = json.loads(out.read_text())
        status = rr.get("md_status") or rr.get("error") or "done"
        return iajd, f"{status} ({dt:.0f}min{'/gpu' if gpu else ''})", \
            rr.get("c0_nm_inv"), rr.get("c0_physics_converged")
    except (json.JSONDecodeError, OSError):
        tail = (proc.stderr or proc.stdout or "no output")[-300:]
        return iajd, f"FAILED ({dt:.0f}min): {tail}", None, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--family", help="family substring (e.g. GA-Tris, PE-Tris); 'all' = whole set")
    g.add_argument("--iajds", help="explicit comma-separated IAJD numbers (overrides order)")
    p.add_argument("--flux-col", default="log10_flux_spleen")
    p.add_argument("--jobs", type=int, default=4, help="concurrent IAJD runs")
    p.add_argument("--threads", type=int, default=4, help="OpenMP threads per run")
    p.add_argument("--gpu-jobs", type=int, default=0,
                   help="max concurrent runs offloaded to the GPU (-nb gpu); keeps the "
                        "GPU busy while CPU cores handle the rest")
    p.add_argument("--prod-ns", type=float, default=80.0)
    p.add_argument("--eq2-ps", type=float, default=4000.0)
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--n-per-leaflet", type=int, default=64)
    p.add_argument("--apl", type=float, default=1.2)
    p.add_argument("--host", action="store_true",
                   help="HOST method (recommended): IAJDs in a stable POPC bilayer (FW-4). "
                        "Pure-bilayer (default) collapses for non-bilayer IAJDs.")
    p.add_argument("--n-iajd", type=int, default=8,
                   help="IAJDs per upper leaflet for the host method (higher = better c0 signal)")
    p.add_argument("--force-check", action="store_true", help="recommended: validate forces vs GROMACS")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--limit", type=int, default=0, help="only run the first N of the order")
    args = p.parse_args()

    if args.iajds:
        order = [int(x) for x in args.iajds.replace(" ", "").split(",") if x]
        fmap, rel = {}, {}
        label = "custom"
    else:
        sub = load_family(args.family, args.flux_col)
        order, fmap, rel = fw3_order(sub, args.flux_col)
        label = args.family
    if args.limit:
        order = order[:args.limit]

    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== IAJD Module-B panel: {label}  ({len(order)} IAJDs, {args.jobs} jobs x "
          f"{args.threads} threads, gpu_jobs={args.gpu_jobs}) ==")
    print(f"   FW-3 order ({args.flux_col}): " + " -> ".join(str(n) for n in order))
    for n in order[:8]:
        print(f"     IAJD{n}: flux={fmap.get(n, float('nan')):.2f}  reliab[{rel.get(n,'n/a')}]"
              if n in fmap else f"     IAJD{n}")
    print("   (every c0 is PROVISIONAL until Module E validates the CG mapping)\n")

    gpu_sem = threading.Semaphore(args.gpu_jobs) if args.gpu_jobs > 0 else None

    def task(iajd: int):
        gpu = False
        if gpu_sem is not None and gpu_sem.acquire(blocking=False):
            gpu = True
        try:
            return run_one(iajd, args, gpu)
        finally:
            if gpu and gpu_sem is not None:
                gpu_sem.release()

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(task, n): n for n in order}
        done = 0
        for f in as_completed(futs):
            iajd, status, c0, conv = f.result()
            done += 1
            results[iajd] = {"c0_nm_inv": c0, "status": status,
                             "physics_converged": conv, "flux": fmap.get(iajd)}
            c0s = f"{c0:+.3f}" if isinstance(c0, (int, float)) else "  nan"
            print(f"  [{done}/{len(order)}] IAJD{iajd}: c0={c0s} "
                  f"flux={fmap.get(iajd)} conv={conv} [{status}]")

    summary = {"family": label, "flux_col": args.flux_col, "order": order,
               "results": results, "elapsed_min": round((time.time() - t0) / 60, 1),
               "provisional": True,
               "note": "c0 PROVISIONAL until Module E; panel value is order + relative shifts"}
    out = DESIGN_DIR / f"iajd_panel_{label}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n== panel written: {out}  ({summary['elapsed_min']} min) ==")
    # quick honest read of the gradient (high vs low flux), if both converged
    conv = {k: v for k, v in results.items()
            if v.get("physics_converged") and isinstance(v.get("c0_nm_inv"), (int, float))}
    if len(conv) >= 2 and fmap:
        hi = max(conv, key=lambda k: fmap.get(k, -1e9))
        lo = min(conv, key=lambda k: fmap.get(k, 1e9))
        print(f"   converged gradient: high-flux IAJD{hi} c0={conv[hi]['c0_nm_inv']:+.3f} "
              f"vs low-flux IAJD{lo} c0={conv[lo]['c0_nm_inv']:+.3f}  "
              f"(Δc0={conv[hi]['c0_nm_inv']-conv[lo]['c0_nm_inv']:+.3f} nm^-1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
