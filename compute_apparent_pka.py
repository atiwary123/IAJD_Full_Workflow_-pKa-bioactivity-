"""
compute_apparent_pka.py — Module A driver: apparent pKa of an ionizable group via
titratable Martini 3 constant-pH MD (Grunewald 2020), on stock GROMACS.

Titratable Martini encodes pH in the force field: a titratable bead whose type carries
an INTRINSIC pKa (e.g. N2_10.2 = amine pKa 10.2, SN6d_4.8 = aniline pKa 4.8) exchanges a
proton particle (POS) with titratable water (WNT/WN), and `#define pH<value>` selects the
pH-dependent water<->proton<->bead interaction strengths from pH_dep_interactions.itp.
Running a pH scan and measuring the average protonation <q>(pH) gives a titration curve;
the apparent pKa = pH where <q> = 0.5 (Henderson-Hasselbalch fit). In a membrane the
apparent pKa is shifted from the intrinsic by the environment — exactly the quantity that
correlates with endosomal escape (build prompt Module A).

This driver ORCHESTRATES the vendored titratable-Martini scripts/mdp/FF
(martini/titratable/) + stock GROMACS; it does not reimplement the physics.

No-proxy: each pH point is a real constant-pH MD run; a non-converged titration -> NaN +
audit. Resumable (skips completed pH dirs). Caches to bundles_caches/physics/design/pka/.

Validation: aniline in water -> pKa ~4.8 (the SN6d_4.8 bead's calibrated value), then
DLin-MC3-DMA in a bilayer -> apparent pKa ~6.44 (membrane-shifted from the ~10 intrinsic).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TITR = ROOT / "martini" / "titratable"
FF_DIR = TITR / "force_fields"
SCRIPTS = TITR / "scripts"
MDP_DIR = TITR / "mdp_files"
CACHE_DIR = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design" / "pka"

DEFAULT_PH = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]


def _gmx() -> List[str]:
    env = os.environ.get("GMX_CMD") or os.environ.get("GMX")
    if env:
        return env.split()
    r = subprocess.run(["which", "gmx"], capture_output=True, text=True)
    return [r.stdout.strip()] if r.returncode == 0 and r.stdout.strip() else ["/opt/homebrew/bin/gmx"]


def _abs_top(top_template: Path, pH: float, out: Path) -> None:
    """Write a concrete system.top: absolute FF includes + the pH define filled in."""
    txt = top_template.read_text()
    txt = txt.replace("../../force_fields", str(FF_DIR)).replace("../force_fields", str(FF_DIR))
    txt = txt.replace("<value>", f"{pH}")
    out.write_text(txt)


def _run(cmd: List[str], cwd: Path, log: Path, timeout: int) -> bool:
    with log.open("a") as fh:
        r = subprocess.run(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode == 0


def run_one_pH(workdir: Path, pH: float, start_gro: Path, top_template: Path,
               gmx: List[str], *, eq_steps: int, prod_steps: int, threads: int,
               nice: int = 10) -> Dict:
    """EM -> NVT eq -> NpT production at one pH. Returns paths + status."""
    d = workdir / f"pH_{pH}"
    if (d / "prod" / "traj_comp.xtc").exists() and (d / "prod" / "topol.tpr").exists():
        return {"pH": pH, "status": "cached", "dir": str(d)}
    d.mkdir(parents=True, exist_ok=True)
    top = d / "system.top"
    _abs_top(top_template, pH, top)
    log = d / "run.log"
    nice_p = ["nice", "-n", str(nice)]

    # 1. EM
    (d / "min").mkdir(exist_ok=True)
    if not _run(gmx + ["grompp", "-f", str(MDP_DIR / "min.mdp"), "-c", str(start_gro),
                       "-p", str(top), "-o", str(d / "min" / "em.tpr"), "-maxwarn", "5"],
                d / "min", log, 600):
        return {"pH": pH, "status": "grompp_min_failed", "dir": str(d)}
    if not _run(nice_p + gmx + ["mdrun", "-s", str(d / "min" / "em.tpr"),
                                "-deffnm", str(d / "min" / "em"), "-ntmpi", "1",
                                "-ntomp", str(threads)], d / "min", log, 3600):
        return {"pH": pH, "status": "mdrun_min_failed", "dir": str(d)}

    # 2. NVT equilibration
    (d / "eq").mkdir(exist_ok=True)
    if not _run(gmx + ["grompp", "-f", str(MDP_DIR / "eq.mdp"), "-c", str(d / "min" / "em.gro"),
                       "-p", str(top), "-o", str(d / "eq" / "eq.tpr"), "-maxwarn", "5"],
                d / "eq", log, 600):
        return {"pH": pH, "status": "grompp_eq_failed", "dir": str(d)}
    if not _run(nice_p + gmx + ["mdrun", "-s", str(d / "eq" / "eq.tpr"), "-deffnm",
                                str(d / "eq" / "eq"), "-nsteps", str(eq_steps),
                                "-ntmpi", "1", "-ntomp", str(threads)], d / "eq", log, 86400):
        return {"pH": pH, "status": "mdrun_eq_failed", "dir": str(d)}

    # 3. NpT production
    (d / "prod").mkdir(exist_ok=True)
    if not _run(gmx + ["grompp", "-f", str(MDP_DIR / "NpT.mdp"), "-c", str(d / "eq" / "eq.gro"),
                       "-p", str(top), "-o", str(d / "prod" / "topol.tpr"), "-maxwarn", "5"],
                d / "prod", log, 600):
        return {"pH": pH, "status": "grompp_prod_failed", "dir": str(d)}
    if not _run(nice_p + gmx + ["mdrun", "-s", str(d / "prod" / "topol.tpr"), "-deffnm",
                                str(d / "prod" / "prod"), "-nsteps", str(prod_steps),
                                "-c", str(d / "prod" / "confout.gro"),
                                "-x", str(d / "prod" / "traj_comp.xtc"),
                                "-ntmpi", "1", "-ntomp", str(threads)], d / "prod", log, 172800):
        return {"pH": pH, "status": "mdrun_prod_failed", "dir": str(d)}
    return {"pH": pH, "status": "ok", "dir": str(d)}


def analyze_pH(d: Path, sel: str, ref: str, gmx: List[str], start_frac: float = 0.4) -> float:
    """Average degree of protonation <q> at one pH via degree_of_deprot.py.

    Uses a .gro topology (not the .tpr): GROMACS-2026 tpr (tpx v138) is too new for
    MDAnalysis' TPR parser, but a .gro is version-independent and carries the bead
    names the distance scheme selects on. Runs under THIS interpreter (venv MDAnalysis).
    """
    prod = d / "prod"
    traj = prod / "traj_comp.xtc"
    topo = prod / "confout.gro"          # version-independent topology
    if not traj.exists() or not topo.exists():
        return float("nan")
    import MDAnalysis as mda
    try:
        n = len(mda.Universe(str(topo), str(traj)).trajectory)
    except Exception:
        n = 0
    b = int(n * start_frac)
    out = prod / "dop.xvg"
    log = d / "analyze.log"
    ok = _run([sys.executable, str(SCRIPTS / "degree_of_deprot.py"), "-f", str(traj),
               "-s", str(topo), "-o", str(out), "-b", str(b), "-ref", ref, "-sel", sel],
              prod, log, 1800)
    if not ok or not out.exists():
        return float("nan")
    vals = []
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else float("nan")


def henderson_hasselbalch(pH: np.ndarray, q: np.ndarray, is_base: bool = True
                          ) -> Tuple[float, float, float]:
    """Fit the DEGREE OF DEPROTONATION <q>(pH) (the degree_of_deprot.py observable:
    0=fully protonated, 1=fully deprotonated, monotonically increasing with pH) to a
    Hill / Henderson-Hasselbalch curve:
        deprot(pH) = 1 / (1 + 10^(n*(pKa - pH)))
    The apparent pKa is the midpoint (deprot = 0.5). This holds for both acids and
    bases since deprotonation always rises with pH; `is_base` is accepted for API
    symmetry but unused. Returns (pKa, hill_n, rmse)."""
    from scipy.optimize import curve_fit

    def model(x, pKa, n):
        return 1.0 / (1.0 + 10.0 ** (n * (pKa - x)))

    m = np.isfinite(q)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        p0 = [float(pH[m][np.argmin(np.abs(q[m] - 0.5))]), 1.0]
        popt, _ = curve_fit(model, pH[m], q[m], p0=p0, maxfev=20000,
                            bounds=([2.0, 0.2], [12.0, 6.0]))
        rmse = float(np.sqrt(np.mean((model(pH[m], *popt) - q[m]) ** 2)))
        return float(popt[0]), float(popt[1]), rmse
    except Exception:
        return float("nan"), float("nan"), float("nan")


def titrate(name: str, start_gro: Path, top_template: Path, sel: str, ref: str, *,
            pH_values: Optional[List[float]] = None, is_base: bool = True,
            prod_ns: float = 20.0, eq_ns: float = 2.0, threads: int = 4,
            dt_ps: float = 0.01, workroot: Optional[Path] = None) -> Dict:
    """Full titration: scan pH, measure <q>, fit apparent pKa. Resumable."""
    gmx = _gmx()
    start_gro = Path(start_gro).resolve()
    top_template = Path(top_template).resolve()
    pH_values = pH_values or DEFAULT_PH
    workdir = (workroot or (ROOT / "physics_cache" / "pka_runs")) / name
    workdir.mkdir(parents=True, exist_ok=True)
    eq_steps = int(eq_ns * 1000 / dt_ps)
    prod_steps = int(prod_ns * 1000 / dt_ps)
    rec: Dict = {"name": name, "sel": sel, "ref": ref, "is_base": is_base,
                 "pH_values": pH_values, "prod_ns": prod_ns, "points": [],
                 "apparent_pKa": float("nan"), "error": None}
    t0 = time.time()
    for pH in pH_values:
        r = run_one_pH(workdir, pH, start_gro, top_template, gmx,
                       eq_steps=eq_steps, prod_steps=prod_steps, threads=threads)
        q = float("nan")
        if r["status"] in ("ok", "cached"):
            q = analyze_pH(Path(r["dir"]), sel, ref, gmx)
        rec["points"].append({"pH": pH, "q": q, "status": r["status"]})
        print(f"  pH {pH}: status={r['status']} <q>={q:.3f}", flush=True)
        _save(rec)
    pHs = np.array([p["pH"] for p in rec["points"]])
    qs = np.array([p["q"] for p in rec["points"]])
    pKa, hill, rmse = henderson_hasselbalch(pHs, qs, is_base=is_base)
    rec["apparent_pKa"] = pKa
    rec["hill_n"] = hill
    rec["fit_rmse"] = rmse
    rec["elapsed_min"] = round((time.time() - t0) / 60, 1)
    _save(rec)
    return rec


def _save(rec: Dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{rec['name']}_pka.json").write_text(json.dumps(rec, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--start-gro", required=True, type=Path)
    p.add_argument("--top", required=True, type=Path, help="system.top template (#define pH<value>)")
    p.add_argument("--sel", required=True, help="MDAnalysis sel of titratable bead, e.g. 'name P2'")
    p.add_argument("--ref", default="name WN", help="reference selection (water)")
    p.add_argument("--acid", action="store_true", help="titrate as acid (default base)")
    p.add_argument("--prod-ns", type=float, default=20.0)
    p.add_argument("--eq-ns", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pH", type=float, nargs="*", default=None)
    args = p.parse_args()
    rec = titrate(args.name, args.start_gro, args.top, args.sel, args.ref,
                  pH_values=args.pH, is_base=not args.acid, prod_ns=args.prod_ns,
                  eq_ns=args.eq_ns, threads=args.threads)
    print(json.dumps({k: v for k, v in rec.items() if k != "points"}, indent=2, default=str))
    print("titration curve:", [(p["pH"], round(p["q"], 3)) for p in rec["points"]])


if __name__ == "__main__":
    raise SystemExit(main())
