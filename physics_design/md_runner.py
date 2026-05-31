"""
physics_design/md_runner.py — run the EM -> tensionless-NPT -> production protocol
for a flat Martini-3 bilayer with stock GROMACS (CPU-only).

Pressure coupling is SEMIISOTROPIC with equal reference pressure laterally and
normally (1 bar / 1 bar). For a planar membrane this enforces zero average surface
tension  gamma = Lz * <P_N - P_L> = 0 — i.e. a *tensionless* bilayer, the required
reference state for a spontaneous-curvature measurement.

Martini-3 standard nonbonded settings (Souza et al., Nat. Methods 2021):
  reaction-field, rcoulomb=rvdw=1.1 nm, epsilon_r=15, epsilon_rf=0 (infinite),
  vdw-modifier=Potential-shift, Verlet scheme. These are the exact forces the
  pressure-profile recompute (pressure_profile.py) reproduces.

Thermal budget: the driver caps OpenMP threads (default 4) and `nice`s the run so
the fanless M-series Air stays cool over long jobs. Checkpointed via gmx .cpt.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EM_MDP = """; Martini 3 EM
integrator   = steep
nsteps       = 10000
emtol        = 100.0
emstep       = 0.01
nstlist      = 20
cutoff-scheme = Verlet
coulombtype  = reaction-field
rcoulomb     = 1.1
epsilon_r    = 15
vdw-type     = cutoff
vdw-modifier = Potential-shift
rvdw         = 1.1
"""

_COMMON = """cutoff-scheme = Verlet
nstlist      = 20
coulombtype  = reaction-field
rcoulomb     = 1.1
epsilon_r    = 15
vdw-type     = cutoff
vdw-modifier = Potential-shift
rvdw         = 1.1
tcoupl       = v-rescale
tc-grps      = MEMB SOL
tau-t        = 1.0 1.0
ref-t        = {temp} {temp}
pcoupltype   = semiisotropic
ref-p        = 1.0 1.0
compressibility = 3e-4 3e-4
"""

EQ1_MDP = """; Martini 3 gentle NPT equilibration (c-rescale)
integrator   = md
dt           = 0.010
nsteps       = {nsteps}
nstxout-compressed = 0
nstenergy    = 1000
nstlog       = 1000
""" + _COMMON + """pcoupl       = c-rescale
tau-p        = 4.0
gen-vel      = yes
gen-temp     = {temp}
gen-seed     = {seed}
refcoord-scaling = all
"""

EQ2_MDP = """; Martini 3 NPT equilibration
integrator   = md
dt           = 0.020
nsteps       = {nsteps}
nstxout-compressed = 0
nstenergy    = 1000
nstlog       = 5000
""" + _COMMON + """pcoupl       = c-rescale
tau-p        = 8.0
gen-vel      = no
"""

PROD_MDP = """; Martini 3 production (tensionless bilayer, frames for pressure profile)
integrator   = md
dt           = 0.020
nsteps       = {nsteps}
nstxout-compressed = {nstxout}
nstenergy    = 5000
nstlog       = 50000
""" + _COMMON + """pcoupl       = parrinello-rahman
tau-p        = 12.0
gen-vel      = no
"""


@dataclass
class BilayerRunSpec:
    workdir: Path
    gro: Path
    top: Path
    gmx_cmd: List[str]
    temp_K: float = 300.0
    seed: int = 1
    eq1_ps: float = 500.0       # gentle dt=10fs
    eq2_ps: float = 3000.0      # dt=20fs, let APL converge
    prod_ns: float = 60.0       # production for pressure profile
    frame_ps: float = 100.0     # save a frame every frame_ps
    n_threads: int = 4
    nice: int = 10


def _make_index(spec: BilayerRunSpec) -> Optional[Path]:
    """Create an index with MEMB (non-water) and SOL (water) groups for the two
    temperature-coupling baths. Uses gmx select via make_ndx echo piping."""
    ndx = spec.workdir / "index.ndx"
    # Build with gmx make_ndx: 'r W' = water; '!r W' = membrane.
    cmd = spec.gmx_cmd + ["make_ndx", "-f", str(spec.gro), "-o", str(ndx)]
    script = "r W\nname 0 SYSTEM\n! r W\n"  # create W group then its complement
    # Simpler & robust: use two commands. Default groups include 'W' if present.
    script = "del 0-50\nr W\nname 0 SOL\n! r W\nname 1 MEMB\nq\n"
    r = subprocess.run(cmd, cwd=str(spec.workdir), input=script,
                       capture_output=True, text=True, timeout=120)
    if not ndx.exists():
        return None
    return ndx


def _grompp_mdrun(spec: BilayerRunSpec, mdp_text: str, gro_in: Path, step: str,
                  ndx: Optional[Path], timeout_s: int,
                  maxwarn: int = 2) -> Tuple[bool, str]:
    wd = spec.workdir
    mdp = wd / f"{step}.mdp"
    mdp.write_text(mdp_text)
    tpr = wd / f"{step}.tpr"
    g = spec.gmx_cmd + ["grompp", "-f", str(mdp), "-c", str(gro_in),
                        "-p", str(spec.top), "-o", str(tpr), "-maxwarn", str(maxwarn)]
    if ndx is not None:
        g += ["-n", str(ndx)]
    r1 = subprocess.run(g, cwd=str(wd), capture_output=True, text=True, timeout=300)
    if r1.returncode != 0:
        return False, "grompp:\n" + (r1.stdout + r1.stderr)[-4000:]
    md = (["nice", "-n", str(spec.nice)] + spec.gmx_cmd +
          ["mdrun", "-s", str(tpr), "-deffnm", str(wd / step),
           "-ntmpi", "1", "-ntomp", str(spec.n_threads), "-pin", "on"])
    r2 = subprocess.run(md, cwd=str(wd), capture_output=True, text=True, timeout=timeout_s)
    ok = r2.returncode == 0 and (wd / f"{step}.gro").exists()
    return ok, (r2.stdout + r2.stderr)[-4000:]


def run_bilayer(spec: BilayerRunSpec) -> Dict:
    spec.workdir.mkdir(parents=True, exist_ok=True)
    audit: Dict = {"workdir": str(spec.workdir), "temp_K": spec.temp_K,
                   "prod_ns": spec.prod_ns, "frame_ps": spec.frame_ps}
    ndx = _make_index(spec)
    if ndx is None:
        audit["status"] = "make_ndx_failed"
        return audit

    # EM (no tcoupl groups needed; use no index to avoid group errors)
    ok, log = _grompp_mdrun(spec, EM_MDP, spec.gro, "em", None, timeout_s=1200)
    if not ok:
        audit.update(status="em_failed", log=log)
        return audit

    eq1 = EQ1_MDP.format(nsteps=int(spec.eq1_ps / 0.010), temp=spec.temp_K,
                         seed=spec.seed)
    ok, log = _grompp_mdrun(spec, eq1, spec.workdir / "em.gro", "eq1", ndx,
                            timeout_s=7200)
    if not ok:
        audit.update(status="eq1_failed", log=log)
        return audit

    eq2 = EQ2_MDP.format(nsteps=int(spec.eq2_ps / 0.020), temp=spec.temp_K)
    ok, log = _grompp_mdrun(spec, eq2, spec.workdir / "eq1.gro", "eq2", ndx,
                            timeout_s=14400)
    if not ok:
        audit.update(status="eq2_failed", log=log)
        return audit

    nstxout = int(spec.frame_ps / 0.020)
    prod = PROD_MDP.format(nsteps=int(spec.prod_ns * 1000 / 0.020),
                           nstxout=nstxout, temp=spec.temp_K)
    ok, log = _grompp_mdrun(spec, prod, spec.workdir / "eq2.gro", "prod", ndx,
                            timeout_s=172800)
    if not ok:
        audit.update(status="prod_failed", log=log)
        return audit

    audit.update(status="ok",
                 traj=str(spec.workdir / "prod.xtc"),
                 tpr=str(spec.workdir / "prod.tpr"),
                 final_gro=str(spec.workdir / "prod.gro"),
                 eq2_gro=str(spec.workdir / "eq2.gro"))
    return audit


if __name__ == "__main__":
    import argparse, json
    from .lipid_library import load_lipid
    from .bilayer import build_system
    p = argparse.ArgumentParser()
    p.add_argument("lipid")
    p.add_argument("--workdir", default="/tmp/bilayer_run")
    p.add_argument("--n-per-leaflet", type=int, default=64)
    p.add_argument("--prod-ns", type=float, default=20.0)
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    gmx = [os.environ.get("GMX", "/opt/homebrew/bin/gmx")]
    wd = Path(args.workdir)
    lip = load_lipid(args.lipid)
    info = build_system(lip, wd, gmx, n_per_leaflet=args.n_per_leaflet)
    print("build:", info["status"], info.get("box_nm"))
    if info["status"] != "ok":
        raise SystemExit(info)
    spec = BilayerRunSpec(workdir=wd, gro=Path(info["gro"]), top=Path(info["top"]),
                          gmx_cmd=gmx, temp_K=args.temp, prod_ns=args.prod_ns,
                          n_threads=args.threads)
    audit = run_bilayer(spec)
    print(json.dumps({k: v for k, v in audit.items() if k != "log"}, indent=2))
    if audit.get("status") != "ok":
        print("LOG:\n", audit.get("log", "")[-2000:])
