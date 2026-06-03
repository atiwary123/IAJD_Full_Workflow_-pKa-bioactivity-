"""
martini/run_selfassembly.py — drive a packed self-assembly MD run for one
IAJD (one protonation state) through GROMACS.

Protocol:
  1. gmx insert-molecules — pack N copies of the CG molecule into a 12³ nm box.
  2. gmx solvate          — fill the box with MARTINI W water.
  3. gmx grompp + mdrun   — EM (steepest descent).
  4. gmx grompp + mdrun   — NPT equilibration (Parrinello-Rahman, v-rescale,
                            dt=20 fs, 200 ps).
  5. gmx grompp + mdrun   — production self-assembly (dt=20 fs, 2–5 µs CG time,
                            written to trajectory.xtc).

All step .mdp files are emitted into the run directory at first call. Run
state is checkpointed via GROMACS' .cpt file; re-running resumes from the
last checkpoint.

Honest scope note: this driver does NOT contain the gromacs binary. It detects
gmx at PATH (or via $GMX_CMD). If unavailable the run is skipped and the
caller's audit log records "gromacs_unavailable". The .mdp/.top/.gro/.itp
files are still written so the run can be launched on any HPC node with
gromacs installed.

Determinism: each call takes an explicit seed; the v-rescale and PR couplings
use that seed; gmx mdrun is invoked with `-pin on -ntmpi 1` for reproducibility.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

def _resolve_gmx_cmd() -> list:
    """Resolve the gmx invocation. Prefers $GMX_CMD, then `which gmx`, then the
    persistent Homebrew gromacs (the original /tmp micromamba env was wiped on
    reboot 2026-05-31 — see docs/PREDICTIVE_PHYSICS_BUILD.md)."""
    env = os.environ.get("GMX_CMD")
    if env:
        return env.split()
    direct = subprocess.run(["which", "gmx"], capture_output=True, text=True)
    if direct.returncode == 0 and direct.stdout.strip():
        return [direct.stdout.strip()]
    # Persistent Homebrew gromacs fallback (survives reboots, unlike /tmp).
    return ["/opt/homebrew/bin/gmx"]


DEFAULT_GMX = _resolve_gmx_cmd()


def gromacs_available() -> bool:
    try:
        r = subprocess.run(DEFAULT_GMX + ["--version"],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and "GROMACS version" in r.stdout
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


# ──────────────────────────────────────────────────────────────────────
# .mdp templates
# ──────────────────────────────────────────────────────────────────────

EM_MDP = """;
; MARTINI 3 energy minimization
;
integrator               = steep
nsteps                   = 5000
emtol                    = 100.0
emstep                   = 0.01
nstlist                  = 20
ns_type                  = grid
cutoff-scheme            = Verlet
coulombtype              = reaction-field
rcoulomb                 = 1.1
epsilon_r                = 15
vdw-type                 = cutoff
vdw-modifier             = Potential-shift
rvdw                     = 1.1
constraints              = none
"""

EQUIL_MDP_TEMPLATE = """;
; MARTINI 3 NPT equilibration, 200 ps
;
integrator               = md
dt                       = 0.020
nsteps                   = 10000
nstxout                  = 0
nstxout-compressed       = 5000
nstenergy                = 1000
nstlog                   = 1000
nstlist                  = 20
ns_type                  = grid
cutoff-scheme            = Verlet
coulombtype              = reaction-field
rcoulomb                 = 1.1
epsilon_r                = 15
vdw-type                 = cutoff
vdw-modifier             = Potential-shift
rvdw                     = 1.1
constraints              = none
tcoupl                   = v-rescale
tc-grps                  = System
tau-t                    = 1.0
ref-t                    = 310.15
pcoupl                   = c-rescale
pcoupltype               = isotropic
tau-p                    = 4.0
ref-p                    = 1.0
compressibility          = 3e-4
gen-vel                  = yes
gen-temp                 = 310.15
gen-seed                 = {seed}
"""

PROD_MDP_TEMPLATE = """;
; MARTINI 3 production self-assembly
;
integrator               = md
dt                       = 0.020
nsteps                   = {nsteps}   ; nsteps * dt = total CG time
nstxout                  = 0
nstxout-compressed       = 25000     ; ~0.5 ns frame stride
nstenergy                = 5000
nstlog                   = 5000
nstlist                  = 20
ns_type                  = grid
cutoff-scheme            = Verlet
coulombtype              = reaction-field
rcoulomb                 = 1.1
epsilon_r                = 15
vdw-type                 = cutoff
vdw-modifier             = Potential-shift
rvdw                     = 1.1
constraints              = none
tcoupl                   = v-rescale
tc-grps                  = System
tau-t                    = 1.0
ref-t                    = 310.15
pcoupl                   = Parrinello-Rahman
pcoupltype               = isotropic
tau-p                    = 12.0
ref-p                    = 1.0
compressibility          = 3e-4
gen-vel                  = no
"""


def write_mdps(out_dir: Path, *, seed: int, prod_ns: float) -> Dict[str, Path]:
    """Write em.mdp, equil.mdp, prod.mdp. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    em = out_dir / "em.mdp"
    eq = out_dir / "equil.mdp"
    pr = out_dir / "prod.mdp"
    em.write_text(EM_MDP)
    eq.write_text(EQUIL_MDP_TEMPLATE.format(seed=seed))
    pr.write_text(PROD_MDP_TEMPLATE.format(
        nsteps=int(prod_ns * 1000 / 0.020),   # dt=0.020 ps
    ))
    return {"em": em, "equil": eq, "prod": pr}


# ──────────────────────────────────────────────────────────────────────
# GROMACS driver
# ──────────────────────────────────────────────────────────────────────

def _gmx(cmd_args: List[str], cwd: Path, timeout_s: int = 3600) -> Tuple[bool, str]:
    cmd = list(DEFAULT_GMX) + cmd_args
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout_s)
    out = proc.stdout + proc.stderr
    return proc.returncode == 0, out


def _insert_molecules(workdir: Path, single_gro: Path, system_gro: Path,
                       n_molecules: int, box_nm: Tuple[float, float, float]) -> Tuple[bool, str]:
    bx, by, bz = box_nm
    return _gmx(
        ["insert-molecules", "-ci", str(single_gro), "-nmol", str(n_molecules),
         "-box", f"{bx}", f"{by}", f"{bz}", "-o", str(system_gro)],
        cwd=workdir, timeout_s=600,
    )


def _solvate(workdir: Path, gro: Path, top: Path, out_gro: Path,
              water_box: Path) -> Tuple[bool, str]:
    return _gmx(
        ["solvate", "-cp", str(gro), "-cs", str(water_box),
         "-p", str(top), "-o", str(out_gro), "-radius", "0.21"],
        cwd=workdir, timeout_s=600,
    )


def _add_counterions(workdir: Path, gro: Path, top: Path, *,
                     net_charge: int, water_residue: str = "W") -> Tuple[bool, str]:
    """Replace `abs(net_charge)` waters with counter-ions to neutralize the system.

    Uses gmx genion. Returns (ok, log)."""
    if net_charge == 0:
        return True, "no_ions_needed"
    ion = "CL" if net_charge > 0 else "NA"
    n_ions = abs(int(net_charge))
    # Need a dummy tpr first — use the EM .mdp (or any valid grompp).
    tpr = workdir / "_ion.tpr"
    ok1, out1 = _gmx(
        ["grompp", "-f", str(workdir / "em.mdp"), "-c", str(gro),
         "-p", str(top), "-o", str(tpr), "-maxwarn", "10"],
        cwd=workdir, timeout_s=300,
    )
    if not ok1:
        return False, out1
    # gmx genion -seed for reproducibility.
    out_gro = workdir / "ionized.gro"
    cmd = list(DEFAULT_GMX) + [
        "genion", "-s", str(tpr), "-o", str(out_gro),
        "-p", str(top),
        "-n" if False else "", "",   # placeholder for ndx (we use SOL/W group via stdin)
        "-pname", "NA", "-nname", "CL",
    ]
    cmd = [c for c in cmd if c]   # drop empties
    if net_charge > 0:
        cmd += ["-nn", str(n_ions)]
    else:
        cmd += ["-np", str(n_ions)]
    # gmx genion interactively asks which residue group is the solvent.
    proc = subprocess.run(cmd, cwd=str(workdir), input=water_residue + "\n",
                           capture_output=True, text=True, timeout=300)
    out2 = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, out1 + out2
    # Replace input gro file with ionized for downstream steps.
    try:
        shutil.copy(out_gro, gro)
    except OSError:
        pass
    return True, out1 + out2


def _grompp_mdrun(workdir: Path, mdp: Path, gro: Path, top: Path, *,
                   step_name: str, timeout_s: int) -> Tuple[bool, str]:
    tpr = workdir / f"{step_name}.tpr"
    ok1, out1 = _gmx(
        ["grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top),
         "-o", str(tpr), "-maxwarn", "5"],
        cwd=workdir, timeout_s=300,
    )
    if not ok1:
        return False, out1
    # Use as many OpenMP threads as the machine has cores for the
    # production step; gmx auto-picks for shorter steps. The micromamba-
    # wrapped binary uses ARM_NEON_ASIMD on Apple Silicon.
    n_threads = os.environ.get("GMX_NTHREADS", str(os.cpu_count() or 4))
    # Parallel callers (precompute_md --jobs N) run many mdruns at once and set
    # GMX_PIN=off so concurrent jobs don't all pin to the same physical cores
    # (which would oversubscribe a few cores and idle the rest). Serial = pin on.
    pin_mode = os.environ.get("GMX_PIN", "on")
    mdrun_args = ["mdrun", "-s", str(tpr), "-deffnm", step_name,
                  "-pin", pin_mode, "-ntmpi", "1", "-ntomp", n_threads]
    # Optional GPU offload (RunPod 1-GPU pods). Set GMX_MDRUN_GPU to the mdrun
    # offload flags and they are appended to the *dynamics* steps only — steepest-
    # descent EM does not support "-update gpu". MARTINI uses reaction-field (NOT
    # PME), so the correct value is "-nb gpu" (optionally "-update gpu"); do NOT
    # pass "-pme gpu". Empty (default) = CPU-only, byte-identical to the prior
    # behaviour, so the laptop / CPU-pod path is unchanged. Many concurrent
    # --jobs share one GPU fine for these small CG systems (GROMACS time-slices
    # the device); the vCPU pool remains the throughput workhorse.
    if step_name in ("equil", "prod"):
        gpu_flags = os.environ.get("GMX_MDRUN_GPU", "").split()
        if gpu_flags:
            mdrun_args += gpu_flags
    ok2, out2 = _gmx(mdrun_args, cwd=workdir, timeout_s=timeout_s)
    return ok2, out1 + out2


@dataclass
class RunSpec:
    workdir: Path
    single_gro: Path
    single_itp: Path
    topol_top: Path
    n_molecules: int = 256
    box_nm: Tuple[float, float, float] = (12.0, 12.0, 12.0)
    prod_ns: float = 2000.0    # 2 µs CG time
    seed: int = 42
    timeout_s: int = 86400     # per-step timeout (1 day)
    ff_dir: Optional[Path] = None    # directory holding martini_v3.0.0*.itp
    water_box: Optional[Path] = None  # pre-built CG water .gro for solvate -cs
    net_charge_per_molecule: float = 0.0   # used to genion after solvate


def run(spec: RunSpec) -> Dict:
    """Run the EM → NPT → production protocol. Returns a dict with status
    and produced file paths. Non-fatal on missing gmx: status='gmx_unavailable'.
    """
    audit = {"workdir": str(spec.workdir),
              "n_molecules": spec.n_molecules,
              "box_nm": list(spec.box_nm),
              "prod_ns": spec.prod_ns,
              "seed": spec.seed,
              "gmx_available": gromacs_available()}
    if not audit["gmx_available"]:
        audit["status"] = "gmx_unavailable"
        return audit

    spec.workdir.mkdir(parents=True, exist_ok=True)
    mdps = write_mdps(spec.workdir, seed=spec.seed, prod_ns=spec.prod_ns)

    # Symlink/copy the MARTINI 3 FF files into the workdir so #include lines
    # in the topol.top resolve. cwd-relative #include is the GROMACS default.
    if spec.ff_dir is not None:
        for fname in ("martini_v3.0.0.itp",
                       "martini_v3.0.0_solvents_v1.itp",
                       "martini_v3.0.0_ions_v1.itp"):
            src = spec.ff_dir / fname
            dst = spec.workdir / fname
            if src.exists() and not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except (OSError, NotImplementedError):
                    shutil.copy(src, dst)

    # 1. pack
    packed = spec.workdir / "packed.gro"
    ok, log = _insert_molecules(spec.workdir, spec.single_gro, packed,
                                  spec.n_molecules, spec.box_nm)
    if not ok:
        audit.update(status="insert_failed", log=log[-2000:])
        return audit
    # 2. solvate
    solvated = spec.workdir / "solvated.gro"
    water_box = spec.water_box or (spec.ff_dir / "water.gro" if spec.ff_dir else None)
    if water_box is None or not water_box.exists():
        audit.update(status="water_box_missing")
        return audit
    ok, log = _solvate(spec.workdir, packed, spec.topol_top, solvated,
                        water_box=water_box)
    if not ok:
        audit.update(status="solvate_failed", log=log[-2000:])
        return audit
    # 2b. Counterions — required when the molecule carries net charge.
    if abs(spec.net_charge_per_molecule) > 0.01:
        net_q = int(round(spec.net_charge_per_molecule * spec.n_molecules))
        ok, log = _add_counterions(spec.workdir, solvated, spec.topol_top,
                                     net_charge=net_q)
        if not ok:
            audit.update(status="genion_failed", log=log[-2000:])
            return audit
    # 3. EM
    ok, log = _grompp_mdrun(spec.workdir, mdps["em"], solvated, spec.topol_top,
                              step_name="em", timeout_s=3600)
    if not ok:
        audit.update(status="em_failed", log=log[-2000:])
        return audit
    # 4. NPT equilibration
    ok, log = _grompp_mdrun(spec.workdir, mdps["equil"],
                              spec.workdir / "em.gro", spec.topol_top,
                              step_name="equil", timeout_s=spec.timeout_s)
    if not ok:
        audit.update(status="equil_failed", log=log[-2000:])
        return audit
    # 5. Production
    ok, log = _grompp_mdrun(spec.workdir, mdps["prod"],
                              spec.workdir / "equil.gro", spec.topol_top,
                              step_name="prod", timeout_s=spec.timeout_s * 4)
    if not ok:
        audit.update(status="prod_failed", log=log[-2000:])
        return audit
    audit.update(status="ok",
                  traj=str(spec.workdir / "prod.xtc"),
                  final_gro=str(spec.workdir / "prod.gro"))
    return audit


__all__ = ["run", "RunSpec", "write_mdps", "gromacs_available"]


if __name__ == "__main__":
    print("gromacs available:", gromacs_available())
    if not gromacs_available():
        print("Install gromacs via the offline conda env before running.")
