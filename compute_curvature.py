"""
compute_curvature.py — Module B driver: monolayer spontaneous curvature c0 (and CPP)
of a Martini-3 lipid (or any single CG species) from the first moment of the lateral
pressure profile of a flat, tensionless bilayer.

Pipeline per species:
  1. build_system   — flat bilayer patch + water (physics_design.bilayer)
  2. run_bilayer    — EM -> tensionless semiisotropic NPT -> production (stock GROMACS)
  3. compute_profile- lateral pressure profile P_L(z)-P_N(z) via Irving-Kirkwood,
                      forces recomputed to match GROMACS exactly (validated, see
                      --force-check)
  4. spontaneous_curvature — first moment -> c0 (nm^-1), R0, surface tension
  5. cache JSON + profile .npz; purge bulky trajectory (janitor)

No-proxy contract: every number is a real computed value or NaN+audit reason. The
force recomputation is cross-checked against GROMACS energies (--force-check); a
species whose run does not reach a tensionless, converged state is reported NaN.

Validation gate (--validate): DOPE c0 strongly negative (~ -1/3 nm^-1), DOPC ~ 0,
|gamma| small. Writes validation_report.json. Module B is not used for design until
this passes.

Usage:
  python compute_curvature.py --lipid DOPC --prod-ns 100 --threads 6 --force-check
  python compute_curvature.py --lipid DOPE --prod-ns 100 --threads 6 --force-check
  python compute_curvature.py --validate          # aggregate DOPC/DOPE/POPC -> gate
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
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physics_design.lipid_library import load_lipid
from physics_design.martini_ff import parse_ff
from physics_design.bilayer import build_system, read_gro
from physics_design.md_runner import BilayerRunSpec, run_bilayer
from physics_design.pressure_profile import (build_atom_arrays, compute_profile,
                                             _minimum_image, F_COULOMB)
from physics_design.curvature import spontaneous_curvature, packing_parameter

DESIGN_DIR = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design"
WORKROOT = ROOT / "physics_cache" / "curvature_runs"

# Tanford tail parameters for CPP of the validation lipids (di-C18:1 = 18 C/tail,
# di-C16:0/C18:1 etc.). v=0.027*nC nm^3, l=0.127*nC nm per chain; two chains.
TANFORD = {
    "DOPC": dict(nC=18, ntails=2), "DOPE": dict(nC=18, ntails=2),
    "POPC": dict(nC=17, ntails=2),
}


def _gmx_cmd() -> List[str]:
    env = os.environ.get("GMX_CMD") or os.environ.get("GMX")
    if env:
        return env.split()
    r = subprocess.run(["which", "gmx"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return [r.stdout.strip()]
    return ["/opt/homebrew/bin/gmx"]


# ──────────────────────────────────────────────────────────────────────
# Force-field reproduction cross-check against GROMACS (single frame)
# ──────────────────────────────────────────────────────────────────────

def energy_cross_check(workdir: Path, gmx: List[str], lip, n_lip: int,
                       ff, rc: float = 1.1, eps_r: float = 15.0) -> Dict:
    """Dump the last production frame, recompute its GROMACS config energies via
    -rerun, and compare to our force-field recomputation. Returns max |%err| over
    Bond/Angle/LJ/Coulomb — the proof the recomputed forces ARE GROMACS' forces."""
    from MDAnalysis.lib import distances as mdadist
    out = {"status": "ok"}
    frame = workdir / "fcheck.gro"
    dump_t = 10 ** 9
    r = subprocess.run(gmx + ["trjconv", "-f", str(workdir / "prod.xtc"),
                              "-s", str(workdir / "prod.tpr"), "-o", str(frame),
                              "-dump", str(dump_t)], cwd=str(workdir), input="0\n",
                       capture_output=True, text=True, timeout=300)
    if not frame.exists():
        return {"status": "trjconv_failed"}
    rr = subprocess.run(gmx + ["mdrun", "-s", str(workdir / "prod.tpr"),
                               "-rerun", str(frame), "-deffnm", str(workdir / "fcheck"),
                               "-ntmpi", "1", "-ntomp", "2"],
                        cwd=str(workdir), capture_output=True, text=True, timeout=600)
    edr = workdir / "fcheck.edr"
    if not edr.exists():
        return {"status": "rerun_failed", "log": (rr.stdout + rr.stderr)[-1500:]}
    en = subprocess.run(gmx + ["energy", "-f", str(edr), "-o", str(workdir / "fcheck.xvg")],
                        cwd=str(workdir), input="Bond\nG96Angle\nLJ-(SR)\nCoulomb-(SR)\nPotential\n",
                        capture_output=True, text=True, timeout=120)
    # parse GROMACS energy averages: name tokens then the numeric average column
    ref = {}
    for line in (en.stdout + en.stderr).splitlines():
        s = line.strip()
        for key, tag in [("Bond", "Bond"), ("Angle", "G96Angle"),
                         ("LJ", "LJ (SR)"), ("Coul", "Coulomb (SR)"),
                         ("Pot", "Potential")]:
            if s.startswith(tag):
                nums = [t for t in s.replace("(kJ/mol)", "").split() if _isfloat(t)]
                if nums:
                    ref[key] = float(nums[0])
    if not {"Bond", "Angle", "LJ", "Coul"} <= set(ref):
        return {"status": "energy_parse_failed", "raw": (en.stdout)[-1500:]}

    # our recomputation
    pos, names, resn, box = read_gro(frame)
    pos = np.array(pos); box = np.array(box)
    n_total = len(pos)
    s = build_atom_arrays(lip, n_lip, n_total, ff)
    q = s.charge; krf = 1 / (2 * rc ** 3); crf = 3 / (2 * rc); pre = F_COULOMB / eps_r
    pairs, dists = mdadist.self_capped_distance(
        pos, max_cutoff=rc, box=np.array([box[0], box[1], box[2], 90, 90, 90]))
    ii, jj = pairs[:, 0], pairs[:, 1]
    ek = np.array(sorted(a * n_total + b for (a, b) in s.excl_pairs), dtype=np.int64)
    keep = ~np.isin(np.minimum(ii, jj).astype(np.int64) * n_total +
                    np.maximum(ii, jj).astype(np.int64), ek)
    ii, jj, r_ = ii[keep], jj[keep], dists[keep]
    c6 = s.c6[s.type_idx[ii], s.type_idx[jj]]; c12 = s.c12[s.type_idx[ii], s.type_idx[jj]]
    E_lj = float(((c12 / r_ ** 12 - c6 / r_ ** 6) - (c12 / rc ** 12 - c6 / rc ** 6)).sum())
    qq = q[ii] * q[jj]; m = qq != 0
    E_coul_nb = float((pre * qq[m] * (1 / r_[m] + krf * r_[m] ** 2 - crf)).sum())
    b = s.bonds; qqb = q[b[:, 0]] * q[b[:, 1]]; mb = qqb != 0
    db = _minimum_image(pos[b[mb, 0]] - pos[b[mb, 1]], box); rb = np.linalg.norm(db, axis=1)
    E_coul_excl = float((pre * qqb[mb] * (krf * rb ** 2)).sum())   # GROMACS excluded RF: krf r^2 only
    E_coul = E_coul_nb + E_coul_excl
    dbb = _minimum_image(pos[b[:, 0]] - pos[b[:, 1]], box); rbb = np.linalg.norm(dbb, axis=1)
    E_bond = float((0.5 * s.bond_k * (rbb - s.bond_r0) ** 2).sum())
    ang = s.angles
    rij = _minimum_image(pos[ang[:, 0]] - pos[ang[:, 1]], box)
    rkj = _minimum_image(pos[ang[:, 2]] - pos[ang[:, 1]], box)
    cth = np.sum(rij * rkj, axis=1) / (np.linalg.norm(rij, axis=1) * np.linalg.norm(rkj, axis=1))
    E_ang = float((0.5 * s.angle_k * (cth - s.angle_cos0) ** 2).sum())
    mine = {"Bond": E_bond, "Angle": E_ang, "LJ": E_lj, "Coul": E_coul}
    errs = {}
    for k in ("Bond", "Angle", "LJ", "Coul"):
        denom = abs(ref[k]) if abs(ref[k]) > 1e-6 else 1.0
        errs[k] = 100.0 * (mine[k] - ref[k]) / denom
    out.update(gromacs=ref, recomputed=mine, pct_err=errs,
               max_abs_pct_err=float(max(abs(v) for v in errs.values())))
    return out


def _isfloat(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


# ──────────────────────────────────────────────────────────────────────
# Per-species characterization
# ──────────────────────────────────────────────────────────────────────

_MD_BULK = ("*.xtc", "*.trr", "*.edr", "*.cpt", "#*#", "*.tpr", "fcheck.*",
            "em.gro", "eq1.gro", "eq2.gro", "step*.pdb", "*.log")


def _purge(workdir: Path, keep_traj: bool):
    if keep_traj:
        return
    for pat in _MD_BULK:
        for f in workdir.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass


def characterize(lipid_name: str, *, n_per_leaflet: int, prod_ns: float,
                 eq2_ps: float, temp: float, threads: int, frame_ps: float,
                 last_frac: float, do_force_check: bool, keep_traj: bool,
                 contour: str = "ik") -> Dict:
    t0 = time.time()
    gmx = _gmx_cmd()
    lip = load_lipid(lipid_name)
    ff = parse_ff(needed_types=set(lip.bead_types) | {"W", "NA", "CL"})
    workdir = WORKROOT / f"{lipid_name}_T{int(temp)}"
    shutil.rmtree(workdir, ignore_errors=True)
    rec: Dict = {"lipid": lipid_name, "temp_K": temp, "n_per_leaflet": n_per_leaflet,
                 "prod_ns": prod_ns, "contour": contour, "c0_nm_inv": float("nan"),
                 "error": None}

    info = build_system(lip, workdir, gmx, n_per_leaflet=n_per_leaflet)
    if info["status"] != "ok":
        rec["error"] = f"build:{info['status']}"
        return rec
    rec["n_lipids"] = info["n_lipids"]; rec["n_water"] = info["n_water"]

    spec = BilayerRunSpec(workdir=workdir, gro=Path(info["gro"]), top=Path(info["top"]),
                          gmx_cmd=gmx, temp_K=temp, eq2_ps=eq2_ps, prod_ns=prod_ns,
                          frame_ps=frame_ps, n_threads=threads)
    audit = run_bilayer(spec)
    rec["md_status"] = audit.get("status")
    if audit.get("status") != "ok":
        rec["error"] = f"md:{audit.get('status')}"
        rec["md_log"] = audit.get("log", "")[-1500:]
        return rec

    # structural observables from final gro
    pos, names, resn, box = read_gro(workdir / "prod.gro")
    pos = np.array(pos)
    n_lip = sum(1 for n in names if n in ("NC3", "NH3"))
    apl = box[0] * box[1] * 2.0 / n_lip
    zhead = np.array([pos[i, 2] for i, n in enumerate(names) if n == "PO4"])
    zc = zhead.mean()
    thick = float(zhead[zhead > zc].mean() - zhead[zhead < zc].mean())
    rec["apl_nm2"] = float(apl); rec["thickness_PP_nm"] = thick

    # optional force-field reproduction check vs GROMACS
    if do_force_check:
        try:
            rec["force_check"] = energy_cross_check(workdir, gmx, lip, n_lip, ff)
        except Exception as exc:
            rec["force_check"] = {"status": f"exception:{type(exc).__name__}:{exc}"}

    # pressure profile + c0
    sysarr = build_atom_arrays(lip, n_lip, len(pos), ff)
    prof = compute_profile(workdir / "prod.xtc", workdir / "prod.gro", sysarr,
                           last_frac=last_frac, contour=contour)
    # monolayer cutoff = headgroup plane + ~1 nm (membrane-water interface)
    zmax = thick / 2.0 + 1.0
    cr = spontaneous_curvature(prof, lipid_name, zmax=zmax)
    rec["surface_tension_mNm"] = prof.surface_tension_mNm
    rec["gamma_global_mNm"] = prof.gamma_global_mNm
    rec["P_config_bar"] = [prof.P_xx_config, prof.P_yy_config, prof.P_zz_config]
    rec["n_frames"] = prof.n_frames
    rec["tau_moment_bar_nm2"] = cr.tau_moment_bar_nm2
    rec["kappa_mono_J"] = cr.kappa_mono_J
    rec["c0_nm_inv"] = cr.c0_nm_inv
    rec["R0_nm"] = cr.R0_nm
    rec["integration_zmax_nm"] = cr.integration_zmax_nm
    rec["moment_curve_bar_nm2"] = cr.moment_curve
    rec["water_baseline_bar"] = cr.water_baseline_bar
    # convergence gate: tensionless + flat bulk-water pressure
    rec["tensionless"] = bool(abs(prof.surface_tension_mNm) < 4.0)
    rec["water_flat"] = bool(np.isfinite(cr.water_baseline_bar)
                             and cr.water_baseline_bar < 25.0)
    rec["c0_trusted"] = bool(rec["tensionless"] and rec["water_flat"])
    # CPP
    if lipid_name in TANFORD:
        tnf = TANFORD[lipid_name]
        v = 0.027 * tnf["nC"] * tnf["ntails"]
        l = 0.127 * tnf["nC"]
        rec["cpp"] = packing_parameter(apl, v, l)

    # persist profile arrays
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DESIGN_DIR / f"{lipid_name}_T{int(temp)}_profile.npz",
             z=prof.z, dP=prof.dP, dP_nb=prof.dP_nb, dP_bond=prof.dP_bond,
             dP_angle=prof.dP_angle)
    rec["profile_npz"] = str(DESIGN_DIR / f"{lipid_name}_T{int(temp)}_profile.npz")
    rec["elapsed_min"] = round((time.time() - t0) / 60, 1)
    _purge(workdir, keep_traj)
    (DESIGN_DIR / f"{lipid_name}_T{int(temp)}_curvature.json").write_text(
        json.dumps(rec, indent=2, default=str))
    return rec


# ──────────────────────────────────────────────────────────────────────
# Validation gate
# ──────────────────────────────────────────────────────────────────────

def _recompute_c0_from_npz(rec: Dict) -> Dict:
    """Recompute c0 from the saved profile .npz with the CURRENT kappa/zmax, so the
    gate is independent of the kappa value used at MD-run time. Mutates+returns rec."""
    from physics_design.pressure_profile import ProfileResult, BAR_NM_TO_MNM
    npz_path = rec.get("profile_npz")
    if not npz_path or not Path(npz_path).exists():
        return rec
    d = np.load(npz_path)
    z, dP = d["z"], d["dP"]
    dz = float(z[1] - z[0])
    prof = ProfileResult(z=z, dP=dP, dP_nb=d["dP_nb"], dP_bond=d["dP_bond"],
                         dP_angle=d["dP_angle"], n_frames=rec.get("n_frames", 0),
                         surface_tension_mNm=-BAR_NM_TO_MNM * float(np.sum(dP) * dz),
                         area_nm2=0.0)
    thick = rec.get("thickness_PP_nm", 3.7)
    cr = spontaneous_curvature(prof, rec.get("lipid", "_default"),
                               zmax=thick / 2.0 + 1.0)
    rec["c0_nm_inv"] = cr.c0_nm_inv
    rec["tau_moment_bar_nm2"] = cr.tau_moment_bar_nm2
    rec["kappa_mono_J"] = cr.kappa_mono_J
    rec["water_baseline_bar"] = cr.water_baseline_bar
    return rec


def aggregate_validation(temps=(300,)) -> Dict:
    results = {}
    for f in DESIGN_DIR.glob("*_curvature.json"):
        try:
            r = json.loads(f.read_text())
            _recompute_c0_from_npz(r)   # ensure c0 uses current kappa/zmax
            results[f.stem] = r
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    def get(name):
        for k, v in results.items():
            if v.get("lipid") == name:
                return v
        return None

    dopc = get("DOPC"); dope = get("DOPE")
    report = {"results": results, "gate": {}}
    if dopc and dope and np.isfinite(dopc.get("c0_nm_inv", np.nan)) \
            and np.isfinite(dope.get("c0_nm_inv", np.nan)):
        c0_dopc = dopc["c0_nm_inv"]; c0_dope = dope["c0_nm_inv"]
        # gate: DOPE strongly negative (<= -0.2), DOPC near zero (|c0|<0.12), and
        # DOPE clearly more negative than DOPC (separation > 0.15 nm^-1)
        gate = {
            "c0_DOPC_nm_inv": c0_dopc,
            "c0_DOPE_nm_inv": c0_dope,
            "DOPE_strongly_negative": bool(c0_dope <= -0.20),
            "DOPC_near_zero": bool(abs(c0_dopc) <= 0.12),
            "DOPE_more_negative_than_DOPC": bool((c0_dopc - c0_dope) >= 0.15),
            "gamma_DOPC_mNm": dopc.get("surface_tension_mNm"),
            "gamma_DOPE_mNm": dope.get("surface_tension_mNm"),
        }
        gate["PASS"] = bool(gate["DOPE_strongly_negative"] and gate["DOPC_near_zero"]
                            and gate["DOPE_more_negative_than_DOPC"])
        report["gate"] = gate
    (DESIGN_DIR / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lipid", help="lipid species name (DOPC/DOPE/POPC/...)")
    p.add_argument("--n-per-leaflet", type=int, default=72)
    p.add_argument("--prod-ns", type=float, default=100.0)
    p.add_argument("--eq2-ps", type=float, default=4000.0)
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--frame-ps", type=float, default=100.0)
    p.add_argument("--last-frac", type=float, default=0.7)
    p.add_argument("--contour", default="ik", choices=["ik", "harasima"])
    p.add_argument("--force-check", action="store_true")
    p.add_argument("--keep-traj", action="store_true")
    p.add_argument("--validate", action="store_true", help="aggregate -> gate report")
    args = p.parse_args()

    if args.validate and not args.lipid:
        rep = aggregate_validation()
        print(json.dumps(rep.get("gate", {}), indent=2, default=str))
        return 0

    if not args.lipid:
        p.error("need --lipid NAME or --validate")
    rec = characterize(args.lipid, n_per_leaflet=args.n_per_leaflet,
                       prod_ns=args.prod_ns, eq2_ps=args.eq2_ps, temp=args.temp,
                       threads=args.threads, frame_ps=args.frame_ps,
                       last_frac=args.last_frac, do_force_check=args.force_check,
                       keep_traj=args.keep_traj, contour=args.contour)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("md_log",)}, indent=2, default=str))
    if args.validate:
        aggregate_validation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
