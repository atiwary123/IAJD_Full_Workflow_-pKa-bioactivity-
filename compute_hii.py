"""
compute_hii.py — Module C driver: H_II / non-bilayer propensity of a lipid (or IAJD)
mixture via Martini-3 self-assembly (build prompt §3 Module C).

Method: insert N copies of (protonated ionizable species + anionic endosomal lipid
±DOPE ±cholesterol) randomly in a box, solvate, run a long unbiased MD, and classify the
self-assembled topology as LAMELLAR (bilayer) vs NON-BILAYER (inverted-hexagonal /
inverted-micellar / cubic). Self-assembly (random start) is used deliberately — a
pre-formed flat bilayer is kinetically trapped lamellar, whereas a random start relaxes to
the true preferred phase.

Non-bilayer metric (documented, semiquantitative — H_II is the leading HYPOTHESIS, not
law; see build prompt §8): from the equilibrated water topology,
  - water_largest_frac  = (largest water cluster) / (total water).  Lamellar water
    percolates (~1) through one connected region; inverted phases fragment water into
    channels/droplets (<1).
  - water_shape         = shape of the largest water region via its gyration-tensor
    eigenvalues l1>=l2>=l3: a SLAB (lamellar) is oblate (l1~l2 >> l3); a TUBE (H_II) is
    prolate (l1 >> l2~l3); a BLOB (micelle) is isotropic. bilayer-likeness =
    (l2/l1)*(1 - l3/l1) is high only for a slab.
  phys_HII_score = 0.5*(1 - water_largest_frac) + 0.5*(1 - bilayer_likeness)  in [0,1];
  phys_nonbilayer = phys_HII_score > 0.5 (bool).

Validation gate (mandatory before use): the DOPE-containing mix must score NON-BILAYER and
the DOPC/PEG mix LAMELLAR (Hafez-Cullis). A failing metric stays NaN, not a guess.

No-proxy: real self-assembly + real topology metric, or NaN+audit. Reuses
martini/run_selfassembly.py (GROMACS driver) + martini/analyze_md.cluster_by_cutoff.
Thermal/janitor identical to the other drivers. Validation MD is long (self-assembly) —
runs serialized after the titrations.
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

from physics_design.lipid_library import load_lipid, write_single_itp, Lipid
from physics_design.bilayer import lipid_template, write_gro, read_gro, _link_ff
from martini.run_selfassembly import _resolve_gmx_cmd, gromacs_available
from martini.analyze_md import cluster_by_cutoff

DESIGN_DIR = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design" / "hii"
WORKROOT = ROOT / "physics_cache" / "hii_runs"
FF_DIR = ROOT / "martini" / "ff"

# Validation mixtures (build prompt §3 Module C): DOPE-rich must go non-bilayer,
# DOPC-rich must stay lamellar.
VALIDATION_MIXES = {
    "DOPE_rich": [("DOPE", 0.8), ("DOPC", 0.2)],
    "DOPC_rich": [("DOPC", 1.0)],
}


def _single_lipid_gro(lip: Lipid, path: Path) -> None:
    """One CG lipid in an extended conformation (for gmx insert-molecules)."""
    coords = lipid_template(lip)            # head-up template, nm
    coords = coords - coords.min(axis=0) + 0.3
    box = tuple(float(coords[:, d].max() + 0.3) for d in range(3))
    write_gro(coords, lip.bead_names, [lip.name] * lip.n_beads, box, path, title=lip.name)


def _insert(gmx, workdir, ci, n, box, out, append: Optional[Path] = None) -> bool:
    cmd = gmx + ["insert-molecules", "-ci", str(ci), "-nmol", str(n),
                 "-box", f"{box[0]}", f"{box[1]}", f"{box[2]}", "-o", str(out)]
    if append is not None:
        cmd = gmx + ["insert-molecules", "-f", str(append), "-ci", str(ci),
                     "-nmol", str(n), "-o", str(out)]
    r = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, timeout=600)
    return out.exists()


SA_MDP = {
    "em": """integrator=steep
nsteps=5000
emtol=100
nstlist=20
cutoff-scheme=Verlet
coulombtype=reaction-field
rcoulomb=1.1
epsilon_r=15
vdw-modifier=Potential-shift
rvdw=1.1
""",
    "prod": """integrator=md
dt=0.020
nsteps={nsteps}
nstxout-compressed=50000
nstlist=20
cutoff-scheme=Verlet
coulombtype=reaction-field
rcoulomb=1.1
epsilon_r=15
vdw-modifier=Potential-shift
rvdw=1.1
tcoupl=v-rescale
tc-grps=System
tau-t=1.0
ref-t={temp}
pcoupl=c-rescale
pcoupltype=isotropic
tau-p=4.0
compressibility=3e-4
ref-p=1.0
gen-vel=yes
gen-temp={temp}
gen-seed={seed}
""",
}


def build_and_run(mix: List[Tuple[str, float]], workdir: Path, gmx: List[str], *,
                  n_total: int = 200, prod_ns: float = 400.0, temp: float = 310.0,
                  threads: int = 4, seed: int = 1) -> Dict:
    """Random-insert the mixture, solvate, EM, long unbiased production. Returns paths."""
    workdir.mkdir(parents=True, exist_ok=True)
    _link_ff(workdir)
    box = (12.0, 12.0, 12.0)
    lipids = {}
    counts = {}
    placed = workdir / "packed.gro"
    prev = None
    total_charge = 0.0
    for i, (name, frac) in enumerate(mix):
        lip = load_lipid(name)
        lipids[name] = lip
        n = max(1, int(round(n_total * frac)))
        counts[name] = n
        total_charge += lip.net_charge * n
        sgro = workdir / f"{name}_single.gro"
        _single_lipid_gro(lip, sgro)
        write_single_itp(lip, workdir / f"{name}.itp")
        ok = _insert(gmx, workdir, sgro, n, box, placed, append=prev)
        if not ok:
            return {"status": f"insert_{name}_failed"}
        prev = placed
    # solvate
    solv = workdir / "solv.gro"
    top = workdir / "system.top"
    _write_top(top, mix, counts, 0)
    r = subprocess.run(gmx + ["solvate", "-cp", str(placed), "-cs", str(FF_DIR / "water.gro"),
                              "-o", str(solv), "-p", str(top), "-radius", "0.21"],
                       cwd=str(workdir), capture_output=True, text=True, timeout=600)
    if not solv.exists():
        return {"status": "solvate_failed", "log": (r.stdout + r.stderr)[-1500:]}
    n_water = _count_res(solv, "W")
    _write_top(top, mix, counts, n_water)
    # EM + production
    for step, mdp in (("em", SA_MDP["em"]),
                      ("prod", SA_MDP["prod"].format(nsteps=int(prod_ns * 1000 / 0.02),
                                                     temp=temp, seed=seed))):
        (workdir / f"{step}.mdp").write_text(mdp)
        tpr = workdir / f"{step}.tpr"
        cin = solv if step == "em" else (workdir / "em.gro")
        g = subprocess.run(gmx + ["grompp", "-f", str(workdir / f"{step}.mdp"), "-c", str(cin),
                                  "-p", str(top), "-o", str(tpr), "-maxwarn", "5"],
                           cwd=str(workdir), capture_output=True, text=True, timeout=300)
        if not tpr.exists():
            return {"status": f"grompp_{step}_failed", "log": (g.stdout + g.stderr)[-1500:]}
        md = (["nice", "-n", "10"] + gmx + ["mdrun", "-s", str(tpr), "-deffnm",
              str(workdir / step), "-ntmpi", "1", "-ntomp", str(threads), "-pin", "on"])
        subprocess.run(md, cwd=str(workdir), capture_output=True, text=True,
                       timeout=259200)
        if not (workdir / f"{step}.gro").exists():
            return {"status": f"mdrun_{step}_failed"}
    return {"status": "ok", "final_gro": str(workdir / "prod.gro"),
            "traj": str(workdir / "prod.xtc"), "n_water": n_water, "counts": counts}


def _write_top(path: Path, mix, counts, n_water):
    L = ['#include "martini_v3.0.0.itp"', '#include "martini_v3.0.0_solvents_v1.itp"',
         '#include "martini_v3.0.0_ions_v1.itp"']
    for name, _ in mix:
        L.append(f'#include "{name}.itp"')
    L += ["", "[ system ]", "HII mix", "", "[ molecules ]"]
    for name, _ in mix:
        L.append(f"{name}  {counts[name]}")
    if n_water:
        L.append(f"W  {n_water}")
    path.write_text("\n".join(L) + "\n")


def _count_res(gro: Path, res: str) -> int:
    pos, names, resn, box = read_gro(gro)
    return sum(1 for r in resn if r == res)


# ── non-bilayer topology metric ────────────────────────────────────────────────
def nonbilayer_score(final_gro: Path, last_frac: float = 0.3) -> Dict:
    """Classify the assembled topology from the final config's water geometry."""
    pos, names, resn, box = read_gro(final_gro)
    box = np.array(box)
    wmask = np.array([r == "W" for r in resn])
    wpos = pos[wmask]
    if len(wpos) < 50:
        return {"phys_HII_score": float("nan"), "error": "too_few_water"}
    clusters = cluster_by_cutoff(wpos, box, cutoff_nm=0.65)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    largest = clusters[int(np.argmax([len(c) for c in clusters]))]
    largest_frac = sizes[0] / len(wpos)
    # gyration-tensor shape of the largest water region (PBC-unwrapped about its seed)
    p = wpos[largest].copy()
    p -= box * np.round((p - p[0]) / box)        # unwrap around seed
    p -= p.mean(axis=0)
    eig = np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1]   # l1>=l2>=l3
    l1, l2, l3 = (eig + 1e-9)
    bilayer_likeness = (l2 / l1) * (1.0 - l3 / l1)         # high only for an oblate slab
    hii = 0.5 * (1.0 - largest_frac) + 0.5 * (1.0 - bilayer_likeness)
    return {"phys_HII_score": float(np.clip(hii, 0, 1)),
            "phys_nonbilayer": bool(hii > 0.5),
            "water_largest_frac": round(float(largest_frac), 3),
            "n_water_clusters": len(sizes),
            "water_shape_eig": [round(float(x), 3) for x in eig],
            "bilayer_likeness": round(float(bilayer_likeness), 3)}


def characterize_mix(name: str, mix: List[Tuple[str, float]], *, prod_ns: float,
                     temp: float, threads: int, keep_traj: bool) -> Dict:
    gmx = _resolve_gmx_cmd()
    if not gromacs_available():
        return {"name": name, "phys_HII_score": float("nan"), "error": "gmx_unavailable"}
    wd = WORKROOT / name
    shutil.rmtree(wd, ignore_errors=True)
    t0 = time.time()
    run = build_and_run(mix, wd, gmx, prod_ns=prod_ns, temp=temp, threads=threads)
    rec = {"name": name, "mix": mix, "prod_ns": prod_ns, "temp_K": temp,
           "md_status": run.get("status")}
    if run.get("status") != "ok":
        rec["error"] = run.get("status")
        rec["phys_HII_score"] = float("nan")
        return rec
    rec.update(nonbilayer_score(Path(run["final_gro"])))
    rec["elapsed_min"] = round((time.time() - t0) / 60, 1)
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    (DESIGN_DIR / f"{name}_hii.json").write_text(json.dumps(rec, indent=2, default=str))
    if not keep_traj:
        for pat in ("*.xtc", "*.trr", "*.tpr", "*.edr", "*.cpt", "#*#"):
            for f in wd.glob(pat):
                f.unlink(missing_ok=True)
    return rec


def validate() -> Dict:
    """Gate: DOPE_rich -> non-bilayer, DOPC_rich -> lamellar."""
    res = {}
    for n in ("DOPE_rich", "DOPC_rich"):
        f = DESIGN_DIR / f"{n}_hii.json"
        if f.exists():
            res[n] = json.loads(f.read_text())
    gate = {}
    if "DOPE_rich" in res and "DOPC_rich" in res:
        de, dc = res["DOPE_rich"], res["DOPC_rich"]
        gate = {"HII_DOPE_rich": de.get("phys_HII_score"),
                "HII_DOPC_rich": dc.get("phys_HII_score"),
                "DOPE_nonbilayer": bool(de.get("phys_nonbilayer")),
                "DOPC_lamellar": bool(not dc.get("phys_nonbilayer", True)),
                "separation": (de.get("phys_HII_score", np.nan) - dc.get("phys_HII_score", np.nan))}
        gate["PASS"] = bool(gate["DOPE_nonbilayer"] and gate["DOPC_lamellar"]
                            and gate.get("separation", 0) > 0.2)
    out = {"results": res, "gate": gate}
    (DESIGN_DIR / "hii_validation_report.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix", help="validation mix name (DOPE_rich/DOPC_rich)")
    p.add_argument("--prod-ns", type=float, default=400.0)
    p.add_argument("--temp", type=float, default=310.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--keep-traj", action="store_true")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate and not args.mix:
        print(json.dumps(validate().get("gate", {}), indent=2, default=str)); return 0
    if not args.mix:
        p.error("need --mix NAME or --validate")
    rec = characterize_mix(args.mix, VALIDATION_MIXES[args.mix], prod_ns=args.prod_ns,
                           temp=args.temp, threads=args.threads, keep_traj=args.keep_traj)
    print(json.dumps({k: v for k, v in rec.items() if k != "log"}, indent=2, default=str))
    if args.validate:
        validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
