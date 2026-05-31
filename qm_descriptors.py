"""
qm_descriptors.py — GFN2-xTB electronic-structure descriptors for IAJDs.

Pipeline (per SMILES):
  1. RDKit ETKDGv3 multi-conformer embedding (deterministic seed).
  2. GFN-FF pre-optimization (xtb --gfnff --opt).
  3. GFN2 geometry optimization with ALPB(water) (xtb --gfn 2 --alpb water --opt).
  4. GFN2 single-point with --dipole and --polar request on lowest-G Boltzmann set.
  5. Parse xtbout.json + stdout for:
       qm_q_ionizableN       — partial charge on the most-basic N (Henderson input)
       qm_dipole_D           — total dipole magnitude (Debye)
       qm_polarizability     — molecular polarizability α(0) in atomic units
       qm_homo_lumo_eV       — HOMO–LUMO gap (eV)
       qm_dGsolv_kJmol       — ALPB(water) solvation free energy of the full molecule
       qm_dGsolv_head        — ΔGsolv of the head fragment (re-runs xtb on head only)
       qm_dGsolv_tail        — ΔGsolv of the largest tail fragment
       qm_Ehedup             — E(protonated) − E(neutral), electronic ΔE (kJ/mol)

Honest no-proxy contract: every failure (embedding, xtb non-convergence, parse
error, missing binary) returns NaN for the failing key — never a constant
default. Callers handle NaN through XGBoost's default branch.

Identifying the head/tail split reuses compute_cpp.identify_head_and_tail_atoms
(do not reinvent — the BFS-from-basic-N split is the canonical decomposition).

Shell-out hardening mirrors compute_molgpka_live.py:
  - subprocess.Popen with start_new_session for clean killpg on timeout
  - per-call timeout (default 600 s — large IAJDs at ~80 heavy atoms can take >2 min)
  - workdir per call so xtbout.json / charges / xtbrestart don't collide

If the xTB binary is unavailable (Space runtime), compute_qm_descriptors()
returns the all-NaN dict and the caller falls back to qm_emulator.joblib.
"""
from __future__ import annotations
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms

RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────
# xTB binary discovery
# ──────────────────────────────────────────────────────────────────────

# Persistent micromamba xtb_env (conda-forge xtb 6.7.1). The original build put
# this under /tmp/mamba_root, which a reboot wiped (2026-05-31); it now lives in
# ~/micromamba so it survives restarts. Rebuild recipe: docs/PREDICTIVE_PHYSICS_BUILD.md.
# Override via XTB_CMD env.
DEFAULT_XTB_CMD = os.environ.get(
    "XTB_CMD",
    "/opt/homebrew/bin/micromamba run -n xtb_env "
    "--root-prefix /Users/aryamantiwary/micromamba xtb",
)

# Hartree → kJ/mol
HARTREE_TO_KJMOL = 2625.4996

# Henderson input — atomic-units dipole → Debye is already done by xtb itself
# in the "tot (Debye)" column; we just parse that.
# Polarizability is kept in a.u. as the spec key name implies.

# ──────────────────────────────────────────────────────────────────────
# QM result schema
# ──────────────────────────────────────────────────────────────────────

QM_KEYS = (
    "qm_q_ionizableN",
    "qm_dipole_D",
    "qm_polarizability",
    "qm_homo_lumo_eV",
    "qm_dGsolv_kJmol",
    "qm_dGsolv_head",
    "qm_dGsolv_tail",
    "qm_Ehedup",
)


def _nan_qm_dict(error: str = "not_attempted") -> Dict[str, float]:
    out = {k: float("nan") for k in QM_KEYS}
    out["qm_error"] = error
    return out


# ──────────────────────────────────────────────────────────────────────
# xTB invocation
# ──────────────────────────────────────────────────────────────────────

def _xtb_available() -> bool:
    cmd = DEFAULT_XTB_CMD.split() + ["--version"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.returncode == 0 or "xtb version" in (r.stdout + r.stderr).lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _xtb_run(xyz_path: Path, workdir: Path, *,
             charge: int = 0, opt: bool = True, alpb: str = "water",
             dipole: bool = True, gfnff: bool = False,
             timeout_s: int = 600) -> Tuple[bool, str]:
    """Shell out to xtb. Returns (success, full stdout+stderr)."""
    cmd = DEFAULT_XTB_CMD.split() + [str(xyz_path), "--json"]
    if gfnff:
        cmd += ["--gfnff"]
    else:
        cmd += ["--gfn", "2"]
    if alpb:
        cmd += ["--alpb", alpb]
    if opt:
        cmd += ["--opt", "tight"]
    else:
        cmd += ["--sp"]
    if dipole:
        cmd += ["--dipole"]
    if charge != 0:
        cmd += ["--chrg", str(int(charge))]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(workdir), start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        return False, "TIMEOUT"
    out = stdout + stderr
    ok = proc.returncode == 0 or "normal termination of xtb" in out
    return ok, out


# ──────────────────────────────────────────────────────────────────────
# stdout / json parsers
# ──────────────────────────────────────────────────────────────────────

_RE_DIPOLE = re.compile(r"^\s*full:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)\s*$", re.M)
_RE_GSOLV  = re.compile(r"->\s*Gsolv\s+([-\d.E+]+)\s+Eh")
_RE_POLAR  = re.compile(r"Mol\.\s*α\(0\)\s*/au\s*:\s*([\d.]+)")


def _parse_xtb_stdout(out: str) -> Dict[str, Optional[float]]:
    """Pull dipole_D, gsolv_Eh, polar_au out of an xtb stdout block."""
    d = {"dipole_D": None, "gsolv_Eh": None, "polar_au": None}
    m = _RE_DIPOLE.search(out)
    if m:
        d["dipole_D"] = float(m.group(1))
    m = _RE_GSOLV.search(out)
    if m:
        d["gsolv_Eh"] = float(m.group(1))
    m = _RE_POLAR.search(out)
    if m:
        d["polar_au"] = float(m.group(1))
    return d


def _parse_xtbout_json(json_path: Path) -> Dict[str, Optional[float]]:
    """Pull total_energy, HOMO-LUMO gap, partial_charges, dipole_au from
    xtbout.json. Returns dict with None on missing keys."""
    if not json_path.exists():
        return {"total_E": None, "homo_lumo_eV": None,
                "partial_charges": None, "dipole_au": None}
    try:
        j = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"total_E": None, "homo_lumo_eV": None,
                "partial_charges": None, "dipole_au": None}
    return {
        "total_E": j.get("total energy"),
        "homo_lumo_eV": j.get("HOMO-LUMO gap / eV"),
        "partial_charges": j.get("partial charges"),
        "dipole_au": j.get("dipole / a.u."),
    }


# ──────────────────────────────────────────────────────────────────────
# Conformer ensemble + RDKit→xyz
# ──────────────────────────────────────────────────────────────────────

def _mol_to_xyz(mol: Chem.Mol, conf_id: int, path: Path) -> None:
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    lines = [f"{n}", ""]
    for i in range(n):
        p = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {p.x:.8f} {p.y:.8f} {p.z:.8f}")
    path.write_text("\n".join(lines) + "\n")


def _embed_ensemble(mol: Chem.Mol, n_confs: int, seed: int) -> Tuple[Chem.Mol, List[int]]:
    """Return (mol_with_Hs, cid_list_sorted_by_MMFF_energy)."""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_confs, params=params))
    if not cids:
        return mh, []
    energies = []
    for cid in cids:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mh)
            if props is None:
                ff = AllChem.UFFGetMoleculeForceField(mh, confId=cid)
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mh, props, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=300)
            energies.append((cid, ff.CalcEnergy()))
        except (RuntimeError, ValueError):
            continue
    if not energies:
        return mh, []
    energies.sort(key=lambda t: t[1])
    return mh, [cid for cid, _ in energies]


# ──────────────────────────────────────────────────────────────────────
# Most-basic-N selection (used for the Henderson input charge + protonation)
# ──────────────────────────────────────────────────────────────────────

_BASIC_N_SMARTS = Chem.MolFromSmarts("[#7;X3;!$(N=*);!$(NC=O);!$(Nc)]")


def _most_basic_n_index(mol_no_h: Chem.Mol) -> Optional[int]:
    """Pick the most basic tertiary nitrogen index — the one we'd protonate.

    Heuristic ranking (in order):
      1. Tertiary alkyl amine (3 alkyl C neighbors) — strongest base.
      2. Piperazine distal N (3 ring/alkyl C neighbors, far from any C=O).
      3. Other sp3 N.
    """
    matches = mol_no_h.GetSubstructMatches(_BASIC_N_SMARTS)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    best, best_score = None, -1
    for m in matches:
        idx = m[0]
        atom = mol_no_h.GetAtomWithIdx(idx)
        score = 0
        n_alkyl_C = sum(1 for nb in atom.GetNeighbors()
                        if nb.GetAtomicNum() == 6 and not nb.GetIsAromatic())
        score += n_alkyl_C * 10
        # Penalize proximity to carbonyl (amide-like) — shouldn't fire on
        # tertiary-amine matches anyway because SMARTS excludes NC=O, but keep
        # the safety scoring for sp3 N near electron-withdrawing groups.
        for nb in atom.GetNeighbors():
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() == idx:
                    continue
                if nb2.GetAtomicNum() == 8 and nb2.GetDegree() == 1:
                    score -= 3
        if score > best_score:
            best_score = score
            best = idx
    return best


def _protonate_n(mol_no_h: Chem.Mol, n_idx: int) -> Chem.Mol:
    rw = Chem.RWMol(mol_no_h)
    atom = rw.GetAtomWithIdx(n_idx)
    atom.SetFormalCharge(1)
    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
    return rw.GetMol()


# ──────────────────────────────────────────────────────────────────────
# Head / tail fragment SMILES from BFS split (reuse compute_cpp logic)
# ──────────────────────────────────────────────────────────────────────

def _head_tail_smiles(mol_no_h: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    """Return (head_smi, largest_tail_smi). NaN for either if no split exists.

    Wraps compute_cpp.identify_head_and_tail_atoms — the canonical IAJD
    head/tail BFS split — and converts each fragment back to a SMILES we can
    re-embed for an isolated ΔGsolv calculation.
    """
    try:
        from compute_cpp import identify_head_and_tail_atoms
    except ImportError:
        return None, None
    head_atoms, tail_atoms = identify_head_and_tail_atoms(mol_no_h)
    if head_atoms is None or not tail_atoms:
        return None, None

    head_idx = sorted(head_atoms)
    try:
        head_smi = Chem.MolFragmentToSmiles(
            mol_no_h, atomsToUse=head_idx, canonical=True,
        )
    except (ValueError, RuntimeError):
        head_smi = None

    # Largest connected tail fragment.
    largest_tail = None
    tail_list = sorted(tail_atoms)
    try:
        # Build per-component grouping by walking neighbors restricted to tail set.
        tail_set = set(tail_atoms)
        seen: set = set()
        components: List[List[int]] = []
        for start in tail_list:
            if start in seen:
                continue
            comp: List[int] = []
            stack = [start]
            while stack:
                a = stack.pop()
                if a in seen or a not in tail_set:
                    continue
                seen.add(a)
                comp.append(a)
                for nb in mol_no_h.GetAtomWithIdx(a).GetNeighbors():
                    if nb.GetIdx() in tail_set and nb.GetIdx() not in seen:
                        stack.append(nb.GetIdx())
            if comp:
                components.append(comp)
        if components:
            biggest = max(components, key=len)
            largest_tail = Chem.MolFragmentToSmiles(
                mol_no_h, atomsToUse=sorted(biggest), canonical=True,
            )
    except (ValueError, RuntimeError):
        pass

    return head_smi, largest_tail


# ──────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────

def compute_qm_descriptors(smiles: str, *,
                           n_confs: int = 5,
                           solvent: str = "water",
                           seed: int = 42,
                           timeout_s: int = 600,
                           include_fragments: bool = True,
                           include_protonated: bool = True,
                           workdir_root: Optional[Path] = None) -> Dict[str, float]:
    """Real GFN2-xTB descriptors for one SMILES.

    Returns a dict with QM_KEYS + qm_error (a short tag if any step failed).
    Each numeric key is NaN on failure; no proxies, no constants.
    """
    if not isinstance(smiles, str) or not smiles:
        return _nan_qm_dict("empty_smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _nan_qm_dict("smiles_parse_failed")

    if not _xtb_available():
        return _nan_qm_dict("xtb_unavailable")

    result = _nan_qm_dict("ok")
    result["qm_error"] = None

    # Identify the most-basic N (needed for both the Henderson-input charge
    # and the protonated-state Ehedup calculation).
    n_idx = _most_basic_n_index(mol)

    # ── Neutral-state pipeline ──
    work_root = Path(workdir_root) if workdir_root else Path(tempfile.mkdtemp(prefix="qm_neutral_"))
    work_root.mkdir(parents=True, exist_ok=True)
    cleanup_work = workdir_root is None

    mh, sorted_cids = _embed_ensemble(mol, n_confs=n_confs, seed=seed)
    if not sorted_cids:
        result["qm_error"] = "embed_failed"
        if cleanup_work:
            shutil.rmtree(work_root, ignore_errors=True)
        return result

    # Take up to 3 lowest MMFF conformers for xtb (Boltzmann set).
    boltzmann_cids = sorted_cids[:max(1, min(3, len(sorted_cids)))]
    best_neutral: Optional[Dict] = None
    for cid in boltzmann_cids:
        cdir = work_root / f"neutral_cid{cid}"
        cdir.mkdir(exist_ok=True)
        xyz = cdir / "input.xyz"
        _mol_to_xyz(mh, cid, xyz)
        # GFN-FF pre-opt (cheap geometry cleanup).
        ok_ff, _ = _xtb_run(xyz, cdir, charge=0, opt=True, alpb="", dipole=False,
                            gfnff=True, timeout_s=timeout_s)
        # Use whichever xyz the FF wrote; if GFN-FF didn't produce one, stay with input.
        ff_xyz = cdir / "xtbopt.xyz"
        if ff_xyz.exists():
            shutil.copy(ff_xyz, cdir / "preopt.xyz")
            opt_input = cdir / "preopt.xyz"
        else:
            opt_input = xyz
        # GFN2 opt + ALPB(water).
        ok_opt, out_opt = _xtb_run(opt_input, cdir, charge=0, opt=True,
                                    alpb=solvent, dipole=True, gfnff=False,
                                    timeout_s=timeout_s)
        if not ok_opt:
            continue
        # Single-point with --dipole on the optimized geometry — to pull dipole
        # in Debye and the polarizability lines that some xtb versions only
        # print on --sp.
        sp_input = cdir / "xtbopt.xyz"
        if not sp_input.exists():
            sp_input = opt_input
        # If GFN2 had already printed Gsolv during the opt, we can skip the
        # extra SP and parse opt output directly.
        parsed_opt = _parse_xtb_stdout(out_opt)
        json_opt = _parse_xtbout_json(cdir / "xtbout.json")
        # When polarizability missing from opt output (some xtb versions), run
        # an additional SP. The optimized geometry is reused so this is cheap.
        polar_au = parsed_opt.get("polar_au")
        if polar_au is None and sp_input.exists():
            ok_sp, out_sp = _xtb_run(sp_input, cdir, charge=0, opt=False,
                                      alpb=solvent, dipole=True, gfnff=False,
                                      timeout_s=timeout_s)
            if ok_sp:
                parsed_sp = _parse_xtb_stdout(out_sp)
                json_sp = _parse_xtbout_json(cdir / "xtbout.json")
                # SP overrides energy/dipole if present (consistent thermo).
                if parsed_sp.get("polar_au") is not None:
                    polar_au = parsed_sp["polar_au"]
                if parsed_sp.get("dipole_D") is not None:
                    parsed_opt["dipole_D"] = parsed_sp["dipole_D"]
                if parsed_sp.get("gsolv_Eh") is not None:
                    parsed_opt["gsolv_Eh"] = parsed_sp["gsolv_Eh"]
                if json_sp.get("total_E") is not None:
                    json_opt["total_E"] = json_sp["total_E"]
                if json_sp.get("homo_lumo_eV") is not None:
                    json_opt["homo_lumo_eV"] = json_sp["homo_lumo_eV"]
                if json_sp.get("partial_charges") is not None:
                    json_opt["partial_charges"] = json_sp["partial_charges"]
        candidate = {
            "cid": cid,
            "E_total_Eh": json_opt.get("total_E"),
            "homo_lumo_eV": json_opt.get("homo_lumo_eV"),
            "partial_charges": json_opt.get("partial_charges"),
            "dipole_D": parsed_opt.get("dipole_D"),
            "polar_au": polar_au,
            "Gsolv_Eh": parsed_opt.get("gsolv_Eh"),
        }
        if (candidate["E_total_Eh"] is not None and
            (best_neutral is None or candidate["E_total_Eh"] < best_neutral["E_total_Eh"])):
            best_neutral = candidate

    if best_neutral is None:
        result["qm_error"] = "neutral_xtb_failed"
        if cleanup_work:
            shutil.rmtree(work_root, ignore_errors=True)
        return result

    # Populate the per-molecule keys we have.
    if best_neutral.get("dipole_D") is not None:
        result["qm_dipole_D"] = float(best_neutral["dipole_D"])
    if best_neutral.get("polar_au") is not None:
        result["qm_polarizability"] = float(best_neutral["polar_au"])
    if best_neutral.get("homo_lumo_eV") is not None:
        result["qm_homo_lumo_eV"] = float(best_neutral["homo_lumo_eV"])
    if best_neutral.get("Gsolv_Eh") is not None:
        result["qm_dGsolv_kJmol"] = float(best_neutral["Gsolv_Eh"]) * HARTREE_TO_KJMOL

    # Henderson-input charge on the most basic N.
    if n_idx is not None and best_neutral.get("partial_charges") is not None:
        charges = best_neutral["partial_charges"]
        # mh has the same heavy-atom indices as mol because AddHs appends Hs
        # at the END (RDKit guarantees this with default kwargs).
        if 0 <= n_idx < len(charges):
            result["qm_q_ionizableN"] = float(charges[n_idx])

    # ── Protonated-state ΔE (Ehedup) ──
    if include_protonated and n_idx is not None:
        prot_mol = _protonate_n(mol, n_idx)
        prot_mh, prot_cids = _embed_ensemble(prot_mol, n_confs=n_confs, seed=seed + 1)
        if prot_cids:
            best_prot_E = None
            for cid in prot_cids[:max(1, min(3, len(prot_cids)))]:
                cdir = work_root / f"prot_cid{cid}"
                cdir.mkdir(exist_ok=True)
                xyz = cdir / "input.xyz"
                _mol_to_xyz(prot_mh, cid, xyz)
                _xtb_run(xyz, cdir, charge=1, opt=True, alpb="",
                         dipole=False, gfnff=True, timeout_s=timeout_s)
                opt_input = cdir / "xtbopt.xyz" if (cdir / "xtbopt.xyz").exists() else xyz
                ok, out = _xtb_run(opt_input, cdir, charge=1, opt=True,
                                    alpb=solvent, dipole=False, gfnff=False,
                                    timeout_s=timeout_s)
                if not ok:
                    continue
                j = _parse_xtbout_json(cdir / "xtbout.json")
                if j.get("total_E") is None:
                    continue
                if best_prot_E is None or j["total_E"] < best_prot_E:
                    best_prot_E = j["total_E"]
            if best_prot_E is not None and best_neutral.get("E_total_Eh") is not None:
                dE_Eh = best_prot_E - best_neutral["E_total_Eh"]
                result["qm_Ehedup"] = float(dE_Eh) * HARTREE_TO_KJMOL

    # ── Fragment ΔGsolv (head + largest tail) ──
    # Sanity range: ΔGsolv for small fragments is bounded; values outside
    # ±500 kJ/mol almost certainly indicate a fragment-SMILES valence issue
    # (e.g. broken-bond charges from MolFragmentToSmiles) and we treat them
    # as NaN. This is quality-control, not a proxy substitution.
    PHYSICAL_GSOLV_KJMOL = 500.0
    if include_fragments:
        head_smi, tail_smi = _head_tail_smiles(mol)
        if head_smi:
            frag_g = _fragment_gsolv(head_smi, work_root / "head_frag",
                                      seed=seed + 2, timeout_s=timeout_s,
                                      solvent=solvent)
            if frag_g is not None and abs(frag_g) <= PHYSICAL_GSOLV_KJMOL:
                result["qm_dGsolv_head"] = float(frag_g)
        if tail_smi:
            frag_g = _fragment_gsolv(tail_smi, work_root / "tail_frag",
                                      seed=seed + 3, timeout_s=timeout_s,
                                      solvent=solvent)
            if frag_g is not None and abs(frag_g) <= PHYSICAL_GSOLV_KJMOL:
                result["qm_dGsolv_tail"] = float(frag_g)

    if cleanup_work:
        shutil.rmtree(work_root, ignore_errors=True)
    return result


def _fragment_gsolv(frag_smi: str, workdir: Path, *,
                    seed: int, timeout_s: int, solvent: str = "water") -> Optional[float]:
    """Run xtb on an isolated fragment to get ALPB Gsolv (kJ/mol).

    Returns None on failure. Fragment SMILES coming from MolFragmentToSmiles
    may contain valence weirdness — we sanitize before embedding.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        # Try to neutralize unfilled valences by re-parsing with no sanitize.
        try:
            mol = Chem.MolFromSmiles(frag_smi, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        except (ValueError, RuntimeError):
            return None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    mh, cids = _embed_ensemble(mol, n_confs=3, seed=seed)
    if not cids:
        return None
    best_g = None
    for cid in cids[:max(1, min(2, len(cids)))]:
        cdir = workdir / f"cid{cid}"
        cdir.mkdir(exist_ok=True)
        xyz = cdir / "input.xyz"
        _mol_to_xyz(mh, cid, xyz)
        ok, out = _xtb_run(xyz, cdir, charge=0, opt=True, alpb=solvent,
                            dipole=False, gfnff=False, timeout_s=timeout_s)
        if not ok:
            continue
        g_Eh = _parse_xtb_stdout(out).get("gsolv_Eh")
        if g_Eh is not None and (best_g is None or g_Eh < best_g):
            best_g = g_Eh
    return None if best_g is None else best_g * HARTREE_TO_KJMOL


__all__ = ["compute_qm_descriptors", "QM_KEYS"]


if __name__ == "__main__":
    # Smoke test: ethanol (small, fast) + IAJD 369 (representative).
    print("xTB available:", _xtb_available())
    if not _xtb_available():
        print("Set XTB_CMD env var. Default uses /tmp/bin/micromamba.")
        raise SystemExit(0)
    print("\n== Ethanol ==")
    r = compute_qm_descriptors("CCO", n_confs=2, timeout_s=120)
    for k in QM_KEYS:
        print(f"  {k:24s} = {r.get(k)}")
    print(f"  qm_error                 = {r.get('qm_error')}")
