"""
physics_status.py — health check for the W-A/W-B/W-C/W-D/W-E build.

Prints what's installed, which caches exist, what emulators are present, and
gives the user a one-glance picture of where the predictive-physics stack is
in its rollout.

Usage:
  python physics_status.py
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"


def _which(cmd: str) -> Optional[str]:
    try:
        r = subprocess.run(["which", cmd], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (FileNotFoundError, OSError):
        return None


def _xtb_version() -> Optional[str]:
    try:
        from qm_descriptors import DEFAULT_XTB_CMD
        cmd = DEFAULT_XTB_CMD.split() + ["--version"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        for line in (r.stdout + r.stderr).splitlines():
            if "xtb version" in line.lower():
                return line.strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def _gmx_version() -> Optional[str]:
    try:
        from martini.run_selfassembly import DEFAULT_GMX
        cmd = list(DEFAULT_GMX) + ["--version"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            if "GROMACS version" in line:
                return line.strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def _cache_rows(p: Path) -> int:
    if not p.exists():
        return -1
    try:
        import pandas as pd
        return int(len(pd.read_csv(p)))
    except Exception:
        return -2


def main():
    print("="*70)
    print(" IAJD predictive-physics build status")
    print("="*70)

    # --- Binaries ---
    print("\n[ Binaries ]")
    print(f"  xtb:    {_xtb_version() or 'NOT FOUND'}")
    print(f"  gmx:    {_gmx_version() or 'NOT FOUND'}")
    crest = _which('crest')
    print(f"  crest:  {crest or 'not installed (W-C falls back to ETKDGv3)'}")

    # --- Caches ---
    print("\n[ Physics caches ]")
    caches = [
        ("qm_cache.csv",            "GFN2-xTB descriptors (W-A)"),
        ("md_cache.csv",            "MARTINI MD observables (W-B)"),
        ("head_area_ensemble.csv",  "Boltzmann head_area (W-C)"),
    ]
    for fname, desc in caches:
        p = PHYS_DIR / fname
        rows = _cache_rows(p)
        if rows >= 0:
            print(f"  {fname:30s} {rows:>4d} rows   ({desc})")
        else:
            print(f"  {fname:30s} MISSING        ({desc})")

    # --- Emulators ---
    print("\n[ Space-side emulators ]")
    for fname in ("qm_emulator.joblib", "md_emulator.joblib", "qmmd_head.joblib"):
        p = PHYS_DIR / fname
        present = "present" if p.exists() else "missing"
        print(f"  {fname:24s} {present}")

    # --- Bundle status ---
    print("\n[ Bioactivity bundle ]")
    import pickle
    bp = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
    if bp.exists():
        try:
            with open(bp, "rb") as f:
                b = pickle.load(f)
            n_feat = b["X_train"].shape[1] if hasattr(b["X_train"], "shape") else len(b["X_train"][0])
            print(f"  bioact_v14_bundle.pkl: {n_feat} features  ({'with Block D-prime' if n_feat == 106 else 'pre-Block-D-prime'})")
            metrics = b.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v}")
        except Exception as exc:
            print(f"  bioact_v14_bundle.pkl: load failed ({exc})")
    else:
        print("  bioact_v14_bundle.pkl: missing")

    # --- Stacker status ---
    sp = ROOT / "IAJD_master/bundles_caches/bioact_stacker_bundle.pkl"
    if sp.exists():
        try:
            with open(sp, "rb") as f:
                sb = pickle.load(f)
            print(f"\n[ Adaptive stacker ]")
            print(f"  version:         {sb.get('version')}")
            print(f"  stack_features:  {sb.get('stack_features')}")
            print(f"  has qmmd head:   {'yes' if sb.get('qmmd_head') is not None else 'no (standalone qmmd_head.joblib used)'}")
        except Exception as exc:
            print(f"\n[ Adaptive stacker ] load failed ({exc})")

    # --- TODO list ---
    print("\n[ Next steps ]")
    qm_p = PHYS_DIR / "qm_cache.csv"
    md_p = PHYS_DIR / "md_cache.csv"
    qm_rows = _cache_rows(qm_p)
    md_rows = _cache_rows(md_p)
    if qm_rows < 270:
        print(f"  - continue python precompute_qm.py --resume "
              f"({qm_rows}/270 done — ~9 min/compound)")
    if md_rows < 270:
        print(f"  - continue python precompute_md.py --resume "
              f"({md_rows} entries — needs gmx in PATH or via xtb_env)")
    if (PHYS_DIR / "qm_cache.csv").exists() and qm_rows >= 30:
        print("  - retrain QM emulator: python train_qm_emulator.py")
    if (PHYS_DIR / "md_cache.csv").exists() and md_rows >= 30:
        print("  - retrain MD emulator: python train_md_emulator.py")
    print("  - full retrain orchestrator:  python retrain_with_physics.py")
    print()


if __name__ == "__main__":
    main()
