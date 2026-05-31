"""
retrain_with_physics.py — sequence the full W-A → W-B → W-D → W-E pipeline so
the model bundles pick up Block D' (MD/QM-grounded physics) end-to-end.

Designed to be run repeatedly: each step checks for existing cache/bundle
files and skips work already done. Honest no-proxy: every step that uses
caches that are missing logs a warning and proceeds with NaN-filled rows.

Order (per the implementation plan §6):
  1. Verify QM cache + qm_emulator (W-A)         [precompute_qm.py / train_qm_emulator.py]
  2. Verify MD cache + md_emulator (W-B)         [precompute_md.py / train_md_emulator.py]
  3. Build Boltzmann head_area_ensemble (W-C)    [head_area_ensemble.py --no-xtb fallback]
  4. Build qmmd training feature block            [adaptive_stacker.build_qmmd_block_for_training]
  5. Rebuild bioact_v14_bundle with Block D'      [bioact_v14_pipeline.main()]
  6. Retrain adaptive_stacker (now with qmmd head) [adaptive_stacker.build_adaptive_stacker()]
  7. Retrain per-organ & binary classifiers        [train_per_organ.py / train_binary_classifier.py]

Each step is idempotent. Use --force-retrain to skip cache checks and retrain
everything from scratch.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"


def _exists_with_rows(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return len(df) >= min_rows
    except (ImportError, Exception):
        return False


def _run_step(label: str, cmd: list, *, optional: bool = False) -> bool:
    print(f"\n=== {label} ===")
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), check=False)
        dt = time.time() - t0
        if r.returncode != 0 and not optional:
            print(f"  FAILED in {dt:.1f}s (rc={r.returncode})")
            return False
        print(f"  done in {dt:.1f}s (rc={r.returncode})")
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  EXC: {exc}")
        return optional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-precompute-qm", action="store_true")
    parser.add_argument("--skip-precompute-md", action="store_true")
    parser.add_argument("--skip-head-area", action="store_true")
    parser.add_argument("--skip-block-rebuild", action="store_true")
    parser.add_argument("--skip-stacker", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--qm-limit", type=int, default=None,
                        help="--limit on precompute_qm.py (sanity-check runs).")
    parser.add_argument("--qm-confs", type=int, default=2)
    parser.add_argument("--md-prod-ns", type=float, default=2000.0)
    args = parser.parse_args()

    print(f"retrain_with_physics: {time.ctime()}")
    print(f"  ROOT = {ROOT}")

    qm_cache = PHYS_DIR / "qm_cache.csv"
    md_cache = PHYS_DIR / "md_cache.csv"
    head_area = PHYS_DIR / "head_area_ensemble.csv"

    # ──── 1. QM precompute (W-A) ────
    if not args.skip_precompute_qm:
        if args.force_retrain or not _exists_with_rows(qm_cache, min_rows=50):
            _run_step("W-A.1: precompute_qm.py",
                       [sys.executable, "precompute_qm.py",
                        f"--n-confs={args.qm_confs}",
                        "--timeout-s=900", "--resume"]
                       + ([f"--limit={args.qm_limit}"] if args.qm_limit else []))
        else:
            print(f"\n=== W-A.1: QM cache present ({qm_cache}) — skip precompute ===")
        # Always (re)train emulator after precompute step.
        _run_step("W-A.2: train_qm_emulator.py",
                   [sys.executable, "train_qm_emulator.py"],
                   optional=True)

    # ──── 2. MD precompute (W-B) ────
    if not args.skip_precompute_md:
        if args.force_retrain or not _exists_with_rows(md_cache, min_rows=50):
            _run_step("W-B.1: precompute_md.py",
                       [sys.executable, "precompute_md.py",
                        "--resume",
                        f"--prod-ns={args.md_prod_ns}"])
        else:
            print(f"\n=== W-B.1: MD cache present ({md_cache}) — skip precompute ===")
        _run_step("W-B.2: train_md_emulator.py",
                   [sys.executable, "train_md_emulator.py"],
                   optional=True)

    # ──── 3. Boltzmann head_area_ensemble (W-C) ────
    if not args.skip_head_area:
        if args.force_retrain or not head_area.exists():
            _run_step("W-C: head_area_ensemble.py",
                       [sys.executable, "head_area_ensemble.py"])

    # ──── 4. qmmd training feature block ────
    _run_step("W-D.1: build qmmd training block",
               [sys.executable, "adaptive_stacker.py", "--build-qmmd-block"],
               optional=False)

    # ──── 5. Rebuild bioact_v14_bundle with Block D' ────
    if not args.skip_block_rebuild:
        _run_step("W-D.2: bioact_v14_pipeline.main() (full rebuild)",
                   [sys.executable, "IAJD_master/code/bioact_v14_pipeline.py"],
                   optional=False)

    # ──── 6. Adaptive stacker (now with qmmd head) ────
    if not args.skip_stacker:
        _run_step("W-E.1: adaptive_stacker.build_adaptive_stacker()",
                   [sys.executable, "adaptive_stacker.py"], optional=False)

    # ──── 7. Per-organ + binary classifier retrain ────
    _run_step("W-E.2: train_per_organ.py",
               [sys.executable, "train_per_organ.py"], optional=True)
    _run_step("W-E.3: train_binary_classifier.py",
               [sys.executable, "train_binary_classifier.py"], optional=True)

    print(f"\nretrain_with_physics complete at {time.ctime()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
