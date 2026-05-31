"""
auto_retrain_watcher.py — long-running watcher that rebuilds Block D' training
artifacts whenever the QM or MD caches grow.

What it does (in a loop):
  1. snapshots qm_cache row count + md_cache row count
  2. if either grew by ≥ `--threshold` since last retrain:
       a. python adaptive_stacker.py --build-qmmd-block  (refresh feature matrix)
       b. python train_qmmd_head_only.py                 (refit head, log CV-MAE)
       c. python train_qm_emulator.py                    (refit emulator if cache ≥30)
       d. python train_md_emulator.py                    (same)
  3. sleeps `--poll-interval` seconds and repeats.

Designed to run in the background alongside the precompute_md / precompute_qm
processes. The retrain takes seconds because the caches are small. Logs each
retrain event to auto_retrain_log.csv.

Stop with Ctrl-C / SIGTERM. Resumable: re-reads the latest snapshot from the
log file on startup.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"
QM_CACHE = PHYS_DIR / "qm_cache.csv"
MD_CACHE = PHYS_DIR / "md_cache.csv"
LOG_CSV = ROOT / "auto_retrain_log.csv"


def _count_rows(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        with open(p) as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


def _last_known(log_path: Path) -> tuple[int, int]:
    if not log_path.exists():
        return 0, 0
    try:
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0, 0
        last = rows[-1]
        return int(last.get("qm_rows", 0)), int(last.get("md_rows", 0))
    except (OSError, ValueError, KeyError):
        return 0, 0


def _append_log(log_path: Path, **kw) -> None:
    header_needed = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(list(kw.keys()))
        w.writerow(list(kw.values()))


def _run(cmd: list, timeout: int = 600) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                            timeout=timeout)
        ok = r.returncode == 0
        return ok, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, str(exc)


def retrain(label: str) -> dict:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] retrain: {label}")
    summary: dict = {"label": label}
    ok1, _ = _run([sys.executable, "adaptive_stacker.py", "--build-qmmd-block"])
    summary["build_qmmd_block_ok"] = ok1
    ok2, out2 = _run([sys.executable, "train_qmmd_head_only.py"])
    summary["train_qmmd_head_ok"] = ok2
    # Pull CV-MAE from last log line if present.
    cv_mae = None
    for line in out2.splitlines():
        if "5-fold CV: MAE=" in line:
            try:
                cv_mae = float(line.split("MAE=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    summary["qmmd_cv_mae"] = cv_mae
    ok3, _ = _run([sys.executable, "train_qm_emulator.py"])
    summary["train_qm_emulator_ok"] = ok3
    ok4, _ = _run([sys.executable, "train_md_emulator.py"])
    summary["train_md_emulator_ok"] = ok4
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=3,
                        help="Retrain when either cache grows by this many rows.")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between cache checks.")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Stop after this many iterations (for testing).")
    args = parser.parse_args()

    print(f"[auto_retrain_watcher] {datetime.now().isoformat(timespec='seconds')}")
    print(f"  threshold={args.threshold}  poll_interval={args.poll_interval}s")
    print(f"  log: {LOG_CSV}")

    last_qm, last_md = _last_known(LOG_CSV)
    print(f"  last known qm_rows={last_qm}, md_rows={last_md}")
    n_iter = 0
    while True:
        n_iter += 1
        qm = _count_rows(QM_CACHE)
        md = _count_rows(MD_CACHE)
        d_qm = qm - last_qm
        d_md = md - last_md
        if d_qm >= args.threshold or d_md >= args.threshold:
            label = f"qm+{d_qm}/md+{d_md}"
            summary = retrain(label)
            _append_log(LOG_CSV,
                         ts=datetime.now().isoformat(timespec="seconds"),
                         qm_rows=qm, md_rows=md,
                         delta_qm=d_qm, delta_md=d_md,
                         **summary)
            last_qm, last_md = qm, md
        else:
            # No retrain — still record a heartbeat every 12 iterations.
            if n_iter % 12 == 0:
                print(f"  [{datetime.now().isoformat(timespec='seconds')}] "
                      f"qm={qm} md={md} (no retrain yet)")
        if args.max_iters and n_iter >= args.max_iters:
            break
        time.sleep(args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
