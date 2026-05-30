"""
molgpka_runner.py — one-shot subprocess wrapper around MolGpKa.

Reads a SMILES from argv, runs the local MolGpKa GCN forward pass, prints
max base pKa to stdout as JSON, exits. Designed to be called with
subprocess.run(timeout=N) so a hung forward pass never blocks the parent.

This is the no-proxy subprocess form of the previously-in-process
_try_live_molgpka helper — same real model, just isolated from the parent's
torch threadpool / chemprop runtime to avoid the deadlock the in-process
path occasionally triggered.

Usage:
    python molgpka_runner.py "<SMILES>"

Output JSON schema:
    {"smiles": "...", "max_base_pka": 6.10, "n_base_atoms": 2}
or on failure:
    {"smiles": "...", "error": "..."}
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOLGPKA_SRC = ROOT / "molgpka_src"

# Force single-thread torch to keep the child process tiny and predictable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if str(MOLGPKA_SRC) not in sys.path:
    sys.path.insert(0, str(MOLGPKA_SRC))

import warnings
warnings.filterwarnings("ignore")


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: molgpka_runner.py <SMILES>"}))
        return 2
    smi = sys.argv[1]
    try:
        from rdkit import Chem, RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(json.dumps({"smiles": smi, "error": "invalid SMILES"}))
            return 3
        from predict_pka import predict as molgpka_predict
        base_dict, acid_dict = molgpka_predict(mol)
        if not base_dict:
            print(json.dumps({"smiles": smi, "error": "no ionizable base atoms"}))
            return 4
        max_base = float(max(base_dict.values()))
        print(json.dumps({
            "smiles": smi,
            "max_base_pka": max_base,
            "n_base_atoms": len(base_dict),
            "n_acid_atoms": len(acid_dict) if acid_dict else 0,
        }))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"smiles": smi, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
