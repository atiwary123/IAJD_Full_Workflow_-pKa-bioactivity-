"""
predict_pka_v91_one.py — one-shot subprocess wrapper around predict_pka_v91.

Reads a single SMILES + family hint from argv, runs live v9.1 (which calls
live MolGpKa + per-family debias internally), prints the result as JSON to
stdout, and exits. Designed to be called via subprocess.run(timeout=…) so a
hung query never blocks the parent.

Usage:
    python predict_pka_v91_one.py "<SMILES>" "<family_hint>"
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: predict_pka_v91_one.py <SMILES> <family>"}))
        sys.exit(2)
    smi = sys.argv[1]
    fam = sys.argv[2]
    os.chdir(ROOT / "IAJD_master/datasets")
    from iajd_pka_v91 import load_v91_bundle, predict_pka_v91
    bundle = load_v91_bundle(
        xlsx_path="IAJD_pKa_v21_final.xlsx",
        debias_path="molgpka_debias_models.joblib",
        molgpka_npy="molgpka_preds.npy",
    )
    r = predict_pka_v91(smi, bundle, family_hint=fam, return_diagnostics=False)
    # Pull only JSON-safe fields
    out = {
        "smiles": smi, "family": fam,
        "pKa_pred": r.get("pKa_pred"),
        "pKa_PI_90": r.get("pKa_PI_90"),
        "confidence_tier": r.get("confidence_tier"),
        "max_tanimoto_to_training": r.get("max_tanimoto_to_training"),
        "family_assigned": r.get("family_assigned"),
        "ood_flag": r.get("ood_flag"),
        "error": r.get("error"),
        "warnings": r.get("warnings"),
    }
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
