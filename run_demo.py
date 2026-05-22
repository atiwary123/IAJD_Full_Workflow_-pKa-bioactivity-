"""
run_demo.py — End-to-end smoke test across all 8 IAJD families.

Reuses the 7 reference SMILES from iajd_tandem_final.py's __main__ block, runs
each through `iajd_predict.predict`, and writes `demo_results.json`. Prints a
one-line summary per case to stdout. Exit code is nonzero if any case errors.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iajd_predict import predict  # noqa: E402

CASES = [
    ("PE-Tris (in-cache training compound)",
     "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1",
     "PE-Tris"),
    ("GA-Tris (ADMET gate override)",
     "CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC",
     "GA-Tris"),
    ("sSS-Nonsym (standard)",
     "CCCCCCCCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCC)c1",
     "sSS-Nonsym"),
    ("Dialkoxybenzyl (standard)",
     "CCCCCCCCCCCCOc1ccc(COC(=O)CCCN2CCN(CCO)CC2)cc1OCCCCCCCCCCCC",
     "Dialkoxybenzyl"),
    ("G1-Janus-Dendrimer (pooled debias)",
     "CCCCCCCCCCCCOc1cc(OCCCCCCCCCCCC)c(OCCCCCCCCCCCC)c(COC(=O)CCCN2CCN(C)CC2)c1",
     "G1-Janus-Dendrimer"),
    ("HTM-Dendrimer (pooled debias)",
     "CCCCCCCCCCCCN(CCO)CCOC(=O)CCCN1CCN(C)CC1",
     "HTM-Dendrimer"),
    ("TT-Dendrimer (pooled debias)",
     "CCCCCCCCCCCCOC(=O)C(NC(=O)CCN(CCO)CCO)(COCCCCCCCCCCCC)COCCCCCCCCCCCC",
     "TT-Dendrimer"),
]


def main() -> int:
    results = []
    failed = 0
    print(f"Running {len(CASES)} smoke cases through the tandem...\n")
    for label, smi, fam in CASES:
        t0 = time.time()
        try:
            r = predict(smi, family=fam, neighbors=3)
            err = r.get("error")
            pka = r.get("pka", {}) or {}
            bio = r.get("bioactivity", {}) or {}
            pka_point = pka.get("point")
            bio_point = bio.get("point")
            ok = err is None and pka_point is not None and bio_point is not None
            failed += 0 if ok else 1
            print(
                f"  {'OK ' if ok else 'FAIL'}  {label:<46}"
                f"pKa={pka_point}   log10_flux={bio_point}   "
                f"t={time.time()-t0:.1f}s"
            )
            if err:
                print(f"        error: {err}")
            results.append({"label": label, "family": fam, "smiles": smi, "result": r})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {label:<46}exception: {type(exc).__name__}: {exc}")
            results.append({"label": label, "family": fam, "smiles": smi,
                            "exception": f"{type(exc).__name__}: {exc}"})

    out = HERE / "demo_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out.name} ({len(results)} cases).")
    print(f"Failures: {failed} / {len(CASES)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
