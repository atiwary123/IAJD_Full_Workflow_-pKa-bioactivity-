"""
eval/extrapolation_curve.py — the "369 problem" plot.

For each LOO held-out compound, plot MAE vs novelty (1 − max Tanimoto to
training-set neighbors), with and without the Block D' qmmd head. The
qmmd head is expected to flatten the error curve at high novelty —
that's the whole point of having a physics-grounded extrapolation head.

Outputs:
  eval/figures/extrapolation_curve.png
  eval/figures/extrapolation_curve.json
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "extrapolation_curve.png"
OUT_JSON = FIG_DIR / "extrapolation_curve.json"


def main():
    bundle_path = ROOT / "IAJD_master/bundles_caches/bioact_stacker_bundle.pkl"
    bioact_path = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
    if not bundle_path.exists() or not bioact_path.exists():
        print(f"ERROR: required bundle missing.\n  stacker: {bundle_path}\n  bioact:  {bioact_path}")
        return 1
    with open(bundle_path, "rb") as f:
        sb = pickle.load(f)
    with open(bioact_path, "rb") as f:
        b14 = pickle.load(f)

    smiles_all = list(b14["smis_train"])
    y_all = np.asarray(b14["y_train"], dtype=float)
    n = len(y_all)
    print(f"Training set: {n} compounds")

    # Compute novelty (1 − max Tanimoto in LOO).
    import sys
    sys.path.insert(0, str(ROOT))
    import rdkit_compat  # noqa: F401 — patches AllChem.GetMorganGenerator
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import BulkTanimotoSimilarity
    fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
    fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_all]
    novelty = np.zeros(n)
    for i in range(n):
        sims = BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(n) if j != i])
        novelty[i] = 1.0 - max(sims) if sims else 1.0

    # Pull LOO predictions for each head from bioact_loo_components.npz.
    comp_path = ROOT / "bioact_loo_components.npz"
    if not comp_path.exists():
        print("  bioact_loo_components.npz missing — produce via loo_bioact_components.py")
        return 1
    d = np.load(comp_path, allow_pickle=True)
    # Only keep numeric LOO arrays (skip families/labels).
    head_loo = {}
    for k in d.files:
        arr = d[k]
        if arr.shape != y_all.shape:
            continue
        if arr.dtype.kind not in ("f", "i"):
            continue
        if k in ("y_true", "max_sim"):
            continue
        head_loo[k] = arr.astype(float)
    print(f"  heads in LOO bundle: {list(head_loo.keys())}")

    # MAE per novelty bin, with and without qmmd.
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]
    has_qmmd = "qmmd" in head_loo
    results = {"bins": bin_labels, "with_qmmd": [], "without_qmmd": [],
                "n_in_bin": []}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (novelty >= lo) & (novelty < hi)
        if mask.sum() == 0:
            results["with_qmmd"].append(None)
            results["without_qmmd"].append(None)
            results["n_in_bin"].append(0)
            continue
        # Without qmmd: mean of available heads (direct/analog/lion/admet etc.).
        no_qmmd_heads = [k for k in head_loo if k != "qmmd"]
        preds_no = np.mean([head_loo[k] for k in no_qmmd_heads], axis=0)
        mae_no = float(np.mean(np.abs(preds_no[mask] - y_all[mask])))
        results["without_qmmd"].append(mae_no)
        if has_qmmd:
            preds_w = np.mean([head_loo[k] for k in head_loo], axis=0)
            mae_w = float(np.mean(np.abs(preds_w[mask] - y_all[mask])))
            results["with_qmmd"].append(mae_w)
        else:
            results["with_qmmd"].append(None)
        results["n_in_bin"].append(int(mask.sum()))
        print(f"  novelty {lo:.1f}-{hi:.1f}: n={mask.sum():3d}  "
              f"without_qmmd={mae_no:.3f}  "
              f"with_qmmd={results['with_qmmd'][-1]}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bin_labels))
    nw = [v if v is not None else 0 for v in results["without_qmmd"]]
    w  = [v if v is not None else 0 for v in results["with_qmmd"]]
    ax.bar(x - 0.2, nw, 0.4, label="Without qmmd head")
    if has_qmmd:
        ax.bar(x + 0.2, w, 0.4, label="With qmmd head")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("Novelty bin (1 − max Tanimoto)")
    ax.set_ylabel("LOO MAE of log10_flux")
    ax.set_title("Extrapolation curve — qmmd impact at high novelty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
