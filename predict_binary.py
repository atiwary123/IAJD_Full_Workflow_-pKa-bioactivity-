"""
predict_binary.py — inference helpers for the binary (tunable-threshold)
bioactivity classifier.

API:
    bundle = load_binary_bundle()
    yhat, p = predict_p_above(smiles, threshold, bundle)
    yhat, p = predict_p_above_batch(smiles_list, threshold, bundle)
"""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

BIN_BUNDLE_PATH = ROOT / "IAJD_master/bundles_caches/bioact_binary_bundle.pkl"
BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"

_BUNDLE_CACHE: Optional[dict] = None
_BIOACT_LOOKUP: Optional[dict] = None
_DESC_CACHE: dict = {}

try:
    _GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    def _fp(m): return _GEN.GetFingerprint(m)
except AttributeError:
    def _fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def load_binary_bundle(path: Path | None = None) -> dict:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    p = path or BIN_BUNDLE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Binary bundle not found: {p}")
    with open(p, "rb") as f:
        _BUNDLE_CACHE = pickle.load(f)
    return _BUNDLE_CACHE


def _bioact_lookup() -> dict:
    global _BIOACT_LOOKUP
    if _BIOACT_LOOKUP is not None:
        return _BIOACT_LOOKUP
    df = pd.read_excel(BIOACT_XLSX)
    out = {}
    for _, r in df.iterrows():
        sm = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(sm):
            continue
        try:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(str(sm)))
        except Exception:
            continue
        out[canon] = r.to_dict()
    _BIOACT_LOOKUP = out
    return out


def _features_for_query(smi: str) -> dict:
    if smi in _DESC_CACHE:
        return _DESC_CACHE[smi]
    from expand_datasets import compute_rdkit_features, compute_3d_features
    d = compute_rdkit_features(smi) or {}
    d.update(compute_3d_features(smi) or {})
    _DESC_CACHE[smi] = d
    return d


def _assemble_X(smiles_list: List[str], family_hint: str = "GA-Tris"):
    from bioact_v14_pipeline import assemble_X
    OUT = ROOT / "IAJD_master/bundles_caches"
    lookup = _bioact_lookup()
    mols, fps_q, rows, canons = [], [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
        fps_q.append(_fp(m) if m else None)
        canon = Chem.MolToSmiles(m) if m else s
        canons.append(canon)
        if canon in lookup:
            row = dict(lookup[canon])
        else:
            row = dict(_features_for_query(canon))
        row["canonical_smi"] = canon
        if not row.get("family") or pd.isna(row.get("family")):
            row["family"] = family_hint
        row["log10_flux_total"] = np.nan
        rows.append(row)
    df_q = pd.DataFrame(rows)
    X, _modes = assemble_X(
        df_q, mols, fps_q, canons,
        lion_cache_path=str(OUT / "lion_cache_v13.json"),
        admet_cache_path=str(OUT / "admet_cache_v13.json"),
        lion_train_fps_path=str(OUT / "lion_train_fps.pkl"),
    )
    return X


def predict_p_above_batch(smiles_list: List[str], threshold: float,
                           bundle: dict | None = None,
                           family_hint: str = "GA-Tris") -> Tuple[np.ndarray, np.ndarray]:
    """Return (yhat, p_above) for each SMILES at the given threshold."""
    b = bundle or load_binary_bundle()
    X = _assemble_X(smiles_list, family_hint=family_hint)
    yhat = b["regressor"].predict(X)
    # Pick nearest grid threshold
    grid = sorted(float(g) for g in b["threshold_calibrators"].keys())
    grid_arr = np.array(grid)
    nearest = grid[int(np.argmin(np.abs(grid_arr - threshold)))]
    c = b["threshold_calibrators"][nearest]
    a, off = c["a"], c["b"]
    # Use the exact requested threshold in the sigmoid, not the snapped grid one,
    # so we get a continuous prediction.
    p = 1.0 / (1.0 + np.exp(-(a * (yhat - threshold) + off)))
    return yhat, p


def predict_p_above(smiles: str, threshold: float,
                     bundle: dict | None = None,
                     family_hint: str = "GA-Tris") -> Tuple[float, float]:
    yhat, p = predict_p_above_batch([smiles], threshold, bundle, family_hint)
    return float(yhat[0]), float(p[0])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="*")
    ap.add_argument("--threshold", type=float, default=8.0)
    args = ap.parse_args()
    if not args.smiles:
        df = pd.read_excel(BIOACT_XLSX)
        test = df[df["IAJD_num"].isin([369, 348, 366, 1])][["IAJD_num","SMILES_canonical","log10_flux_total"]]
        for _, r in test.iterrows():
            yhat, p = predict_p_above(r["SMILES_canonical"], args.threshold)
            print(f"IAJD {int(r['IAJD_num']):3d}  y_obs={r['log10_flux_total']:.2f}  "
                  f"ŷ={yhat:.2f}  P(≥{args.threshold})={p:.2f}")
    else:
        for s in args.smiles:
            yhat, p = predict_p_above(s, args.threshold)
            print(f"ŷ={yhat:.2f}  P(≥{args.threshold})={p:.2f}  smiles={s}")
