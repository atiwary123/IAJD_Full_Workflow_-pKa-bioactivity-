"""
predict_pka_v92.py — inference helper for the v9.2 three-head blend.

Loads the saved bundle once and exposes:
    predict_pka_v92(smiles, family_hint=None) -> dict

The dict contains pKa_pred plus per-head contributions and the blend weights,
so callers can see how the prediction was composed.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

BUNDLE_PATH = ROOT / "IAJD_master/bundles_caches/pka_v92_bundle.joblib"

_V92_BUNDLE: Optional[Dict[str, Any]] = None


def _morgan_fp(mol):
    try:
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_v92_bundle() -> Dict[str, Any]:
    global _V92_BUNDLE
    if _V92_BUNDLE is None:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"v9.2 bundle not found at {BUNDLE_PATH}")
        _V92_BUNDLE = joblib.load(BUNDLE_PATH)
    return _V92_BUNDLE


def _live_molgpka(mol) -> float:
    """Run live MolGpKa GCN and return max basic pKa (NaN if no basic site)."""
    sys.path.insert(0, str(ROOT / "molgpka_src"))
    try:
        from predict_pka import predict as molgpka_predict
    except Exception:
        return float("nan")
    try:
        base_dict, _ = molgpka_predict(mol)
        if base_dict:
            return float(max(base_dict.values()))
    except Exception:
        pass
    return float("nan")


def _v52_features_for_smiles(smiles: str) -> np.ndarray:
    """Compute the 30-d v52 base feature vector for a query SMILES."""
    cwd = os.getcwd()
    os.chdir(ROOT / "IAJD_master/datasets")
    try:
        from iajd_pka_v52 import compute_features, compute_3d_features_from_mol, tokens_from_mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full(30, np.nan)
        tokens = tokens_from_mol(mol)
        features_3d = compute_3d_features_from_mol(mol)
        feats = compute_features(mol, tokens, features_3d=features_3d)
    finally:
        os.chdir(cwd)
    return np.asarray(feats, dtype=float)


def _detect_family(smiles: str, family_hint: Optional[str]) -> str:
    if family_hint:
        return family_hint
    cwd = os.getcwd()
    os.chdir(ROOT / "IAJD_master/datasets")
    try:
        from iajd_pka_v52 import tokens_from_mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "sSS-Nonsym"
        tokens = tokens_from_mol(mol)
        return tokens.family
    finally:
        os.chdir(cwd)


def predict_pka_v92(smiles: str, family_hint: Optional[str] = None) -> Dict[str, Any]:
    """Predict pKa via the v9.2 three-head blend.

    Returns:
        {
          "pKa_pred": float,
          "components": {"analog": float, "xgb_pure": float,
                          "molgpka_debiased": float},
          "weights": {"analog": w1, "xgb_pure": w2, "molgpka_debiased": w3},
          "family_assigned": str,
          "warnings": [list of strings]
        }
    """
    b = load_v92_bundle()
    out: Dict[str, Any] = {"warnings": []}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "invalid SMILES", "pKa_pred": None}
    canonical = Chem.MolToSmiles(mol)
    fam = _detect_family(smiles, family_hint)
    out["family_assigned"] = fam

    # Head 1: analog (Tanimoto neighbors)
    q_fp = _morgan_fp(mol)
    train_fps = b["analog"]["fps"]
    train_pkas = np.array(b["analog"]["pkas"], dtype=float)
    sims = np.array([TanimotoSimilarity(q_fp, tf) for tf in train_fps])
    order = np.argsort(-sims)
    K = b["analog"]["k_query_max"]
    thresh = b["analog"]["sim_threshold"]
    min_n = b["analog"]["min_neighbors"]
    wp = b["analog"]["weight_power"]
    top = order[:K]
    above = [int(j) for j in top if sims[j] >= thresh]
    if len(above) < min_n:
        above = [int(j) for j in top[:min_n]]
    s_arr = sims[np.array(above)]
    w = s_arr ** wp
    w = w / w.sum()
    p_analog = float(np.sum(w * train_pkas[np.array(above)]))
    max_tanimoto = float(np.max(sims))

    # Head 2: pure XGB on 30 features
    q_feats = _v52_features_for_smiles(canonical)
    nan_mask = ~np.isfinite(q_feats)
    if nan_mask.any():
        # Fill with column medians from training (the scaler captures them)
        med = b["xgb_pure"]["scaler"].mean_
        q_feats = np.where(nan_mask, med, q_feats)
        out["warnings"].append(f"IMPUTED_{int(nan_mask.sum())}_FEATURES")
    q_scaled = b["xgb_pure"]["scaler"].transform(q_feats.reshape(1, -1))
    p_xgb = float(b["xgb_pure"]["model"].predict(q_scaled)[0])

    # Head 3: live MolGpKa → per-family debias
    raw = _live_molgpka(mol)
    debias = b["molgpka_debiased"]["debias_models"]
    if fam in debias and np.isfinite(raw):
        d = debias[fam]
        p_molgpka = float(d["slope"] * raw + d["intercept"])
    elif np.isfinite(raw):
        # Pooled (n-weighted)
        tot_n = sum(d_["n"] for d_ in debias.values())
        slope = sum(d_["slope"] * d_["n"] for d_ in debias.values()) / tot_n
        intercept = sum(d_["intercept"] * d_["n"] for d_ in debias.values()) / tot_n
        p_molgpka = float(slope * raw + intercept)
        out["warnings"].append(f"POOLED_DEBIAS_for_family={fam}")
    else:
        p_molgpka = float("nan")
        out["warnings"].append("MOLGPKA_LIVE_FAILED")

    weights = b["weights"]
    components = {"analog": p_analog, "xgb_pure": p_xgb,
                   "molgpka_debiased": p_molgpka}
    # Renormalize weights if MolGpKa unavailable
    if not np.isfinite(p_molgpka):
        w_a = weights["analog"]; w_x = weights["xgb_pure"]
        s = w_a + w_x
        pka = (w_a * p_analog + w_x * p_xgb) / s if s > 0 else p_analog
    else:
        pka = (weights["analog"] * p_analog
                + weights["xgb_pure"] * p_xgb
                + weights["molgpka_debiased"] * p_molgpka)
    out["pKa_pred"] = round(float(pka), 4)
    out["components"] = {k: round(v, 4) if np.isfinite(v) else None
                          for k, v in components.items()}
    out["weights"] = weights
    out["raw_molgpka"] = round(raw, 4) if np.isfinite(raw) else None
    out["max_tanimoto_to_training"] = round(max_tanimoto, 4)
    return out


if __name__ == "__main__":
    import json
    test = [
        "CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCCC)c1",   # MPRZ-12
        "CCCCC(CC)CCOc1cc(COC(=O)CCCN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCC",  # GA-tris H2EPRZ (IAJD 347)
    ]
    for sm in test:
        r = predict_pka_v92(sm)
        print(json.dumps(r, indent=2, default=str))
