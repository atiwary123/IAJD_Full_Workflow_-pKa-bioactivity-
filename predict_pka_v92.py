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


_LIVE_MOLGPKA_CACHE: dict = {}

def _live_molgpka(mol, timeout_s: float = 60.0) -> float:
    """Run live MolGpKa GCN via SUBPROCESS (T5 #14 hardened path) and return
    max basic pKa (NaN on failure or timeout).

    Why subprocess: the in-process MolGpKa load + forward pass can hang when
    the parent process already has chemprop's torch initialized — the two
    torch threadpools / OMP runtimes occasionally deadlock during state-dict
    deserialization. A subprocess with `start_new_session=True` isolates
    MolGpKa from whatever torch state lives in the parent.

    No-proxy: if MolGpKa subprocess fails or times out we return NaN — never
    a family median or constant. The v9.2 caller then drops the MolGpKa head
    weight to 0 and re-normalizes the blend over the remaining heads.
    """
    import subprocess, json
    canon = Chem.MolToSmiles(mol) if mol else None
    if canon is None:
        return float("nan")
    if canon in _LIVE_MOLGPKA_CACHE:
        return _LIVE_MOLGPKA_CACHE[canon]
    runner = ROOT / "molgpka_runner.py"
    if not runner.exists():
        _LIVE_MOLGPKA_CACHE[canon] = float("nan")
        return float("nan")
    clean_env = {k: v for k, v in os.environ.items()
                  if not k.startswith(("DYLD_", "LD_LIBRARY_PATH", "PYTHON"))}
    clean_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    clean_env["OMP_NUM_THREADS"] = "1"
    clean_env["MKL_NUM_THREADS"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(runner), canon],
            capture_output=True, text=True,
            timeout=timeout_s,
            env=clean_env, start_new_session=True,
        )
        if proc.returncode != 0:
            _LIVE_MOLGPKA_CACHE[canon] = float("nan")
            return float("nan")
        result = json.loads(proc.stdout.strip())
        if "error" in result:
            _LIVE_MOLGPKA_CACHE[canon] = float("nan")
            return float("nan")
        val = float(result.get("max_base_pka", float("nan")))
        _LIVE_MOLGPKA_CACHE[canon] = val
        return val
    except subprocess.TimeoutExpired:
        _LIVE_MOLGPKA_CACHE[canon] = float("nan")
        return float("nan")
    except Exception:
        _LIVE_MOLGPKA_CACHE[canon] = float("nan")
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

    # Head 2: pure XGB on 30 features. No-proxy: NaN inputs are routed via
    # XGBoost's default branch (the StandardScaler can't handle NaN, so we
    # do the equivalent z-score manually with NaN passthrough).
    q_feats = _v52_features_for_smiles(canonical)
    scaler = b["xgb_pure"]["scaler"]
    nan_mask = ~np.isfinite(q_feats)
    if nan_mask.any():
        out["warnings"].append(f"NAN_PASSTHROUGH_{int(nan_mask.sum())}_FEATURES")
    # Z-score the finite entries, leave NaN as NaN
    q_scaled = np.full_like(q_feats, np.nan, dtype=float)
    finite = ~nan_mask
    q_scaled[finite] = (q_feats[finite] - scaler.mean_[finite]) / scaler.scale_[finite]
    p_xgb = float(b["xgb_pure"]["model"].predict(q_scaled.reshape(1, -1))[0])

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
    # Resolve weights: prefer per-family LOO-optimized triple; fall back to
    # global LOO-optimized triple. No constants — both come from real CV.
    if "per_family" in weights and fam in weights["per_family"]:
        triple = weights["per_family"][fam]
        w_a, w_x, w_m = float(triple[0]), float(triple[1]), float(triple[2])
        out["weights_source"] = f"per_family[{fam}]"
    elif "global" in weights:
        g = weights["global"]
        w_a = float(g["analog"]); w_x = float(g["xgb_pure"]); w_m = float(g["molgpka_debiased"])
        out["weights_source"] = "global"
    else:
        # Legacy schema fallback (flat dict)
        w_a = float(weights.get("analog", 0.0))
        w_x = float(weights.get("xgb_pure", 0.0))
        w_m = float(weights.get("molgpka_debiased", 0.0))
        out["weights_source"] = "legacy_flat"

    components = {"analog": p_analog, "xgb_pure": p_xgb,
                   "molgpka_debiased": p_molgpka}
    # Renormalize over the remaining heads if MolGpKa unavailable. No proxy:
    # we drop MolGpKa's weight and redistribute proportionally, rather than
    # substituting a default pKa.
    if not np.isfinite(p_molgpka):
        s = w_a + w_x
        if s > 0:
            pka = (w_a * p_analog + w_x * p_xgb) / s
        else:
            pka = p_analog
    else:
        pka = w_a * p_analog + w_x * p_xgb + w_m * p_molgpka
    out["pKa_pred"] = round(float(pka), 4)
    out["components"] = {k: round(v, 4) if np.isfinite(v) else None
                          for k, v in components.items()}
    out["weights_applied"] = {"analog": w_a, "xgb_pure": w_x, "molgpka_debiased": w_m}
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
