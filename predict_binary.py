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
_V91_BUNDLE_CACHE = None        # v9.1 pKa bundle, lazy-loaded
_LIVE_PKA_CACHE: dict = {}      # canonical SMILES -> live (pKa, pKa_sd)

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


def _live_pka_for_smiles(smi: str, family_hint: str = "GA-Tris") -> Tuple[float, float]:
    """Live v9.1 pKa prediction (uses live MolGpKa + per-family debias).

    Replaces the previous hardcoded `pka = 6.3` fallback in
    bioact_v14_pipeline._compute_block_a_from_row when a query SMILES has no
    measured pKa. No proxy fallback: if v9.1 returns no valid value the caller
    has to decide what to do (we let it fall through to NaN).
    """
    global _V91_BUNDLE_CACHE
    if smi in _LIVE_PKA_CACHE:
        return _LIVE_PKA_CACHE[smi]
    try:
        from iajd_pka_v91 import load_v91_bundle, predict_pka_v91
        if _V91_BUNDLE_CACHE is None:
            import os
            xlsx_dir = str((ROOT / "IAJD_master/datasets").resolve())
            cwd = os.getcwd()
            try:
                os.chdir(xlsx_dir)
                _V91_BUNDLE_CACHE = load_v91_bundle(
                    xlsx_path="IAJD_pKa_v21_final.xlsx",
                    debias_path="molgpka_debias_models.joblib",
                    molgpka_npy="molgpka_preds.npy",
                )
            finally:
                os.chdir(cwd)
        r = predict_pka_v91(smi, _V91_BUNDLE_CACHE, family_hint=family_hint,
                              return_diagnostics=False)
        if "error" in r:
            pair = (float("nan"), float("nan"))
        else:
            pi90 = r.get("pKa_PI_90") or [r.get("pKa_pred"), r.get("pKa_pred")]
            pka = float(r.get("pKa_pred"))
            sd = float((pi90[1] - pi90[0]) / 3.29) if len(pi90) == 2 else 0.3
            pair = (pka, sd)
    except Exception:
        pair = (float("nan"), float("nan"))
    _LIVE_PKA_CACHE[smi] = pair
    return pair


def _extend_caches_live(canon_smis: List[str]) -> None:
    """No-proxy: run live LION + ADMET subprocesses for any canonical SMILES
    not already cached. Populates the JSON caches in place so the subsequent
    assemble_X call hits real values instead of RDKit proxies.

    No-op when extend_caches module can't be imported (cache then stays as-is
    and Block B/C may fall back to RDKit proxies, but we log it honestly).
    """
    import json
    sys.path.insert(0, str(ROOT / "IAJD_master/code"))
    try:
        from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles
    except Exception as exc:
        print(f"  [WARN] live cache extension unavailable: {exc}")
        return
    OUT = ROOT / "IAJD_master/bundles_caches"
    lion_path = OUT / "lion_cache_v13.json"
    admet_path = OUT / "admet_cache_v13.json"
    if not lion_path.exists() or not admet_path.exists():
        return
    lion_cache = json.load(open(lion_path))
    admet_cache = json.load(open(admet_path))
    to_lion = sorted({c for c in canon_smis if c not in lion_cache})
    to_admet = sorted({c for c in canon_smis if c not in admet_cache})
    if to_admet:
        try:
            predict_admet_for_smiles(to_admet, cache_path=str(admet_path), verbose=False)
        except Exception as exc:
            print(f"  [WARN] live ADMET extension failed: {exc}")
    if to_lion:
        try:
            predict_lion_for_smiles(to_lion, cache_path=str(lion_path), verbose=False)
        except Exception as exc:
            print(f"  [WARN] live LION extension failed: {exc}")


def _assemble_X(smiles_list: List[str], family_hint: str = "GA-Tris",
                 sample_prep: dict | None = None,
                 include_dprime: bool | None = None):
    """Build the feature stack for one or more SMILES.

    `include_dprime`:
      None  (default) → return the full 106-col stack (Block A+B+C+D+form+E+D').
      False           → legacy 92-col stack (skip Block D'), for compatibility
                        with bundles trained before the W-A/W-B/W-C upgrade.

    Callers loading a binary bundle can pass `include_dprime=False` if their
    bundle was trained against the 92-feature layout, OR they can leave it as
    None and the bioact_v14_pipeline auto-detects from the consumer model.

    `sample_prep` (optional): dict of {pH_sample, T_hours, inj_route} to
    populate Block E at inference. Missing keys → NaN (XGBoost's default
    branch handles it). When None, all Block E entries are NaN — equivalent
    to the model running without sample-prep context.

    No-proxy: before assembling the feature stack, run live LION + ADMET
    subprocesses to populate Block B/C with REAL values (not RDKit proxies)
    for any SMILES not already in the per-cache JSON.
    """
    from bioact_v14_pipeline import assemble_X
    OUT = ROOT / "IAJD_master/bundles_caches"
    lookup = _bioact_lookup()
    sp = sample_prep or {}
    mols, fps_q, rows, canons = [], [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
        fps_q.append(_fp(m) if m else None)
        canon = Chem.MolToSmiles(m) if m else s
        canons.append(canon)
    # No-proxy: live LION + ADMET extension for any new SMILES, BEFORE the
    # per-row row-dict construction (so when assemble_X reads caches, the
    # entries are present).
    _extend_caches_live(canons)
    for canon in canons:
        if canon in lookup:
            row = dict(lookup[canon])
        else:
            row = dict(_features_for_query(canon))
            # Live v9.1 pKa + per-family debias (no proxy fallback)
            if pd.isna(row.get("pKa")):
                pka_pred, pka_sd = _live_pka_for_smiles(canon, family_hint=family_hint)
                if np.isfinite(pka_pred):
                    row["pKa"] = pka_pred
                    row["pKa_sd"] = pka_sd
        row["canonical_smi"] = canon
        if not row.get("family") or pd.isna(row.get("family")):
            row["family"] = family_hint
        row["log10_flux_total"] = np.nan
        # Block E sample-prep (per-query override if provided)
        if "pH_sample" in sp and sp["pH_sample"] is not None:
            row["pH_sample"] = sp["pH_sample"]
        if "T_hours" in sp and sp["T_hours"] is not None:
            row["T_hours"] = sp["T_hours"]
        if "inj_route" in sp and sp["inj_route"] is not None:
            row["inj_route"] = sp["inj_route"]
        rows.append(row)
    df_q = pd.DataFrame(rows)
    # Default to the new (106-col) layout; callers can force legacy 92-col by
    # passing include_dprime=False.
    use_dp = True if include_dprime is None else bool(include_dprime)
    X, _modes = assemble_X(
        df_q, mols, fps_q, canons,
        lion_cache_path=str(OUT / "lion_cache_v13.json"),
        admet_cache_path=str(OUT / "admet_cache_v13.json"),
        lion_train_fps_path=str(OUT / "lion_train_fps.pkl"),
        include_dprime=use_dp,
    )
    return X


def predict_p_above_batch(smiles_list: List[str], threshold: float,
                           bundle: dict | None = None,
                           family_hint: str = "GA-Tris") -> Tuple[np.ndarray, np.ndarray]:
    """Return (yhat, p_above) for each SMILES at the given threshold."""
    b = bundle or load_binary_bundle()
    # Match the bundle's feature dimensionality: pre-Block-D' regressors were
    # trained on 92 cols. Post-W-D bundles expect 106 (with Block D').
    reg_in = getattr(b["regressor"], "n_features_in_", 92)
    include_dp = reg_in >= 100
    X = _assemble_X(smiles_list, family_hint=family_hint,
                     include_dprime=include_dp)
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
