"""
iajd_neighbors.py — Tanimoto nearest-neighbor lookup over IAJD training sets.

Loads the pKa (v21, 246 compounds) and bioact (v13, 335 rows / 274 unique) tables,
builds Morgan-2/2048 fingerprints, caches them under .cache/ at the project root,
and returns the top-k most similar training IAJDs for any query SMILES.

Public API
----------
    tanimoto_neighbors(smiles, k=5, scope="both") -> dict
        scope is "pka", "bioact", or "both". Returns {'pka': [...], 'bioact': [...]}.
        Each neighbor dict carries the IAJD ID, family, measured label, and Tanimoto.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "IAJD_master" / "datasets"
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

PKA_XLSX = DATA_DIR / "IAJD_pKa_v21_final.xlsx"
BIOACT_XLSX = DATA_DIR / "IAJD_Bioact_v13_clean.xlsx"

PKA_CACHE = CACHE_DIR / "pka_fps.pkl"
BIOACT_CACHE = CACHE_DIR / "bioact_fps.pkl"


def _morgan_fp(mol):
    # rdkit >= 2023.03 ships the new generator; older builds still expose the legacy helper.
    try:
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _build_pka_index() -> Dict[str, Any]:
    if PKA_CACHE.exists():
        with open(PKA_CACHE, "rb") as f:
            return pickle.load(f)
    df = pd.read_excel(PKA_XLSX, sheet_name="Dataset")
    rows: List[Dict[str, Any]] = []
    fps = []
    for _, r in df.iterrows():
        smi = str(r.get("SMILES", "")).strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(_morgan_fp(mol))
        rows.append({
            "iajd_id": r.get("IAJD"),
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "family": r.get("family"),
            "pKa": float(r.get("pKa")) if pd.notna(r.get("pKa")) else None,
            "pKa_sd": float(r.get("pKa_sd")) if pd.notna(r.get("pKa_sd")) else None,
        })
    idx = {"rows": rows, "fps": fps}
    with open(PKA_CACHE, "wb") as f:
        pickle.dump(idx, f)
    return idx


def _build_bioact_index() -> Dict[str, Any]:
    # Novel GA-Tris IAJDs (347, 348, 365, 366, 367, 369, 372, 373) reintegrated 2026-05-28.
    if BIOACT_CACHE.exists():
        with open(BIOACT_CACHE, "rb") as f:
            return pickle.load(f)
    df = pd.read_excel(BIOACT_XLSX)
    rows: List[Dict[str, Any]] = []
    fps = []
    for _, r in df.iterrows():
        smi = str(r.get("SMILES_canonical") or r.get("SMILES") or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(_morgan_fp(mol))
        def _f(col):
            v = r.get(col)
            return float(v) if pd.notna(v) else None
        rows.append({
            "iajd_id": r.get("IAJD_id"),
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "family": r.get("family"),
            "log10_flux_total": _f("log10_flux_total"),
            "log10_flux_lung":  _f("log10_flux_lung"),
            "log10_flux_liver": _f("log10_flux_liver"),
            "log10_flux_spleen": _f("log10_flux_spleen"),
            "log10_flux_LN":    _f("log10_flux_LN"),
            "log10_flux_heart": _f("log10_flux_heart"),
        })
    idx = {"rows": rows, "fps": fps}
    with open(BIOACT_CACHE, "wb") as f:
        pickle.dump(idx, f)
    return idx


_INDICES: Dict[str, Dict[str, Any]] = {}


def _get_index(scope: str) -> Dict[str, Any]:
    if scope not in _INDICES:
        _INDICES[scope] = (_build_pka_index() if scope == "pka" else _build_bioact_index())
    return _INDICES[scope]


def _query_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"INVALID_SMILES: {smiles}")
    return _morgan_fp(mol)


def _top_k(scope: str, query_fp, k: int) -> List[Dict[str, Any]]:
    idx = _get_index(scope)
    if not idx["fps"]:
        return []
    sims = np.array(BulkTanimotoSimilarity(query_fp, idx["fps"]))
    order = np.argsort(-sims)[:k]
    out = []
    for j in order:
        row = dict(idx["rows"][int(j)])
        row["tanimoto"] = round(float(sims[int(j)]), 4)
        out.append(row)
    return out


def tanimoto_neighbors(smiles: str, k: int = 5, scope: str = "both") -> Dict[str, List[Dict[str, Any]]]:
    """Return the top-k IAJDs by Tanimoto similarity from each requested training set."""
    q_fp = _query_fp(smiles)
    result: Dict[str, List[Dict[str, Any]]] = {}
    if scope in ("pka", "both"):
        result["pka"] = _top_k("pka", q_fp, k)
    if scope in ("bioact", "both"):
        result["bioact"] = _top_k("bioact", q_fp, k)
    return result


if __name__ == "__main__":
    test_smi = "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(C)CC1"
    n = tanimoto_neighbors(test_smi, k=3)
    for scope, rows in n.items():
        print(f"\n== {scope.upper()} neighbors ==")
        for r in rows:
            print(f"  {r['iajd_id']!s:>10}  tanimoto={r['tanimoto']:.3f}  family={r['family']}")
