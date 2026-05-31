"""
train_qm_emulator.py — train a small multi-output XGBoost regressor that
predicts the 8 QM descriptors from cheap RDKit features.

Why this exists:
  qm_cache.csv covers every SMILES in the training set + every proposer
  candidate the Mac-offline driver has already run. For live novel queries
  that the Space hits at inference time without a cache hit, the emulator
  bridges the gap. It is NOT a replacement for the cache — keep the cache
  coverage high.

Pipeline:
  features  = RDKit descriptors (~30 cheap ones: MW, logP, TPSA, fragment
              counts, fp-derived counts, Crippen, Chi, BertzCT, etc.)
  targets   = QM_KEYS (8 outputs)
  model     = MultiOutputRegressor(XGBRegressor) with conservative depth.

Output:
  IAJD_master/bundles_caches/physics/qm_emulator.joblib

Contract on predict():
  emulate_qm(smiles) -> dict with the same 8 keys (NaN on RDKit parse failure
  or model unavailable).
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

RDLogger.logger().setLevel(RDLogger.ERROR)

from qm_descriptors import QM_KEYS

CACHE_CSV = ROOT / "IAJD_master/bundles_caches/physics/qm_cache.csv"
EMULATOR_PATH = ROOT / "IAJD_master/bundles_caches/physics/qm_emulator.joblib"
REPORT_PATH = ROOT / "IAJD_master/bundles_caches/physics/qm_emulator_report.json"


# ──────────────────────────────────────────────────────────────────────
# Feature vector — cheap, deterministic, RDKit-only
# ──────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "MW", "HeavyAtomCount", "MolLogP", "TPSA", "LabuteASA", "FractionCSP3",
    "RotatableBonds", "BertzCT", "Chi0v", "Chi1v", "HallKierAlpha",
    "NumAromaticRings", "NumHDonors", "NumHAcceptors", "NumNitrogens",
    "NumOxygens", "NumSp3Carbons", "NumSp2Carbons", "NumEsters", "NumAmides",
    "NumEthers", "NumTertiaryAmines", "HasPiperazine",
    "n_amines_basic_tertN", "n_aliphatic_C", "n_aromatic_C",
    "GasteigerN_min", "GasteigerN_max", "GasteigerN_mean",
    "HBD_HBA_ratio", "Crippen_MR",
]


def featurize(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Gasteiger charges (deterministic).
        AllChem.ComputeGasteigerCharges(mol)
        n_charges = [float(a.GetProp("_GasteigerCharge"))
                     for a in mol.GetAtoms()
                     if a.GetAtomicNum() == 7
                     and a.HasProp("_GasteigerCharge")]
        gN_min = min(n_charges) if n_charges else 0.0
        gN_max = max(n_charges) if n_charges else 0.0
        gN_mean = float(np.mean(n_charges)) if n_charges else 0.0
    except (RuntimeError, ValueError):
        gN_min = gN_max = gN_mean = 0.0

    ester = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    ether = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    tert_amine = Chem.MolFromSmarts("[NX3]([#6])([#6])[#6]")
    pip = Chem.MolFromSmarts("C1CNCCN1")
    basic_n = Chem.MolFromSmarts("[#7;X3;!$(N=*);!$(NC=O);!$(Nc)]")

    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    n_O = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_sp3C = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3)
    n_sp2C = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP2)
    n_arom_C = sum(1 for a in mol.GetAtoms()
                    if a.GetAtomicNum() == 6 and a.GetIsAromatic())
    n_aliph_C = sum(1 for a in mol.GetAtoms()
                     if a.GetAtomicNum() == 6 and not a.GetIsAromatic())

    feats = [
        Descriptors.ExactMolWt(mol),
        mol.GetNumHeavyAtoms(),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.BertzCT(mol),
        Descriptors.Chi0v(mol),
        Descriptors.Chi1v(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.NumAromaticRings(mol),
        hbd,
        hba,
        n_N,
        n_O,
        n_sp3C,
        n_sp2C,
        len(mol.GetSubstructMatches(ester)),
        len(mol.GetSubstructMatches(amide)),
        len(mol.GetSubstructMatches(ether)),
        len(mol.GetSubstructMatches(tert_amine)),
        1 if mol.HasSubstructMatch(pip) else 0,
        len(mol.GetSubstructMatches(basic_n)),
        n_aliph_C,
        n_arom_C,
        gN_min,
        gN_max,
        gN_mean,
        (hbd / max(hba, 1)) if hba > 0 else 0.0,
        Crippen.MolMR(mol),
    ]
    return np.array(feats, dtype=float)


# ──────────────────────────────────────────────────────────────────────
# Train + report
# ──────────────────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=2.0,
    min_child_weight=2, objective="reg:squarederror",
    tree_method="hist", n_jobs=2, random_state=42, verbosity=0,
)


def train(qm_cache_path: Path, emulator_out: Path) -> Dict:
    df = pd.read_csv(qm_cache_path)
    print(f"Loaded {len(df)} cached QM records from {qm_cache_path.name}")
    print("Featurizing…")
    feats: List[Optional[np.ndarray]] = [featurize(s) for s in df["smiles_canonical"]]
    valid_mask = np.array([f is not None for f in feats])
    X_all = np.vstack([f for f in feats if f is not None])
    df_valid = df[valid_mask].reset_index(drop=True)

    print(f"  {X_all.shape[0]} rows × {X_all.shape[1]} features")
    Y_all = df_valid[list(QM_KEYS)].to_numpy(dtype=float)

    # Train per-target individually so missing rows don't bias one target's
    # global model.
    print("\nTraining one XGB per target (NaN-masked):")
    models: Dict[str, XGBRegressor] = {}
    per_target_metrics: Dict[str, Dict] = {}
    for j, key in enumerate(QM_KEYS):
        y = Y_all[:, j]
        mask = np.isfinite(y)
        if mask.sum() < 30:
            print(f"  {key:24s} insufficient data ({mask.sum()}) — skip")
            continue
        Xj = X_all[mask]
        yj = y[mask]
        # CV-validated metric
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros_like(yj)
        for tr, te in kf.split(Xj):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(Xj[tr], yj[tr])
            preds[te] = m.predict(Xj[te])
        cv_mae = float(mean_absolute_error(yj, preds))
        cv_r2 = float(r2_score(yj, preds))
        # Fit final model on all valid rows.
        final = XGBRegressor(**XGB_PARAMS)
        final.fit(Xj, yj)
        models[key] = final
        per_target_metrics[key] = {
            "n_train": int(mask.sum()),
            "cv_mae": cv_mae,
            "cv_r2": cv_r2,
            "y_std": float(np.std(yj)),
        }
        print(f"  {key:24s} n={mask.sum():>4d}  CV-MAE={cv_mae:8.3f}  "
              f"R²={cv_r2:+.3f}  σ(y)={np.std(yj):.3f}")

    bundle = {
        "version": "qm_emulator_v1",
        "feature_names": FEATURE_NAMES,
        "qm_keys": list(QM_KEYS),
        "models": models,
        "per_target_metrics": per_target_metrics,
        "n_train_records": int(len(df_valid)),
    }
    emulator_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, emulator_out)
    print(f"\nSaved → {emulator_out}")
    REPORT_PATH.write_text(json.dumps(per_target_metrics, indent=2))
    return bundle


# ──────────────────────────────────────────────────────────────────────
# Inference (used by physics_cache_io fallback)
# ──────────────────────────────────────────────────────────────────────

def emulate_qm(smiles: str, bundle_path: Optional[Path] = None) -> Dict[str, float]:
    """Predict QM_KEYS for a SMILES using the emulator. NaN on any failure."""
    bp = bundle_path or EMULATOR_PATH
    if not bp.exists():
        return {k: float("nan") for k in QM_KEYS}
    try:
        bundle = joblib.load(bp)
    except (OSError, ValueError):
        return {k: float("nan") for k in QM_KEYS}
    feat = featurize(smiles)
    if feat is None:
        return {k: float("nan") for k in QM_KEYS}
    X = feat.reshape(1, -1)
    out = {}
    for k in QM_KEYS:
        m = bundle["models"].get(k)
        if m is None:
            out[k] = float("nan")
        else:
            try:
                out[k] = float(m.predict(X)[0])
            except (ValueError, RuntimeError):
                out[k] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE_CSV)
    parser.add_argument("--out", type=Path, default=EMULATOR_PATH)
    args = parser.parse_args()
    if not args.cache.exists():
        print(f"ERROR: cache not found: {args.cache}")
        return 2
    train(args.cache, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
