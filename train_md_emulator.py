"""
train_md_emulator.py — train a per-target XGBoost regressor that predicts the
~10 MARTINI MD observables (md_a_head, md_bilayer_thick, md_cpp_*, etc.) from
cheap features: grammar features + RDKit descriptors + xTB charges where
available.

Cache-first policy: this emulator is a Space-side fallback for novel queries
that aren't already in md_cache.csv. Most training-set + proposer compounds
should hit the real cache. The emulator never "fills in" rows already in the
cache — both pipelines coexist via physics_cache_io.

Inputs:
  IAJD_master/bundles_caches/physics/md_cache.csv  (key smiles_canonical + MD_KEYS)
  IAJD_master/bundles_caches/physics/qm_cache.csv  (optional, for xTB charge)
  IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx  (for grammar features)

Output:
  IAJD_master/bundles_caches/physics/md_emulator.joblib
  IAJD_master/bundles_caches/physics/md_emulator_report.json
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
RDLogger.logger().setLevel(RDLogger.ERROR)

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"
MD_CACHE = PHYS_DIR / "md_cache.csv"
QM_CACHE = PHYS_DIR / "qm_cache.csv"
EMULATOR_PATH = PHYS_DIR / "md_emulator.joblib"
REPORT_PATH = PHYS_DIR / "md_emulator_report.json"
BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"

# Targets — match physics_cache_io.MD_KEYS.
MD_KEYS = (
    "md_assembles", "md_n_agg",
    "md_a_head_neutral_nm2", "md_a_head_prot_nm2", "md_delta_a_head_nm2",
    "md_bilayer_thick_nm", "md_order_param", "md_radius_gyration_nm",
    "md_water_penetration",
    "md_cpp_neutral", "md_cpp_prot", "md_delta_cpp",
    "md_c0_spontaneous",
)


# ──────────────────────────────────────────────────────────────────────
# Feature vector — same shape as qm_emulator's featurize (so reuse it),
# plus grammar features (family, head, linker_n, tails) and optional xTB
# charges.
# ──────────────────────────────────────────────────────────────────────

def _grammar_features(seed) -> Dict[str, float]:
    """One-hot family + head + linkage + linker_n + tail descriptors."""
    out = {
        "fam_sSSNonsym":     1.0 if seed.family == "sSS-Nonsym" else 0.0,
        "fam_GATris":        1.0 if seed.family == "GA-Tris" else 0.0,
        "fam_PETris":        1.0 if seed.family == "PE-Tris" else 0.0,
        "fam_PEGallic":      1.0 if seed.family == "PE-Gallic" else 0.0,
        "fam_Dialkoxybenzyl":1.0 if seed.family == "Dialkoxybenzyl" else 0.0,
        "head_DMA":          1.0 if seed.head == "DMA" else 0.0,
        "head_MPRZ":         1.0 if seed.head == "MPRZ" else 0.0,
        "head_PIP":          1.0 if seed.head == "PIP" else 0.0,
        "head_HPRZ":         1.0 if seed.head == "HPRZ" else 0.0,
        "head_H2EPRZ":       1.0 if seed.head == "H2EPRZ" else 0.0,
        "head_DMBA":         1.0 if seed.head == "DMBA" else 0.0,
        "linkage_ester":     1.0 if seed.linkage == "ester" else 0.0,
        "linker_n":          float(seed.linker_n),
        "n_tails":           float(len(seed.tails)),
    }
    nC = []
    branched = 0
    for t in seed.tails:
        m = Chem.MolFromSmiles(t)
        if m:
            nC.append(sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6))
            if any(len(list(a.GetNeighbors())) > 2
                    for a in m.GetAtoms() if a.GetAtomicNum() == 6):
                branched += 1
    out["tail_nC_mean"]   = float(np.mean(nC)) if nC else 0.0
    out["tail_nC_max"]    = float(np.max(nC)) if nC else 0.0
    out["tail_nC_min"]    = float(np.min(nC)) if nC else 0.0
    out["tail_branched"]  = float(branched)
    return out


def featurize(smiles: str, row: Optional[dict] = None,
               qm_row: Optional[dict] = None,
               seed=None) -> Optional[np.ndarray]:
    from train_qm_emulator import featurize as qm_feat
    feat_qm = qm_feat(smiles)
    if feat_qm is None:
        return None
    # Grammar features need the row; if missing we use a zero vector.
    if seed is None and row is not None:
        from iajd_grammar import decompose_row
        seed = decompose_row(row)
    gf = _grammar_features(seed) if seed is not None else {}
    # Pack in a deterministic order.
    grammar_keys = [
        "fam_sSSNonsym", "fam_GATris", "fam_PETris", "fam_PEGallic", "fam_Dialkoxybenzyl",
        "head_DMA", "head_MPRZ", "head_PIP", "head_HPRZ", "head_H2EPRZ", "head_DMBA",
        "linkage_ester", "linker_n", "n_tails",
        "tail_nC_mean", "tail_nC_max", "tail_nC_min", "tail_branched",
    ]
    g_vec = np.array([gf.get(k, 0.0) for k in grammar_keys], dtype=float)
    # Optional QM cache contribution (4 keys most predictive for MD).
    qm_keys = ["qm_q_ionizableN", "qm_dipole_D", "qm_dGsolv_kJmol", "qm_polarizability"]
    if qm_row is not None:
        q_vec = np.array([float(qm_row.get(k, np.nan)) for k in qm_keys], dtype=float)
    else:
        q_vec = np.full(len(qm_keys), np.nan, dtype=float)
    return np.concatenate([feat_qm, g_vec, q_vec])


FEATURE_NAMES_EXTRA = [
    "fam_sSSNonsym", "fam_GATris", "fam_PETris", "fam_PEGallic", "fam_Dialkoxybenzyl",
    "head_DMA", "head_MPRZ", "head_PIP", "head_HPRZ", "head_H2EPRZ", "head_DMBA",
    "linkage_ester", "linker_n", "n_tails",
    "tail_nC_mean", "tail_nC_max", "tail_nC_min", "tail_branched",
    "qm_q_ionizableN", "qm_dipole_D", "qm_dGsolv_kJmol", "qm_polarizability",
]


# ──────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.7, reg_lambda=3.0,
    min_child_weight=2, objective="reg:squarederror",
    tree_method="hist", n_jobs=2, random_state=42, verbosity=0,
)


def train(md_cache: Path, qm_cache: Optional[Path], xlsx: Path,
           emulator_out: Path) -> Dict:
    print(f"Loading MD cache: {md_cache}")
    df_md = pd.read_csv(md_cache)
    print(f"  {len(df_md)} rows")
    df_qm = pd.read_csv(qm_cache).set_index("smiles_canonical") if qm_cache and qm_cache.exists() else None
    df_bio = pd.read_excel(xlsx)
    bio_by_smi = {}
    for _, r in df_bio.iterrows():
        m = Chem.MolFromSmiles(r.get("SMILES_canonical") or "")
        if m is None:
            continue
        bio_by_smi[Chem.MolToSmiles(m)] = r.to_dict()

    feats_list = []
    rows_keep = []
    for _, row in df_md.iterrows():
        smi = row["smiles_canonical"]
        qm_row = df_qm.loc[smi].to_dict() if (df_qm is not None and smi in df_qm.index) else None
        bio_row = bio_by_smi.get(smi)
        f = featurize(smi, row=bio_row, qm_row=qm_row)
        if f is None:
            continue
        feats_list.append(f)
        rows_keep.append(row)
    X = np.vstack(feats_list)
    df_train = pd.DataFrame(rows_keep).reset_index(drop=True)
    print(f"  features: X = {X.shape}")

    models = {}
    per_target_metrics = {}
    for key in MD_KEYS:
        if key not in df_train.columns:
            continue
        y = df_train[key].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 30:
            print(f"  {key:24s} insufficient data ({mask.sum()})")
            continue
        Xj = X[mask]
        yj = y[mask]
        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros_like(yj)
        for tr, te in kf.split(Xj):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(Xj[tr], yj[tr])
            preds[te] = m.predict(Xj[te])
        cv_mae = float(mean_absolute_error(yj, preds))
        cv_r2 = float(r2_score(yj, preds))
        final = XGBRegressor(**XGB_PARAMS)
        final.fit(Xj, yj)
        models[key] = final
        per_target_metrics[key] = {
            "n_train": int(mask.sum()),
            "cv_mae": cv_mae, "cv_r2": cv_r2,
            "y_std": float(np.std(yj)),
        }
        print(f"  {key:24s} n={mask.sum():>4d}  CV-MAE={cv_mae:8.3f}  "
              f"R²={cv_r2:+.3f}  σ(y)={np.std(yj):.3f}")

    bundle = {
        "version": "md_emulator_v1",
        "feature_names_extra": FEATURE_NAMES_EXTRA,
        "md_keys": list(MD_KEYS),
        "models": models,
        "per_target_metrics": per_target_metrics,
        "n_train_records": int(len(df_train)),
    }
    emulator_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, emulator_out)
    REPORT_PATH.write_text(json.dumps(per_target_metrics, indent=2))
    print(f"\nSaved emulator → {emulator_out}")
    return bundle


def emulate_md(smiles: str, bundle_path: Optional[Path] = None,
                row: Optional[dict] = None,
                qm_row: Optional[dict] = None) -> Dict[str, float]:
    bp = bundle_path or EMULATOR_PATH
    if not bp.exists():
        return {k: float("nan") for k in MD_KEYS}
    bundle = joblib.load(bp)
    feat = featurize(smiles, row=row, qm_row=qm_row)
    if feat is None:
        return {k: float("nan") for k in MD_KEYS}
    X = feat.reshape(1, -1)
    out = {}
    for k in MD_KEYS:
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
    parser.add_argument("--md-cache", type=Path, default=MD_CACHE)
    parser.add_argument("--qm-cache", type=Path, default=QM_CACHE)
    parser.add_argument("--xlsx", type=Path, default=BIOACT_XLSX)
    parser.add_argument("--out", type=Path, default=EMULATOR_PATH)
    args = parser.parse_args()
    if not args.md_cache.exists():
        print(f"ERROR: md_cache not found at {args.md_cache}")
        return 2
    train(args.md_cache, args.qm_cache, args.xlsx, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
