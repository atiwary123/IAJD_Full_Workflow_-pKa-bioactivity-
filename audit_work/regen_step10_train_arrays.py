"""regen_step10_train_arrays.py — rebuild the stacker's train-aligned feature
arrays so they match the NEW bioact_v14 bundle EXACTLY (row count + order).

The pre-regen agile_embeddings_v14_train.npy / cpp_features_v14_train.npy were
(335, …) while the bundle had 247 rows — a latent misalignment. We rebuild both
strictly aligned to bundle['smis_train'] (236 rows post-audit):

  agile_embeddings_v14_train.npy : (n, 512)  AGILE 60k frozen encoder on smis_train
  cpp_features_v14_train.npy     : (n, 23)   compute_cpp_features(smi, pKa)

CPP pKa input per compound (real, corrected-structure-consistent):
  measured 'pKa'  ->  'pKa_paper'  ->  v11 predicted_pKa (by row_id, last resort).
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master" / "code"))
BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
BIOACT = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
V11 = ROOT / "v11_pka_flux/predicted_pka_cache.csv"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def build_pka_map():
    b = pd.read_excel(BIOACT, sheet_name=0)
    v11 = pd.read_csv(V11) if V11.exists() else None
    v11_by_rowid = dict(zip(v11["row_id"], v11["predicted_pKa"])) if v11 is not None else {}
    m = {}
    for _, r in b.iterrows():
        c = canon(r.get("SMILES_canonical") or r.get("SMILES"))
        if not c:
            continue
        pka = None
        for col in ("pKa", "pKa_paper"):
            v = r.get(col)
            if pd.notna(v):
                try:
                    pka = float(v); break
                except (TypeError, ValueError):
                    pass
        if pka is None:
            rid = r.get("row_id")
            if rid in v11_by_rowid and pd.notna(v11_by_rowid[rid]):
                pka = float(v11_by_rowid[rid])
        if c not in m and pka is not None:
            m[c] = pka
    return m


def main():
    with open(BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    smis = list(bundle["smis_train"])
    n = len(smis)
    print(f"bundle smis_train: {n}")

    pka_map = build_pka_map()
    pk_hit = sum(1 for s in smis if s in pka_map)
    print(f"pKa available for CPP: {pk_hit}/{n}")

    # ---- AGILE embeddings aligned to smis_train ----
    from agile_embeddings import load_agile_encoder, extract_embeddings
    model = load_agile_encoder()
    agile = extract_embeddings(smis, model)
    assert agile.shape == (n, 512), f"agile shape {agile.shape}"
    nan_rows = int(np.isnan(agile).any(axis=1).sum())
    np.save(ROOT / "agile_embeddings_v14_train.npy", agile)
    print(f"agile_embeddings_v14_train.npy {agile.shape}  NaN_rows={nan_rows}")

    # ---- CPP features aligned to smis_train ----
    from compute_cpp import compute_cpp_features, CPP_FEATURE_NAMES
    rows = []
    n_none = 0
    for s in smis:
        feats = compute_cpp_features(s, pka=pka_map.get(s))
        if feats is None:
            n_none += 1
            rows.append({k: np.nan for k in CPP_FEATURE_NAMES})
        else:
            rows.append({k: feats.get(k, np.nan) for k in CPP_FEATURE_NAMES})
    X_cpp = pd.DataFrame(rows)[CPP_FEATURE_NAMES].to_numpy(dtype=float)
    assert X_cpp.shape == (n, len(CPP_FEATURE_NAMES)), f"cpp shape {X_cpp.shape}"
    np.save(ROOT / "cpp_features_v14_train.npy", X_cpp)
    print(f"cpp_features_v14_train.npy {X_cpp.shape}  "
          f"rows_with_no_CPP={n_none}  NaN_frac={np.isnan(X_cpp).mean():.3f}")

    print("\nDONE — both arrays aligned to bundle smis_train order.")


if __name__ == "__main__":
    main()
