"""
compute_molgpka_live.py — run REAL MolGpKa (Xundrug/MolGpKa GCN) for every
SMILES in the pKa training table and replace molgpka_preds.npy with the live
predictions. Eliminates the back-calculated-via-inverse-debias proxies that
rebuild_pka_model.py used for cross-populated rows.

The convention this codebase uses for the "MolGpKa value" of a compound (as
embedded into the v9.1 31-feature vector) is the *maximum predicted base pKa*
across all detected basic ionization sites; acid pKas and minor sites are
discarded. This matches `_try_live_molgpka()` in
IAJD_master/code/iajd_pka_v71.py:267 which stores `float(max(base_dict.values()))`.

Outputs:
  IAJD_master/bundles_caches/molgpka_preds.npy  (n_compounds float64)
  IAJD_master/datasets/molgpka_preds.npy        (mirror)
  molgpka_live_audit.json                        (per-compound details)
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
MOLGPKA_SRC = ROOT / "molgpka_src"
sys.path.insert(0, str(MOLGPKA_SRC))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

import torch
from utils.descriptor import mol2vec
from utils.ionization_group import get_ionization_aid
from utils.net import GCNNet

PKA_XLSX = ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
OUT_BUNDLES = ROOT / "IAJD_master/bundles_caches/molgpka_preds.npy"
OUT_DATASETS = ROOT / "IAJD_master/datasets/molgpka_preds.npy"
AUDIT_JSON = ROOT / "molgpka_live_audit.json"
MODEL_BASE = ROOT / "molgpka_models/weight_base.pth"
MODEL_ACID = ROOT / "molgpka_models/weight_acid.pth"


def _load(model_path: Path) -> GCNNet:
    m = GCNNet().to("cpu")
    m.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    m.eval()
    return m


def _predict_for_aid(mol, aid: int, model: GCNNet) -> float:
    data = mol2vec(mol, aid)
    with torch.no_grad():
        out = model(data.to("cpu"))
    return float(out.cpu().numpy()[0][0])


def molgpka_max_basic_pka(smiles: str, model_base: GCNNet,
                          model_acid: GCNNet | None = None) -> dict:
    """Return {'value': max_basic_pka, 'base_pkas': {...}, 'acid_pkas': {...}}.

    Mirrors `_try_live_molgpka` in iajd_pka_v71.py: `max(base_dict.values())`.
    """
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"value": float("nan"), "base_pkas": {}, "acid_pkas": {},
                "error": "smiles_parse_failed"}
    try:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol_h = AllChem.AddHs(mol)
    except Exception as exc:
        return {"value": float("nan"), "base_pkas": {}, "acid_pkas": {},
                "error": f"uncharge_failed:{exc}"}
    base_pkas = {}
    base_aids = get_ionization_aid(mol_h, acid_or_base="base")
    for aid in base_aids:
        try:
            base_pkas[int(aid)] = _predict_for_aid(mol_h, aid, model_base)
        except Exception as exc:
            base_pkas[int(aid)] = float("nan")
    acid_pkas = {}
    if model_acid is not None:
        acid_aids = get_ionization_aid(mol_h, acid_or_base="acid")
        for aid in acid_aids:
            try:
                acid_pkas[int(aid)] = _predict_for_aid(mol_h, aid, model_acid)
            except Exception as exc:
                acid_pkas[int(aid)] = float("nan")
    if base_pkas and not all(np.isnan(v) for v in base_pkas.values()):
        value = float(max(v for v in base_pkas.values() if np.isfinite(v)))
    else:
        value = float("nan")
    return {"value": value, "base_pkas": base_pkas, "acid_pkas": acid_pkas}


def main():
    print(f"Loading pKa table from {PKA_XLSX.name}…")
    # Read the first sheet (post-audit canonical file uses 'Sheet1', not 'Dataset').
    df = pd.read_excel(PKA_XLSX, sheet_name=0)
    # Match iajd_pka_v52.build_bundle: exclude audit-flagged rows so the saved
    # molgpka_preds.npy aligns 1:1 (length + order) with the pKa training bundle.
    if "audit_status" in df.columns:
        _flag = df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
        if int(_flag.sum()):
            print(f"  excluding {int(_flag.sum())} audit-flagged rows (align with bundle)")
        df = df[~_flag].reset_index(drop=True)
    smis = df["SMILES"].tolist()
    print(f"  {len(smis)} compounds")

    print(f"Loading MolGpKa weights…")
    model_base = _load(MODEL_BASE)
    model_acid = _load(MODEL_ACID)
    print(f"  base + acid models loaded (n_features=29, hidden=1024)")

    print(f"\nRunning live MolGpKa on every compound…")
    values = np.full(len(smis), np.nan)
    audit = []
    failed = 0
    for i, smi in enumerate(smis):
        res = molgpka_max_basic_pka(smi, model_base, model_acid)
        values[i] = res["value"]
        if np.isnan(res["value"]):
            failed += 1
        audit.append({
            "iajd": int(df.iloc[i]["IAJD"]) if pd.notna(df.iloc[i]["IAJD"]) else None,
            "smiles": smi,
            "value": res["value"],
            "n_base": len(res["base_pkas"]),
            "n_acid": len(res["acid_pkas"]),
            "error": res.get("error"),
        })
        if (i + 1) % 25 == 0 or i == len(smis) - 1:
            print(f"  {i+1}/{len(smis)}  failed so far: {failed}", flush=True)

    print(f"\nDone. {failed}/{len(smis)} failed.")
    np.save(OUT_BUNDLES, values)
    np.save(OUT_DATASETS, values)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))
    print(f"  wrote {OUT_BUNDLES.name} (× 2) and {AUDIT_JSON.name}")

    # Quick stats
    valid = values[np.isfinite(values)]
    print(f"\n  Range: [{valid.min():.2f}, {valid.max():.2f}]  median: {np.median(valid):.2f}")
    print(f"  vs. measured pKa correlation:")
    pkas = df["pKa"].values
    mask = np.isfinite(values) & np.isfinite(pkas)
    if mask.sum() >= 3:
        r = float(np.corrcoef(values[mask], pkas[mask])[0, 1])
        print(f"    Pearson r = {r:.3f} on {mask.sum()} pairs")


if __name__ == "__main__":
    main()
