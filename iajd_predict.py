"""
iajd_predict.py — End-to-end IAJD prediction (pKa + bioactivity) with Tanimoto routing.

Wraps the tandem bundle in `IAJD_master/code/iajd_tandem_final.py`, computes a 60%
prediction interval to accompany the bundle's native 90% PI, and surfaces the
top Tanimoto neighbors from both training sets (pKa v21 and bioact v13).

Usage
-----
    python iajd_predict.py --smiles "<SMILES>" [--family PE-Tris] \
        [--neighbors 5] [--json out.json] [--quiet]

`--family` is optional. Allowed values:
    sSS-Nonsym, PE-Tris, GA-Tris, PE-Gallic, Dialkoxybenzyl,
    G1-Janus-Dendrimer, HTM-Dendrimer, TT-Dendrimer

Programmatic entry point: `predict(smiles, family=None, neighbors=5)`.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / "IAJD_master" / "code"
CACHE_DIR = HERE / "IAJD_master" / "bundles_caches"
DATA_DIR = HERE / "IAJD_master" / "datasets"

# Locate the external-model venvs for live LION + ADMET-AI inference.
# These run as fully isolated subprocesses so they never share Python state
# (or libomp / libtorch threadpools) with the main v15 process.
_LION_REPO_LOCAL = HERE / "lion_repo"
_LION_PY_LOCAL = HERE / "lion_env" / "bin" / "python3"
_ADMET_PY_LOCAL = HERE / "admet_env" / "bin" / "python3"
if _LION_REPO_LOCAL.exists():
    os.environ.setdefault("LION_REPO", str(_LION_REPO_LOCAL))
if _LION_PY_LOCAL.exists():
    os.environ.setdefault("LION_VENV_PYTHON", str(_LION_PY_LOCAL))
elif not os.environ.get("LION_VENV_PYTHON"):
    os.environ["LION_VENV_PYTHON"] = sys.executable
if _ADMET_PY_LOCAL.exists():
    os.environ.setdefault("ADMET_VENV_PYTHON", str(_ADMET_PY_LOCAL))
elif not os.environ.get("ADMET_VENV_PYTHON"):
    os.environ["ADMET_VENV_PYTHON"] = sys.executable
os.environ.setdefault("IAJD_OUT_DIR", str(CACHE_DIR))

sys.path.insert(0, str(CODE_DIR))

from iajd_neighbors import tanimoto_neighbors  # noqa: E402
from iajd_v15 import (  # noqa: E402
    load_v15, predict_pka_v15, predict_bioactivity_v15, V15Bundle,
)
from iajd_structural import (  # noqa: E402
    structural_features, refine_analog_prediction, feature_dim,
)
from iajd_family import (  # noqa: E402
    resolve_family, CHEMICAL_FAMILIES, BIOACT_ONLY_SUBARCHS,
)

ALLOWED_FAMILIES = {
    "sSS-Nonsym", "PE-Tris", "GA-Tris", "PE-Gallic", "Dialkoxybenzyl",
    "G1-Janus-Dendrimer",
}

# 90% -> sigma -> 60%: sigma = half90 / 1.645; half60 = 0.8416 * sigma.
_PI60_FROM_PI90 = 0.8416 / 1.6449  # ≈ 0.5116

_BUNDLE = None
_V15: V15Bundle | None = None
_REFIT_CACHE = HERE / ".cache" / "bioact_direct_xgb_refit.pkl"
_STACKER_BUNDLE_PATH = CACHE_DIR / "bioact_stacker_bundle.pkl"
_STACKER_BUNDLE = None  # lazy-loaded


def _load_bioact_stacker():
    """Lazy-load the bioactivity stacker bundle (direct/lion/admet/stacker XGBs).
    Returns None if the bundle isn't present (production falls back to v15 path)."""
    global _STACKER_BUNDLE
    if _STACKER_BUNDLE is not None:
        return _STACKER_BUNDLE
    if not _STACKER_BUNDLE_PATH.exists():
        return None
    import pickle
    try:
        with open(_STACKER_BUNDLE_PATH, "rb") as f:
            _STACKER_BUNDLE = pickle.load(f)
    except Exception:  # noqa: BLE001
        _STACKER_BUNDLE = None
    return _STACKER_BUNDLE


def _bioact_stacker_predict(X_full_row, analog_pred, smiles=None, max_tanimoto=None):
    """Apply the bioactivity stacker. Adaptive version (v3):
    dynamically weights 6 heads based on query novelty.

    Returns {"point": float, "direct": float, "analog": float,
             "lion": float, "admet": float, "agile": float, "cpp": float,
             "novelty": float, "weights": dict}
    or None if the bundle isn't loaded."""
    import numpy as np
    sb = _load_bioact_stacker()
    if sb is None:
        return None
    x = np.asarray(X_full_row, dtype=float).reshape(1, -1)
    bl_a, bl_b = sb["block_lion"]; bc_a, bc_b = sb["block_admet"]
    direct = float(sb["direct_head"].predict(x)[0])
    lion = float(sb["lion_head"].predict(x[:, bl_a:bl_b])[0])
    admet = float(sb["admet_head"].predict(x[:, bc_a:bc_b])[0])

    # AGILE head
    agile_val = None
    if "agile_head" in sb and smiles:
        try:
            agile_val = _agile_predict_single(smiles, sb)
        except Exception:
            agile_val = None

    # CPP head
    cpp_val = None
    if "cpp_head" in sb and smiles:
        try:
            cpp_val = _cpp_predict_single(smiles, sb)
        except Exception:
            cpp_val = None

    head_preds = {"direct": direct, "analog": float(analog_pred),
                  "lion": lion, "admet": admet}
    if agile_val is not None:
        head_preds["agile"] = agile_val
    if cpp_val is not None:
        head_preds["cpp"] = cpp_val

    # Adaptive weighting if params available
    adaptive = sb.get("adaptive_params")
    if adaptive and max_tanimoto is not None:
        novelty = 1.0 - max_tanimoto
        base_w = adaptive["base_weights"]
        k_decay = adaptive["k_decay"]
        k_grow = adaptive["k_grow"]

        HEAD_TYPES = {'direct':'stable','analog':'similarity','lion':'gnn',
                      'admet':'gnn','agile':'physics','cpp':'physics'}
        raw_w = {}
        for h in head_preds:
            bw = base_w.get(h, 0.01)
            ht = HEAD_TYPES.get(h, 'stable')
            if ht == 'similarity':
                raw_w[h] = bw * np.exp(-k_decay * novelty)
            elif ht == 'gnn':
                raw_w[h] = bw * np.exp(-k_decay * 0.5 * novelty)
            elif ht == 'physics':
                raw_w[h] = bw * np.exp(k_grow * novelty)
            else:
                raw_w[h] = bw

        total = sum(raw_w.values())
        if total == 0: total = 1.0
        weights = {h: raw_w[h]/total for h in raw_w}
        point = sum(weights[h] * head_preds[h] for h in head_preds)

        result = {"point": point, **head_preds, "novelty": novelty, "weights": weights,
                  "mode": "adaptive"}
    else:
        # Fallback: static stacker if available
        if "stacker" in sb:
            feats_list = [direct, float(analog_pred), lion, admet]
            if agile_val is not None and "agile" in sb.get("stack_features", []):
                feats_list.append(agile_val)
            if cpp_val is not None and "cpp" in sb.get("stack_features", []):
                feats_list.append(cpp_val)
            feats = np.asarray([feats_list])
            try:
                point = float(sb["stacker"].predict(feats)[0])
            except Exception:
                point = float(np.mean(list(head_preds.values())))
        else:
            point = float(np.mean(list(head_preds.values())))

        result = {"point": point, **head_preds, "mode": "static"}

    return result


_AGILE_CACHE = {}

def _agile_predict_single(smiles, sb):
    """Get AGILE prediction for a SMILES. Uses precomputed cache or
    lazy-loads the encoder if memory allows."""
    import numpy as np
    if smiles in _AGILE_CACHE:
        return _AGILE_CACHE[smiles]

    try:
        import torch
        from agile_embeddings import load_agile_encoder, smiles_to_graph
        from torch_geometric.data import Batch

        global _AGILE_ENCODER
        if '_AGILE_ENCODER' not in globals():
            _AGILE_ENCODER = None
        if _AGILE_ENCODER is None:
            _AGILE_ENCODER = load_agile_encoder()
        if _AGILE_ENCODER is False:
            return None

        g = smiles_to_graph(smiles)
        if g is None:
            return None
        with torch.no_grad():
            h, _ = _AGILE_ENCODER(Batch.from_data_list([g]))
        emb = h.cpu().numpy()
        scaled = sb["agile_scaler"].transform(emb)
        pca_emb = sb["agile_pca"].transform(scaled)
        val = float(sb["agile_head"].predict(pca_emb)[0])
        _AGILE_CACHE[smiles] = val
        return val
    except Exception:
        return None


_CPP_CACHE = {}

def _cpp_predict_single(smiles, sb):
    """Compute CPP features and predict via CPP head."""
    if smiles in _CPP_CACHE:
        return _CPP_CACHE[smiles]
    import numpy as np
    try:
        from compute_cpp import compute_cpp_features, CPP_FEATURE_NAMES
        feats = compute_cpp_features(smiles, pka=None)
        if feats is None:
            return None
        x = np.array([[feats.get(k, 0) for k in CPP_FEATURE_NAMES]])
        # Impute NaN with 0
        x = np.nan_to_num(x, nan=0.0)
        val = float(sb["cpp_head"].predict(x)[0])
        _CPP_CACHE[smiles] = val
        return val
    except Exception:
        return None


def _warm_agile_cache():
    """Precompute AGILE predictions for all training SMILES at startup."""
    import numpy as np
    sb = _load_bioact_stacker()
    if sb is None or "agile_head" not in sb:
        return
    emb_path = Path(__file__).resolve().parent / "agile_embeddings_v14_train.npy"
    smi_path = CACHE_DIR.parent / "bioact_v14_bundle.pkl"
    if not emb_path.exists() or not smi_path.exists():
        return
    try:
        import pickle
        with open(smi_path, "rb") as f:
            smiles_list = list(pickle.load(f)["smis_train"])
        emb = np.load(emb_path)
        scaled = sb["agile_scaler"].transform(emb)
        pca_emb = sb["agile_pca"].transform(scaled)
        preds = sb["agile_head"].predict(pca_emb)
        for smi, pred in zip(smiles_list, preds):
            _AGILE_CACHE[smi] = float(pred)
    except Exception:
        pass

# Cache of structural feature vectors keyed by canonical SMILES. Training-set
# entries are populated lazily as neighbors surface from v15; same vector dim
# (`feature_dim()`) for queries and training. ChemDraw queries that supply
# their own 2D coords bypass this cache (their features get computed from the
# user-drawn coordinates).
_STRUCT_FEAT_CACHE: Dict[str, "np.ndarray"] = {}


def _struct_feat_for_smiles(smiles: str):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    cached = _STRUCT_FEAT_CACHE.get(smiles)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        vec = np.zeros(feature_dim(), dtype=float)
    else:
        AllChem.Compute2DCoords(mol)
        try:
            vec = structural_features(mol)
        except Exception:  # noqa: BLE001
            vec = np.zeros(feature_dim(), dtype=float)
    _STRUCT_FEAT_CACHE[smiles] = vec
    return vec


def _struct_feat_for_mol(mol):
    import numpy as np
    if mol is None:
        return np.zeros(feature_dim(), dtype=float)
    try:
        return structural_features(mol)
    except Exception:  # noqa: BLE001
        return np.zeros(feature_dim(), dtype=float)


def _apply_structural_refinement(
    stage_block: Dict[str, Any],
    analog_neighbors: list,
    query_feat,
    beta: float = 0.25,
) -> Dict[str, Any]:
    """Recompute analog-delta with structurally-blended weights and rebuild
    the final blend = alpha * direct + (1 - alpha) * analog_refined.

    Returns a refinement dict that includes the refined point + a struct trace.
    Caller is expected to attach it to the stage block; the *original* values
    in `stage_block` are preserved untouched."""
    if not analog_neighbors:
        return {"applied": False, "reason": "no_analog_neighbors"}
    neighbor_feats = [_struct_feat_for_smiles(n["smiles"]) for n in analog_neighbors]
    refined = refine_analog_prediction(
        analog_neighbors, query_feat, neighbor_feats, beta=beta,
    )
    if refined["refined_point"] is None:
        return {"applied": False, "reason": "no_refined_point"}
    direct = stage_block.get("pred_direct", stage_block.get("direct_xgb_pred"))
    analog = stage_block.get("pred_analog", stage_block.get("analog_delta_pred"))
    alpha = stage_block.get("alpha_used")
    refined_pt = float(refined["refined_point"])
    if direct is None or alpha is None:
        new_point = refined_pt
    else:
        new_point = float(alpha) * float(direct) + (1.0 - float(alpha)) * refined_pt
    # Preserve the same CI half-widths (the refinement nudges the point but
    # tier confidence is dictated by max_tanimoto, which hasn't changed).
    sigma = stage_block.get("sigma")
    ci60 = ci90 = None
    if sigma is not None:
        h60 = 0.8416 * float(sigma)
        h90 = 1.6449 * float(sigma)
        ci60 = [round(new_point - h60, 3), round(new_point + h60, 3)]
        ci90 = [round(new_point - h90, 3), round(new_point + h90, 3)]
    return {
        "applied": True,
        "beta": beta,
        "point": round(new_point, 3),
        "analog_refined": round(refined_pt, 3),
        "ci_60": ci60,
        "ci_90": ci90,
        "delta_from_original": round(new_point - float(stage_block["point"]), 3),
        "sim_blend_per_neighbor": refined["sim_blend_per_neighbor"],
        "weight_per_neighbor": refined["weight_per_neighbor"],
        "struct_sim_per_neighbor": refined["struct_sim_per_neighbor"],
    }


def _rescue_direct_model(bio_bundle) -> bool:
    """Replace the pickled XGBoost direct model if it predicts nonsense.

    XGBoost models pickled in an older version often unpickle without error but
    yield meaningless predictions (the regressor we observed predicted ~0.3 on
    training rows whose y was ~7.5). When that happens we refit on the stored
    X_train / y_train. Sanity criterion: MAE on a held-out 20% > 2.0 log units.

    Returns True if the model was replaced.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    X = np.asarray(bio_bundle["X_train"], dtype=float)
    y = np.asarray(bio_bundle["y_train"], dtype=float)
    try:
        preds = bio_bundle["direct_model_full"].predict(X[:32])
        sample_mae = float(np.mean(np.abs(preds - y[:32])))
    except Exception:  # noqa: BLE001
        sample_mae = float("inf")

    if sample_mae <= 2.0:
        return False  # model is fine

    if _REFIT_CACHE.exists():
        import pickle as _pickle
        with open(_REFIT_CACHE, "rb") as f:
            bio_bundle["direct_model_full"] = _pickle.load(f)
        return True

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    fresh = xgb.XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=1,
    )
    fresh.fit(Xtr, ytr, verbose=False)
    # quick honesty check
    held_out_mae = float(np.mean(np.abs(fresh.predict(Xte) - yte)))
    # train on full set for production
    fresh.fit(X, y, verbose=False)
    bio_bundle["direct_model_full"] = fresh
    bio_bundle.setdefault("metrics", {})["refit_holdout_mae"] = held_out_mae
    import pickle as _pickle
    _REFIT_CACHE.parent.mkdir(exist_ok=True)
    with open(_REFIT_CACHE, "wb") as f:
        _pickle.dump(fresh, f)
    return True


def _stage_pka_artifacts() -> None:
    """The v91 loader resolves molgpka_*.joblib/.npy relative to the pKa xlsx's
    directory. The HF Spaces deployment ships these files directly in datasets/;
    other layouts ship them in bundles_caches/ and we copy them across.

    Defensive against pre-existing broken symlinks (older bundles symlinked to
    an absolute developer path that doesn't exist on every host).
    """
    import shutil
    for name in ("molgpka_debias_models.joblib", "molgpka_preds.npy"):
        src = CACHE_DIR / name
        dst = DATA_DIR / name
        # A broken symlink reports is_symlink()==True but exists()==False — bin it.
        if dst.is_symlink() and not dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        if dst.exists():
            continue  # already staged correctly
        if not src.exists():
            continue  # nothing to copy from; loader will error later if needed
        try:
            shutil.copy2(src, dst)
        except OSError:
            try:
                dst.symlink_to(src)
            except OSError:
                pass


def _load_bundle():
    """Load (and memoize) the tandem bundle.

    Handles two real-world failure modes seen on first install:
      1. xgboost version-skew warnings when unpickling — harmless, warning swallowed.
      2. v14 bundle fails to unpickle entirely (e.g., sklearn ABI break) — fall
         back to the v13 production bundle and surface a BUNDLE_FALLBACK_V13 flag.
    """
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    _stage_pka_artifacts()
    _prev_cwd = os.getcwd()
    os.chdir(CACHE_DIR)  # the bundle loader uses HERE-relative paths via iajd_tandem_final
    try:
        from iajd_tandem_final import load_tandem_bundle

        # Both bundle paths must be absolute to dodge the loader's HERE-relative dance.
        _BUNDLE = load_tandem_bundle(
            pka_xlsx=str(DATA_DIR / "IAJD_pKa_v21_final.xlsx"),
            bioact_bundle=str(CACHE_DIR / "bioact_v14_bundle.pkl"),
            verbose=False,
        )
        _BUNDLE._fallback_flag = None
    except Exception as exc:  # noqa: BLE001
        fallback = HERE / "IAJD_master" / "optional" / "iajd_bioact_v13_production_bundle.pkl"
        if not fallback.exists():
            raise RuntimeError(
                f"v14 bundle failed to load ({type(exc).__name__}: {exc}) and v13 "
                "fallback is missing."
            ) from exc
        from iajd_tandem_final import load_tandem_bundle  # type: ignore

        _BUNDLE = load_tandem_bundle(
            pka_xlsx=str(DATA_DIR / "IAJD_pKa_v21_final.xlsx"),
            bioact_bundle=str(fallback),
            verbose=False,
        )
        _BUNDLE._fallback_flag = f"BUNDLE_FALLBACK_V13: {type(exc).__name__}: {exc}"
    finally:
        os.chdir(_prev_cwd)

    # Rescue corrupt-pickled XGBoost direct model (version skew between the
    # env that built the bundle and the env loading it).
    rescued = _rescue_direct_model(_BUNDLE.bioact_bundle)
    if rescued:
        existing = getattr(_BUNDLE, "_fallback_flag", None)
        msg = "DIRECT_MODEL_REFIT: pickled XGBoost predicted nonsense; refit on stored X_train/y_train"
        _BUNDLE._fallback_flag = f"{existing}; {msg}" if existing else msg
    return _BUNDLE


def _get_v15() -> V15Bundle:
    """Build (once) the v15 predictor on top of the tandem bundle."""
    global _V15
    if _V15 is None:
        _V15 = load_v15(_load_bundle())
        _warm_agile_cache()
    return _V15


def _half60(half90: float) -> float:
    return round(float(half90) * _PI60_FROM_PI90, 3)


def _ci_from_pi90(point: float, pi90) -> Dict[str, Any]:
    lo, hi = float(pi90[0]), float(pi90[1])
    half90 = (hi - lo) / 2.0
    sigma = half90 / 1.6449
    h60 = _half60(half90)
    return {
        "point": round(float(point), 3),
        "sigma": round(float(sigma), 3),
        "ci_60": [round(point - h60, 3), round(point + h60, 3)],
        "ci_90": [round(lo, 3), round(hi, 3)],
    }


def predict(
    smiles: str,
    family: Optional[str] = None,
    neighbors: int = 5,
    measured_pka: Optional[float] = None,
    measured_pka_sd: Optional[float] = None,
    mol_2d: Optional["Chem.Mol"] = None,  # type: ignore[name-defined]
    structural_refine: bool = True,
    structural_beta: float = 0.25,
    input_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the v15 tandem (count-Tanimoto + analog-delta + global α) and emit
    a result block with 60% / 90% CIs, neighbors per stage, and organ partition.

    When `mol_2d` is provided (e.g. parsed from a ChemDraw upload), its 2D
    coordinates feed the structural feature extractor; otherwise RDKit-generated
    canonical 2D coords are used. `structural_refine=True` (default) recomputes
    the analog-delta path with similarity weights blended with structural
    cosine similarity, producing a refined point estimate that better respects
    positional/topological cues the count fingerprint can't see.
    """
    if family is not None and family not in ALLOWED_FAMILIES:
        raise ValueError(
            f"family={family!r} is not one of {sorted(ALLOWED_FAMILIES)}"
        )

    bundle = _load_bundle()
    v15 = _get_v15()

    summary: Dict[str, Any] = {
        "smiles": smiles,
        "input_label": input_label,
        "version": "v15 + struct-refine v1 + stacker v1 + family-detect v1",
        "warnings": [],
    }
    if getattr(bundle, "_fallback_flag", None):
        summary["warnings"].append(bundle._fallback_flag)

    # Resolve family BEFORE the v15 calls so both pKa and bioact see a single,
    # consistent family assignment.  The detector collapses the three bioact-
    # only sub-arch labels (HTM/TT/G1-Janus) to PE-Gallic per project policy;
    # the original label, if any, is kept in `family_resolution.subarch_label`
    # for downstream display.
    from rdkit import Chem as _Chem
    _mol_for_family = mol_2d if mol_2d is not None else _Chem.MolFromSmiles(smiles)
    family_resolution = resolve_family(_mol_for_family, user_hint=family)
    resolved_family = family_resolution.get("family_assigned")
    summary["family_resolution"] = family_resolution
    if resolved_family is None and "error" in family_resolution:
        summary["error"] = family_resolution["error"]
        return summary
    # `family` is the variable v15 calls expect; use the resolved value.
    family = resolved_family

    # Pre-compute the structural feature vector for the query. If the caller
    # passed a Mol with 2D coords (typically from a ChemDraw file), use that;
    # otherwise fall back to canonical 2D from the SMILES.
    query_feat = None
    query_struct_source = None
    if structural_refine:
        if mol_2d is not None and mol_2d.GetNumConformers() > 0:
            query_feat = _struct_feat_for_mol(mol_2d)
            query_struct_source = "user_chemdraw_2d"
        else:
            query_feat = _struct_feat_for_smiles(smiles)
            query_struct_source = "rdkit_2d_from_smiles"

    # ---- Stage 1: pKa via v15 ----
    if measured_pka is not None:
        pka_point = float(measured_pka)
        pka_sd = float(measured_pka_sd) if measured_pka_sd is not None else 0.05
        half90 = 1.6449 * pka_sd
        h60 = 0.8416 * pka_sd
        pka_result = {
            "point": round(pka_point, 3),
            "sigma": round(pka_sd, 3),
            "ci_60": [round(pka_point - h60, 3), round(pka_point + h60, 3)],
            "ci_90": [round(pka_point - half90, 3), round(pka_point + half90, 3)],
            "source": "measured",
            "tier": "MEASURED",
            "max_tanimoto": 1.0,
        }
    else:
        pka_result = predict_pka_v15(smiles, v15, family_hint=family)

    if "error" in pka_result:
        summary["error"] = pka_result["error"]
        summary["smiles"] = smiles
        return summary

    summary["canonical_smiles"] = pka_result.get("canonical_smiles", smiles)
    summary["family_used"] = pka_result.get("family_assigned") or family

    # Repack pKa for the standard output shape used by the CLI/server
    pka_block = {
        "point": pka_result["point"],
        "sigma": pka_result["sigma"],
        "ci_60": pka_result["ci_60"],
        "ci_90": pka_result["ci_90"],
        "source": pka_result.get("source"),
        "tier": pka_result.get("tier"),
        "max_tanimoto_to_training": pka_result.get("max_tanimoto"),
        "ood_flag": pka_result.get("max_tanimoto", 0) < 0.5,
        "molgpka_debiased": pka_result.get("debiased_molgpka_feature"),
        "direct_xgb_pred": pka_result.get("direct_pred"),
        "analog_delta_pred": pka_result.get("analog_pred"),
        "n_neighbors_used": pka_result.get("n_neighbors_used"),
        "iajd_id_if_measured": pka_result.get("iajd_id"),
        "warnings": pka_result.get("warnings", []),
    }
    # Structural refinement on the pKa stage (post-hoc reweighting of the
    # analog-delta neighbors with 2D-positional similarity).
    if structural_refine and query_feat is not None:
        ref = _apply_structural_refinement(
            stage_block={
                "point": pka_block["point"],
                "sigma": pka_block["sigma"],
                "pred_direct": pka_block["direct_xgb_pred"],
                "pred_analog": pka_block["analog_delta_pred"],
                "alpha_used": pka_result.get("alpha_used"),
            },
            analog_neighbors=pka_result.get("analog_neighbors", []) or [],
            query_feat=query_feat,
            beta=structural_beta,
        )
        if ref.get("applied"):
            pka_block["point_v15_original"] = pka_block["point"]
            pka_block["ci_60_v15_original"] = pka_block["ci_60"]
            pka_block["ci_90_v15_original"] = pka_block["ci_90"]
            pka_block["point"] = ref["point"]
            if ref["ci_60"] is not None: pka_block["ci_60"] = ref["ci_60"]
            if ref["ci_90"] is not None: pka_block["ci_90"] = ref["ci_90"]
        pka_block["structural_refinement"] = ref
        pka_block["query_struct_source"] = query_struct_source

    summary["pka"] = pka_block
    summary["warnings"].extend(pka_block.get("warnings", []) or [])

    # ---- Stage 2: bioactivity via v15 (uses the v15 pKa as injected feature) ----
    # Propagate the structurally-refined pKa into the bioactivity feature
    # vector so the refinement compounds through the tandem.
    injected_pka = float(pka_block["point"])
    bio_result = predict_bioactivity_v15(
        smiles, v15, family_hint=family or pka_result.get("family_assigned"),
        injected_pka=injected_pka,
        injected_pka_sd=pka_result["sigma"],
    )

    if "error" in bio_result:
        summary["bioactivity"] = {"error": bio_result["error"]}
    else:
        bio_block = {
            "point": bio_result["point"],
            "sigma": bio_result["sigma"],
            "ci_60": bio_result["ci_60"],
            "ci_90": bio_result["ci_90"],
            "source": bio_result.get("source"),
            "tier": bio_result.get("tier"),
            "max_tanimoto": bio_result.get("max_tanimoto"),
            "alpha_used": bio_result.get("alpha_used"),
            "pred_direct": bio_result.get("direct_pred"),
            "pred_analog": bio_result.get("analog_pred"),
            "n_neighbors_used": bio_result.get("n_neighbors_used"),
            "block_B_active": True, "block_C_active": True,
            "block_B_real": bio_result.get("lion_real"),
            "lion_live_call": bio_result.get("lion_live_call"),
            "admet_live_call": bio_result.get("admet_live_call"),
            "iajd_id_if_measured": bio_result.get("iajd_id"),
            "per_organ_measured": bio_result.get("per_organ_measured"),
            "pka_used_in_features": injected_pka,
            "warnings": bio_result.get("warnings", []),
        }
        # Structural refinement on the analog leg (recomputes the
        # similarity-weighted neighbor consensus with structural blending).
        analog_for_stacker = bio_block["pred_analog"]
        if structural_refine and query_feat is not None:
            ref_b = _apply_structural_refinement(
                stage_block=bio_block,
                analog_neighbors=bio_result.get("analog_neighbors", []) or [],
                query_feat=query_feat,
                beta=structural_beta,
            )
            if ref_b.get("applied"):
                bio_block["point_v15_original"] = bio_block["point"]
                bio_block["ci_60_v15_original"] = bio_block["ci_60"]
                bio_block["ci_90_v15_original"] = bio_block["ci_90"]
                bio_block["point"] = ref_b["point"]
                if ref_b["ci_60"] is not None: bio_block["ci_60"] = ref_b["ci_60"]
                if ref_b["ci_90"] is not None: bio_block["ci_90"] = ref_b["ci_90"]
                analog_for_stacker = ref_b.get("analog_refined", analog_for_stacker)
            bio_block["structural_refinement"] = ref_b
            bio_block["query_struct_source"] = query_struct_source

        # Apply the bioactivity stacker (v1) on top: feeds the v14-feature
        # vector x_full through 3 standalone heads (direct, lion, admet) and
        # the analog leg (refined if structural_refine fired) into an XGBoost
        # stacker fit on LOO OOF predictions (5-fold CV MAE 0.3856 vs baseline
        # 0.4010 → −0.0154 improvement, see weight_eval_v2_results.json).
        x_full = bio_result.get("x_full")
        if x_full is not None:
            stack_out = _bioact_stacker_predict(
                x_full, analog_for_stacker,
                smiles=summary.get("canonical_smiles"),
                max_tanimoto=bio_block.get("max_tanimoto"))
        else:
            stack_out = None
        if stack_out is not None:
            bio_block["point_pre_stacker"] = bio_block["point"]
            bio_block["ci_60_pre_stacker"] = bio_block["ci_60"]
            bio_block["ci_90_pre_stacker"] = bio_block["ci_90"]
            new_point = float(stack_out["point"])
            bio_block["point"] = round(new_point, 3)
            sigma = bio_block.get("sigma")
            if sigma is not None:
                h60 = 0.8416 * float(sigma)
                h90 = 1.6449 * float(sigma)
                bio_block["ci_60"] = [round(new_point - h60, 3), round(new_point + h60, 3)]
                bio_block["ci_90"] = [round(new_point - h90, 3), round(new_point + h90, 3)]
            bio_block["stacker"] = {
                "applied": True, "version": "bioact_stacker_v1",
                "components": {
                    "direct_head_pred": round(stack_out["direct"], 3),
                    "analog_pred":      round(stack_out["analog"], 3),
                    "lion_head_pred":   round(stack_out["lion"], 3),
                    "admet_head_pred":  round(stack_out["admet"], 3),
                },
                "expected_loo_mae": 0.3934,   # true LOO of the stacker
                "expected_5fold_cv_mae": 0.3856,
                "baseline_loo_mae": 0.4010,
                "improvement": 0.0076,
            }
        else:
            bio_block["stacker"] = {"applied": False,
                                      "reason": ("no_x_full" if x_full is None
                                                 else "stacker_bundle_missing")}
        summary["bioactivity"] = bio_block
        summary["warnings"].extend(bio_block.get("warnings", []) or [])
        # Use v15's neighbor-derived organ partition (real similarities now)
        if bio_result.get("organ_delivery"):
            summary["organ_delivery"] = bio_result["organ_delivery"]
        elif bio_result.get("per_organ_measured"):
            # Training-row exact match — convert measured per-organ to the same shape
            pom = bio_result["per_organ_measured"]
            valid = {k: v for k, v in pom.items() if v is not None}
            if valid:
                linear = {k: 10 ** v for k, v in valid.items()}
                tot = sum(linear.values())
                summary["organ_delivery"] = {
                    "target_organ": max(valid.items(), key=lambda kv: kv[1])[0],
                    "target_log10_flux": round(max(valid.values()), 3),
                    "log10_flux_by_organ": {k: round(v, 3) for k, v in valid.items()},
                    "partition_pct": {k: round(100.0 * v / tot, 1) for k, v in linear.items()},
                    "n_neighbors_used": 1,
                    "method": "training-row exact match (measured per-organ values)",
                }

    # ---- Display neighbors: prefer v15's analog neighbors (count-Tanimoto);
    # fall back to the iajd_neighbors module when the exact-match shortcut fired
    # and didn't surface any (so the GUI still shows the closest training IAJDs).
    pka_neigh = (pka_result.get("analog_neighbors", []) or [])[:neighbors]
    bio_neigh = _dedupe_by_iajd((bio_result.get("analog_neighbors", []) or []))[:neighbors]
    for n in pka_neigh:
        n["pKa"] = n.pop("neighbor_y", None)
    for n in bio_neigh:
        n["log10_flux_total"] = n.pop("neighbor_y", None)

    if not pka_neigh or not bio_neigh:
        try:
            from iajd_v15 import count_fp, bulk_tanimoto
            mol_q = __import__("rdkit").Chem.MolFromSmiles(summary["canonical_smiles"])
            fp_q = count_fp(mol_q)
            if not pka_neigh:
                sims = bulk_tanimoto(fp_q, v15.pka.count_fps)
                order = sims.argsort()[::-1][:neighbors]
                pka_neigh = [{
                    "iajd_id": v15.pka.ids[int(j)],
                    "smiles":  v15.pka.smiles[int(j)],
                    "family":  v15.pka.families[int(j)],
                    "tanimoto": round(float(sims[int(j)]), 4),
                    "pKa": float(v15.pka.y[int(j)]),
                } for j in order]
            if not bio_neigh:
                sims = bulk_tanimoto(fp_q, v15.bioact.count_fps)
                order = sims.argsort()[::-1][:max(neighbors * 2, 10)]
                raw = [{
                    "iajd_id": v15.bioact.ids[int(j)],
                    "smiles":  v15.bioact.smiles[int(j)],
                    "family":  v15.bioact.families[int(j)],
                    "tanimoto": round(float(sims[int(j)]), 4),
                    "log10_flux_total": float(v15.bioact.y[int(j)]),
                } for j in order]
                bio_neigh = _dedupe_by_iajd(raw)[:neighbors]
        except Exception:  # noqa: BLE001
            pass

    summary["neighbors"] = {"pka": pka_neigh, "bioact": bio_neigh}
    return summary


def predict_batch(
    content: bytes | str,
    filename: Optional[str] = None,
    family: Optional[str] = None,
    neighbors: int = 5,
    structural_refine: bool = True,
    structural_beta: float = 0.25,
) -> Dict[str, Any]:
    """Parse a SMILES string, multi-line SMILES, or uploaded ChemDraw/.cdxml/
    .mol/.sdf file and run the tandem predictor over every molecule found.

    Returns:
        {
            "n_inputs": int,
            "source": "smiles"|"chemdraw"|"molfile"|"sdf",
            "results": [ <one result dict per molecule, same shape as predict()> ],
            "errors": [ {label, error} ... ],
        }
    """
    from iajd_structural import parse_input
    parsed = parse_input(content, filename=filename)
    out_results: list = []
    errors: list = []
    if not parsed:
        return {
            "n_inputs": 0, "source": None,
            "results": [], "errors": [{"label": None, "error": "no_parseable_structures"}],
        }
    src = parsed[0].source
    for p in parsed:
        # Per-row hint precedence: file-level `family=` token  >  global `family`
        # >  auto-detect (resolved inside predict()).
        row_family = p.family_hint or family
        try:
            r = predict(
                smiles=p.smiles,
                family=row_family,
                neighbors=neighbors,
                mol_2d=p.mol if p.had_coords else None,
                structural_refine=structural_refine,
                structural_beta=structural_beta,
                input_label=p.label,
            )
            r["coord_source"] = "chemdraw_file" if p.had_coords else "rdkit_canonical"
            r["family_hint_source"] = (
                "per_row" if p.family_hint else
                ("user_global" if family else "auto")
            )
            out_results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append({"label": p.label, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "n_inputs": len(parsed),
        "source": src,
        "results": out_results,
        "errors": errors,
    }


def _dedupe_by_iajd(neighbors: list) -> list:
    """Keep only the highest-similarity row per IAJD id (the bioact xlsx has
    repeated rows for the same IAJD at different dose/timepoint conditions)."""
    seen = {}
    for n in neighbors:
        key = str(n.get("iajd_id") or n.get("smiles"))
        if key not in seen or n.get("tanimoto", 0) > seen[key].get("tanimoto", 0):
            seen[key] = n
    return sorted(seen.values(), key=lambda r: -r.get("tanimoto", 0))


_ORGAN_COLS = [
    ("lung",   "log10_flux_lung"),
    ("liver",  "log10_flux_liver"),
    ("spleen", "log10_flux_spleen"),
    ("LN",     "log10_flux_LN"),
    ("heart",  "log10_flux_heart"),
]


def _organ_partition(bioact_neighbors: list) -> Dict[str, Any]:
    """Predict per-organ delivery from the top Tanimoto neighbors.

    For each of the 5 organs, take a Tanimoto-weighted average of the
    neighbors' log10 per-organ flux (sim^4 weighting, same as the bundle's
    analog-delta path). The predicted target organ is the one with the
    largest weighted log10 flux. Also reports the relative organ partition
    in linear flux space so users can read it as "X% to lung, Y% to liver".
    """
    if not bioact_neighbors:
        return {"target_organ": None, "reason": "no_neighbors"}

    import math
    # Restrict to top-2 closest neighbors that actually have per-organ data.
    # This is a nearest-neighbor similarity lookup, not an ML prediction.
    usable = [n for n in bioact_neighbors[:2]
              if any(n.get(col) is not None for _, col in _ORGAN_COLS)]
    if not usable:
        return {"target_organ": None, "reason": "no_per_organ_data_in_neighbors"}

    weights = []
    per_organ = {label: [] for label, _ in _ORGAN_COLS}
    for n in usable:
        w = max(float(n.get("tanimoto", 0.0)), 0.0) ** 4
        for label, col in _ORGAN_COLS:
            v = n.get(col)
            if v is None:
                continue
            per_organ[label].append((w, float(v)))
        weights.append(w)

    if sum(weights) <= 0:
        return {"target_organ": None, "reason": "zero_total_weight"}

    log_flux_by_organ: Dict[str, float] = {}
    for label, pairs in per_organ.items():
        if not pairs:
            continue
        wsum = sum(w for w, _ in pairs)
        if wsum <= 0:
            continue
        log_flux_by_organ[label] = sum(w * v for w, v in pairs) / wsum

    if not log_flux_by_organ:
        return {"target_organ": None, "reason": "no_organ_values_after_weighting"}

    # Convert to linear flux for partition percentages
    linear = {k: 10 ** v for k, v in log_flux_by_organ.items()}
    total = sum(linear.values())
    partition = {k: round(100.0 * v / total, 1) for k, v in linear.items()}
    target = max(log_flux_by_organ.items(), key=lambda kv: kv[1])

    return {
        "target_organ": target[0],
        "target_log10_flux": round(target[1], 3),
        "log10_flux_by_organ": {k: round(v, 3) for k, v in log_flux_by_organ.items()},
        "partition_pct": partition,
        "n_neighbors_used": len(usable),
        "method": "similarity-score (Tanimoto-weighted) over top-2 nearest training IAJDs — NOT an ML prediction",
    }


def _format_text(r: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("IAJD Tandem Prediction")
    lines.append("=" * 78)
    lines.append(f"SMILES (canonical) : {r.get('canonical_smiles')}")
    lines.append(f"Family used        : {r.get('family_used')}")
    src = r.get("family_sources", {}) or {}
    lines.append(
        f"  (pka stage detected: {src.get('pka')} | bioact stage detected: {src.get('bioact')} "
        f"| user hint: {src.get('user_hint')})"
    )
    lines.append("")

    if "error" in r:
        lines.append(f"ERROR: {r['error']}")
        return "\n".join(lines)

    pka = r.get("pka", {}) or {}
    if "point" in pka:
        lines.append(
            f"pKa                : {pka['point']:.3f}   tier={pka.get('tier')}  "
            f"source={pka.get('source')}"
        )
        lines.append(
            f"  60% CI           : [{pka['ci_60'][0]:.3f}, {pka['ci_60'][1]:.3f}]   "
            f"(sigma={pka['sigma']:.3f})"
        )
        lines.append(
            f"  90% CI           : [{pka['ci_90'][0]:.3f}, {pka['ci_90'][1]:.3f}]"
        )
        if pka.get("max_tanimoto_to_training") is not None:
            lines.append(
                f"  max Tanimoto      : {pka['max_tanimoto_to_training']:.3f}   "
                f"OOD={pka.get('ood_flag')}"
            )
    else:
        lines.append("pKa                : (unavailable)")

    bio = r.get("bioactivity", {}) or {}
    lines.append("")
    if "point" in bio:
        lines.append(
            f"log10 flux total   : {bio['point']:.3f}   tier={bio.get('tier')}   "
            f"alpha={bio.get('alpha_used')}"
        )
        lines.append(
            f"  60% CI           : [{bio['ci_60'][0]:.3f}, {bio['ci_60'][1]:.3f}]   "
            f"(sigma={bio['sigma']:.3f})"
        )
        lines.append(
            f"  90% CI           : [{bio['ci_90'][0]:.3f}, {bio['ci_90'][1]:.3f}]"
        )
        lines.append(
            f"  max Tanimoto      : {bio.get('max_tanimoto'):.3f}   "
            f"LION_real={bio.get('block_B_real')}   "
            f"B={bio.get('block_B_active')}  C={bio.get('block_C_active')}"
        )
    elif "error" in bio:
        lines.append(f"bioactivity ERROR  : {bio['error']}")

    organ = r.get("organ_delivery", {}) or {}
    if organ.get("target_organ"):
        lines.append("")
        parts = organ.get("partition_pct", {})
        partition_str = ", ".join(f"{k}={parts[k]}%" for k in sorted(parts, key=lambda x: -parts[x]))
        lines.append(
            f"Predicted target organ : {organ['target_organ']}  "
            f"(log10_flux={organ.get('target_log10_flux')}, "
            f"n_neighbors={organ.get('n_neighbors_used')})"
        )
        lines.append(f"  Partition           : {partition_str}")

    # Neighbors
    nb = r.get("neighbors", {}) or {}
    lines.append("")
    lines.append("Nearest pKa training IAJDs (Tanimoto, Morgan-2/2048):")
    if isinstance(nb.get("pka"), list):
        for n in nb["pka"]:
            lines.append(
                f"  #{n['iajd_id']!s:<10}  tan={n['tanimoto']:.3f}  fam={n['family']:<22}"
                f"  pKa={n.get('pKa')}"
            )
    else:
        lines.append(f"  (lookup failed: {nb.get('error')})")

    lines.append("")
    lines.append("Nearest bioactivity training IAJDs:")
    if isinstance(nb.get("bioact"), list):
        for n in nb["bioact"]:
            lines.append(
                f"  #{n['iajd_id']!s:<10}  tan={n['tanimoto']:.3f}  fam={n['family']:<22}"
                f"  log10_flux={n.get('log10_flux_total')}"
            )
    else:
        lines.append(f"  (lookup failed: {nb.get('error')})")

    warns = r.get("warnings", []) or []
    if warns:
        lines.append("")
        lines.append("Warnings:")
        for w in warns:
            lines.append(f"  - {w}")

    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="IAJD tandem prediction (pKa + bioactivity).")
    ap.add_argument("--smiles", required=True, help="Query SMILES.")
    ap.add_argument(
        "--family",
        default=None,
        choices=sorted(ALLOWED_FAMILIES) + [None],  # type: ignore
        help="Optional family hint. Required for non-PE-Tris auto-detect.",
    )
    ap.add_argument("--neighbors", type=int, default=5, help="Top-k neighbors per training set.")
    ap.add_argument("--json", dest="json_out", default=None, help="Write full result as JSON to this path.")
    ap.add_argument("--quiet", action="store_true", help="Suppress pretty stdout (JSON only).")
    ap.add_argument("--measured-pka", type=float, default=None, help="Bypass pKa prediction with this measured value.")
    ap.add_argument("--measured-pka-sd", type=float, default=None, help="SD for the measured pKa (default 0.05).")
    args = ap.parse_args()

    result = predict(
        smiles=args.smiles,
        family=args.family,
        neighbors=args.neighbors,
        measured_pka=args.measured_pka,
        measured_pka_sd=args.measured_pka_sd,
    )

    if not args.quiet:
        print(_format_text(result))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        if not args.quiet:
            print(f"\n(wrote {args.json_out})")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
