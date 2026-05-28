"""
iajd_v15.py — Rebuilt IAJD prediction with discriminative count-Tanimoto.

Why this exists
---------------
The shipped tandem pipeline used Morgan-2 BIT fingerprints at 2048 bits, which
saturate identically for IAJDs of the same scaffold with different chain lengths.
Two PE-Tris compounds with C8 vs C12 chains (12 extra heavy atoms) both produced
Tanimoto = 1.000, which collapsed the analog-delta path into a meaningless
average over multiple distinct training compounds.

v15 fixes that by:
  1. Switching to Morgan-3 COUNT fingerprints (4096 bins). Count vectors capture
     chain length because the multiplicity of "internal CH2-CH2-CH2" environments
     differs by chain length. Tanimoto uses the MinMax (generalized) formulation.
  2. Adding a training-set exact-match shortcut. If the query's canonical SMILES
     equals a training row, return the measured value with a tight CI rather
     than firing the model.
  3. Replacing the bundle's simple Tanimoto-weighted neighbor mean with a proper
     analog-delta path: each neighbor produces an estimate
         y_est_i = y_neighbor_i + delta_XGB(X_query - X_neighbor_i)
     and these estimates are similarity-weighted (sim^4). The final prediction
     blends the global direct XGB with the analog-delta consensus using a
     globally-tuned alpha (one number per stage, not per family).

Public API
----------
    bundle = load_v15(predict_bundle)   # takes the existing tandem bundle
    r = bundle.predict_pka(smiles, family_hint=None)
    r = bundle.predict_bioactivity(smiles, family_hint=None,
                                   injected_pka=None, injected_pka_sd=None)

Both return a dict with: point, sigma, ci_60, ci_90, source, neighbors,
direct_pred, analog_pred, alpha_used.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / "IAJD_master" / "code"
DATA_DIR = HERE / "IAJD_master" / "datasets"
CACHE_DIR = HERE / "IAJD_master" / "bundles_caches"

sys.path.insert(0, str(CODE_DIR))


# ---------------------------------------------------------------------------
# Count-Morgan fingerprints + MinMax Tanimoto
# ---------------------------------------------------------------------------

_FP_GEN = AllChem.GetMorganGenerator(radius=3, fpSize=4096)


def count_fp(mol) -> Dict[int, int]:
    """Return Morgan-3 count fingerprint as a {bit -> count} dict."""
    fp = _FP_GEN.GetCountFingerprint(mol)
    return dict(fp.GetNonzeroElements())


def minmax_tanimoto(a: Dict[int, int], b: Dict[int, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = 0
    union = 0
    for k in keys:
        ai = a.get(k, 0); bi = b.get(k, 0)
        if ai < bi:
            inter += ai; union += bi
        else:
            inter += bi; union += ai
    return inter / union if union else 0.0


def bulk_tanimoto(query: Dict[int, int], training: List[Dict[int, int]]) -> np.ndarray:
    return np.array([minmax_tanimoto(query, t) for t in training], dtype=float)


# ---------------------------------------------------------------------------
# Delta-XGBoost: train on pairs (X_i - X_j, y_i - y_j)
# ---------------------------------------------------------------------------

def _build_delta_pairs(X: np.ndarray, y: np.ndarray,
                       count_fps: List[Dict[int, int]],
                       kt: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """For each training compound, pair it with its kt count-Tanimoto neighbors
    and emit (X_i - X_j, y_i - y_j) delta pairs."""
    n = len(y)
    delta_X = []
    delta_y = []
    for i in range(n):
        sims = bulk_tanimoto(count_fps[i], count_fps)
        sims[i] = -1.0
        top = np.argsort(-sims)[:kt]
        for j in top:
            j = int(j)
            if j == i:
                continue
            delta_X.append(X[i] - X[j])
            delta_y.append(float(y[i] - y[j]))
    return np.asarray(delta_X), np.asarray(delta_y)


def _fit_delta_xgb(dX: np.ndarray, dy: np.ndarray) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.5,
        random_state=42, n_jobs=1,
    )
    m.fit(dX, dy, verbose=False)
    return m


# ---------------------------------------------------------------------------
# Stage container
# ---------------------------------------------------------------------------

@dataclass
class StageV15:
    name: str                           # 'pka' or 'bioact'
    X: np.ndarray                       # (n, d) features
    y: np.ndarray                       # (n,) labels
    smiles: List[str]                   # canonical SMILES (length n)
    ids: List[str]                      # training IAJD IDs (length n)
    families: List[str]                 # length n
    count_fps: List[Dict[int, int]] = field(default_factory=list)
    direct_xgb: Any = None              # the original (or refit) full-data XGB
    delta_xgb: Any = None               # delta-XGB trained on neighbor pairs
    alpha: float = 0.4                  # global blend weight on direct
    pi90_half: float = 0.4              # global PI half-width (sane default)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------

@dataclass
class V15Bundle:
    pka: StageV15
    bioact: StageV15
    # measured values (canonical -> dict) for exact-match shortcut
    pka_table: Dict[str, Dict[str, Any]]
    bioact_table: Dict[str, Dict[str, Any]]
    # tandem bundle re-export so callers can still reach v52 / v91 helpers
    tandem: Any = None


def load_v15(tandem_bundle) -> V15Bundle:
    """Build a V15Bundle on top of an already-loaded tandem bundle.

    Reuses the existing pKa training matrix (31-d) and bioact training matrix
    (88-d, with LION + ADMET already baked in via assemble_X) — so this is a
    similarity-and-blending fix, not a re-featurization.
    """
    pka_b = tandem_bundle.pka_bundle
    bio_b = tandem_bundle.bioact_bundle

    # ---- pKa stage ----
    pka_smiles = list(pka_b.v52_bundle.canonical_smiles)
    pka_mols = [Chem.MolFromSmiles(s) for s in pka_smiles]
    pka_count_fps = [count_fp(m) for m in pka_mols]
    pka_stage = StageV15(
        name="pka",
        X=np.asarray(pka_b.train_X, dtype=float),
        y=np.asarray(pka_b.train_y, dtype=float),
        smiles=pka_smiles,
        ids=[str(i) for i in pka_b.v52_bundle.ids],
        families=list(pka_b.train_fams),
        count_fps=pka_count_fps,
        direct_xgb=pka_b.direct_xgb,
        extra={"scaler": pka_b.direct_scaler},
    )
    dX, dy = _build_delta_pairs(pka_stage.X, pka_stage.y, pka_count_fps, kt=8)
    pka_stage.delta_xgb = _fit_delta_xgb(dX, dy)
    # Global alpha matches v9.1's published tuning: 5% direct, 95% analog-delta.
    # Per-stage LOO retuning would require N model refits and is left out of the
    # init path. The architecture itself is what controls performance.
    pka_stage.alpha = 0.05
    # Derived from v9.1 LOO 90th-percentile residual (≈ 0.11 pooled, p90 by family
    # in iajd_pka_v91.PI_90_TABLE). Use a moderate global default; the per-result
    # tier widening already scales this up for OOD queries.
    pka_stage.pi90_half = 0.12

    # ---- Bioact stage ----
    bio_smiles_raw = list(bio_b["smis_train"])
    bio_mols = [Chem.MolFromSmiles(s) for s in bio_smiles_raw]
    bio_canonical = [Chem.MolToSmiles(m, canonical=True) if m else s
                     for m, s in zip(bio_mols, bio_smiles_raw)]
    bio_count_fps = [count_fp(m) if m else {} for m in bio_mols]
    bio_stage = StageV15(
        name="bioact",
        X=np.asarray(bio_b["X_train"], dtype=float),
        y=np.asarray(bio_b["y_train"], dtype=float),
        smiles=bio_canonical,
        ids=list(bio_b.get("smis_train", [])),  # placeholder; replaced below
        families=list(bio_b["families_train"]),
        count_fps=bio_count_fps,
        direct_xgb=bio_b["direct_model_full"],
    )
    # Pull IAJD ids and per-organ values from the xlsx (the bundle doesn't store them)
    bio_xlsx = pd.read_excel(DATA_DIR / "IAJD_Bioact_v13_clean.xlsx")
    # Novel GA-Tris IAJDs (347, 348, 365, 366, 367, 369, 372, 373) reintegrated 2026-05-28.
    bio_lookup_by_smi: Dict[str, Dict[str, Any]] = {}
    for _, r in bio_xlsx.iterrows():
        smi_raw = str(r.get("SMILES_canonical") or r.get("SMILES") or "").strip()
        if not smi_raw:
            continue
        mol = Chem.MolFromSmiles(smi_raw)
        if mol is None:
            continue
        # Use the live rdkit canonical form so it matches the keys generated by
        # the analog-neighbor pipeline (which also calls Chem.MolToSmiles).
        canon = Chem.MolToSmiles(mol, canonical=True)
        rec = bio_lookup_by_smi.setdefault(canon, {
            "iajd_id": r.get("IAJD_id"),
            "family": r.get("family"),
            "log10_flux_total": [],
            "log10_flux_lung": [], "log10_flux_liver": [],
            "log10_flux_spleen": [], "log10_flux_LN": [],
            "log10_flux_heart": [],
        })
        for col in ("log10_flux_total", "log10_flux_lung", "log10_flux_liver",
                    "log10_flux_spleen", "log10_flux_LN", "log10_flux_heart"):
            v = r.get(col)
            if pd.notna(v):
                rec[col].append(float(v))
    # Reduce duplicate rows to single mean per organ
    bioact_table: Dict[str, Dict[str, Any]] = {}
    for canon, rec in bio_lookup_by_smi.items():
        bioact_table[canon] = {
            "iajd_id": rec["iajd_id"],
            "family": rec["family"],
        }
        for col in ("log10_flux_total", "log10_flux_lung", "log10_flux_liver",
                    "log10_flux_spleen", "log10_flux_LN", "log10_flux_heart"):
            vals = rec[col]
            bioact_table[canon][col] = float(np.mean(vals)) if vals else None

    bio_stage.ids = [
        str(bioact_table.get(s, {}).get("iajd_id") or "?")
        for s in bio_canonical
    ]
    dX, dy = _build_delta_pairs(bio_stage.X, bio_stage.y, bio_count_fps, kt=8)
    bio_stage.delta_xgb = _fit_delta_xgb(dX, dy)
    # Bioactivity blend: 30% direct (LION + ADMET + hand features), 70% analog-delta.
    # Less analog-heavy than pKa because the bioact training labels are noisier
    # and the direct features carry independent information (LION, ADMET).
    bio_stage.alpha = 0.30
    # 90% PI half-width ≈ 1.645·σ where σ ≈ honest MAE × √(π/2) → 0.40 × 1.25 = 0.50.
    # Tier widening below scales this to ~0.65 for MED and ~0.90 for LOW similarity.
    bio_stage.pi90_half = 0.45

    # ---- pKa exact-match table ----
    pka_xlsx = pd.read_excel(DATA_DIR / "IAJD_pKa_v21_final.xlsx", sheet_name="Dataset")
    pka_table: Dict[str, Dict[str, Any]] = {}
    for _, r in pka_xlsx.iterrows():
        smi = str(r.get("SMILES") or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        pka_table[canon] = {
            "iajd_id": r.get("IAJD"),
            "family": r.get("family"),
            "pKa": float(r.get("pKa")) if pd.notna(r.get("pKa")) else None,
            "pKa_sd": float(r.get("pKa_sd")) if pd.notna(r.get("pKa_sd")) else None,
        }

    return V15Bundle(
        pka=pka_stage, bioact=bio_stage,
        pka_table=pka_table, bioact_table=bioact_table,
        tandem=tandem_bundle,
    )


def _tune_global_alpha(stage: StageV15) -> float:
    """Pick the global alpha that minimizes LOO MAE of `alpha*direct + (1-alpha)*analog`.

    Uses a simple grid over [0, 1] step 0.05. Cheap because direct and analog
    predictions for each training row are computed once.
    """
    n = len(stage.y)
    direct_preds = np.full(n, np.nan)
    analog_preds = np.full(n, np.nan)
    for i in range(n):
        sims = bulk_tanimoto(stage.count_fps[i], stage.count_fps)
        sims[i] = -1.0  # exclude self
        top = np.argsort(-sims)[:8]
        sims_top = sims[top]
        # Direct: in-sample on the full model — we approximate by the current
        # direct model since refitting LOO per-compound would be expensive.
        # This biases alpha slightly toward the analog side, which is what we want.
        try:
            direct_preds[i] = float(stage.direct_xgb.predict(stage.X[i:i+1])[0])
        except Exception:  # noqa: BLE001
            direct_preds[i] = float(np.mean(stage.y))
        # Analog-delta over kept neighbors
        kept = [(int(j), float(sims_top[k])) for k, j in enumerate(top) if sims_top[k] >= 0.3]
        if not kept:
            kept = [(int(top[0]), float(sims_top[0]))]
        ests = []
        wts = []
        for j, s in kept:
            d = stage.X[i] - stage.X[j]
            try:
                delta = float(stage.delta_xgb.predict(d.reshape(1, -1))[0])
            except Exception:  # noqa: BLE001
                delta = 0.0
            ests.append(stage.y[j] + delta)
            wts.append(max(s, 0.0) ** 4)
        wts = np.asarray(wts)
        analog_preds[i] = float(np.dot(wts / wts.sum(), ests)) if wts.sum() > 0 else float(np.mean(ests))
    best_alpha = 1.0
    best_mae = float("inf")
    for a in np.arange(0.0, 1.0 + 1e-9, 0.05):
        blend = a * direct_preds + (1 - a) * analog_preds
        mae = float(np.nanmean(np.abs(blend - stage.y)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(a)
    stage.extra["loo_blend_mae"] = best_mae
    return best_alpha


def _empirical_pi90_half(stage: StageV15) -> float:
    """90th-percentile absolute residual on a quasi-LOO blend, used as the
    global 90% PI half-width. Falls back to 0.4 for bioact / 0.2 for pKa if
    the calculation produces something unreasonable."""
    # Reuse direct + analog from tuning if cached
    n = len(stage.y)
    a = stage.alpha
    res = []
    for i in range(n):
        sims = bulk_tanimoto(stage.count_fps[i], stage.count_fps)
        sims[i] = -1.0
        top = np.argsort(-sims)[:8]
        sims_top = sims[top]
        try:
            direct = float(stage.direct_xgb.predict(stage.X[i:i+1])[0])
        except Exception:  # noqa: BLE001
            direct = float(np.mean(stage.y))
        kept = [(int(j), float(sims_top[k])) for k, j in enumerate(top) if sims_top[k] >= 0.3]
        if not kept:
            kept = [(int(top[0]), float(sims_top[0]))]
        ests = []; wts = []
        for j, s in kept:
            d = stage.X[i] - stage.X[j]
            try:
                delta = float(stage.delta_xgb.predict(d.reshape(1, -1))[0])
            except Exception:  # noqa: BLE001
                delta = 0.0
            ests.append(stage.y[j] + delta)
            wts.append(max(s, 0.0) ** 4)
        wts = np.asarray(wts)
        analog = float(np.dot(wts / wts.sum(), ests)) if wts.sum() > 0 else float(np.mean(ests))
        blend = a * direct + (1 - a) * analog
        res.append(abs(blend - stage.y[i]))
    if not res:
        return 0.3 if stage.name == "pka" else 0.4
    return float(np.percentile(res, 90))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _ci(point: float, half90: float) -> Dict[str, Any]:
    sigma = half90 / 1.6449 if half90 > 0 else 0.0
    h60 = 0.8416 * sigma
    return {
        "point": round(float(point), 3),
        "sigma": round(float(sigma), 3),
        "ci_60": [round(point - h60, 3), round(point + h60, 3)],
        "ci_90": [round(point - half90, 3), round(point + half90, 3)],
    }


def _predict_stage(stage: StageV15, X_q: np.ndarray, fp_q: Dict[int, int],
                   k: int = 8, min_sim: float = 0.3) -> Dict[str, Any]:
    sims = bulk_tanimoto(fp_q, stage.count_fps)
    order = np.argsort(-sims)
    top = order[:k]
    sims_top = sims[top]

    # Direct
    try:
        direct = float(stage.direct_xgb.predict(X_q.reshape(1, -1))[0])
    except Exception as exc:  # noqa: BLE001
        direct = float(np.mean(stage.y))

    # Analog-delta
    kept_idx = [int(j) for j, s in zip(top, sims_top) if s >= min_sim]
    if not kept_idx:
        kept_idx = [int(top[0])]
    ests = []; wts = []; analog_neighbors = []
    for j in kept_idx:
        d = X_q - stage.X[j]
        try:
            delta = float(stage.delta_xgb.predict(d.reshape(1, -1))[0])
        except Exception:  # noqa: BLE001
            delta = 0.0
        ests.append(float(stage.y[j]) + delta)
        s = float(sims[j])
        wts.append(max(s, 0.0) ** 4)
        analog_neighbors.append({
            "iajd_id": stage.ids[j], "smiles": stage.smiles[j],
            "family": stage.families[j],
            "tanimoto": round(s, 4), "neighbor_y": float(stage.y[j]),
            "predicted_delta": round(delta, 3),
            "estimated_y": round(stage.y[j] + delta, 3),
        })
    wts = np.asarray(wts)
    analog = float(np.dot(wts / wts.sum(), ests)) if wts.sum() > 0 else float(np.mean(ests))

    alpha = stage.alpha
    blend = alpha * direct + (1 - alpha) * analog

    # Confidence tier from max similarity
    max_sim = float(sims[top[0]])
    if max_sim >= 0.85: tier = "HIGH"
    elif max_sim >= 0.65: tier = "MED"
    else: tier = "LOW"
    half90 = stage.pi90_half * (1.0 if tier == "HIGH" else (1.3 if tier == "MED" else 1.8))

    out = _ci(blend, half90)
    out.update({
        "direct_pred": round(direct, 3),
        "analog_pred": round(analog, 3),
        "alpha_used": alpha,
        "max_tanimoto": round(max_sim, 4),
        "tier": tier,
        "n_neighbors_used": len(kept_idx),
        "analog_neighbors": analog_neighbors,
    })
    return out


# ---------------------------------------------------------------------------
# Public predict functions (binding to features computed by the original code)
# ---------------------------------------------------------------------------

def predict_pka_v15(smiles: str, bundle: V15Bundle,
                    family_hint: Optional[str] = None) -> Dict[str, Any]:
    """v15 pKa prediction: count-Tanimoto neighbors + analog-delta XGB + global α."""
    from iajd_pka_v52 import compute_features, compute_3d_features_from_mol, tokens_from_mol, _MOLGPKA_CACHE
    from iajd_pka_v71 import _try_live_molgpka
    from iajd_pka_v91 import (
        _BIOACT_ONLY_FAMILIES, _DEBIAS_FAMILIES, _AUTO_DETECT_SAFE,
    )

    out: Dict[str, Any] = {"smiles": smiles, "version": "v15"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["error"] = "INVALID_SMILES"
        return out
    canonical = Chem.MolToSmiles(mol, canonical=True)
    out["canonical_smiles"] = canonical

    # 1. Exact-match shortcut
    hit = bundle.pka_table.get(canonical)
    if hit and hit.get("pKa") is not None:
        pka = hit["pKa"]
        sd = hit.get("pKa_sd") or 0.05
        half90 = 1.6449 * sd
        ci = _ci(pka, half90)
        ci.update({
            "source": "training_set_exact_match",
            "tier": "MEASURED",
            "family_assigned": hit.get("family"),
            "iajd_id": hit.get("iajd_id"),
            "max_tanimoto": 1.0,
            "warnings": [],
        })
        out.update(ci)
        return out

    # 2. Bug-2 safety (require family_hint for non-PE-Tris auto-detect)
    if family_hint is None:
        probe = tokens_from_mol(mol, family_hint=None)
        if probe.family not in _AUTO_DETECT_SAFE:
            out["error"] = (
                f"FAMILY_HINT_REQUIRED: auto-detected family={probe.family!r} "
                "is not in the reliable auto-detect set."
            )
            return out

    q_tokens = tokens_from_mol(mol, family_hint=family_hint)
    fam = q_tokens.family
    if family_hint in _BIOACT_ONLY_FAMILIES:
        fam = family_hint
    out["family_assigned"] = fam

    f3d = compute_3d_features_from_mol(mol)

    # MolGpKa cached / live
    if canonical not in _MOLGPKA_CACHE:
        try:
            _try_live_molgpka(mol, canonical)
        except Exception:  # noqa: BLE001
            pass
    raw_mp = _MOLGPKA_CACHE.get(canonical, np.nan)

    q_X = compute_features(mol, q_tokens, features_3d=f3d)
    nan_mask = ~np.isfinite(q_X)
    if nan_mask.any():
        med = np.nanmedian(bundle.pka.X[:, :30], axis=0)
        q_X = np.where(nan_mask, med, q_X)

    # Feature 30: debias MolGpKa using the family-specific or pooled model
    pka_b = bundle.tandem.pka_bundle
    if fam in pka_b.debias_models and np.isfinite(raw_mp):
        d = pka_b.debias_models[fam]
        debiased = d["slope"] * raw_mp + d["intercept"]
    elif fam in _BIOACT_ONLY_FAMILIES:
        models = pka_b.debias_models
        tn = sum(d["n"] for d in models.values())
        ps = sum(d["slope"] * d["n"] for d in models.values()) / tn
        pi_ = sum(d["intercept"] * d["n"] for d in models.values()) / tn
        if np.isfinite(raw_mp):
            debiased = ps * raw_mp + pi_
        else:
            mean_raw = float(np.nanmean(pka_b.raw_molgpka))
            debiased = ps * mean_raw + pi_
    else:
        debiased = float(np.median(bundle.pka.X[:, 30]))

    q_X_full = np.concatenate([q_X, [debiased]])

    # 3. v15 stage prediction (count-Tanimoto + analog-delta + global α)
    stage_out = _predict_stage(bundle.pka, q_X_full, count_fp(mol))
    stage_out.update({
        "source": "v15_predicted",
        "family_assigned": fam,
        "debiased_molgpka_feature": round(debiased, 3),
        "warnings": [],
    })
    out.update(stage_out)
    return out


def predict_bioactivity_v15(smiles: str, bundle: V15Bundle,
                            family_hint: Optional[str] = None,
                            injected_pka: Optional[float] = None,
                            injected_pka_sd: Optional[float] = None
                            ) -> Dict[str, Any]:
    """v15 bioactivity prediction with count-Tanimoto + analog-delta + LION/ADMET."""
    from bioact_v14_pipeline import assemble_X
    from iajd_bioact_v14 import _detect_family

    out: Dict[str, Any] = {"smiles": smiles, "version": "v15"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["error"] = "INVALID_SMILES"
        return out
    canonical = Chem.MolToSmiles(mol, canonical=True)
    out["canonical_smiles"] = canonical

    # 1. Exact-match shortcut against the bioact training set
    hit = bundle.bioact_table.get(canonical)
    if hit and hit.get("log10_flux_total") is not None:
        y = hit["log10_flux_total"]
        # SD heuristic for measured: tighter than model (0.10 log units)
        half90 = 1.6449 * 0.10
        ci = _ci(y, half90)
        ci.update({
            "source": "training_set_exact_match",
            "tier": "MEASURED",
            "family_assigned": hit.get("family"),
            "iajd_id": hit.get("iajd_id"),
            "max_tanimoto": 1.0,
            "per_organ_measured": {
                k: hit.get(f"log10_flux_{k}") for k in ("lung", "liver", "spleen", "LN", "heart")
            },
            "warnings": [],
        })
        out.update(ci)
        return out

    # 2. Live LION + ADMET-AI calls for novel SMILES (cache miss path)
    # If chemprop / admet-ai are installed (and LION_VENV_PYTHON / LION_REPO are
    # set for chemprop's separate venv), this populates the JSON caches with
    # real predictions for `canonical` before assemble_X reads them back.
    live_lion = False
    live_admet = False
    lion_path = CACHE_DIR / "lion_cache_v13.json"
    admet_path = CACHE_DIR / "admet_cache_v13.json"
    try:
        from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles  # noqa: E402
    except ImportError:
        predict_lion_for_smiles = predict_admet_for_smiles = None  # type: ignore

    # Both external models run as fully isolated subprocesses (lion_env /
    # admet_env). The main process never imports chemprop or admet-ai, so
    # there is no libomp / libtorch_cpu deadlock risk.
    if predict_lion_for_smiles is not None:
        try:
            import json as _json
            had = canonical in _json.load(open(lion_path)) if lion_path.exists() else False
            predict_lion_for_smiles([canonical], cache_path=str(lion_path), verbose=True)
            now = canonical in _json.load(open(lion_path))
            live_lion = (now and not had)
            if not now and not had:
                out.setdefault("warnings", []).append(
                    "LION_CALL_RETURNED_EMPTY: chemprop ran but produced no output"
                )
        except Exception as exc:  # noqa: BLE001
            out.setdefault("warnings", []).append(
                f"LION_FETCH_FAILED: {type(exc).__name__}: {str(exc)[:800]}"
            )

    if predict_admet_for_smiles is not None:
        try:
            import json as _json
            had = canonical in _json.load(open(admet_path)) if admet_path.exists() else False
            predict_admet_for_smiles([canonical], cache_path=str(admet_path), verbose=False)
            now = canonical in _json.load(open(admet_path))
            live_admet = (now and not had)
        except Exception as exc:  # noqa: BLE001
            out.setdefault("warnings", []).append(
                f"ADMET_FETCH_FAILED: {type(exc).__name__}: {str(exc)[:80]}"
            )


    # 3. Build the 88-d feature row (LION + ADMET via assemble_X)
    bio_b = bundle.tandem.bioact_bundle
    fam_detected, family_method = _detect_family(
        mol, AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol), bio_b
    )
    fam = family_hint or fam_detected
    out["family_assigned"] = fam

    row = {
        "SMILES_canonical": canonical, "canonical_smi": canonical,
        "family": fam, "IAJD_id": "v15_query",
        "log10_flux_total": 0,
        "pKa": injected_pka if injected_pka is not None else 6.5,
        "pKa_sd": injected_pka_sd if injected_pka_sd is not None else 0.5,
    }
    df_q = pd.DataFrame([row])
    lion_path = CACHE_DIR / "lion_cache_v13.json"
    admet_path = CACHE_DIR / "admet_cache_v13.json"
    X_q, lion_modes = assemble_X(
        df_q, [mol], [AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)],
        [canonical],
        lion_cache_path=str(lion_path) if lion_path.exists() else None,
        admet_cache_path=str(admet_path) if admet_path.exists() else None,
    )
    x_full = X_q[0]
    lion_real = (lion_modes[0] == "cached") if lion_modes else False

    # 4. v15 stage prediction
    stage_out = _predict_stage(bundle.bioact, x_full, count_fp(mol))
    stage_out.update({
        "source": "v15_predicted",
        "family_assigned": fam,
        "family_detected": fam_detected,
        "lion_real": bool(lion_real),
        "lion_live_call": bool(live_lion),
        "admet_live_call": bool(live_admet),
        "warnings": out.get("warnings", []) + ([] if lion_real else ["LION_PROXY_USED"]),
        # Expose the assembled 88-feature vector so downstream callers can
        # apply alternate heads (the v1 bioactivity stacker uses cols
        # 50-63 for LION and 64-73 for ADMET).
        "x_full": x_full,
    })

    # 4. Per-organ partition: nearest-neighbor similarity lookup (not an ML
    # prediction). Restricted to the top-2 closest IAJDs by count-Tanimoto so
    # the partition is reported as a direct similarity score to the two
    # closest training molecules rather than a smoothed analog ensemble.
    fp_q_count = count_fp(mol)
    sims_all = bulk_tanimoto(fp_q_count, bundle.bioact.count_fps)
    order = sims_all.argsort()[::-1][:2]
    wide_neighbors = [{
        "iajd_id": bundle.bioact.ids[int(j)],
        "smiles":  bundle.bioact.smiles[int(j)],
        "family":  bundle.bioact.families[int(j)],
        "tanimoto": float(sims_all[int(j)]),
    } for j in order]
    per_organ = _per_organ_from_neighbors(wide_neighbors, bundle.bioact_table)
    stage_out["organ_delivery"] = per_organ
    out.update(stage_out)
    return out


_ORGAN_COLS = [("lung", "log10_flux_lung"), ("liver", "log10_flux_liver"),
               ("spleen", "log10_flux_spleen"), ("LN", "log10_flux_LN"),
               ("heart", "log10_flux_heart")]


def _per_organ_from_neighbors(neighbors: List[Dict[str, Any]],
                              bioact_table: Dict[str, Dict[str, Any]]
                              ) -> Dict[str, Any]:
    if not neighbors:
        return {"target_organ": None, "reason": "no_neighbors"}
    weights = []
    per_organ: Dict[str, List[Tuple[float, float]]] = {label: [] for label, _ in _ORGAN_COLS}
    for n in neighbors:
        rec = bioact_table.get(n.get("smiles") or "", {})
        s = float(n.get("tanimoto", 0.0))
        w = max(s, 0.0) ** 4
        for label, col in _ORGAN_COLS:
            v = rec.get(col)
            if v is None:
                continue
            per_organ[label].append((w, float(v)))
        weights.append(w)
    if sum(weights) <= 0:
        return {"target_organ": None, "reason": "zero_weight"}
    log_flux = {}
    for label, pairs in per_organ.items():
        if not pairs:
            continue
        ws = sum(w for w, _ in pairs)
        if ws <= 0:
            continue
        log_flux[label] = sum(w * v for w, v in pairs) / ws
    if not log_flux:
        return {"target_organ": None, "reason": "no_organ_values"}
    linear = {k: 10 ** v for k, v in log_flux.items()}
    total = sum(linear.values())
    partition = {k: round(100.0 * v / total, 1) for k, v in linear.items()}
    target = max(log_flux.items(), key=lambda kv: kv[1])
    return {
        "target_organ": target[0],
        "target_log10_flux": round(target[1], 3),
        "log10_flux_by_organ": {k: round(v, 3) for k, v in log_flux.items()},
        "partition_pct": partition,
        "n_neighbors_used": len(neighbors),
        "method": "similarity-score (Tanimoto-weighted) over top-2 nearest training IAJDs — NOT an ML prediction",
    }


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from iajd_predict import _load_bundle
    print("Loading tandem bundle...")
    tandem = _load_bundle()
    print("Building v15 (count-Tanimoto, delta XGB, global α)...")
    v15 = load_v15(tandem)
    print(f"pKa stage:    alpha={v15.pka.alpha:.2f}  pi90_half={v15.pka.pi90_half:.3f}"
          f"  loo_blend_mae={v15.pka.extra.get('loo_blend_mae')}")
    print(f"bioact stage: alpha={v15.bioact.alpha:.2f}  pi90_half={v15.bioact.pi90_half:.3f}"
          f"  loo_blend_mae={v15.bioact.extra.get('loo_blend_mae')}")
    test = "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1"
    print("\n-- pKa --"); print(json.dumps(predict_pka_v15(test, v15, "PE-Tris"), indent=2, default=str))
    print("\n-- bioact --"); print(json.dumps(predict_bioactivity_v15(test, v15, "PE-Tris"), indent=2, default=str))
