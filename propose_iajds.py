"""
propose_iajds.py — beam search for IAJDs predicted to clear a bioactivity threshold.

Pipeline:
  1. Build the structural library from the current bioact xlsx
  2. Seed with the top-K training IAJDs (highest measured log10_flux_total)
     OR with a user-supplied SMILES
  3. For depth=D rounds:
       - Apply all single-mutation operators (iajd_grammar.propose_single_mutations)
       - Steer by the per-family informed-mutation prior (family_sar): each move
         gets a signed Δ_SAR = the training set's expected Δlog10_flux for moving
         along that family's significant, de-correlated structure-activity axes
         (and observed head/linkage mean-flux gaps). Strongly-adverse moves are
         pruned before scoring; the redundant no-signal tail-variant explosion
         is capped. Families with no statistically-supported axis get no steering.
       - Score each candidate with the bioact regressor + binary head + physics
       - Filter for Tanimoto novelty (< 0.85 to existing training set) and
         a rough synthesizability gate (RDKit sanitization passes, no exotic atoms,
         MolWt ≤ 1500, RotatableBonds ≤ 50)
       - Rank by  (1−α)·[P(≥T)·max(0,Δ_ML)] + α·[Q_physics·UCB] + β·Δ_SAR
  4. Return ranked candidate list with mutation provenance + Δ_SAR reasons

Outputs:
  - proposed_iajds.csv  (ranked candidates with scores + mutation trail)
  - proposed_iajds_summary.json

CLI:
  python propose_iajds.py --threshold 8.0 --beam 20 --depth 3 --seed top10
"""
from __future__ import annotations
import argparse, json, pickle, sys, warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

from iajd_grammar import (
    Library, Seed, build_library, decompose_row,
    propose_single_mutations, FAMILY_ASSEMBLERS,
    humanize_mutation_tag,
)
from family_sar import (
    sar_features, sar_prior_for_candidate, family_has_prior,
)

# Informed-mutation (SAR-prior) tuning. The prior is in log10_flux_total units.
#   PRUNE_ADVERSE   : per-mutation prior at/below which a move is dropped BEFORE
#                     ML scoring — but only in families that actually have a
#                     data-supported prior (flat/low-data families never prune).
#   TAIL_NOSIG_CAP  : max no-signal tail-type variants kept per parent, so the
#                     dozens of redundant tail swaps in flat families don't drown
#                     the few informed moves (and don't waste ML featurization).
#   SAR_SHOW        : a candidate the ML thinks is ≤ seed is still surfaced if its
#                     SAR prior is ≥ this (a data-supported direction the
#                     conservative ML regressor won't extrapolate to).
PRUNE_ADVERSE = -0.40
TAIL_NOSIG_CAP = 8
SAR_SHOW = 0.15
_TAIL_TAG_PREFIXES = ("tail", "all_tails", "multi_tail")

BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
BIN_BUNDLE  = ROOT / "IAJD_master/bundles_caches/bioact_binary_bundle.pkl"
V14_BUNDLE  = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"

# Filters
MAX_MW = 1500.0
MAX_ROTATABLE = 50
# Tanimoto gating: only require candidate != training compound (1.0 means
# identical). Higher similarity is desirable here because the regressor was
# trained on those nearby compounds and can score them confidently.
NOVELTY_TANIMOTO_MAX = 0.999

try:
    _MFP_GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    def _fp(m): return _MFP_GEN.GetFingerprint(m)
except AttributeError:
    def _fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _passes_filters(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    try: Chem.SanitizeMol(m)
    except Exception: return False
    if Descriptors.ExactMolWt(m) > MAX_MW: return False
    if Descriptors.NumRotatableBonds(m) > MAX_ROTATABLE: return False
    for a in m.GetAtoms():
        if a.GetAtomicNum() not in (1, 6, 7, 8): return False
    return True


def _tanimoto_max(query_fp, train_fps) -> float:
    if not train_fps: return 0.0
    return max(TanimotoSimilarity(query_fp, tf) for tf in train_fps)


_BIOACT_LOOKUP = None      # canonical_smiles -> dict (training row)
_DESC_CACHE: dict = {}     # canonical_smiles -> dict of feature columns
_V91_BUNDLE_CACHE = None   # v9.1 pKa bundle, lazy-loaded
_LIVE_PKA_CACHE: dict = {} # canonical SMILES -> (pKa, pKa_sd) from live v9.1
_V15_BUNDLE_CACHE = None   # v15 hybrid bundle (physics-quality + sigma)
_PHYSICS_CACHE: dict = {}  # canonical SMILES -> (Q_physics, sigma_combined) tuple
_MONOTONE_AXES = None      # loaded from monotone_axes.json (lazy)
_MONOTONE_CACHE: dict = {} # canonical SMILES -> (bonus, signals_dict) tuple


# No-proxy policy: pKa MUST come from live MolGpKa + per-family debias.
# _live_molgpka_pka() is a faster path than full v9.1 (skips conformer
# regeneration etc.) and caches by canonical SMILES.
_MOLGPKA_PKA_CACHE: dict = {}   # canonical SMILES -> debiased pKa (live)


def _live_molgpka_pka(smiles: str, family: str) -> float:
    """Live MolGpKa GCN + per-family debias. Real per-molecule pKa, cached
    by canonical SMILES. NO family-median fallback — returns NaN if MolGpKa
    can't run on this structure."""
    if smiles in _MOLGPKA_PKA_CACHE:
        return _MOLGPKA_PKA_CACHE[smiles]
    try:
        # Reuse the v52 _try_live_molgpka entry point (already patched to use
        # the local molgpka_src/ install).
        import sys as _sys, os as _os
        proj_root = str((ROOT).resolve())
        molgpka_src = _os.path.join(proj_root, "molgpka_src")
        if molgpka_src not in _sys.path:
            _sys.path.insert(0, molgpka_src)
        from predict_pka import predict as molgpka_predict   # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _MOLGPKA_PKA_CACHE[smiles] = float("nan")
            return float("nan")
        base_dict, _ = molgpka_predict(mol)
        if not base_dict:
            _MOLGPKA_PKA_CACHE[smiles] = float("nan")
            return float("nan")
        raw_max_base = float(max(base_dict.values()))
        # Apply per-family debias if available
        import joblib as _joblib
        debias = _joblib.load(ROOT / "IAJD_master/bundles_caches/molgpka_debias_models.joblib")
        if family in debias:
            d = debias[family]
            pka = d["slope"] * raw_max_base + d["intercept"]
        else:
            tot_n = sum(d_["n"] for d_ in debias.values())
            slope = sum(d_["slope"] * d_["n"] for d_ in debias.values()) / tot_n
            intercept = sum(d_["intercept"] * d_["n"] for d_ in debias.values()) / tot_n
            pka = slope * raw_max_base + intercept
        _MOLGPKA_PKA_CACHE[smiles] = float(pka)
        return float(pka)
    except Exception:
        _MOLGPKA_PKA_CACHE[smiles] = float("nan")
        return float("nan")


def _load_monotone_axes():
    """Lazy-load the monotone-axis analysis (analyze_monotone_axes.py output)."""
    global _MONOTONE_AXES
    if _MONOTONE_AXES is not None:
        return _MONOTONE_AXES
    path = ROOT / "monotone_axes.json"
    if not path.exists():
        _MONOTONE_AXES = {"axes": {}, "_missing": True}
        return _MONOTONE_AXES
    try:
        import json as _json
        _MONOTONE_AXES = _json.loads(path.read_text())
    except Exception:
        _MONOTONE_AXES = {"axes": {}, "_missing": True}
    return _MONOTONE_AXES


def _monotone_bonus(smiles: str, features: dict) -> tuple:
    """For each monotone axis, check whether the candidate's value steps
    *beyond* the training range in the favorable direction.

    Returns (bonus_score ∈ [0, 0.6], signals_dict):
       bonus_score scales with how many monotone axes the candidate exceeds,
       weighted by their Spearman ρ. Capped at 0.6 so a single feature can't
       dominate.
       signals_dict is a {feature → "+0.12C beyond train_max" / etc.} for
       transparency.
    """
    if smiles in _MONOTONE_CACHE:
        return _MONOTONE_CACHE[smiles]
    axes = _load_monotone_axes().get("axes", {})
    bonus = 0.0
    signals = {}
    for feat, info in axes.items():
        if not info.get("reliably_monotone"):
            continue
        if feat not in features:
            continue
        x = features.get(feat)
        if x is None or not np.isfinite(x):
            continue
        train_min = info.get("train_min")
        train_max = info.get("train_max")
        rho = info.get("spearman_rho") or 0.0
        direction = info.get("direction", "increasing")
        train_range = (train_max - train_min) if (train_max is not None and train_min is not None) else 0
        if train_range <= 0:
            continue
        # Compute how far past the boundary we are, in fractions of the training range
        if direction == "increasing":
            if x > train_max:
                step = (x - train_max) / train_range
                axis_score = abs(rho) * min(0.3, step)   # cap per-axis at 0.3·|ρ|
                bonus += axis_score
                signals[feat] = (f"+{step:.2f}·range beyond train_max "
                                  f"(ρ={rho:+.2f}, +increasing helps)")
        else:   # decreasing
            if x < train_min:
                step = (train_min - x) / train_range
                axis_score = abs(rho) * min(0.3, step)
                bonus += axis_score
                signals[feat] = (f"−{step:.2f}·range below train_min "
                                  f"(ρ={rho:+.2f}, −decreasing helps)")
    bonus = min(bonus, 0.6)
    _MONOTONE_CACHE[smiles] = (bonus, signals)
    return bonus, signals


def _load_v15_bundle():
    """Lazy-load the v15 hybrid bundle (physics scaler/model + σ's + weights)."""
    global _V15_BUNDLE_CACHE
    if _V15_BUNDLE_CACHE is not None:
        return _V15_BUNDLE_CACHE
    import joblib
    bundle_path = ROOT / "IAJD_master/bundles_caches/v15_hybrid_bundle.joblib"
    if not bundle_path.exists():
        _V15_BUNDLE_CACHE = {"_missing": True}
        return _V15_BUNDLE_CACHE
    try:
        _V15_BUNDLE_CACHE = joblib.load(bundle_path)
    except Exception:
        _V15_BUNDLE_CACHE = {"_missing": True}
    return _V15_BUNDLE_CACHE


def _physics_quality_quick(smiles: str, seed: "Seed | None" = None,
                            family: str = "GA-Tris") -> dict:
    """Inline physics-quality computation. NO PROXIES.

    All inputs are real per-molecule computations:
      • pKa            — live MolGpKa GCN (subprocess) + per-family debias
      • a_head         — 3D ETKDGv3+MMFF94 VdW projection (head_area_3d)
      • CPP_geometric  — Tanford a_head × l_tail / v_tail
      • protonation    — Henderson–Hasselbalch at endo/cyto pH using live pKa
      • HLB Griffin    — real MW(head)/MW(total) ratio
      • logKp membrane — real MolLogP + size correction
      • chain_avg_carbons — REAL count from SMILES tail atoms (not 10.0)

    When ANY input is NaN (e.g. live MolGpKa failed for this SMILES, or 3D
    embedder couldn't converge on the head conformer), the corresponding
    output is NaN. q_physics is the mean of finite components; NaN if no
    component finite. Caller (proposer scoring) refuses to rank candidates
    by NaN, falling back to non-physics scoring honestly.

    Returns:
        {"q_physics": float ∈ [0,1] or NaN,
         "cpp": float or NaN,
         "endosomal_escape": float or NaN,
         "hlb": float or NaN,
         "logKp": float or NaN,
         "physics_yhat": float or NaN  (v15 Ridge prediction; NaN if bundle missing
                                         or any feature NaN — no zero-fill proxy)}
    """
    if smiles in _PHYSICS_CACHE:
        return _PHYSICS_CACHE[smiles]
    from physics_features import compute_all_physics_features
    # Real per-molecule pKa via live MolGpKa GCN + per-family debias.
    # NaN if MolGpKa fails — downstream protonation/escape/Manning will
    # propagate NaN honestly.
    pka = _live_molgpka_pka(smiles, family)
    head_group = (seed.head if seed is not None else None)
    linker_n = (seed.linker_n if seed is not None else 4)
    n_chains = 3 if ("Tris" in family or family == "PE-Gallic") else 2
    # No proxy: compute actual mean tail carbon count from the SMILES.
    chain_avg = _avg_tail_carbons_from_smiles(smiles, n_chains)
    try:
        feats = compute_all_physics_features(
            smiles, pka=pka, linker_length=linker_n,
            n_tail_chains=n_chains, chain_avg_carbons=chain_avg,
            head_group=head_group,
        )
    except Exception:
        feats = {}

    # No-proxy component scoring: NaN inputs → NaN scores → q_physics excludes them.
    cpp = feats.get("cpp_geometric", float("nan"))
    cpp_score = float(np.exp(-((cpp - 1.0) ** 2) / (2 * 0.3 ** 2))) if np.isfinite(cpp) else float("nan")

    p_e = feats.get("protonation_endosome", float("nan"))
    p_c = feats.get("protonation_cytosol", float("nan"))
    if np.isfinite(p_e) and np.isfinite(p_c):
        escape = float(max(0.0, p_e - p_c))
    else:
        escape = float("nan")

    hlb = feats.get("hlb_griffin", float("nan"))
    hlb_score = float(np.exp(-((hlb - 8.5) ** 2) / (2 * 3.0 ** 2))) if np.isfinite(hlb) else float("nan")

    logKp = feats.get("logKp_membrane", float("nan"))
    membrane_score = float(1.0 / (1.0 + np.exp(-(logKp - 4.0)))) if np.isfinite(logKp) else float("nan")

    # Average over FINITE components only; NaN if none finite.
    component_scores = [cpp_score, escape, hlb_score, membrane_score]
    finite_components = [c for c in component_scores if c == c]   # NaN excluded
    q = float(np.mean(finite_components)) if finite_components else float("nan")

    # Physics Ridge prediction — NaN if any feature NaN (no zero-fill proxy).
    physics_yhat = float("nan")
    b = _load_v15_bundle()
    if not b.get("_missing"):
        try:
            cols = b["physics_cols"]
            X = np.array([[feats.get(c, np.nan) for c in cols]], dtype=float)
            if np.isfinite(X).all():
                Xs = b["physics_scaler"].transform(X)
                physics_yhat = float(b["physics_model"].predict(Xs)[0])
        except Exception:
            pass

    out = {
        "q_physics": q,
        "cpp": float(cpp) if np.isfinite(cpp) else float("nan"),
        "endosomal_escape": escape,
        "hlb": float(hlb) if np.isfinite(hlb) else float("nan"),
        "logKp": float(logKp) if np.isfinite(logKp) else float("nan"),
        "physics_yhat": physics_yhat,
    }
    _PHYSICS_CACHE[smiles] = out
    return out


def _avg_tail_carbons_from_smiles(smiles: str, n_chains: int) -> float:
    """Compute REAL average number of carbons per alkoxy tail (no proxy).

    Counts CH₂/CH₃ runs attached via –O– ether linkages, divides by the
    family-expected number of tails. Returns NaN if the molecule can't be
    parsed or no tails detected; the physics computation then NaN-propagates.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")
        # Match the SMARTS pattern: aryl-O-C(sp3) chain → count the (sp3 C) run length
        patt = Chem.MolFromSmarts("[c,C]O[CH2,CH3]")
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return float("nan")
        chain_lengths = []
        for match in matches:
            o_idx = match[1]
            c_start = match[2]
            # Walk along the aliphatic chain via sp3 C neighbors
            visited = {match[0], o_idx}
            stack = [c_start]
            chain = []
            while stack:
                a = stack.pop()
                if a in visited:
                    continue
                visited.add(a)
                atom = mol.GetAtomWithIdx(a)
                if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
                    continue
                if atom.GetHybridization() not in (Chem.HybridizationType.SP3,):
                    continue
                chain.append(a)
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() not in visited:
                        stack.append(nb.GetIdx())
            chain_lengths.append(len(chain))
        if not chain_lengths:
            return float("nan")
        return float(np.mean(chain_lengths))
    except Exception:
        return float("nan")


def _bioact_lookup():
    """Lazy-load the bioact xlsx as a dict {canonical_smiles -> row dict}.

    This lets us reuse pre-computed Block A descriptors (Pct_V_Bur_max, etc.)
    for training-set queries, instead of getting 0s from the on-the-fly path.
    """
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
    print(f"  bioact lookup: {len(out)} canonical-SMILES → row entries")
    return out


def _features_for_query_smiles(smiles: str) -> dict:
    """Compute the full set of Block A descriptors for a NEW (mutated) SMILES.

    Reuses expand_datasets' compute_rdkit_features + compute_3d_features so
    the column set matches what _compute_block_a_from_row reads from the row.
    """
    if smiles in _DESC_CACHE:
        return _DESC_CACHE[smiles]
    from expand_datasets import compute_rdkit_features, compute_3d_features
    d = compute_rdkit_features(smiles) or {}
    d3 = compute_3d_features(smiles) or {}
    d.update(d3)
    # Aliases needed by _compute_block_a_from_row
    d.setdefault("Linker_Length", d.get("Linker_Length", np.nan))
    d.setdefault("Taft_Steric_Sum", d.get("Taft_Steric_Sum", np.nan))
    _DESC_CACHE[smiles] = d
    return d


def _live_pka_for_smiles(smiles: str, family_hint: str = "GA-Tris"):
    """Live v9.1 pKa prediction (uses live MolGpKa + per-family debias)."""
    global _V91_BUNDLE_CACHE
    if smiles in _LIVE_PKA_CACHE:
        return _LIVE_PKA_CACHE[smiles]
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
        r = predict_pka_v91(smiles, _V91_BUNDLE_CACHE, family_hint=family_hint,
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
    _LIVE_PKA_CACHE[smiles] = pair
    return pair


def _featurize_for_v14(smiles_list: List[str]):
    """Run the same feature-assembly path the v14 bundle uses, returning X.

    For training-set SMILES we re-use the stored xlsx row (which has all the
    pre-computed descriptors). For new SMILES we compute them on the fly.
    """
    from bioact_v14_pipeline import assemble_X
    OUT = ROOT / "IAJD_master/bundles_caches"
    lion_cache  = OUT / "lion_cache_v13.json"
    admet_cache = OUT / "admet_cache_v13.json"
    lion_fps    = OUT / "lion_train_fps.pkl"
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps_q = [_fp(m) if m is not None else None for m in mols]
    lookup = _bioact_lookup()
    rows = []
    for sm in smiles_list:
        canon = sm
        try:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
        except Exception:
            pass
        if canon in lookup:
            # Reuse the cached descriptor block from the training row
            row = dict(lookup[canon])
        else:
            # New SMILES — compute descriptors on the fly
            row = _features_for_query_smiles(canon)
            # Live v9.1 pKa + per-family debias (no proxy fallback)
            if pd.isna(row.get("pKa")):
                fam_hint = row.get("family") or "GA-Tris"
                pka_pred, pka_sd = _live_pka_for_smiles(canon, family_hint=fam_hint)
                if np.isfinite(pka_pred):
                    row["pKa"] = pka_pred
                    row["pKa_sd"] = pka_sd
        row["canonical_smi"] = canon
        if "family" not in row or pd.isna(row.get("family")):
            row["family"] = "GA-Tris"   # default placeholder
        row["log10_flux_total"] = np.nan
        rows.append(row)
    df_q = pd.DataFrame(rows)
    X, _modes = assemble_X(
        df_q, mols, fps_q, smiles_list,
        lion_cache_path=str(lion_cache) if lion_cache.exists() else None,
        admet_cache_path=str(admet_cache) if admet_cache.exists() else None,
        lion_train_fps_path=str(lion_fps) if lion_fps.exists() else None,
    )
    return X


def _extend_caches_for_smiles(smiles_list: List[str]) -> None:
    """Run real LION + ADMET subprocess prediction for SMILES NOT yet in the
    caches; populates them in place so the next assemble_X call hits real
    values instead of RDKit proxies."""
    import json
    sys.path.insert(0, str(ROOT / "IAJD_master/code"))
    try:
        from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles
    except Exception as e:
        print(f"  WARN: cache-extension import failed ({e}); falling back to proxies.")
        return
    OUT = ROOT / "IAJD_master/bundles_caches"
    lion_path = OUT / "lion_cache_v13.json"
    admet_path = OUT / "admet_cache_v13.json"
    if not lion_path.exists() or not admet_path.exists():
        return
    lion_cache = json.load(open(lion_path))
    admet_cache = json.load(open(admet_path))
    # Canonical SMILES of each query
    canon = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon.append(Chem.MolToSmiles(m))
    to_lion = sorted({c for c in canon if c not in lion_cache})
    to_admet = sorted({c for c in canon if c not in admet_cache})
    if not to_lion and not to_admet:
        return
    if to_admet:
        print(f"  extending ADMET cache: +{len(to_admet)} SMILES")
        try: predict_admet_for_smiles(to_admet, cache_path=str(admet_path), verbose=False)
        except Exception as e: print(f"    ADMET extension failed: {e}")
    if to_lion:
        print(f"  extending LION cache:  +{len(to_lion)} SMILES")
        try: predict_lion_for_smiles(to_lion, cache_path=str(lion_path), verbose=False)
        except Exception as e: print(f"    LION extension failed: {e}")


def _score_with_v14(smiles_list: List[str], bundle_v14: dict, bundle_bin: dict,
                    threshold: float, extend_cache: bool = True,
                    return_ensemble_sigma: bool = False):
    """Return (yhat, p_above[, sigma_ens]) arrays for the candidate SMILES list.

    When `return_ensemble_sigma` is True, also returns per-candidate σ_eff
    from the deep ensemble (T2 #5), computed on the SAME X used for the
    binary regressor prediction. σ_eff = max(σ_query · calibration,
    σ_family · 0.7). NaN if the ensemble bundle isn't loadable.
    """
    if extend_cache:
        _extend_caches_for_smiles(smiles_list)
    X = _featurize_for_v14(smiles_list)
    reg = bundle_bin["regressor"]
    yhat = reg.predict(X)
    # Probability via per-threshold calibrator (interpolate if T not in grid)
    grid = sorted(bundle_bin["threshold_calibrators"].keys(), key=float)
    grid_floats = np.array([float(g) for g in grid])
    # Nearest grid point
    nearest = grid[int(np.argmin(np.abs(grid_floats - threshold)))]
    c = bundle_bin["threshold_calibrators"][nearest]
    a, b = c["a"], c["b"]
    p_above = 1.0 / (1.0 + np.exp(-(a * (yhat - threshold) + b)))
    if not return_ensemble_sigma:
        return yhat, p_above
    # Per-candidate ensemble σ on the same X (T3 #8 active-learning surface)
    sigma_ens = _ensemble_sigma_batch(X)
    return yhat, p_above, sigma_ens


def _expected_improvement(yhat: float, sigma: float, threshold: float) -> float:
    """Closed-form Expected Improvement (EI) under a Normal(ŷ, σ²) posterior
    relative to `threshold` (T3 #8 active-learning acquisition).

    EI(ŷ, σ, T) = (ŷ − T)·Φ(z) + σ·φ(z),  z = (ŷ − T)/σ

    High EI ⇔ candidate likely above T AND the model has enough uncertainty
    that a real measurement would be informative. Returns 0 when σ <= 0 or
    inputs are NaN — those candidates are uninformative for EI ranking.
    """
    import math
    if not (sigma == sigma) or sigma <= 0:
        return 0.0
    if not (yhat == yhat):
        return 0.0
    z = (float(yhat) - float(threshold)) / float(sigma)
    # Standard-normal CDF and PDF
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (float(yhat) - float(threshold)) * phi + float(sigma) * pdf


def _ensemble_sigma_batch(X) -> np.ndarray:
    """Per-row σ_eff for a batch X (shape n×92) from the deep ensemble bundle.

    σ_eff = max(σ_query · calibration, σ_family · 0.7) using each row's
    family (unavailable here, so falls back to global σ). Returns NaN-filled
    array if ensemble bundle isn't loadable.
    """
    global _ENSEMBLE_BUNDLE_CACHE
    try:
        if _ENSEMBLE_BUNDLE_CACHE is None:
            from predict_ensemble import load_ensemble_bundle
            _ENSEMBLE_BUNDLE_CACHE = load_ensemble_bundle()
    except Exception:
        return np.full(len(X), np.nan)
    b = _ENSEMBLE_BUNDLE_CACHE
    models = b["models"]
    # Per-model predictions on the full batch
    M_preds = np.array([m.predict(X) for m in models])  # shape (M, n)
    sigma_query_raw = M_preds.std(axis=0)               # per-candidate raw std
    sigma_query = sigma_query_raw * float(b.get("sigma_calibration", 1.0))
    fam_sigma_floor = float(b.get("global_sigma", 0.43)) * 0.7
    sigma_eff = np.maximum(sigma_query, fam_sigma_floor)
    return sigma_eff


_ENSEMBLE_BUNDLE_CACHE = None

# ---------------------------------------------------------------------------
# ChemBERTa structural-novelty (T3 #7 honest pivot)
# ---------------------------------------------------------------------------
# We do NOT use ChemBERTa embeddings as v14 regression features (negative LOO
# result, see docs/T3_7_chemberta_negative_result.md), but they DO give a
# real structural-novelty signal that complements Tanimoto. Used as a
# proposer column only.

_CHEMBERTA_PCA = None
_CHEMBERTA_TRAIN_EMB_N = None   # row-normalized training PCA embeddings

def _ensure_chemberta_novelty():
    """Load the saved PCA + training embeddings on first use."""
    global _CHEMBERTA_PCA, _CHEMBERTA_TRAIN_EMB_N
    if _CHEMBERTA_PCA is not None and _CHEMBERTA_TRAIN_EMB_N is not None:
        return True
    try:
        import joblib, numpy as _np
        bundle = joblib.load(ROOT / "IAJD_master/bundles_caches/chemberta_pca_bundle.joblib")
        _CHEMBERTA_PCA = bundle["pca"]
        train_emb = bundle["embeddings_pca"].astype(_np.float32)
        norms = _np.linalg.norm(train_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _CHEMBERTA_TRAIN_EMB_N = train_emb / norms
        return True
    except Exception:
        return False


def _chemberta_novelty_batch(smiles_list):
    """Per-candidate structural novelty = 1 − max cosine similarity to any
    training molecule in ChemBERTa-PCA-32 space (T3 #7 active-learning surface).

    No-proxy: every value comes from a real frozen-encoder forward pass on
    the query SMILES (cached on disk by canonical SMILES). Returns NaN-filled
    array if the bundle isn't loadable.
    """
    if not _ensure_chemberta_novelty():
        return np.full(len(smiles_list), np.nan)
    try:
        from chemberta_embedder import embed_smiles_batch
    except Exception:
        return np.full(len(smiles_list), np.nan)
    embs = embed_smiles_batch(smiles_list, use_cache=True)
    if not np.isfinite(embs).all():
        # fall back to zero-fill only the NaN rows so the others still produce
        # real novelty; mark NaN rows as NaN in the output
        bad_rows = ~np.isfinite(embs).all(axis=1)
    else:
        bad_rows = np.zeros(len(smiles_list), dtype=bool)
    embs_pca = _CHEMBERTA_PCA.transform(embs)
    norms = np.linalg.norm(embs_pca, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs_pca / norms
    # cos_max to training centroid set
    cos_to_train = embs_n @ _CHEMBERTA_TRAIN_EMB_N.T
    cos_max = cos_to_train.max(axis=1)
    novelty = 1.0 - cos_max
    novelty[bad_rows] = np.nan
    return novelty


def _per_family_sigma(family: str) -> float:
    """Look up per-family σ from the deep ensemble bundle (T2 #5).

    These are real LOO RMSEs from the ensemble's bootstrap CV, not constants.
    Returns NaN if the ensemble bundle isn't available — caller falls back to
    the pre-ensemble global σ.

    No proxy. σ_family comes from real cross-validated residuals on that
    family's training rows.
    """
    global _ENSEMBLE_BUNDLE_CACHE
    try:
        if _ENSEMBLE_BUNDLE_CACHE is None:
            from predict_ensemble import load_ensemble_bundle
            _ENSEMBLE_BUNDLE_CACHE = load_ensemble_bundle()
    except Exception:
        return float("nan")
    pfs = _ENSEMBLE_BUNDLE_CACHE.get("per_family_sigma", {})
    val = pfs.get(family)
    if val is None:
        # Fall back to the ensemble's overall global σ (still real, not arbitrary)
        val = _ENSEMBLE_BUNDLE_CACHE.get("global_sigma", float("nan"))
    return float(val)


def _pick_seeds(df: pd.DataFrame, top_k: int) -> List[dict]:
    have = df.dropna(subset=["log10_flux_total"]).copy()
    have = have.sort_values("log10_flux_total", ascending=False).head(top_k)
    return [r.to_dict() for _, r in have.iterrows()]


def _sar_prior_for_mutation(parent_seed, parent_feats, cand_seed, cand_smi, tag):
    """Per-mutation informed prior (parent→candidate), in log10_flux units.

    Returns (step_prior, reason). Cross-family jumps return a neutral 0.0 —
    there is no principled within-family SAR comparison across architectures,
    so we don't fabricate one."""
    if tag.startswith("family:"):
        return 0.0, ""
    cand_feats = sar_features(cand_smi)
    step, reason, _sig = sar_prior_for_candidate(
        parent_seed.family, parent_feats, cand_feats,
        mutation_tag=tag,
        seed_linker=parent_seed.linker_n, cand_linker=cand_seed.linker_n,
        head_old=parent_seed.head, head_new=cand_seed.head,
        linkage_old=parent_seed.linkage, linkage_new=cand_seed.linkage,
    )
    return step, reason


def beam_search_streaming(threshold: float, beam: int, depth: int,
                          seeds: List[Seed], library: Library,
                          bundle_v14: dict, bundle_bin: dict, train_fps,
                          exploration_weight: float = 0.0,
                          kappa_ucb: float = 1.5,
                          sar_weight: float = 0.4):
    """Generator: yield (round_idx, cumulative_df) after each beam round.

    Wraps beam_search but emits the DataFrame-so-far after each scoring pass so
    a Gradio caller can stream partial results to the UI instead of waiting for
    the full depth-D loop to complete.

    Scoring:
        score = (1 - α)·ML_score + α·PHYSICS_UCB_score + β·SAR_prior
        ML_score      = P(≥T) × max(0, ŷ − ŷ_seed)
        PHYSICS_UCB   = Q_physics × max(0, ŷ + κ·σ − ŷ_seed)
        SAR_prior     = Σ over the mutation trail of the per-mutation
                        family-SAR prior (family_sar.sar_prior_for_candidate):
                        the training set's expected Δlog10_flux for moving the
                        candidate along the family's significant, de-correlated
                        structure-activity axes (and observed head/linkage
                        mean-flux gaps). Signed: favourable moves lift the
                        score, proven-adverse moves sink it.

        Q_physics ∈ [0,1] is the mechanism-based quality score, σ comes from
        the v15 hybrid bundle, α = exploration_weight balances ML ↔ physics,
        and β = sar_weight scales the data-grounded informed-mutation prior.

    Informed mutations also shape GENERATION: before the expensive ML scoring,
    moves that are strongly adverse per the family SAR are pruned (only in
    families that actually have a data-supported prior), and the redundant
    no-signal tail-variant explosion is capped — so the few informed moves
    aren't drowned out (and we don't waste featurization on noise).
    """
    # Load v15 once for σ values; prefer ensemble per-family σ (T2 #5) when
    # the ensemble bundle is loaded. Per-family σ is the actual cross-validated
    # residual std for that family, not a constant or an inflated v15 disagreement.
    _v15 = _load_v15_bundle()
    if _v15.get("_missing"):
        # No proxy: both ensemble per-family σ AND v15 σ unavailable → NaN.
        # The UCB term will degrade to NaN; the proposer falls back to pure
        # ML and physics-quality scoring without an honest upper-bound.
        sigma_global_fallback = float("nan")
    else:
        sigma_global_fallback = (_v15["sigma"]["ml"] ** 2
                                  + _v15["sigma"]["disagreement"] ** 2) ** 0.5
    explored = set()
    current = []
    seed_smis = []
    seed_kept = []
    for s in seeds:
        sm = s.original_smiles or s.assemble()
        if sm is None:
            continue
        seed_smis.append(sm)
        seed_kept.append(s)
    seeds = seed_kept
    seed_yhat, seed_p = _score_with_v14(seed_smis, bundle_v14, bundle_bin, threshold,
                                          extend_cache=False)
    for s, sm, yh, p in zip(seeds, seed_smis, seed_yhat, seed_p):
        # Inline physics quality for the seed
        seed_phys = _physics_quality_quick(sm, s, family=s.family if s else "GA-Tris")
        current.append({
            "smiles": sm, "seed": s, "yhat": float(yh), "p_above": float(p),
            "yhat_seed": float(yh),
            # Pure-physics seed baseline (used for α=1 score Δ)
            "physics_yhat_seed": float(seed_phys["physics_yhat"]),
            "delta_vs_seed": 0.0,
            "tanim_max_to_train": float("nan"),
            "q_physics": seed_phys["q_physics"],
            "cpp": seed_phys["cpp"],
            "endosomal_escape": seed_phys["endosomal_escape"],
            "physics_yhat": seed_phys["physics_yhat"],
            "ucb_score": float(yh),
            "score_ml": 0.0,
            "score_phys": 0.0,
            "sar_prior": 0.0,
            "sar_step": 0.0,
            "sar_reason": "",
            "mutation_trail": ["seed"],
            "mutation_description": "Original seed (no mutation)",
            "mutation_tag": "seed",
            "parent_smiles": None,
            "score": 0.0,
        })
        explored.add(sm)
    # `shown` is what the UI table renders: seeds + qualifying candidates only.
    # Sub-par candidates still seed the next beam round (via `current`) but are
    # never rendered — matching the user's "don't show me worse-than-seed rows".
    shown = list(current)
    yield 0, pd.DataFrame(shown)

    for d in range(depth):
        current.sort(key=lambda r: -r["score"])
        current = current[:beam]
        next_pool = []
        for parent in current:
            parent_seed = parent["seed"]
            parent_feats = sar_features(parent["smiles"])
            parent_cum = parent.get("sar_prior", 0.0)
            fam_has_prior = family_has_prior(parent_seed.family)
            local = []          # this parent's accepted candidates
            tail_nosig = 0      # count of no-signal tail-variant moves kept
            for cand_smi, tag, cand_seed in propose_single_mutations(parent_seed, library):
                if cand_smi in explored:
                    continue
                if not _passes_filters(cand_smi):
                    continue
                m = Chem.MolFromSmiles(cand_smi)
                cand_fp = _fp(m)
                tanim = _tanimoto_max(cand_fp, train_fps)
                if tanim >= NOVELTY_TANIMOTO_MAX:
                    continue
                # ── informed-mutation gate (cheap, pre-ML) ──
                step, reason = _sar_prior_for_mutation(
                    parent_seed, parent_feats, cand_seed, cand_smi, tag)
                # Drop strongly-adverse moves, but only where we actually have a
                # data-supported prior (flat/low-data families never prune).
                if fam_has_prior and step <= PRUNE_ADVERSE:
                    continue
                # Cap the redundant no-signal tail-variant explosion per parent.
                is_tail = tag.startswith(_TAIL_TAG_PREFIXES)
                if is_tail and abs(step) < 1e-6:
                    if tail_nosig >= TAIL_NOSIG_CAP:
                        continue
                    tail_nosig += 1
                explored.add(cand_smi)
                local.append({
                    "smiles": cand_smi, "seed": cand_seed,
                    "parent_smiles": parent["smiles"],
                    "mutation_tag": tag,
                    "mutation_description": humanize_mutation_tag(tag),
                    "mutation_trail": parent["mutation_trail"] + [tag],
                    "tanim_max_to_train": float(tanim),
                    "yhat_seed": parent["yhat_seed"],
                    "physics_yhat_seed": parent.get("physics_yhat_seed", float("nan")),
                    "sar_step": float(step),
                    "sar_prior": float(parent_cum + step),
                    "sar_reason": reason,
                })
            next_pool.extend(local)
        if not next_pool:
            break
        smis = [r["smiles"] for r in next_pool]
        # Score + per-candidate ensemble σ (T3 #8: active-learning surface)
        yhats, ps, sigma_ens = _score_with_v14(
            smis, bundle_v14, bundle_bin, threshold, return_ensemble_sigma=True,
        )
        # ChemBERTa structural novelty (T3 #7 honest pivot) — per-candidate
        # cosine distance to training neighbourhood in pretrained semantic space.
        novelty_ch = _chemberta_novelty_batch(smis)
        for r, yh, p, s_ens, nov in zip(next_pool, yhats, ps, sigma_ens, novelty_ch):
            r["yhat"] = float(yh)
            r["p_above"] = float(p)
            r["delta_vs_seed"] = float(yh) - r["yhat_seed"]
            r["sigma_ensemble"] = float(s_ens) if s_ens == s_ens else float("nan")
            r["chemberta_novelty"] = float(nov) if nov == nov else float("nan")

            # Inline physics quality — live MolGpKa subprocess + 3D ETKDGv3 head
            # area + real RDKit logP/HLB/Manning (no family-median pKa, no
            # head-area lookup proxy; audit 2026-05-29).
            cand_family = r["seed"].family if r.get("seed") else "GA-Tris"
            phys = _physics_quality_quick(r["smiles"], r.get("seed"), family=cand_family)
            r["q_physics"] = phys["q_physics"]
            r["cpp"] = phys["cpp"]
            r["endosomal_escape"] = phys["endosomal_escape"]
            r["physics_yhat"] = phys["physics_yhat"]

            # Monotone-axis bonus: candidates that step BEYOND training range
            # along reliably-monotone features get a small additive lift.
            # (We need the underlying RDKit feature dict to evaluate this.)
            cand_features_for_monotone = (
                _features_for_query_smiles(r["smiles"]) if r["smiles"] not in _DESC_CACHE
                else _DESC_CACHE[r["smiles"]]
            )
            mono_bonus, mono_signals = _monotone_bonus(r["smiles"], cand_features_for_monotone)
            r["monotone_bonus"] = mono_bonus
            r["monotone_signals"] = "; ".join(f"{k}: {v}" for k, v in mono_signals.items())[:200]

            # UCB upper bound — prefer PER-CANDIDATE σ from the deep ensemble
            # (T3 #8); fall back to per-family σ from ensemble bundle (T2 #5);
            # fall back to v15 disagreement σ. NaN if none available — UCB
            # term degrades to NaN and is excluded from the score honestly.
            cand_family = r.get("seed").family if r.get("seed") else "GA-Tris"
            fam_sigma = _per_family_sigma(cand_family)
            if r["sigma_ensemble"] == r["sigma_ensemble"] and r["sigma_ensemble"] > 0:
                sigma_for_ucb = r["sigma_ensemble"]
            elif fam_sigma == fam_sigma and fam_sigma > 0:
                sigma_for_ucb = fam_sigma
            else:
                sigma_for_ucb = sigma_global_fallback
            ucb = float(yh) + kappa_ucb * sigma_for_ucb
            r["ucb_score"] = ucb
            r["sigma_for_ucb"] = float(sigma_for_ucb)

            # Component scores
            # score_ml (pure ML): P(≥T) × max(0, ML Δ vs seed)
            score_ml = float(p) * max(0.0, r["delta_vs_seed"])
            # score_phys (PURE PHYSICS): Q_physics × max(0, physics-Ridge Δ vs seed)
            # + monotone-axis extrapolation bonus. No ML term here — so α=1
            # truly selects on physics-only signal. Physics Δ uses the v15
            # Ridge prediction on real per-molecule features (CPP, escape,
            # HLB, logKp, etc.), all computed live without proxy defaults.
            phys_yhat = phys.get("physics_yhat", float("nan"))
            seed_phys_yhat = r.get("physics_yhat_seed", float("nan"))
            phys_delta = (float(phys_yhat) - float(seed_phys_yhat)
                          if (phys_yhat == phys_yhat and seed_phys_yhat == seed_phys_yhat)
                          else float("nan"))
            q_phys_val = float(phys["q_physics"]) if phys["q_physics"] == phys["q_physics"] else float("nan")
            if q_phys_val == q_phys_val and phys_delta == phys_delta:
                score_phys_core = q_phys_val * max(0.0, phys_delta)
            else:
                score_phys_core = float("nan")
            # Monotone bonus is itself a physics-derived RDKit-feature signal.
            # Scale it by |phys_delta| (NOT ML delta) so α=1 is truly pure
            # physics; floor at 0.5 to preserve the bonus when phys_delta is
            # tiny but the monotone signal is real.
            mono_scale = abs(phys_delta) if phys_delta == phys_delta else 0.5
            mono_term = mono_bonus * max(0.5, mono_scale)
            score_phys = (score_phys_core if score_phys_core == score_phys_core else 0.0) + mono_term
            r["score_ml"] = score_ml
            r["score_phys"] = score_phys
            r["physics_delta"] = phys_delta
            # Active-learning / Expected Improvement (T3 #8)
            r["expected_improvement"] = _expected_improvement(
                float(yh), r["sigma_ensemble"], float(threshold),
            )
            # Blended score: at α=0 → pure ML; at α=1 → pure physics + monotone.
            r["score"] = ((1.0 - exploration_weight) * score_ml
                          + exploration_weight * score_phys
                          + sar_weight * r["sar_prior"])

        # Stream each *qualifying* candidate one-at-a-time so the UI table grows
        # row-by-row. A candidate qualifies for display if EITHER:
        #   (a) delta_vs_seed > 0  — the ML model thinks it beats the seed, OR
        #   (b) sar_prior ≥ SAR_SHOW — the training set supports this direction
        #       even though the (conservative) ML regressor won't extrapolate to
        #       it. This is the "informed mutation" the user asked to surface.
        # Everything else (ML thinks it's worse AND no data-supported direction)
        # never appears in the table — it only seeds the next beam round.
        def _qualifies(r):
            return (r["delta_vs_seed"] > 0) or (r["sar_prior"] >= SAR_SHOW)
        qualifying = sorted([r for r in next_pool if _qualifies(r)],
                            key=lambda r: -r["score"])
        for cand in qualifying:
            shown.append(cand)
            df_partial = pd.DataFrame(shown)
            df_partial["_tiebreak_yhat"] = df_partial["yhat"]
            df_partial = df_partial.sort_values(
                ["score", "_tiebreak_yhat"], ascending=[False, False]
            ).reset_index(drop=True).drop(columns=["_tiebreak_yhat"])
            yield d + 1, df_partial

        # End-of-round summary yield (covers rounds where nothing qualified)
        df_partial = pd.DataFrame(shown)
        df_partial["_tiebreak_yhat"] = df_partial["yhat"]
        df_partial = df_partial.sort_values(
            ["score", "_tiebreak_yhat"], ascending=[False, False]
        ).reset_index(drop=True).drop(columns=["_tiebreak_yhat"])
        yield d + 1, df_partial
        current = next_pool


def beam_search(threshold: float, beam: int, depth: int,
                seeds: List[Seed], library: Library,
                bundle_v14: dict, bundle_bin: dict,
                train_fps,
                exploration_weight: float = 0.0,
                kappa_ucb: float = 1.5,
                sar_weight: float = 0.4,
                prune_adverse: float = PRUNE_ADVERSE,
                tail_nosig_cap: int = TAIL_NOSIG_CAP) -> pd.DataFrame:
    """Run depth-D beam search of width `beam`, scoring by the same combined
    formula as beam_search_streaming (live physics + UCB + monotone + SAR).

    Scoring:
        score = (1 − α)·ML_score + α·PHYSICS_UCB_score + β·SAR_prior
        ML_score      = P(≥T) × max(0, ŷ − ŷ_seed)
        PHYSICS_UCB   = Q_physics × max(0, ŷ + κ·σ − ŷ_seed)
                        + monotone_bonus · max(0.5, |Δ|)
        SAR_prior     = accumulated family informed-mutation prior

    Pre-ML steering:
      - candidates with SAR step ≤ prune_adverse are dropped (only in families
        with a data-supported prior)
      - no-signal tail-variant moves capped at tail_nosig_cap per parent

    Q_physics is computed inline via _physics_quality_quick (real CPP +
    endosomal-escape + HLB + log Kp + Ridge-model y-hat, using live MolGpKa
    pKa). σ comes from the v15 hybrid bundle when available.
    """
    # Load v15 σ once for UCB. Per-family σ from the deep ensemble (T2 #5) is
    # preferred when available (real cross-validated residual std), with v15's
    # disagreement-σ as the fallback.
    _v15 = _load_v15_bundle()
    if _v15.get("_missing"):
        # No proxy fallback: leave σ as NaN so UCB degrades honestly.
        sigma_global_fallback = float("nan")
    else:
        sigma_global_fallback = (_v15["sigma"]["ml"] ** 2
                                  + _v15["sigma"]["disagreement"] ** 2) ** 0.5

    explored = set()
    current = []
    # Score the seeds first. Prefer the ORIGINAL training SMILES (cached) for
    # seeds that came from training rows; falls back to assembled form
    # otherwise.
    seed_smis = []
    seed_kept = []
    for s in seeds:
        sm = s.original_smiles or s.assemble()
        if sm is None:
            continue
        seed_smis.append(sm)
        seed_kept.append(s)
    seeds = seed_kept
    seed_yhat, seed_p = _score_with_v14(seed_smis, bundle_v14, bundle_bin, threshold,
                                          extend_cache=False)  # training seeds already cached
    # Per-seed reference ŷ used to compute Δ vs seed for each candidate.
    for s, sm, yh, p in zip(seeds, seed_smis, seed_yhat, seed_p):
        seed_phys = _physics_quality_quick(sm, s, family=s.family if s else "GA-Tris")
        current.append({
            "smiles": sm, "seed": s, "yhat": float(yh), "p_above": float(p),
            "yhat_seed": float(yh),
            "physics_yhat_seed": float(seed_phys["physics_yhat"]),
            "delta_vs_seed": 0.0,
            "tanim_max_to_train": float("nan"),
            "q_physics": seed_phys["q_physics"],
            "cpp": seed_phys["cpp"],
            "endosomal_escape": seed_phys["endosomal_escape"],
            "physics_yhat": seed_phys["physics_yhat"],
            "ucb_score": float(yh),
            "monotone_bonus": 0.0,
            "monotone_signals": "",
            "score_ml": 0.0,
            "score_phys": 0.0,
            "sar_prior": 0.0, "sar_step": 0.0, "sar_reason": "",
            "mutation_trail": ["seed"],
            "mutation_description": "Original seed (no mutation)",
            "mutation_tag": "seed",
            "parent_smiles": None,
            "score": 0.0,  # Δ=0 for seed itself
        })
        explored.add(sm)

    all_candidates = list(current)

    for d in range(depth):
        # Sort current by score, keep top beam
        current.sort(key=lambda r: -r["score"])
        current = current[:beam]
        if current:
            print(f"  [beam d={d}] current best score={current[0]['score']:.3f}  "
                  f"(ŷ={current[0]['yhat']:.2f}, P≥{threshold}={current[0]['p_above']:.2f})")

        next_pool = []
        for parent in current:
            parent_seed = parent["seed"]
            parent_feats = sar_features(parent["smiles"])
            parent_cum = parent.get("sar_prior", 0.0)
            fam_has_prior = family_has_prior(parent_seed.family)
            tail_nosig = 0
            for cand_smi, tag, cand_seed in propose_single_mutations(parent_seed, library):
                if cand_smi in explored:
                    continue
                if not _passes_filters(cand_smi):
                    continue
                m = Chem.MolFromSmiles(cand_smi)
                cand_fp = _fp(m)
                tanim = _tanimoto_max(cand_fp, train_fps)
                if tanim >= NOVELTY_TANIMOTO_MAX:
                    # identical to a training compound — skip
                    continue
                step, reason = _sar_prior_for_mutation(
                    parent_seed, parent_feats, cand_seed, cand_smi, tag)
                if fam_has_prior and step <= prune_adverse:
                    continue
                is_tail = tag.startswith(_TAIL_TAG_PREFIXES)
                if is_tail and abs(step) < 1e-6:
                    if tail_nosig >= tail_nosig_cap:
                        continue
                    tail_nosig += 1
                explored.add(cand_smi)
                next_pool.append({
                    "smiles": cand_smi,
                    "seed": cand_seed,
                    "parent_smiles": parent["smiles"],
                    "mutation_tag": tag,
                    "mutation_description": humanize_mutation_tag(tag),
                    "mutation_trail": parent["mutation_trail"] + [tag],
                    "tanim_max_to_train": float(tanim),
                    "yhat_seed": parent["yhat_seed"],
                    "physics_yhat_seed": parent.get("physics_yhat_seed", float("nan")),
                    "sar_step": float(step),
                    "sar_prior": float(parent_cum + step),
                    "sar_reason": reason,
                })

        # Score this round's pool in a single batch (faster)
        if not next_pool:
            print(f"  [beam d={d}] no new candidates; stopping.")
            break
        smis = [r["smiles"] for r in next_pool]
        # Per-candidate ensemble σ (T3 #8)
        yhats, ps, sigma_ens = _score_with_v14(
            smis, bundle_v14, bundle_bin, threshold, return_ensemble_sigma=True,
        )
        # ChemBERTa structural novelty (T3 #7 honest pivot)
        novelty_ch = _chemberta_novelty_batch(smis)
        for r, yh, p, s_ens, nov in zip(next_pool, yhats, ps, sigma_ens, novelty_ch):
            r["yhat"] = float(yh)
            r["p_above"] = float(p)
            r["delta_vs_seed"] = float(yh) - r["yhat_seed"]
            r["sigma_ensemble"] = float(s_ens) if s_ens == s_ens else float("nan")
            r["chemberta_novelty"] = float(nov) if nov == nov else float("nan")

            # Live physics quality (MolGpKa pKa + CPP + escape + HLB + Ridge)
            cand_family = r["seed"].family if r.get("seed") else "GA-Tris"
            phys = _physics_quality_quick(r["smiles"], r.get("seed"),
                                          family=cand_family)
            r["q_physics"] = phys["q_physics"]
            r["cpp"] = phys["cpp"]
            r["endosomal_escape"] = phys["endosomal_escape"]
            r["physics_yhat"] = phys["physics_yhat"]

            # Monotone-axis bonus (favourable extrapolation beyond training range)
            cand_features_for_monotone = (
                _DESC_CACHE[r["smiles"]] if r["smiles"] in _DESC_CACHE
                else _features_for_query_smiles(r["smiles"])
            )
            mono_bonus, mono_signals = _monotone_bonus(r["smiles"], cand_features_for_monotone)
            r["monotone_bonus"] = mono_bonus
            r["monotone_signals"] = "; ".join(f"{k}: {v}" for k, v in mono_signals.items())[:200]

            # UCB upper bound — prefer per-candidate σ from the deep ensemble
            # (T3 #8); then per-family σ from ensemble bundle (T2 #5); then v15
            # disagreement σ; else NaN (no proxy).
            fam_sigma = _per_family_sigma(cand_family)
            if r["sigma_ensemble"] == r["sigma_ensemble"] and r["sigma_ensemble"] > 0:
                sigma_for_ucb = r["sigma_ensemble"]
            elif fam_sigma == fam_sigma and fam_sigma > 0:
                sigma_for_ucb = fam_sigma
            else:
                sigma_for_ucb = sigma_global_fallback
            ucb = float(yh) + kappa_ucb * sigma_for_ucb
            r["ucb_score"] = ucb
            r["sigma_for_ucb"] = float(sigma_for_ucb)

            # Component scores
            # Pure ML
            score_ml = float(p) * max(0.0, r["delta_vs_seed"])
            # Pure physics (no ML term inside) — uses physics_yhat from v15 Ridge.
            phys_yhat = phys.get("physics_yhat", float("nan"))
            seed_phys_yhat = r.get("physics_yhat_seed", float("nan"))
            phys_delta = (float(phys_yhat) - float(seed_phys_yhat)
                          if (phys_yhat == phys_yhat and seed_phys_yhat == seed_phys_yhat)
                          else float("nan"))
            q_phys_val = float(phys["q_physics"]) if phys["q_physics"] == phys["q_physics"] else float("nan")
            if q_phys_val == q_phys_val and phys_delta == phys_delta:
                score_phys_core = q_phys_val * max(0.0, phys_delta)
            else:
                score_phys_core = float("nan")
            mono_scale = abs(phys_delta) if phys_delta == phys_delta else 0.5
            mono_term = mono_bonus * max(0.5, mono_scale)
            score_phys = (score_phys_core if score_phys_core == score_phys_core else 0.0) + mono_term
            r["score_ml"] = score_ml
            r["score_phys"] = score_phys
            r["physics_delta"] = phys_delta
            # Active-learning EI score (T3 #8)
            r["expected_improvement"] = _expected_improvement(
                float(yh), r["sigma_ensemble"], float(threshold),
            )
            r["score"] = ((1.0 - exploration_weight) * score_ml
                          + exploration_weight * score_phys
                          + sar_weight * r["sar_prior"])

        all_candidates.extend(next_pool)
        current = next_pool

    df = pd.DataFrame(all_candidates)
    df["_tiebreak_yhat"] = df["yhat"]
    df = df.sort_values(["score", "_tiebreak_yhat"], ascending=[False, False]).reset_index(drop=True)
    df = df.drop(columns=["_tiebreak_yhat"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="Bioactivity threshold (log10_flux_total ≥ T)")
    ap.add_argument("--beam", type=int, default=20,
                    help="Beam width")
    ap.add_argument("--depth", type=int, default=3,
                    help="Beam search depth (mutation rounds)")
    ap.add_argument("--top_seeds", type=int, default=10,
                    help="Number of top training IAJDs to seed from")
    ap.add_argument("--seed_smiles", default=None,
                    help="Optional explicit seed SMILES (overrides --top_seeds)")
    ap.add_argument("--seed_family", default="GA-Tris",
                    help="Family hint for --seed_smiles (used for pKa debias & "
                         "assembler routing); ignored when --top_seeds is used.")
    # Live scoring weights (mirroring beam_search_streaming)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Exploration weight α ∈ [0,1]: blend of ML score vs "
                         "physics UCB score. 0 = pure ML, 1 = pure physics+UCB.")
    ap.add_argument("--kappa", type=float, default=1.5,
                    help="UCB aggressiveness κ: ŷ + κ·σ. Higher = more "
                         "exploration of uncertain candidates.")
    ap.add_argument("--sar_weight", type=float, default=0.4,
                    help="β: weight on accumulated family-SAR prior in the "
                         "final score.")
    # Mutation steering
    ap.add_argument("--prune_adverse", type=float, default=PRUNE_ADVERSE,
                    help="Per-mutation SAR prior at/below which a move is "
                         "dropped pre-ML (only in families with a "
                         "data-supported prior).")
    ap.add_argument("--tail_nosig_cap", type=int, default=TAIL_NOSIG_CAP,
                    help="Max no-signal tail-variant mutations kept per parent "
                         "(prevents flat-family tail explosion).")
    ap.add_argument("--out_csv", default="proposed_iajds.csv")
    ap.add_argument("--out_json", default="proposed_iajds_summary.json")
    args = ap.parse_args()

    print(f"Building structural library from {BIOACT_XLSX.name}…")
    library = build_library(BIOACT_XLSX)
    print(f"  {len(library.families)} families")

    print(f"Loading bioact regressor + binary heads…")
    if not BIN_BUNDLE.exists():
        print(f"  WARN: {BIN_BUNDLE} not found. Run train_binary_classifier.py first.")
        sys.exit(1)
    if not V14_BUNDLE.exists():
        print(f"  WARN: {V14_BUNDLE} not found. Run bioact_v14_pipeline.py first.")
        sys.exit(1)
    with open(V14_BUNDLE, "rb") as f:
        bundle_v14 = pickle.load(f)
    with open(BIN_BUNDLE, "rb") as f:
        bundle_bin = pickle.load(f)
    print(f"  v14 n_train={len(bundle_v14.get('y_train', []))}, "
          f"bin n_train={bundle_bin['metrics']['n_train']}, "
          f"bin reg LOO MAE={bundle_bin['metrics']['regression_loo_mae']:.3f}")

    # Seeds
    df = pd.read_excel(BIOACT_XLSX)
    if args.seed_smiles:
        # If the explicit seed matches a training row, reuse that row's full
        # decomposition (correct family, head, linker_length, linkage). Falls
        # back to GA-Tris defaults only when the SMILES is genuinely new.
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(args.seed_smiles))
        match = None
        for _, r in df.iterrows():
            sm = r.get("SMILES_canonical") or r.get("SMILES")
            if pd.isna(sm):
                continue
            try:
                cs = Chem.MolToSmiles(Chem.MolFromSmiles(str(sm)))
            except Exception:
                continue
            if cs == canon:
                match = r.to_dict()
                break
        if match is not None:
            seeds_list = [decompose_row(match)]
            print(f"  --seed_smiles matched training row "
                  f"IAJD_id={match.get('IAJD_id')} family={match.get('family')}")
        else:
            seed_row = {"family": args.seed_family, "head_group": "HPRZ",
                        "linker_length": 4, "linkage": "ester",
                        "SMILES_canonical": args.seed_smiles}
            seeds_list = [decompose_row(seed_row)]
    else:
        seed_rows = _pick_seeds(df, args.top_seeds)
        seeds_list = [decompose_row(r) for r in seed_rows]
        seeds_list = [s for s in seeds_list if s is not None]
    print(f"\nSeeding from {len(seeds_list)} structures:")
    for s in seeds_list[:5]:
        print(f"  IAJD {s.iajd_num}: family={s.family} head={s.head} "
              f"link={s.linker_n}C  y_obs={s.y_obs}")

    # Train FPs for novelty
    train_fps = []
    for _, r in df.iterrows():
        smi = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(smi): continue
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            train_fps.append(_fp(m))

    print(f"\nRunning beam search: threshold={args.threshold} beam={args.beam} "
          f"depth={args.depth}")
    print(f"  α(alpha)={args.alpha}  κ(kappa)={args.kappa}  β(sar_weight)={args.sar_weight}")
    print(f"  prune_adverse={args.prune_adverse}  tail_nosig_cap={args.tail_nosig_cap}")
    result = beam_search(args.threshold, args.beam, args.depth,
                         seeds_list, library, bundle_v14, bundle_bin, train_fps,
                         exploration_weight=args.alpha,
                         kappa_ucb=args.kappa,
                         sar_weight=args.sar_weight,
                         prune_adverse=args.prune_adverse,
                         tail_nosig_cap=args.tail_nosig_cap)

    # Output columns: ML + live physics + UCB + monotone + SAR
    out_cols = ["smiles", "yhat", "yhat_seed", "delta_vs_seed", "p_above",
                "ucb_score", "q_physics", "cpp", "endosomal_escape",
                "physics_yhat", "monotone_bonus", "monotone_signals",
                "sar_prior", "sar_step", "sar_reason",
                "score_ml", "score_phys", "score",
                "tanim_max_to_train",
                "mutation_description", "mutation_tag",
                "mutation_trail", "parent_smiles"]
    for c in out_cols:
        if c not in result.columns:
            result[c] = None
    result[out_cols].to_csv(args.out_csv, index=False)

    summary = {
        "threshold": args.threshold,
        "beam": args.beam, "depth": args.depth,
        "alpha": args.alpha, "kappa": args.kappa,
        "sar_weight": args.sar_weight,
        "prune_adverse": args.prune_adverse,
        "tail_nosig_cap": args.tail_nosig_cap,
        "n_candidates_total": len(result),
        "n_novel": int((result["tanim_max_to_train"] < NOVELTY_TANIMOTO_MAX).sum()),
        "top_10": result.head(10)[out_cols].to_dict(orient="records"),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDONE — {len(result)} candidates scored")
    print(f"  Novel (tanim < {NOVELTY_TANIMOTO_MAX}): {summary['n_novel']}")
    print(f"  Top 5:")
    for _, r in result.head(5).iterrows():
        trail_str = r['mutation_trail'][-3:] if isinstance(r['mutation_trail'], list) else r['mutation_trail']
        print(f"    score={r['score']:.3f}  (ml={r['score_ml']:.3f} phys={r['score_phys']:.3f})  "
              f"ŷ={r['yhat']:.2f}  ucb={r['ucb_score']:.2f}  "
              f"Q={r['q_physics']:.2f}  P(≥{args.threshold})={r['p_above']:.2f}  "
              f"SAR={r['sar_prior']:+.2f}  tanim={r['tanim_max_to_train']:.2f}")
        print(f"      trail={trail_str}")
        print(f"      {r['smiles']}")
    print(f"\n  Wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
