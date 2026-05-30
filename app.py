"""
app.py — Gradio entrypoint for STRIDE on Hugging Face Spaces.

STRIDE = STRuctural Ranking + Informed Design Engine: a tandem pKa +
bioactivity predictor for ionizable amphiphilic Janus dendrimers (IAJDs),
plus a beam-search proposer that suggests novel candidates steered by
per-family SAR priors.

Wraps `predict()` and `predict_batch()` from iajd_predict.py and renders the
single-SMILES and multi-molecule batch flows that the localhost HTTP server
provides, formatted as Markdown for Gradio.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# Compat shim: older rdkit builds (notably the one pinned in this env) don't
# expose AllChem.GetMorganGenerator. Many of our modules reference it at
# import time, so patch in a thin wrapper that falls back to the legacy API.
try:
    from rdkit.Chem import AllChem as _AllChem_shim
    if not hasattr(_AllChem_shim, "GetMorganGenerator"):
        class _ShimGen:
            def __init__(self, radius, fpSize):
                self.radius = radius; self.fpSize = fpSize
            def GetFingerprint(self, mol):
                return _AllChem_shim.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.fpSize)
            def GetCountFingerprint(self, mol):
                return _AllChem_shim.GetHashedMorganFingerprint(
                    mol, radius=self.radius, nBits=self.fpSize)
        _AllChem_shim.GetMorganGenerator = (
            lambda radius=2, fpSize=2048, **kw: _ShimGen(radius, fpSize)
        )
except Exception:
    pass

# IMPORTANT: load XGBoost bundles BEFORE torch/gradio/chemprop. Once torch
# (or gradio→torch) initializes its OpenMP runtime, deserializing the
# xgboost pickle in the same process can crash. Loading the XGBoost bundle
# first warms the cache and ordering the OpenMP init xgboost→torch is safe.
try:
    from predict_binary import load_binary_bundle, predict_p_above
    _BIN_BUNDLE = load_binary_bundle()
    BIN_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    BIN_AVAILABLE = False
    _BIN_BUNDLE = None
    predict_p_above = None
    print(f"[binary head] not available: {exc}")

# Install chemprop 1.6.1 at startup if not present (HF Spaces can't fit it
# in a Docker image alongside torch, so we install at runtime).
try:
    from chemprop.train.make_predictions import make_predictions as _  # noqa: F401
except ImportError:
    print("Installing chemprop 1.6.1 + deps at startup...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                          "numpy<2", "chemprop==1.6.1", "tensorboard",
                          "hyperopt", "typed-argument-parser"])
    print("chemprop installed.")

# Fix torch.load for chemprop 1.6.1: newer torch defaults to weights_only=True
# which rejects the pickled argparse.Namespace in chemprop checkpoints.
try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

import gradio as gr

from iajd_predict import (
    predict, predict_batch, _load_bundle, ALLOWED_FAMILIES,
)
from iajd_family import CHEMICAL_FAMILIES, BIOACT_ONLY_SUBARCHS

# Deep-ensemble (T2 #5) + Per-organ (T3 #9) — lazy loaded on first use.
# Eager-loading both XGBoost bundles at import time triggers a native
# thread-init crash when the pka-curve XGB is built right after at module init.
ENS_AVAILABLE = True
_ENS_BUNDLE = None
_predict_ensemble = None
_predict_with_p_above_ensemble = None
_p_above_gaussian = None
_assemble_X_binary = None
PER_ORGAN_AVAILABLE = True
_PER_ORGAN_BUNDLE = None
_predict_per_organ = None


def _lazy_load_ensemble():
    """Load deep-ensemble bundle on first use. Returns True on success."""
    global _ENS_BUNDLE, _predict_ensemble, _predict_with_p_above_ensemble
    global _p_above_gaussian, _assemble_X_binary, ENS_AVAILABLE
    if _ENS_BUNDLE is not None:
        return True
    if not ENS_AVAILABLE:
        return False
    try:
        from predict_ensemble import (
            load_ensemble_bundle, predict_ensemble as _pe,
            predict_with_p_above as _pwpa, p_above_gaussian as _pag,
        )
        from predict_binary import _assemble_X as _aX
        _ENS_BUNDLE = load_ensemble_bundle()
        _predict_ensemble = _pe
        _predict_with_p_above_ensemble = _pwpa
        _p_above_gaussian = _pag
        _assemble_X_binary = _aX
        print(f"[ensemble] lazy-loaded — M={_ENS_BUNDLE['metrics']['M']}, "
              f"calibration={_ENS_BUNDLE['sigma_calibration']:.3f}, "
              f"global σ={_ENS_BUNDLE['global_sigma']:.3f}")
        return True
    except Exception as exc:  # noqa: BLE001
        ENS_AVAILABLE = False
        print(f"[ensemble] lazy load failed: {exc}")
        return False


def _lazy_load_per_organ():
    """Load per-organ bundle on first use. Returns True on success."""
    global _PER_ORGAN_BUNDLE, _predict_per_organ, PER_ORGAN_AVAILABLE
    if _PER_ORGAN_BUNDLE is not None:
        return True
    if not PER_ORGAN_AVAILABLE:
        return False
    try:
        from predict_per_organ import load_per_organ_bundle, predict_per_organ as _ppo
        _PER_ORGAN_BUNDLE = load_per_organ_bundle()
        _predict_per_organ = _ppo
        print(f"[per-organ] lazy-loaded — {len(_PER_ORGAN_BUNDLE['organ_models'])} organ models")
        return True
    except Exception as exc:  # noqa: BLE001
        PER_ORGAN_AVAILABLE = False
        print(f"[per-organ] lazy load failed: {exc}")
        return False

# pKa v9.2 three-head blend (analog @ K=5 + pure XGB + debiased MolGpKa)
# with per-family optimal weights. Live MolGpKa GCN runs at inference for
# any new SMILES.
try:
    from predict_pka_v92 import predict_pka_v92, load_v92_bundle
    _PKA_V92_BUNDLE = load_v92_bundle()
    PKA_V92_AVAILABLE = True
    print(f"[pKa v9.2] loaded — blend LOO MAE "
          f"{_PKA_V92_BUNDLE['metrics']['loo_mae_blend']:.4f} on "
          f"{_PKA_V92_BUNDLE['metrics']['n_train']} compounds")
except Exception as exc:  # noqa: BLE001
    PKA_V92_AVAILABLE = False
    _PKA_V92_BUNDLE = None
    predict_pka_v92 = None
    print(f"[pKa v9.2] not available: {exc}")

# Fragment-swap proposer — added 2026-05-28.
try:
    from iajd_grammar import build_library, decompose_row, Seed, humanize_mutation_tag
    import propose_iajds as _propose
    _PROPOSE_LIB = None
    PROPOSE_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    PROPOSE_AVAILABLE = False
    print(f"[proposer] not available: {exc}")


def _mutation_story(trail, max_steps: int = 6) -> str:
    """Humanize the FULL mutation trail (every round) into one readable chain,
    so a multi-round candidate shows its complete derivation from the seed —
    not just the most recent mutation. Single-step candidates render as the
    lone description; multi-step ones are round-labelled, e.g.
    'R1 Linker shortened: 5C → 3C  ▸  R2 Synthetic head added: DEHPRZ'.
    Steps are joined by ' ▸ ' (distinct from the ' → ' inside each step)."""
    if not isinstance(trail, (list, tuple)):
        return ""
    steps = [t for t in trail if t and t != "seed"]
    if not steps:
        return "Original seed (no mutation)"
    labels = [humanize_mutation_tag(t) for t in steps]
    if len(labels) > max_steps:
        labels = labels[:max_steps] + [f"… (+{len(labels) - max_steps} more)"]
    if len(labels) == 1:
        return labels[0]
    return "  ▸  ".join(f"R{i+1} {lab}" for i, lab in enumerate(labels))

# v11 pKa-dominant independent baseline (M2 bundle). Optional — the Space
# still works if the bundle file isn't shipped.
try:
    from v11_pka_predictor import (
        predict_v11_bioact, tail_descriptors_from_mol, is_available as _v11_ok,
    )
    V11_AVAILABLE = _v11_ok()
except Exception:  # noqa: BLE001
    V11_AVAILABLE = False
    predict_v11_bioact = None
    tail_descriptors_from_mol = None

# Warm the bundle at import — HF Spaces shares the process across requests, so
# this one-time ~10s cost is amortized.  Wrapped in try/except so import-time
# errors don't kill the whole Space.
try:
    _load_bundle()
    BUNDLE_OK = True
    BUNDLE_ERR = None
except Exception as exc:  # noqa: BLE001
    BUNDLE_OK = False
    BUNDLE_ERR = f"{type(exc).__name__}: {exc}"

# pKa-flux curve model (fitted quadratic + structural corrector; pKa input carries indirect similarity from v9.1)
_PKA_CURVE_READY = False
_PKA_CURVES = None
_PKA_CURVE_XGB = None
_PKA_CURVE_MEDIANS = None
try:
    import numpy as _np
    from scipy.optimize import curve_fit as _curve_fit
    import xgboost as _xgb_mod
    from pka_primary_model import (
        extract_structural_features as _extract_sf,
        STRUCT_FEATURE_NAMES as _SF_NAMES,
        fit_pka_flux_curves as _fit_curves,
        predict_curve as _pred_curve,
        RESIDUAL_XGB_HP as _RES_HP,
    )
    import pandas as _pd_curve

    # Novel GA-Tris IAJDs (347, 348, 365, 366, 367, 369, 372, 373) reintegrated 2026-05-28.
    _bio_c = _pd_curve.read_excel(HERE / "IAJD_master" / "datasets" / "IAJD_Bioact_v13_clean.xlsx")
    _pka_c = _pd_curve.read_csv(HERE / "v11_pka_flux" / "predicted_pka_cache.csv")
    _mg = _bio_c.merge(_pka_c[["row_id", "predicted_pKa"]], on="row_id", how="left")
    _vd = _mg.dropna(subset=["log10_flux_total", "predicted_pKa"])
    _fams = _vd["family"].values
    _pkas = _vd["predicted_pKa"].values
    _fluxes = _vd["log10_flux_total"].values
    _feat_r = []
    for _, _r in _vd.iterrows():
        _s = _r.get("SMILES_canonical") or _r.get("SMILES")
        _feat_r.append(_extract_sf(str(_s)) if _pd_curve.notna(_s) else {k: _np.nan for k in _SF_NAMES})
    # No-proxy: train XGBoost on the raw NaN-bearing matrix so the model
    # learns real default-branch behavior; do NOT median-fill at training.
    _Xs = _pd_curve.DataFrame(_feat_r)[_SF_NAMES].values
    _PKA_CURVES = _fit_curves(_pkas, _fluxes, _fams)
    _cp = _np.array([_pred_curve(p, f, _PKA_CURVES) for p, f in zip(_pkas, _fams)])
    _res = _fluxes - _cp
    _Xf = _np.column_stack([_pkas, _Xs])
    _PKA_CURVE_XGB = _xgb_mod.XGBRegressor(**_RES_HP)
    _PKA_CURVE_XGB.fit(_Xf, _res, verbose=False)
    _PKA_CURVE_READY = True
except Exception:
    pass

def _predict_pka_curve(pred_pka, family, smiles):
    """pKa-flux curve + XGBoost structural residual.

    No-proxy (audit 2026-05-29): missing structural features pass NaN through
    to XGBoost's default branch (which was learned on the same-shape NaN-aware
    training matrix). We do not median-fill or zero-fill at inference.
    """
    curve_val = _pred_curve(pred_pka, family, _PKA_CURVES)
    feats = _extract_sf(smiles)
    x = _np.array(
        [[pred_pka] + [feats.get(f, float("nan")) for f in _SF_NAMES]],
        dtype=float,
    )
    resid = float(_PKA_CURVE_XGB.predict(x)[0])
    return curve_val + resid

_FAMILY_OPTIONS = ["(auto-detect)"] + sorted(ALLOWED_FAMILIES)


def _log_with_sci(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:  # noqa: BLE001
        return str(v)
    linear = 10 ** v
    return f"{v:.3f} (10^={linear:.2e})"


def _ci_with_sci(arr):
    if not arr or len(arr) != 2:
        return "—"
    lo, hi = float(arr[0]), float(arr[1])
    return f"[{lo:.3f} (10^={10**lo:.2e}), {hi:.3f} (10^={10**hi:.2e})]"


def _render_result_markdown(r: dict, idx: int = 0) -> str:
    if not r:
        return "_(no result)_"
    if r.get("error"):
        return f"### Error: {r.get('input_label') or 'mol ' + str(idx+1)}\n\n**Error:** {r['error']}"

    label = r.get("input_label") or f"mol {idx + 1}"
    pka = r.get("pka") or {}
    bio = r.get("bioactivity") or {}
    organ = r.get("organ_delivery") or {}
    fr = r.get("family_resolution") or {}
    det = fr.get("detection") or {}
    cands = det.get("candidates") or []
    nb = r.get("neighbors") or {}
    stk = bio.get("stacker") or {}
    v11 = r.get("bioactivity_v11_pka_dominant") or {}

    md = []
    md.append(f"### `{label}`")
    md.append(f"`{r.get('canonical_smiles', '')}`")
    md.append("")
    fam = fr.get('family_assigned') or '—'
    md.append(f"**Family:** `{fam}` (confidence {det.get('confidence', 0)*100:.0f}%)")
    md.append("")

    # ── pKa (v9.2) ──
    v92 = r.get("pka_v92") or {}
    if v92 and "pKa_pred" in v92 and v92["pKa_pred"] is not None:
        md.append(f"### pKa = {v92['pKa_pred']:.3f}")
    else:
        md.append(f"### pKa = {pka.get('point', '—')}")
    if pka.get("ci_90"):
        md.append(f"90% CI: [{pka['ci_90'][0]:.3f}, {pka['ci_90'][1]:.3f}]")
    md.append("")

    # ── Bioactivity (v14 + adaptive stacker only) ──
    is_lookup = bio.get("source") == "training_set_exact_match"
    bio_point = bio.get("point")

    if is_lookup:
        iajd_id = bio.get("iajd_id_if_measured", "")
        md.append(f"### log₁₀ flux = {_log_with_sci(bio_point)}")
        md.append(f"Measured value from training set (IAJD {iajd_id}, Tanimoto = 1.000).")
    else:
        md.append(f"### log₁₀ flux = {_log_with_sci(bio_point)}")
        if bio.get("ci_90"):
            md.append(f"90% CI: {_ci_with_sci(bio['ci_90'])}")
        if bio.get("max_tanimoto") is not None:
            md.append(f"Max Tanimoto to training: {bio['max_tanimoto']:.3f}")
    md.append("")

    # ── Binary threshold verdict — derived from the deep ensemble's per-candidate
    #    σ (T2 #5). Falls back to lookup match or v14+stacker σ=0.43 if ensemble
    #    not loaded.
    ens = r.get("ensemble") or {}
    ens_bin = r.get("binary_above_threshold_ensemble") or {}
    bin_info = r.get("binary_above_threshold") or {}
    T = (ens_bin.get("threshold") if ens_bin else None) or bin_info.get("threshold")
    if T is not None and bio_point is not None:
        import math
        if ens_bin and ens_bin.get("p_above") is not None:
            p = float(ens_bin["p_above"])
            verdict_source = ens_bin.get("source", "ensemble Gaussian")
            if not is_lookup:
                verdict_source = (
                    f"deep ensemble (M={ens.get('M','?')}), σ_eff = {ens.get('sigma_eff','?')} "
                    f"(per-candidate σ_q={ens.get('sigma_query','?')}, "
                    f"per-family σ={ens.get('sigma_family','?')})"
                )
            verdict = "above" if p >= 0.5 else "below"
            md.append(f"### P(log₁₀ flux ≥ {T:.2f}) = {p:.1%} — likely {verdict}")
            md.append(f"_{verdict_source}_")
        elif is_lookup:
            p = 1.0 if bio_point >= T else 0.0
            verdict = "above" if p >= 0.5 else "below"
            md.append(f"### P(log₁₀ flux ≥ {T:.2f}) = {p:.1%} — likely {verdict}")
            md.append(f"_from measured value_")
        else:
            # No proxy: ensemble bundle not loaded → no honest σ available, so
            # refuse to fabricate a Gaussian P(≥T). Report point estimate only.
            md.append(f"### P(log₁₀ flux ≥ {T:.2f}) = (unavailable)")
            md.append("_Ensemble uncertainty model not loaded; cannot compute P(≥T) "
                      "without a real σ. Point estimate above is the v14+stacker ŷ._")
        if ens.get("ood_warning"):
            md.append(f"_⚠ OOD: {ens['ood_warning']}_")
        md.append("")

    # ── Per-organ ML predictions (T3 #9): one XGB per organ, real ML signal.
    per_organ_ml = r.get("per_organ_ml") or {}
    if per_organ_ml and not per_organ_ml.get("error"):
        md.append("### Per-organ ML predictions")
        md.append("| organ | log₁₀ flux | σ_family | LOO MAE | n_train |")
        md.append("|---|---|---|---|---|")
        organ_order = ["lung", "liver", "spleen", "LN"]
        for organ in organ_order:
            info = per_organ_ml.get(organ)
            if info is None:
                continue
            md.append(
                f"| {organ} | {info['yhat']:.2f} | {info['sigma_family']:.2f} | "
                f"{info['loo_mae']:.3f} | {info['n_train']} |"
            )
        md.append("_Separate XGBoost regressor per organ on the same 92-feature stack._")
        md.append("")
    elif organ.get("target_organ"):
        # Fall back to the legacy similarity-weighted lookup table if ML not loaded.
        md.append(f"### Per-organ flux (similarity-weighted, not ML)")
        md.append("| organ | partition | log10 flux |")
        md.append("|---|---|---|")
        parts = organ.get("partition_pct") or {}
        lfs = organ.get("log10_flux_by_organ") or {}
        for k in sorted(parts, key=lambda x: -parts[x]):
            md.append(f"| {k} | {parts[k]}% | {_log_with_sci(lfs.get(k))} |")
        md.append("")

    # Neighbors (collapsed)
    if nb.get("pka") or nb.get("bioact"):
        md.append("### Nearest training neighbors")
        if nb.get("pka"):
            md.append("**pKa:** " + ", ".join(
                f"{n.get('iajd_id')} (Tan={n.get('tanimoto',0):.2f}, pKa={n.get('pKa')})"
                for n in nb["pka"][:3]))
        if nb.get("bioact"):
            md.append("**Bioact:** " + ", ".join(
                f"{n.get('iajd_id')} (Tan={n.get('tanimoto',0):.2f}, flux={n.get('log10_flux_total','?')})"
                for n in nb["bioact"][:3]))
        md.append("")

    if r.get("warnings"):
        md.append("_Warnings: " + "; ".join(r["warnings"]) + "_")
    return "\n".join(md)


def _attach_v11(r: dict) -> dict:
    """Enrich a single predict() result with v11 pKa-dominant + pKa-flux curve."""
    pka = r.get("pka") or {}
    if pka.get("point") is None:
        return r
    fam = (r.get("family_resolution") or {}).get("family_assigned") or \
           r.get("family_used")
    smi = r.get("canonical_smiles")

    # v11 M2 pKa-dominant
    if V11_AVAILABLE and predict_v11_bioact is not None:
        tail = None
        if smi:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smi)
                if mol is not None and tail_descriptors_from_mol is not None:
                    tail = tail_descriptors_from_mol(mol)
            except Exception:
                tail = None
        try:
            v11_out = predict_v11_bioact(
                canonical_smiles=smi or "", family=fam or "",
                predicted_pKa=float(pka["point"]), tail_descriptors=tail,
            )
            r["bioactivity_v11_pka_dominant"] = v11_out
        except Exception as exc:
            r["bioactivity_v11_pka_dominant"] = {"error": str(exc)}

    # pKa-flux curve (flux step has no similarity; pKa input carries indirect similarity)
    if smi and _PKA_CURVE_READY:
        try:
            pred_pka = float(pka["point"])
            curve_pred = _predict_pka_curve(pred_pka, fam or "GA-Tris", smi)
            r["bioactivity_pka_curve"] = {"point": round(curve_pred, 3)}
        except Exception as exc:
            r["bioactivity_pka_curve"] = {"error": str(exc)}

    return r


def _attach_pka_v92(r: dict) -> dict:
    """Run the v9.2 three-head pKa blend (live MolGpKa + analog K=5 + XGB)."""
    if not PKA_V92_AVAILABLE or predict_pka_v92 is None:
        return r
    smi = r.get("canonical_smiles")
    if not smi:
        return r
    fam = (r.get("family_resolution") or {}).get("family_assigned")
    try:
        v92 = predict_pka_v92(smi, family_hint=fam)
        r["pka_v92"] = v92
    except Exception as exc:
        r["pka_v92"] = {"error": str(exc)}
    return r


def _attach_binary(r: dict, threshold: float) -> dict:
    """Add binary classifier P(≥threshold) to a predict() result."""
    if not BIN_AVAILABLE or predict_p_above is None:
        return r
    smi = r.get("canonical_smiles")
    if not smi:
        return r
    fam = (r.get("family_resolution") or {}).get("family_assigned") or "GA-Tris"
    try:
        yhat, p = predict_p_above(smi, threshold, _BIN_BUNDLE, family_hint=fam)
        r["binary_above_threshold"] = {
            "threshold": float(threshold),
            "yhat_regressor": round(float(yhat), 3),
            "p_above": round(float(p), 4),
            "decision": bool(p >= 0.5),
        }
    except Exception as exc:
        r["binary_above_threshold"] = {"error": str(exc)}
    return r


def _attach_ensemble_and_per_organ(
    r: dict,
    threshold: float,
    sample_prep: dict | None = None,
) -> dict:
    """T2 #5 + T3 #9: attach per-candidate ensemble σ and per-organ predictions.

    Block E (pH_sample, T_hours, inj_route) is threaded into the 92-feature
    stack via `sample_prep` so the model sees the actual experimental
    conditions instead of NaN.

    Adds three keys to `r`:
      - "ensemble":   M-model mean, per-candidate σ, per-family σ
      - "per_organ":  lung/liver/spleen/LN predictions (each with σ_family)
      - "binary_above_threshold_ensemble": Gaussian P(≥T) using ensemble σ
    """
    if not _lazy_load_ensemble():
        return r
    smi = r.get("canonical_smiles") or r.get("smiles")
    if not smi:
        return r
    fam = (r.get("family_resolution") or {}).get("family_assigned") or "GA-Tris"
    try:
        X = _assemble_X_binary([smi], family_hint=fam, sample_prep=sample_prep)
    except Exception as exc:  # noqa: BLE001
        r["ensemble"] = {"error": f"X assembly failed: {exc}"}
        return r

    # Max-Tanimoto to training (for OOD flag) — reuse what attach_binary already
    # computed if available; otherwise compute now.
    max_tan = (r.get("bioactivity") or {}).get("max_tanimoto")
    try:
        ens = _predict_ensemble(X, family=fam)
    except Exception as exc:  # noqa: BLE001
        r["ensemble"] = {"error": f"ensemble inference failed: {exc}"}
        return r
    r["ensemble"] = {
        "yhat_mean": round(ens["yhat_mean"], 3),
        "sigma_query": round(ens["sigma_query"], 3),
        "sigma_family": round(ens["sigma_family"], 3),
        "sigma_eff": round(ens["sigma_eff"], 3),
        "calibration": round(ens["calibration"], 3),
        "M": ens["M"],
        "family": fam,
        "max_tanimoto_to_training": (
            round(float(max_tan), 3) if max_tan is not None and max_tan == max_tan else None
        ),
        "ood_warning": (
            f"max Tanimoto to training = {max_tan:.2f} (< 0.50): outside the model's "
            "training neighborhood; reported σ likely underestimates true uncertainty."
        ) if (max_tan is not None and max_tan == max_tan and float(max_tan) < 0.5) else None,
    }

    # Gaussian P(≥T) using ensemble's per-candidate σ_eff (replaces the
    # hardcoded σ=0.43 binary head).
    bio = r.get("bioactivity") or {}
    is_lookup = bio.get("source") == "training_set_exact_match"
    bio_point = bio.get("point")
    if is_lookup and bio_point is not None:
        p_ens = 1.0 if float(bio_point) >= float(threshold) else 0.0
        r["binary_above_threshold_ensemble"] = {
            "threshold": float(threshold),
            "p_above": p_ens,
            "sigma_used": 0.0,
            "source": "from measured value",
        }
    else:
        p_ens = _p_above_gaussian(ens["yhat_mean"], ens["sigma_eff"], float(threshold))
        r["binary_above_threshold_ensemble"] = {
            "threshold": float(threshold),
            "p_above": round(float(p_ens), 4) if p_ens == p_ens else None,
            "sigma_used": round(ens["sigma_eff"], 3),
            "source": "deep ensemble (M={}) + per-family σ + calibration ×{:.2f}".format(
                ens["M"], ens["calibration"]),
        }

    # Per-organ predictions (lazy load)
    if _lazy_load_per_organ():
        try:
            organ_preds = _predict_per_organ(X, family=fam)
            r["per_organ_ml"] = {
                organ: {
                    "yhat": round(info["yhat"], 3),
                    "sigma_family": round(info["sigma_family"], 3),
                    "loo_mae": round(info["loo_mae"], 3),
                    "n_train": info["n_train"],
                }
                for organ, info in organ_preds.items()
            }
        except Exception as exc:  # noqa: BLE001
            r["per_organ_ml"] = {"error": str(exc)}
    return r


def _parse_optional_float(x, lo: float | None = None, hi: float | None = None):
    """Parse a Textbox-style optional float. Blank / whitespace / non-numeric
    returns None (downstream uses NaN → XGBoost default branch). Out-of-range
    also returns None rather than raising — UI hint should already discourage
    this, and we'd rather route to the default branch than crash a prediction."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if v != v:    # NaN
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def single_predict(smiles: str, family_choice: str, neighbors: int,
                    threshold: float = 8.0,
                    pH_sample=None,
                    T_hours=None,
                    inj_route: str | None = None):
    if not smiles or not smiles.strip():
        return "_Enter a SMILES._", ""
    family = None if (not family_choice or family_choice == "(auto-detect)") else family_choice
    try:
        r = predict(smiles.strip(), family=family, neighbors=int(neighbors))
    except Exception as exc:  # noqa: BLE001
        return f"### Prediction failed\n\n```\n{type(exc).__name__}: {exc}\n```", ""
    _attach_v11(r)
    _attach_pka_v92(r)
    _attach_binary(r, float(threshold))
    # Build sample-prep dict from UI inputs. Blank / out-of-range → None →
    # XGBoost default branch (equivalent to the pre-Block-E model).
    sample_prep = {
        "pH_sample": _parse_optional_float(pH_sample, lo=4.0, hi=9.0),
        "T_hours":   _parse_optional_float(T_hours,   lo=0.0, hi=24.0),
        "inj_route": inj_route if inj_route and inj_route != "(unspecified)" else None,
    }
    r["sample_prep_inputs"] = sample_prep
    _attach_ensemble_and_per_organ(r, float(threshold), sample_prep=sample_prep)
    md = _render_result_markdown(r, 0)
    raw = json.dumps(r, indent=2, default=str)
    return md, raw


def propose_better(seed_smiles: str, threshold: float, beam: int, depth: int,
                    top_seeds: int, exploration_weight: float = 0.0,
                    kappa_ucb: float = 1.5, sar_weight: float = 0.4,
                    rank_by: str = "score (default)"):
    """Run the fragment-swap proposer; render top candidates as Markdown.

    exploration_weight (α ∈ [0,1]):
        0 → pure ML scoring: P(≥T) × max(0, ŷ_ML − ŷ_seed_ML)
        1 → pure physics scoring: Q_physics × max(0, ŷ_physics − ŷ_seed_physics)
             + monotone-axis bonus. ŷ_physics is the v15 Ridge prediction on
             real CPP / endosomal-escape / HLB / logKp / Manning features
             (NO ML term — pure physics at α=1, audited 2026-05-29).
        in-between blends the two — useful when ML is bounded by training
        distribution and you want extrapolation hypotheses surfaced

    rank_by (T3 #8 active-learning surface):
        "score (default)"          — combined ML + physics + SAR score
        "expected_improvement (…)" — EI under N(ŷ, σ²) vs T; picks
                                      candidates likely above T AND with σ
                                      large enough that a real measurement
                                      is informative
        "sigma_ensemble (…)"        — pure exploration; picks the candidates
                                      the ensemble disagrees most about
    """
    if not PROPOSE_AVAILABLE:
        return "_Proposer module unavailable._", ""
    global _PROPOSE_LIB
    if _PROPOSE_LIB is None:
        _PROPOSE_LIB = build_library(HERE / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")

    import pandas as _pd
    df_bio = _pd.read_excel(HERE / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx")
    seeds = []
    if seed_smiles and seed_smiles.strip():
        from rdkit import Chem as _Chem
        m = _Chem.MolFromSmiles(seed_smiles.strip())
        if m is None:
            return f"_Could not parse SMILES: {seed_smiles}_", ""
        canon = _Chem.MolToSmiles(m)
        # Find a matching training row for family/head/linker context
        row = None
        for _, r in df_bio.iterrows():
            sm = r.get("SMILES_canonical") or r.get("SMILES")
            if _Chem.MolToSmiles(_Chem.MolFromSmiles(str(sm))) == canon:
                row = r.to_dict()
                break
        if row is None:
            # Synthesize a generic GA-Tris row
            row = {"family": "GA-Tris", "head_group": "HPRZ",
                   "linker_length": 4, "linkage": "ester",
                   "SMILES_canonical": canon}
        seed = decompose_row(row)
        if seed is not None:
            seeds = [seed]
    if not seeds:
        # default to top-K winners
        rows = (df_bio.dropna(subset=["log10_flux_total"])
                       .sort_values("log10_flux_total", ascending=False)
                       .head(int(top_seeds)))
        for _, r in rows.iterrows():
            s = decompose_row(r.to_dict())
            if s: seeds.append(s)
    if not seeds:
        return "_No usable seeds found._", ""

    # Train fingerprints for novelty
    train_fps = []
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AC
    try:
        _gen = _AC.GetMorganGenerator(radius=2, fpSize=2048)
        _fp_fn = lambda m: _gen.GetFingerprint(m)
    except AttributeError:
        _fp_fn = lambda m: _AC.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    for _, r in df_bio.iterrows():
        sm = r.get("SMILES_canonical") or r.get("SMILES")
        if _pd.isna(sm): continue
        m = _Chem.MolFromSmiles(str(sm))
        if m is not None:
            train_fps.append(_fp_fn(m))

    import pickle as _pickle
    with open(HERE / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl", "rb") as f:
        bundle_v14 = _pickle.load(f)

    # Resolve the rank-by selection to a DataFrame column for sorting
    if rank_by.startswith("expected_improvement"):
        sort_col, sort_label = "expected_improvement", "EI (active learning)"
    elif rank_by.startswith("sigma_ensemble"):
        sort_col, sort_label = "sigma_ensemble", "σ_ensemble (pure exploration)"
    elif rank_by.startswith("chemberta_novelty"):
        sort_col, sort_label = "chemberta_novelty", "ChemBERTa structural novelty"
    else:
        sort_col, sort_label = "score", "blended score"

    def _render(result_df, round_idx: int, is_final: bool):
        if is_final:
            header = f"## Proposed IAJDs — DONE"
        else:
            header = f"## Proposed IAJDs — live (round {round_idx}/{depth})"
        rank_desc = (f"α={exploration_weight:.2f}·[`P(≥T)·Δ_ML` ⊕ `Q_phys·UCB`] "
                      f"(κ={kappa_ucb:.2f})  +  β={sar_weight:.2f}·`Δ_SAR` "
                      f"(family informed-mutation prior)")
        # Re-sort the displayed top-20 by the user-selected rank-by criterion.
        # The DataFrame upstream is sorted by `score`; we honor that as default.
        if sort_col in result_df.columns and sort_col != "score":
            view_df = result_df.sort_values(sort_col, ascending=False, na_position="last")
        else:
            view_df = result_df
        body = [header,
                f"_T={threshold}, beam={beam}, depth={depth}, "
                f"{len(seeds)} seed(s); composite score = {rank_desc}. "
                f"Top-20 sorted by **{sort_label}**. "
                f"{len(result_df)} candidates scored._",
                ""]
        body.append(
            "| rank | Δ vs seed | Δ_SAR | ŷ ML | σ_ens | EI | ChemB_nov | seed ŷ | P(≥T) | Q_phys | escape | Tanim | Mutations — full derivation (all rounds) · latest SAR why | SMILES |"
        )
        body.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(view_df.head(20).iterrows()):
            # Full derivation: every mutation across all rounds, not just the last.
            desc = (_mutation_story(r.get("mutation_trail"))
                    or r.get("mutation_description") or "—")
            sar_why = r.get("sar_reason")
            if isinstance(sar_why, str) and sar_why.strip():
                desc = f"{desc} · _latest: {sar_why}_"
            def _fmt(v, d=2):
                if v is None or v != v:
                    return "—"
                return f"{v:.{d}f}"
            delta = r.get("delta_vs_seed", 0)
            delta_s = f"{'+' if delta >= 0 else ''}{delta:.2f}" if delta == delta else "—"
            sar = r.get("sar_prior")
            sar_s = f"{'+' if sar >= 0 else ''}{sar:.2f}" if (sar is not None and sar == sar) else "—"
            seed_y = r.get("yhat_seed")
            seed_y_s = f"{seed_y:.2f}" if seed_y == seed_y else "—"
            tanim = r.get("tanim_max_to_train")
            tanim_s = f"{tanim:.2f}" if tanim == tanim else "—"
            q_p = r.get("q_physics")
            esc = r.get("endosomal_escape")
            sig = r.get("sigma_ensemble")
            ei = r.get("expected_improvement")
            nov = r.get("chemberta_novelty")
            body.append(
                f"| {i+1} | **{delta_s}** | {sar_s} | {r['yhat']:.2f} | "
                f"{_fmt(sig)} | {_fmt(ei)} | {_fmt(nov)} | {seed_y_s} | "
                f"{r['p_above']:.0%} | {_fmt(q_p)} | "
                f"{_fmt(esc)} | {tanim_s} | {desc} | `{r['smiles']}` |"
            )
        if is_final:
            body.append("")
            tan_mask = result_df["tanim_max_to_train"] < 0.85
            body.append(f"_Total scored: {len(result_df)}; "
                          f"novel (Tanim < 0.85): {int(tan_mask.sum())}_")
        csv_df = result_df.head(50).copy()
        if "mutation_trail" in csv_df.columns:
            # Readable full-derivation column alongside the raw trail.
            csv_df.insert(0, "mutation_story", csv_df["mutation_trail"].apply(_mutation_story))
        csv = csv_df.to_csv(index=False)
        return "\n".join(body), csv

    last_md, last_csv = "_Running…_", ""
    try:
        result_iter = _propose.beam_search_streaming(
            float(threshold), int(beam), int(depth), seeds, _PROPOSE_LIB,
            bundle_v14, _BIN_BUNDLE, train_fps,
            exploration_weight=float(exploration_weight),
            kappa_ucb=float(kappa_ucb),
            sar_weight=float(sar_weight),
        )
        last_partial_df = None
        for round_idx, partial_df in result_iter:
            last_partial_df = partial_df
            last_md, last_csv = _render(partial_df, round_idx, is_final=False)
            yield last_md, last_csv
        # Final render with DONE header on the last DataFrame we got. Re-render
        # the in-memory DataFrame directly (no CSV round-trip, which would
        # stringify mutation_trail lists and coerce empty sar_reason to NaN).
        if last_partial_df is not None and len(last_partial_df):
            md_final, csv_final = _render(last_partial_df, depth, is_final=True)
            yield md_final, csv_final
    except Exception as exc:
        yield f"### Proposer failed\n\n```\n{type(exc).__name__}: {exc}\n```", last_csv
        return


def batch_predict(file_obj, smiles_text: str, family_choice: str, neighbors: int):
    family = None if (not family_choice or family_choice == "(auto-detect)") else family_choice
    content = None; filename = None
    if file_obj is not None:
        try:
            with open(file_obj.name, "rb") as f:
                content = f.read()
            filename = Path(file_obj.name).name
        except Exception as exc:  # noqa: BLE001
            return f"_File read failed: {exc}_", ""
    elif smiles_text and smiles_text.strip():
        content = smiles_text
        filename = "pasted.smi"
    else:
        return "_Upload a file or paste SMILES._", ""
    try:
        rb = predict_batch(content=content, filename=filename,
                            family=family, neighbors=int(neighbors))
    except Exception as exc:  # noqa: BLE001
        return f"### Batch failed\n\n```\n{type(exc).__name__}: {exc}\n```", ""
    parts = [f"### Parsed {rb.get('n_inputs', 0)} structures from `{rb.get('source')}`"]
    if rb.get("errors"):
        for e in rb["errors"]:
            parts.append(f"- ❌ `{e.get('label')}`: {e.get('error')}")
    for i, r in enumerate(rb.get("results", [])):
        _attach_v11(r)
        parts.append("\n---\n")
        parts.append(_render_result_markdown(r, i))
    raw = json.dumps(rb, indent=2, default=str)
    return "\n".join(parts), raw


def build_ui() -> gr.Blocks:
    intro = """
# STRIDE

**ST**ructural **R**anking + **I**nformed **D**esign **E**ngine. Predicts **pKa** and **bioactivity (log₁₀ total flux)** for ionizable amphiphilic Janus dendrimers from molecular input, and proposes novel IAJDs via beam search over structural mutations.

| Property | Model | LOO MAE (95% bootstrap CI) |
|---|---|---|
| pKa | v9.2 three-head blend (analog + XGB + **hardened-subprocess** live MolGpKa) | 0.125 [0.110, 0.141] (n=278) |
| log₁₀ flux (direct) | v14, 92 features inc. sample-prep | 0.443 [0.405, 0.486] (n=247) |
| log₁₀ flux (stacker) | v14 + adaptive stacker (5-fold CV) | 0.386 |
| Per-candidate σ | Deep ensemble (M=7 bootstrap-XGBs), calibrated ×2.61 | per query |
| Per-organ flux | One XGBoost regressor per organ | lung 0.642, liver 0.493, spleen 0.477, LN 0.564 |
| Structural novelty | ChemBERTa-77M-MTR cosine distance (proposer table column) | per query |

**Input formats:** SMILES, ChemDraw (.cdxml), SDF, MOL. Family auto-detected from six chemical families.

**Tabs:**
- *Single SMILES* — predict one molecule
- *Batch* — file upload or multi-line SMILES paste
- *Propose better IAJDs* — beam search over single-step structural mutations of a seed
"""

    with gr.Blocks(title="STRIDE") as demo:
        gr.Markdown(intro)
        if not BUNDLE_OK:
            gr.Markdown(f"**Bundle load failed at startup.** {BUNDLE_ERR}")

        with gr.Tab("Single SMILES"):
            with gr.Row():
                smiles_in = gr.Textbox(
                    label="SMILES",
                    value="CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1",
                    lines=2,
                )
            with gr.Row():
                fam = gr.Dropdown(_FAMILY_OPTIONS, value="(auto-detect)", label="Family (optional)")
                neighbors = gr.Number(value=5, label="Neighbors", precision=0, minimum=1, maximum=20)
                threshold = gr.Slider(
                    minimum=6.5, maximum=9.5, step=0.25, value=8.0,
                    label="Bioactivity threshold T (log10 flux); model returns P(≥ T)"
                )
                go = gr.Button("Predict", variant="primary")
            with gr.Accordion(
                "Sample-prep conditions (Block E, optional — leave blank to use NaN)",
                open=False,
            ):
                gr.Markdown(
                    "_The bioactivity model was retrained with sample-prep covariates "
                    "(pH at preparation, ageing time, injection route). Leaving these "
                    "blank routes XGBoost through its default NaN branch, equivalent to "
                    "the pre-Block-E model. Set them to the actual experimental "
                    "conditions for an honest, condition-specific prediction._"
                )
                with gr.Row():
                    pH_in = gr.Textbox(
                        label="pH at sample prep (4–9)",
                        value="", placeholder="(blank = unspecified)",
                    )
                    T_in = gr.Textbox(
                        label="Ageing time T (hours, 0–24)",
                        value="", placeholder="(blank = unspecified)",
                    )
                    route_in = gr.Dropdown(
                        ["(unspecified)", "intravenous", "retro-orbital"],
                        value="(unspecified)", label="Injection route",
                    )
            out_md = gr.Markdown()
            with gr.Accordion("Raw JSON result", open=False):
                out_json = gr.Code(language="json")
            go.click(
                single_predict,
                inputs=[smiles_in, fam, neighbors, threshold, pH_in, T_in, route_in],
                outputs=[out_md, out_json],
            )

        with gr.Tab("Batch (file upload or multi-SMILES paste)"):
            gr.Markdown(
                "Upload a **ChemDraw (.cdxml)**, **MOL**, **SDF**, or **SMILES (.smi/.txt)** "
                "file with one or many IAJDs — *family auto-detected per molecule*. "
                "You can also paste multi-line SMILES with optional per-row family hints: "
                "`<SMILES> [family=<FAMILY>] [<label>]`."
            )
            with gr.Row():
                upfile = gr.File(label="File (.cdxml / .mol / .sdf / .smi)",
                                  file_types=[".cdxml", ".cdx", ".mol", ".molfile",
                                                ".sdf", ".smi", ".txt"])
                fam_b = gr.Dropdown(_FAMILY_OPTIONS, value="(auto-detect)",
                                     label="Family (fallback only)")
                neighbors_b = gr.Number(value=5, label="Neighbors", precision=0,
                                          minimum=1, maximum=20)
            smiles_text = gr.Textbox(
                label="…or paste SMILES (one per line)",
                placeholder="# comment lines and blanks are ignored\nCCCCCCCCOCC(...)CCN1CCN(CCO)CC1  IAJD-119\nCCCCCCCCCCOCC(...)CCN1CCN(CCO)CC1  family=PE-Tris IAJD-120",
                lines=6,
            )
            go_b = gr.Button("Predict batch", variant="primary")
            out_md_b = gr.Markdown()
            with gr.Accordion("Raw JSON result", open=False):
                out_json_b = gr.Code(language="json")
            go_b.click(batch_predict, inputs=[upfile, smiles_text, fam_b, neighbors_b],
                       outputs=[out_md_b, out_json_b])

        if PROPOSE_AVAILABLE and BIN_AVAILABLE:
            with gr.Tab("Propose better IAJDs"):
                gr.Markdown(
                    "Search for IAJDs predicted to clear a target bioactivity threshold. "
                    "Applies expanded-grammar structural mutations (head/linker/tail swaps, "
                    "tail extension, cross-family jumps, synthetic head variants) and ranks "
                    "by a blend of **ML score** (`P(≥T) · Δ vs seed`) and **physics UCB** "
                    "(`Q_physics · (ŷ + κ·σ − seed_ŷ)`) — the `α` slider controls the "
                    "balance.\n\n"
                    "• **α = 0**: pure ML. Score = P(≥T) × max(0, ŷ_ML − seed_ŷ_ML).\n"
                    "• **α = 1**: PURE physics. Score = Q_physics × max(0, ŷ_physics − seed_ŷ_physics) "
                    "  + monotone-axis bonus. ŷ_physics is the v15 Ridge prediction on real CPP, "
                    "  endosomal escape, HLB-Griffin, log-Kp_membrane, Manning condensation — all "
                    "  per-molecule live (3D ETKDG head area, live MolGpKa pKa, real RDKit "
                    "  descriptors). **No ML term appears in the α=1 score** (audited 2026-05-29). "
                    "  Useful when the seed is at the top of the training distribution and ML "
                    "  can't see anything above it.\n"
                    "• **κ** = exploration aggressiveness (UCB constant). 0 = no uncertainty "
                    "  boost, 2-3 = aggressive extrapolation.\n"
                    "• **β (SAR prior)** = informed-mutation steering. Each mutation gets a "
                    "  signed `Δ_SAR` prior — the training set's expected change in "
                    "  log₁₀-flux for moving along that family's *significant, de-correlated* "
                    "  structure-activity axes (e.g. GA-Tris: shorter linker ↑, more H-bond "
                    "  acceptors ↑, HPRZ/H2EPRZ heads ≫ MPRZ; PE-Tris: less lipophilic ↑, "
                    "  more H-bond donors ↑). Favourable moves are surfaced even when the "
                    "  (conservative) ML regressor won't extrapolate to them; strongly-adverse "
                    "  moves are pruned before scoring. Families with no statistically-supported "
                    "  axis (PE-Gallic, Dialkoxybenzyl) get **no** steering — an honest neutral, "
                    "  not a guess. Priors are computed by `analyze_family_sar.py`."
                )
                with gr.Row():
                    seed_in = gr.Textbox(
                        label="Seed SMILES (blank ⇒ top training IAJDs)",
                        value="",
                        lines=2,
                    )
                with gr.Row():
                    p_threshold = gr.Slider(6.5, 9.5, value=8.0, step=0.25,
                                              label="Threshold T")
                    p_beam = gr.Slider(5, 30, value=10, step=1, label="Beam width")
                    p_depth = gr.Slider(1, 3, value=2, step=1, label="Mutation depth")
                    p_seeds = gr.Slider(1, 15, value=5, step=1, label="# top training seeds")
                with gr.Row():
                    p_alpha = gr.Slider(0.0, 1.0, value=0.10, step=0.05,
                                          label="α (0 = pure ML, 1 = pure physics-extrapolation) — "
                                                "default 0.10 is LOO-Spearman-optimal with no-proxy physics; "
                                                "crank to 0.3–0.6 for extrapolation searches above a top seed")
                    p_kappa = gr.Slider(0.0, 3.0, value=1.5, step=0.25,
                                          label="κ UCB (exploration aggressiveness)")
                    p_sar = gr.Slider(0.0, 1.0, value=0.4, step=0.05,
                                       label="β SAR prior (informed-mutation steering; "
                                             "0 = off, 0.4 = default, higher = steer harder "
                                             "toward the family's data-supported directions)")
                with gr.Row():
                    p_rank_by = gr.Radio(
                        choices=[
                            "score (default)",
                            "expected_improvement (active learning)",
                            "sigma_ensemble (pure exploration)",
                            "chemberta_novelty (structural diversity)",
                        ],
                        value="score (default)",
                        label="Rank top-20 by — EI: likely above T AND informative; "
                              "σ: most uncertain; chemberta_novelty: most structurally "
                              "different from training in a pretrained semantic space",
                    )
                go_p = gr.Button("Propose", variant="primary")
                out_p_md = gr.Markdown()
                with gr.Accordion("Candidate CSV (top 50)", open=False):
                    out_p_csv = gr.Code()
                go_p.click(propose_better,
                            inputs=[seed_in, p_threshold, p_beam, p_depth,
                                    p_seeds, p_alpha, p_kappa, p_sar, p_rank_by],
                            outputs=[out_p_md, out_p_csv])

        gr.Markdown(
            "---\n"
            "Training set: 278 pKa measurements, 273 bioactivity measurements, "
            "across six chemical families."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
