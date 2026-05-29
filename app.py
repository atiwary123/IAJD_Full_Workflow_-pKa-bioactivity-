"""
app.py — Gradio entrypoint for the IAJD Tandem Predictor on Hugging Face Spaces.

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

# Binary (tunable-threshold) classifier head — added 2026-05-28.
try:
    from predict_binary import load_binary_bundle, predict_p_above
    _BIN_BUNDLE = load_binary_bundle()
    BIN_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    BIN_AVAILABLE = False
    _BIN_BUNDLE = None
    predict_p_above = None
    print(f"[binary head] not available: {exc}")

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
    from iajd_grammar import build_library, decompose_row, Seed
    import propose_iajds as _propose
    _PROPOSE_LIB = None
    PROPOSE_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    PROPOSE_AVAILABLE = False
    print(f"[proposer] not available: {exc}")

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
    _Xs = _pd_curve.DataFrame(_feat_r)[_SF_NAMES].values
    for _col in range(_Xs.shape[1]):
        _m = ~_np.isfinite(_Xs[:, _col])
        if _m.any():
            _Xs[_m, _col] = _np.nanmedian(_Xs[:, _col])
    _PKA_CURVES = _fit_curves(_pkas, _fluxes, _fams)
    _cp = _np.array([_pred_curve(p, f, _PKA_CURVES) for p, f in zip(_pkas, _fams)])
    _res = _fluxes - _cp
    _Xf = _np.column_stack([_pkas, _Xs])
    _PKA_CURVE_MEDIANS = _np.nanmedian(_Xf, axis=0)
    _PKA_CURVE_XGB = _xgb_mod.XGBRegressor(**_RES_HP)
    _PKA_CURVE_XGB.fit(_Xf, _res, verbose=False)
    _PKA_CURVE_READY = True
except Exception:
    pass

def _predict_pka_curve(pred_pka, family, smiles):
    curve_val = _pred_curve(pred_pka, family, _PKA_CURVES)
    feats = _extract_sf(smiles)
    x = _np.array([[pred_pka] + [feats.get(f, 0) for f in _SF_NAMES]])
    for col in range(x.shape[1]):
        if not _np.isfinite(x[0, col]):
            x[0, col] = _PKA_CURVE_MEDIANS[col]
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

    # ── Binary threshold verdict — derived from the single v14+stacker
    #    prediction (or measured value if lookup), Gaussian assumption with
    #    σ = v14+stacker LOO RMSE. NO separate model, NO calibrator that can
    #    disagree with the point estimate.
    bin_info = r.get("binary_above_threshold") or {}
    T = bin_info.get("threshold")
    if T is not None and bio_point is not None:
        import math
        if is_lookup:
            p = 1.0 if bio_point >= T else 0.0
            verdict_source = "from measured value"
        else:
            sigma_v14 = 0.43   # v14+stacker LOO RMSE
            z = (float(bio_point) - float(T)) / sigma_v14
            p = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
            verdict_source = f"Gaussian on v14+stacker (σ = {sigma_v14})"
        verdict = "above" if p >= 0.5 else "below"
        md.append(f"### P(log₁₀ flux ≥ {T:.2f}) = {p:.1%} — likely {verdict}")
        md.append(f"_{verdict_source}_")
        md.append("")

    # Organ delivery
    if organ.get("target_organ"):
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


def single_predict(smiles: str, family_choice: str, neighbors: int,
                    threshold: float = 8.0):
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
    md = _render_result_markdown(r, 0)
    raw = json.dumps(r, indent=2, default=str)
    return md, raw


def propose_better(seed_smiles: str, threshold: float, beam: int, depth: int,
                    top_seeds: int, exploration_weight: float = 0.0,
                    kappa_ucb: float = 1.5, sar_weight: float = 0.4):
    """Run the fragment-swap proposer; render top candidates as Markdown.

    exploration_weight (α ∈ [0,1]):
        0 → pure ML scoring (Δ vs seed × P(≥T))
        1 → pure physics-justified UCB (Q_physics × upper bound)
        in-between blends the two — useful when ML is bounded by training
        distribution and you want extrapolation hypotheses surfaced
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

    def _render(result_df, round_idx: int, is_final: bool):
        if is_final:
            header = f"## Proposed IAJDs — DONE"
        else:
            header = f"## Proposed IAJDs — live (round {round_idx}/{depth})"
        rank_desc = (f"α={exploration_weight:.2f}·[`P(≥T)·Δ_ML` ⊕ `Q_phys·UCB`] "
                      f"(κ={kappa_ucb:.2f})  +  β={sar_weight:.2f}·`Δ_SAR` "
                      f"(family informed-mutation prior)")
        body = [header,
                f"_T={threshold}, beam={beam}, depth={depth}, "
                f"{len(seeds)} seed(s); ranked by {rank_desc}. "
                f"{len(result_df)} candidates shown._",
                ""]
        body.append("| rank | Δ vs seed | Δ_SAR | ŷ ML | seed ŷ | P(≥T) | Q_phys | escape | Tanim | What changed (· why, per training SAR) | SMILES |")
        body.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(result_df.head(20).iterrows()):
            desc = r.get("mutation_description") or " → ".join((r.get("mutation_trail") or [])[-2:])
            sar_why = r.get("sar_reason")
            if isinstance(sar_why, str) and sar_why.strip():
                desc = f"{desc} · _{sar_why}_"
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
            body.append(
                f"| {i+1} | **{delta_s}** | {sar_s} | {r['yhat']:.2f} | {seed_y_s} | "
                f"{r['p_above']:.0%} | {_fmt(q_p)} | "
                f"{_fmt(esc)} | {tanim_s} | {desc} | `{r['smiles']}` |"
            )
        if is_final:
            body.append("")
            tan_mask = result_df["tanim_max_to_train"] < 0.85
            body.append(f"_Total scored: {len(result_df)}; "
                          f"novel (Tanim < 0.85): {int(tan_mask.sum())}_")
        csv = result_df.head(50).to_csv(index=False)
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
# IAJD Tandem Predictor

Predicts **pKa** and **bioactivity (log₁₀ total flux)** for ionizable amphiphilic Janus dendrimers from molecular input.

| Property | Model | LOO MAE |
|---|---|---|
| pKa | v9.2 three-head blend (analog + XGB + live MolGpKa) | 0.125 |
| log₁₀ flux | v14 + adaptive stacker | 0.43 |

**Input formats:** SMILES, ChemDraw (.cdxml), SDF, MOL. Family auto-detected from six chemical families.

**Tabs:**
- *Single SMILES* — predict one molecule
- *Batch* — file upload or multi-line SMILES paste
- *Propose better IAJDs* — beam search over single-step structural mutations of a seed
"""

    with gr.Blocks(title="IAJD Tandem Predictor") as demo:
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
            out_md = gr.Markdown()
            with gr.Accordion("Raw JSON result", open=False):
                out_json = gr.Code(language="json")
            go.click(single_predict, inputs=[smiles_in, fam, neighbors, threshold],
                     outputs=[out_md, out_json])

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
                    "• **α = 0**: pure ML. Ranks candidates that look like training data and "
                    "  beat the seed by ML's estimate.\n"
                    "• **α = 1**: pure physics-justified extrapolation. Ranks by mechanism-based "
                    "  quality (CPP near 1, endosomal escape Δ, HLB optimal, membrane affinity) "
                    "  combined with the model's uncertainty σ — useful when the seed is at the "
                    "  top of the training distribution and ML can't see anything above it.\n"
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
                go_p = gr.Button("Propose", variant="primary")
                out_p_md = gr.Markdown()
                with gr.Accordion("Candidate CSV (top 50)", open=False):
                    out_p_csv = gr.Code()
                go_p.click(propose_better,
                            inputs=[seed_in, p_threshold, p_beam, p_depth,
                                    p_seeds, p_alpha, p_kappa, p_sar],
                            outputs=[out_p_md, out_p_csv])

        gr.Markdown(
            "---\n"
            "Training set: 278 pKa measurements, 273 bioactivity measurements, "
            "across six chemical families."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
