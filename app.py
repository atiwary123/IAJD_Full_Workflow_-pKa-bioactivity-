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

    # ── pKa ──
    md.append(f"## pKa = {pka.get('point', '—')}")
    md.append(f"_v9.1 analog-delta XGBoost (278 training compounds, 6 families, LOO MAE 0.065) "
              f"| tier `{pka.get('tier', '—')}` | max Tanimoto {pka.get('max_tanimoto_to_training', '—')}_")
    if pka.get("ci_90"):
        md.append(f"_90% CI: [{pka['ci_90'][0]:.3f}, {pka['ci_90'][1]:.3f}]_")
    md.append("")

    # ── Three-model bioactivity comparison ──
    is_lookup = bio.get("source") == "training_set_exact_match"

    if is_lookup:
        iajd_id = bio.get("iajd_id_if_measured", "")
        md.append(f"## Bioactivity: MEASURED VALUE (training set exact match)")
        md.append(f"_This compound ({iajd_id}) is in the training set (Tanimoto=1.0). "
                  f"The value below is the **stored experimental measurement**, not a model prediction._")
        md.append("")
        md.append(f"## log10 flux total = {_log_with_sci(bio.get('point'))}")
        md.append("")
    else:
        md.append("## Bioactivity Predictions (log10 flux total)")
    md.append("")
    md.append("| Model | Prediction | Description |")
    md.append("|---|---|---|")

    # Model 1: v14 + stacker
    bio_point = _log_with_sci(bio.get('point'))
    stk_mode = stk.get("mode", "static") if stk.get("applied") else "off"
    novelty = stk.get("components", {}).get("novelty")
    nov_str = f" novelty={novelty:.2f}" if novelty is not None else ""
    if is_lookup:
        md.append(f"| **v14 + adaptive stacker** | **{bio_point}** | "
                  f"LOOKUP (exact match in training set, not a prediction) |")
    else:
        md.append(f"| **v14 + adaptive stacker** | **{bio_point}** | "
                  f"6-head ensemble (direct + analog + LION + ADMET + AGILE + CPP) with OOD-aware dynamic weighting; "
                  f"physics heads (CPP/AGILE) dominate as queries drift from training;{nov_str} "
                  f"LOO MAE 0.403 on 335 compounds |")

    # Model 2: v11 M2 pKa-dominant
    if v11.get("point") is not None:
        v11_point = _log_with_sci(v11.get('point'))
        md.append(f"| **v11 pKa-dominant** | **{v11_point}** | "
                  f"pKa + family + pKa x family interaction + 8 tail descriptors; "
                  f"LOO MAE {v11.get('loo_mae', 0):.3f} on {v11.get('n_train_rows', '?')} compounds |")

    # Model 3: pKa-flux curve (if available in result)
    pka_curve = r.get("bioactivity_pka_curve") or {}
    if pka_curve.get("point") is not None:
        pc_point = _log_with_sci(pka_curve.get('point'))
        md.append(f"| **pKa-flux curve** | **{pc_point}** | "
                  f"Per-family fitted pKa-to-flux quadratic + structural residual corrector; "
                  f"flux step uses no similarity, but pKa input carries indirect similarity from v9.1; LOO MAE ~0.50 |")

    md.append("")

    # Confidence info
    if bio.get("ci_90"):
        md.append(f"_v14+stacker 90% CI: {_ci_with_sci(bio['ci_90'])}_")
    if bio.get("max_tanimoto") is not None:
        md.append(f"_max Tanimoto to bioact training: {bio['max_tanimoto']:.3f}_")
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


def single_predict(smiles: str, family_choice: str, neighbors: int):
    if not smiles or not smiles.strip():
        return "_Enter a SMILES._", ""
    family = None if (not family_choice or family_choice == "(auto-detect)") else family_choice
    try:
        r = predict(smiles.strip(), family=family, neighbors=int(neighbors))
    except Exception as exc:  # noqa: BLE001
        return f"### Prediction failed\n\n```\n{type(exc).__name__}: {exc}\n```", ""
    _attach_v11(r)
    md = _render_result_markdown(r, 0)
    raw = json.dumps(r, indent=2, default=str)
    return md, raw


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
    intro = f"""
# IAJD Tandem Predictor

Predicts **pKa** and **bioactivity (log10 total flux)** for ionizable amphiphilic Janus dendrimers.
Three independent bioactivity models run in parallel on every query:

| Model | What it does | LOO MAE |
|---|---|---|
| **v14 + adaptive stacker** | 6-head ensemble with OOD-aware dynamic weighting: as queries drift from training, physics heads (CPP packing parameter + AGILE GNN) automatically dominate over similarity-dependent heads | 0.403 |
| **v11 pKa-dominant** | Predicted pKa + family one-hot + pKa x family interactions + 8 tail descriptors | 0.475 |
| **pKa-flux curve** | Per-family fitted pKa-to-flux quadratic + structural residual corrector; flux step uses no similarity, but pKa input carries indirect similarity from v9.1 | ~0.50 |

**pKa model:** v9.1 analog-delta XGBoost, 278 training compounds across 6 families, LOO MAE 0.065.

Accepts SMILES, ChemDraw (.cdxml), SDF, MOL. Family auto-detected (6 chemical families + 2 bioactivity-only subarchitectures).

**Status:** {"loaded" if BUNDLE_OK else f"failed — {BUNDLE_ERR}"} | v11 {"loaded" if V11_AVAILABLE else "unavailable"}
"""

    with gr.Blocks(title="IAJD Tandem Predictor") as demo:
        gr.Markdown(intro)
        if not BUNDLE_OK:
            gr.Markdown(f"⚠️ **Bundle load failed at startup.** Predictions will not work until this is resolved.")

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
                go = gr.Button("Predict", variant="primary")
            out_md = gr.Markdown()
            with gr.Accordion("Raw JSON result", open=False):
                out_json = gr.Code(language="json")
            go.click(single_predict, inputs=[smiles_in, fam, neighbors],
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

        gr.Markdown(
            "---\n"
            "_Training: 278 pKa compounds (6 families) + 273 bioactivity measurements (6 families)._ "
            "_pKa v9.1 LOO MAE 0.065. Adaptive stacker (6-head, OOD-aware) LOO MAE 0.403. v11 M2 LOO MAE 0.475._"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
