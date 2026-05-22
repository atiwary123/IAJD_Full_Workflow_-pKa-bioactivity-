"""
app.py — Gradio entrypoint for the IAJD Tandem Predictor on Hugging Face Spaces.

Wraps `predict()` and `predict_batch()` from iajd_predict.py and renders the
single-SMILES and multi-molecule batch flows that the localhost HTTP server
provides, formatted as Markdown for Gradio.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

import gradio as gr

from iajd_predict import (
    predict, predict_batch, _load_bundle, ALLOWED_FAMILIES,
)
from iajd_family import CHEMICAL_FAMILIES, BIOACT_ONLY_SUBARCHS

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
        return f"### ❌ {r.get('input_label') or 'mol ' + str(idx+1)}\n\n**Error:** {r['error']}"

    label = r.get("input_label") or f"mol {idx + 1}"
    pka = r.get("pka") or {}
    bio = r.get("bioactivity") or {}
    organ = r.get("organ_delivery") or {}
    fr = r.get("family_resolution") or {}
    det = fr.get("detection") or {}
    cands = det.get("candidates") or []
    nb = r.get("neighbors") or {}
    stk = bio.get("stacker") or {}

    md = []
    md.append(f"### `{label}`")
    md.append(f"`{r.get('canonical_smiles', '')}`")
    md.append("")
    fam_src = (fr.get("source") or "—").replace("auto:", "")
    md.append(f"**Family:** `{fr.get('family_assigned') or '—'}`  · source: `{fam_src}`")
    if fr.get("subarch_label"):
        md.append(f"_(original label `{fr['subarch_label']}` collapsed to PE-Gallic for bioact routing)_")
    if det.get("confidence") is not None:
        md.append(f"_detect confidence: {det['confidence']*100:.1f}% · max Tanimoto to training: {det.get('max_tanimoto')}_")
    if len(cands) > 1:
        alt_txt = ", ".join(f"{c['family']} ({c['score']*100:.0f}%)" for c in cands[1:3])
        md.append(f"_alternatives: {alt_txt}_")
    md.append("")

    # pKa block
    md.append("#### pKa")
    md.append(f"**{pka.get('point', '—')}**  · σ={pka.get('sigma', '—')} · tier `{pka.get('tier', '—')}` · src `{pka.get('source', '—')}`")
    if pka.get("ci_60"):
        md.append(f"- 60% CI: `[{pka['ci_60'][0]:.3f}, {pka['ci_60'][1]:.3f}]`")
    if pka.get("ci_90"):
        md.append(f"- 90% CI: `[{pka['ci_90'][0]:.3f}, {pka['ci_90'][1]:.3f}]`")
    if pka.get("max_tanimoto_to_training") is not None:
        md.append(f"- max Tanimoto to training: {pka['max_tanimoto_to_training']} · OOD={pka.get('ood_flag')}")
    if pka.get("point_v15_original") is not None:
        sr = pka.get("structural_refinement") or {}
        md.append(f"- struct refine: Δ={sr.get('delta_from_original')} from v15 baseline ({pka['point_v15_original']})")
    md.append("")

    # Bioact block
    md.append("#### log₁₀ flux total (linear flux in parentheses)")
    md.append(f"**{_log_with_sci(bio.get('point'))}**  · σ={bio.get('sigma', '—')} · tier `{bio.get('tier', '—')}` · α={bio.get('alpha_used')}")
    if bio.get("ci_60"):
        md.append(f"- 60% CI: `{_ci_with_sci(bio['ci_60'])}`")
    if bio.get("ci_90"):
        md.append(f"- 90% CI: `{_ci_with_sci(bio['ci_90'])}`")
    if bio.get("max_tanimoto") is not None:
        md.append(f"- max Tanimoto: {bio['max_tanimoto']:.3f} · LION_real={bio.get('block_B_real')}")
    if bio.get("point_v15_original") is not None:
        md.append(f"- v15 (no struct refine): {_log_with_sci(bio['point_v15_original'])}")
    if bio.get("point_pre_stacker") is not None:
        comps = stk.get("components") or {}
        md.append(f"- pre-stacker: {_log_with_sci(bio['point_pre_stacker'])}  "
                  f"· stacker components D={comps.get('direct_head_pred')} A={comps.get('analog_pred')} "
                  f"L={comps.get('lion_head_pred')} M={comps.get('admet_head_pred')}")
        md.append(f"- stacker LOO MAE {stk.get('expected_loo_mae')} (vs baseline {stk.get('baseline_loo_mae')})")
    md.append("")

    # Organ
    if organ.get("target_organ"):
        md.append(f"#### Predicted target organ: **{organ['target_organ']}**")
        md.append(f"_{organ.get('method')} · n_neighbors={organ.get('n_neighbors_used')}_")
        md.append("")
        md.append("| organ | partition | log₁₀ flux (linear) |")
        md.append("|---|---|---|")
        parts = organ.get("partition_pct") or {}
        lfs = organ.get("log10_flux_by_organ") or {}
        for k in sorted(parts, key=lambda x: -parts[x]):
            md.append(f"| {k} | {parts[k]}% | {_log_with_sci(lfs.get(k))} |")
        md.append("")

    # Neighbors
    if nb.get("pka"):
        md.append("#### Nearest pKa training IAJDs")
        md.append("| IAJD | Tanimoto | family | pKa |")
        md.append("|---|---|---|---|")
        for n in nb["pka"]:
            md.append(f"| {n.get('iajd_id')} | {n.get('tanimoto', 0):.3f} | {n.get('family')} | {n.get('pKa')} |")
        md.append("")
    if nb.get("bioact"):
        md.append("#### Nearest bioactivity training IAJDs")
        md.append("| IAJD | Tanimoto | family | log₁₀ flux (linear) |")
        md.append("|---|---|---|---|")
        for n in nb["bioact"]:
            md.append(f"| {n.get('iajd_id')} | {n.get('tanimoto', 0):.3f} | {n.get('family')} | {_log_with_sci(n.get('log10_flux_total'))} |")
        md.append("")

    if r.get("warnings"):
        md.append("**Warnings:** " + "; ".join(r["warnings"]))
    return "\n".join(md)


def single_predict(smiles: str, family_choice: str, neighbors: int):
    if not smiles or not smiles.strip():
        return "_Enter a SMILES._", ""
    family = None if (not family_choice or family_choice == "(auto-detect)") else family_choice
    try:
        r = predict(smiles.strip(), family=family, neighbors=int(neighbors))
    except Exception as exc:  # noqa: BLE001
        return f"### Prediction failed\n\n```\n{type(exc).__name__}: {exc}\n```", ""
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
        parts.append("\n---\n")
        parts.append(_render_result_markdown(r, i))
    raw = json.dumps(rb, indent=2, default=str)
    return "\n".join(parts), raw


def build_ui() -> gr.Blocks:
    intro = f"""
# IAJD Tandem Predictor

Predicts **pKa (v9.1)** and **log₁₀ flux total bioactivity (v14 + stacker v1)** for
ionizable amino-lipid janus dendrimers, with auto-detected family (5-class chemical
axis; 96% LOO recall), ChemDraw / `.cdxml` / `.sdf` / `.mol` / multi-SMILES upload,
2D positional structural refinement on the analog leg, and an XGBoost stacker
over (direct, analog, LION, ADMET) heads (true-LOO MAE **0.3934** vs **0.4010**
v14 baseline).

**Bundle status:** {"loaded ✓" if BUNDLE_OK else f"failed — {BUNDLE_ERR}"}

Reference: https://github.com/atiwary123/IAJD_Full_Workflow_-pKa-bioactivity-
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
            "_Stacker v1: bioact LOO MAE 0.3934 vs 0.4010 baseline (−0.0076)._ "
            "_pKa v9.1: LOO MAE 0.0653._ "
            "_Family detector: 96% LOO recall on 5 chemical families (HTM / TT / "
            "G1-Janus collapsed to PE-Gallic — SMILES audit confirmed they share "
            "canonical SMILES with PE-Gallic entries)._"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
