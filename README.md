---
title: IAJD Tandem Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# IAJD Tandem Predictor

Predicts **pKa (v9.1)** and **log₁₀ flux total bioactivity (v14 + stacker v1)**
for ionizable amino-lipid janus dendrimers (IAJDs).

## What's in here

- **`app.py`** — Gradio UI for HF Spaces (single SMILES + batch upload tabs).
- **`iajd_predict.py`** — public `predict()` and `predict_batch()` entry points.
- **`iajd_server.py`** — alternate localhost HTTP server (not used by Spaces).
- **`iajd_v15.py`** — count-Tanimoto + analog-delta v15 base.
- **`iajd_structural.py`** — ChemDraw / `.cdxml` / `.mol` / `.sdf` parser
  with 2D positional-encoding structural features.
- **`iajd_family.py`** — SMARTS + count-Morgan kNN family detector
  (96% pooled LOO recall on 5 chemical families; HTM / TT / G1-Janus collapsed
  to PE-Gallic per SMILES audit).
- **`IAJD_master/`** — datasets, source bundles, supporting code from the
  research pipeline.

## Headline metrics (LOO)

| Stage | Baseline | Final | Δ |
|---|---|---|---|
| pKa v9.1 | 0.0653 | 0.0653 | 0.0 |
| Bioactivity (log₁₀ flux) | 0.4010 | **0.3934** | **−0.0076** |

The bioactivity stacker is an XGBoost (depth=3, n=300, lr=0.03) over four
heads — full-feature direct XGB, Tanimoto-weighted analog-delta consensus,
LION-block-only XGB, ADMET-block-only XGB — trained on LOO out-of-fold
predictions.

## Inputs

- **SMILES** — single or one-per-line; optional `family=` token.
- **ChemDraw `.cdxml` / `.cdx`** — multiple IAJDs per file supported.
- **MDL `.mol` / `.sdf`** — single or multi-record.
- **`.smi` / `.txt`** — one SMILES per line; tokens
  `<SMILES> [family=<FAMILY>] [<label>]`.

Family is **auto-detected per molecule** (96% LOO accuracy on 5 chemical
classes); user override available globally and per row.

## Outputs

For each molecule: predicted **pKa** + 60% / 90% CIs, **log₁₀ flux total
bioactivity** with linear-flux in scientific notation alongside every value,
predicted target organ + per-organ partition, nearest training analogues,
stacker component breakdown, and structural-refinement trace.
