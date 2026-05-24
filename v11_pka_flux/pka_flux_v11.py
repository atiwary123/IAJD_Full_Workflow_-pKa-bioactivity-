"""
pka_flux_v11.py — pKa-dominant flux prediction experiment.

Builds 4 models on log10(total flux) and evaluates them under one protocol:
  M0  predicted_pKa only (1-feature null)
  M1  predicted_pKa + family one-hot + (pKa × family) interaction
  M2  M1 + tail/shape descriptors (the honest pKa-dominant model)
  M3  v14 cascade baseline (refit per fold, same features + HP as deployed v14)

Evaluation: family-stratified k=5 with full refit per fold (Section 5 fallback
authorized in the prompt; full nested LOO was deemed too expensive).

For predicted_pKa generation:
  - bioact rows whose canonical SMILES is in the v21 pKa training set: use
    preds_v91_final.npy LOO predictions (v9.1, pooled LOO MAE 0.0672).
  - bioact rows NOT in v21: call predict_pka_v91 with the full v21 bundle.
  - No pKa target leaks into the flux model: the only rows where measured
    pKa exists AND would be in the flux training set get the v21 LOO value,
    which was computed with that row held out at the pKa level.

Reads:  IAJD_master/datasets/{IAJD_Bioact_v13_clean.xlsx, IAJD_pKa_v21_final.xlsx}
        IAJD_master/bundles_caches/bioact_v14_bundle.pkl
        preds_v91_final.npy
Writes: v11_pka_flux/{predictions.csv, metrics.json, shap_M2.png,
        calibration.png, family_optima.csv, REPORT.md, predicted_pka_cache.csv}

Run:    .venv/bin/python v11_pka_flux/pka_flux_v11.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "IAJD_master" / "datasets"
BUNDLES = ROOT / "IAJD_master" / "bundles_caches"
CODE = ROOT / "IAJD_master" / "code"
OUT = ROOT / "v11_pka_flux"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE))


# ---------------------------------------------------------------------------
# 1. Load tables and align IDs
# ---------------------------------------------------------------------------

def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    bio = pd.read_excel(DATA / "IAJD_Bioact_v13_clean.xlsx")
    pka_tbl = pd.read_excel(DATA / "IAJD_pKa_v21_final.xlsx", sheet_name="Dataset")
    pka_loo = np.load(ROOT / "preds_v91_final.npy")
    assert len(pka_loo) == len(pka_tbl), "preds_v91_final.npy row count mismatch"
    return bio, pka_tbl, pka_loo


# ---------------------------------------------------------------------------
# 2. Predicted pKa for every bioact row
# ---------------------------------------------------------------------------

PKA_CACHE = OUT / "predicted_pka_cache.csv"


def _canon(s: str) -> str:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m is not None else ""


def generate_predicted_pka(bio: pd.DataFrame, pka_tbl: pd.DataFrame,
                            pka_loo: np.ndarray) -> pd.DataFrame:
    if PKA_CACHE.exists():
        cached = pd.read_csv(PKA_CACHE)
        if len(cached) == len(bio):
            print(f"[pKa] cache hit ({len(cached)} rows) — skipping")
            return cached
        print(f"[pKa] cache stale ({len(cached)} vs {len(bio)}), regenerating")

    print("[pKa] importing v9.1 …")
    from iajd_pka_v91 import load_v91_bundle, predict_pka_v91

    cwd = os.getcwd()
    os.chdir(DATA)
    try:
        bundle = load_v91_bundle(
            xlsx_path="IAJD_pKa_v21_final.xlsx",
            debias_path="molgpka_debias_models.joblib",
            molgpka_npy="molgpka_preds.npy",
        )
    finally:
        os.chdir(cwd)

    # Build canonical-SMILES → v21-LOO map
    pka_canon = pka_tbl["SMILES"].map(_canon)
    loo_map: Dict[str, Tuple[float, float, str]] = {}
    for i, can in enumerate(pka_canon):
        if can:
            loo_map[can] = (float(pka_loo[i]),
                            float(pka_tbl.iloc[i]["pKa"]),
                            str(pka_tbl.iloc[i]["family"]))

    out = []
    family_mismatch = 0
    n_loo = 0
    n_full = 0
    t0 = time.time()
    for idx, row in bio.iterrows():
        smi = row.get("SMILES_canonical") or row.get("SMILES")
        can = _canon(smi)
        fam_table = row.get("family")
        rec = {"row_id": row.get("row_id"), "IAJD_id": row.get("IAJD_id"),
               "family_table": fam_table, "canonical_smiles": can}
        if can in loo_map:
            pred, meas, fam_v21 = loo_map[can]
            rec.update({"predicted_pKa": pred, "measured_pKa_v21": meas,
                        "family_v21": fam_v21, "pka_source": "v21_loo"})
            if fam_v21 != fam_table:
                family_mismatch += 1
            n_loo += 1
        else:
            try:
                r = predict_pka_v91(can, bundle, family_hint=str(fam_table),
                                     return_diagnostics=False)
                if "error" in r:
                    rec.update({"predicted_pKa": np.nan,
                                "pka_source": "error:" + r["error"]})
                else:
                    rec.update({"predicted_pKa": float(r["pKa_pred"]),
                                "measured_pKa_v21": np.nan,
                                "family_v21": r.get("family_assigned"),
                                "pka_source": "v91_full_bundle"})
                n_full += 1
            except Exception as exc:  # noqa: BLE001
                rec.update({"predicted_pKa": np.nan,
                            "pka_source": f"exception:{type(exc).__name__}"})
        out.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"  [pKa] {idx + 1}/{len(bio)} done ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(out)
    df.to_csv(PKA_CACHE, index=False)
    print(f"[pKa] {n_loo} v21-LOO + {n_full} full-bundle calls; "
          f"{family_mismatch} family mismatches; cached → {PKA_CACHE}")

    # Sanity check: pooled MAE on rows with measured pKa
    have_meas = df.dropna(subset=["measured_pKa_v21"])
    if len(have_meas):
        mae = (have_meas["predicted_pKa"] - have_meas["measured_pKa_v21"]).abs().mean()
        print(f"[pKa] sanity: pooled MAE on {len(have_meas)} v21-LOO rows "
              f"= {mae:.4f} (expect ~0.067)")
    return df


# ---------------------------------------------------------------------------
# 3. Feature builders for M0 / M1 / M2
# ---------------------------------------------------------------------------

FAMILIES_ORDER = ["sSS-Nonsym", "PE-Tris", "GA-Tris", "PE-Gallic",
                  "Dialkoxybenzyl", "G1-Janus-Dendrimer"]

# Tail/shape descriptors already present in the bioact table.
# These are the minimum-set "honest" features that the mechanism section
# of the prompt calls out: tail size/saturation/branching + head polarity
# + cone-shape proxy via Crippen-MR ratio (LabuteASA is the rdkit
# stand-in).  Kept narrow to keep M2 honest.
TAIL_DESCRIPTORS = [
    "MolLogP",          # tail lipophilicity proxy
    "FractionCSP3",     # tail saturation
    "RotatableBonds",   # chain flexibility / length
    "NumAromaticRings", # head aromaticity
    "TPSA",             # head polarity
    "LabuteASA",        # overall accessible surface (cone-shape proxy)
    "HeavyAtomCount",   # size
    "linker_carbons",   # explicit linker length
]


def _family_onehot(fams: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {f"fam_{f}": (fams == f).astype(int) for f in FAMILIES_ORDER},
        index=fams.index,
    )


def build_features_m0(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = ["predicted_pKa"]
    return df[cols].to_numpy(dtype=float), cols


def build_features_m1(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    oh = _family_onehot(df["family"])
    pka = df["predicted_pKa"].to_numpy(dtype=float).reshape(-1, 1)
    inter = oh.to_numpy() * pka  # broadcast: pka × family
    inter_cols = [f"pKa_x_{c}" for c in oh.columns]
    X = np.hstack([pka, oh.to_numpy(), inter])
    cols = ["predicted_pKa"] + list(oh.columns) + inter_cols
    return X, cols


def build_features_m2(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    Xm1, m1_cols = build_features_m1(df)
    tail = df[TAIL_DESCRIPTORS].to_numpy(dtype=float)
    # fill missing values with column medians (computed on the full table)
    med = np.nanmedian(tail, axis=0)
    nan_mask = ~np.isfinite(tail)
    tail = np.where(nan_mask, med, tail)
    X = np.hstack([Xm1, tail])
    cols = m1_cols + TAIL_DESCRIPTORS
    return X, cols


def feature_sha(cols: List[str]) -> str:
    return hashlib.sha256("|".join(cols).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 4. Models (fit-on-train, predict-on-fold)
# ---------------------------------------------------------------------------

GBR_HP = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=1,
)


def fit_predict_gbr(X_tr, y_tr, X_te):
    m = xgb.XGBRegressor(**GBR_HP)
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict(X_te), m


def fit_predict_quad_ols(X_tr, y_tr, X_te):
    """log_flux ~ a + b*pKa + c*pKa^2 — M0 sanity-check baseline."""
    pka_tr = X_tr[:, 0]
    pka_te = X_te[:, 0]
    A = np.column_stack([np.ones_like(pka_tr), pka_tr, pka_tr ** 2])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    Ate = np.column_stack([np.ones_like(pka_te), pka_te, pka_te ** 2])
    return Ate @ coef


# ---------------------------------------------------------------------------
# 5. M3 = v14 refit-per-fold
# ---------------------------------------------------------------------------

V14_HP = dict(
    n_estimators=600, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=1, objective="reg:squarederror",
)


def load_v14_artifacts():
    import pickle
    with open(BUNDLES / "bioact_v14_bundle.pkl", "rb") as f:
        b = pickle.load(f)
    return {
        "X": np.asarray(b["X_train"], dtype=float),
        "y": np.asarray(b["y_train"], dtype=float),
        "fps": list(b["fps_train"]),
        "fams": np.asarray(b["families_train"]),
        "smis": np.asarray(b["smis_train"]),
        "alpha_per_family": dict(b["best_alpha_per_family"]),
        "block_b_active": dict(b["block_b_active_per_family"]),
        "block_c_active": dict(b["block_c_active_per_family"]),
        "block_slices": dict(b["block_slices"]),
    }


def fit_predict_m3(v14, train_smis, test_smis, test_fams):
    """Refit v14-class direct head + analog-delta blend on rows whose
    canonical SMILES is NOT in test_smis. Predict on test_smis."""
    canon = {s: i for i, s in enumerate(v14["smis"])}
    train_idx = [canon[s] for s in train_smis if s in canon]
    test_idx_map = {s: canon.get(s) for s in test_smis}

    X = v14["X"][train_idx]
    y = v14["y"][train_idx]
    fps_tr = [v14["fps"][i] for i in train_idx]
    fams_tr = v14["fams"][train_idx]

    # Per-family gating (same as v14): apply block masks per query family
    bsl = v14["block_slices"]
    direct = xgb.XGBRegressor(**V14_HP)
    direct.fit(X, y, verbose=False)

    preds = []
    for smi, fam in zip(test_smis, test_fams):
        i = test_idx_map.get(smi)
        if i is None:
            preds.append(np.nan)
            continue
        x = v14["X"][i].copy()
        gate_b = v14["block_b_active"].get(fam, True)
        gate_c = v14["block_c_active"].get(fam, True)
        if not gate_b:
            x[bsl["B"][0]:bsl["B"][1]] = 0
        if not gate_c:
            x[bsl["C"][0]:bsl["C"][1]] = 0
        direct_pred = float(direct.predict(x.reshape(1, -1))[0])

        fp_q = v14["fps"][i]
        sims = np.array(BulkTanimotoSimilarity(fp_q, fps_tr))
        top = np.argsort(-sims)[:8]
        max_sim = float(sims[top[0]]) if len(top) else 0.0
        if max_sim >= 0.4:
            w = sims[top] ** 4
            w /= w.sum() if w.sum() > 0 else 1.0
            delta_pred = float(np.sum(w * y[top]))
        else:
            delta_pred = None

        alpha = v14["alpha_per_family"].get(fam, 0.6)
        if delta_pred is None:
            preds.append(direct_pred)
        else:
            preds.append(alpha * direct_pred + (1 - alpha) * delta_pred)
    return np.array(preds)


# ---------------------------------------------------------------------------
# 6. k=5 family-stratified eval loop
# ---------------------------------------------------------------------------

def run_eval(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    df = df.dropna(subset=["log10_flux_total", "predicted_pKa"]).reset_index(drop=True)
    y = df["log10_flux_total"].to_numpy(dtype=float)
    fams = df["family"].to_numpy()

    X0, c0 = build_features_m0(df)
    X1, c1 = build_features_m1(df)
    X2, c2 = build_features_m2(df)
    print(f"[features] M0 cols={c0} sha={feature_sha(c0)}")
    print(f"[features] M1 sha={feature_sha(c1)} ({len(c1)} cols)")
    print(f"[features] M2 sha={feature_sha(c2)} ({len(c2)} cols)")

    v14 = load_v14_artifacts()
    smis = df["SMILES_canonical"].to_numpy()

    # Bin small families together for stratification so we don't get
    # singletons in a fold (e.g. TT-Dendrimer n=11)
    counts = pd.Series(fams).value_counts()
    strat = np.where(pd.Series(fams).isin(counts[counts >= 15].index),
                     fams, "other")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), strat)):
        print(f"\n[fold {k + 1}/{n_splits}] train={len(tr_idx)} test={len(te_idx)}")
        y_tr, y_te = y[tr_idx], y[te_idx]

        m0_pred, _ = fit_predict_gbr(X0[tr_idx], y_tr, X0[te_idx])
        m0q_pred = fit_predict_quad_ols(X0[tr_idx], y_tr, X0[te_idx])
        m1_pred, _ = fit_predict_gbr(X1[tr_idx], y_tr, X1[te_idx])
        m2_pred, m2_model = fit_predict_gbr(X2[tr_idx], y_tr, X2[te_idx])
        m3_pred = fit_predict_m3(v14, smis[tr_idx].tolist(),
                                  smis[te_idx].tolist(), fams[te_idx].tolist())

        for j, i in enumerate(te_idx):
            fold_results.append({
                "fold": k, "row_id": int(df.iloc[i]["row_id"])
                                       if "row_id" in df.columns else int(i),
                "smiles": smis[i], "family": fams[i],
                "true": float(y[i]),
                "M0": float(m0_pred[j]), "M0_quad": float(m0q_pred[j]),
                "M1": float(m1_pred[j]), "M2": float(m2_pred[j]),
                "M3": float(m3_pred[j]) if np.isfinite(m3_pred[j]) else np.nan,
                "predicted_pKa": float(df.iloc[i]["predicted_pKa"]),
            })
        # checkpoint after each fold
        pd.DataFrame(fold_results).to_csv(OUT / "predictions.csv", index=False)

    return pd.DataFrame(fold_results), m2_model, c2, X2


# ---------------------------------------------------------------------------
# 7. Metrics + plots + inverted-U fits
# ---------------------------------------------------------------------------

NOISE_FLOOR = 0.28  # log-unit replicate noise from bioact SI


def _metrics_block(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"n": int(mask.sum())}
    t = np.asarray(true)[mask]; p = np.asarray(pred)[mask]
    return {
        "n": int(mask.sum()),
        "MAE": float(mean_absolute_error(t, p)),
        "RMSE": float(np.sqrt(mean_squared_error(t, p))),
        "R2": float(r2_score(t, p)),
        "Spearman": float(spearmanr(t, p).correlation),
    }


def compute_metrics(preds: pd.DataFrame) -> Dict:
    models = ["M0", "M0_quad", "M1", "M2", "M3"]
    out: Dict = {"pooled": {}, "per_family": {}, "noise_floor": NOISE_FLOOR}
    for m in models:
        out["pooled"][m] = _metrics_block(preds["true"], preds[m])
    for fam, g in preds.groupby("family"):
        out["per_family"][fam] = {m: _metrics_block(g["true"], g[m])
                                    for m in models}
    return out


def plot_calibration(preds: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fams = sorted(preds["family"].unique())
    cmap = plt.get_cmap("tab10")
    fam_colors = {f: cmap(i % 10) for i, f in enumerate(fams)}
    for ax, model in zip(axes.flat, ["M0", "M1", "M2", "M3"]):
        for f in fams:
            g = preds[preds["family"] == f]
            ax.scatter(g["true"], g[model], s=18, alpha=0.75,
                        c=[fam_colors[f]], label=f)
        lo = min(preds["true"].min(), preds[model].min(skipna=True))
        hi = max(preds["true"].max(), preds[model].max(skipna=True))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        mae = mean_absolute_error(
            preds["true"][preds[model].notna()],
            preds[model].dropna(),
        )
        ax.set_title(f"{model}  (MAE={mae:.3f})")
        ax.set_xlabel("measured log10 flux")
        ax.set_ylabel("predicted log10 flux")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def shap_m2(preds: pd.DataFrame, df: pd.DataFrame, path: Path):
    """SHAP on M2 refit on the full table (representative model).

    Also returns the mean |SHAP value| ranking so the report can quote the
    top-5 features without forcing the reader to interpret the PNG.
    """
    import shap
    full = df.dropna(subset=["log10_flux_total", "predicted_pKa"]).reset_index(drop=True)
    X, cols = build_features_m2(full)
    y = full["log10_flux_total"].to_numpy(dtype=float)
    model = xgb.XGBRegressor(**GBR_HP)
    model.fit(X, y, verbose=False)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(sv, X, feature_names=cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    ranking = [(cols[i], float(mean_abs[i])) for i in order]
    return ranking


def inverted_u_fits(preds: pd.DataFrame, path: Path):
    """Per-family inverted-U fit log_flux ~ a − b(pKa − pKa_opt)^2.

    pKa_opt bounded to [3, 12] (physical pKa window) so the fit can't drift
    into nonsense regions. Curvature b is also bounded ≥ 0 (b<0 means a
    U-shape, which is not the literature-canonical inverted-U story —
    those families are flagged 'not_inverted_U').
    """
    def f(x, a, b, x0):
        return a - b * (x - x0) ** 2

    rows = []
    for fam, g in preds.groupby("family"):
        n = len(g)
        rec = {"family": fam, "n": n, "pKa_opt": None,
               "curvature_b": None, "a": None, "rmse": None, "note": None}
        if n < 6:
            rec["note"] = "too_few_rows"
            rows.append(rec); continue
        x = g["predicted_pKa"].to_numpy(dtype=float)
        y = g["true"].to_numpy(dtype=float)
        try:
            p0 = [float(y.max()), 1.0, float(x[np.argmax(y)])]
            popt, _ = curve_fit(
                f, x, y, p0=p0, maxfev=10000,
                bounds=([-np.inf, 0.0, 3.0], [np.inf, np.inf, 12.0]),
            )
            yhat = f(x, *popt)
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            rec.update({
                "a": float(popt[0]),
                "curvature_b": float(popt[1]),
                "pKa_opt": float(popt[2]),
                "rmse": rmse,
            })
            # If pKa_opt pinned to either bound, treat as no-clear-optimum
            if abs(popt[2] - 3.0) < 1e-3 or abs(popt[2] - 12.0) < 1e-3:
                rec["note"] = "optimum_at_boundary_no_clear_peak"
            elif popt[1] < 1e-3:
                rec["note"] = "flat_response_no_inverted_U"
        except Exception as exc:  # noqa: BLE001
            rec["note"] = f"fit_failed:{type(exc).__name__}"
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """# pKa-Dominant Flux Model — v11 Experiment Report

**Hypothesis.** Predicted molecular pKa, central to endosomal escape, can carry
most of the signal for ionizable-amine-driven log10 flux. Compared head-to-head
with the v14 cascade (3D + LION + ADMET + Family-α blend) on the same strict-LOO
protocol family-stratified k=5 fold (the prompt's sanctioned fallback to true nested LOO).

**Setup.**
- Bioact table: {n_rows} rows with `log10_flux_total` and `predicted_pKa`.
- Predicted pKa: v9.1 LOO on {n_loo} rows present in v21 (pooled MAE on those = {pka_mae:.4f}, expected ≈0.067), v9.1 full-bundle on {n_full} remaining rows.
- Family-stratified k=5; per-fold full refit of M0/M1/M2 (XGB depth-4) and the M3 v14 direct head + analog-delta blend on n−fold rows.
- Replicate noise floor (bioact SI): {noise:.2f} log-units.

## Models
- **M0**: predicted_pKa only (1 col). Plus a quadratic OLS sanity check.
- **M1**: M0 + family one-hot + (pKa × family) interaction.
- **M2**: M1 + tail/shape descriptors: {tail_cols}.
- **M3**: v14 baseline (88-feature pipeline, direct XGB + analog-delta blend with per-family α and B/C gates), refit per fold on the bundle's stored `X_train/y_train/fps_train`.

## Pooled results
{pooled_table}

## Per-family MAE
{family_table}

## Family pKa optima (inverted-U fits on held-out predictions)
{optima_table}

## SHAP top features for M2 (mean |SHAP value|, full-table refit)
{shap_table}

## Caveats
- {n_family_mismatch} of the v21-LOO rows had a family-table label that disagreed with the v9.1 detector. They are routed through the table's family for M1/M2 features (consistent with how the bioact pipeline would run them).
- M3 baseline was refit per fold using the v14 bundle's stored `X_train/y_train/fps_train`. {n_m3_unmatched} bioact rows have a canonical SMILES that does not match any v14 training row and were skipped in the M3 column (M3 n={m3_n} vs n={total_n}). They remain in M0/M1/M2 columns.
- `linker_carbons` is mostly NaN in the bioact table (342/361) and gets median-imputed in M2; treat any importance from that column as proxy for family rather than chain length.

Files emitted alongside this report:
- `predictions.csv` — per-fold, per-row predictions and residuals.
- `metrics.json` — full pooled + per-family metric table.
- `shap_M2.png` — SHAP summary for M2 (refit on full table) showing whether predicted_pKa is the dominant feature in practice.
- `calibration.png` — 2×2 panel of predicted vs measured for M0/M1/M2/M3.
- `family_optima.csv` — per-family fitted pKa_opt and curvature.
- `predicted_pka_cache.csv` — pKa values per row (cached for reuse).

## Honest interpretation
- The single-feature M0 establishes the raw pKa→flux signal floor.
- M1 quantifies how much of the variance is *family-specific pKa response* vs *family-level intercept differences*.
- M2 adds the minimum tail-descriptor set the literature argues survives pKa-control — if SHAP shows tail descriptors out-importance predicted_pKa, the "pKa-dominant" framing is too strong.
- M3 is the reference. Beating M3 with M2 would suggest the v14 3D-descriptor stack is overfitting; tying within noise would suggest the pKa-centric representation is a strictly cheaper and more interpretable choice for similar accuracy; underperforming by more than ~0.05 log-units says the 3D/LION/ADMET features carry orthogonal signal.

## Verdict
{verdict}
"""


def md_table(d: Dict, models: List[str], col_order=None) -> str:
    if col_order is None:
        col_order = ["MAE", "RMSE", "R2", "Spearman", "n"]
    header = "| Model | " + " | ".join(col_order) + " |"
    sep = "|" + "|".join(["---"] * (len(col_order) + 1)) + "|"
    rows = [header, sep]
    for m in models:
        b = d.get(m, {})
        vals = []
        for c in col_order:
            v = b.get(c)
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        rows.append(f"| {m} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def family_md_table(per_family: Dict, models: List[str]) -> str:
    fams = sorted(per_family.keys())
    header = "| Family | n | " + " | ".join(f"{m}-MAE" for m in models) + " |"
    sep = "|" + "|".join(["---"] * (len(models) + 2)) + "|"
    rows = [header, sep]
    for f in fams:
        b = per_family[f]
        n = b.get(models[0], {}).get("n", "—")
        maes = [f"{b.get(m, {}).get('MAE', float('nan')):.3f}" for m in models]
        rows.append(f"| {f} | {n} | " + " | ".join(maes) + " |")
    return "\n".join(rows)


def optima_md_table(df: pd.DataFrame) -> str:
    rows = ["| Family | n | pKa_opt | curvature b | rmse | note |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        def _fmt(v):
            return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
                   else f"{v:.3f}"
        note = r["note"] if "note" in df.columns and r.get("note") else "—"
        rows.append(
            f"| {r['family']} | {r['n']} | {_fmt(r['pKa_opt'])} | "
            f"{_fmt(r['curvature_b'])} | {_fmt(r['rmse'])} | {note} |"
        )
    return "\n".join(rows)


def shap_md_table(ranking, n=10) -> str:
    rows = ["| Rank | Feature | mean |SHAP\\| |", "|---|---|---|"]
    for i, (name, val) in enumerate(ranking[:n], 1):
        rows.append(f"| {i} | `{name}` | {val:.4f} |")
    return "\n".join(rows)


def pick_verdict(pooled: Dict) -> str:
    m2 = pooled.get("M2", {}).get("MAE")
    m3 = pooled.get("M3", {}).get("MAE")
    if m2 is None or m3 is None:
        return "_Inconclusive — one of the models did not produce a usable MAE._"
    delta = m2 - m3
    if abs(delta) <= 0.02:
        return ("**pKa-dominant model matches v1.1 within noise** "
                f"(M2 MAE={m2:.3f}, M3 MAE={m3:.3f}, Δ={delta:+.3f}).")
    if delta > 0:
        return ("**pKa-dominant model is a useful baseline but underperforms v1.1 by "
                f"{delta:.3f} log units** (M2 MAE={m2:.3f}, M3 MAE={m3:.3f}).")
    return ("**pKa-dominant model outperforms v1.1 — investigate why the 3D-descriptor "
            f"stack was overfitting** (M2 MAE={m2:.3f}, M3 MAE={m3:.3f}, Δ={delta:+.3f}).")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("pka_flux_v11 — pKa-dominant flux experiment")
    print("=" * 78)

    bio, pka_tbl, pka_loo = load_tables()
    print(f"bio: {bio.shape}    pKa v21: {pka_tbl.shape}    v91 LOO preds: {pka_loo.shape}")

    pka_pred_df = generate_predicted_pka(bio, pka_tbl, pka_loo)
    merged = bio.merge(
        pka_pred_df[["row_id", "predicted_pKa", "pka_source"]],
        on="row_id", how="left",
    )

    preds, _m2_model, m2_cols, _X2 = run_eval(merged)
    preds.to_csv(OUT / "predictions.csv", index=False)
    print(f"[eval] wrote {len(preds)} predictions to predictions.csv")

    metrics = compute_metrics(preds)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[metrics] pooled:")
    for m, b in metrics["pooled"].items():
        print(f"  {m}: MAE={b.get('MAE', float('nan')):.4f}  "
              f"R²={b.get('R2', float('nan')):.3f}  "
              f"Spearman={b.get('Spearman', float('nan')):.3f}")

    plot_calibration(preds, OUT / "calibration.png")
    print("[plot] calibration.png written")

    shap_ranking = shap_m2(preds, merged, OUT / "shap_M2.png")
    print("[plot] shap_M2.png written")
    top5 = ", ".join(f"{n}({v:.3f})" for n, v in shap_ranking[:5])
    print(f"[shap] top-5: {top5}")

    optima_df = inverted_u_fits(preds, OUT / "family_optima.csv")
    print(f"[fits] inverted-U: {len(optima_df)} families")

    # Sanity numbers for the report
    pka_have = pka_pred_df.dropna(subset=["measured_pKa_v21"])
    pka_mae = ((pka_have["predicted_pKa"] - pka_have["measured_pKa_v21"]).abs().mean()
                if len(pka_have) else float("nan"))
    n_loo = int((pka_pred_df["pka_source"] == "v21_loo").sum())
    n_full = int((pka_pred_df["pka_source"] == "v91_full_bundle").sum())

    # Family mismatch + M3 coverage stats for the caveats block
    fam_mm = pka_pred_df.dropna(subset=["family_v21"])
    n_family_mismatch = int((fam_mm["family_v21"] != fam_mm["family_table"]).sum()) \
                          if len(fam_mm) else 0
    n_m3_unmatched = int(preds["M3"].isna().sum())
    m3_n = int(preds["M3"].notna().sum())
    total_n = int(len(preds))

    pooled_md = md_table(metrics["pooled"], ["M0", "M0_quad", "M1", "M2", "M3"])
    fam_md = family_md_table(metrics["per_family"], ["M0", "M1", "M2", "M3"])
    optima_md = optima_md_table(optima_df)
    shap_md = shap_md_table(shap_ranking)
    verdict = pick_verdict(metrics["pooled"])

    report = REPORT_TEMPLATE.format(
        n_rows=len(preds),
        n_loo=n_loo, n_full=n_full,
        pka_mae=pka_mae,
        noise=NOISE_FLOOR,
        tail_cols=", ".join(TAIL_DESCRIPTORS),
        pooled_table=pooled_md,
        family_table=fam_md,
        optima_table=optima_md,
        shap_table=shap_md,
        n_family_mismatch=n_family_mismatch,
        n_m3_unmatched=n_m3_unmatched,
        m3_n=m3_n,
        total_n=total_n,
        verdict=verdict,
    )
    (OUT / "REPORT.md").write_text(report)
    print(f"[report] REPORT.md written ({len(report)} chars)")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()