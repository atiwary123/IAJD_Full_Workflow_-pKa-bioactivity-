"""
tune_alpha_blend.py — find the α that maximizes ranking quality of the
blended proposer score against measured log10_flux_total on the training set.

The proposer's blended score is:
    score(α) = (1-α) · score_ML  +  α · (score_phys + monotone_bonus)
    score_ML   = P(≥T_default) · max(0, ŷ_LOO − mean_y_train)
    score_phys = Q_physics · max(0, (ŷ_LOO + κ·σ) − mean_y_train)
    monotone_bonus = Σ |ρ_i| · min(0.3, step_past_boundary) over reliably-monotone features

We evaluate Spearman rank correlation between score(α) and the actual
log10_flux_total, both on the FULL training set and on the TOP QUARTILE
(the extrapolation-relevant regime). The optimal α is the one that ranks
the highest-flux compounds at the top of the list.

If pure-ML (α=0) wins on the full set but loses on the top quartile, that
confirms the physics layer helps where it's supposed to: at the high-flux
frontier where ML extrapolation breaks down.

Outputs:
  alpha_tuning_report.json — per-α metrics
  Pushes the optimal α as the new default in app.py.
"""
from __future__ import annotations
import json, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physics_features import compute_all_physics_features

BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
LOO_BIOACT = ROOT / "bioact_loo_components.npz"
STACKER_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_stacker_bundle.pkl"
V15_BUNDLE = ROOT / "IAJD_master/bundles_caches/v15_hybrid_bundle.joblib"
BIN_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_binary_bundle.pkl"
MONOTONE_JSON = ROOT / "monotone_axes.json"
OUT_JSON = ROOT / "alpha_tuning_report.json"

DEFAULT_T = 8.0
KAPPA = 1.5


def _ml_loo_predictions(df_train: pd.DataFrame):
    """Pull LOO ML predictions from stacker bundle."""
    d = np.load(LOO_BIOACT, allow_pickle=True)
    direct = d["direct"]; analog = d["analog"]
    lion = d["lion"]; admet = d["admet"]
    y_loo = d["y_true"]
    with open(STACKER_BUNDLE, "rb") as f:
        stk = pickle.load(f)
    X_stack = np.column_stack([direct, analog, lion, admet])
    preds = stk["stacker"].predict(X_stack)
    return preds, y_loo


def _p_above(yhat: np.ndarray, threshold: float, bin_bundle):
    """Binary head probability via the per-threshold sigmoid calibrator."""
    grid = sorted(bin_bundle["threshold_calibrators"].keys(), key=float)
    grid_floats = np.array([float(g) for g in grid])
    nearest = grid[int(np.argmin(np.abs(grid_floats - threshold)))]
    c = bin_bundle["threshold_calibrators"][nearest]
    a, b = c["a"], c["b"]
    return 1.0 / (1.0 + np.exp(-(a * (yhat - threshold) + b)))


def _physics_quality_array(df_train: pd.DataFrame) -> np.ndarray:
    """Compute Q_physics for each row."""
    Q = np.zeros(len(df_train))
    for i, (_, r) in enumerate(df_train.iterrows()):
        smi = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(smi):
            Q[i] = 0.5
            continue
        fam = r.get("family", "GA-Tris")
        linker_n = int(r["linker_length"]) if pd.notna(r.get("linker_length")) else 4
        n_chains = 3 if ("Tris" in str(fam) or fam == "PE-Gallic") else 2
        pka = float(r.get("pKa", 6.5)) if pd.notna(r.get("pKa")) else 6.5
        feats = compute_all_physics_features(
            str(smi), pka=pka, linker_length=linker_n,
            n_tail_chains=n_chains, chain_avg_carbons=10.0,
        )
        cpp = feats.get("cpp_geometric", 1.0)
        if not np.isfinite(cpp): cpp = 1.0
        cpp_score = float(np.exp(-((cpp - 1.0) ** 2) / (2 * 0.3 ** 2)))
        p_e = feats.get("protonation_endosome", 0.5)
        p_c = feats.get("protonation_cytosol", 0.1)
        if not np.isfinite(p_e): p_e = 0.5
        if not np.isfinite(p_c): p_c = 0.1
        escape = max(0.0, p_e - p_c)
        hlb = feats.get("hlb_griffin", 8.5)
        if not np.isfinite(hlb): hlb = 8.5
        hlb_score = float(np.exp(-((hlb - 8.5) ** 2) / (2 * 3.0 ** 2)))
        logKp = feats.get("logKp_membrane", 5.0)
        if not np.isfinite(logKp): logKp = 5.0
        membrane_score = 1.0 / (1.0 + np.exp(-(logKp - 4.0)))
        Q[i] = float(np.mean([cpp_score, escape, hlb_score, membrane_score]))
    return Q


def _monotone_bonus_array(df_train: pd.DataFrame) -> np.ndarray:
    """Compute monotone bonus for each row (uses pre-computed RDKit cols)."""
    axes_data = json.loads(MONOTONE_JSON.read_text())
    axes = {k: v for k, v in axes_data["axes"].items() if v.get("reliably_monotone")}
    bonuses = np.zeros(len(df_train))
    for i, (_, r) in enumerate(df_train.iterrows()):
        bonus = 0.0
        for feat, info in axes.items():
            if feat not in df_train.columns:
                continue
            x = r.get(feat)
            if pd.isna(x):
                continue
            x = float(x)
            train_min = info["train_min"]; train_max = info["train_max"]
            rho = info.get("spearman_rho") or 0.0
            direction = info.get("direction", "increasing")
            train_range = train_max - train_min
            if train_range <= 0: continue
            if direction == "increasing" and x > train_max:
                step = (x - train_max) / train_range
                bonus += abs(rho) * min(0.3, step)
            elif direction == "decreasing" and x < train_min:
                step = (train_min - x) / train_range
                bonus += abs(rho) * min(0.3, step)
        bonuses[i] = min(bonus, 0.6)
    return bonuses


def main():
    print("Loading training data + bundles…", flush=True)
    df = pd.read_excel(BIO_XLSX)
    df_used = df.dropna(subset=["log10_flux_total"]).reset_index(drop=True)
    df_used = df_used[df_used["SMILES_canonical"].notna()].reset_index(drop=True)
    print(f"  n_train = {len(df_used)}", flush=True)

    print("Computing LOO ML predictions…", flush=True)
    yhat_loo, y_loo = _ml_loo_predictions(df_used)
    assert len(yhat_loo) == len(df_used), \
        f"row count mismatch: {len(yhat_loo)} vs {len(df_used)}"
    y = df_used["log10_flux_total"].values.astype(float)

    print("Loading binary head + computing P(≥T)…", flush=True)
    with open(BIN_BUNDLE, "rb") as f:
        bin_b = pickle.load(f)
    P_loo = _p_above(yhat_loo, DEFAULT_T, bin_b)

    print("Computing Q_physics for all training rows…", flush=True)
    Q = _physics_quality_array(df_used)

    print("Computing monotone bonus for all training rows…", flush=True)
    mono = _monotone_bonus_array(df_used)

    print("Loading v15 σ_combined…", flush=True)
    v15 = joblib.load(V15_BUNDLE)
    sigma = float(np.sqrt(v15["sigma"]["ml"] ** 2 + v15["sigma"]["disagreement"] ** 2))
    print(f"  σ_combined = {sigma:.3f}", flush=True)

    # Reference baseline for Δ — use training mean (no specific seed)
    mean_y = float(np.mean(y))
    print(f"  mean_y_train = {mean_y:.3f}", flush=True)

    print("\nSweeping α from 0 to 1…", flush=True)
    alphas = np.linspace(0.0, 1.0, 21)
    results = []
    for alpha in alphas:
        score_ml = P_loo * np.maximum(0.0, yhat_loo - mean_y)
        ucb = yhat_loo + KAPPA * sigma
        score_phys = Q * np.maximum(0.0, ucb - mean_y) + mono
        score = (1 - alpha) * score_ml + alpha * score_phys

        rho_full, _ = spearmanr(score, y)
        # Top quartile = compounds in the top 25% by flux
        q75 = np.quantile(y, 0.75)
        top_mask = y >= q75
        rho_top, _ = spearmanr(score[top_mask], y[top_mask])
        # MAE between scaled score and y? Not meaningful since score isn't calibrated.
        # Instead: precision@10 — fraction of top-10-by-score that are also top-10-by-flux
        top10_by_score = np.argsort(-score)[:10]
        top10_by_y = set(np.argsort(-y)[:10])
        prec10 = len(set(top10_by_score) & top10_by_y) / 10.0
        # NDCG@20
        order = np.argsort(-score)[:20]
        dcg = sum(y[order[k]] / np.log2(k + 2) for k in range(len(order)))
        ideal_order = np.argsort(-y)[:20]
        idcg = sum(y[ideal_order[k]] / np.log2(k + 2) for k in range(len(ideal_order)))
        ndcg20 = dcg / idcg if idcg > 0 else 0.0
        results.append({
            "alpha": float(alpha),
            "spearman_full": float(rho_full),
            "spearman_top_quartile": float(rho_top) if rho_top == rho_top else None,
            "precision_at_10": prec10,
            "ndcg_at_20": float(ndcg20),
        })

    print("\n  α     Spearman(full)   Spearman(top-Q)   Precision@10   NDCG@20", flush=True)
    print("  " + "─" * 70, flush=True)
    for r in results:
        sq = r["spearman_top_quartile"]
        sq_s = f"{sq:+.4f}" if sq is not None else "    —    "
        print(f"  {r['alpha']:.2f}    {r['spearman_full']:+.4f}        "
              f"{sq_s}        {r['precision_at_10']:.2f}           "
              f"{r['ndcg_at_20']:.4f}", flush=True)

    # Pick optimal α: combination of full-set Spearman + top-quartile Spearman.
    # The proposer's actual use case is finding high-flux candidates, so weight
    # the top-quartile heavily.
    def score_alpha(r):
        sq = r["spearman_top_quartile"]
        if sq is None: sq = 0.0
        return 0.4 * r["spearman_full"] + 0.4 * sq + 0.2 * r["ndcg_at_20"]

    optimal = max(results, key=score_alpha)
    print(f"\nOPTIMAL α = {optimal['alpha']:.2f}", flush=True)
    print(f"  Spearman(full) = {optimal['spearman_full']:+.4f}", flush=True)
    print(f"  Spearman(top-Q) = {optimal['spearman_top_quartile']:+.4f}", flush=True)
    print(f"  Precision@10  = {optimal['precision_at_10']:.2f}", flush=True)
    print(f"  NDCG@20       = {optimal['ndcg_at_20']:.4f}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "kappa": KAPPA, "default_T": DEFAULT_T, "sigma_combined": sigma,
        "results": results, "optimal": optimal,
    }, indent=2, default=str))
    print(f"\n  wrote {OUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
