"""
train_v15_physics_ml.py — train the v15 hybrid: ML (v14 stacker) + physics (new
descriptors) blended with LOO-fit weights, plus uncertainty quantification
from inter-layer disagreement.

Pipeline:
  1. Load 273-row bioact xlsx (real measured log10_flux_total)
  2. Compute physics features per compound via physics_features.compute_all
  3. Pull LOO predictions for the ML side:
       - v14+stacker (already cached in bioact_loo_components.npz)
  4. Fit a physics-only Ridge regression on the physics features → log10_flux
     (LOO-validated)
  5. Stack the two LOO prediction vectors and learn blend weights via
     constrained convex optimization (simplex)
  6. Use the LOO residual std as σ for each layer; combined σ adds in
     quadrature for UCB acquisition: score = ŷ + κ · σ
  7. Save IAJD_master/bundles_caches/v15_hybrid_bundle.joblib for inference

Outputs:
  v15_hybrid_bundle.joblib   — ML weights, physics model, blend weights, σ's
  v15_loo_report.json        — per-layer + combined LOO MAE
  preds_v15_final.npy        — LOO blended predictions (for downstream caches)
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physics_features import compute_all_physics_features

BIO_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
LOO_BIOACT = ROOT / "bioact_loo_components.npz"
V14_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"
STACKER_BUNDLE = ROOT / "IAJD_master/bundles_caches/bioact_stacker_bundle.pkl"
PKA_CACHE = ROOT / "v11_pka_flux/predicted_pka_cache.csv"

OUT_BUNDLE = ROOT / "IAJD_master/bundles_caches/v15_hybrid_bundle.joblib"
OUT_REPORT = ROOT / "v15_loo_report.json"
OUT_PREDS = ROOT / "preds_v15_final.npy"

PHYSICS_COLS = [
    "cpp_geometric", "lamellar_d_nm", "logKp_membrane",
    "protonation_endosome", "protonation_cytosol",
    "endosomal_escape_score", "cmc_log10_molar", "hlb_griffin",
    # W-A/W-B/W-C grounded extensions — real QM + MD-derived columns. NaN
    # passthrough downstream; the Ridge re-fit absorbs scale differences.
    "md_a_head_prot_nm2", "md_delta_a_head_nm2",
    "md_cpp_prot", "md_delta_cpp",
    "md_bilayer_thick_nm", "md_order_param",
    "qm_q_ionizableN", "qm_dipole_D",
    "qm_dGsolv_kJmol", "qm_homo_lumo_eV",
    "dG_escape_helfrich",
]


def _physics_matrix(df: pd.DataFrame, pka_lookup: dict) -> np.ndarray:
    """Compute physics features per row; return matrix [n × k].

    No-proxy policy: pKa is REQUIRED to be real (from cache or row). If a
    row has no real pKa, its protonation/escape/Manning rows stay NaN —
    they're NOT filled with family-median estimates. Final imputation uses
    column medians from the rows that DO have real values (so the Ridge
    can still train), but each NaN is logged as "physics not fully real
    for row X". head_group is pulled from the bioact row, so a_head is
    per-molecule.
    """
    rows = []
    n_nan_pka = 0
    for _, r in df.iterrows():
        smi = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(smi):
            rows.append({k: np.nan for k in PHYSICS_COLS})
            continue
        # Real pKa only: cache lookup OR measured pKa column. No family-median.
        pka = pka_lookup.get(int(r["row_id"]), np.nan) if "row_id" in r else np.nan
        if not np.isfinite(pka):
            pka_meas = r.get("pKa")
            pka = float(pka_meas) if pd.notna(pka_meas) else np.nan
        if not np.isfinite(pka):
            n_nan_pka += 1
        linker_n = int(r["linker_length"]) if pd.notna(r.get("linker_length")) else None
        n_chains = 3 if "Tris" in str(r.get("family", "")) else 2
        head_group = r.get("head_group") if pd.notna(r.get("head_group")) else None
        # Pull MD/QM inputs (real or NaN) from physics_cache_io so the
        # Helfrich-form escape is enabled where MD data is cached.
        try:
            from physics_cache_io import load_physics
            pr = load_physics(str(smi), head_group=head_group,
                               pka=pka if np.isfinite(pka) else None)
            md_c0  = pr.md.get("md_c0_spontaneous", float("nan"))
            md_t   = pr.md.get("md_bilayer_thick_nm", float("nan"))
            md_a_p = pr.md.get("md_a_head_prot_nm2", float("nan"))
        except Exception:
            md_c0 = md_t = md_a_p = float("nan")
            pr = None
        feats = compute_all_physics_features(
            str(smi), pka=pka, linker_length=linker_n,
            n_tail_chains=n_chains, chain_avg_carbons=10.0,
            head_group=head_group,
            md_c0_spontaneous=md_c0,
            md_bilayer_thick_nm=md_t,
            md_a_head_prot_nm2=md_a_p,
        )
        # Merge the W-A/W-B columns the new PHYSICS_COLS schema expects.
        if pr is not None:
            for k in ("md_a_head_prot_nm2", "md_delta_a_head_nm2",
                      "md_cpp_prot", "md_delta_cpp",
                      "md_bilayer_thick_nm", "md_order_param"):
                feats[k] = pr.md.get(k, float("nan"))
            for k in ("qm_q_ionizableN", "qm_dipole_D",
                      "qm_dGsolv_kJmol", "qm_homo_lumo_eV"):
                feats[k] = pr.qm.get(k, float("nan"))
            feats["dG_escape_helfrich"] = pr.dG_escape_helfrich
        rows.append({k: feats.get(k, np.nan) for k in PHYSICS_COLS})
    print(f"  rows without real pKa: {n_nan_pka}/{len(df)}", flush=True)
    X = pd.DataFrame(rows)[PHYSICS_COLS].values.astype(float)
    X[~np.isfinite(X)] = np.nan
    # Median imputation only for training-data column gaps; this is a fit
    # convenience, not a proxy substitution at inference.
    for c in range(X.shape[1]):
        m = ~np.isfinite(X[:, c])
        if m.any():
            med = np.nanmedian(X[:, c])
            X[m, c] = med if np.isfinite(med) else 0.0
    return X


def physics_loo(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """LOO physics-only Ridge predictions."""
    n = len(y)
    preds = np.zeros(n)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = Ridge(alpha=alpha).fit(Xs[mask], y[mask])
        preds[i] = float(m.predict(Xs[i:i+1])[0])
    return preds


def get_ml_loo(df: pd.DataFrame) -> np.ndarray:
    """Pull the v14+stacker LOO predictions for each bioact row.

    Stacker LOO is reconstructed from bioact_loo_components.npz: the stacker's
    OOF prediction = stacker.predict([direct, analog, lion, admet]_loo).
    """
    import pickle
    d = np.load(LOO_BIOACT, allow_pickle=True)
    direct = d["direct"]; analog = d["analog"]
    lion = d["lion"]; admet = d["admet"]
    y_loo = d["y_true"]
    with open(STACKER_BUNDLE, "rb") as f:
        stk = pickle.load(f)
    X_stack = np.column_stack([direct, analog, lion, admet])
    preds = stk["stacker"].predict(X_stack)
    return preds, y_loo


def optimize_blend(P: np.ndarray, y: np.ndarray):
    """Simplex blend weights minimizing LOO MAE."""
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * P.shape[1]
    starts = [np.ones(P.shape[1])/P.shape[1],
              np.array([0.8, 0.2]),
              np.array([0.5, 0.5]),
              np.array([0.95, 0.05]),
              np.array([0.7, 0.3])]
    best_w = None; best_loss = float("inf")
    for w0 in starts:
        r = minimize(lambda w: float(np.mean(np.abs(y - P @ w))),
                      w0, method="SLSQP", bounds=bnds, constraints=cons,
                      options={"maxiter": 500, "ftol": 1e-8})
        if r.fun < best_loss:
            best_loss, best_w = r.fun, r.x
    return best_w, best_loss


def main():
    print("="*70)
    print("[v15] training hybrid ML+physics bioactivity predictor")
    print("="*70)
    df_bio = pd.read_excel(BIO_XLSX)
    print(f"  bioact xlsx: {df_bio.shape}", flush=True)

    # Pull stacker LOO preds; these are aligned to a 247-row subset (bioact rows
    # with non-null log10_flux_total + non-null SMILES_canonical).
    print("\n[1/4] loading v14+stacker LOO predictions…", flush=True)
    ml_loo, y_loo = get_ml_loo(df_bio)
    print(f"  stacker LOO: {len(ml_loo)} predictions, MAE={np.mean(np.abs(ml_loo - y_loo)):.4f}",
          flush=True)

    # Match the 247-row subset back to df_bio for physics-feature computation
    bio_used = df_bio.dropna(subset=["log10_flux_total"]).reset_index(drop=True)
    bio_used = bio_used[bio_used["SMILES_canonical"].notna()].reset_index(drop=True)
    assert len(bio_used) == len(ml_loo), \
           f"row count mismatch: {len(bio_used)} vs {len(ml_loo)}"

    # Load predicted_pKa per row_id
    pka_df = pd.read_csv(PKA_CACHE)
    pka_lookup = dict(zip(pka_df["row_id"].astype(int), pka_df["predicted_pKa"]))

    print("\n[2/4] computing physics features for 247 rows…", flush=True)
    X_phys = _physics_matrix(bio_used, pka_lookup)
    print(f"  physics matrix: {X_phys.shape}", flush=True)
    print(f"  physics feature ranges:", flush=True)
    for j, col in enumerate(PHYSICS_COLS):
        print(f"    {col:<28s}  [{np.nanmin(X_phys[:,j]):>+8.3f}, "
              f"{np.nanmax(X_phys[:,j]):>+8.3f}]", flush=True)

    print("\n[3/4] LOO physics-only Ridge…", flush=True)
    phys_loo = physics_loo(X_phys, y_loo, alpha=1.0)
    phys_mae = float(np.mean(np.abs(phys_loo - y_loo)))
    print(f"  physics LOO MAE: {phys_mae:.4f}", flush=True)

    print("\n[4/4] optimizing ML+physics blend…", flush=True)
    P = np.column_stack([ml_loo, phys_loo])
    w, blend_mae = optimize_blend(P, y_loo)
    print(f"  optimal weights: ml={w[0]:.3f}  physics={w[1]:.3f}", flush=True)
    print(f"  blend LOO MAE:   {blend_mae:.4f}", flush=True)

    blend = P @ w
    # Per-layer residual std as σ
    sigma_ml   = float(np.std(ml_loo - y_loo))
    sigma_phys = float(np.std(phys_loo - y_loo))
    sigma_blend = float(np.std(blend - y_loo))
    # Inter-layer disagreement σ for UCB
    sigma_disagreement = float(np.std(ml_loo - phys_loo))
    print(f"\n  σ_ml={sigma_ml:.3f}  σ_phys={sigma_phys:.3f}  "
          f"σ_blend={sigma_blend:.3f}  σ_disagreement={sigma_disagreement:.3f}",
          flush=True)

    # Fit a FINAL physics model on all 247 rows for inference
    sc = StandardScaler().fit(X_phys)
    final_physics = Ridge(alpha=1.0).fit(sc.transform(X_phys), y_loo)

    bundle = {
        "version": "v15_hybrid_2026-05-28",
        "weights": {"ml": float(w[0]), "physics": float(w[1])},
        "physics_model": final_physics,
        "physics_scaler": sc,
        "physics_cols": PHYSICS_COLS,
        "sigma": {
            "ml": sigma_ml, "physics": sigma_phys,
            "blend": sigma_blend, "disagreement": sigma_disagreement,
        },
        "metrics": {
            "n_train": len(y_loo),
            "loo_mae_ml": float(np.mean(np.abs(ml_loo - y_loo))),
            "loo_mae_physics": phys_mae,
            "loo_mae_blend": blend_mae,
            "improvement_vs_ml": float(np.mean(np.abs(ml_loo - y_loo))) - blend_mae,
        },
    }
    joblib.dump(bundle, OUT_BUNDLE)
    np.save(OUT_PREDS, blend)
    OUT_REPORT.write_text(json.dumps(bundle["metrics"] | bundle["sigma"], indent=2, default=str))
    print(f"\n  saved {OUT_BUNDLE.name}", flush=True)
    print(f"  saved {OUT_PREDS.name}", flush=True)
    print(f"  saved {OUT_REPORT.name}", flush=True)

    # Per-family breakdown
    families = bio_used["family"].values
    print(f"\n  per-family blend MAE:", flush=True)
    for fam in sorted(set(families)):
        mask = families == fam
        mae = float(np.mean(np.abs(blend[mask] - y_loo[mask])))
        n = int(mask.sum())
        print(f"    {fam:<22s}  n={n:>3}  MAE={mae:.3f}", flush=True)


if __name__ == "__main__":
    main()
