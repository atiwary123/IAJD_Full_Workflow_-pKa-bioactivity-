"""pKa-primary bioactivity model — no similarity, no stacker.

Architecture:
  SMILES → pKa(v9.1) → per-family pKa-flux curve → base_flux
  SMILES → RDKit structural features → XGBoost → residual_correction
  final_flux = base_flux + residual_correction

The pKa-flux curve is a fitted quadratic (inverted-U) per family.
The residual corrector uses only direct molecular descriptors,
no Tanimoto similarity, no neighbor averaging.

Evaluation:
  - LOO on full training set (361 rows)
  - Out-of-sample on novel IAJDs (if SMILES provided)
"""
from __future__ import annotations
import sys, os, warnings, json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, AllChem, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "IAJD_master" / "datasets"
CODE = ROOT / "IAJD_master" / "code"

sys.path.insert(0, str(CODE))

# ── pKa-flux curve ──────────────────────────────────────────────────────

def inverted_u(pka, a, b, c):
    """Quadratic: flux = a*(pka - b)^2 + c.  a < 0 → inverted U."""
    return a * (pka - b) ** 2 + c

def fit_pka_flux_curves(pka_vals, flux_vals, families):
    """Fit per-family inverted-U pKa→flux curves."""
    curves = {}
    for fam in sorted(set(families)):
        mask = (families == fam) & np.isfinite(pka_vals) & np.isfinite(flux_vals)
        n = mask.sum()
        if n < 4:
            curves[fam] = {"type": "mean", "value": float(np.mean(flux_vals[mask])), "n": n}
            continue
        x, y = pka_vals[mask], flux_vals[mask]
        try:
            popt, _ = curve_fit(inverted_u, x, y, p0=[-2.0, np.mean(x), np.max(y)],
                                maxfev=5000)
            resid = y - inverted_u(x, *popt)
            curves[fam] = {
                "type": "quadratic",
                "a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]),
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "n": int(n),
            }
        except Exception:
            curves[fam] = {"type": "mean", "value": float(np.mean(y)), "n": int(n)}
    return curves

def predict_curve(pka, family, curves):
    """Predict base flux from pKa using fitted curve."""
    if family not in curves:
        return np.nanmean([c.get("value", c.get("c", 7.5)) for c in curves.values()])
    c = curves[family]
    if c["type"] == "quadratic":
        return inverted_u(pka, c["a"], c["b"], c["c"])
    return c["value"]


# ── Structural features (no similarity) ─────────────────────────────────

def extract_structural_features(smiles: str) -> dict:
    """Extract structural descriptors directly from SMILES. No fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    d = {}
    d["MolLogP"] = Crippen.MolLogP(mol)
    d["TPSA"] = Descriptors.TPSA(mol)
    d["HeavyAtomCount"] = float(mol.GetNumHeavyAtoms())
    d["FractionCSP3"] = Lipinski.FractionCSP3(mol)
    d["RotatableBonds"] = float(Lipinski.NumRotatableBonds(mol))
    d["NumHDonors"] = float(Lipinski.NumHDonors(mol))
    d["NumHAcceptors"] = float(Lipinski.NumHAcceptors(mol))
    d["NumAromaticRings"] = float(Lipinski.NumAromaticRings(mol))
    d["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
    d["NumEthers"] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4][OX2][CX4]"))))
    d["NumEsters"] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2]"))))
    d["NumAmides"] = float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))))

    # Head group polarity proxy: count OH groups on piperazine N
    d["NumOH_on_head"] = float(len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("[NX3]CCO"))))

    # Linker length: count sp3 carbons between C=O and first piperazine N
    linker_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3][CX4]")
    d["has_ester_linker"] = 1.0 if mol.HasSubstructMatch(linker_pat) else 0.0

    # Approximate linker carbon count from chain between ester and amine
    d["ExactMolWt"] = Descriptors.ExactMolWt(mol)

    # Gasteiger charge on most basic N
    try:
        m2 = Chem.RWMol(mol)
        AllChem.ComputeGasteigerCharges(m2)
        charges = []
        for atom in m2.GetAtoms():
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 3:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isfinite(q):
                    charges.append(q)
        d["GasteigerN_max"] = max(charges) if charges else 0.0
    except Exception:
        d["GasteigerN_max"] = 0.0

    return d

STRUCT_FEATURE_NAMES = [
    "MolLogP", "TPSA", "HeavyAtomCount", "FractionCSP3", "RotatableBonds",
    "NumHDonors", "NumHAcceptors", "NumAromaticRings", "LabuteASA",
    "NumEthers", "NumEsters", "NumAmides", "NumOH_on_head",
    "has_ester_linker", "ExactMolWt", "GasteigerN_max",
]


# ── Main model ──────────────────────────────────────────────────────────

RESIDUAL_XGB_HP = dict(
    n_estimators=100, max_depth=3, learning_rate=0.08,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
    reg_lambda=3.0, random_state=42, n_jobs=1,
)


def build_and_evaluate(novel_data: Optional[pd.DataFrame] = None):
    """Build pKa-primary model, evaluate LOO + novels.

    Novel GA-Tris IAJDs (347, 348, 365, 366, 367, 369, 372, 373) reintegrated 2026-05-28.
    """
    # Load training data
    bio = pd.read_excel(DATA / "IAJD_Bioact_v13_clean.xlsx")
    pka_cache = pd.read_csv(ROOT / "v11_pka_flux" / "predicted_pka_cache.csv")

    merged = bio.merge(
        pka_cache[["row_id", "predicted_pKa"]],
        on="row_id", how="left"
    )

    valid = merged.dropna(subset=["log10_flux_total", "predicted_pKa"]).copy()
    print(f"Training set: {len(valid)} rows with both flux + predicted pKa")

    families = valid["family"].values
    pka_vals = valid["predicted_pKa"].values
    flux_vals = valid["log10_flux_total"].values

    # Extract structural features for all training compounds
    print("Extracting structural features...")
    feat_rows = []
    for _, row in valid.iterrows():
        smi = row.get("SMILES_canonical") or row.get("SMILES")
        if pd.isna(smi):
            feat_rows.append({k: np.nan for k in STRUCT_FEATURE_NAMES})
            continue
        feat_rows.append(extract_structural_features(str(smi)))
    X_struct = pd.DataFrame(feat_rows)[STRUCT_FEATURE_NAMES].values
    # Impute NaN
    for col in range(X_struct.shape[1]):
        nan_mask = ~np.isfinite(X_struct[:, col])
        if nan_mask.any():
            X_struct[nan_mask, col] = np.nanmedian(X_struct[:, col])

    # ── LOO evaluation ──────────────────────────────────────────────────
    print("\n=== LOO EVALUATION ===")
    n = len(valid)
    loo_curve_preds = np.full(n, np.nan)
    loo_final_preds = np.full(n, np.nan)

    # Use stratified 5-fold (full LOO on 336 compounds is expensive)
    fam_labels = np.array([f if f in ['sSS-Nonsym','PE-Tris','GA-Tris','Dialkoxybenzyl','PE-Gallic']
                           else 'other' for f in families])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_struct, fam_labels)):
        # Fit curves on training fold
        curves = fit_pka_flux_curves(pka_vals[train_idx], flux_vals[train_idx],
                                     families[train_idx])

        # Curve predictions
        curve_preds_train = np.array([predict_curve(p, f, curves)
                                       for p, f in zip(pka_vals[train_idx], families[train_idx])])
        curve_preds_test = np.array([predict_curve(p, f, curves)
                                      for p, f in zip(pka_vals[test_idx], families[test_idx])])

        # Residuals for training
        residuals_train = flux_vals[train_idx] - curve_preds_train

        # Build feature matrix: pKa + structural features
        X_train = np.column_stack([pka_vals[train_idx], X_struct[train_idx]])
        X_test = np.column_stack([pka_vals[test_idx], X_struct[test_idx]])

        # Train residual corrector
        xgb_model = xgb.XGBRegressor(**RESIDUAL_XGB_HP)
        xgb_model.fit(X_train, residuals_train, verbose=False)

        # Predict
        resid_preds = xgb_model.predict(X_test)
        final_preds = curve_preds_test + resid_preds

        loo_curve_preds[test_idx] = curve_preds_test
        loo_final_preds[test_idx] = final_preds

    # Metrics
    valid_mask = np.isfinite(loo_final_preds)
    curve_mae = mean_absolute_error(flux_vals[valid_mask], loo_curve_preds[valid_mask])
    final_mae = mean_absolute_error(flux_vals[valid_mask], loo_final_preds[valid_mask])
    curve_resid = flux_vals[valid_mask] - loo_curve_preds[valid_mask]
    final_resid = flux_vals[valid_mask] - loo_final_preds[valid_mask]

    ss_tot = np.sum((flux_vals[valid_mask] - np.mean(flux_vals[valid_mask]))**2)
    curve_r2 = 1 - np.sum(curve_resid**2) / ss_tot
    final_r2 = 1 - np.sum(final_resid**2) / ss_tot

    print(f"\n  POOLED (n={valid_mask.sum()}):")
    print(f"    Curve-only MAE: {curve_mae:.4f}  R²: {curve_r2:.4f}")
    print(f"    + Corrector MAE: {final_mae:.4f}  R²: {final_r2:.4f}")

    print(f"\n  PER-FAMILY:")
    for fam in sorted(set(families)):
        fam_mask = valid_mask & (families == fam)
        if fam_mask.sum() < 3:
            continue
        fam_curve_mae = mean_absolute_error(flux_vals[fam_mask], loo_curve_preds[fam_mask])
        fam_final_mae = mean_absolute_error(flux_vals[fam_mask], loo_final_preds[fam_mask])
        mean_bias = np.mean(flux_vals[fam_mask] - loo_final_preds[fam_mask])
        print(f"    {fam:25s} n={fam_mask.sum():>3}  curve={fam_curve_mae:.3f}  final={fam_final_mae:.3f}  bias={mean_bias:+.3f}")

    # ── Novel evaluation ────────────────────────────────────────────────
    if novel_data is not None and len(novel_data) > 0:
        print(f"\n=== NOVEL COMPOUND EVALUATION (n={len(novel_data)}) ===")

        # Fit curves on FULL training set
        curves_full = fit_pka_flux_curves(pka_vals, flux_vals, families)
        curve_preds_full_train = np.array([predict_curve(p, f, curves_full)
                                            for p, f in zip(pka_vals, families)])
        residuals_full = flux_vals - curve_preds_full_train
        X_full = np.column_stack([pka_vals, X_struct])
        xgb_full = xgb.XGBRegressor(**RESIDUAL_XGB_HP)
        xgb_full.fit(X_full, residuals_full, verbose=False)

        for _, row in novel_data.iterrows():
            iajd = row["IAJD"]
            pred_pka = row["pred_pka"]
            exp_flux = row["exp_flux"]
            family = row.get("family", "GA-Tris")
            smiles = row.get("SMILES", None)

            curve_pred = predict_curve(pred_pka, family, curves_full)

            if pd.notna(smiles):
                feats = extract_structural_features(str(smiles))
                x_novel = np.array([[pred_pka] + [feats.get(f, 0) for f in STRUCT_FEATURE_NAMES]])
                for col in range(x_novel.shape[1]):
                    if not np.isfinite(x_novel[0, col]):
                        x_novel[0, col] = np.nanmedian(X_full[:, col])
                resid_pred = float(xgb_full.predict(x_novel)[0])
            else:
                resid_pred = 0.0  # no structural correction without SMILES

            final_pred = curve_pred + resid_pred
            delta = final_pred - exp_flux

            print(f"  IAJD {iajd:>3}: pKa={pred_pka:.2f}  curve={curve_pred:.2f}  "
                  f"+corr={resid_pred:+.2f}  final={final_pred:.2f}  "
                  f"exp={exp_flux:.2f}  Δ={delta:+.2f}")

        # Summary
        novel_finals = []
        for _, row in novel_data.iterrows():
            cp = predict_curve(row["pred_pka"], row.get("family", "GA-Tris"), curves_full)
            if pd.notna(row.get("SMILES")):
                feats = extract_structural_features(str(row["SMILES"]))
                x = np.array([[row["pred_pka"]] + [feats.get(f, 0) for f in STRUCT_FEATURE_NAMES]])
                for col in range(x.shape[1]):
                    if not np.isfinite(x[0, col]):
                        x[0, col] = np.nanmedian(X_full[:, col])
                rp = float(xgb_full.predict(x)[0])
            else:
                rp = 0.0
            novel_finals.append(cp + rp)

        novel_exp = novel_data["exp_flux"].values
        novel_preds = np.array(novel_finals)
        novel_mae = mean_absolute_error(novel_exp, novel_preds)
        novel_bias = np.mean(novel_preds - novel_exp)
        print(f"\n  Novel MAE: {novel_mae:.3f}")
        print(f"  Novel mean bias: {novel_bias:+.3f}")
        print(f"  Novel direction: {'ALL UNDER' if all(novel_preds < novel_exp) else 'MIXED'}")

    # Print curve parameters for reference
    print(f"\n=== FITTED pKa-FLUX CURVES ===")
    curves_final = fit_pka_flux_curves(pka_vals, flux_vals, families)
    for fam, c in sorted(curves_final.items()):
        if c["type"] == "quadratic":
            print(f"  {fam:25s} flux = {c['a']:.2f}*(pKa-{c['b']:.2f})² + {c['c']:.2f}  "
                  f"(RMSE={c['rmse']:.3f}, n={c['n']})")
        else:
            print(f"  {fam:25s} flux = {c['value']:.2f}  (mean, n={c['n']})")

    return curves_final


if __name__ == "__main__":
    # Novel compounds from the discrepancy report (no SMILES available)
    novel = pd.DataFrame({
        "IAJD": [347, 348, 365, 366, 367, 369, 372, 373],
        "exp_flux": [8.63, 9.10, 8.83, 9.04, 8.75, 9.39, 8.38, 8.84],
        "pred_pka": [6.75, 6.71, 6.73, 6.72, 6.72, 6.74, 6.70, 6.73],
        "family": ["GA-Tris"] * 8,
        "head": ["diol", "mono", "diol", "mono", "mono", "diol", "mono", "diol"],
        "SMILES": [None] * 8,  # not available — curve-only evaluation
        # Stacker predictions from the report for comparison
        "stacker_pred": [7.04, 7.67, 7.57, 7.04, 7.19, 7.23, 7.14, 6.92],
    })

    curves = build_and_evaluate(novel_data=novel)
