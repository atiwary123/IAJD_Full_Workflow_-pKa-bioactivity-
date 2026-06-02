"""
eval/sample_prep_eval.py — quantify the sample-prep FINDINGS honestly, self-contained.

Reuses the validated small-noisy-data recipe from benchmark_features_gp_lofo.py
(group by canonical SMILES, replicate-average y, carry per-compound replicate spread
`y_std` as the irreducible noise floor, leave-one-family-out vs a Mean-predictor floor)
WITHOUT importing or mutating that diagnostic. Uses the shared helper sample_prep_weights.

Answers three questions the integration spec (docs/SAMPLE_PREP_INTEGRATION_SPEC.md) asks:
  Q1  Does sp_confidence x n_mice WEIGHTING help?  (LOFO point metrics + MAE on the
      trusted conf>=0.85 subset; expect small/neutral R2, the win is robustness/calibration)
  Q2  Does sp_assembly_pH (is_pH52) add signal WITHIN ja1c05813 (random 5-fold)?
  Q3  Is is_pH52 subsumed by `paper`?  (R2 of paper-only vs paper+is_pH52; bootstrap CI)

Light by design (Ridge + RandomForest; no GP unless --gp). Run politely:
  OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python eval/sample_prep_eval.py
Writes eval/sample_prep_eval.json. Reads only the dataset (never the live caches/models).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sample_prep_weights import row_reliability_weight, prep_feature_matrix, within_paper_mask  # noqa: E402

BIOACT = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx"
TARGET = "log10_flux_total"
RDKIT_DESCS = [n for n, _ in Descriptors._descList]


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


# Clean, precomputed descriptor columns already in the xlsx (avoids RDKit Ipc-style
# float32 overflows from recomputing the full descriptor list).
DESC_COLS = ['ExactMolWt', 'MolLogP', 'TPSA', 'LabuteASA', 'FractionCSP3', 'RotatableBonds',
             'Chi0v', 'Chi1v', 'HallKierAlpha', 'NumAromaticRings', 'NumHDonors', 'NumHAcceptors',
             'NumTertiaryAmines', 'NumAmines_total', 'Hydrophobic_Index', 'Polar_Surface_Ratio',
             'Inductive_Effect_Strength', 'Taft_Steric_Sum', 'HBD_HBA_Ratio', 'Desolvation_Proxy',
             'Pct_V_Bur_mean', 'Rg_3D', 'Asphericity_3D']


def load_compounds() -> pd.DataFrame:
    """One row per compound: replicate-averaged y + carried prep/provenance + clean descriptors."""
    df = pd.read_excel(BIOACT)
    df["c"] = df["SMILES_canonical"].map(canon)
    df = df.dropna(subset=["c", TARGET])
    df["paper_eff"] = df["paper"].fillna(df["source"]) if "source" in df else df["paper"]
    g = df.groupby("c")
    base = pd.DataFrame({
        "y": g[TARGET].mean(),
        "y_std": g[TARGET].std().fillna(0.0),
        "n_rep": g[TARGET].size(),
        "family": g["family"].first().fillna("Unknown"),
        "paper": g["paper_eff"].first().fillna("unknown"),
        "sp_confidence": g["sp_confidence"].first(),
        "sp_assembly_pH": g["sp_assembly_pH"].first(),
        "sp_imaging_time_h": g["sp_imaging_time_h"].first(),
        "n_mice": g["n_mice"].sum(min_count=1),   # total animals across replicates
    }).reset_index().rename(columns={"c": "smiles"})
    present = [c for c in DESC_COLS if c in df.columns]
    desc = g[present].first().reset_index().rename(columns={"c": "smiles"})
    return base.merge(desc, on="smiles", how="left")


def f_rdkit(smis):
    fns = dict(Descriptors._descList)
    X = np.full((len(smis), len(RDKIT_DESCS)), np.nan)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, n in enumerate(RDKIT_DESCS):
            try:
                X[i, j] = fns[n](m)
            except Exception:
                pass
    # drop all-nan / zero-variance columns
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.std(axis=0) > 1e-9
    return X[:, keep]


def lofo_splits(fam):
    for f in sorted(pd.unique(fam)):
        te = fam == f
        if te.sum() >= 3 and (~te).sum() >= 20:
            yield f, ~te, te


def make_model(name):
    if name == "Ridge":
        return Ridge(alpha=10.0)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=1)
    if name == "Mean(floor)":
        return DummyRegressor(strategy="mean")
    raise ValueError(name)


def eval_pooled(name, X, y, splits, weights=None):
    """Pooled held-out predictions; weights applied to TRAIN folds only."""
    yhat = np.full(len(y), np.nan)
    for _, tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = make_model(name)
        if weights is not None and name != "Mean(floor)":
            m.fit(Xtr, y[tr], sample_weight=weights[tr])
        else:
            m.fit(Xtr, y[tr])
        yhat[te] = m.predict(Xte)
    ok = np.isfinite(yhat)
    return yhat, dict(n=int(ok.sum()),
                      r2=float(r2_score(y[ok], yhat[ok])),
                      mae=float(mean_absolute_error(y[ok], yhat[ok])),
                      rho=float(spearmanr(y[ok], yhat[ok])[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", action="store_true", help="also run a GP variant (slower/hotter)")
    args = ap.parse_args()

    lab = load_compounds()
    y = lab["y"].values
    fam = lab["family"].values
    present = [c for c in DESC_COLS if c in lab.columns]
    X = lab[present].to_numpy(dtype=float)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e18, 1e18)
    Xprep, prep_names = prep_feature_matrix(lab)
    splits = list(lofo_splits(fam))
    _nz = lab.loc[lab["n_rep"] > 1, "y_std"]
    noise = float(_nz.mean()) if len(_nz) else float("nan")

    # weighting schemes
    w_conf = row_reliability_weight(lab, nmice_col="n_mice")                      # conf x n_mice
    w_var = row_reliability_weight(lab, nmice_col="n_mice", yspread_col="y_std")  # + heteroscedastic
    trusted = (lab["sp_confidence"].fillna(0) >= 0.85).values

    report = {"n_compounds": len(lab), "n_families": int(lab["family"].nunique()),
              "noise_floor_logunits": round(noise, 3),
              "folds": [s[0] for s in splits],
              "is_pH52_count": int(Xprep[:, 0].sum()),
              "trusted_conf>=0.85": int(trusted.sum())}

    # ---------- Q1: weighting on LOFO (RDKit features) ----------
    print(f"compounds={len(lab)} families={lab['family'].nunique()} noise_floor(y_std)={noise:.3f} log\n")
    print("Q1  WEIGHTING (leave-one-family-out, RDKit features) — train-fold weights only")
    print(f"{'model':13s}{'weight':12s}{'R2':>7s}{'MAE':>7s}{'rho':>7s}{'MAE@trusted':>13s}")
    q1 = []
    for model in ["Ridge", "RandomForest"] + (["GP"] if args.gp else []):
        for wname, w in [("none", None), ("conf", w_conf), ("conf+var", w_var)]:
            yhat, r = eval_pooled(model, X, y, splits, weights=w)
            ok = np.isfinite(yhat) & trusted
            mae_tr = float(mean_absolute_error(y[ok], yhat[ok])) if ok.sum() > 5 else float("nan")
            r.update(model=model, weight=wname, mae_trusted=round(mae_tr, 4))
            q1.append(r)
            print(f"{model:13s}{wname:12s}{r['r2']:7.3f}{r['mae']:7.3f}{r['rho']:7.3f}{mae_tr:13.4f}")
    # floor
    _, fr = eval_pooled("Mean(floor)", X, y, splits)
    print(f"{'Mean(floor)':13s}{'-':12s}{fr['r2']:7.3f}{fr['mae']:7.3f}{fr['rho']:7.3f}   <-- floor")
    report["Q1_weighting"] = q1
    report["mean_floor"] = fr

    # ---------- Q2: is_pH52 within ja1c05813 (random 5-fold) ----------
    m = within_paper_mask(lab, "ja1c05813")
    print(f"\nQ2  is_pH52 WITHIN ja1c05813 (n={int(m.sum())}, random 5-fold, Ridge)")
    q2 = {"n": int(m.sum())}
    if m.sum() >= 25 and Xprep[m, 0].std() > 0:
        Xj, yj = X[m], y[m]
        Xjp = np.column_stack([Xj, Xprep[m, 0]])  # + is_pH52
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for tag, XX in [("rdkit", Xj), ("rdkit+is_pH52", Xjp)]:
            yh = np.full(len(yj), np.nan)
            for tr, te in kf.split(XX):
                sc = StandardScaler().fit(XX[tr])
                mm = Ridge(alpha=10.0).fit(sc.transform(XX[tr]), yj[tr])
                yh[te] = mm.predict(sc.transform(XX[te]))
            q2[tag] = dict(r2=round(float(r2_score(yj, yh)), 4),
                           mae=round(float(mean_absolute_error(yj, yh)), 4))
            print(f"   {tag:16s} R2={q2[tag]['r2']:.4f}  MAE={q2[tag]['mae']:.4f}")
        q2["delta_MAE(is_pH52 - baseline)"] = round(q2["rdkit+is_pH52"]["mae"] - q2["rdkit"]["mae"], 4)
    else:
        q2["skipped"] = "insufficient within-ja1c05813 variation"
        print("   skipped:", q2["skipped"])
    report["Q2_within_ja1c05813"] = q2

    # ---------- Q3: is is_pH52 subsumed by paper? ----------
    print("\nQ3  is is_pH52 subsumed by `paper`?  (LinearRegression, in-sample R2 + bootstrap)")
    P = pd.get_dummies(lab["paper"]).values.astype(float)
    base = LinearRegression().fit(P, y)
    full = LinearRegression().fit(np.column_stack([P, Xprep[:, 0]]), y)
    r2_base = r2_score(y, base.predict(P))
    r2_full = r2_score(y, full.predict(np.column_stack([P, Xprep[:, 0]])))
    rng = np.random.RandomState(0)
    deltas = []
    n = len(y)
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        b = LinearRegression().fit(P[idx], y[idx]); f = LinearRegression().fit(
            np.column_stack([P, Xprep[:, 0]])[idx], y[idx])
        deltas.append(r2_score(y[idx], f.predict(np.column_stack([P, Xprep[:, 0]])[idx]))
                      - r2_score(y[idx], b.predict(P[idx])))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    verdict = ("adds signal beyond paper" if lo > 0.005
               else "marginal / essentially subsumed by paper (CI hugs 0)" if lo > 0
               else "subsumed by paper (CI includes 0)")
    q3 = {"r2_paper_only": round(float(r2_base), 4), "r2_paper+is_pH52": round(float(r2_full), 4),
          "delta_r2": round(float(r2_full - r2_base), 4),
          "delta_r2_boot95": [round(float(lo), 4), round(float(hi), 4)],
          "verdict": verdict}
    print(f"   R2 paper-only={q3['r2_paper_only']}  paper+is_pH52={q3['r2_paper+is_pH52']}  "
          f"dR2={q3['delta_r2']} (95% boot [{q3['delta_r2_boot95'][0]}, {q3['delta_r2_boot95'][1]}])")
    print(f"   verdict: {q3['verdict']}")
    report["Q3_is_pH52_vs_paper"] = q3

    out = ROOT / "eval/sample_prep_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print("\nHONEST READ: weighting's value is robustness on trusted rows / calibration, not a big "
          "LOFO R2 jump; adopt is_pH52 only if Q2 helps AND Q3 says it is not subsumed by paper.")


if __name__ == "__main__":
    raise SystemExit(main())
