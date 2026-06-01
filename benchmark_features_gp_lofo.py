"""
benchmark_features_gp_lofo.py — honest small-data benchmark for IAJD flux prediction.

Implements the validated recipe in docs/SMALL_NOISY_DATA_METHODS.md:
  * replicate-averaged y  (+ per-compound spread as a real noise estimate)
  * FEATURE SETS compared head-to-head: Morgan FP, RDKit descriptors, physics (QM/MD),
    and combinations  ->  "physics-as-features", the highest-confidence lever
  * core model = homoscedastic Gaussian Process; baselines = RandomForest, Ridge,
    and a MEAN-PREDICTOR floor (the honest null: does anything beat predicting the
    training mean for a held-out family?)
  * LEAVE-ONE-FAMILY-OUT cross-validation (default) or KMeans cluster split
  * reports pooled held-out R2 / MAE / Spearman per (feature_set x model)

The goal is NOT to maximize a number: label noise caps the *measured* score, not the
model's true accuracy (Kolmar & Grulke, J Cheminform 2021). The goal is a fair,
family-held-out comparison against the mean-predictor floor — what actually generalizes.

Hook: once Module A/B physics (apparent pKa, spontaneous curvature, H_II) are computed
for the IAJDs, add them as another feature set here — that's the real test of the design
levers. An AGILE frozen-encoder embedding can be dropped in the same way (load a .npy and
register it as a feature set), to benchmark transfer vs physics-features (expect possible
OOD negative transfer for dendrimers).

  ./.venv/bin/python benchmark_features_gp_lofo.py
  ./.venv/bin/python benchmark_features_gp_lofo.py --split cluster --n-clusters 6
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.logger().setLevel(RDLogger.ERROR)
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent
BIOACT = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
QM = ROOT / "IAJD_master/bundles_caches/physics/qm_cache.csv"
MD = ROOT / "IAJD_master/bundles_caches/physics/md_cache.csv"
TARGET = "log10_flux_total"
RDKIT_DESCS = [n for n, _ in Descriptors._descList]


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def load_labels() -> pd.DataFrame:
    """Replicate-averaged y, per-compound spread (noise estimate), family — one row/compound."""
    df = pd.read_excel(BIOACT)
    df["c"] = df["SMILES_canonical"].map(canon)
    df = df.dropna(subset=["c", TARGET])
    g = df.groupby("c")
    out = pd.DataFrame({
        "y": g[TARGET].mean(),
        "y_std": g[TARGET].std().fillna(0.0),     # replicate spread = per-point noise
        "n_rep": g[TARGET].size(),
        "family": g["family"].first().fillna("Unknown"),
    }).reset_index().rename(columns={"c": "smiles"})
    return out


def f_morgan(smis, radius=3, nbits=1024):
    # RDKit 2022.09 API (GetMorganGenerator was added later)
    X = np.zeros((len(smis), nbits))
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
            X[i] = np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")
    return X


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
    return X


def f_physics(smis, cache: Path, prefix: str):
    if not cache.exists():
        return np.zeros((len(smis), 0))
    c = pd.read_csv(cache)
    c["c"] = c["smiles_canonical"].map(canon)
    cols = [x for x in c.columns if x.startswith(prefix) and x != f"{prefix}error"]
    cmap = c.dropna(subset=["c"]).drop_duplicates("c").set_index("c")[cols]
    X = np.full((len(smis), len(cols)), np.nan)
    for i, s in enumerate(smis):
        if s in cmap.index:
            X[i] = pd.to_numeric(cmap.loc[s], errors="coerce").values
    return X


def models():
    def gp():
        k = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                        n_restarts_optimizer=6, random_state=0)
    return {
        "GP": gp,
        "RandomForest": lambda: RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
        "Ridge": lambda: Ridge(alpha=10.0),
        "Mean(floor)": lambda: DummyRegressor(strategy="mean"),
    }


def lofo_splits(fam: np.ndarray):
    for f in sorted(pd.unique(fam)):
        te = fam == f
        if te.sum() >= 3 and (~te).sum() >= 20:
            yield f, ~te, te


def cluster_splits(X: np.ndarray, k: int):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    lab = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
    for c in range(k):
        te = lab == c
        if te.sum() >= 3 and (~te).sum() >= 20:
            yield f"clust{c}", ~te, te


def evaluate(make_model, X, y, splits):
    """Pooled held-out predictions across folds, scored once (honest aggregate)."""
    yhat = np.full(len(y), np.nan)
    for _, tr, te in splits:
        sc = StandardScaler().fit(np.nan_to_num(X[tr]))
        Xtr, Xte = sc.transform(np.nan_to_num(X[tr])), sc.transform(np.nan_to_num(X[te]))
        m = make_model()
        m.fit(Xtr, y[tr])
        yhat[te] = m.predict(Xte)
    ok = np.isfinite(yhat)
    if ok.sum() < 10:
        return None
    return dict(n=int(ok.sum()), r2=r2_score(y[ok], yhat[ok]),
                mae=mean_absolute_error(y[ok], yhat[ok]),
                rho=spearmanr(y[ok], yhat[ok])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["lofo", "cluster"], default="lofo")
    ap.add_argument("--n-clusters", type=int, default=6)
    args = ap.parse_args()

    lab = load_labels()
    smis = lab["smiles"].tolist()
    fam = lab["family"].values
    y = lab["y"].values
    print(f"compounds: {len(lab)} | families: {lab['family'].nunique()} "
          f"| target std={y.std():.2f}")
    print(f"NOISE FLOOR — mean per-compound replicate spread (y_std) = {lab['y_std'].mean():.3f} "
          f"log units (compounds with >1 rep: {(lab['n_rep']>1).sum()}); "
          f"this is the irreducible measurement noise — don't expect to predict below it.\n")

    print("Building feature sets (Morgan / RDKit / physics-QM)…")
    feats = {
        "morgan": f_morgan(smis),
        "rdkit": f_rdkit(smis),
        "qm": f_physics(smis, QM, "qm_"),
    }
    feats["qm+morgan"] = np.hstack([feats["qm"], feats["morgan"]])
    feats["qm+rdkit"] = np.hstack([feats["qm"], feats["rdkit"]])
    feats["all"] = np.hstack([feats["qm"], feats["rdkit"], feats["morgan"]])
    # (When Module A/B physics for the IAJDs exist, add e.g.
    #  feats["design"] = f_physics(smis, DESIGN_CACHE, "phys_") and combos.)

    splits = list(lofo_splits(fam)) if args.split == "lofo" \
        else list(cluster_splits(feats["all"], args.n_clusters))
    print(f"\nVALIDATION = {args.split} | {len(splits)} held-out folds: "
          f"{[s[0] for s in splits]}\n")
    print(f"{'feature_set':12s} {'model':14s} {'n':>4s} {'R2':>7s} {'MAE':>6s} {'Spearman':>9s}")
    print("-" * 56)
    for fs, X in feats.items():
        for mname, mk in models().items():
            if mname == "Mean(floor)" and fs != "qm":
                continue   # floor is feature-independent; show once
            r = evaluate(mk, X, y, splits)
            if r:
                tag = "  <-- floor" if mname == "Mean(floor)" else ""
                print(f"{fs:12s} {mname:14s} {r['n']:4d} {r['r2']:7.3f} "
                      f"{r['mae']:6.3f} {r['rho']:9.3f}{tag}")
    print("\nRead it honestly: a feature_set/model only 'works' if it beats the Mean(floor)"
          "\nrow under family-held-out validation. Modest R2 is expected (noise ceiling).")


if __name__ == "__main__":
    raise SystemExit(main())
