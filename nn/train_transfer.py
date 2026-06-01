"""
nn/train_transfer.py — transfer-learning pipeline for IAJD organ flux, with the HONEST gate
built in (docs/IAJD_NN_TRANSFER_LEARNING_PLAN.md).

Two transfer mechanisms, in increasing fidelity:
  (RUNNABLE NOW) "delivery-prior" transfer: train a delivery model on LNPDB in-vivo lipids
      (RDKit features), then feed its prediction for each IAJD as an extra feature to the
      IAJD GP. Simple, robust, no GNN-repo fragility — runs on the pod immediately.
  (UPGRADE) GNN-embedding transfer: swap the RDKit featurizer for AGILE's frozen 60k-MolCLR
      encoder embedding (--encoder agile). Same GP head downstream.

THE GATE (non-negotiable, honesty): transfer is reported as useful ONLY if it beats the
no-transfer baseline (same GP, IAJD features only) on **leave-one-FAMILY-out** CV — the real
test of generalizing to a held-out architecture. We print BOTH and let the numbers decide;
a pretrained net is not assumed better.

Usage (on the pod, after nn/fetch_data.sh):
  python nn/train_transfer.py --lnpdb lnpdb_invivo.csv --target log10_flux_spleen
  python nn/train_transfer.py --no-lnpdb --target log10_flux_spleen   # baseline only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IAJD_CSV = HERE / "iajd_transfer_input.csv"


def morgan_features(smiles_list, n_bits=2048, radius=2):
    """RDKit Morgan fingerprint + a few physchem descriptors. Returns (X, ok_mask)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    X, ok = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(str(smi)) if smi is not None else None
        if m is None:
            X.append(np.zeros(n_bits + 6)); ok.append(False); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float64)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                rdMolDescriptors.CalcNumRotatableBonds(m),
                rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m)]
        X.append(np.concatenate([arr, desc])); ok.append(True)
    return np.array(X), np.array(ok)


def _featurize(smiles, encoder):
    """Molecular features: 'morgan' (fingerprint+descriptors, CPU) or 'agile' (frozen 60k
    MolCLR embedding, the GNN transfer encoder; GPU-accelerated if available)."""
    if encoder == "agile":
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent))
        from agile_embed import agile_embeddings
        return agile_embeddings(list(smiles))
    return morgan_features(smiles)[0]


def gp_loo_family(X, y, families, label):
    """Leave-one-FAMILY-out GP CV. Returns (spearman, r2, n) over held-out predictions."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr
    preds = np.full(len(y), np.nan)
    fams = pd.Series(families).fillna("?").values
    for fam in pd.unique(fams):
        tr = fams != fam; te = fams == fam
        if tr.sum() < 5 or te.sum() < 1:
            continue
        # standardize, then PCA-reduce (Morgan is high-dim+sparse; ARD over 2k bits would
        # overfit at n~tens and is slow) -> isotropic Matern GP on the components.
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        npc = int(min(30, Xtr.shape[1], max(2, tr.sum() - 1)))
        pca = PCA(n_components=npc).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        k = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-2)
        # random_state pins the optimizer restart for reproducibility. NOTE: the pooled
        # leave-one-family-out Spearman is still noisy at n~tens / 6 families — for a
        # trustworthy verdict, repeat over several random_states/CV folds and average
        # (a single run's baseline has swung +-0.2 between runs). Do that before believing
        # any "transfer helps" claim for design use.
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-6,
                                      n_restarts_optimizer=2, random_state=0)
        gp.fit(Ztr, y[tr])
        preds[te] = gp.predict(Zte)
    m = ~np.isnan(preds)
    if m.sum() < 5:
        return float("nan"), float("nan"), int(m.sum())
    rho = float(spearmanr(y[m], preds[m]).correlation)
    ss = 1 - np.sum((y[m] - preds[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
    print(f"  [{label}] leave-one-family-out: Spearman={rho:+.3f}  R2={ss:+.3f}  (n={m.sum()})")
    return rho, ss, int(m.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="log10_flux_spleen")
    p.add_argument("--lnpdb", type=Path, help="LNPDB in-vivo CSV (cols: smiles, value[, organ])")
    p.add_argument("--no-lnpdb", action="store_true", help="baseline only (no transfer)")
    p.add_argument("--encoder", default="morgan", choices=["morgan", "agile"],
                   help="featurizer: morgan (runs now) or agile (frozen 60k MolCLR; upgrade)")
    args = p.parse_args()

    from rdkit import Chem
    iajd = pd.read_csv(IAJD_CSV)
    iajd = iajd.dropna(subset=[args.target, "smiles"]).reset_index(drop=True)
    valid = iajd["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    if (~valid).any():
        print(f"  dropping {int((~valid).sum())} unparseable SMILES")
    iajd = iajd[valid].reset_index(drop=True)
    print(f"IAJDs with {args.target}: {len(iajd)}  ({iajd['family'].nunique()} families)")
    Xi = _featurize(iajd["smiles"].tolist(), args.encoder)
    y = iajd[args.target].to_numpy(float)
    fams = iajd["family"] if "family" in iajd else pd.Series(["?"] * len(iajd))

    # physics features (pKa, c0) appended where present
    phys = []
    for c in ("pKa", "c0_nm_inv"):
        if c in iajd:
            v = pd.to_numeric(iajd[c], errors="coerce").to_numpy(float)
            v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
            phys.append(v.reshape(-1, 1))
    Xi_phys = np.hstack([Xi] + phys) if phys else Xi

    results = {}
    print("\n== BASELINE (IAJD features only, no transfer) ==")
    results["baseline"] = gp_loo_family(Xi_phys, y, fams, "baseline")

    if not args.no_lnpdb and args.lnpdb and Path(args.lnpdb).exists():
        print("\n== TRANSFER (LNPDB in-vivo delivery prior as an extra feature) ==")
        ln = pd.read_csv(args.lnpdb).dropna(subset=["smiles", "value"])
        # per-study z-normalization: LNPDB values span scales across ~18 papers, so a raw
        # prior mixes incomparable units. Normalize within each study before training.
        if "study" in ln.columns:
            ln["value"] = ln.groupby("study")["value"].transform(
                lambda v: (v - v.mean()) / (v.std() + 1e-9))
            ln = ln.dropna(subset=["value"])
        # organ-match: if the target names an organ, train the prior on THAT organ's rows
        organ = next((o for o in ("spleen", "liver", "lung", "muscle", "heart", "kidney")
                      if o in args.target), None)
        if organ and "organ" in ln.columns and int((ln["organ"] == organ).sum()) >= 30:
            ln = ln[ln["organ"] == organ]
            print(f"  organ-matched prior: {organ} ({len(ln)} LNPDB rows)")
        else:
            print(f"  generic in-vivo prior ({len(ln)} LNPDB rows, per-study normalized)")
        ln = ln[ln["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)]
        Xl = _featurize(ln["smiles"].tolist(), args.encoder)
        from sklearn.ensemble import HistGradientBoostingRegressor
        prior = HistGradientBoostingRegressor(max_iter=300).fit(Xl, ln["value"].to_numpy(float))
        iajd_prior = prior.predict(Xi).reshape(-1, 1)   # LNPDB-learned delivery prior per IAJD
        Xi_tr = np.hstack([Xi_phys, iajd_prior])
        results["transfer"] = gp_loo_family(Xi_tr, y, fams, "transfer")
        b, t = results["baseline"][0], results["transfer"][0]
        verdict = ("TRANSFER HELPS" if (np.isfinite(t) and np.isfinite(b) and t > b + 0.02)
                   else "NO LIFT — keep baseline/GP-GAM")
        print(f"\n  VERDICT: Spearman baseline={b:+.3f} -> transfer={t:+.3f}  => {verdict}")
    else:
        print("\n  (no LNPDB given -> baseline only; fetch LNPDB on the pod for the transfer arm)")

    (HERE / "transfer_results.json").write_text(json.dumps(
        {"target": args.target, "encoder": args.encoder, "results": results}, indent=2, default=str))
    print(f"\nwrote {HERE/'transfer_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
