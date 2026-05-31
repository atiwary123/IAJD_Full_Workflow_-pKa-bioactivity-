"""
physics_calibrate.py — Steps 0-3 of the physics-design protocol
(docs/PHYSICS_DESIGN_PROTOCOL.md): truth-weight the flux data, assemble the mechanistic
coordinate axes, and adjudicate — per family, per organ — which axes are *corroborated*
(mechanism prior ∩ reliability-weighted data), leniently and with honest CIs.

This is the instrument that decides what the design loop is allowed to push on. It is
NOT a flux predictor and NOT an axis-picker by p-value; it reports, for each axis, whether
the data SUPPORTS / is too NOISY to confirm / CONTRADICTS its mechanistic prior, weighted
by how trustworthy each compound's flux is.

Runnable now on the existing dataset + physics features (pKa, c0/CPP, qm descriptors); it
consumes the new Module A/B/C outputs as they land (axis registry). No proxies: a compound
with no real value for an axis is excluded from that axis, never imputed.

Steps 4-7 (composite direction, coupled-space design, robustness ranking, falsification)
live in the design loop and are scaffolded at the bottom (pending the full physics panel).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
BIOACT = ROOT / "IAJD_master" / "datasets" / "IAJD_Bioact_v13_clean.xlsx"
PHYS = ROOT / "IAJD_master" / "bundles_caches" / "physics"
OUT = PHYS / "design" / "calibration_report.json"

# ── Step 2: mechanistic priors (from biophysics, NOT the dataset) ──────────────
# direction: +1 = higher axis -> higher escape/flux (monotonic); 'window' = interior
# optimum (too-high AND too-low are bad). prior_strength in [0,1] = how established.
@dataclass
class AxisPrior:
    direction: object        # +1, -1, or 'window'
    prior_strength: float    # 0..1
    note: str

MECHANISM_PRIORS: Dict[str, AxisPrior] = {
    # apparent pKa: the best-validated correlate BUT windowed and liver-specific; for
    # spleen the window is unknown -> test for ANY interior-peak structure, weak prior.
    "pKa":            AxisPrior("window", 0.6, "near-neutral-blood/charged-endosome; liver optimum 6.2-6.5 does NOT transfer to spleen"),
    "phys_apparent_pKa": AxisPrior("window", 0.6, "Module A membrane apparent pKa (preferred over the dataset pKa once available)"),
    # spontaneous curvature: more-negative c0 -> more H_II drive -> more escape.
    "phys_c0":        AxisPrior(-1, 0.5, "Module B: more-negative c0 favors Lα->H_II"),
    "md_c0_spontaneous": AxisPrior(+1, 0.3, "legacy Helfrich-proxy delta-a_head; sign per its own convention, weak"),
    # packing parameter: wedge CPP>1 -> negative curvature -> escape.
    "phys_CPP":       AxisPrior(+1, 0.5, "Module B: CPP>1 cone"),
    "md_cpp_prot":    AxisPrior(+1, 0.4, "MD packing parameter (protonated)"),
    # H_II propensity: more non-bilayer -> more escape (leading hypothesis, not law).
    "phys_HII_score": AxisPrior(+1, 0.4, "Module C: inverted-phase propensity (hypothesis)"),
    # qm head charge: more protonatable N -> (weakly) toward the pKa story.
    "qm_q_ionizableN": AxisPrior(+1, 0.2, "xTB intrinsic head charge; weak intrinsic anchor only"),
}

ORGAN_FLUX = {"spleen": "log10_flux_spleen", "liver": "log10_flux_liver",
              "lung": "log10_flux_lung", "total": "log10_flux_total"}


# ── Step 0: truth-weight the flux data ─────────────────────────────────────────
def reliability_weights(df: pd.DataFrame, organ: str) -> np.ndarray:
    """Per-compound flux reliability in [0,1]: count-based (replicates, mice) × inverse
    measurement noise (SEM) × structural-twin self-consistency. Every term is explicit
    and documented — no hidden trust."""
    n = len(df)
    # count term: more replicates/mice -> more reliable (sqrt, saturating)
    nrep = pd.to_numeric(df.get("n_replicates"), errors="coerce").fillna(1).clip(lower=1)
    nmice = pd.to_numeric(df.get("n_mice"), errors="coerce").fillna(1).clip(lower=1)
    count = np.sqrt(nrep.values * nmice.values)
    count = count / np.nanmax(count) if np.nanmax(count) > 0 else np.ones(n)
    # noise term: inverse-variance from SEM relative to the flux spread (where present)
    sem = pd.to_numeric(df.get("flux_total_SEM"), errors="coerce").values
    flux = pd.to_numeric(df.get(ORGAN_FLUX[organ]), errors="coerce").values
    spread = np.nanstd(flux) if np.isfinite(np.nanstd(flux)) else 1.0
    rel_sem = np.where(np.isfinite(sem) & (sem > 0), sem / (spread + 1e-9), np.nan)
    noise = 1.0 / (1.0 + np.nan_to_num(rel_sem, nan=np.nanmedian(rel_sem[np.isfinite(rel_sem)]) if np.isfinite(rel_sem).any() else 0.5))
    # structural-twin consistency: a compound whose nearest structural twin disagrees on
    # flux (beyond the family spread) is sitting on the noise floor -> downweight.
    twin = _twin_consistency(df, organ)
    w = count * noise * twin
    return np.clip(w / (np.nanmax(w) + 1e-9), 0.0, 1.0)


def _twin_consistency(df: pd.DataFrame, organ: str) -> np.ndarray:
    """1 - normalized |flux - nearest-Tanimoto-twin flux| / family_spread, in [0,1]."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        return np.ones(len(df))
    smis = df.get("SMILES_canonical", df.get("SMILES")).fillna("").tolist()
    flux = pd.to_numeric(df.get(ORGAN_FLUX[organ]), errors="coerce").values
    fps = []
    for s in smis:
        m = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)
    spread = np.nanstd(flux) + 1e-9
    out = np.ones(len(df))
    for i in range(len(df)):
        if fps[i] is None or not np.isfinite(flux[i]):
            continue
        best, bj = -1.0, -1
        for j in range(len(df)):
            if j == i or fps[j] is None or not np.isfinite(flux[j]):
                continue
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if t > best:
                best, bj = t, j
        if bj >= 0 and best > 0.85:   # only judge by genuinely near twins
            disagree = abs(flux[i] - flux[bj]) / spread
            out[i] = float(np.clip(1.0 - 0.5 * disagree, 0.2, 1.0))
    return out


# ── Step 1: assemble the coordinate axes from real sources (no imputation) ──────
def assemble_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Merge available physics axes onto the bioactivity rows by canonical SMILES.
    Pulls dataset pKa + the physics caches + (future) Module A/B/C design caches."""
    out = df.copy()
    key = "SMILES_canonical" if "SMILES_canonical" in out else "SMILES"
    for cache, cols in [
        (PHYS / "qm_cache.csv", ["qm_q_ionizableN", "qm_dipole_D"]),
        (PHYS / "md_cache.csv", ["md_c0_spontaneous", "md_cpp_prot"]),
        (PHYS / "design" / "pka" / "_merged.csv", ["phys_apparent_pKa"]),
        (PHYS / "design" / "design_cache.csv", ["phys_c0", "phys_CPP", "phys_HII_score"]),
    ]:
        if cache.exists():
            try:
                c = pd.read_csv(cache)
                ck = "smiles_canonical" if "smiles_canonical" in c else key
                keep = [x for x in cols if x in c.columns]
                if keep and ck in c.columns:
                    out = out.merge(c[[ck] + keep].rename(columns={ck: key}), on=key, how="left")
            except Exception:
                pass
    return out


# ── Step 1: GAM response curves — locate each axis's optimum, CV-validate ───────
# Per the canonical optimization protocol (docs/DESIGN_OPTIMIZATION_PROTOCOL.md):
# mechanism -> GAM response curves (find the peak, not the edge; CV so a peak isn't
# noise; distrust a data-peak that contradicts mechanism) -> GP/BO -> confirm. We fit a
# reliability-WEIGHTED smooth GAM of flux vs each axis, in-family, and adjudicate the
# response against the mechanistic prior.
@dataclass
class Verdict:
    family: str
    organ: str
    axis: str
    n_eff: float
    gam_peak: Optional[float]   # axis value at the response maximum (the optimum)
    peak_interior: bool         # is the optimum interior (a true window) vs at an edge?
    response_range: float       # max-min of the smoothed flux response (effect size)
    cv_r2: float                # k-fold pseudo-R^2 of the GAM — is the signal real?
    edf: float                  # effective dof (>~1.5 => genuinely non-monotonic)
    mechanism_consistent: bool  # does the data response match the prior direction/shape?
    verdict: str                # SUPPORTED / NOISY / CONTRARY / INSUFFICIENT / NO_DATA
    weight: float               # design weight = prior_strength * cv_support * reliability


def _kfold_r2(X, Y, W, n_splines, lam, k=5):
    """Reliability-weighted k-fold pseudo-R^2 for a single-term GAM (signal vs noise)."""
    from pygam import LinearGAM, s
    n = len(Y)
    if n < 8:
        k = max(2, n // 3)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    sse, sst = 0.0, 0.0
    ybar = np.average(Y, weights=W)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        if len(tr) < n_splines + 1 or len(f) == 0:
            continue
        try:
            g = LinearGAM(s(0, n_splines=n_splines), lam=lam).fit(X[tr], Y[tr], weights=W[tr])
            pred = g.predict(X[f])
        except Exception:
            continue
        sse += np.sum(W[f] * (Y[f] - pred) ** 2)
        sst += np.sum(W[f] * (Y[f] - ybar) ** 2)
    return 1.0 - sse / sst if sst > 0 else np.nan


def adjudicate(df: pd.DataFrame, axis: str, organ: str, family: str) -> Verdict:
    from pygam import LinearGAM, s
    prior = MECHANISM_PRIORS.get(axis)
    fcol = ORGAN_FLUX[organ]
    sub = df[df["family"] == family] if family != "ALL" else df
    if axis not in sub.columns or fcol not in sub.columns:
        return Verdict(family, organ, axis, 0, None, False, np.nan, np.nan, np.nan, False, "NO_DATA", 0.0)
    x = pd.to_numeric(sub[axis], errors="coerce").values
    y = pd.to_numeric(sub[fcol], errors="coerce").values
    w = reliability_weights(sub, organ)
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    n = int(m.sum())
    if n < 6 or np.nanstd(x[m]) < 1e-9:
        return Verdict(family, organ, axis, n, None, False, np.nan, np.nan, np.nan, False, "INSUFFICIENT", 0.0)
    X = x[m].reshape(-1, 1); Y = y[m]; W = w[m]
    n_splines = int(np.clip(n // 3, 4, 10))
    try:
        gam = LinearGAM(s(0, n_splines=n_splines)).gridsearch(
            X, Y, weights=W, lam=np.logspace(-1, 3, 9), progress=False)
    except Exception:
        return Verdict(family, organ, axis, n, None, False, np.nan, np.nan, np.nan, False, "INSUFFICIENT", 0.0)
    grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    pred = gam.predict(grid)
    peak = float(grid[np.argmax(pred), 0])
    rng_x = X.max() - X.min()
    peak_interior = bool(X.min() + 0.1 * rng_x < peak < X.max() - 0.1 * rng_x)
    response_range = float(pred.max() - pred.min())
    edf = float(gam.statistics_.get("edof", np.nan))
    cv_r2 = _kfold_r2(X, Y, W, n_splines, gam.lam[0][0] if hasattr(gam, "lam") else 1.0)

    # mechanism consistency: window -> interior peak; +/-1 -> monotone sign of the curve
    if prior and prior.direction == "window":
        consistent = peak_interior
    elif prior and prior.direction in (+1, -1):
        slope = np.corrcoef(grid.ravel(), pred)[0, 1] * np.sign(prior.direction)
        consistent = bool(slope > 0)
    else:
        consistent = True
    reliability = float(np.average(W))
    real = (np.isfinite(cv_r2) and cv_r2 > 0.05 and response_range > 0.15 * (np.nanstd(Y) + 1e-9))
    if not real:
        verdict = "NOISY"
    elif consistent:
        verdict = "SUPPORTED"
    else:
        verdict = "CONTRARY"     # a real, CV-validated response that DISAGREES with mechanism
    cv_support = max(0.0, cv_r2) if np.isfinite(cv_r2) else 0.0
    weight = (prior.prior_strength if prior else 0.2) * cv_support * reliability * (1.0 if consistent else 0.0)
    return Verdict(family, organ, axis, n, round(peak, 2), peak_interior,
                   round(response_range, 3), round(float(cv_r2), 3) if np.isfinite(cv_r2) else np.nan,
                   round(edf, 2) if np.isfinite(edf) else np.nan, consistent, verdict, round(weight, 4))


def run_calibration(organs: Tuple[str, ...] = ("spleen", "total"),
                    min_family_n: int = 8) -> Dict:
    df = pd.read_excel(BIOACT)
    df = assemble_axes(df)
    fams = [f for f, n in df["family"].value_counts().items() if n >= min_family_n]
    report = {"families_scored": {f: int((df["family"] == f).sum()) for f in fams},
              "organs": list(organs), "axes": list(MECHANISM_PRIORS), "verdicts": []}
    for organ in organs:
        for fam in fams:
            for axis in MECHANISM_PRIORS:
                v = adjudicate(df, axis, organ, fam)
                if v.verdict != "NO_DATA":
                    report["verdicts"].append(v.__dict__)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    return report


# ── Steps 4-7 (design loop) — scaffold; needs the full physics panel ───────────
def composite_direction(report: Dict, family: str, organ: str) -> Dict:
    """Step 4: confidence-weighted composite design direction from the SUPPORTED/NOISY
    axes (weight already = prior × data_support × reliability). Returns axis->weight."""
    w = {}
    for v in report["verdicts"]:
        if v["family"] == family and v["organ"] == organ and v["verdict"] in ("SUPPORTED", "NOISY"):
            w[v["axis"]] = v["weight"]
    s = sum(w.values()) or 1.0
    return {k: round(val / s, 3) for k, val in sorted(w.items(), key=lambda kv: -kv[1])}
# ── Step 2: GP surrogate + Bayesian optimization (propose next candidate) ───────
def gp_surrogate(df: pd.DataFrame, axes: List[str], organ: str, family: str):
    """Fit a reliability-noised, ARD Matern-2.5 GP of flux vs the supported axes; return
    the fitted GP, the scaler, LOO-CV R^2, 1-sigma calibration coverage, and per-axis ARD
    length-scales (a free relevance read-out). Honest: a miscalibrated GP is flagged."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    sub = df[df["family"] == family] if family != "ALL" else df
    cols = [a for a in axes if a in sub.columns]
    X = sub[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(sub[ORGAN_FLUX[organ]], errors="coerce")
    w = reliability_weights(sub, organ)
    m = X.notna().all(axis=1).values & np.isfinite(y.values) & (w > 0)
    if m.sum() < max(6, len(cols) + 3):
        return {"status": "insufficient", "n": int(m.sum()), "axes": cols}
    Xs = StandardScaler().fit(X.values[m])
    Xn = Xs.transform(X.values[m]); Y = y.values[m]; W = w[m]
    # heteroscedastic noise floor from reliability: low-reliability points get more noise
    alpha = (np.nanvar(Y) * (1.0 - 0.8 * W) + 1e-3)
    kern = C(1.0) * Matern(length_scale=np.ones(len(cols)), nu=2.5) + WhiteKernel(1e-1)
    gp = GaussianProcessRegressor(kernel=kern, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=4, random_state=0).fit(Xn, Y)
    # LOO-CV R^2 + calibration (does the 1-sigma band cover truth at ~68%?)
    preds, sds = [], []
    for i in range(len(Y)):
        tr = np.setdiff1d(np.arange(len(Y)), [i])
        g = GaussianProcessRegressor(kernel=kern, alpha=alpha[tr], normalize_y=True,
                                     random_state=0).fit(Xn[tr], Y[tr])
        mu, sd = g.predict(Xn[i:i+1], return_std=True)
        preds.append(mu[0]); sds.append(sd[0])
    preds = np.array(preds); sds = np.array(sds)
    r2 = 1 - np.sum((Y - preds) ** 2) / (np.sum((Y - Y.mean()) ** 2) + 1e-9)
    cover = float(np.mean(np.abs(Y - preds) <= sds))
    ls = dict(zip(cols, np.round(gp.kernel_.k1.k2.length_scale, 2))) if hasattr(gp.kernel_.k1.k2, "length_scale") else {}
    return {"status": "ok", "n": int(m.sum()), "axes": cols, "loo_r2": round(float(r2), 3),
            "calibration_1sigma": round(cover, 2), "ard_length_scales": ls,
            "gp": gp, "scaler": Xs, "y_best": float(np.max(Y))}


def expected_improvement(gp, scaler, X_cand: np.ndarray, y_best: float, xi: float = 0.01):
    """EI acquisition over candidate descriptor rows (the next-to-evaluate ranking)."""
    from scipy.stats import norm
    mu, sd = gp.predict(scaler.transform(X_cand), return_std=True)
    imp = mu - y_best - xi
    z = np.where(sd > 0, imp / sd, 0.0)
    ei = np.where(sd > 0, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
    return mu, sd, ei
# Steps 5-7 (coupled-space analog design over the GP via constrained-EI, robustness
# ranking, retrodiction/escalation) attach once Modules A/B/C populate design_cache.csv.


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--organs", nargs="*", default=["spleen", "total"])
    args = p.parse_args()
    rep = run_calibration(organs=tuple(args.organs))
    print(f"families scored: {rep['families_scored']}")
    print("\nGAM ADJUDICATION (Step 1) — reliability-weighted flux response vs each axis:")
    print(f"{'family':16s} {'organ':6s} {'axis':18s} {'n':>3s} {'peak':>6s} {'int':>3s} "
          f"{'range':>6s} {'cvR2':>6s} {'mech':>4s} {'verdict':10s} {'w':>6s}")
    for v in sorted(rep["verdicts"], key=lambda d: (-d["weight"])):
        if v["verdict"] in ("SUPPORTED", "CONTRARY") or v["weight"] > 0.02:
            pk = f"{v['gam_peak']:+.2f}" if v['gam_peak'] is not None else "  -  "
            print(f"{v['family']:16s} {v['organ']:6s} {v['axis']:18s} {v['n_eff']:3.0f} "
                  f"{pk:>6s} {'Y' if v['peak_interior'] else 'n':>3s} "
                  f"{(v['response_range'] or 0):6.2f} {(v['cv_r2'] if v['cv_r2']==v['cv_r2'] else 0):6.2f} "
                  f"{'Y' if v['mechanism_consistent'] else 'N':>4s} {v['verdict']:10s} {v['weight']:6.3f}")
    for fam in ("GA-Tris", "sSS-Nonsym", "PE-Tris", "G1-Janus-Dendrimer"):
        if fam in rep["families_scored"]:
            print(f"composite design direction [{fam}/spleen] (Step 4):",
                  composite_direction(rep, fam, "spleen"))
