"""
sample_prep_weights.py — single source of truth for USING the `sp_*` sample-prep
block (see docs/SAMPLE_PREP_PARAMETERS.md) inside model training.

All functions are PURE: they take a row-aligned DataFrame and return numpy arrays.
The caller is responsible for passing a df whose row order matches its X / y.

DEFAULT-OFF integration contract
--------------------------------
Trainers wire in `maybe_sample_weight(df)` at their `.fit(...)` site:

    from sample_prep_weights import maybe_sample_weight
    sw = maybe_sample_weight(df)        # None unless IAJD_USE_PREP_WEIGHTS is set
    model.fit(X, y, sample_weight=sw)   # sample_weight=None == unweighted (sklearn/xgb default)

So adding the call is a **no-op** for every existing / in-flight retrain until a user
explicitly runs with `IAJD_USE_PREP_WEIGHTS=1`. This is what lets the integration land
without disturbing the ongoing retrains (`retrain_flagfix.sh`, `finalize_after_qm.sh`).

Findings encoded (from docs/SAMPLE_PREP_PARAMETERS.md / SAMPLE_PREP_INTEGRATION_SPEC.md):
  * Weighting is the primary, robust win: down-weight low-provenance rows
    (sp_confidence: novel_2026=0.35, bm4c01599=0.65, unknown=0.25) and n_mice==1.
  * The only varying formulation covariate is sp_assembly_pH (is_pH52); it is
    CONFOUNDED with paper/family — test it within ja1c05813 with `paper` in the model.
  * Data-quality flags (n_mice<=1; ja1c09585 imaging_time>7 h SI-unsupported) feed
    weights; labels are NEVER edited here.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ENV_FLAG = "IAJD_USE_PREP_WEIGHTS"
_TRUE = {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUE


def _num(df: pd.DataFrame, col: str, default=np.nan) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def row_reliability_weight(
    df: pd.DataFrame,
    *,
    conf_col: str = "sp_confidence",
    nmice_col: str = "n_mice",
    yspread_col: str | None = None,
    conf_floor: float = 0.05,
    conf_missing: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Per-row training weight  w_i = sp_confidence_i * n_eff_i  [/ (floor + spread^2)].

    - sp_confidence: provenance reliability (1.0/0.85/0.65/0.35/0.25). Missing -> conf_missing.
    - n_eff: max(n_mice, 1). Missing n_mice -> 1.
    - if yspread_col given (e.g. replicate spread y_std), divide by (floor + spread^2)
      so noisy compounds are down-weighted (heteroscedastic). floor = median positive spread^2.
    Normalized to mean 1 so it does not rescale regularization vs the unweighted fit.
    """
    n = len(df)
    conf = _num(df, conf_col, conf_missing)
    conf = np.where(np.isfinite(conf), conf, conf_missing)
    conf = np.clip(conf, conf_floor, 1.0)

    neff = _num(df, nmice_col, 1.0)
    neff = np.where(np.isfinite(neff) & (neff >= 1.0), neff, 1.0)

    w = conf * neff

    if yspread_col and yspread_col in df.columns:
        s = _num(df, yspread_col, 0.0)
        s = np.where(np.isfinite(s), s, 0.0)
        pos = s[s > 0]
        floor = float(np.median(pos) ** 2) if pos.size else 1e-3
        w = w / (floor + s ** 2)

    if not np.all(np.isfinite(w)) or np.all(w <= 0):
        good = w[np.isfinite(w) & (w > 0)]
        fill = float(good.min()) if good.size else 1.0
        w = np.where(np.isfinite(w) & (w > 0), w, fill)

    if normalize and w.sum() > 0:
        w = w * (len(w) / w.sum())
    return w.astype(float)


def gp_alpha(
    df: pd.DataFrame,
    *,
    base: float = 1e-2,
    conf_col: str = "sp_confidence",
    yspread_col: str = "y_std",
    conf_missing: float = 0.5,
) -> np.ndarray:
    """Per-point GP noise variance for sklearn GaussianProcessRegressor(alpha=...):
        alpha_i = base * (eps + spread_i^2) / sp_confidence_i
    Low-confidence / high-spread points get more slack (not deletion)."""
    n = len(df)
    conf = _num(df, conf_col, conf_missing)
    conf = np.clip(np.where(np.isfinite(conf), conf, conf_missing), 0.05, 1.0)
    s = _num(df, yspread_col, 0.0)
    s = np.where(np.isfinite(s), s, 0.0)
    pos = s[s > 0]
    eps = float(np.median(pos) ** 2) if pos.size else 1e-3
    return (base * (eps + s ** 2) / conf).astype(float)


# Only the VARYING prep covariates (constant sp_* columns are deliberately excluded).
PREP_FEATURE_NAMES = ["is_pH52", "imaging_time_c"]


def prep_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(X, names) for the prep covariates that actually vary in this corpus:
        is_pH52        = 1 if assembly buffer pH == 5.2 (ja1c05813 high-pH DNPs), else 0
        imaging_time_c = (imaging_time_h - 4.0), 0-filled  (uses sp_imaging_time_h or T_hours)
    """
    ph = _num(df, "sp_assembly_pH", np.nan)
    is_ph52 = (ph == 5.2).astype(float)
    t = _num(df, "sp_imaging_time_h", np.nan)
    if not np.any(np.isfinite(t)):
        t = _num(df, "T_hours", np.nan)
    t_c = np.where(np.isfinite(t), t - 4.0, 0.0)
    X = np.column_stack([is_ph52, t_c]).astype(float)
    return X, list(PREP_FEATURE_NAMES)


def dq_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality booleans (feed weights / ablations; never used to edit labels):
        dq_low_n        : n_mice <= 1  (high per-row variance)
        dq_time_suspect : imaging_time_h > 7  (SI-unsupported for ja1c09585; max 6 h)
    """
    nmice = _num(df, "n_mice", np.nan)
    t = _num(df, "sp_imaging_time_h", np.nan)
    if not np.any(np.isfinite(t)):
        t = _num(df, "T_hours", np.nan)
    return pd.DataFrame(
        {
            "dq_low_n": np.where(np.isfinite(nmice), nmice <= 1, False),
            "dq_time_suspect": np.where(np.isfinite(t), t > 7.0, False),
        },
        index=df.index,
    )


def within_paper_mask(df, paper, paper_col="paper", source_col="source") -> np.ndarray:
    """Boolean mask of rows from a given paper (coalescing paper<-source)."""
    if paper_col in df.columns:
        p = df[paper_col]
        if source_col in df.columns:
            p = p.fillna(df[source_col])
    elif source_col in df.columns:
        p = df[source_col]
    else:
        return np.zeros(len(df), dtype=bool)
    return (p.astype(str) == str(paper)).to_numpy()


def maybe_sample_weight(df: pd.DataFrame, **kw):
    """Return row weights iff env IAJD_USE_PREP_WEIGHTS is set, else None (no-op).
    This is the default-off hook trainers call so the integration never changes an
    existing/ongoing retrain unless explicitly enabled."""
    if not _enabled():
        return None
    return row_reliability_weight(df, **kw)


def bundle_weight(b14, bioact_xlsx, smis_key="smis_train", **kw) -> np.ndarray:
    """Weights aligned to a training bundle whose rows are identified by canonical
    SMILES (b14[smis_key]). Pulls sp_confidence (per-compound, first) and n_mice
    (summed over replicates) from the bioact xlsx, joined on canonical SMILES.
    Unmatched rows (e.g. the known stale SMILES_canonical join-key) fall back to
    neutral conf=0.5 / n_mice=1 via row_reliability_weight's missing handling."""
    from rdkit import Chem, RDLogger  # lazy: keep the module import-light
    RDLogger.logger().setLevel(RDLogger.ERROR)

    def _canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None

    smis = [s for s in b14[smis_key]]
    df = pd.read_excel(bioact_xlsx)
    smcol = "SMILES_canonical" if "SMILES_canonical" in df.columns else "SMILES"
    df = df.copy()
    df["__c"] = df[smcol].map(_canon)
    g = df.dropna(subset=["__c"]).groupby("__c")
    conf = g["sp_confidence"].first() if "sp_confidence" in df.columns else pd.Series(dtype=float)
    nmice = g["n_mice"].sum(min_count=1) if "n_mice" in df.columns else pd.Series(dtype=float)
    rows = pd.DataFrame({
        "sp_confidence": [conf.get(s, np.nan) for s in smis],
        "n_mice": [nmice.get(s, np.nan) for s in smis],
    })
    return row_reliability_weight(rows, nmice_col="n_mice", **kw)


def maybe_bundle_weight(b14, bioact_xlsx, **kw):
    """Bundle-aligned weights iff env IAJD_USE_PREP_WEIGHTS is set, else None (no-op)."""
    if not _enabled():
        return None
    return bundle_weight(b14, bioact_xlsx, **kw)


def smiles_weight(smiles_list, xlsx, *, use_nmice=True, yspread_col=None, **kw) -> np.ndarray:
    """Weights aligned to an explicit canonical-SMILES list, joined to `xlsx`.
    Generalizes bundle_weight for any model whose rows are SMILES-identified.
    For pKa use use_nmice=False, yspread_col='pKa_sd' (n_mice is irrelevant to pKa;
    the relevant per-row noise is the reported pKa standard deviation)."""
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    def _canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None

    df = pd.read_excel(xlsx)
    smcol = "SMILES_canonical" if "SMILES_canonical" in df.columns else "SMILES"
    df = df.copy()
    df["__c"] = df[smcol].map(_canon)
    g = df.dropna(subset=["__c"]).groupby("__c")
    keys = [_canon(s) for s in smiles_list]
    conf = g["sp_confidence"].first() if "sp_confidence" in df.columns else pd.Series(dtype=float)
    rows = {"sp_confidence": [conf.get(k, np.nan) for k in keys]}
    nmcol = "__no_nmice__"
    if use_nmice and "n_mice" in df.columns:
        nm = g["n_mice"].sum(min_count=1)
        rows["n_mice"] = [nm.get(k, np.nan) for k in keys]
        nmcol = "n_mice"
    if yspread_col and yspread_col in df.columns:
        ys = g[yspread_col].mean()
        rows[yspread_col] = [ys.get(k, np.nan) for k in keys]
    return row_reliability_weight(pd.DataFrame(rows), nmice_col=nmcol, yspread_col=yspread_col, **kw)


def maybe_pka_weight(smiles_list, pka_xlsx, **kw):
    """pKa-appropriate weights (sp_confidence, /pKa_sd^2; NO n_mice) iff the env flag
    is set, else None. pKa is a molecular property — animal count is irrelevant; the
    per-row noise is the reported pKa_sd (present for ~16 rows)."""
    if not _enabled():
        return None
    return smiles_weight(smiles_list, pka_xlsx, use_nmice=False, yspread_col="pKa_sd", **kw)


if __name__ == "__main__":  # tiny self-check on the live datasets (no model fit)
    import sys

    root = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(root, "IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx")
    d = pd.read_excel(f)
    w = row_reliability_weight(d, nmice_col="n_mice")
    X, names = prep_feature_matrix(d)
    fl = dq_flags(d)
    print(f"rows={len(d)}  weight[min/mean/max]={w.min():.3f}/{w.mean():.3f}/{w.max():.3f}")
    print(f"prep features {names}: is_pH52 sum={int(X[:,0].sum())}, "
          f"imaging_time_c nonzero={int((X[:,1]!=0).sum())}")
    print(f"dq_low_n={int(fl['dq_low_n'].sum())}  dq_time_suspect={int(fl['dq_time_suspect'].sum())}")
    print(f"env {ENV_FLAG} enabled? {_enabled()} -> maybe_sample_weight returns "
          f"{'array' if maybe_sample_weight(d) is not None else 'None'}")
    sys.exit(0)
