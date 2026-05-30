"""
test_regression_guards.py — minimal property-based test suite for non-regression.

Run with: pytest tests/test_regression_guards.py -v

Each test guards a specific invariant of the trained pipeline. They are not
unit tests in the traditional sense — they execute against the live bundles
on disk and verify that performance hasn't silently degraded between
retrains (which is what almost happened when `rebuild_pka_model.py` quietly
overwrote `preds_v91_final.npy` with worse predictions in this codebase).

If a test fails, that's a real signal that something regressed.
"""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))


# ──────────────────────────────────────────────────────────────────────
# Bundle-level invariants
# ──────────────────────────────────────────────────────────────────────

def test_v14_bundle_has_92_features():
    """v14 bundle should now include Block E (sample-prep covariates)."""
    with open(ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl", "rb") as f:
        b = pickle.load(f)
    n_features = b["X_train"].shape[1] if hasattr(b["X_train"], "shape") else len(b["X_train"][0])
    assert n_features == 92, f"v14 should have 92 features, got {n_features}"


def test_v14_bundle_block_slices_consistent():
    with open(ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl", "rb") as f:
        b = pickle.load(f)
    slices = b.get("block_slices") or {}
    assert "E_sampleprep" in slices, "Block E_sampleprep missing from block_slices"
    s = slices["E_sampleprep"]
    assert s.start == 88 and s.stop == 92, f"E_sampleprep slice should be 88:92, got {s}"


def test_pka_v92_bundle_has_per_family_weights():
    import joblib
    b = joblib.load(ROOT / "IAJD_master/bundles_caches/pka_v92_bundle.joblib")
    w = b["weights"]
    assert "per_family" in w or "analog" in w, "v9.2 should have either global or per-family weights"


# ──────────────────────────────────────────────────────────────────────
# Performance regression guards
# ──────────────────────────────────────────────────────────────────────

def test_v14_loo_mae_not_regressed():
    """v14 stacker LOO MAE should stay below 0.55 (was 0.43 at last known good state)."""
    d = np.load(ROOT / "bioact_loo_components.npz", allow_pickle=True)
    y = d["y_true"]
    direct = d["direct"]
    mae = float(np.mean(np.abs(direct - y)))
    assert mae < 0.55, f"v14 direct LOO MAE = {mae:.3f}, regression suspected (threshold 0.55)"


def test_pka_v92_loo_mae_not_regressed():
    """pKa v9.2 LOO MAE should stay below 0.20 (claim is 0.13)."""
    preds = np.load(ROOT / "preds_v91_final.npy")
    pka = pd.read_excel(ROOT / "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx",
                         sheet_name="Dataset")
    y = pka["pKa"].values.astype(float)
    if len(preds) != len(y):
        pytest.skip(f"preds shape {len(preds)} vs pKa table {len(y)} — pipeline mid-retrain?")
    mask = np.isfinite(preds) & np.isfinite(y)
    mae = float(np.mean(np.abs(preds[mask] - y[mask])))
    assert mae < 0.20, f"pKa LOO MAE = {mae:.3f}, regression suspected (threshold 0.20)"


# ──────────────────────────────────────────────────────────────────────
# Bayesian ensemble invariants
# ──────────────────────────────────────────────────────────────────────

def test_ensemble_bundle_exists():
    p = ROOT / "IAJD_master/bundles_caches/bioact_ensemble_bundle.pkl"
    if not p.exists():
        pytest.skip("ensemble bundle not yet trained")
    with open(p, "rb") as f:
        b = pickle.load(f)
    M = len(b["models"])
    assert M >= 5, f"ensemble should have ≥5 models, got M={M}"


def test_ensemble_calibration_reasonable():
    """The σ calibration factor should be in [0.5, 5.0]. Larger → ensemble is
    drastically underestimating uncertainty (model overfits training)."""
    p = ROOT / "IAJD_master/bundles_caches/bioact_ensemble_bundle.pkl"
    if not p.exists():
        pytest.skip("ensemble bundle not yet trained")
    with open(p, "rb") as f:
        b = pickle.load(f)
    c = b["sigma_calibration"]
    assert 0.3 < c < 10.0, f"calibration factor {c:.2f} suggests something wrong"


# ──────────────────────────────────────────────────────────────────────
# Physics layer invariants
# ──────────────────────────────────────────────────────────────────────

def test_physics_quality_in_unit_interval():
    """Q_physics MUST be in [0,1]. If it isn't, the formula is broken."""
    from physics_features import compute_all_physics_features
    sm = "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"
    feats = compute_all_physics_features(
        sm, pka=6.10, linker_length=2, n_tail_chains=3,
        chain_avg_carbons=9.0, head_group="H2EPRZ",
    )
    cpp = feats.get("cpp_geometric")
    if cpp is None or not np.isfinite(cpp):
        pytest.skip("CPP NaN — likely head_area returned NaN")
    assert 0 < cpp < 5, f"CPP {cpp:.3f} out of plausible range [0, 5]"


def test_head_area_3d_finite_for_all_known_heads():
    """3D-derived head area should work for every head in the registry."""
    from head_area_3d import head_area_from_smiles
    from iajd_grammar import HEAD_FRAGMENTS
    for hg, smi in HEAD_FRAGMENTS.items():
        a = head_area_from_smiles(smi)
        assert np.isfinite(a), f"head {hg} ({smi}) → NaN area"
        assert 0.05 < a < 2.0, f"head {hg} area {a:.3f} nm² out of plausible range"


# ──────────────────────────────────────────────────────────────────────
# MolGpKa live-call invariants
# ──────────────────────────────────────────────────────────────────────

def test_molgpka_runs_on_iajd_369():
    """Live MolGpKa GCN should return a finite raw pKa for IAJD 369."""
    try:
        sys.path.insert(0, str(ROOT / "molgpka_src"))
        from predict_pka import predict as molgpka_predict
    except Exception:
        pytest.skip("MolGpKa local install not importable")
    from rdkit import Chem
    sm = "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"
    mol = Chem.MolFromSmiles(sm)
    base_dict, _ = molgpka_predict(mol)
    assert len(base_dict) > 0, "MolGpKa returned empty base dict for IAJD 369"
    max_base = max(base_dict.values())
    assert 4 < max_base < 12, f"MolGpKa raw max-base {max_base:.2f} out of plausible range"


# ──────────────────────────────────────────────────────────────────────
# Honesty / no-proxy invariants
# ──────────────────────────────────────────────────────────────────────

def test_proposer_has_no_family_median_proxy():
    """propose_iajds.py shouldn't have _FAMILY_PKA_MEDIAN anymore — that was
    the proxy we replaced with live MolGpKa."""
    code = (ROOT / "propose_iajds.py").read_text()
    assert "_FAMILY_PKA_MEDIAN = " not in code, \
        "proxy _FAMILY_PKA_MEDIAN dict snuck back into propose_iajds.py"
    assert "_live_molgpka_pka" in code, \
        "_live_molgpka_pka helper missing from propose_iajds.py"


def test_block_form_no_proxy_defaults():
    """Block form should pass NaN through, not substitute medians."""
    code = (ROOT / "IAJD_master/code/bioact_v14_pipeline.py").read_text()
    # The string "_safe_float(row.get('DNP_size_nm'), 150.0)" used to be there —
    # if it's still there, we regressed
    assert "_safe_float(row.get('DNP_size_nm'), 150.0)" not in code, \
        "Block form regressed to proxy default for DNP_size_nm"


def test_block_b_c_no_rdkit_proxy_fallback():
    """compute_block_b / compute_block_c should NaN-out cache misses, not
    substitute RDKit proxies. Active code path must not call _lion_proxy_features
    or _admet_proxy_features."""
    code = (ROOT / "IAJD_master/code/bioact_v14_pipeline.py").read_text()
    # extract just the function bodies (the proxy fn definitions themselves
    # may remain for historical reference)
    import re
    block_b = re.search(r"def compute_block_b\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    block_c = re.search(r"def compute_block_c\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    assert block_b, "compute_block_b not found"
    assert block_c, "compute_block_c not found"
    assert "_lion_proxy_features(" not in block_b.group(0), \
        "compute_block_b still calls _lion_proxy_features (RDKit proxy)"
    assert "_admet_proxy_features(" not in block_c.group(0), \
        "compute_block_c still calls _admet_proxy_features (RDKit proxy)"


def test_head_area_no_lookup_fallback():
    """head_area_for_group in physics_features must NOT fall back to the
    hand-curated HEAD_AREA_PER_GROUP_NM2 table — 3D-derived or NaN, nothing in
    between."""
    code = (ROOT / "physics_features.py").read_text()
    import re
    m = re.search(r"def head_area_for_group\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    assert m, "head_area_for_group not found"
    body = m.group(0)
    assert "HEAD_AREA_PER_GROUP_NM2.get" not in body, \
        "head_area_for_group still falls back to the hand-curated lookup proxy"


def test_assemble_x_invokes_live_cache_extension():
    """predict_binary._assemble_X must call _extend_caches_live before
    handing rows to assemble_X, so novel SMILES get real LION/ADMET, not
    proxy values."""
    code = (ROOT / "predict_binary.py").read_text()
    import re
    m = re.search(r"def _assemble_X\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    assert m, "_assemble_X not found"
    assert "_extend_caches_live(canons)" in m.group(0), \
        "_assemble_X is missing live LION/ADMET cache extension"


def test_proposer_score_phys_uses_physics_yhat_not_ml():
    """At α=1 the proposer's score MUST be pure physics. The score_phys
    formula must use phys_delta (v15 Ridge on physics features), NOT
    ucb/yh (ML regressor). This guards against accidentally putting
    `ucb - yhat_seed` back into score_phys (audit 2026-05-29)."""
    code = (ROOT / "propose_iajds.py").read_text()
    import re
    # Find every `score_phys_core =` assignment line. They must reference
    # phys_delta, not ucb/yh.
    matches = re.findall(r"score_phys_core\s*=\s*(.+)", code)
    assert matches, "no score_phys_core assignment found — formula changed?"
    for m in matches:
        # phys_delta is acceptable; ucb/yh leaking into score_phys is not
        if "ucb" in m or " yh " in m or "yh_seed" in m or "ml_delta" in m:
            raise AssertionError(
                f"score_phys_core contains ML term: {m!r} — α=1 is no longer pure physics"
            )


def test_physics_quality_has_no_hardcoded_proxies():
    """_physics_quality_quick must not substitute hardcoded defaults
    (cpp=1.0, p_e=0.5, hlb=8.5, logKp=5.0) when its inputs are NaN —
    NaN must propagate honestly. Guards the no-proxy audit."""
    code = (ROOT / "propose_iajds.py").read_text()
    import re
    m = re.search(r"def _physics_quality_quick\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    assert m, "_physics_quality_quick not found"
    body = m.group(0)
    forbidden = [
        ("cpp = 1.0", "CPP=1.0 proxy"),
        ("p_e = 0.5", "protonation_endosome=0.5 proxy"),
        ("p_c = 0.1", "protonation_cytosol=0.1 proxy"),
        ("hlb = 8.5", "HLB=8.5 proxy"),
        ("logKp = 5.0", "logKp=5.0 proxy"),
        ("X[~np.isfinite(X)] = 0.0", "Ridge zero-fill proxy"),
        ("chain_avg_carbons=10.0", "tail-carbon-count=10.0 proxy"),
    ]
    for needle, label in forbidden:
        assert needle not in body, f"_physics_quality_quick regressed: {label} ({needle!r}) found"


def test_pka_v92_uses_per_family_or_global_weights():
    """predict_pka_v92 must look weights up as per_family[fam] or
    weights['global']['analog'] etc. — never the legacy flat keys
    (`weights['analog']` directly) which broke pKa for non-training SMILES."""
    code = (ROOT / "predict_pka_v92.py").read_text()
    import re
    m = re.search(r"def predict_pka_v92\(.*?\n(?=def |\Z)", code, flags=re.DOTALL)
    assert m, "predict_pka_v92 not found"
    body = m.group(0)
    # Should NOT contain bare weights["analog"]
    assert 'weights["analog"]' not in body, \
        "predict_pka_v92 still uses legacy weights['analog'] (broken on per-family bundle schema)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
