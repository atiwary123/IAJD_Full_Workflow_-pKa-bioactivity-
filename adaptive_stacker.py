"""Adaptive stacker with OOD-aware dynamic weighting.

Architecture:
  6 heads: direct, analog, LION, ADMET, AGILE, CPP

  Each query gets a "novelty score" = 1 - max_tanimoto_to_training.
  As novelty increases:
    - Similarity-dependent heads (analog, LION, ADMET) get downweighted
    - Physics/pretrained heads (CPP, AGILE) get upweighted
    - Direct XGB stays roughly constant (trained on descriptors, moderate extrapolation)

  The weighting function:
    w_i(novelty) = base_w_i * decay_i(novelty)

  Where decay functions are:
    - analog:  exp(-k_sim * novelty)     (decays fastest — pure similarity)
    - LION:    exp(-k_gnn * novelty)     (decays moderate — pretrained but on different data)
    - ADMET:   exp(-k_gnn * novelty)     (same as LION)
    - direct:  1.0                        (constant — descriptor-based)
    - AGILE:   exp(+k_phys * novelty)    (increases — pretrained on lipids)
    - CPP:     exp(+k_phys * novelty)    (increases — physics-based, best extrapolator)

  The k parameters are fit on the training set by simulating OOD via LOO:
    for each LOO fold, the held-out compound's max Tanimoto to training
    gives a real novelty signal. We optimize k values to minimize LOO MAE.
"""
from __future__ import annotations
import pickle, warnings
from pathlib import Path
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

from compute_cpp import compute_cpp_features, CPP_FEATURE_NAMES


def compute_novelty_scores(smiles_list, train_fps):
    """Compute novelty = 1 - max_tanimoto for each query vs training set."""
    try:
        fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
        _fp = lambda m: fpgen.GetFingerprint(m)
    except AttributeError:
        _fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(1.0)
            continue
        qfp = _fp(mol)
        sims = BulkTanimotoSimilarity(qfp, train_fps)
        scores.append(1.0 - max(sims) if sims else 1.0)
    return np.array(scores)


def adaptive_blend(head_preds, novelty, weights_at_zero, k_decay, k_grow):
    """Compute adaptive weighted prediction.

    head_preds: dict of head_name → prediction value
    novelty: float 0-1 (0 = exact match, 1 = completely OOD)
    weights_at_zero: base weights when novelty=0
    k_decay: decay rate for similarity heads
    k_grow: growth rate for physics heads

    Returns: weighted prediction
    """
    HEAD_TYPES = {
        'direct': 'stable',
        'analog': 'similarity',
        'lion': 'gnn',
        'admet': 'gnn',
        'agile': 'physics',
        'cpp': 'physics',
        # MD/QM physics head (Block D'): xTB charges + MARTINI a_head/CPP +
        # Helfrich escape. First-principles physics, designed to extrapolate
        # outside the analog-similarity regime, so it grows with novelty.
        'qmmd': 'physics',
    }

    raw_weights = {}
    for head, base_w in weights_at_zero.items():
        htype = HEAD_TYPES.get(head, 'stable')
        if htype == 'similarity':
            raw_weights[head] = base_w * np.exp(-k_decay * novelty)
        elif htype == 'gnn':
            raw_weights[head] = base_w * np.exp(-k_decay * 0.5 * novelty)
        elif htype == 'physics':
            raw_weights[head] = base_w * np.exp(k_grow * novelty)
        else:  # stable
            raw_weights[head] = base_w

    # Normalize
    total = sum(raw_weights.values())
    if total == 0:
        total = 1.0

    pred = sum(raw_weights[h] / total * head_preds[h] for h in head_preds)
    return pred, {h: raw_weights[h]/total for h in raw_weights}


def fit_adaptive_weights(head_loo_preds, y_true, novelty_scores):
    """Fit the k_decay and k_grow parameters + base weights on LOO data."""

    heads = list(head_loo_preds.keys())
    n = len(y_true)

    def objective(params):
        base_weights = {h: max(params[i], 0.01) for i, h in enumerate(heads)}
        k_decay = max(params[len(heads)], 0.0)
        k_grow = max(params[len(heads)+1], 0.0)

        total_loss = 0
        for j in range(n):
            hp = {h: head_loo_preds[h][j] for h in heads}
            pred, _ = adaptive_blend(hp, novelty_scores[j], base_weights, k_decay, k_grow)
            total_loss += abs(pred - y_true[j])
        return total_loss / n

    # Initial: equal weights, moderate decay/grow
    x0 = [1.0/len(heads)] * len(heads) + [3.0, 2.0]

    # Try multiple starting points
    best_result = None
    best_loss = float('inf')

    for k_d_init in [1.0, 3.0, 5.0, 8.0]:
        for k_g_init in [1.0, 2.0, 4.0]:
            x0_trial = [1.0/len(heads)] * len(heads) + [k_d_init, k_g_init]
            result = minimize(objective, x0_trial, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-5})
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result

    params = best_result.x
    base_weights = {h: max(params[i], 0.01) for i, h in enumerate(heads)}
    k_decay = max(params[len(heads)], 0.0)
    k_grow = max(params[len(heads)+1], 0.0)

    return base_weights, k_decay, k_grow, best_loss


def build_adaptive_stacker():
    """Build the full adaptive stacker with all 6 heads."""

    # Load training data
    with open('IAJD_master/bundles_caches/bioact_v14_bundle.pkl', 'rb') as f:
        b = pickle.load(f)
    X_all = np.asarray(b["X_train"], dtype=float)
    y_all = np.asarray(b["y_train"], dtype=float)
    families_all = np.asarray(b["families_train"])
    smiles_all = list(b["smis_train"])
    n = len(y_all)

    # LOO components
    d = np.load('bioact_loo_components.npz', allow_pickle=True)
    direct_loo = d["direct"]; analog_loo = d["analog"]
    lion_loo = d["lion"]; admet_loo = d["admet"]

    # AGILE
    agile_emb = np.load('agile_embeddings_v14_train.npy')
    AGILE_HP = dict(n_estimators=150, max_depth=3, learning_rate=0.08,
                    min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
                    reg_lambda=3.0, random_state=42, n_jobs=1)

    # CPP
    X_cpp = np.load('cpp_features_v14_train.npy')
    CPP_HP = dict(n_estimators=150, max_depth=3, learning_rate=0.08,
                  min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=3.0, random_state=42, n_jobs=1)

    # QM/MD physics head (Block D' from W-A/W-B/W-C). Pulled via
    # physics_cache_io.load_physics so the same code-path serves training and
    # inference. Falls through to NaN where the cache/emulator are missing.
    X_qmmd_path = ROOT / 'qmmd_features_v14_train.npy'
    QMMD_HP = dict(n_estimators=200, max_depth=3, learning_rate=0.06,
                    min_child_weight=4, subsample=0.85, colsample_bytree=0.7,
                    reg_lambda=3.0, random_state=42, n_jobs=1)
    if X_qmmd_path.exists():
        X_qmmd = np.load(X_qmmd_path)
        print(f"  QM/MD physics block: shape={X_qmmd.shape}")
    else:
        X_qmmd = None
        print(f"  QM/MD physics block: not found ({X_qmmd_path.name}) — "
              f"build via build_qmmd_block_for_training() first")

    # Compute training fingerprints for novelty scoring
    try:
        fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
        _fp = lambda m: fpgen.GetFingerprint(m)
    except AttributeError:
        _fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)
    train_fps = []
    for smi in smiles_all:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(_fp(mol))

    # Compute LOO novelty scores (each compound's max Tanimoto to rest of training)
    print("Computing LOO novelty scores...")
    novelty_loo = np.zeros(n)
    for i in range(n):
        mol = Chem.MolFromSmiles(smiles_all[i])
        if mol is None:
            novelty_loo[i] = 1.0
            continue
        qfp = _fp(mol)
        other_fps = [train_fps[j] for j in range(n) if j != i]
        sims = BulkTanimotoSimilarity(qfp, other_fps)
        novelty_loo[i] = 1.0 - max(sims) if sims else 1.0

    print(f"  Novelty range: {novelty_loo.min():.3f} - {novelty_loo.max():.3f}")
    print(f"  Novelty mean: {novelty_loo.mean():.3f}")

    # Generate AGILE and CPP LOO predictions
    fam_labels = np.array([f if f in ['sSS-Nonsym','PE-Tris','GA-Tris','Dialkoxybenzyl','PE-Gallic']
                           else 'other' for f in families_all])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    agile_loo = np.full(n, np.nan)
    cpp_loo = np.full(n, np.nan)
    qmmd_loo = np.full(n, np.nan) if X_qmmd is not None else None
    for tr, te in skf.split(X_cpp, fam_labels):
        # AGILE
        sc = StandardScaler().fit(agile_emb[tr])
        pc = PCA(n_components=16, random_state=42)
        a_tr = pc.fit_transform(sc.transform(agile_emb[tr]))
        a_te = pc.transform(sc.transform(agile_emb[te]))
        m = xgb.XGBRegressor(**AGILE_HP).fit(a_tr, y_all[tr], verbose=False)
        agile_loo[te] = m.predict(a_te)
        # CPP
        m2 = xgb.XGBRegressor(**CPP_HP).fit(X_cpp[tr], y_all[tr], verbose=False)
        cpp_loo[te] = m2.predict(X_cpp[te])
        # QM/MD
        if X_qmmd is not None:
            m3 = xgb.XGBRegressor(**QMMD_HP).fit(X_qmmd[tr], y_all[tr], verbose=False)
            qmmd_loo[te] = m3.predict(X_qmmd[te])

    print(f"  AGILE LOO MAE: {mean_absolute_error(y_all, agile_loo):.4f}")
    print(f"  CPP LOO MAE:   {mean_absolute_error(y_all, cpp_loo):.4f}")
    if qmmd_loo is not None and np.isfinite(qmmd_loo).all():
        print(f"  QM/MD LOO MAE: {mean_absolute_error(y_all, qmmd_loo):.4f}")

    # ── Fit adaptive weights ──
    print("\nFitting adaptive weights...")
    head_loo = {
        'direct': direct_loo,
        'analog': analog_loo,
        'lion': lion_loo,
        'admet': admet_loo,
        'agile': agile_loo,
        'cpp': cpp_loo,
    }
    if qmmd_loo is not None:
        head_loo['qmmd'] = qmmd_loo

    base_weights, k_decay, k_grow, opt_mae = fit_adaptive_weights(
        head_loo, y_all, novelty_loo)

    print(f"\n  Optimal parameters:")
    print(f"    k_decay (similarity falloff): {k_decay:.2f}")
    print(f"    k_grow (physics boost):       {k_grow:.2f}")
    print(f"    Base weights:")
    for h, w in sorted(base_weights.items(), key=lambda x: -x[1]):
        print(f"      {h:10s}: {w:.4f}")
    print(f"    Adaptive MAE: {opt_mae:.4f}")

    # Compare: static 4-head stacker
    STACKER_HP = dict(n_estimators=300, max_depth=3, learning_rate=0.03, random_state=42, n_jobs=1)
    X_stack_4 = np.column_stack([direct_loo, analog_loo, lion_loo, admet_loo])
    s4_loo = np.full(n, np.nan)
    for tr, te in skf.split(X_stack_4, fam_labels):
        s4 = xgb.XGBRegressor(**STACKER_HP).fit(X_stack_4[tr], y_all[tr], verbose=False)
        s4_loo[te] = s4.predict(X_stack_4[te])
    static_mae = mean_absolute_error(y_all, s4_loo)

    # Adaptive predictions on LOO
    adaptive_preds = np.full(n, np.nan)
    adaptive_weight_log = []
    for i in range(n):
        hp = {h: head_loo[h][i] for h in head_loo}
        pred, weights = adaptive_blend(hp, novelty_loo[i], base_weights, k_decay, k_grow)
        adaptive_preds[i] = pred
        adaptive_weight_log.append(weights)

    adaptive_mae = mean_absolute_error(y_all, adaptive_preds)

    print(f"\n  {'Model':35s} {'LOO MAE':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Static 4-head stacker':35s} {static_mae:>10.4f}")
    print(f"  {'Adaptive 6-head (this)':35s} {adaptive_mae:>10.4f}")
    print(f"  {'Improvement':35s} {static_mae - adaptive_mae:>+10.4f}")

    # Show weight behavior at different novelty levels
    print(f"\n  Weight behavior by novelty:")
    for nov in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        _, w = adaptive_blend({h: 0 for h in head_loo}, nov, base_weights, k_decay, k_grow)
        parts = " ".join(f"{h[:3]}={w[h]:.2f}" for h in ['direct','analog','lion','admet','agile','cpp'])
        print(f"    novelty={nov:.1f}: {parts}")

    # Per-family
    print(f"\n  Per-family:")
    for fam in sorted(set(families_all)):
        mask = families_all == fam
        if mask.sum() < 3: continue
        s_mae = mean_absolute_error(y_all[mask], s4_loo[mask])
        a_mae = mean_absolute_error(y_all[mask], adaptive_preds[mask])
        avg_nov = novelty_loo[mask].mean()
        print(f"    {fam:25s} n={mask.sum():>3}  static={s_mae:.3f}  adaptive={a_mae:.3f}  Δ={a_mae-s_mae:+.3f}  avg_nov={avg_nov:.3f}")

    # ── Train production heads ──
    print("\nTraining production heads...")
    BLOCK_LION = slice(50, 64); BLOCK_ADMET = slice(64, 74)

    direct_head = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1
    ).fit(X_all, y_all, verbose=False)
    lion_head = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1
    ).fit(X_all[:, BLOCK_LION], y_all, verbose=False)
    admet_head = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=1
    ).fit(X_all[:, BLOCK_ADMET], y_all, verbose=False)

    agile_scaler = StandardScaler().fit(agile_emb)
    agile_pca = PCA(n_components=16, random_state=42)
    agile_pca_data = agile_pca.fit_transform(agile_scaler.transform(agile_emb))
    agile_head = xgb.XGBRegressor(**AGILE_HP).fit(agile_pca_data, y_all, verbose=False)

    cpp_head = xgb.XGBRegressor(**CPP_HP).fit(X_cpp, y_all, verbose=False)
    qmmd_head = None
    if X_qmmd is not None:
        qmmd_head = xgb.XGBRegressor(**QMMD_HP).fit(X_qmmd, y_all, verbose=False)

    # Static 4-head meta-stacker on [direct, analog, lion, admet] LOO preds, kept
    # in the bundle so legacy consumers (train_v15_physics_ml, iajd_predict's
    # static fallback) still find a "stacker" key; adaptive_params drives the
    # preferred path.
    static_stacker = xgb.XGBRegressor(**STACKER_HP).fit(
        np.column_stack([direct_loo, analog_loo, lion_loo, admet_loo]),
        y_all, verbose=False)

    # Save bundle
    stack_features = ["direct", "analog", "lion", "admet", "agile", "cpp"]
    if X_qmmd is not None:
        stack_features.append("qmmd")
    bundle = {
        "version": "adaptive_stacker_v2" if X_qmmd is not None else "adaptive_stacker_v1",
        "direct_head": direct_head,
        "lion_head": lion_head,
        "admet_head": admet_head,
        "agile_head": agile_head,
        "agile_pca": agile_pca,
        "agile_scaler": agile_scaler,
        "cpp_head": cpp_head,
        "qmmd_head": qmmd_head,
        "stacker": static_stacker,
        "block_lion": (BLOCK_LION.start, BLOCK_LION.stop),
        "block_admet": (BLOCK_ADMET.start, BLOCK_ADMET.stop),
        "adaptive_params": {
            "base_weights": base_weights,
            "k_decay": float(k_decay),
            "k_grow": float(k_grow),
        },
        "stack_features": stack_features,
        "train_fps": train_fps,
        "train_metrics": {
            "n_train": n,
            "static_4head_loo_mae": round(static_mae, 4),
            "adaptive_loo_mae": round(adaptive_mae, 4),
            "improvement": round(static_mae - adaptive_mae, 4),
            "has_qmmd_head": qmmd_head is not None,
        },
    }

    out_path = 'IAJD_master/bundles_caches/bioact_stacker_bundle.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\nSaved adaptive stacker → {out_path}")

    return bundle


def build_qmmd_block_for_training(smiles_list=None, families_list=None,
                                    out_path=None):
    """Materialize the qmmd_features_v14_train.npy feature block from the
    physics caches. Call this once before retraining the stacker so the
    qmmd head trains on the cached QM/MD values.

    Schema: one row per training compound. Columns = physics_cache_io
    BLOCK_DPRIME_KEYS (14 features). NaN where the cache/emulator is missing.
    """
    if smiles_list is None or families_list is None:
        # Load from the v14 bundle.
        with open('IAJD_master/bundles_caches/bioact_v14_bundle.pkl', 'rb') as f:
            b = pickle.load(f)
        smiles_list = list(b["smis_train"])
        families_list = list(b["families_train"])
    from physics_cache_io import load_physics, BLOCK_DPRIME_KEYS
    # Need the row dict for head_group + pKa, so re-load the bioact xlsx.
    import pandas as pd
    df_bio = pd.read_excel(ROOT / 'IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx')
    bio_by_smi = {}
    for _, r in df_bio.iterrows():
        m = Chem.MolFromSmiles(r.get("SMILES_canonical") or "")
        if m is None:
            continue
        bio_by_smi[Chem.MolToSmiles(m)] = r.to_dict()
    n = len(smiles_list)
    X = np.full((n, len(BLOCK_DPRIME_KEYS)), np.nan)
    for i, smi in enumerate(smiles_list):
        row = bio_by_smi.get(smi, {})
        head = row.get("head_group")
        pka = row.get("pKa") or row.get("pKa_paper")
        try:
            pka_v = float(pka) if pka is not None and not pd.isna(pka) else None
        except (TypeError, ValueError):
            pka_v = None
        rec = load_physics(smi, head_group=head, pka=pka_v)
        flat = rec.as_block_dprime()
        for j, k in enumerate(BLOCK_DPRIME_KEYS):
            X[i, j] = float(flat.get(k, float("nan")))
    out_path = out_path or (ROOT / "qmmd_features_v14_train.npy")
    np.save(out_path, X)
    print(f"Saved qmmd training block: {out_path}  shape={X.shape}  "
          f"NaN fraction={np.isnan(X).mean():.2%}")
    return X


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--build-qmmd-block":
        build_qmmd_block_for_training()
    else:
        bundle = build_adaptive_stacker()
