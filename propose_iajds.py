"""
propose_iajds.py — beam search for IAJDs predicted to clear a bioactivity threshold.

Pipeline:
  1. Build the structural library from the current bioact xlsx
  2. Seed with the top-K training IAJDs (highest measured log10_flux_total)
     OR with a user-supplied SMILES
  3. For depth=D rounds:
       - Apply all single-mutation operators (iajd_grammar.propose_single_mutations)
       - Score each candidate with the bioact regressor + binary head
       - Filter for Tanimoto novelty (< 0.85 to existing training set) and
         a rough synthesizability gate (RDKit sanitization passes, no exotic atoms,
         MolWt ≤ 1500, RotatableBonds ≤ 50)
       - Keep top-N by combined criterion: predicted log10_flux × P(≥T)
  4. Return ranked candidate list with mutation provenance

Outputs:
  - proposed_iajds.csv  (ranked candidates with scores + mutation trail)
  - proposed_iajds_summary.json

CLI:
  python propose_iajds.py --threshold 8.0 --beam 20 --depth 3 --seed top10
"""
from __future__ import annotations
import argparse, json, pickle, sys, warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IAJD_master/code"))

from iajd_grammar import (
    Library, Seed, build_library, decompose_row,
    propose_single_mutations, FAMILY_ASSEMBLERS,
)

BIOACT_XLSX = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
BIN_BUNDLE  = ROOT / "IAJD_master/bundles_caches/bioact_binary_bundle.pkl"
V14_BUNDLE  = ROOT / "IAJD_master/bundles_caches/bioact_v14_bundle.pkl"

# Filters
MAX_MW = 1500.0
MAX_ROTATABLE = 50
# Tanimoto gating: only require candidate != training compound (1.0 means
# identical). Higher similarity is desirable here because the regressor was
# trained on those nearby compounds and can score them confidently.
NOVELTY_TANIMOTO_MAX = 0.999

try:
    _MFP_GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    def _fp(m): return _MFP_GEN.GetFingerprint(m)
except AttributeError:
    def _fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _passes_filters(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    try: Chem.SanitizeMol(m)
    except Exception: return False
    if Descriptors.ExactMolWt(m) > MAX_MW: return False
    if Descriptors.NumRotatableBonds(m) > MAX_ROTATABLE: return False
    for a in m.GetAtoms():
        if a.GetAtomicNum() not in (1, 6, 7, 8): return False
    return True


def _tanimoto_max(query_fp, train_fps) -> float:
    if not train_fps: return 0.0
    return max(TanimotoSimilarity(query_fp, tf) for tf in train_fps)


_BIOACT_LOOKUP = None      # canonical_smiles -> dict (training row)
_DESC_CACHE: dict = {}     # canonical_smiles -> dict of feature columns
_V91_BUNDLE_CACHE = None   # v9.1 pKa bundle, lazy-loaded
_LIVE_PKA_CACHE: dict = {} # canonical SMILES -> (pKa, pKa_sd) from live v9.1


def _bioact_lookup():
    """Lazy-load the bioact xlsx as a dict {canonical_smiles -> row dict}.

    This lets us reuse pre-computed Block A descriptors (Pct_V_Bur_max, etc.)
    for training-set queries, instead of getting 0s from the on-the-fly path.
    """
    global _BIOACT_LOOKUP
    if _BIOACT_LOOKUP is not None:
        return _BIOACT_LOOKUP
    df = pd.read_excel(BIOACT_XLSX)
    out = {}
    for _, r in df.iterrows():
        sm = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(sm):
            continue
        try:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(str(sm)))
        except Exception:
            continue
        out[canon] = r.to_dict()
    _BIOACT_LOOKUP = out
    print(f"  bioact lookup: {len(out)} canonical-SMILES → row entries")
    return out


def _features_for_query_smiles(smiles: str) -> dict:
    """Compute the full set of Block A descriptors for a NEW (mutated) SMILES.

    Reuses expand_datasets' compute_rdkit_features + compute_3d_features so
    the column set matches what _compute_block_a_from_row reads from the row.
    """
    if smiles in _DESC_CACHE:
        return _DESC_CACHE[smiles]
    from expand_datasets import compute_rdkit_features, compute_3d_features
    d = compute_rdkit_features(smiles) or {}
    d3 = compute_3d_features(smiles) or {}
    d.update(d3)
    # Aliases needed by _compute_block_a_from_row
    d.setdefault("Linker_Length", d.get("Linker_Length", np.nan))
    d.setdefault("Taft_Steric_Sum", d.get("Taft_Steric_Sum", np.nan))
    _DESC_CACHE[smiles] = d
    return d


def _live_pka_for_smiles(smiles: str, family_hint: str = "GA-Tris"):
    """Live v9.1 pKa prediction (uses live MolGpKa + per-family debias)."""
    global _V91_BUNDLE_CACHE
    if smiles in _LIVE_PKA_CACHE:
        return _LIVE_PKA_CACHE[smiles]
    try:
        from iajd_pka_v91 import load_v91_bundle, predict_pka_v91
        if _V91_BUNDLE_CACHE is None:
            import os
            xlsx_dir = str((ROOT / "IAJD_master/datasets").resolve())
            cwd = os.getcwd()
            try:
                os.chdir(xlsx_dir)
                _V91_BUNDLE_CACHE = load_v91_bundle(
                    xlsx_path="IAJD_pKa_v21_final.xlsx",
                    debias_path="molgpka_debias_models.joblib",
                    molgpka_npy="molgpka_preds.npy",
                )
            finally:
                os.chdir(cwd)
        r = predict_pka_v91(smiles, _V91_BUNDLE_CACHE, family_hint=family_hint,
                              return_diagnostics=False)
        if "error" in r:
            pair = (float("nan"), float("nan"))
        else:
            pi90 = r.get("pKa_PI_90") or [r.get("pKa_pred"), r.get("pKa_pred")]
            pka = float(r.get("pKa_pred"))
            sd = float((pi90[1] - pi90[0]) / 3.29) if len(pi90) == 2 else 0.3
            pair = (pka, sd)
    except Exception:
        pair = (float("nan"), float("nan"))
    _LIVE_PKA_CACHE[smiles] = pair
    return pair


def _featurize_for_v14(smiles_list: List[str]):
    """Run the same feature-assembly path the v14 bundle uses, returning X.

    For training-set SMILES we re-use the stored xlsx row (which has all the
    pre-computed descriptors). For new SMILES we compute them on the fly.
    """
    from bioact_v14_pipeline import assemble_X
    OUT = ROOT / "IAJD_master/bundles_caches"
    lion_cache  = OUT / "lion_cache_v13.json"
    admet_cache = OUT / "admet_cache_v13.json"
    lion_fps    = OUT / "lion_train_fps.pkl"
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps_q = [_fp(m) if m is not None else None for m in mols]
    lookup = _bioact_lookup()
    rows = []
    for sm in smiles_list:
        canon = sm
        try:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
        except Exception:
            pass
        if canon in lookup:
            # Reuse the cached descriptor block from the training row
            row = dict(lookup[canon])
        else:
            # New SMILES — compute descriptors on the fly
            row = _features_for_query_smiles(canon)
            # Live v9.1 pKa + per-family debias (no proxy fallback)
            if pd.isna(row.get("pKa")):
                fam_hint = row.get("family") or "GA-Tris"
                pka_pred, pka_sd = _live_pka_for_smiles(canon, family_hint=fam_hint)
                if np.isfinite(pka_pred):
                    row["pKa"] = pka_pred
                    row["pKa_sd"] = pka_sd
        row["canonical_smi"] = canon
        if "family" not in row or pd.isna(row.get("family")):
            row["family"] = "GA-Tris"   # default placeholder
        row["log10_flux_total"] = np.nan
        rows.append(row)
    df_q = pd.DataFrame(rows)
    X, _modes = assemble_X(
        df_q, mols, fps_q, smiles_list,
        lion_cache_path=str(lion_cache) if lion_cache.exists() else None,
        admet_cache_path=str(admet_cache) if admet_cache.exists() else None,
        lion_train_fps_path=str(lion_fps) if lion_fps.exists() else None,
    )
    return X


def _extend_caches_for_smiles(smiles_list: List[str]) -> None:
    """Run real LION + ADMET subprocess prediction for SMILES NOT yet in the
    caches; populates them in place so the next assemble_X call hits real
    values instead of RDKit proxies."""
    import json
    sys.path.insert(0, str(ROOT / "IAJD_master/code"))
    try:
        from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles
    except Exception as e:
        print(f"  WARN: cache-extension import failed ({e}); falling back to proxies.")
        return
    OUT = ROOT / "IAJD_master/bundles_caches"
    lion_path = OUT / "lion_cache_v13.json"
    admet_path = OUT / "admet_cache_v13.json"
    if not lion_path.exists() or not admet_path.exists():
        return
    lion_cache = json.load(open(lion_path))
    admet_cache = json.load(open(admet_path))
    # Canonical SMILES of each query
    canon = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon.append(Chem.MolToSmiles(m))
    to_lion = sorted({c for c in canon if c not in lion_cache})
    to_admet = sorted({c for c in canon if c not in admet_cache})
    if not to_lion and not to_admet:
        return
    if to_admet:
        print(f"  extending ADMET cache: +{len(to_admet)} SMILES")
        try: predict_admet_for_smiles(to_admet, cache_path=str(admet_path), verbose=False)
        except Exception as e: print(f"    ADMET extension failed: {e}")
    if to_lion:
        print(f"  extending LION cache:  +{len(to_lion)} SMILES")
        try: predict_lion_for_smiles(to_lion, cache_path=str(lion_path), verbose=False)
        except Exception as e: print(f"    LION extension failed: {e}")


def _score_with_v14(smiles_list: List[str], bundle_v14: dict, bundle_bin: dict,
                    threshold: float, extend_cache: bool = True):
    """Return (yhat, p_above) arrays for the candidate SMILES list."""
    if extend_cache:
        _extend_caches_for_smiles(smiles_list)
    X = _featurize_for_v14(smiles_list)
    reg = bundle_bin["regressor"]
    yhat = reg.predict(X)
    # Probability via per-threshold calibrator (interpolate if T not in grid)
    grid = sorted(bundle_bin["threshold_calibrators"].keys(), key=float)
    grid_floats = np.array([float(g) for g in grid])
    # Nearest grid point
    nearest = grid[int(np.argmin(np.abs(grid_floats - threshold)))]
    c = bundle_bin["threshold_calibrators"][nearest]
    a, b = c["a"], c["b"]
    p_above = 1.0 / (1.0 + np.exp(-(a * (yhat - threshold) + b)))
    return yhat, p_above


def _pick_seeds(df: pd.DataFrame, top_k: int) -> List[dict]:
    have = df.dropna(subset=["log10_flux_total"]).copy()
    have = have.sort_values("log10_flux_total", ascending=False).head(top_k)
    return [r.to_dict() for _, r in have.iterrows()]


def beam_search(threshold: float, beam: int, depth: int,
                seeds: List[Seed], library: Library,
                bundle_v14: dict, bundle_bin: dict,
                train_fps) -> pd.DataFrame:
    """Run depth-D beam search of width `beam`, keeping the best by combined score."""
    explored = set()
    current = []
    # Score the seeds first. Prefer the ORIGINAL training SMILES (cached) for
    # seeds that came from training rows; falls back to assembled form
    # otherwise.
    seed_smis = []
    seed_kept = []
    for s in seeds:
        sm = s.original_smiles or s.assemble()
        if sm is None:
            continue
        seed_smis.append(sm)
        seed_kept.append(s)
    seeds = seed_kept
    seed_yhat, seed_p = _score_with_v14(seed_smis, bundle_v14, bundle_bin, threshold,
                                          extend_cache=False)  # training seeds already cached
    for s, sm, yh, p in zip(seeds, seed_smis, seed_yhat, seed_p):
        current.append({
            "smiles": sm, "seed": s, "yhat": float(yh), "p_above": float(p),
            "mutation_trail": ["seed"],
            "score": float(yh) * float(p),
        })
        explored.add(sm)

    all_candidates = list(current)

    for d in range(depth):
        # Sort current by score, keep top beam
        current.sort(key=lambda r: -r["score"])
        current = current[:beam]
        print(f"  [beam d={d}] current best score={current[0]['score']:.3f}  "
              f"(ŷ={current[0]['yhat']:.2f}, P≥{threshold}={current[0]['p_above']:.2f})")

        next_pool = []
        for parent in current:
            for cand_smi, tag, cand_seed in propose_single_mutations(parent["seed"], library):
                if cand_smi in explored:
                    continue
                if not _passes_filters(cand_smi):
                    continue
                m = Chem.MolFromSmiles(cand_smi)
                cand_fp = _fp(m)
                tanim = _tanimoto_max(cand_fp, train_fps)
                if tanim >= NOVELTY_TANIMOTO_MAX:
                    # identical to a training compound — skip
                    continue
                explored.add(cand_smi)
                next_pool.append({
                    "smiles": cand_smi,
                    "seed": cand_seed,
                    "parent_smiles": parent["smiles"],
                    "mutation_tag": tag,
                    "mutation_trail": parent["mutation_trail"] + [tag],
                    "tanim_max_to_train": float(tanim),
                })

        # Score this round's pool in a single batch (faster)
        if not next_pool:
            print(f"  [beam d={d}] no new candidates; stopping.")
            break
        smis = [r["smiles"] for r in next_pool]
        yhats, ps = _score_with_v14(smis, bundle_v14, bundle_bin, threshold)
        for r, yh, p in zip(next_pool, yhats, ps):
            r["yhat"] = float(yh)
            r["p_above"] = float(p)
            r["score"] = float(yh) * float(p)
        all_candidates.extend(next_pool)
        current = next_pool

    df = pd.DataFrame(all_candidates)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="Bioactivity threshold (log10_flux_total ≥ T)")
    ap.add_argument("--beam", type=int, default=20,
                    help="Beam width")
    ap.add_argument("--depth", type=int, default=3,
                    help="Beam search depth (mutation rounds)")
    ap.add_argument("--top_seeds", type=int, default=10,
                    help="Number of top training IAJDs to seed from")
    ap.add_argument("--seed_smiles", default=None,
                    help="Optional explicit seed SMILES (overrides --top_seeds)")
    ap.add_argument("--out_csv", default="proposed_iajds.csv")
    ap.add_argument("--out_json", default="proposed_iajds_summary.json")
    args = ap.parse_args()

    print(f"Building structural library from {BIOACT_XLSX.name}…")
    library = build_library(BIOACT_XLSX)
    print(f"  {len(library.families)} families")

    print(f"Loading bioact regressor + binary heads…")
    if not BIN_BUNDLE.exists():
        print(f"  WARN: {BIN_BUNDLE} not found. Run train_binary_classifier.py first.")
        sys.exit(1)
    if not V14_BUNDLE.exists():
        print(f"  WARN: {V14_BUNDLE} not found. Run bioact_v14_pipeline.py first.")
        sys.exit(1)
    with open(V14_BUNDLE, "rb") as f:
        bundle_v14 = pickle.load(f)
    with open(BIN_BUNDLE, "rb") as f:
        bundle_bin = pickle.load(f)
    print(f"  v14 n_train={len(bundle_v14.get('y_train', []))}, "
          f"bin n_train={bundle_bin['metrics']['n_train']}, "
          f"bin reg LOO MAE={bundle_bin['metrics']['regression_loo_mae']:.3f}")

    # Seeds
    df = pd.read_excel(BIOACT_XLSX)
    if args.seed_smiles:
        seed_row = {"family": "GA-Tris", "head_group": "HPRZ",
                    "linker_length": 4, "linkage": "ester",
                    "SMILES_canonical": args.seed_smiles}
        seeds_list = [decompose_row(seed_row)]
    else:
        seed_rows = _pick_seeds(df, args.top_seeds)
        seeds_list = [decompose_row(r) for r in seed_rows]
        seeds_list = [s for s in seeds_list if s is not None]
    print(f"\nSeeding from {len(seeds_list)} structures:")
    for s in seeds_list[:5]:
        print(f"  IAJD {s.iajd_num}: family={s.family} head={s.head} "
              f"link={s.linker_n}C  y_obs={s.y_obs}")

    # Train FPs for novelty
    train_fps = []
    for _, r in df.iterrows():
        smi = r.get("SMILES_canonical") or r.get("SMILES")
        if pd.isna(smi): continue
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            train_fps.append(_fp(m))

    print(f"\nRunning beam search: threshold={args.threshold} beam={args.beam} depth={args.depth}")
    result = beam_search(args.threshold, args.beam, args.depth,
                         seeds_list, library, bundle_v14, bundle_bin, train_fps)

    # Trim columns for CSV
    out_cols = ["smiles", "yhat", "p_above", "score", "tanim_max_to_train",
                "mutation_tag", "mutation_trail", "parent_smiles"]
    for c in out_cols:
        if c not in result.columns:
            result[c] = None
    result[out_cols].to_csv(args.out_csv, index=False)

    summary = {
        "threshold": args.threshold,
        "beam": args.beam, "depth": args.depth,
        "n_candidates_total": len(result),
        "n_novel": int((result["tanim_max_to_train"] < NOVELTY_TANIMOTO_MAX).sum()),
        "top_10": result.head(10)[out_cols].to_dict(orient="records"),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDONE — {len(result)} candidates scored")
    print(f"  Novel (tanim < {NOVELTY_TANIMOTO_MAX}): {summary['n_novel']}")
    print(f"  Top 5:")
    for _, r in result.head(5).iterrows():
        print(f"    score={r['score']:.3f}  ŷ={r['yhat']:.2f}  "
              f"P(≥{args.threshold})={r['p_above']:.2f}  "
              f"tanim={r['tanim_max_to_train']:.2f}  "
              f"trail={r['mutation_trail'][-3:]}")
        print(f"      {r['smiles']}")
    print(f"\n  Wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
