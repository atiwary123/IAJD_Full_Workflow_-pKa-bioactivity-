"""
family_sar.py — shared SAR feature space + per-family informed-mutation prior.

This module is the single source of truth for "what does the training set say
a favourable mutation looks like, for this family?" It is imported by BOTH
  - analyze_family_sar.py  (offline: builds family_sar_priors.json), and
  - propose_iajds.py       (online: scores/prunes/prioritises candidates),
so the priors and the candidate values are always computed by the EXACT same
function — no definitional drift, no 3D dependency, no proxy.

Why a dedicated fast 2D feature set (not the proposer's full featurizer)?
  The bioact xlsx columns were computed by an older feature function than the
  live featurizer (e.g. xlsx Hydrophobic_Index = MolLogP/HeavyAtomCount ≈ 0.205
  vs live MolLogP/MolWt ≈ 0.015; xlsx NumEthers counts aryl-O-alkyl, live
  counts only aliphatic C-O-C → 0). Standardising a candidate's live value
  against an xlsx-derived mean/std would be apples-to-oranges. And the live
  featurizer needs ~20 s/molecule (3D conformer search). So we define one
  cheap (~0.5 ms/mol), unambiguous 2D space here and use it everywhere.

The prior is honest by construction:
  - continuous axes: contribution = ρ · Δz · σ_flux, i.e. the expected change
    in log10_flux_total if the (monotone) training relationship held, where Δz
    is the candidate-minus-seed change of the feature in family-training SDs.
    ρ is the signed Spearman correlation, σ_flux the family flux SD. Only axes
    that pass a significance gate AND survive de-correlation contribute.
  - categorical axes (head_group, linkage): contribution = the observed
    (shrunk) mean-flux gap between the new and old level. Used INSTEAD of the
    continuous term for pure head/linkage swaps, so a head's effect is never
    double-counted via its induced descriptor changes.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as _rd

RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent
PRIORS_JSON = ROOT / "family_sar_priors.json"

# Cap a single axis's standardized move so one runaway feature can't dominate.
_CAP_Z = 2.0
# Cap total prior magnitude (log10-flux units).
_CAP_TOTAL = 2.0

_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ALKOXY = Chem.MolFromSmarts("[#6][OX2][#6]")   # any C-O-C, incl. aryl-O-alkyl

# Canonical fast 2D descriptors. linker_length is handled separately as a
# structural integer (read from the Seed), not computed here.
SAR_FEATURE_NAMES = [
    "MolLogP", "TPSA", "ExactMolWt", "FractionCSP3", "RotatableBonds",
    "NumHAcceptors", "NumHDonors", "NumAromaticRings", "AlkoxyEthers",
    "NumEsters", "LogP_per_HA", "HeavyAtoms",
]

_FEAT_CACHE: dict = {}
_PRIORS_CACHE = None


def sar_features(smiles: str) -> Optional[dict]:
    """Fast, deterministic 2D descriptor dict for one SMILES (or None on parse
    failure). Cached by input string."""
    if smiles in _FEAT_CACHE:
        return _FEAT_CACHE[smiles]
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        _FEAT_CACHE[smiles] = None
        return None
    ha = m.GetNumHeavyAtoms()
    logp = float(Crippen.MolLogP(m))
    n_ester = len(m.GetSubstructMatches(_ESTER))
    ether = max(len(m.GetSubstructMatches(_ALKOXY)) - n_ester, 0)
    d = {
        "MolLogP": logp,
        "TPSA": float(_rd.CalcTPSA(m)),
        "ExactMolWt": float(Descriptors.ExactMolWt(m)),
        "FractionCSP3": float(_rd.CalcFractionCSP3(m)),
        "RotatableBonds": float(_rd.CalcNumRotatableBonds(m)),
        "NumHAcceptors": float(_rd.CalcNumHBA(m)),
        "NumHDonors": float(_rd.CalcNumHBD(m)),
        "NumAromaticRings": float(_rd.CalcNumAromaticRings(m)),
        "AlkoxyEthers": float(ether),
        "NumEsters": float(n_ester),
        "LogP_per_HA": float(logp / ha) if ha else 0.0,
        "HeavyAtoms": float(ha),
    }
    _FEAT_CACHE[smiles] = d
    return d


def load_sar_priors(path: str | Path = PRIORS_JSON) -> dict:
    """Lazy-load + cache family_sar_priors.json. Returns {} if missing."""
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None:
        return _PRIORS_CACHE
    p = Path(path)
    if not p.exists():
        _PRIORS_CACHE = {"families": {}, "_missing": True}
        return _PRIORS_CACHE
    try:
        _PRIORS_CACHE = json.loads(p.read_text())
    except Exception:
        _PRIORS_CACHE = {"families": {}, "_missing": True}
    return _PRIORS_CACHE


def family_has_prior(family: str) -> bool:
    """True iff the family has any data-supported lever that can actually steer
    a mutation: a selected continuous axis, OR a categorical (head/linkage) with
    ≥2 levels (a single-level categorical has delta_vs_family ≡ 0, so it can't
    steer — e.g. PE-Gallic, whose only head is DMBA → honest no-prior)."""
    fam = load_sar_priors().get("families", {}).get(family)
    if not fam:
        return False
    has_cont = any(a.get("selected_for_prior") for a in fam.get("continuous", {}).values())
    def _n_levels(cat):
        return sum(1 for k in fam.get(cat, {}) if k != "_family_mean")
    has_cat = _n_levels("head_group") >= 2 or _n_levels("linkage") >= 2
    return has_cont or has_cat


def _kind_for_tag(tag: str) -> str:
    """Route a mutation tag to a prior 'kind' so head/linkage effects are
    scored categorically (and not double-counted via descriptors)."""
    if tag.startswith("head:"):
        return "head"
    if tag.startswith("linkage:"):
        return "linkage"
    # synth_head, tail*, linker, all_tails, multi_tail, family, seed → continuous
    return "structural"


def sar_prior_for_candidate(
    family: str,
    seed_feats: Optional[dict],
    cand_feats: Optional[dict],
    *,
    mutation_tag: str = "",
    seed_linker: Optional[float] = None,
    cand_linker: Optional[float] = None,
    head_old: Optional[str] = None,
    head_new: Optional[str] = None,
    linkage_old: Optional[str] = None,
    linkage_new: Optional[str] = None,
) -> tuple[float, str, dict]:
    """Signed informed-mutation prior in log10_flux_total units.

    Returns (prior, human_reason, signals_dict). prior > 0 means the training
    SAR expects this mutation to RAISE activity; < 0 means LOWER. 0.0 means
    'no data-supported signal for this move in this family' (e.g. a flat family
    or a low-data family with no gated axes) — an honest neutral, not a guess.
    """
    priors = load_sar_priors()
    fam = priors.get("families", {}).get(family)
    if not fam:
        return 0.0, "", {}
    flux_std = float(fam.get("flux_std") or 0.5)
    kind = _kind_for_tag(mutation_tag) if mutation_tag else "structural"
    signals: dict = {}
    total = 0.0

    # ── categorical levers (routed; used instead of continuous to avoid
    #    double-counting a head/linkage's induced descriptor changes) ──
    if kind == "head" and head_new and head_old and head_new != head_old:
        hg = fam.get("head_group", {})
        a, b = hg.get(head_old), hg.get(head_new)
        if isinstance(a, dict) and isinstance(b, dict):
            d = float(b["delta_vs_family"]) - float(a["delta_vs_family"])
            total += d
            signals["head"] = f"head {head_old}→{head_new}: {d:+.2f}"
        return _finish(total, signals)

    if kind == "linkage" and linkage_new and linkage_old and linkage_new != linkage_old:
        lk = fam.get("linkage", {})
        a, b = lk.get(linkage_old), lk.get(linkage_new)
        if isinstance(a, dict) and isinstance(b, dict):
            d = float(b["delta_vs_family"]) - float(a["delta_vs_family"])
            total += d
            signals["linkage"] = f"linkage {linkage_old}→{linkage_new}: {d:+.2f}"
        return _finish(total, signals)

    # ── continuous axes (structural / tail / linker / synthetic-head) ──
    for f, a in fam.get("continuous", {}).items():
        if not a.get("selected_for_prior"):
            continue
        rho = float(a.get("rho") or 0.0)
        sd = float(a.get("train_std") or 0.0)
        if sd <= 0 or rho == 0.0:
            continue
        if a.get("source") == "structural" and f == "linker_length":
            if seed_linker is None or cand_linker is None:
                continue
            xs, xc = float(seed_linker), float(cand_linker)
        else:
            if not seed_feats or not cand_feats:
                continue
            xs, xc = seed_feats.get(f), cand_feats.get(f)
            if xs is None or xc is None:
                continue
        try:
            xs, xc = float(xs), float(xc)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(xs) and math.isfinite(xc)):
            continue
        dz = max(-_CAP_Z, min(_CAP_Z, (xc - xs) / sd))
        if abs(dz) < 1e-9:
            continue
        contrib = rho * dz * flux_std
        total += contrib
        arrow = "↑" if xc > xs else "↓"
        signals[f] = f"{f}{arrow}{abs(xc - xs) / sd:.1f}σ(ρ{rho:+.2f}):{contrib:+.2f}"

    return _finish(total, signals)


def _finish(total: float, signals: dict) -> tuple[float, str, dict]:
    total = max(-_CAP_TOTAL, min(_CAP_TOTAL, total))
    reason = "; ".join(signals.values())[:240]
    return float(total), reason, signals
