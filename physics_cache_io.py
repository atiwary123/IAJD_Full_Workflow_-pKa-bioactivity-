"""
physics_cache_io.py — unified cache lookup + emulator fallback + NaN contract
for the QM (xTB) and MD (MARTINI) physics layers.

This is the single integration surface between the offline-precomputed physics
caches and the in-Space scoring pipeline. Block D' in bioact_v14_pipeline reads
through this module; it never touches the cache files or emulators directly.

The contract:
  load_physics(smiles)
    ──► canonicalize SMILES
    ──► look up qm_cache.csv  : hit → real QM values
                                  miss → qm_emulator.joblib (Space-side regressor)
                                          → if no emulator, all-NaN
    ──► look up md_cache.csv  : hit → real MD observables
                                  miss → md_emulator.joblib
                                          → if no emulator, all-NaN
    ──► look up head_area_ensemble.csv : hit → Boltzmann-weighted head_area_nm2
                                             miss → live single-conformer head_area
                                             (head_area_3d.head_area_for_group)

Honest no-proxy invariant: anything we couldn't compute returns NaN — never a
median, never a constant. XGBoost's default branch handles NaN in Block D'.

Performance:
  - caches load once and stay in memory (~hundreds of rows × <20 cols).
  - emulators load lazily on first miss.
"""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent
PHYS_DIR = ROOT / "IAJD_master/bundles_caches/physics"

QM_CACHE_CSV   = PHYS_DIR / "qm_cache.csv"
MD_CACHE_CSV   = PHYS_DIR / "md_cache.csv"
HEAD_AREA_CSV  = PHYS_DIR / "head_area_ensemble.csv"
QM_EMU_JOBLIB  = PHYS_DIR / "qm_emulator.joblib"
MD_EMU_JOBLIB  = PHYS_DIR / "md_emulator.joblib"

# Keep these strings in lock-step with qm_descriptors.QM_KEYS / md.OBSERVABLE
# keys so a typo in either file is loud (KeyError at first miss).
QM_KEYS = (
    "qm_q_ionizableN", "qm_dipole_D", "qm_polarizability",
    "qm_homo_lumo_eV", "qm_dGsolv_kJmol",
    "qm_dGsolv_head", "qm_dGsolv_tail", "qm_Ehedup",
)
MD_KEYS = (
    "md_assembles", "md_n_agg",
    "md_a_head_neutral_nm2", "md_a_head_prot_nm2", "md_delta_a_head_nm2",
    "md_bilayer_thick_nm", "md_order_param", "md_radius_gyration_nm",
    "md_water_penetration",
    "md_cpp_neutral", "md_cpp_prot", "md_delta_cpp",
    "md_c0_spontaneous",
)

# Block D' final consumer keys (assembled in load_physics()).
BLOCK_DPRIME_KEYS = (
    "md_a_head_prot_nm2", "md_delta_a_head_nm2", "md_cpp_prot", "md_delta_cpp",
    "md_bilayer_thick_nm", "md_order_param",
    "md_assembles", "md_n_agg",
    "qm_q_ionizableN", "qm_dipole_D", "qm_dGsolv_kJmol", "qm_homo_lumo_eV",
    "dG_escape_helfrich",
    "head_area_nm2",  # ensemble or single-conformer fallback
)


# ──────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ──────────────────────────────────────────────────────────────────────

_lock = threading.RLock()
_qm_df: Optional[pd.DataFrame] = None
_md_df: Optional[pd.DataFrame] = None
_head_area_df: Optional[pd.DataFrame] = None
_qm_emu = None
_md_emu = None


def _canonical(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else ""


def _load_qm_cache() -> pd.DataFrame:
    global _qm_df
    with _lock:
        if _qm_df is None:
            if QM_CACHE_CSV.exists():
                _qm_df = pd.read_csv(QM_CACHE_CSV)
            else:
                _qm_df = pd.DataFrame(columns=["smiles_canonical", *QM_KEYS])
            _qm_df = _qm_df.set_index("smiles_canonical")
    return _qm_df


def _load_md_cache() -> pd.DataFrame:
    global _md_df
    with _lock:
        if _md_df is None:
            if MD_CACHE_CSV.exists():
                _md_df = pd.read_csv(MD_CACHE_CSV)
            else:
                _md_df = pd.DataFrame(columns=["smiles_canonical", *MD_KEYS])
            _md_df = _md_df.set_index("smiles_canonical")
    return _md_df


def _load_head_area_cache() -> pd.DataFrame:
    global _head_area_df
    with _lock:
        if _head_area_df is None:
            if HEAD_AREA_CSV.exists():
                _head_area_df = pd.read_csv(HEAD_AREA_CSV)
            else:
                _head_area_df = pd.DataFrame(columns=["head_group", "head_area_nm2"])
            _head_area_df = _head_area_df.set_index("head_group")
    return _head_area_df


def _load_qm_emulator():
    global _qm_emu
    with _lock:
        if _qm_emu is None and QM_EMU_JOBLIB.exists():
            try:
                import joblib
                _qm_emu = joblib.load(QM_EMU_JOBLIB)
            except (OSError, ImportError, ValueError):
                _qm_emu = False
        if _qm_emu is None:
            _qm_emu = False  # mark as "tried, missing" so we don't retry
    return _qm_emu if _qm_emu is not False else None


def _load_md_emulator():
    global _md_emu
    with _lock:
        if _md_emu is None and MD_EMU_JOBLIB.exists():
            try:
                import joblib
                _md_emu = joblib.load(MD_EMU_JOBLIB)
            except (OSError, ImportError, ValueError):
                _md_emu = False
        if _md_emu is None:
            _md_emu = False
    return _md_emu if _md_emu is not False else None


# ──────────────────────────────────────────────────────────────────────
# Per-source lookup
# ──────────────────────────────────────────────────────────────────────

def lookup_qm(smiles: str) -> Dict[str, float]:
    """Cache → emulator → NaN dict. Returns dict keyed by QM_KEYS."""
    canon = _canonical(smiles)
    if not canon:
        return {k: float("nan") for k in QM_KEYS}
    qm = _load_qm_cache()
    if canon in qm.index:
        row = qm.loc[canon]
        # Handle both Series (single hit) and DataFrame (collision) gracefully.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {k: float(row.get(k, float("nan"))) for k in QM_KEYS}
    emu = _load_qm_emulator()
    if emu is None:
        return {k: float("nan") for k in QM_KEYS}
    try:
        from train_qm_emulator import featurize
    except ImportError:
        return {k: float("nan") for k in QM_KEYS}
    feat = featurize(canon)
    if feat is None:
        return {k: float("nan") for k in QM_KEYS}
    X = feat.reshape(1, -1)
    out = {}
    for k in QM_KEYS:
        m = emu.get("models", {}).get(k) if isinstance(emu, dict) else None
        if m is None:
            out[k] = float("nan")
        else:
            try:
                out[k] = float(m.predict(X)[0])
            except (ValueError, RuntimeError):
                out[k] = float("nan")
    return out


def lookup_md(smiles: str) -> Dict[str, float]:
    """Cache → emulator → NaN dict. Returns dict keyed by MD_KEYS."""
    canon = _canonical(smiles)
    if not canon:
        return {k: float("nan") for k in MD_KEYS}
    md = _load_md_cache()
    if canon in md.index:
        row = md.loc[canon]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {k: float(row.get(k, float("nan"))) for k in MD_KEYS}
    emu = _load_md_emulator()
    if emu is None:
        return {k: float("nan") for k in MD_KEYS}
    try:
        from train_md_emulator import featurize as md_feat
    except ImportError:
        return {k: float("nan") for k in MD_KEYS}
    feat = md_feat(canon)
    if feat is None:
        return {k: float("nan") for k in MD_KEYS}
    X = feat.reshape(1, -1)
    out = {}
    for k in MD_KEYS:
        m = emu.get("models", {}).get(k) if isinstance(emu, dict) else None
        if m is None:
            out[k] = float("nan")
        else:
            try:
                out[k] = float(m.predict(X)[0])
            except (ValueError, RuntimeError):
                out[k] = float("nan")
    return out


def lookup_head_area(head_group: Optional[str], smiles_for_fallback: Optional[str] = None) -> float:
    """Head-area lookup for the ensemble cache. Falls back to live
    head_area_3d.head_area_for_group on miss. NaN if both fail."""
    if head_group:
        df = _load_head_area_cache()
        if head_group in df.index:
            row = df.loc[head_group]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            val = row.get("head_area_nm2")
            if val is not None and np.isfinite(float(val)):
                return float(val)
        try:
            from head_area_3d import head_area_for_group as live
            v = live(head_group)
            if v is not None and np.isfinite(float(v)):
                return float(v)
        except (ImportError, RuntimeError):
            pass
    return float("nan")


# ──────────────────────────────────────────────────────────────────────
# Helfrich escape — assembled here so callers see Block D' as a flat dict
# ──────────────────────────────────────────────────────────────────────

def helfrich_escape(*, md_c0_spontaneous: float,
                    bilayer_thick_nm: float,
                    a_head_nm2: float,
                    delta_protonation: float) -> float:
    """ΔG_escape ≈ ½ · κ_b · (c_prot − c_neutral)² · Δprotonation,
    with κ_b = t² / a (Helfrich/Evans-Skalak proxy in k_B T units).

    All inputs must be finite; NaN propagates honestly.
    """
    if not all(np.isfinite([md_c0_spontaneous, bilayer_thick_nm, a_head_nm2,
                             delta_protonation])):
        return float("nan")
    if bilayer_thick_nm <= 0 or a_head_nm2 <= 0:
        return float("nan")
    kappa_kBT = (bilayer_thick_nm ** 2) / a_head_nm2
    return 0.5 * kappa_kBT * (md_c0_spontaneous ** 2) * delta_protonation


# ──────────────────────────────────────────────────────────────────────
# Block D' assembly
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsRecord:
    smiles_canonical: str
    qm: Dict[str, float] = field(default_factory=dict)
    md: Dict[str, float] = field(default_factory=dict)
    head_area_nm2: float = float("nan")
    dG_escape_helfrich: float = float("nan")

    def as_block_dprime(self) -> Dict[str, float]:
        """Flat dict in BLOCK_DPRIME_KEYS order — what Block D' consumes."""
        merged = {**self.qm, **self.md,
                  "head_area_nm2": self.head_area_nm2,
                  "dG_escape_helfrich": self.dG_escape_helfrich}
        return {k: float(merged.get(k, float("nan"))) for k in BLOCK_DPRIME_KEYS}


def load_physics(smiles: str, *, head_group: Optional[str] = None,
                  pka: Optional[float] = None,
                  ph_endosome: float = 5.5,
                  ph_cytosol: float = 7.4) -> PhysicsRecord:
    """Full physics lookup for one molecule.

    Returns PhysicsRecord with QM, MD, head_area, and computed Helfrich escape.
    NaN-passthrough throughout; XGBoost handles NaN in Block D'.
    """
    canon = _canonical(smiles)
    rec = PhysicsRecord(smiles_canonical=canon or smiles)
    rec.qm = lookup_qm(canon)
    rec.md = lookup_md(canon)
    rec.head_area_nm2 = lookup_head_area(head_group, canon)

    # If MD didn't give us a real a_head, use the head_area cache value.
    a_head_for_helfrich = rec.md.get("md_a_head_prot_nm2", float("nan"))
    if not np.isfinite(a_head_for_helfrich):
        a_head_for_helfrich = rec.head_area_nm2

    # Henderson-Hasselbalch delta protonation, only if pka was passed (real,
    # not proxy). The caller passes the live MolGpKa+debias prediction.
    delta_prot = float("nan")
    if pka is not None and np.isfinite(pka):
        f_endo = 1.0 / (1.0 + 10 ** (ph_endosome - pka))
        f_cyto = 1.0 / (1.0 + 10 ** (ph_cytosol - pka))
        delta_prot = f_endo - f_cyto

    rec.dG_escape_helfrich = helfrich_escape(
        md_c0_spontaneous=rec.md.get("md_c0_spontaneous", float("nan")),
        bilayer_thick_nm=rec.md.get("md_bilayer_thick_nm", float("nan")),
        a_head_nm2=a_head_for_helfrich,
        delta_protonation=delta_prot,
    )
    return rec


def cache_status() -> Dict:
    """Diagnostic snapshot — used by app startup to log what's loaded."""
    qm = _load_qm_cache(); md = _load_md_cache(); ha = _load_head_area_cache()
    return {
        "qm_cache_rows": int(len(qm)),
        "md_cache_rows": int(len(md)),
        "head_area_cache_rows": int(len(ha)),
        "qm_emulator": QM_EMU_JOBLIB.exists(),
        "md_emulator": MD_EMU_JOBLIB.exists(),
        "qm_cache_path": str(QM_CACHE_CSV),
        "md_cache_path": str(MD_CACHE_CSV),
    }


__all__ = [
    "load_physics", "lookup_qm", "lookup_md", "lookup_head_area",
    "helfrich_escape", "cache_status", "PhysicsRecord",
    "QM_KEYS", "MD_KEYS", "BLOCK_DPRIME_KEYS",
]


if __name__ == "__main__":
    import sys
    print(json.dumps(cache_status(), indent=2))
    if len(sys.argv) > 1:
        smi = sys.argv[1]
        rec = load_physics(smi)
        print(json.dumps({
            "smiles": rec.smiles_canonical,
            "qm": rec.qm, "md": rec.md,
            "head_area_nm2": rec.head_area_nm2,
            "dG_escape_helfrich": rec.dG_escape_helfrich,
            "block_dprime": rec.as_block_dprime(),
        }, indent=2, default=str))
