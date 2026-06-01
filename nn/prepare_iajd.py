"""
nn/prepare_iajd.py — assemble the IAJD transfer-learning target table: SMILES + organ flux
labels + the physics descriptors (Module A pKa, Module B c0 when available) + family, for
the frozen-encoder + GP head (docs/IAJD_NN_TRANSFER_LEARNING_PLAN.md stage 3).

This is the LOCAL, fully-testable half of the NN pipeline (no GPU/repos needed). The pod
side (fetch LNPDB/AGILE, pretrain, embed) consumes this CSV. Honest: every row is a REAL
measured IAJD; no synthetic augmentation here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_DS = ROOT / "IAJD_master" / "datasets"
# prefer the audit-corrected SMILES (2026-06-01: 18 PE-Gallic SMILES fixed vs SI). The NN
# computes features fresh from SMILES, so the audit's "features stale" caveat doesn't apply.
BIOACT = (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx") if \
    (_DS / "IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx").exists() else \
    (_DS / "IAJD_Bioact_v13_clean.xlsx")
DESIGN = ROOT / "IAJD_master" / "bundles_caches" / "physics" / "design"
OUT = Path(__file__).resolve().parent / "iajd_transfer_input.csv"


def _col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def physics_features() -> dict:
    """Pull any computed Module A/B values keyed by IAJD_num (provisional ones included,
    flagged). Returns {iajd_num: {c0_nm_inv, c0_trusted, apparent_pKa, ...}}."""
    feats: dict = {}
    if not DESIGN.exists():
        return feats
    for f in DESIGN.glob("IAJD*_curvature.json"):
        try:
            r = json.loads(f.read_text())
            n = int(r.get("iajd"))
            d = feats.setdefault(n, {})
            d["c0_nm_inv"] = r.get("c0_nm_inv")
            d["c0_method"] = r.get("method", "pure")
            d["c0_physics_converged"] = r.get("c0_physics_converged")
            d["c0_trusted"] = r.get("c0_trusted")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return feats


def main():
    df = pd.read_excel(BIOACT)
    sm = _col(df, "SMILES_canonical", "smiles_canonical", "SMILES")
    fam = _col(df, "family", "Family", "architecture")
    keep = {"IAJD_num": "iajd_num", sm: "smiles"}
    for c in ("log10_flux_spleen", "log10_flux_liver", "log10_flux_total",
              "flux_total_SEM", "n_replicates", "n_mice", "pKa", "pKa_paper", "pKa_sd",
              "audit_status"):
        if c in df.columns:
            keep[c] = c
    if fam:
        keep[fam] = "family"
    out = df[list(keep)].rename(columns=keep).copy()
    out = out.dropna(subset=["smiles", "iajd_num"])
    out["iajd_num"] = out["iajd_num"].astype(int)

    pf = physics_features()
    out["c0_nm_inv"] = out["iajd_num"].map(lambda n: pf.get(n, {}).get("c0_nm_inv"))
    out["c0_trusted"] = out["iajd_num"].map(lambda n: pf.get(n, {}).get("c0_trusted"))

    OUT.write_text(out.to_csv(index=False))
    print(f"wrote {OUT}  ({len(out)} IAJDs)")
    print(f"  columns: {list(out.columns)}")
    nfam = out["family"].nunique() if "family" in out else 0
    print(f"  families: {nfam} | with c0: {out['c0_nm_inv'].notna().sum()} | "
          f"flux_spleen non-null: {out['log10_flux_spleen'].notna().sum() if 'log10_flux_spleen' in out else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
