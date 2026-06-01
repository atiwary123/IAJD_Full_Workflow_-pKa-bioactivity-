"""regen_caches.py — repopulate LiON / ADMET caches for the CORRECTED dataset.

Computes real predictions (local envs via extend_caches.py) for every non-flagged
bioactivity canonical SMILES, so the v14 pipeline finds a real cache entry for
each training compound (no proxy fallback). extend_caches auto-skips already-cached
SMILES, so only the corrected/new ones are actually computed.

Usage:  python audit_work/regen_caches.py {admet|lion|both}
Key = canonical SMILES (matches bioact_v14_pipeline lookup).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "IAJD_master" / "code"))
BIOACT = ROOT / "IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx"
CACHES = ROOT / "IAJD_master/bundles_caches"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def training_smiles():
    df = pd.read_excel(BIOACT, sheet_name=0)
    if "audit_status" in df.columns:
        flag = df["audit_status"].astype(str).str.contains("UNRESOLVED|FLAG", na=False)
        df = df[~flag]
    df = df[df["log10_flux_total"].notna()]
    smis = []
    for s in df["SMILES_canonical"]:
        c = canon(s)
        if c:
            smis.append(c)
    # unique, preserve order
    seen = set(); out = []
    for s in smis:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def coverage(cache_path, smis):
    if not Path(cache_path).exists():
        return 0, len(smis)
    cache = json.load(open(cache_path))
    have = sum(1 for s in smis if s in cache)
    return have, len(smis) - have


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    smis = training_smiles()
    print(f"Non-flagged bioact training canonical SMILES: {len(smis)} unique")

    if which in ("admet", "both"):
        from extend_caches import predict_admet_for_smiles
        ap = CACHES / "admet_cache_v13.json"
        have, miss = coverage(ap, smis)
        print(f"\n[ADMET] before: {have} cached, {miss} missing")
        if miss:
            predict_admet_for_smiles(smis, cache_path=str(ap), verbose=True)
        have2, miss2 = coverage(ap, smis)
        print(f"[ADMET] after:  {have2} cached, {miss2} missing")

    if which in ("lion", "both"):
        from extend_caches import predict_lion_for_smiles
        lp = CACHES / "lion_cache_v13.json"
        have, miss = coverage(lp, smis)
        print(f"\n[LiON] before: {have} cached, {miss} missing")
        if miss:
            predict_lion_for_smiles(smis, cache_path=str(lp), verbose=True)
        have2, miss2 = coverage(lp, smis)
        print(f"[LiON] after:  {have2} cached, {miss2} missing")


if __name__ == "__main__":
    main()
