#!/usr/bin/env bash
# nn/fetch_data.sh — fetch the transfer-learning corpora and build the LNPDB in-vivo CSV.
# Run on the pod (needs internet). MIT-licensed sources (verified 2026-06-01):
#   LNPDB   github.com/evancollins1/LNPDB        12,845 ionizable lipids, 2,388 in-vivo organ rows
#   AGILE   github.com/bowang-lab/AGILE          60k-lipid MolCLR pretrained encoder (model.pth)
#   LANTERN github.com/AsalMehradfar/LANTERN     cleaned 1,100-lipid in-vitro labels
set -euo pipefail
WORK="${WORK:-/workspace}"
REPO="${REPO:-$WORK/IAJD}"
cd "$WORK"; mkdir -p extern; cd extern

clone() { [ -d "$2" ] || git clone --depth 1 "$1" "$2"; }
clone https://github.com/evancollins1/LNPDB.git   LNPDB
clone https://github.com/bowang-lab/AGILE.git      AGILE
clone https://github.com/AsalMehradfar/LANTERN.git LANTERN

# Build the in-vivo subset (smiles, value, organ) — schema-robust column detection.
python - "$REPO" <<'PY'
import pandas as pd, glob, sys, os
repo = sys.argv[1]
cands = sorted(glob.glob('LNPDB/**/LNPDB*.csv', recursive=True), key=os.path.getsize, reverse=True)
assert cands, "LNPDB csv not found"
df = pd.read_csv(cands[0], low_memory=False)
def find(cols, *keys, avoid=()):
    for c in cols:
        u = c.upper()
        if all(k.upper() in u for k in keys) and not any(a.upper() in u for a in avoid):
            return c
    return None
sm  = find(df.columns, 'SMILES', avoid=('PROT',)) or find(df.columns, 'SMILES')
mt  = find(df.columns, 'MODEL', 'TYPE') or find(df.columns, 'IN_VIVO')
val = find(df.columns, 'VALUE') or find(df.columns, 'EFFICACY') or find(df.columns, 'EXPERIMENT', 'VALUE')
org = find(df.columns, 'TARGET') or find(df.columns, 'ORGAN')
sub = df[df[mt].astype(str).str.contains('vivo', case=False, na=False)] if mt else df
cols = {sm: 'smiles', val: 'value'}
if org: cols[org] = 'organ'
out = sub[list(cols)].rename(columns=cols).dropna(subset=['smiles', 'value'])
dest = os.path.join(repo, 'nn', 'lnpdb_invivo.csv')
out.to_csv(dest, index=False)
print(f"wrote {dest}: {len(out)} in-vivo rows; columns {list(out.columns)}")
if 'organ' in out: print("  organs:", out['organ'].value_counts().head(8).to_dict())
PY
echo "DONE. Now: python $REPO/nn/train_transfer.py --lnpdb $REPO/nn/lnpdb_invivo.csv --target log10_flux_spleen"
