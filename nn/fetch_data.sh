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
# LNPDB is REQUIRED (the in-vivo corpus); AGILE/LANTERN are only for the GNN-encoder /
# cleaned-label UPGRADES, so their clone failing must NOT abort the LNPDB build.
# (Also: nn/lnpdb_invivo.csv is already committed in the repo, so the Morgan transfer path
#  runs even without this script.)
clone https://github.com/evancollins1/LNPDB.git   LNPDB
clone https://github.com/bowang-lab/AGILE.git      AGILE   || echo "[warn] AGILE clone failed (GNN-encoder upgrade only)"
clone https://github.com/AsalMehradfar/LANTERN.git LANTERN || echo "[warn] LANTERN clone failed (cleaned-labels upgrade only)"
# AGILE's pretrained checkpoint (model.pth) is git-LFS; pull it so agile_embed.py can load it.
[ -d AGILE ] && ( cd AGILE && git lfs install --local 2>/dev/null
  git lfs pull 2>&1 | tail -1 ) || echo "[warn] AGILE LFS pull skipped (checkpoint may be a pointer)"

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
# the in-vivo FLAG is detected BY VALUE (the real LNPDB schema has a `Model` column with
# in_vitro/in_vivo; `Model_type` is cell lines — name-based detection picks the wrong one).
flag = None
for c in df.columns:
    v = set(df[c].astype(str).str.lower().unique())
    if 'in_vivo' in v and 'in_vitro' in v:
        flag = c; break
sm  = find(df.columns, 'SMILES', avoid=('PROT', 'HEAD', 'LINKER', 'TAIL', 'HL_', 'CHL', 'PEG')) or 'IL_SMILES'
val = find(df.columns, 'EXPERIMENT', 'VALUE') or find(df.columns, 'VALUE') or find(df.columns, 'EFFICACY')
org = find(df.columns, 'MODEL', 'TARGET') or find(df.columns, 'TARGET') or find(df.columns, 'ORGAN')
sub = df[df[flag].astype(str).str.lower() == 'in_vivo'] if flag else df
print(f"  in-vivo flag column = {flag}; SMILES={sm}; value={val}; organ={org}")
cols = {sm: 'smiles', val: 'value'}
if org: cols[org] = 'organ'
out = sub[list(cols)].rename(columns=cols).dropna(subset=['smiles', 'value'])
dest = os.path.join(repo, 'nn', 'lnpdb_invivo.csv')
out.to_csv(dest, index=False)
print(f"wrote {dest}: {len(out)} in-vivo rows; columns {list(out.columns)}")
if 'organ' in out: print("  organs:", out['organ'].value_counts().head(8).to_dict())
PY
echo "DONE. Now: python $REPO/nn/train_transfer.py --lnpdb $REPO/nn/lnpdb_invivo.csv --target log10_flux_spleen"
