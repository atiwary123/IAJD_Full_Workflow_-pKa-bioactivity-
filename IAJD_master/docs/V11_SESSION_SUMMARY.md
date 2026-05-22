# v11 Session Summary — G1-Janus Dendrimer Expansion

## Quick Status

**v10 → v11 dataset improvement:**
- Rows: 351 → 360 (+9)
- Unique IAJDs: 254 → 263 (+9)
- **8 new G1-Janus dendrimers MALDI-validated** + integrated with full 3D conformer features
- **First LOO estimate for G1-Janus family**: MAE 0.6161 (n=8, default α=0.5)

## Architecture decoded

The G1-Janus dendrimers follow a **2-arm amphiphilic Janus dendrimer architecture** (Percec lab Generation-1 design):

```
Hydrophobic Ph (3,5-bis(R₁-O)(R₂-O), 1-CH₂-bridge)
  → -CH₂-O-CO-  (or -CH₂-NH-CO-) bridge to upper ring
  → Hydrophilic Ph (3,4,5-trisubstituted)
    - 3,5-positions: -O-(CH₂CH₂O)₃-CO-(CH₂)₃-N(piperidine)  [TEG-piperidinyl-butanoate]
    - 4-position:    -O-(CH₂CH₂O)₃-CH₂-C₆H₅                 [TEG-benzyl ether]
```

## ✓ 8 G1-Janus IAJDs MALDI-validated (Δ ≤ 0.05 Da each)

| IAJD | Architecture | Reported [M+H]⁺ | Computed | Δ (Da) |
|------|--------------|------------------|----------|--------|
| 110 | G1-J-C12/C12-ester | 1422.0 | 1421.96 | 0.04 |
| 111 | G1-J-C11/C11-ester | 1393.9 | 1393.92 | 0.02 |
| 131 | G1-J-C11/C18-ester | 1492.0 | 1492.03 | 0.03 |
| 155 | G1-J-C11/EH-ester | 1351.9 | 1351.88 | 0.02 |
| 156 | G1-J-C11/C14-ester | 1436.0 | 1435.97 | 0.03 |
| 157 | G1-J-C11/C15-ester | 1450.0 | 1449.99 | 0.01 |
| 158 | G1-J-C11/C17-ester | 1478.0 | 1478.02 | 0.02 |
| 159 | G1-J-C11/C17-amide | 1477.0 | 1477.03 | 0.03 |

## ⏸ Still deferred

**IAJD 33** — "SS family" lung-targeting compound with highest published flux (1.44×10⁸ total, 1.1×10⁹ lung). Synthesis source not located in available SI files (ja2c00273, pharm1572, ja1c05813, ja1c09585). Likely from an even earlier Percec paper. Exists in v11 as a row with no SMILES.

## 3D conformer ensemble

ETKDGv3 + MMFF94 with `useRandomCoords` (large molecules require this). Saved to `g1janus_3d.pkl`:

| IAJD | Rg (Å) | Asphericity | %V_Bur_max |
|------|--------|-------------|------------|
| 110 | 8.5 | ~0.5 | 76.5 |
| 111 | 8.4 | ~0.5 | 75 |
| 131 | 6.71 | — | 70.2 |
| 155 | 8.3 | — | — |
| 156 | 6.97 | — | 63.2 |
| 157 | 7.04 | — | 67.2 |
| 158 | 7.11 | — | 68.8 |
| 159 | 7.29 | — | 69.8 |

## v11 LOO baseline (α-routed, collapsed replicates)

| Subset | n | MAE |
|--------|---|-----|
| **v11 all (with G1-Janus)** | 195 | **0.4178** |
| v11 excluding G1-Janus | 187 | 0.4093 |
| v10 reference (no G1-Janus) | 187 | 0.3982 |
| **G1-Janus only** | **8** | **0.6161** |

### Per-family breakdown (v11 collapsed, α-routed)
| Family | n | α | MAE |
|--------|---|---|-----|
| sSS-Nonsym | 135 | 0.05 | 0.4041 |
| PE-Gallic | 23 | 0.85 | 0.6145 |
| PE-Tris | 14 | 0.0 | 0.2062 |
| GA-Tris | 10 | 1.0 | 0.3036 |
| **G1-Janus-Dendrimer** | **8** | **0.5** | **0.6161** |
| Dialkoxybenzyl | 5 | 1.0 | 0.3842 |

## Strategic value of this expansion

**Why include G1-Janus despite high MAE?**
1. **Coverage critical for lung-targeting predictions** — IAJDs 33, 78, 110, 111, 155-159 are the top lung-delivering compounds. Without them, the model literally cannot reason about lung architecture-flux relationships.
2. **Architecture diversity for OOD detection** — having even 8 G1-Janus compounds in the training set means OOD-flagging tests can distinguish "in-domain G1-Janus" from "novel architecture".
3. **Building toward larger n** — 8 compounds is the seed; future Percec papers may add more, and the model can incrementally improve.

## Files saved (in /home/claude/pharm1572_extract/)

### G1-Janus deliverables
- `g1janus_smiles_final.pkl` — 8 SMILES + arch + MALDI data
- `g1janus_3d.pkl` — 3D features for all 8

### v11 dataset
- `IAJD_Bioact_v11_clean.pkl/.xlsx` — 360 rows, 263 IAJDs, with G1-Janus integrated

### LOO outputs
- `v11_collapsed_y.npy`, `v11_collapsed_final.npy`, `v11_collapsed_families.npy`

## Next session pickup points

1. **Find IAJD 33 source paper** — likely an early Percec one-component paper not in current /mnt/project. Once found, add to v12.
2. **Build SMILES for v21 IAJDs 27, 28, 29, 31, 34, 46** — these are the **phenylacetate-Bn variants** of the G1-Janus core (different from the OBn variants in pharm1572). v21 already has SMILES for them; cross-check MALDI.
3. **Tune G1-Janus α** — with only 8 compounds, α=0.5 is a guess. Compare α ∈ {0.0, 0.5, 1.0} explicitly.
4. **Run full Stage A/B/C cascade on v11** with proper 3D + Gasteiger features.
5. **Stage B per-organ regressors** — now that we have lung-flux for 20 IAJDs (Fig 12) and total flux for 195, train per-organ models.
