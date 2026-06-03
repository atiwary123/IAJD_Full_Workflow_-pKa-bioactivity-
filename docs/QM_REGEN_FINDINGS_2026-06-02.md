# QM Regeneration — Findings & Anything Interesting (2026-06-02/03)

Real xTB QM (8 descriptors) was recomputed for the **corrected** IAJD dataset and merged into the
datasets. This is the "anything interesting" report. Companion to `docs/DATASET_REGEN_EXECUTION_2026-06-01.md`.
**Be skeptical-honest:** several family signals are small-n; correlation ≠ causation.

## Status
- **QM complete: 49 corrected compounds computed, every non-flagged training SMILES now has real QM.**
  Coverage after append: **bioact 273/273 (100%)**, **pKa 262/278 (94%)** — the 16 pKa blanks are still-flagged
  (UNRESOLVED) rows, deliberately left blank (never proxied).
- QM descriptors appended to all 4 datasets (`IAJD_Bioact_v13_clean{,.AUDIT_FIXED}.xlsx`,
  `IAJD_pKa_v21_final{,.AUDIT_FIXED}.xlsx`) as `qm_*` columns + a `qm_source` flag.
- Full bioact stack re-finalized with **real QM in Block D'** (0% emulator-filled, was 65%).

## Headline finding — the SMILES audit changed *shape & solvation*, not charge
Comparing each corrected compound's QM to its **pre-audit (wrong-SMILES)** QM (n=43–48):

| QM descriptor | mean abs change | max | what it is |
|---|---|---|---|
| q_ionizableN | **0.037** | 0.13 | charge on the protonatable N |
| dipole_D | **2.99 D** | 10.0 | molecular dipole |
| polarizability | **19.2** | 87.6 | electronic polarizability (~size) |
| ΔGsolv (total) | **14.4 kJ/mol** | 40.3 | solvation free energy |
| ΔGsolv_tail | **13.2 kJ/mol** | 74.9 | tail solvation |
| ΔGsolv_head | 12.5 kJ/mol | 51.3 | head solvation |
| Ehedup | **31.4** | 90.4 | head-dehydration term |

**Intuition:** the audit corrected the molecules' *periphery* (OH→OMe caps, phenylacetate→OBn ethers,
twin/dendron rebuilds) — which moved size/polarity/solvation a lot — but the **protonation-site charge
barely moved (0.037)**. So the old wrong-SMILES models had materially wrong "size/greasiness/solvation"
features but ~correct charge. That's precisely why the membrane/solvation-driven predictions needed
retraining, and why the pKa *intrinsic* charge term was relatively robust.

## Per-family QM ↔ property signal (the interesting trends)
Pooled QM↔target correlations are weak (flux: polarizability −0.22; pKa: ΔGsolv_head −0.17) because a
large, QM-flat family (sSS-Nonsym) dilutes them. Stratified by family (Spearman, |ρ|≥0.4):

**pKa** (QM's home turf — strongest signal):
- **PE-Gallic: ΔGsolv_head ↔ pKa = +0.76 (n=52)** — well-powered, the standout. Head solvation tracks pKa.
- **GA-Tris: ΔGsolv_tail ↔ pKa = −0.73 (n=12)** — stable across every check tonight.
- G1-Janus: polarizability −0.65, ΔGsolv_tail +0.61, dipole +0.55 (n=8, small).
- Dialkoxybenzyl: HOMO-LUMO +0.58, q_ionizableN −0.55, ΔGsolv +0.51 (n=12).

**flux (bioactivity)** — weaker, fewer (flux is a noisy biological endpoint):
- **PE-Tris: ΔGsolv ↔ flux = −0.41 (n=41)** — best-powered flux signal.
- **G1-Janus: polarizability +0.55, ΔGsolv_tail +0.45 (n=21).**
- Dialkoxybenzyl: Ehedup −0.64 (n=10).
- **sSS-Nonsym (n=144): nothing ≥0.4** — QM doesn't explain its flux (matches the known "sSS levers are flat" SAR).

**Two recurring themes:** (1) **ΔGsolv is the workhorse** — it's the descriptor that keeps appearing for
both pKa and flux (desolvation governs both protonation and membrane partitioning). (2) **pKa is far more
QM-explainable than bioactivity** (more families, stronger ρ).

**Caveats:** the strong G1-Janus/GA-Tris/Dialkoxybenzyl hits are small-n (8–12); across ~96
family×descriptor×target tests a couple are chance. Bank on: **PE-Gallic ΔGsolv_head↔pKa (n=52)**,
**PE-Tris ΔGsolv↔flux (n=41)**, **GA-Tris ΔGsolv_tail↔pKa (n=12, consistent)**.

## Model metrics — honest read
| Model | before | after (corrected + real QM) |
|---|---|---|
| bioact v14 LOO MAE | 0.411 | **0.431** (n=247) |
| pKa v92 blend LOO MAE | 0.118 (n=255) | **0.126** (n=278) |
| v15 hybrid blend LOO MAE | — | 0.259 (ml=1.0, physics=0.0) |

**The pooled MAEs rose slightly — and that's expected, not a regression.** The training sets *grew*
(the parallel reconstruction un-flagged previously-excluded **G1-Janus** and other hard/novel compounds),
so the models are now scored on a harder set. Per-family (v15): GA-Tris 0.15, Dialkoxybenzyl 0.18, PE-Tris
0.22, sSS 0.28, PE-Gallic 0.29, **G1-Janus 0.34** — the reconstructed G1-Janus family is the hard one
pulling the average up. The real win is **correctness** (right structures + real QM), not a metric drop.
v15's blend put **0 weight on the physics head** (ml=1.0) — the Block-D' physics is informative per-family
but doesn't beat the descriptor/LiON/ADMET stack on pooled LOO; it earns its keep on novelty extrapolation,
not in-distribution accuracy.

## Failures & the janitor bug (the one real gotcha)
3 `FileNotFoundError` losses, all the **same** mechanism: `qm_scratch_janitor.sh` deleted **live** scratch.
Its old check (`-maxdepth 1 -mmin +30` on the parent dir) flagged a dir "idle >30 min" whenever a single
xtb call on a huge twin/G1-Janus ran >30 min writing only to *subdirs* — so it deleted scratch mid-run.
**Fixed** (commit `ed23a8e`): recursive 60-min-idle check (delete only if *nothing inside* changed in 60 min).
After the fix the thrice-failed PE-Gallic twin completed cleanly. Net QM data lost: **0** (all retried).

## Operational story
- **Pace:** small PE-Gallic single-singles ~10 min each; ~200-atom twin-twins + G1-Janus dendrimers ~1–2 h
  each on the fanless M-series CPU → the run spanned ~2 days, not "overnight."
- **Power:** the laptop repeatedly dropped to battery (`caffeinate -s` is ignored on battery), causing system
  sleeps that *froze* compute (and triggered the first janitor race). No data lost (per-compound checkpointing
  + hourly GitHub autosave), but it stalled the run; AC was required to finish.
- **Moving target:** corrections continued in parallel (G1-Janus reconstructions un-flagged after launch); the
  `--resume` sweeps picked them up, so the final QM covers them too.
- **Saving:** per-compound `--checkpoint-every 1` + the autosave loop pushed `qm_cache` to GitHub throughout;
  nothing computed was ever lost.

## Follow-ups
- A few pKa rows (16) remain QM-blank (still UNRESOLVED SMILES) — they get real QM once their structures are resolved.
- The Block-D' physics head adds little to *in-distribution* pooled accuracy; its value is OOD extrapolation +
  per-family mechanism (ΔGsolv), worth keeping but not over-weighting.
