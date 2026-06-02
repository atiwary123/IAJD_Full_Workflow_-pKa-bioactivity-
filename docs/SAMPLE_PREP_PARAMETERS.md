# Sample-Preparation Parameters for the IAJD datasets

**Added:** 2026-06-02 · **Builder:** `build_sample_prep_features.py` · **Audit JSON:** `sample_prep_protocols.json`

## Why
The IAJD bioactivity (flux) and pKa labels were measured across ~9 source papers whose
DNP (dendrimersome-nanoparticle) **formulation** and **in-vivo assay** conditions are not
guaranteed identical. Unmodeled prep variation (assembly pH, IAJD:mRNA ratio, imaging time,
dialysis, measurement method, one-component-vs-multi-component, batch/era) shows up as label
noise. This adds a self-contained, **provenance-tagged**, quantitative `sp_*` block extracted
from the source papers so models can either use prep as features or down-weight low-provenance rows.

Every value is either **reported** in a source paper, **derived** from a paper's single stated
protocol, or explicitly flagged as **inferred** — no fabricated numbers (see provenance below).

## Where it was added (live datasets only)
- `IAJD_master/datasets/IAJD_Bioact_v13_clean.AUDIT_FIXED.xlsx`  (+31 cols, 273 rows)
- `IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx`  (canonical; == AUDIT_FIXED)
- `IAJD_master/datasets/IAJD_pKa_v21_final.AUDIT_FIXED.xlsx`  (+31 cols, 278 rows)
- `IAJD_master/datasets/IAJD_pKa_v21_final.xlsx`  (canonical; == AUDIT_FIXED)
- `nn/iajd_transfer_input.csv`  (+31 cols, merged by IAJD number; 273/273 matched)

Pre-edit copies saved as `*.PRE_SAMPLEPREP.*`. `*.PRE_AUDIT/PRE_RECON/PRE_RECOMPUTE` backups and
`workflow_snapshot_*` were intentionally **not** modified. No existing column was changed or dropped;
the new block is namespaced `sp_` so it never collides with legacy columns
(`buffer`, `pH_sample`, `inj_route`, `mice_strain`, `dose_mRNA_ug`, `T_hours`, `DNP_size_nm`, …).

## The parameters

| column | type | meaning | values in corpus | varies? |
|---|---|---|---|---|
| `sp_prep_method` / `sp_prep_method_code` | str/int | assembly method | ethanol_injection (1) | constant |
| `sp_n_components` | int | # formulation components | 1 (one-component IAJD; no helper lipid/chol/PEG) | constant |
| `sp_assembly_buffer` | str | injection buffer | acetate (citrate for ja1c05813 IAJD 9) | rare |
| `sp_assembly_buffer_mM` | float | buffer conc | 10 | constant |
| **`sp_assembly_pH`** | float | **nominal assembly buffer pH** | **4.0 / 5.2 (3.0 for IAJD 9)** | **YES** |
| `sp_IAJD_conc_final_mg_mL` | float | final IAJD conc | 4.0 | constant |
| `sp_mRNA_conc_final_mg_mL` | float | final mRNA conc | 0.10 (0.025 for IAJD 9) | rare |
| **`sp_IAJD_mRNA_mass_ratio`** | float | **C_IAJD / C_mRNA (mass)** | **40 (160 for IAJD 9)** | rare |
| `sp_ethanol_vol_frac` | float | EtOH v/v at injection | 0.0500 (= 25/500.5 µL) | constant |
| `sp_vortex_time_s` | float | vortex time | 5 (ja3c13569 also tested 20) | constant |
| `sp_assembly_temp_C` | float | mixing temp | 23 — **INFERRED** (room temp; never stated for the mixing step) | constant |
| `sp_dialyzed` | int | post-assembly dialysis | 0 (1 for ja1c05813 IAJD 9) | rare |
| `sp_cargo` / `sp_cargo_code` | str/int | mRNA cargo | Luc-mRNA (1) | constant |
| `sp_mRNA_polyA_nt` | float | poly(A) length | 101 (stated 2023+; same Weissman construct) | constant |
| `sp_assay_system` / `sp_assay_system_code` | str/int | assay | in_vivo (1) | constant* |
| `sp_inj_route` / `sp_inj_route_iv` | str/int | injection route | retro-orbital, IV=1 | constant |
| `sp_dose_mRNA_ug` | float | mRNA dose | 10 | constant |
| `sp_inj_volume_uL` | float | injection volume | 100 | constant |
| `sp_mouse_strain` | str | mouse strain | BALB/c | constant |
| **`sp_imaging_time_h`** | float | hours injection→imaging | **4–7 (=T_hours where present, else paper default)** | **YES** |
| `sp_luciferin_mg_kg` | float | D-luciferin dose | 150 | constant |
| `sp_pKa_method` / `sp_pKa_method_code` | str/int | IAJD pKa method | ethanolic HCl titration, half-equiv (1); TNS available for ja5c07232 (2) | constant |
| `sp_pKa_IAJD_conc_mg_mL` | float | pKa titration conc | 1.5 | constant |
| **`sp_provenance`** | str | how the row's prep was sourced | reported_per_row / paper_protocol / inferred_group_standard / unknown_paper | **YES** |
| **`sp_confidence`** | float | provenance weight | 1.00 / 0.85 / 0.65 / 0.35 / 0.25 | **YES** |
| `sp_notes` | str | per-paper caveats | — | per-paper |

\* `sp_assay_system` is constant (`in_vivo`) here because every flux label in these datasets is an
in-vivo organ/total IVIS measurement; it would split if in-vitro HEK293T rows were ever added.

## Provenance / confidence (the honesty layer)
- **reported_per_row (1.00)** — read per-compound from a paper data table. Only `ja1c05813`
  (Table S7 gives per-IAJD buffer/pH/conc/size/PDI/time). 48 Bioact rows / 52 pKa rows.
- **paper_protocol (0.85)** — paper states ONE Methods protocol applied to all its IAJDs; confirmed
  by reading each SI. `pharmaceutics1501572, ja2c00273, ja3c07337, ja3c13569, ja1c09585, ja5c07232`.
- **inferred_group_standard (0.65 / 0.35)** — `bm4c01599` (0.65; synthesis-only SI, formulation
  inferred but main paper carries in-vivo metadata) and `novel_2026` (0.35; GA-Tris-345-EH compounds
  347–373 absent from every cited paper — reconstructed from a CDXML + flux sheet, standard inferred).
- **unknown_paper (0.25)** — pKa rows whose source paper is unidentifiable (23 rows): only universal
  facts filled (ethanol injection, one-component, Luc-mRNA, pKa titration); paper-specific values null.

**Suggested model use:** pass `sp_confidence` as a sample weight (or drop `<0.5`), and use
`sp_assembly_pH` + `sp_imaging_time_h` as covariates.

## Source-paper protocol summary (extracted this session from SI Methods + data tables)
Standard one-component Percec IAJD protocol (confirmed across 7 papers): rapid **ethanol injection**;
IAJD 80 mg/mL in EtOH; Luc-mRNA 4.0 mg/mL in water; **12.5 µL mRNA + 463 µL acetate (10 mM, pH 4.0)
+ 25 µL IAJD-EtOH; vortex 5 s** → final C_IAJD = 4.0, C_mRNA = 0.10 mg/mL, **IAJD:mRNA = 40:1**,
EtOH 5.0% v/v, 50 µg mRNA/batch. In vivo: **retro-orbital (IV), 100 µL / 10 µg**, BALB/c 6–8 wk,
IVIS Spectrum, i.p. D-luciferin 150 mg/kg, image ~4 h. pKa: half-equivalence titration, 1.5 mg/mL
in NaCl-sat ethanol, 0.1 M HCl, 7.5 µL increments.

Deviations honored (see `sp_notes`): **ja1c05813** screens assembly pH (acetate 4.0/5.2; IAJD 9 =
citrate 3.0 + PBS dialysis, 160:1); **ja3c13569** vortex 5 *or* 20 s, pH rises 4.0→~4.8 post-injection;
**ja5c07232** adds a TNS DNP-surface pKa and a lymph-node organ; **ja1c09585** is an early paper with
in-vitro HEK293T data and SI imaging only 4–6 h (some dataset `T_hours` up to 10.25 are **not**
supported by the SI). `s41467-024-50619` (AGILE) is a **multi-component LNP / intramuscular / robot-mix**
paper — *not* an IAJD source; it does not apply to any IAJD row.

## Honest assessment — how much noise can this actually explain?
The Percec IAJD corpus is **remarkably standardized**: in the bioactivity table only **3** of the new
columns genuinely vary — `sp_assembly_pH` (4.0 vs 5.2), `sp_imaging_time_h` (4–7 h), and
`sp_confidence`. Everything else is a protocol constant (it documents conditions and would gain
variance only if non-Percec or in-vitro data were added).

`sp_assembly_pH` **does carry real signal** — pH-5.2-assembled DNPs are larger and far less active:

| sp_assembly_pH | n | median DNP size | median PDI | median log10 flux_total |
|---|---|---|---|---|
| 4.0 | 223 | 150 nm | 0.287 | **7.71** |
| 5.2 | 39 | 171 nm | 0.323 | **6.20** |

A ~1.5-log flux drop, consistent with the ja1c05813 finding that lower assembly pH → smaller, more
active DNPs. **Caveat (confound):** every pH-5.2 row is from `ja1c05813`, and the pH-4.0 subset within
that paper were partly the better-performing compounds re-run at pH 4.0 — so `sp_assembly_pH` is
partly confounded with paper identity and compound selection. Treat it as informative but not a clean
causal lever inside this corpus.

**Other noise sources worth a look (not "sample prep" per se):** several `pharmaceutics1501572` rows
have `n_mice = 1` (high per-row measurement variance); `ja1c09585` `T_hours` values >7 h appear to be
data-entry/averaging artifacts (SI max = 6 h); replicate counts (`n_replicates`, `n_mice`) vary widely
and would make a better variance/weighting signal than most prep constants.

## Reproduce
```bash
python build_sample_prep_features.py --dry-run   # compute + verify, no writes
python build_sample_prep_features.py             # apply to the 4 xlsx + NN csv, write audit JSON
```
Idempotent: re-running drops and rebuilds the `sp_` block. To roll back, restore `*.PRE_SAMPLEPREP.*`.
