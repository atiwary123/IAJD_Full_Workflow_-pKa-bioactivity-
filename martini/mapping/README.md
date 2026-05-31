# MARTINI 3 fragment mapping for IAJD self-assembly

This directory holds the MARTINI 3 bead assignments for every fragment in the
IAJD grammar (see `iajd_grammar.py`). `martini/build_cg.py` glues fragments
together to emit a per-molecule topology and a packed initial configuration.
`martini/run_selfassembly.py` then runs the EM → NPT → production protocol
through GROMACS, and `martini/analyze_md.py` extracts the observables that
feed `md_cache.csv`.

## Fragment classes

| Fragment | Bead types (MARTINI 3) | Notes |
| --- | --- | --- |
| Alkyl tail (C2–C18, branched & linear) | C1 / TC for terminal CH3, C1 every 4 carbons | length × 4-to-1 atomistic-to-CG mapping |
| 2-ethylhexyl branch | C1 + 1 SC1 branch bead | matches the 8-C branched tail in IAJD library |
| Ester linker `-OC(=O)-` | N4a + SP2 backbone bead | charge 0; H-bond acceptor |
| Amide linker `-NC(=O)-` | P1 backbone bead | stronger H-bond donor, used in PE-Gallic |
| Benzyl `-CH2-c1ccccc1` (aromatic core) | TC5 × 3 ring beads + SC1 for the benzyl CH2 | rigid ring constraints |
| Gallic / 3,4,5-trisubstituted phenyl | TC5 × 3 + 3 ether O beads (N4a) | shared between GA-Tris/PE-Tris/PE-Gallic |
| Piperazine (`N1CCNCC1`) | N6d (proximal N) + N5a (distal N) + SC2 backbone | distal-N protonation toggles N5a → Q1 (+1) |
| 4-methylpiperazine (MPRZ) | piperazine + TC1 methyl | distal N is the most-basic site |
| 4-hydroxyethyl piperazine (HPRZ) | piperazine + SP1 + SP1 (CH2-CH2-OH) | H-bond donor |
| H2EPRZ (`-CH2CH2OCH2CH2OH`) | piperazine + N4a + SP1 + SP1 | most flexible head |
| Piperidine (PIP) | N6d + SC2 ring | no distal-N |
| DMA (`N(CH3)(CH3)`) | N6d (proximal N) + 2× TC1 methyls | small head |
| DMBA (`Cc1ccc(N(C)C)cc1`) | benzyl ring (TC5×3) + N6d + 2× TC1 | PE-Gallic head |

## Protonation state encoding

Each piperazine and tertiary-amine head fragment ships **two** mapping files:

- `*_neutral.map` — distal-N is N5a (neutral, uncharged).
- `*_prot.map`    — distal-N is Q1 (+1 charge), accompanied by a counter-ion
  Q1 (Cl−) in the bulk to neutralize the total system.

`build_cg.py` selects the right file based on a `protonated: bool` flag and
emits a `*_neutral.itp` / `*_prot.itp` pair per molecule.

## Bead parameter sources

- MARTINI 3 force-field files: `martini_v3.0.0.itp` (downloaded separately into
  this directory once the offline conda env is built — see
  `requirements-offline.txt`).
- Ion parameters: `martini_v3.0.0_ions.itp`.
- Water bead: regular W (4-to-1 H2O mapping); enable polarizable variant `WP`
  when running protonated systems for better Coulombic screening.

## Honest scope note

The fragment vocabulary is bounded (5 families × ~6 heads × ~10 tails). Each
fragment is mapped *once*; molecules are then assembled by concatenating the
per-fragment topologies. This is the only way MARTINI is tractable here —
mapping each whole IAJD individually would be infeasible.

Convergence of the self-assembly run is **per-molecule** and reported in
`md_audit.json`; replicate disagreement is logged and propagated as honest
low-confidence (`md_assembles < 1`) rather than imputed.
