"""
physics_features.py — physics-grounded descriptors for IAJD self-assembly
and endosomal-delivery prediction. These are NOT learned from training data;
they're computed from molecular structure + first-principles theory. That
means they can extrapolate outside the training range, in contrast to the
ML predictor's piecewise-constant trees.

Each function takes either an RDKit Mol or a SMILES and returns a single
floating-point value or a dict of values. Failures return NaN.

Categories:
  • Geometry-based (CPP, d-spacing, V_tail, l_tail, a_head)
  • Thermodynamic (membrane partition Kp, solvation ΔG, CMC)
  • Electrostatic (protonation fraction, Manning condensation)
  • Mechanical (bending modulus, splay/tilt moduli)
  • Composite (endosomal escape, fusogenicity)

Most expressions are simplified analytical forms with parameters fit to the
training set's structure-property correlations. Where a parameter is
domain-empirical we cite the source.
"""
from __future__ import annotations
import math
from typing import Optional, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP

# ──────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────
R = 8.314e-3   # kJ/mol/K
T_KELVIN = 310.15   # body temperature (37 °C)
RT = R * T_KELVIN   # ≈ 2.58 kJ/mol
KB = 1.380649e-23   # J/K
N_A = 6.022e23      # /mol
NM_PER_A = 0.1       # angstrom → nm

# Bilayer / membrane reference values (lipid bilayer, body T)
BILAYER_THICKNESS_NM = 4.0
HEAD_AREA_REF_NM2 = 0.70    # typical PC head area
TAIL_VOLUME_REF_NM3 = 0.95  # typical 14-C saturated chain volume

# Tunable scaling from training-set calibration (set heuristically; tuned in
# train_physics_predictor.py against measured log10_flux residuals)
KP_SCALE = 0.43        # multiplier on logP → ΔG_transfer kJ/mol
ESCAPE_GAIN = 1.0
BENDING_GAIN = 1.0


# ──────────────────────────────────────────────────────────────────────
# 1. CPP (critical packing parameter) — geometry-driven
# ──────────────────────────────────────────────────────────────────────

def cpp_geometric(v_tail_nm3: float, l_tail_nm: float, a_head_nm2: float) -> float:
    """CPP = V_tail / (a_head · l_tail).
    < 1/3 → spherical micelle, ~ 1/3..1/2 → cylindrical, ~ 1/2..1 → bilayer/lamellar,
    > 1 → inverted hexagonal.  For fusogenic IAJD-DNP we want CPP ~ 0.9-1.1.
    """
    if a_head_nm2 <= 0 or l_tail_nm <= 0:
        return float("nan")
    return v_tail_nm3 / (a_head_nm2 * l_tail_nm)


# ──────────────────────────────────────────────────────────────────────
# 2. Lamellar d-spacing (extends Peterca's empirical formula)
# ──────────────────────────────────────────────────────────────────────

def lamellar_d_spacing_nm(linker_length: int, chain_avg_carbons: float,
                           head_size_nm: float = 0.5) -> float:
    """Estimated lamellar d-spacing for an IAJD bilayer.

    Peterca's empirical formula (J. Am. Chem. Soc. 2008): d ≈ 0.13·linker
    + 0.125·chain_avg, in nm. Extended with explicit head contribution.
    """
    if linker_length is None or chain_avg_carbons is None:
        return float("nan")
    return 0.13 * float(linker_length) + 0.125 * float(chain_avg_carbons) + head_size_nm


# ──────────────────────────────────────────────────────────────────────
# 3. Membrane partition coefficient (Kp)
# ──────────────────────────────────────────────────────────────────────

def membrane_partition_logKp(mol: Chem.Mol) -> float:
    """Bilayer partition coefficient via Walter-Gulati: log Kp ≈ α·logP − β·TPSA + γ.
    Approximates the free-energy of inserting the IAJD into the lipid bilayer.
    Higher logKp → stronger membrane affinity → better endosomal interaction.
    """
    try:
        logp = MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        # Walter-Gulati-like fit; α=0.85, β=0.018, γ=−0.4 calibrated on
        # typical drug-like compounds. For very lipophilic IAJDs this gives
        # logKp in the 5-10 range (membrane-bound regime).
        return 0.85 * logp - 0.018 * tpsa - 0.4
    except Exception:
        return float("nan")


def membrane_dG_transfer_kJmol(mol: Chem.Mol) -> float:
    """ΔG_transfer (water → bilayer) = −RT · ln(10) · log Kp.

    Negative values mean transfer to bilayer is favorable (membrane-affine).
    """
    logKp = membrane_partition_logKp(mol)
    if not np.isfinite(logKp):
        return float("nan")
    return -RT * math.log(10) * logKp


# ──────────────────────────────────────────────────────────────────────
# 4. Endosomal-escape proxy
# ──────────────────────────────────────────────────────────────────────

def henderson_hasselbalch_fraction(pka: float, ph: float = 5.5) -> float:
    """Protonation fraction at given pH. ph=5.5 = early endosome."""
    if not np.isfinite(pka):
        return float("nan")
    return 1.0 / (1.0 + 10 ** (ph - pka))


def endosomal_escape_score(pka: float, cpp: float,
                            ph_endosome: float = 5.5, ph_cytosol: float = 7.4) -> float:
    """Composite escape score.

    The "proton sponge" / membrane-disruption picture:
      1. Endosomal acidification protonates the head (Δprotonation).
      2. Protonation increases head area → CPP shifts AWAY from 1 (lamellar)
         toward 0.5 (micellar) or > 1 (inverted hexagonal), destabilizing the
         endosomal membrane.
      3. |CPP - 1| × Δprotonation approximates fusogenic potential.
    """
    f_endo = henderson_hasselbalch_fraction(pka, ph_endosome)
    f_cyto = henderson_hasselbalch_fraction(pka, ph_cytosol)
    if not (np.isfinite(f_endo) and np.isfinite(f_cyto) and np.isfinite(cpp)):
        return float("nan")
    delta_protonation = f_endo - f_cyto       # 0 to 1
    cpp_departure = abs(cpp - 1.0)            # destabilization magnitude
    return ESCAPE_GAIN * delta_protonation * cpp_departure


# ──────────────────────────────────────────────────────────────────────
# 5. Bending / tilt modulus proxy (Helfrich theory)
# ──────────────────────────────────────────────────────────────────────

def bending_modulus_kBT(bilayer_thickness_nm: float = BILAYER_THICKNESS_NM,
                        area_per_lipid_nm2: float = HEAD_AREA_REF_NM2) -> float:
    """Helfrich bending modulus, in units of k_B T.

    κ_b ≈ k_b · t² / a   (Evans-Skalak-type)
    where t is bilayer thickness and a is area per lipid.
    For fusogenic IAJDs we want LOW κ_b (more deformable membrane → easier fusion).
    """
    if bilayer_thickness_nm <= 0 or area_per_lipid_nm2 <= 0:
        return float("nan")
    return (bilayer_thickness_nm ** 2) / area_per_lipid_nm2


def splay_modulus_kBT(l_tail_nm: float, n_tail_chains: int = 2) -> float:
    """Splay modulus K_3 ≈ k_b · n_chains / l_tail."""
    if l_tail_nm <= 0:
        return float("nan")
    return float(n_tail_chains) / l_tail_nm


# ──────────────────────────────────────────────────────────────────────
# 6. Manning counterion condensation
# ──────────────────────────────────────────────────────────────────────

def manning_fraction(charge_density: float, dielectric: float = 80.0) -> float:
    """Fraction of fixed charge neutralized by counterion condensation.

    For a polyelectrolyte: ξ = (Bjerrum_length / spacing) > 1 → condensation.
    For IAJDs binding mRNA, higher Manning fraction = stronger charge
    neutralization = better mRNA condensation.

    charge_density is in elementary charges per nm.
    """
    bjerrum_nm = 0.7 / dielectric * 56.0   # ≈ 0.7 nm at ε=80
    if charge_density <= 0:
        return 0.0
    xi = bjerrum_nm * charge_density
    if xi >= 1.0:
        return 1.0 - 1.0 / xi
    return 0.0


# ──────────────────────────────────────────────────────────────────────
# 7. Critical micelle concentration (CMC)
# ──────────────────────────────────────────────────────────────────────

def cmc_log_molar(n_tail_carbons: float, head_hydrophilicity: float = 1.0) -> float:
    """Klevens-type CMC formula: log10(CMC) = a − b · n_tail_carbons.

    Calibrated typical values: a=1.7, b=0.30 for monovalent surfactants.
    head_hydrophilicity scales the intercept (more hydrophilic head → higher CMC).
    Lower log CMC means stronger self-assembly tendency.
    """
    if n_tail_carbons is None or not np.isfinite(n_tail_carbons):
        return float("nan")
    return (1.7 + 0.4 * (head_hydrophilicity - 1.0)) - 0.30 * float(n_tail_carbons)


# ──────────────────────────────────────────────────────────────────────
# 8. Hydrophobic-hydrophilic balance (HLB)
# ──────────────────────────────────────────────────────────────────────

def hlb_griffin(mol: Chem.Mol) -> float:
    """Griffin HLB ≈ 20 · M_hydrophilic / M_total.

    HLB 0-3 = anti-foaming/water-in-oil, 4-6 = w/o emulsifier, 7-9 = wetting,
    10-13 = oil-in-water, 13-18 = solubilizer.  Fusogenic IAJDs typically
    score 6-11.
    """
    try:
        total_mw = Descriptors.ExactMolWt(mol)
        # Hydrophilic mass: heteroatoms (O, N) plus their nearest H
        hydro_mass = 0.0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in (7, 8):
                hydro_mass += atom.GetMass()
                hydro_mass += atom.GetTotalNumHs() * 1.008
        if total_mw <= 0:
            return float("nan")
        return 20.0 * hydro_mass / total_mw
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────
# 9. Composite all-features wrapper
# ──────────────────────────────────────────────────────────────────────

def compute_all_physics_features(mol_or_smiles, *,
                                  pka: Optional[float] = None,
                                  linker_length: Optional[int] = None,
                                  n_tail_chains: int = 3,
                                  chain_avg_carbons: Optional[float] = None,
                                  v_tail_nm3: Optional[float] = None,
                                  l_tail_nm: Optional[float] = None,
                                  a_head_nm2: Optional[float] = None) -> Dict[str, float]:
    """Compute all physics-grounded descriptors for a SMILES (or Mol).

    The geometry inputs (v_tail_nm3, l_tail_nm, a_head_nm2) come from the
    existing compute_cpp.py pipeline; pass them in for the most-accurate CPP.
    If omitted, falls back to RDKit-derived estimates.
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return {}

    out = {}

    # CPP (geometric, if inputs provided)
    if v_tail_nm3 is not None and l_tail_nm is not None and a_head_nm2 is not None:
        out["cpp_geometric"] = cpp_geometric(v_tail_nm3, l_tail_nm, a_head_nm2)
    else:
        # Cheap RDKit fallback: V_tail ≈ 0.027 × heavy_aliphatic, l_tail ≈ 0.127 × n_C
        n_aliphatic_c = sum(1 for a in mol.GetAtoms()
                              if a.GetAtomicNum() == 6 and not a.GetIsAromatic())
        n_chains = max(1, n_tail_chains)
        out["cpp_geometric"] = (
            0.027 * n_aliphatic_c / n_chains  # V_tail per chain
        ) / (HEAD_AREA_REF_NM2 * (0.127 * n_aliphatic_c / n_chains + 0.3))

    # Lamellar d-spacing
    if linker_length is not None and chain_avg_carbons is not None:
        out["lamellar_d_nm"] = lamellar_d_spacing_nm(linker_length, chain_avg_carbons)
    else:
        out["lamellar_d_nm"] = float("nan")

    # Thermodynamic
    out["logKp_membrane"] = membrane_partition_logKp(mol)
    out["dG_transfer_kJmol"] = membrane_dG_transfer_kJmol(mol)

    # Mechanical
    out["bending_kappa_kBT"] = bending_modulus_kBT()
    if l_tail_nm is not None:
        out["splay_modulus_kBT"] = splay_modulus_kBT(l_tail_nm, n_tail_chains)

    # Electrostatic / escape
    if pka is not None:
        out["protonation_endosome"] = henderson_hasselbalch_fraction(pka, 5.5)
        out["protonation_cytosol"] = henderson_hasselbalch_fraction(pka, 7.4)
        if np.isfinite(out["cpp_geometric"]):
            out["endosomal_escape_score"] = endosomal_escape_score(
                pka, out["cpp_geometric"]
            )

    # Self-assembly thermodynamics
    n_aliphatic = sum(1 for a in mol.GetAtoms()
                       if a.GetAtomicNum() == 6 and not a.GetIsAromatic())
    out["cmc_log10_molar"] = cmc_log_molar(n_aliphatic / max(1, n_tail_chains))

    # Hydrophobic/hydrophilic balance
    out["hlb_griffin"] = hlb_griffin(mol)

    return out


__all__ = [
    "cpp_geometric", "lamellar_d_spacing_nm",
    "membrane_partition_logKp", "membrane_dG_transfer_kJmol",
    "henderson_hasselbalch_fraction", "endosomal_escape_score",
    "bending_modulus_kBT", "splay_modulus_kBT",
    "manning_fraction", "cmc_log_molar", "hlb_griffin",
    "compute_all_physics_features",
]


if __name__ == "__main__":
    # Smoke test on IAJD 369
    sm = "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC"
    feats = compute_all_physics_features(
        sm, pka=6.5, linker_length=2, n_tail_chains=3, chain_avg_carbons=9.0,
        v_tail_nm3=0.9, l_tail_nm=1.2, a_head_nm2=0.7,
    )
    import json
    print(json.dumps(feats, indent=2, default=str))
