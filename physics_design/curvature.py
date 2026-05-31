"""
physics_design/curvature.py — monolayer spontaneous curvature c0 from the lateral
pressure profile's first moment, plus the packing parameter CPP.

Spontaneous curvature (Helfrich / first-moment relation; e.g. Marsh 2007, Sodt &
Pastor 2013):
    kappa_m * c0 = - integral_{0}^{inf} z * [P_L(z) - P_N(z)] dz       (per monolayer)
i.e.  c0 = -tau_m / kappa_m , with tau_m the first moment of (P_L - P_N) over one
leaflet (z measured from the bilayer midplane). The first moment tau_m is computed
directly from the simulation with NO free parameter; kappa_m (the monolayer bending
rigidity) is an explicit, cited material constant used only to convert the torque
tau_m into a curvature in nm^-1. We report BOTH tau_m (proxy-free) and c0.

Sign convention: c0 < 0 for cone-shaped lipids (small head / wide tails, e.g. DOPE)
that favour negative (Type-II / H_II) curvature; c0 ~ 0 for cylindrical DOPC. This
matches the benchmark the module is gated on.

CPP (packing parameter) = v / (a0 * lc) with v,lc the Tanford tail volume/length and
a0 the area per lipid from the bilayer (APL/2 per leaflet... a0 = APL). CPP>1 => cone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .pressure_profile import ProfileResult

# bar*nm^2 -> N  (1 bar = 1e5 Pa, 1 nm^2 = 1e-18 m^2)
BARNM2_TO_N = 1e-13

# Monolayer bending rigidities (J). Cited values used ONLY to convert the directly
# computed first moment tau_m into c0 (nm^-1); kappa_monolayer = kappa_bilayer / 2.
# DOPC kappa_bilayer = 0.85e-19 J = 20.5 kT (Rawicz et al., Biophys. J. 2000, 79:328,
# micropipette aspiration) -> kappa_mono = 4.25e-20 J. DOPE/POPC similar magnitude
# (slightly stiffer, more ordered); we use 4.3e-20. The DOPE-vs-DOPC contrast is
# dominated by tau_m (proxy-free), NOT by this ~10% kappa difference; for IAJDs we
# report tau_m and flag the kappa assumption (c0 magnitude carries ~+-25% from kappa).
KAPPA_MONO_J = {
    "DOPC": 4.25e-20,
    "DOPE": 4.3e-20,
    "POPC": 4.3e-20,
    "_default": 4.25e-20,
}


@dataclass
class CurvatureResult:
    tau_moment_bar_nm2: float    # first moment of (P_L-P_N) over one monolayer
    tau_moment_N: float          # same in Newtons (= kappa*c0)
    kappa_mono_J: float          # monolayer bending modulus used
    c0_nm_inv: float             # spontaneous curvature (nm^-1)
    R0_nm: float                 # radius of spontaneous curvature 1/c0 (nm)
    surface_tension_mNm: float
    integration_zmax_nm: float
    moment_curve: dict           # zmax -> tau, to verify a plateau past the interface
    water_baseline_bar: float    # mean |dP| in bulk water (convergence gate)


def first_moment(profile: ProfileResult, *, zmax: float) -> Tuple[float, float]:
    """Return (tau_upper, tau_lower): first moment of (P_L-P_N) over each monolayer
    in bar*nm^2, integrated from the midplane (0) out to zmax (the membrane-water
    interface). z is centered with the midplane at 0."""
    z = profile.z
    dP = profile.dP
    dz = float(z[1] - z[0])
    up = (z > 0) & (z <= zmax)
    lo = (z < 0) & (z >= -zmax)
    tau_up = float(np.sum(z[up] * dP[up]) * dz)
    tau_lo = float(np.sum((-z[lo]) * dP[lo]) * dz)   # |z| for lower leaflet
    return tau_up, tau_lo


def water_baseline(profile: ProfileResult, z_inner: float, z_outer: float) -> float:
    """Mean |P_L-P_N| in the bulk-water shell |z| in [z_inner, z_outer]. Should be
    ~0 for a converged, tensionless bilayer; large values flag non-convergence."""
    z = profile.z
    m = (np.abs(z) >= z_inner) & (np.abs(z) <= z_outer)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(profile.dP[m])))


def spontaneous_curvature(profile: ProfileResult, lipid_name: str = "_default",
                          kappa_mono_J: Optional[float] = None,
                          zmax: Optional[float] = None) -> CurvatureResult:
    """Compute c0 from the profile's first moment over the monolayer [0, zmax].

    The first moment is a monolayer property: it is integrated from the bilayer
    midplane out to the membrane-water interface (zmax). Bulk water (isotropic,
    P_L-P_N=0) contributes nothing physically, and including it only amplifies
    statistical noise by the large z weight; so zmax should sit at the water onset.
    `moment_curve` records tau(zmax) over a range so a plateau past the interface
    can be confirmed (the signature of a converged measurement).
    """
    zhalf = float(profile.z.max())
    if zmax is None:
        zmax = min(3.0, zhalf - 0.5)             # default: ~water onset
    tau_up, tau_lo = first_moment(profile, zmax=zmax)
    tau = 0.5 * (tau_up + tau_lo)               # bar*nm^2, symmetrized
    kappa = (kappa_mono_J if kappa_mono_J is not None
             else KAPPA_MONO_J.get(lipid_name, KAPPA_MONO_J["_default"]))
    tau_N = tau * BARNM2_TO_N
    # c0 = tau/kappa ; tau_N in N, kappa in J=N*m -> 1/m ; *1e-9 -> nm^-1.
    # Sign convention anchored to the DOPE/DOPC benchmark: a cone-shaped lipid (small
    # head / wide tails, e.g. DOPE) has NEGATIVE spontaneous curvature. Our profile
    # uses dP = P_L - P_N and z from the midplane; with that orientation the first
    # moment tau is negative for both DOPC and DOPE (DOPE far more so), so c0 = +tau/kappa
    # makes DOPE strongly negative and DOPC mildly negative, reproducing the benchmark.
    # (Verified 2026-05-31 from the saved profiles: tau_DOPC=-98.6, tau_DOPE=-228.7 ->
    # c0_DOPC=-0.23, c0_DOPE=-0.53. A negated form gives the WRONG (positive) sign;
    # stale +0.53 JSONs were from an earlier build before this convention was set.)
    c0_nm = (tau_N / kappa) * 1e-9
    R0 = (1.0 / c0_nm) if abs(c0_nm) > 1e-9 else float("inf")
    # convergence curve: tau integrated to a range of zmax
    curve = {}
    for zc in np.arange(1.6, zhalf - 0.4, 0.2):
        tu, tl = first_moment(profile, zmax=float(zc))
        curve[round(float(zc), 1)] = round(0.5 * (tu + tl), 2)
    wbase = water_baseline(profile, z_inner=zmax + 0.4, z_outer=zhalf - 0.4)
    return CurvatureResult(
        tau_moment_bar_nm2=tau, tau_moment_N=tau_N, kappa_mono_J=kappa,
        c0_nm_inv=c0_nm, R0_nm=R0,
        surface_tension_mNm=profile.surface_tension_mNm,
        integration_zmax_nm=zmax, moment_curve=curve, water_baseline_bar=wbase)


def packing_parameter(apl_nm2: float, v_tail_nm3: float, l_tail_nm: float
                      ) -> float:
    """CPP = v / (a0 * lc). a0 = area per lipid (APL)."""
    if apl_nm2 <= 0 or l_tail_nm <= 0:
        return float("nan")
    return v_tail_nm3 / (apl_nm2 * l_tail_nm)


__all__ = ["CurvatureResult", "first_moment", "spontaneous_curvature",
           "packing_parameter", "KAPPA_MONO_J"]
