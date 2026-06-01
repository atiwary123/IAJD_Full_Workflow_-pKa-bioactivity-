"""
physics_design/pressure_profile.py — lateral pressure profile P_L(z) - P_N(z) of a
flat Martini-3 bilayer, computed by recomputing the force field from a stock-GROMACS
trajectory. This is the same physics GROMACS-LS computes internally (the
Irving-Kirkwood-Noll configurational stress with a central-force decomposition of
3-body terms); we reimplement it in Python because GROMACS-LS is frozen at GROMACS
4.6.6 and cannot read 2026.x trajectories.

Why only the *configurational* (virial) part is needed:
  P_L(z) - P_N(z) = [P_xx+P_yy]/2 - P_zz . The kinetic contribution is isotropic on
  time-average (equipartition: 1/2 kT per d.o.f. per axis), so it cancels in L-N.
  Hence we need positions only — no velocities.

For each saved frame we sum, over every interaction, the central pairwise force f_ij
(direction d = r_i - r_j) and assign its lateral-minus-normal virial
  s_ij = (f_ij/|d|) * [ 1/2 (dx^2+dy^2) - dz^2 ]
to z via the Harasima contour (half at z_i, half at z_j), giving
  P_L(z)-P_N(z) = <sum_pairs s_ij assigned to slab> / (A * dz) .
3-body angle forces are first decomposed into central pair forces (CFD) so they too
enter as s_ij. Bonded 1-2 pairs are excluded from the nonbonded sum (nrexcl=1).

Interactions reproduced EXACTLY as GROMACS integrated them (Verlet + Potential-shift):
  LJ:   V = C12/r^12 - C6/r^6 ,  f = (12 C12/r^12 - 6 C6/r^6)/r , hard cutoff at rvdw
  RF:   V = (fE/eps_r) qi qj [1/r + krf r^2 - crf] ,  f = (fE/eps_r) qi qj [1/r^2 - 2 krf r]
        with eps_rf = inf (GROMACS epsilon_rf=0)  => krf = 1/(2 rc^3), crf = 3/(2 rc)
  bond: V = 1/2 k (r-r0)^2 ,  f = -k (r-r0)
  angle (G96 cosine, func 2): V = 1/2 k (cos th - cos th0)^2 -> analytic forces -> CFD

Pressure unit conversion: 1 kJ/(mol nm^3) = 16.6054 bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lipid_library import Lipid
from .martini_ff import MartiniFF

# Physical constants (GROMACS units: kJ/mol, nm, ps, e)
F_COULOMB = 138.935458       # kJ mol^-1 nm e^-2
KJMOLNM3_TO_BAR = 16.6054    # 1 kJ/(mol nm^3) in bar
BAR_NM_TO_MNM = 0.1          # 1 bar*nm = 0.1 mN/m  (for surface tension)


@dataclass
class AtomArrays:
    type_idx: np.ndarray        # (M,) int index into `type_names`
    type_names: List[str]
    charge: np.ndarray          # (M,) float
    c6: np.ndarray              # (T,T) C6 matrix over type_names
    c12: np.ndarray             # (T,T) C12 matrix
    bonds: np.ndarray           # (B,2) int global atom indices
    bond_r0: np.ndarray         # (B,)
    bond_k: np.ndarray          # (B,)
    angles: np.ndarray          # (A,3) int global atom indices (i,j,k); vertex j
    angle_cos0: np.ndarray      # (A,) cos(theta0)
    angle_k: np.ndarray         # (A,)
    excl_pairs: set             # set of (min,max) intramolecular bonded pairs
    n_lipid_atoms: int
    excl_qq_mask: Optional[np.ndarray] = None   # (B,) bool: bond is charged-charged


def build_atom_arrays(lip: Lipid, n_lipids: int, n_total: int,
                      ff: MartiniFF, water_type: str = "W") -> AtomArrays:
    """Assemble per-atom arrays for a system of `n_lipids` copies of `lip` followed
    by (n_total - n_lipids*nbeads) water beads."""
    nb = lip.n_beads
    n_lip_atoms = n_lipids * nb
    n_water = n_total - n_lip_atoms
    if n_water < 0:
        raise ValueError(f"n_total {n_total} < lipid atoms {n_lip_atoms}")

    lip_types = lip.bead_types
    lip_charges = [c for _, _, c in lip.beads]
    used_types = sorted(set(lip_types) | {water_type})
    type_index = {t: i for i, t in enumerate(used_types)}
    # dense C6/C12 over used types (raises if any pair missing -> no proxy)
    _, c6, c12 = ff.c6_c12_matrix(used_types)

    type_idx = np.empty(n_total, dtype=np.int32)
    charge = np.zeros(n_total, dtype=np.float64)
    for L in range(n_lipids):
        base = L * nb
        for b in range(nb):
            type_idx[base + b] = type_index[lip_types[b]]
            charge[base + b] = lip_charges[b]
    wt = type_index[water_type]
    type_idx[n_lip_atoms:] = wt
    # water charge 0 already

    # bonds / angles / exclusions, offset per lipid copy
    bonds = []
    bond_r0 = []
    bond_k = []
    excl = set()
    for L in range(n_lipids):
        base = L * nb
        for (i, j, r0, k) in lip.bonds:
            a, b = base + i, base + j
            bonds.append((a, b))
            bond_r0.append(r0)
            bond_k.append(k)
            excl.add((a, b) if a < b else (b, a))
    angles = []
    angle_cos0 = []
    angle_k = []
    for L in range(n_lipids):
        base = L * nb
        for (i, j, k, th0, kk) in lip.angles:
            angles.append((base + i, base + j, base + k))
            angle_cos0.append(np.cos(np.deg2rad(th0)))
            angle_k.append(kk)

    bonds_arr = np.array(bonds, dtype=np.int64) if bonds else np.zeros((0, 2), np.int64)
    if len(bonds_arr):
        qq_bond = charge[bonds_arr[:, 0]] * charge[bonds_arr[:, 1]]
        excl_qq_mask = qq_bond != 0.0
    else:
        excl_qq_mask = np.zeros(0, dtype=bool)
    return AtomArrays(
        type_idx=type_idx, type_names=used_types, charge=charge,
        c6=c6, c12=c12,
        bonds=bonds_arr,
        bond_r0=np.array(bond_r0), bond_k=np.array(bond_k),
        angles=np.array(angles, dtype=np.int64) if angles else np.zeros((0, 3), np.int64),
        angle_cos0=np.array(angle_cos0), angle_k=np.array(angle_k),
        excl_pairs=excl, n_lipid_atoms=n_lip_atoms, excl_qq_mask=excl_qq_mask,
    )


def build_atom_arrays_mixed(species, n_total: int, ff: MartiniFF,
                            water_type: str = "W") -> AtomArrays:
    """Like build_atom_arrays but for a MIXED system. `species` is an ORDERED list of
    (Lipid, count) laid out contiguously to match the .gro / [molecules] order, followed
    by (n_total - sum) water beads. Used by the host-method mixed bilayer (IAJD + POPC):
    every per-bead type/charge/bond/angle/exclusion is the real value for whichever
    species owns that bead — no proxy. The force recompute (energy_cross_check) validates
    the assembled arrays against GROMACS exactly as for a single species."""
    used_types = {water_type}
    for lip, _cnt in species:
        used_types |= set(lip.bead_types)
    used_types = sorted(used_types)
    type_index = {t: i for i, t in enumerate(used_types)}
    _, c6, c12 = ff.c6_c12_matrix(used_types)

    type_idx = np.empty(n_total, dtype=np.int32)
    charge = np.zeros(n_total, dtype=np.float64)
    bonds, bond_r0, bond_k = [], [], []
    angles, angle_cos0, angle_k = [], [], []
    excl = set()
    off = 0
    for lip, cnt in species:
        nb = lip.n_beads
        lt = lip.bead_types
        lc = [c for _, _, c in lip.beads]
        for L in range(cnt):
            base = off + L * nb
            for b in range(nb):
                type_idx[base + b] = type_index[lt[b]]
                charge[base + b] = lc[b]
            for (i, j, r0, k) in lip.bonds:
                a, bb = base + i, base + j
                bonds.append((a, bb)); bond_r0.append(r0); bond_k.append(k)
                excl.add((a, bb) if a < bb else (bb, a))
            for (i, j, k, th0, kk) in lip.angles:
                angles.append((base + i, base + j, base + k))
                angle_cos0.append(np.cos(np.deg2rad(th0))); angle_k.append(kk)
        off += cnt * nb
    n_lip_atoms = off
    if n_total < n_lip_atoms:
        raise ValueError(f"n_total {n_total} < molecule atoms {n_lip_atoms}")
    type_idx[n_lip_atoms:] = type_index[water_type]

    bonds_arr = np.array(bonds, dtype=np.int64) if bonds else np.zeros((0, 2), np.int64)
    if len(bonds_arr):
        qq_bond = charge[bonds_arr[:, 0]] * charge[bonds_arr[:, 1]]
        excl_qq_mask = qq_bond != 0.0
    else:
        excl_qq_mask = np.zeros(0, dtype=bool)
    return AtomArrays(
        type_idx=type_idx, type_names=used_types, charge=charge, c6=c6, c12=c12,
        bonds=bonds_arr, bond_r0=np.array(bond_r0), bond_k=np.array(bond_k),
        angles=np.array(angles, dtype=np.int64) if angles else np.zeros((0, 3), np.int64),
        angle_cos0=np.array(angle_cos0), angle_k=np.array(angle_k),
        excl_pairs=excl, n_lipid_atoms=n_lip_atoms, excl_qq_mask=excl_qq_mask,
    )


def _minimum_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply orthorhombic minimum image to displacement vectors d (..,3)."""
    return d - box * np.round(d / box)


@dataclass
class ProfileResult:
    z: np.ndarray            # (nbins,) bin-center positions, midplane at 0 (nm)
    dP: np.ndarray           # (nbins,) P_L(z)-P_N(z) in bar (total)
    dP_nb: np.ndarray        # nonbonded LJ+RF contribution
    dP_bond: np.ndarray      # bond contribution
    dP_angle: np.ndarray     # angle (CFD) contribution
    n_frames: int
    surface_tension_mNm: float        # from the local profile integral
    area_nm2: float                   # mean lateral area
    # global configurational pressure tensor (bar) — cross-check vs GROMACS
    P_xx_config: float = float("nan")
    P_yy_config: float = float("nan")
    P_zz_config: float = float("nan")
    gamma_global_mNm: float = float("nan")   # Lz*(Pzz-(Pxx+Pyy)/2), config only
    contour: str = "ik"


def _deposit(acc: np.ndarray, s: np.ndarray, zi: np.ndarray, dz_z: np.ndarray,
             Lz: float, zmin: float, dz: float, contour: str = "ik", K: int = 10):
    """Distribute each pair stress s onto the z grid along the MINIMUM-IMAGE z path.

    The contour runs from z_i (t=0) to z_j = z_i - dz_z (t=1), where dz_z is the
    minimum-image z separation (= (r_i-r_j)_z). Each sampled point is wrapped into
    [-Lz/2, Lz/2] before binning, so a pair straddling the periodic z-boundary
    deposits its stress along the SHORT path across the boundary (bulk water at the
    box edge) rather than the long path through the membrane — fixing the spurious
    water-region baseline.

    'harasima' : half at z_i, half at z_j (spiky; fast).
    'ik'       : Irving-Kirkwood — s spread uniformly along the path by K-point
                 midpoint quadrature (smooth; the rigorous choice).
    """
    nb = acc.shape[0]
    if contour == "harasima":
        ts = np.array([0.0, 1.0])
        w = 0.5
    else:
        ts = (np.arange(K) + 0.5) / K          # midpoints of K sub-segments
        w = 1.0 / K
    # z positions along the min-image path (npairs, len(ts)), wrapped into [-Lz/2,Lz/2]
    zpos = zi[:, None] - ts[None, :] * dz_z[:, None]
    zpos = ((zpos + 0.5 * Lz) % Lz) - 0.5 * Lz
    bins = np.floor((zpos - zmin) / dz).astype(np.int64)
    m = (bins >= 0) & (bins < nb)
    vals = np.broadcast_to((w * s)[:, None], zpos.shape)
    np.add.at(acc, bins[m], vals[m])


def _lj_rf_scalar(r: np.ndarray, c6: np.ndarray, c12: np.ndarray,
                  qq: np.ndarray, eps_r: float, rc: float) -> np.ndarray:
    """Central scalar force f (force on i along r_i-r_j; +=repulsive) for LJ+RF."""
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    f_lj = (12.0 * c12 * inv_r12 - 6.0 * c6 * inv_r6) * inv_r
    krf = 1.0 / (2.0 * rc ** 3)
    f_rf = (F_COULOMB / eps_r) * qq * (inv_r2 - 2.0 * krf * r)
    return f_lj + f_rf


def compute_profile(traj: Path, topol: Path, sysarr: AtomArrays, *,
                    rc: float = 1.1, eps_r: float = 15.0,
                    dz: float = 0.1, last_frac: float = 0.6,
                    stride: int = 1, max_frames: Optional[int] = None,
                    contour: str = "ik", ik_K: int = 10) -> ProfileResult:
    """Compute the lateral pressure profile from `traj` (xtc) + `topol` (gro/tpr)."""
    import MDAnalysis as mda
    from MDAnalysis.lib import distances as mdadist

    u = mda.Universe(str(topol), str(traj))
    n_total = len(u.atoms)
    if n_total != sysarr.type_idx.shape[0]:
        raise ValueError(f"atom count mismatch: traj {n_total} vs arrays "
                         f"{sysarr.type_idx.shape[0]}")
    n_frames_total = len(u.trajectory)
    start = int(n_frames_total * (1.0 - last_frac))
    frame_ids = list(range(start, n_frames_total, stride))
    if max_frames:
        frame_ids = frame_ids[:max_frames]

    bonds = sysarr.bonds
    angles = sysarr.angles
    c6m, c12m = sysarr.c6, sysarr.c12
    tix = sysarr.type_idx
    q = sysarr.charge
    lip_mask = np.zeros(n_total, dtype=bool)
    lip_mask[:sysarr.n_lipid_atoms] = True
    # encode excluded (bonded 1-2) pairs as int64 keys for vectorized filtering
    if sysarr.excl_pairs:
        excl_keys = np.array(sorted(a * n_total + b for (a, b) in sysarr.excl_pairs),
                             dtype=np.int64)
    else:
        excl_keys = np.zeros(0, dtype=np.int64)

    # fixed z grid, symmetric about midplane; sized from first frame box
    u.trajectory[frame_ids[0]]
    Lz0 = float(u.dimensions[2]) / 10.0
    zhalf = Lz0 / 2.0 + 0.5
    nbins = int(np.ceil(2 * zhalf / dz))
    zmin = -zhalf
    zc = zmin + (np.arange(nbins) + 0.5) * dz

    acc_nb = np.zeros(nbins)
    acc_bond = np.zeros(nbins)
    acc_angle = np.zeros(nbins)
    area_sum = 0.0
    sumP = np.zeros(3)        # running sum of per-frame config P_xx,P_yy,P_zz (bar)
    sum_gamma_g = 0.0
    nfr = 0

    for fi in frame_ids:
        u.trajectory[fi]
        box_nm = np.array(u.dimensions[:3]) / 10.0        # Lx,Ly,Lz nm
        pos = u.atoms.positions / 10.0                    # (M,3) nm
        A = box_nm[0] * box_nm[1]
        V = A * box_nm[2]
        area_sum += A
        invA = 1.0 / A
        Wframe = np.zeros(3)   # global config virial diagonal (kJ/mol)

        # center membrane: shift so lipid COM z -> box center, then z'=z-Lz/2
        comz = pos[lip_mask, 2].mean()
        pos[:, 2] = np.mod(pos[:, 2] - comz + box_nm[2] / 2.0, box_nm[2])
        zprime = pos[:, 2] - box_nm[2] / 2.0

        # ---- nonbonded pairs within rc (PBC) ----
        pairs, dists = mdadist.self_capped_distance(
            pos, max_cutoff=rc, box=np.array([box_nm[0], box_nm[1], box_nm[2],
                                              90.0, 90.0, 90.0]))
        if len(pairs):
            ii = pairs[:, 0]
            jj = pairs[:, 1]
            # exclusions: drop intramolecular bonded (1-2) pairs (vectorized)
            if excl_keys.size:
                lo = np.minimum(ii, jj).astype(np.int64)
                hi = np.maximum(ii, jj).astype(np.int64)
                keys = lo * n_total + hi
                keep = ~np.isin(keys, excl_keys)
                ii, jj = ii[keep], jj[keep]
                dists = dists[keep]
            d = _minimum_image(pos[ii] - pos[jj], box_nm)
            r = dists
            c6 = c6m[tix[ii], tix[jj]]
            c12 = c12m[tix[ii], tix[jj]]
            qq = q[ii] * q[jj]
            f = _lj_rf_scalar(r, c6, c12, qq, eps_r, rc)
            _add_pairs(acc_nb, Wframe, f, d, r, zprime[ii], box_nm[2],
                       invA, zmin, dz, contour, ik_K)

        # ---- bonds ----
        if len(bonds):
            ia, ib = bonds[:, 0], bonds[:, 1]
            d = _minimum_image(pos[ia] - pos[ib], box_nm)
            r = np.linalg.norm(d, axis=1)
            fb = -sysarr.bond_k * (r - sysarr.bond_r0)
            _add_pairs(acc_bond, Wframe, fb, d, r, zprime[ia], box_nm[2],
                       invA, zmin, dz, contour, ik_K)

        # ---- reaction-field correction for EXCLUDED (bonded) charged pairs ----
        # GROMACS RF applies the homogeneous reaction-field term (k_rf r^2) to
        # excluded pairs too (the bonded charges still polarise the continuum);
        # only the bare 1/r Coulomb is excluded. Force = (fE/eps)*qq*(-2 krf r).
        if sysarr.excl_qq_mask is not None and sysarr.excl_qq_mask.any():
            ea, eb = bonds[sysarr.excl_qq_mask, 0], bonds[sysarr.excl_qq_mask, 1]
            qqe = q[ea] * q[eb]
            d = _minimum_image(pos[ea] - pos[eb], box_nm)
            r = np.linalg.norm(d, axis=1)
            krf = 1.0 / (2.0 * rc ** 3)
            f_excl = (F_COULOMB / eps_r) * qqe * (-2.0 * krf * r)
            _add_pairs(acc_nb, Wframe, f_excl, d, r, zprime[ea], box_nm[2],
                       invA, zmin, dz, contour, ik_K)

        # ---- angles (analytic forces -> central-force decomposition) ----
        if len(angles):
            _accumulate_angles(acc_angle, Wframe, pos, box_nm, zprime, angles,
                               sysarr.angle_cos0, sysarr.angle_k, invA, zmin, dz,
                               contour, ik_K)

        # global config pressure tensor this frame (bar)
        Pframe = Wframe / V * KJMOLNM3_TO_BAR
        sumP += Pframe
        sum_gamma_g += BAR_NM_TO_MNM * box_nm[2] * (
            Pframe[2] - 0.5 * (Pframe[0] + Pframe[1]))
        nfr += 1

    # finalize: divide by dz * n_frames, convert kJ/mol/nm^3 -> bar
    norm = KJMOLNM3_TO_BAR / (dz * max(1, nfr))
    dP_nb = acc_nb * norm
    dP_bond = acc_bond * norm
    dP_angle = acc_angle * norm
    dP = dP_nb + dP_bond + dP_angle
    gamma = -BAR_NM_TO_MNM * float(np.sum(dP) * dz)      # mN/m, from local profile
    Pcfg = sumP / max(1, nfr)
    return ProfileResult(z=zc, dP=dP, dP_nb=dP_nb, dP_bond=dP_bond,
                         dP_angle=dP_angle, n_frames=nfr,
                         surface_tension_mNm=gamma,
                         area_nm2=area_sum / max(1, nfr),
                         P_xx_config=float(Pcfg[0]), P_yy_config=float(Pcfg[1]),
                         P_zz_config=float(Pcfg[2]),
                         gamma_global_mNm=sum_gamma_g / max(1, nfr),
                         contour=contour)


def _add_pairs(acc, Wframe, f, d, r, zi, Lz, invA, zmin, dz, contour, K):
    """Add a batch of central pair forces (scalar f along d=r_a-r_b, |d|=r) to the
    local profile `acc` (P_L-P_N stress via the min-image z contour) and to the
    global configurational virial diagonal `Wframe` (kJ/mol). `zi` is the centered
    z of the first atom; the partner's z is z_i - d_z (min image)."""
    fr = f / r
    wxx = fr * d[:, 0] ** 2
    wyy = fr * d[:, 1] ** 2
    wzz = fr * d[:, 2] ** 2
    Wframe[0] += wxx.sum()
    Wframe[1] += wyy.sum()
    Wframe[2] += wzz.sum()
    s = (0.5 * (wxx + wyy) - wzz) * invA
    _deposit(acc, s, zi, d[:, 2], Lz, zmin, dz, contour, K)


def _accumulate_angles(acc, Wframe, pos, box_nm, zprime, angles, cos0, kk, invA,
                       zmin, dz, contour, K):
    """G96 cosine angle forces -> central-force decomposition -> stress + virial."""
    i, j, k = angles[:, 0], angles[:, 1], angles[:, 2]
    rij = _minimum_image(pos[i] - pos[j], box_nm)
    rkj = _minimum_image(pos[k] - pos[j], box_nm)
    nij = np.linalg.norm(rij, axis=1)
    nkj = np.linalg.norm(rkj, axis=1)
    uij = rij / nij[:, None]
    ukj = rkj / nkj[:, None]
    cth = np.sum(uij * ukj, axis=1)
    cth = np.clip(cth, -1.0, 1.0)
    dVdcos = kk * (cth - cos0)                       # dV/dcos(theta)
    # Fi = -dV/dcos * dcos/dri ; dcos/dri = (ukj - cth*uij)/nij
    Fi = -dVdcos[:, None] * (ukj - cth[:, None] * uij) / nij[:, None]
    Fk = -dVdcos[:, None] * (uij - cth[:, None] * ukj) / nkj[:, None]
    Fj = -(Fi + Fk)

    # CFD: solve for central pair scalars a_ij, a_ik, a_jk (exact for 3 bodies)
    rik = _minimum_image(pos[i] - pos[k], box_nm)
    nik = np.linalg.norm(rik, axis=1)
    uik = rik / nik[:, None]
    ujk = -ukj                                       # (rj - rk) direction
    n_ang = angles.shape[0]
    M = np.zeros((n_ang, 9, 3))
    M[:, 0:3, 0] = uij           # a_ij on i: +uij
    M[:, 3:6, 0] = -uij          # on j: -uij
    M[:, 0:3, 1] = uik           # a_ik on i: +uik
    M[:, 6:9, 1] = -uik          # on k: -uik
    M[:, 3:6, 2] = ujk           # a_jk on j: +ujk
    M[:, 6:9, 2] = -ujk          # on k: -ujk
    F = np.concatenate([Fi, Fj, Fk], axis=1)         # (n_ang, 9)
    MtM = np.einsum('aij,aik->ajk', M, M)
    MtF = np.einsum('aij,ai->aj', M, F)
    MtM += 1e-9 * np.eye(3)[None, :, :]
    a = np.linalg.solve(MtM, MtF)                    # (n_ang,3): a_ij,a_ik,a_jk

    # each central pair contributes like a normal pair force of scalar a along
    # dvec = r_pa - r_pb (so the contour partner z_pb = z_pa - dvec_z is correct).
    Lz = box_nm[2]
    for col, (pa, pb, dvec, dist) in enumerate([
            (i, j, rij, nij), (i, k, rik, nik), (j, k, -rkj, nkj)]):
        _add_pairs(acc, Wframe, a[:, col], dvec, dist, zprime[pa], Lz,
                   invA, zmin, dz, contour, K)


__all__ = ["AtomArrays", "build_atom_arrays", "ProfileResult", "compute_profile"]
