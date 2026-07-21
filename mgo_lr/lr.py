"""Standalone long-range (LR) Hamiltonian processor.

Unit/sign convention (see also constants.py — this module is the ONLY
place the Coulomb prefactor and sign enter):

    Z*  dimensionless (units of e);  u in Å;  d_k = Z~*_k u_k^rel (e·Å)
    phi(G) = -i (4π/Ω) C_COUL [Σ_k G·d_k e^(-iG·R0_k)] / (G·ε∞·G) f_Ewald(G)
    V_LR(G) = LR_SIGN · phi(G)          # electron potential energy, eV
    V_LR(r) = Σ_{G∈𝒢} V_LR(G) e^(+iG·r);   V(G=0) = 0 (fixed gauge)
    f_Ewald(G) = exp(-(G·ε∞·G)/(4Λ²))

Λ is part of the dataset definition: the damped reciprocal-space sum alone
IS the LR definition (no compensating real-space term), so H^LR depends on
Λ by construction.  G-set requirements (inversion symmetry, G=0 excluded,
no duplicates) are hard: the realness of V^LR depends on them.
"""
import json
import os

import numpy as np

from .constants import C_COUL, LR_SIGN
from .convert import key_str, parse_key


def gmax_squared(lam, tol):
    """Bound on G·ε∞·G from the f_Ewald floor `tol`."""
    return 4.0 * float(lam) ** 2 * np.log(1.0 / float(tol))


def reciprocal_set(rec_cell, eps, gmax_sq):
    """Integer combinations of supercell reciprocal vectors inside the
    dielectric ellipsoid G·ε∞·G <= gmax_sq, G=0 excluded.  The symmetric
    cutoff makes the set inversion-symmetric by construction."""
    rec = np.asarray(rec_cell, float)
    eps = np.asarray(eps, float)
    eps_min = float(np.linalg.eigvalsh(0.5 * (eps + eps.T)).min())
    if eps_min <= 0.0:
        raise ValueError("dielectric tensor not positive definite")
    gmax_cart = np.sqrt(gmax_sq / eps_min)
    real = 2.0 * np.pi * np.linalg.inv(rec).T          # rows a_i
    nmax = [int(np.ceil(gmax_cart * np.linalg.norm(a) / (2.0 * np.pi)))
            for a in real]
    ns, gs = [], []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                if n1 == n2 == n3 == 0:
                    continue
                g = np.array([n1, n2, n3], float) @ rec
                if float(g @ eps @ g) <= gmax_sq:
                    ns.append((n1, n2, n3))
                    gs.append(g)
    return np.array(ns, int).reshape(-1, 3), np.array(gs).reshape(-1, 3)


def check_reciprocal_set(n_int):
    tuples = [tuple(int(x) for x in v) for v in np.asarray(n_int).reshape(-1, 3)]
    s = set(tuples)
    rep = {"number_of_vectors": len(tuples),
           "excludes_G_zero": (0, 0, 0) not in s,
           "no_duplicates": len(s) == len(tuples),
           "inversion_symmetric": all((-a, -b, -c) in s for a, b, c in s)}
    rep["ok"] = (rep["excludes_G_zero"] and rep["no_duplicates"]
                 and rep["inversion_symmetric"])
    return rep


def lr_coefficients(g_cart, dipoles, ref_positions, eps, lam, volume):
    """V_LR(G) with the reference-position phase convention (exactly linear
    in u^rel)."""
    g = np.asarray(g_cart, float)
    eps = np.asarray(eps, float)
    geg = np.einsum("ga,ab,gb->g", g, eps, g)
    f_ewald = np.exp(-geg / (4.0 * float(lam) ** 2))
    gd = g @ np.asarray(dipoles, float).T                       # (M,N)
    phases = np.exp(-1j * (g @ np.asarray(ref_positions, float).T))
    s_g = np.sum(gd * phases, axis=1)
    phi = -1j * (4.0 * np.pi / float(volume)) * C_COUL * s_g / geg * f_ewald
    return LR_SIGN * phi


def evaluate_potential(g_cart, coeffs, points):
    ph = np.exp(1j * (np.asarray(points, float) @ np.asarray(g_cart, float).T))
    return ph @ np.asarray(coeffs)


def imaginary_residual(v, delta):
    v = np.asarray(v)
    return float(np.linalg.norm(np.imag(v))
                 / (np.linalg.norm(np.real(v)) + float(delta)))


def minimum_image_displacements(cell, cart, ref_cart):
    """u = cart - ref wrapped to the nearest image (valid for |u| << cell)."""
    cell = np.asarray(cell, float)
    dfrac = (np.asarray(cart, float) - np.asarray(ref_cart, float)) \
        @ np.linalg.inv(cell)
    dfrac -= np.round(dfrac)
    return dfrac @ cell


def assemble_lr_hamiltonian(overlap_blocks, v_atom):
    """H^LR_ij(R) = (V_i + V_j)/2 * S_ij(R) over every stored overlap key.
    Hermiticity is inherited from S."""
    v_atom = np.asarray(v_atom, float)
    out = {}
    for k, s in overlap_blocks.items():
        _, _, _, i, j = parse_key(k)                    # 1-based
        out[k] = 0.5 * (v_atom[i - 1] + v_atom[j - 1]) * np.asarray(s, float)
    return out


def blocks_norm(blocks):
    return float(np.sqrt(sum(float(np.sum(v * v)) for v in blocks.values())))


def blocks_diff_norm(a, b):
    """Frobenius norm of a - b over the union of keys (absent -> zero)."""
    tot = 0.0
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        d = va - vb if va is not None and vb is not None \
            else (va if vb is None else -vb)
        tot += float(np.sum(d * d))
    return float(np.sqrt(tot))
