import cmath
import math

import numpy as np
import pytest

from mgo_lr import lr
from mgo_lr.constants import C_COUL, LR_SIGN
from mgo_lr.convert import key_str
from mgo_lr.structures import reciprocal

EPS_I = np.eye(3)


def _cube(L=8.0):
    cell = L * np.eye(3)
    return cell, reciprocal(cell), abs(np.linalg.det(cell))


def test_reciprocal_set_properties():
    cell, rec, vol = _cube()
    gmax_sq = lr.gmax_squared(1.0, 1e-10)
    n_int, g_cart = lr.reciprocal_set(rec, EPS_I, gmax_sq)
    rep = lr.check_reciprocal_set(n_int)
    assert rep["ok"] and rep["number_of_vectors"] == len(n_int) > 0
    assert rep["excludes_G_zero"] and rep["inversion_symmetric"]
    # every vector satisfies the ellipsoidal cutoff
    assert all(g @ EPS_I @ g <= gmax_sq + 1e-9 for g in g_cart)


def test_check_flags_broken_set():
    n_int, _ = lr.reciprocal_set(_cube()[1], EPS_I, lr.gmax_squared(1.0, 1e-6))
    broken = n_int[1:]                      # drop one vector -> asymmetric
    assert lr.check_reciprocal_set(broken)["ok"] is False
    with_zero = np.vstack([n_int, [0, 0, 0]])
    assert lr.check_reciprocal_set(with_zero)["excludes_G_zero"] is False


def test_sign_and_prefactor_against_filtered_dipole():
    """Production vectorized implementation vs a deliberately slow loop
    reference building the SAME filtered coefficients, plus the sign pin:
    an electron just above the positive lobe of a +z dipole has NEGATIVE
    potential energy."""
    cell, rec, vol = _cube(8.0)
    lam, tol = 0.6, 1e-8
    n_int, g_cart = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(lam, tol))
    dipoles = np.array([[0.0, 0.0, 0.1]])
    refpos = np.array([[4.0, 4.0, 4.0]])
    points = np.array([[4.0, 4.0, 5.0], [4.0, 4.0, 3.0], [5.0, 4.0, 4.0]])
    coeffs = lr.lr_coefficients(g_cart, dipoles, refpos, EPS_I, lam, vol)
    v = lr.evaluate_potential(g_cart, coeffs, points)

    slow = np.zeros(len(points), complex)
    for nvec in n_int:
        g = np.asarray(nvec, float) @ rec
        geg = float(g @ EPS_I @ g)
        f = math.exp(-geg / (4.0 * lam * lam))
        s = sum((g @ d) * cmath.exp(-1j * float(g @ r0))
                for d, r0 in zip(dipoles, refpos))
        vg = LR_SIGN * (-1j) * (4.0 * math.pi / vol) * C_COUL * s / geg * f
        for p, r in enumerate(points):
            slow[p] += vg * cmath.exp(1j * float(g @ r))
    assert np.allclose(v, slow, atol=1e-10)
    assert lr.imaginary_residual(v, 1e-12) < 1e-10
    vr = np.real(v)
    assert vr[0] < 0.0 < vr[1]              # sign pin (electron energy)
    assert abs(vr[0] + vr[1]) < 1e-10       # odd in z


def test_coefficients_linear_in_dipoles():
    cell, rec, vol = _cube()
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(0.8, 1e-8))
    rng = np.random.default_rng(0)
    d = rng.standard_normal((4, 3)) * 0.01
    r0 = rng.uniform(0, 8, (4, 3))
    c1 = lr.lr_coefficients(g, d, r0, EPS_I, 0.8, vol)
    c2 = lr.lr_coefficients(g, 2.0 * d, r0, EPS_I, 0.8, vol)
    assert np.allclose(c2, 2.0 * c1)        # exactly linear in u_rel
    c0 = lr.lr_coefficients(g, 0.0 * d, r0, EPS_I, 0.8, vol)
    assert np.allclose(c0, 0.0)             # equilibrium -> exact zero


def test_uniform_translation_exact_zero():
    from mgo_lr.displacements import remove_uniform_translation
    u = np.tile([[0.03, -0.01, 0.02]], (6, 1))
    assert np.allclose(remove_uniform_translation(u), 0.0)


def test_realness_requires_inversion_symmetry():
    cell, rec, vol = _cube()
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(0.8, 1e-6))
    d = np.array([[0.0, 0.0, 0.05]])
    r0 = np.array([[1.234, 2.345, 3.456]])
    c = lr.lr_coefficients(g, d, r0, EPS_I, 0.8, vol)
    pts = np.array([[0.5, 1.5, 2.5]])
    assert lr.imaginary_residual(lr.evaluate_potential(g, c, pts), 1e-12) < 1e-10
    # drop the highest-weight vector (its -G partner stays): residual blows up
    mask = np.ones(len(g), bool)
    mask[int(np.argmax(np.abs(c)))] = False
    v_bad = lr.evaluate_potential(g[mask], c[mask], pts)
    assert lr.imaginary_residual(v_bad, 1e-12) > 1e-4


def test_gmax_convergence_at_fixed_lambda():
    cell, rec, vol = _cube()
    lam, tol = 0.8, 1e-10
    d = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    r0 = np.array([[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]])
    pts = r0 + 0.01
    vs = []
    for scale in (1.0, 1.5 ** 2):
        n_int, g = lr.reciprocal_set(rec, EPS_I,
                                     lr.gmax_squared(lam, tol) * scale)
        c = lr.lr_coefficients(g, d, r0, EPS_I, lam, vol)
        vs.append(np.real(lr.evaluate_potential(g, c, pts)))
    rel = np.linalg.norm(vs[1] - vs[0]) / (np.linalg.norm(vs[1]) + 1e-12)
    assert rel < 1e-6                       # converged at fixed Lambda


def test_minimum_image_displacements():
    cell = 10.0 * np.eye(3)
    ref = np.array([[0.5, 0.5, 0.5]])
    cart = np.array([[9.8, 0.5, 0.5]])      # wrapped: really at -0.2
    u = lr.minimum_image_displacements(cell, cart, ref)
    assert np.allclose(u, [[-0.7, 0.0, 0.0]])


def test_assemble_and_hermiticity_and_small_amplitude():
    """H^LR = (V_i+V_j)/2 S inherits hermiticity from S; with u-dependent S
    the sign-reversal and linearity errors DECREASE with amplitude."""
    cell, rec, vol = _cube(8.0)
    lam = 0.8
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(lam, 1e-8))
    # asymmetric positions: a mirror-symmetric pair (2,2,2)/(6,2,2) makes
    # both dipoles equal AND kills the linear term of V by symmetry
    ref = np.array([[2.0, 2.0, 2.0], [5.0, 3.0, 2.0]])
    z = np.array([np.eye(3) * 2.0, np.eye(3) * -2.0])

    def h_lr(amp):
        u = np.array([[amp, 0.0, 0.0], [-amp, 0.0, 0.0]])
        from mgo_lr.displacements import remove_uniform_translation
        u_rel = remove_uniform_translation(u)
        d = np.einsum("nab,nb->na", z, u_rel)
        pos = ref + u
        c = lr.lr_coefficients(g, d, ref, EPS_I, lam, vol)
        v = np.real(lr.evaluate_potential(g, c, pos))
        dist = np.linalg.norm(pos[0] - pos[1])
        s12 = np.array([[math.exp(-dist / 4.0)]])   # u-dependent overlap
        s = {key_str((0, 0, 0), 0, 0): np.array([[1.0]]),
             key_str((0, 0, 0), 1, 1): np.array([[1.0]]),
             key_str((0, 0, 0), 0, 1): s12,
             key_str((0, 0, 0), 1, 0): s12.T.copy()}
        return lr.assemble_lr_hamiltonian(s, v)

    h = h_lr(0.02)
    assert np.allclose(h[key_str((0, 0, 0), 0, 1)],
                       h[key_str((0, 0, 0), 1, 0)].T)    # hermitian
    delta = 1e-12

    def e_sign(a):
        return lr.blocks_diff_norm(h_lr(a), {k: -v for k, v in
                                             h_lr(-a).items()}) \
            / (lr.blocks_norm(h_lr(a)) + delta)

    def e_linear(a):
        h2, h1 = h_lr(2 * a), h_lr(a)
        return lr.blocks_diff_norm(h2, {k: 2.0 * v for k, v in h1.items()}) \
            / (2.0 * lr.blocks_norm(h1) + delta)

    assert e_sign(0.005) < e_sign(0.01) < e_sign(0.02)
    assert e_linear(0.005) < e_linear(0.01) < e_linear(0.02)
