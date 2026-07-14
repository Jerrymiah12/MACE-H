import numpy as np
import torch
import pytest

from maceh.epc.supercell import SupercellMap, fold_key
from maceh.epc.derivative import (DerivativeData, finite_difference,
                                  acoustic_sum_rule, hermitize_blocks)
from maceh.epc.assemble import compute_epc_cartesian

A = 3.0  # unit-cell lattice constant along x
L_SC = np.diag([2 * A, 10.0, 10.0])  # 2x1x1 supercell lattice (rows)
# supercell hopping keys [R'x, R'y, R'z, I, J], 1-based, directed, incl. onsite
KEYS = [[0, 0, 0, 1, 1], [0, 0, 0, 2, 2],
        [0, 0, 0, 1, 2], [0, 0, 0, 2, 1],
        [-1, 0, 0, 1, 2], [1, 0, 0, 2, 1],
        [1, 0, 0, 1, 2], [-1, 0, 0, 2, 1],
        [1, 0, 0, 1, 1], [-1, 0, 0, 1, 1],
        [1, 0, 0, 2, 2], [-1, 0, 0, 2, 2],
        [2, 0, 0, 2, 1], [-2, 0, 0, 1, 2]]


def edge_vec(pos, key):
    R = np.array(key[:3], dtype=np.float64)
    return pos[key[4] - 1] + R @ L_SC - pos[key[3] - 1]


def predict_fn(positions):
    pos = positions.detach().numpy()
    return {str(k): np.array([[np.linalg.norm(edge_vec(pos, k))]]) for k in KEYS}


def analytic_deriv(pos, key, kappa_sc, alpha):
    # d|v|/d tau_{kappa,alpha} where v = r_J + R'.L - r_I; displacing a supercell
    # atom moves only that atom (its supercell periodic images are other sc atoms)
    v = edge_vec(pos, key)
    n = np.linalg.norm(v)
    if n == 0:
        return 0.0
    d = 0.0
    if key[4] - 1 == kappa_sc:
        d += v[alpha] / n
    if key[3] - 1 == kappa_sc:
        d -= v[alpha] / n
    return d


def test_finite_difference_matches_analytic():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    norb_cumsum = np.array([0, 1])
    deriv = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                              grad_threshold=1e-12)
    assert deriv.n_uc_atoms == 1 and deriv.norb_tot == 1
    # displacing uc atom 0 displaces supercell atom 0 (home cell)
    pos_np = pos0.numpy()
    for alpha in range(3):
        found = deriv.blocks[(0, alpha)]
        for key in KEYS:
            expected = analytic_deriv(pos_np, key, kappa_sc=0, alpha=alpha)
            p, R, i, j = fold_key(key, smap)
            got = found.get((p, R), np.zeros((1, 1)))[i, j]
            assert got == pytest.approx(expected, abs=1e-6), (key, alpha)


def test_grad_threshold_drops_far_blocks():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    deriv = finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=1e-4,
                              grad_threshold=1e30)
    assert all(len(v) == 0 for v in deriv.blocks.values())


def test_acoustic_sum_rule_zero_for_translation_invariant_model():
    # the stub depends only on relative positions, so the sum rule is exact
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    deriv = finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=1e-4,
                              grad_threshold=1e-12)
    assert acoustic_sum_rule(deriv) < 1e-6


def test_atom_indices_out_of_range_rejected():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=1e-4,
                          atom_indices=[1])


def test_delta_zero_rejected():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=0.0)


def test_hermitize_blocks():
    H = {'[1, 0, 0, 1, 1]': np.array([[2.0]]),
         '[-1, 0, 0, 1, 1]': np.array([[4.0]]),
         '[0, 0, 0, 1, 1]': np.array([[1.0]])}
    out = hermitize_blocks(H)
    assert out['[1, 0, 0, 1, 1]'][0, 0] == pytest.approx(3.0)
    assert out['[-1, 0, 0, 1, 1]'][0, 0] == pytest.approx(3.0)
    assert out['[0, 0, 0, 1, 1]'][0, 0] == pytest.approx(1.0)


def test_hermitize_blocks_missing_partner():
    with pytest.raises(AssertionError):
        hermitize_blocks({'[1, 0, 0, 1, 1]': np.array([[2.0]])})


def test_g_hermiticity_invariant():
    # differentiating a Hermitian-consistent H must give g(k, q)^dagger = g(k+q, -q);
    # q must be commensurate with the supercell grid, k is arbitrary. Exercises the
    # full hermitize -> FD -> fold-back -> Fourier chain.
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    deriv = finite_difference(lambda pos: hermitize_blocks(predict_fn(pos)),
                              pos0, smap, np.array([0, 1]), delta=1e-4,
                              grad_threshold=1e-12)
    k = np.array([0.3, 0.0, 0.0])
    q = np.array([0.5, 0.0, 0.0])
    res = compute_epc_cartesian(deriv, kpts=[k, k + q], qpts=[q, -q])
    g = res['g']  # shape (2, 2, 1, 3, 1, 1)
    for alpha in range(3):
        assert np.allclose(g[0, 0, 0, alpha].conj().T, g[1, 1, 0, alpha], atol=1e-8), alpha
