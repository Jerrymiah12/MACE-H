import numpy as np

from mgo_lr import convert


def test_us_are_signed_permutations():
    for l in range(4):
        u = convert.orbital_u(l)
        assert u.shape == (2 * l + 1, 2 * l + 1)
        assert np.allclose(u @ u.T, np.eye(2 * l + 1))       # orthogonal
        assert np.allclose(np.abs(u).sum(axis=0), 1.0)       # permutation
        assert np.allclose(np.abs(u).sum(axis=1), 1.0)


def test_p_transform_hand_checked():
    # DeepH-pack: U[1] = eye(3)[[1,2,0]] with rows [0,1] negated, so an
    # ABACUS p-vector (a0, a1, a2) maps to (-a1, -a2, +a0).
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(convert.orbital_u(1) @ v, [-2.0, -3.0, 1.0])


def test_d_transform_hand_checked():
    # U[2] = eye(5)[[0,3,4,1,2]] with rows [3,4] negated:
    # (a0..a4) -> (a0, a3, a4, -a1, -a2)
    v = np.arange(5, dtype=float)
    assert np.allclose(convert.orbital_u(2) @ v, [0.0, 3.0, 4.0, -1.0, -2.0])


def test_atom_u_block_diagonal():
    u = convert.atom_u([0, 0, 1])
    assert u.shape == (5, 5)
    assert np.allclose(u[:2, :2], np.eye(2))
    assert np.allclose(u[2:, 2:], convert.orbital_u(1))
    assert np.allclose(u[:2, 2:], 0.0)


def test_transform_preserves_hermitian_pairs():
    rng = np.random.default_rng(3)
    li, lj = [0, 1], [1]
    a = rng.standard_normal((4, 3))          # H_ij(R)
    b = a.T.copy()                           # H_ji(-R) = H_ij(R)^T
    ta = convert.transform_block(a, li, lj)
    tb = convert.transform_block(b, lj, li)
    assert np.allclose(tb, ta.T)


def test_transform_involution_via_orthogonality():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((3, 3))
    t = convert.transform_block(a, [1], [1])
    u = convert.orbital_u(1)
    assert np.allclose(u.T @ t @ u, a)       # U^T (U a U^T) U = a
