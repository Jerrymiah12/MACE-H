"""ABACUS -> DeepH-E3/MACE-H data conversion.

The per-l orbital permutation/sign table is copied verbatim from
DeepH-pack deeph/preprocess/abacus_get_data.py (class OrbAbacus2DeepH).
A silent error here corrupts every matrix — the table is pinned by
tests/test_orbital_reorder.py and must not be re-derived.
"""
import numpy as np
from scipy.linalg import block_diag


def _build_us():
    us = {0: np.eye(1),
          1: np.eye(3)[[1, 2, 0]],
          2: np.eye(5)[[0, 3, 4, 1, 2]],
          3: np.eye(7)}
    minus = {1: [0, 1], 2: [3, 4], 3: [1, 2, 5, 6]}
    for l, rows in minus.items():
        us[l][rows] *= -1.0
    return us


_U_ABACUS2DEEPH = _build_us()


def orbital_u(l):
    if l not in _U_ABACUS2DEEPH:
        raise NotImplementedError(f"only l <= 3 supported, got l={l}")
    return _U_ABACUS2DEEPH[l]


def atom_u(orbital_types_atom):
    """Block-diagonal transform for one atom's full AO set."""
    return block_diag(*[orbital_u(l) for l in orbital_types_atom])


def transform_block(mat, l_left, l_right):
    """U_i @ mat @ U_j.T for an atom-pair block."""
    return atom_u(l_left) @ np.asarray(mat, float) @ atom_u(l_right).T
