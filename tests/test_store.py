import json

import numpy as np
import h5py
import pytest

from maceh.epc.store import H5DerivativeStore


def write_fixture(path):
    with h5py.File(path, 'w') as f:
        f['n_grid'] = np.array([2, 1, 1], dtype=int)
        f['n_uc_atoms'] = 1
        f['delta'] = 0.01
        f['norb_cumsum'] = np.array([0, 1])
        # (kappa=0, alpha=0) has two (p, R) blocks; alpha=1 has one; alpha=2 none
        f['dH/0/x/[0, 0, 0, 0, 0, 0]'] = np.array([[1.0]])
        f['dH/0/x/[1, 0, 0, 2, 0, 0]'] = np.array([[0.5]])
        f['dH/0/y/[0, 0, 0, 0, 0, 0]'] = np.array([[2.0]])


def test_group_order_is_canonical_not_hdf5_link_order(tmp_path):
    # HDF5 iterates links lexicographically by name, so '[..., 10, 0, 0]' comes back
    # before '[..., 2, 0, 0]'; group() must undo that or the non-associative Fourier
    # sum lands on different bits than the in-memory backend
    path = str(tmp_path / 'dH.h5')
    keys = [((0, 0, 0), (10, 0, 0)), ((0, 0, 0), (2, 0, 0)), ((0, 0, 0), (-1, 0, 0))]
    with h5py.File(path, 'w') as f:
        f['n_grid'] = np.array([2, 1, 1], dtype=int)
        f['n_uc_atoms'] = 1
        f['delta'] = 0.01
        f['norb_cumsum'] = np.array([0, 1])
        for p, R in keys:
            f[f'dH/0/x/{str(list(p) + list(R))}'] = np.array([[1.0]])
        assert list(f['dH/0/x']) == ['[0, 0, 0, -1, 0, 0]', '[0, 0, 0, 10, 0, 0]',
                                     '[0, 0, 0, 2, 0, 0]']
    assert list(H5DerivativeStore(path).group(0, 0)) == sorted(keys)


def test_store_metadata(tmp_path):
    path = str(tmp_path / 'dH.h5')
    write_fixture(path)
    store = H5DerivativeStore(path)
    assert store.n_grid == (2, 1, 1)
    assert store.n_uc_atoms == 1
    assert store.delta == pytest.approx(0.01)
    assert store.norb_tot == 1


def test_store_pairs_and_groups(tmp_path):
    path = str(tmp_path / 'dH.h5')
    write_fixture(path)
    store = H5DerivativeStore(path)
    assert set(store.pairs()) == {(0, 0), (0, 1)}
    g = store.group(0, 0)
    assert g[((0, 0, 0), (0, 0, 0))][0, 0] == pytest.approx(1.0)
    assert g[((1, 0, 0), (2, 0, 0))][0, 0] == pytest.approx(0.5)
    assert store.group(0, 1)[((0, 0, 0), (0, 0, 0))][0, 0] == pytest.approx(2.0)
    assert store.group(0, 2) == {}   # absent alpha -> empty
    assert store.group(9, 0) == {}   # absent kappa -> empty
