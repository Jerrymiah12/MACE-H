import numpy as np
import torch
import pytest

from maceh.epc.supercell import SupercellMap
from maceh.epc.derivative import finite_difference, stream_finite_difference
from maceh.epc.assemble import compute_epc_cartesian
from tests.test_derivative_fd import predict_fn, A


def _smap_pos():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    return smap, pos0


def test_streamed_groups_equal_in_memory(tmp_path):
    smap, pos0 = _smap_pos()
    norb_cumsum = np.array([0, 1])
    mem = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                            grad_threshold=1e-12)
    store = stream_finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                                     out_path=str(tmp_path / 'dH.h5'),
                                     grad_threshold=1e-12)
    assert set(store.pairs()) == set(mem.pairs())
    for kappa, alpha in mem.pairs():
        gm, gs = mem.group(kappa, alpha), store.group(kappa, alpha)
        assert set(gm) == set(gs)
        for key in gm:
            assert np.allclose(gm[key], gs[key], atol=0, rtol=0)


def test_duplicate_atom_indices_are_deduplicated(tmp_path):
    # a repeated index used to collide on the dH dataset name and raise OSError
    smap, pos0 = _smap_pos()
    norb_cumsum = np.array([0, 1])
    store = stream_finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                                     out_path=str(tmp_path / 'dH.h5'),
                                     atom_indices=[0, 0], grad_threshold=1e-12)
    mem = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                            atom_indices=[0, 0], grad_threshold=1e-12)
    assert set(store.pairs()) == set(mem.pairs()) == {(0, 0), (0, 1), (0, 2)}


def test_store_nbytes_matches_in_memory(tmp_path):
    smap, pos0 = _smap_pos()
    norb_cumsum = np.array([0, 1])
    mem = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                            grad_threshold=1e-12)
    store = stream_finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                                     out_path=str(tmp_path / 'dH.h5'),
                                     grad_threshold=1e-12)
    assert store.nbytes() == mem.nbytes() > 0


def test_streamed_pipeline_reproduces_g(tmp_path):
    # spec step 12: end-to-end g must be identical from either derivative source
    smap, pos0 = _smap_pos()
    norb_cumsum = np.array([0, 1])
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    qpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mem = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                            grad_threshold=1e-12)
    store = stream_finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                                     out_path=str(tmp_path / 'dH.h5'),
                                     grad_threshold=1e-12)
    g_mem = compute_epc_cartesian(mem, kpts, qpts)['g']
    g_str = compute_epc_cartesian(store, kpts, qpts)['g']
    assert np.array_equal(g_mem, g_str)
