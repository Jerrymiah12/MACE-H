import numpy as np
import h5py
import pytest

from maceh.epc.supercell import Structure
from maceh.epc.derivative import DerivativeData
from maceh.epc.store import H5DerivativeStore
from maceh.epc.assemble import compute_epc_cartesian, write_epc_cartesian_h5


def make_deriv():
    # blocks only for (kappa=0, alpha=0); alpha=1,2 empty
    blocks = {(0, 0): {((0, 0, 0), (0, 0, 0)): np.array([[1.0]]),
                       ((1, 0, 0), (2, 0, 0)): np.array([[0.5]])},
              (0, 1): {}, (0, 2): {}}
    return DerivativeData(n_grid=(2, 1, 1), n_uc_atoms=1, delta=0.01,
                          norb_cumsum=np.array([0, 1]), blocks=blocks)


def test_derivativedata_accessor_interface():
    deriv = make_deriv()
    assert set(deriv.pairs()) == {(0, 0), (0, 1), (0, 2)}
    g0 = deriv.group(0, 0)
    assert g0[((0, 0, 0), (0, 0, 0))][0, 0] == pytest.approx(1.0)
    assert g0[((1, 0, 0), (2, 0, 0))][0, 0] == pytest.approx(0.5)
    assert deriv.group(0, 1) == {}
    assert deriv.group(5, 0) == {}  # absent pair -> empty, never KeyError


def test_compute_epc_cartesian_phases():
    deriv = make_deriv()
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    qpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    res = compute_epc_cartesian(deriv, kpts, qpts)
    g = res['g']
    assert g.shape == (2, 2, 1, 3, 1, 1)
    assert np.array_equal(res['atom_indices'], [0])
    # k=0, q=0: 1 + 0.5
    assert g[0, 0, 0, 0, 0, 0] == pytest.approx(1.5)
    # k=0, q=(1/2,0,0): 1 + 0.5 * exp(2pi i * 0.5 * 1) = 1 - 0.5
    assert g[0, 1, 0, 0, 0, 0] == pytest.approx(0.5)
    # k=(1/2,0,0), q=0: 1 + 0.5 * exp(2pi i * 0.5 * 2) = 1.5
    assert g[1, 0, 0, 0, 0, 0] == pytest.approx(1.5)
    # k=(1/2,0,0), q=(1/2,0,0): 1 + 0.5 * (-1) * (1) = 0.5
    assert g[1, 1, 0, 0, 0, 0] == pytest.approx(0.5)
    # untouched directions are zero
    assert np.all(g[:, :, :, 1:, :, :] == 0)


def test_write_epc_cartesian_h5(tmp_path):
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    qpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    res = compute_epc_cartesian(deriv, kpts, qpts)
    path = str(tmp_path / 'epc_cartesian_pred.h5')
    write_epc_cartesian_h5(path, struct, deriv, kpts, qpts,
                           {'units': 'g in eV/Angstrom', 'spinful': False})
    with h5py.File(path, 'r') as f:
        assert f['g_real'].shape == (2, 2, 1, 3, 1, 1)
        assert f['g_imag'].shape == (2, 2, 1, 3, 1, 1)
        # streamed output equals the in-memory computation
        assert np.allclose(f['g_real'][()] + 1j * f['g_imag'][()], res['g'])
        assert f['g_real'][0, 0, 0, 0, 0, 0] == pytest.approx(1.5)
        assert np.array_equal(f['atomic_numbers'][()], [79])
        assert np.array_equal(f['atom_indices'][()], [0])
        assert [d.decode() for d in f['cartesian_directions'][()]] == ['x', 'y', 'z']
        assert np.array_equal(f['orbital_indices'][()], [0])
        assert np.allclose(f['lattice'][()], 3.0 * np.eye(3))
        assert np.allclose(f['supercell_matrix'][()], np.diag([2, 1, 1]))
        assert f['finite_difference_delta'][()] == pytest.approx(0.01)
        assert f.attrs['units'] == 'g in eV/Angstrom'
        assert 'dH' not in f
    assert list(tmp_path.glob('*.tmp')) == []


def test_write_epc_cartesian_h5_saves_derivatives(tmp_path):
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    path = str(tmp_path / 'epc_cartesian_pred.h5')
    write_epc_cartesian_h5(path, struct, deriv, np.zeros((1, 3)), np.zeros((1, 3)),
                           {'units': 'g in eV/Angstrom'}, save_derivatives=True)
    with h5py.File(path, 'r') as f:
        assert f['dH/0/x/[0, 0, 0, 0, 0, 0]'][()] == pytest.approx(1.0)
        assert f['dH/0/x/[1, 0, 0, 2, 0, 0]'][()] == pytest.approx(0.5)


class CountingDeriv:
    r''' wraps a derivative source and counts group() fetches, standing in for the
    per-fetch file read an H5DerivativeStore does '''

    def __init__(self, inner):
        self.inner = inner
        self.fetches = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def group(self, kappa, alpha):
        self.fetches += 1
        return self.inner.group(kappa, alpha)


def test_write_epc_fetches_each_group_once(tmp_path):
    # a q-major transform refetched the whole store per q-point; one fetch per
    # (kappa, alpha) is the invariant that keeps stage 2 off the disk
    deriv = CountingDeriv(make_deriv())
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    qpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    write_epc_cartesian_h5(str(tmp_path / 'epc.h5'), struct, deriv,
                           np.zeros((2, 3)), qpts, {}, save_derivatives=True)
    assert deriv.fetches == len(deriv.pairs()) == 3


def test_compute_epc_cartesian_fetches_each_group_once():
    deriv = CountingDeriv(make_deriv())
    compute_epc_cartesian(deriv, np.zeros((2, 3)), np.zeros((4, 3)))
    assert deriv.fetches == 3


def test_backends_agree_bitwise_under_cancellation(tmp_path):
    # summation order is the whole point: 1e-16 + 1e-16 + 1.0 != 1.0 + 1e-16 + 1e-16,
    # and HDF5 link order used to disagree with in-memory insertion order
    keys = [((0, 0, 0), (10, 0, 0)), ((0, 0, 0), (2, 0, 0)), ((0, 0, 0), (-1, 0, 0))]
    vals = [1e-16, 1e-16, 1.0]
    path = str(tmp_path / 'dH.h5')
    with h5py.File(path, 'w') as f:
        f['n_grid'] = np.array([2, 1, 1], dtype=int)
        f['n_uc_atoms'] = 1
        f['delta'] = 0.01
        f['norb_cumsum'] = np.array([0, 1])
        for (p, R), v in zip(keys, vals):
            f[f'dH/0/x/{str(list(p) + list(R))}'] = np.array([[v]])
    store = H5DerivativeStore(path)
    mem = DerivativeData(n_grid=(2, 1, 1), n_uc_atoms=1, delta=0.01,
                         norb_cumsum=np.array([0, 1]),
                         blocks={(0, 0): {k: np.array([[v]]) for k, v in zip(keys, vals)},
                                 (0, 1): {}, (0, 2): {}})
    # guard against a vacuous test: these values really are order-sensitive
    assert vals[0] + vals[1] + vals[2] != vals[2] + vals[1] + vals[0]
    g_mem = compute_epc_cartesian(mem, np.zeros((1, 3)), np.zeros((1, 3)))['g']
    g_h5 = compute_epc_cartesian(store, np.zeros((1, 3)), np.zeros((1, 3)))['g']
    assert np.array_equal(g_mem, g_h5)
    # canonical (p, R) order is -1 < 2 < 10, i.e. 1.0 + 1e-16 + 1e-16
    assert g_mem[0, 0, 0, 0, 0, 0] == 1.0


def test_write_epc_k_batching_is_bitwise_transparent(tmp_path, monkeypatch):
    # batching only splits the k axis, so it must not perturb a single bit
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    rng = np.random.default_rng(0)
    kpts, qpts = rng.random((7, 3)), rng.random((3, 3))
    ref = compute_epc_cartesian(deriv, kpts, qpts)['g']
    for batch_bytes in (16 * 1024 ** 2, 1):    # unbatched, then one k-point per batch
        monkeypatch.setattr('maceh.epc.assemble.BATCH_BYTES', batch_bytes)
        path = str(tmp_path / f'epc_{batch_bytes}.h5')
        write_epc_cartesian_h5(path, struct, deriv, kpts, qpts, {})
        with h5py.File(path, 'r') as f:
            assert f['g_real'].chunks[0] == (7 if batch_bytes > 1 else 1)
            assert np.array_equal(f['g_real'][()] + 1j * f['g_imag'][()], ref)


def test_write_epc_chunks_do_not_span_q_or_atom(tmp_path):
    # partially filled chunks would make every hyperslab write a read-modify-write
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    path = str(tmp_path / 'epc.h5')
    write_epc_cartesian_h5(path, struct, deriv, np.zeros((5, 3)), np.zeros((4, 3)), {})
    with h5py.File(path, 'r') as f:
        assert f['g_real'].chunks[1:4] == (1, 1, 1)
        assert f['g_imag'].chunks[1:4] == (1, 1, 1)


def test_write_epc_cartesian_h5_failure_leaves_no_partial_file(tmp_path):
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    path = str(tmp_path / 'epc_cartesian_pred.h5')
    # an existing result must survive a failed rewrite
    with h5py.File(path, 'w') as f:
        f['sentinel'] = 1
    with pytest.raises(TypeError):
        # dict attrs cannot be stored by h5py -> write fails mid-file
        write_epc_cartesian_h5(path, struct, deriv, np.zeros((1, 3)), np.zeros((1, 3)),
                               {'bad': {'nested': 'dict'}})
    with h5py.File(path, 'r') as f:
        assert f['sentinel'][()] == 1
    assert list(tmp_path.glob('*.tmp')) == []
