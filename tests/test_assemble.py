import numpy as np
import h5py
import pytest

from maceh.epc.supercell import Structure
from maceh.epc.derivative import DerivativeData
from maceh.epc.assemble import compute_epc_cartesian, write_epc_cartesian_h5


def make_deriv():
    # blocks only for (kappa=0, alpha=0); alpha=1,2 empty
    blocks = {(0, 0): {((0, 0, 0), (0, 0, 0)): np.array([[1.0]]),
                       ((1, 0, 0), (2, 0, 0)): np.array([[0.5]])},
              (0, 1): {}, (0, 2): {}}
    return DerivativeData(n_grid=(2, 1, 1), n_uc_atoms=1, delta=0.01,
                          norb_cumsum=np.array([0, 1]), blocks=blocks)


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
