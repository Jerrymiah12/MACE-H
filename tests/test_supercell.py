import numpy as np
import pytest

from maceh.epc.supercell import (Structure, SupercellMap, build_supercell,
                                 fold_key, uniform_grid, load_structure)


def make_uc():
    return Structure(positions=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                     lattice=4.0 * np.eye(3),
                     numbers=np.array([79, 79]))


def test_supercell_map_roundtrip():
    smap = SupercellMap((2, 3, 1), n_uc_atoms=2)
    assert smap.n_cells == 6
    assert smap.cells[0] == (0, 0, 0)
    for i in range(2):
        for p in smap.cells:
            sc = smap.sc_index(i, p)
            assert smap.uc_of(sc) == (i, p)
    # home cell atoms come first
    assert smap.sc_index(0, (0, 0, 0)) == 0
    assert smap.sc_index(1, (0, 0, 0)) == 1


def test_build_supercell():
    sc, smap = build_supercell(make_uc(), (2, 1, 1))
    assert sc.positions.shape == (4, 3)
    assert np.allclose(sc.lattice, np.diag([8.0, 4.0, 4.0]))
    assert np.array_equal(sc.numbers, [79, 79, 79, 79])
    # atom 1 in cell (1,0,0) sits at uc position + a1
    idx = smap.sc_index(1, (1, 0, 0))
    assert np.allclose(sc.positions[idx], [6.0, 2.0, 2.0])


def test_fold_key():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=2)
    # bra atom: uc atom 1 in cell (1,0,0)  -> sc index 3 -> 1-based 4
    # ket atom: uc atom 0 in cell (0,0,0)  -> sc index 0 -> 1-based 1
    # supercell image shift R' = (1, 0, 0)
    p, R, i, j = fold_key([1, 0, 0, 4, 1], smap)
    assert (i, j) == (1, 0)
    # R = p_j + n*R' - p_i = (0 + 2*1 - 1, 0, 0)
    assert R == (1, 0, 0)
    # p = -p_i mod n = (-1) % 2 = 1
    assert p == (1, 0, 0)


def test_uniform_grid():
    g = uniform_grid((2, 1, 2))
    assert g.shape == (4, 3)
    assert np.allclose(g[0], [0, 0, 0])
    assert np.allclose(g[1], [0, 0, 0.5])
    assert np.allclose(g[2], [0.5, 0, 0])


def test_load_structure(tmp_path):
    # files use the DeepH convention: columns are atoms / lattice vectors
    np.savetxt(tmp_path / 'site_positions.dat', np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]))
    np.savetxt(tmp_path / 'element.dat', np.array([79.0, 79.0]))
    np.savetxt(tmp_path / 'lat.dat', 4.0 * np.eye(3))
    s = load_structure(str(tmp_path))
    assert s.positions.shape == (2, 3)
    assert np.allclose(s.positions[1], [2.0, 2.0, 2.0])
    assert np.allclose(s.lattice, 4.0 * np.eye(3))
    assert np.array_equal(s.numbers, [79, 79])
