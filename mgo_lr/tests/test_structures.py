import numpy as np

from mgo_lr.structures import Supercell, make_supercell, reciprocal, rocksalt_primitive


def test_rocksalt_primitive():
    a = 4.2
    cell, frac, species = rocksalt_primitive(a)
    assert species == ["Mg", "O"]
    # fcc primitive vectors a/2 (0,1,1) etc.
    assert np.allclose(np.abs(np.linalg.det(cell)), a**3 / 4.0)
    assert np.allclose(frac[0], [0.0, 0.0, 0.0])
    assert np.allclose(frac[1], [0.5, 0.5, 0.5])
    # nearest neighbour Mg-O distance is a/2
    cart = frac @ cell
    d = np.linalg.norm(cart[1] - cart[0] - cell[0])
    assert abs(min(np.linalg.norm(cart[1] - cart[0]), d) - a / 2.0) < 1e-12


def test_make_supercell_ordering():
    cell, frac, species = rocksalt_primitive(4.2)
    sc = make_supercell(cell, frac, species, 2)
    assert isinstance(sc, Supercell)
    assert len(sc.species) == 16
    assert sc.species[:8] == ["Mg"] * 8 and sc.species[8:] == ["O"] * 8
    assert sc.cell_index.shape == (16, 3)
    # first Mg at cell (0,0,0), second at (0,0,1) (np.ndindex order)
    assert tuple(sc.cell_index[0]) == (0, 0, 0)
    assert tuple(sc.cell_index[1]) == (0, 0, 1)
    assert np.allclose(sc.cell, 2 * cell)
    assert sc.basis_index[0] == 0 and sc.basis_index[8] == 1


def test_reciprocal():
    cell, _, _ = rocksalt_primitive(4.2)
    rec = reciprocal(cell)
    assert np.allclose(rec @ cell.T, 2 * np.pi * np.eye(3), atol=1e-12)
