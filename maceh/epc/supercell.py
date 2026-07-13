import os
import itertools
from collections import namedtuple

import numpy as np

# positions: (N, 3) cartesian Angstrom; lattice: (3, 3) rows are lattice vectors;
# numbers: (N,) atomic numbers
Structure = namedtuple('Structure', ['positions', 'lattice', 'numbers'])


def load_structure(structure_dir):
    positions = np.loadtxt(os.path.join(structure_dir, 'site_positions.dat')).T
    numbers = np.loadtxt(os.path.join(structure_dir, 'element.dat'))
    lattice = np.loadtxt(os.path.join(structure_dir, 'lat.dat')).T
    if numbers.ndim == 0:
        numbers = numbers[None]
        positions = positions[None, :]
    return Structure(positions.astype(np.float64), lattice.astype(np.float64),
                     numbers.astype(int))


class SupercellMap:
    r''' cell-major ordering: supercell atom index = cell_lin(p) * n_uc_atoms + i '''

    def __init__(self, n_grid, n_uc_atoms):
        self.n_grid = tuple(int(n) for n in n_grid)
        self.n_uc_atoms = int(n_uc_atoms)
        self.cells = list(itertools.product(range(self.n_grid[0]),
                                            range(self.n_grid[1]),
                                            range(self.n_grid[2])))

    @property
    def n_cells(self):
        return len(self.cells)

    def cell_lin(self, p):
        return (p[0] * self.n_grid[1] + p[1]) * self.n_grid[2] + p[2]

    def sc_index(self, i, p):
        return self.cell_lin(p) * self.n_uc_atoms + i

    def uc_of(self, sc_i):
        return sc_i % self.n_uc_atoms, self.cells[sc_i // self.n_uc_atoms]


def build_supercell(struct, n_grid):
    smap = SupercellMap(n_grid, len(struct.numbers))
    sc_lattice = struct.lattice * np.array(smap.n_grid, dtype=np.float64)[:, None]
    positions, numbers = [], []
    for p in smap.cells:
        shift = np.array(p, dtype=np.float64) @ struct.lattice
        positions.append(struct.positions + shift)
        numbers.append(struct.numbers)
    return Structure(np.concatenate(positions), sc_lattice, np.concatenate(numbers)), smap


def fold_key(key, smap):
    r''' fold supercell hopping key [Rx, Ry, Rz, I, J] (I, J 1-based; R in supercell
    lattice units) into unit-cell labels, for displacement of a home-cell atom.
    Returns (p, R, i, j): p = cell of the displaced atom relative to the bra atom's
    cell (reduced mod n_grid, exact for q commensurate with the grid); R = bra->ket
    offset in unit-cell lattice units; i, j = 0-based unit-cell atom indices '''
    n = smap.n_grid
    i, p_i = smap.uc_of(key[3] - 1)
    j, p_j = smap.uc_of(key[4] - 1)
    R = tuple(p_j[a] + n[a] * key[a] - p_i[a] for a in range(3))
    p = tuple((-p_i[a]) % n[a] for a in range(3))
    return p, R, i, j


def uniform_grid(n_grid):
    return np.array([[p[0] / n_grid[0], p[1] / n_grid[1], p[2] / n_grid[2]]
                     for p in itertools.product(range(n_grid[0]), range(n_grid[1]),
                                                range(n_grid[2]))])
