from dataclasses import dataclass

import numpy as np


def rocksalt_primitive(a):
    """Rocksalt MgO primitive cell. Rows of `cell` are lattice vectors (Å)."""
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0]])
    frac = np.array([[0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5]])
    return cell, frac, ["Mg", "O"]


@dataclass
class Supercell:
    cell: np.ndarray        # (3,3) rows = supercell lattice vectors, Å
    cart: np.ndarray        # (N,3) Cartesian positions, Å
    species: list           # length N
    cell_index: np.ndarray  # (N,3) int, primitive-cell offset n_l of each atom
    basis_index: np.ndarray # (N,) int, index into the primitive basis


def make_supercell(cell, frac, species, n):
    """n x n x n supercell, species-major then cell-minor atom ordering."""
    cart_prim = frac @ cell
    cells = np.array(list(np.ndindex(n, n, n)))          # (n^3, 3)
    pos, spec, cidx, bidx = [], [], [], []
    for b, s in enumerate(species):
        for c in cells:
            pos.append(cart_prim[b] + c @ cell)
            spec.append(s)
            cidx.append(c)
            bidx.append(b)
    return Supercell(cell=n * np.asarray(cell, float),
                     cart=np.array(pos),
                     species=spec,
                     cell_index=np.array(cidx, dtype=int),
                     basis_index=np.array(bidx, dtype=int))


def reciprocal(cell):
    """Reciprocal lattice, rows b_i, includes the 2*pi factor. 1/Å."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
