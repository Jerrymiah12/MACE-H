import json
from dataclasses import dataclass

import numpy as np
import torch

from ..graph import get_graph, get_edge_fea
from .supercell import fold_key


def build_supercell_graph(struct, radius, default_dtype_torch):
    r''' radius-based graph for an in-memory structure. Returns a Data object with
    x = atomic numbers, edge_attr = [dist, dx, dy, dz] and edge_key = [Rx,Ry,Rz,i,j]
    (1-based). Asserts that edge_attr can be recomputed exactly from positions and
    edge_key -- the invariant the finite-difference driver relies on. '''
    assert radius > 0, 'EPC graph construction requires an explicit cutoff radius'
    lattice = torch.tensor(struct.lattice, dtype=default_dtype_torch)
    cart_coords = torch.tensor(struct.positions, dtype=default_dtype_torch)
    frac_coords = cart_coords @ torch.linalg.inv(lattice)
    numbers = torch.tensor(struct.numbers, dtype=torch.int64)
    data = get_graph(cart_coords, frac_coords, numbers, stru_id='epc_supercell',
                     r=radius, max_num_nbr=0, edge_Aij=False, lattice=lattice,
                     default_dtype_torch=default_dtype_torch, data_folder=None,
                     target_file_name='overlaps.h5', inference=True, only_ij=False,
                     create_from_DFT=False)
    recomputed = get_edge_fea(data.pos, data.lattice[0], default_dtype_torch, data.edge_key)
    assert torch.allclose(recomputed, data.edge_attr, atol=1e-7), \
        'edge_attr recomputed from positions does not match graph construction'
    return data
