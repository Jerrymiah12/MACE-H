import numpy as np
import torch
import pytest

from maceh.epc.supercell import Structure
from maceh.epc.derivative import build_supercell_graph
from maceh.graph import get_edge_fea


def test_build_supercell_graph():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        struct = Structure(positions=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                           lattice=4.0 * np.eye(3),
                           numbers=np.array([79, 79]))
        data = build_supercell_graph(struct, radius=4.5, default_dtype_torch=torch.float64)

        assert data.x.dtype == torch.int64
        assert torch.equal(data.x, torch.tensor([79, 79]))
        assert data.edge_key.shape[1] == 5
        assert data.edge_key.shape[0] == data.edge_index.shape[1] == data.edge_attr.shape[0]
        # keys are 1-based
        assert data.edge_key[:, 3].min() >= 1 and data.edge_key[:, 4].min() >= 1
        # the self-check inside build_supercell_graph already asserted consistency;
        # verify the recomputation contract explicitly too
        recomputed = get_edge_fea(data.pos, data.lattice[0], torch.float64, data.edge_key)
        assert torch.allclose(recomputed, data.edge_attr, atol=1e-10)
        # directed graph: for every (i->j, R) there is (j->i, -R)
        keys = set(map(tuple, data.edge_key.tolist()))
        for (r1, r2, r3, i, j) in keys:
            assert (-r1, -r2, -r3, j, i) in keys
    finally:
        torch.set_default_dtype(prev)
