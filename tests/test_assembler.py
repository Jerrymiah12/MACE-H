import numpy as np
import pytest
import torch

from maceh.epc.assembler import HoppingAssembler, _edge_key_strings
from maceh.epc.derivative import hermitize_blocks


class _FakeDatasetInfo:
    def __init__(self, spinful, index_to_Z, orbital_types):
        self.spinful = spinful
        self.index_to_Z = torch.tensor(index_to_Z)
        self.orbital_types = orbital_types


class _FakeNetOutInfo:
    def __init__(self, blocks, slices):
        self.blocks = blocks
        self.slices = slices


class _FakeTrainConfig:
    np_dtype = np.float64


class _FakeKernel:
    r''' just enough of DeepHE3Kernel for update_hopping and the assembler '''

    def __init__(self, dataset_info, net_out_info):
        self.dataset_info = dataset_info
        self.net_out_info = net_out_info
        self.train_config = _FakeTrainConfig()

    update_hopping = None  # bound below from the real implementation


from maceh.kernel import DeepHE3Kernel  # noqa: E402
_FakeKernel.update_hopping = DeepHE3Kernel.update_hopping


def _build_graph(seed=0, n_atoms=6, n_cells=3):
    r''' a directed edge set closed under (R, i, j) -> (-R, j, i), as radius graphs are '''
    rng = np.random.default_rng(seed)
    shifts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)][:n_cells]
    edges = set()
    for i in range(n_atoms):
        for j in range(n_atoms):
            for R in shifts:
                if rng.random() < 0.5:
                    edges.add((R[0], R[1], R[2], i + 1, j + 1))
                    edges.add((-R[0], -R[1], -R[2], j + 1, i + 1))
    edges = sorted(edges)
    edge_key = np.array(edges, dtype=np.int64)
    edge_index = np.stack([edge_key[:, 3] - 1, edge_key[:, 4] - 1])
    return edge_key, edge_index


def _species_setup(n_atoms, rng):
    # different orbital counts per species, so the destination blocks are ragged:
    # species 0 = s + p (4 orbitals), species 1 = s + p + p (7 orbitals)
    index_to_Z = [6, 79]
    orbital_types = [[0, 1], [0, 1, 1]]
    species = rng.integers(0, 2, size=n_atoms)
    info = _FakeDatasetInfo(False, index_to_Z, orbital_types)
    return info, species


def _net_out_info(info, which=None):
    r''' one target per equivariant orbital block, mirroring the real layout: within a
    target every species pair has the same block size, while different targets cover
    different sub-blocks and some orbitals stay uncovered (and therefore NaN) '''
    Z = [int(z) for z in info.index_to_Z]
    pairs = [f'{Z[a]} {Z[b]}' for a in range(2) for b in range(2)]
    # (row_lo, row_hi, col_lo, col_hi) sub-blocks common to every species pair
    subblocks = [(0, 1, 0, 1),      # s-s   1x1
                 (0, 1, 1, 4),      # s-p   1x3
                 (1, 4, 0, 1)]      # p-s   3x1
    if which is not None:
        subblocks = [subblocks[i] for i in which]
    blocks, slices, cursor = [], [0], 0
    for (r0, r1, c0, c1) in subblocks:
        blocks.append({p: [r0, r1, c0, c1] for p in pairs})
        cursor += (r1 - r0) * (c1 - c0)
        slices.append(cursor)
    return _FakeNetOutInfo(blocks, slices), cursor


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_assembler_matches_update_hopping_plus_hermitize(seed):
    rng = np.random.default_rng(seed + 100)
    n_atoms = 6
    edge_key, edge_index = _build_graph(seed=seed, n_atoms=n_atoms)
    info, species = _species_setup(n_atoms, rng)
    noi, n_out = _net_out_info(info)
    kernel = _FakeKernel(info, noi)
    contexts = [(kernel, None, None)]

    H_pred = rng.standard_normal((edge_key.shape[0], n_out))

    # reference: the general path
    ref = {}
    kernel.update_hopping(ref, H_pred, torch.tensor(species),
                          torch.tensor(edge_index), torch.tensor(edge_key))
    ref = hermitize_blocks(ref)

    # the vectorised plan
    asm = HoppingAssembler(contexts, edge_key, edge_index, species)
    assert asm.supported
    got = asm([H_pred])

    assert set(got) == set(ref)
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(got[k])
        assert a.shape == b.shape, k
        # NaN in the same places (orbitals no target block covers), equal elsewhere
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        m = ~np.isnan(a)
        assert np.array_equal(a[m], b[m]), f'{k}: not bit-for-bit identical'


def test_assembler_merges_multiple_models():
    r''' two models each predicting one target must merge exactly as update_hopping does '''
    rng = np.random.default_rng(7)
    n_atoms = 5
    edge_key, edge_index = _build_graph(seed=3, n_atoms=n_atoms)
    info, species = _species_setup(n_atoms, rng)
    noi_a, n_out_a = _net_out_info(info, which=[0])        # model A predicts s-s
    noi_b, n_out_b = _net_out_info(info, which=[1, 2])     # model B predicts s-p and p-s

    k_a = _FakeKernel(info, noi_a)
    k_b = _FakeKernel(info, noi_b)
    pred_a = rng.standard_normal((edge_key.shape[0], n_out_a))
    pred_b = rng.standard_normal((edge_key.shape[0], n_out_b))

    ref = {}
    k_a.update_hopping(ref, pred_a, torch.tensor(species),
                       torch.tensor(edge_index), torch.tensor(edge_key))
    k_b.update_hopping(ref, pred_b, torch.tensor(species),
                       torch.tensor(edge_index), torch.tensor(edge_key))
    ref = hermitize_blocks(ref)

    asm = HoppingAssembler([(k_a, None, None), (k_b, None, None)],
                           edge_key, edge_index, species)
    got = asm([pred_a, pred_b])
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(got[k])
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        m = ~np.isnan(a)
        assert np.array_equal(a[m], b[m]), k


def test_edge_key_strings_match_tolist_format():
    edge_key = np.array([[0, 0, 0, 1, 2], [-1, 2, -3, 4, 5]], dtype=np.int64)
    got = _edge_key_strings(edge_key)
    want = [str(torch.tensor(r).tolist()) for r in edge_key]
    assert got == want


def test_assembler_rejects_open_edge_set():
    r''' a graph missing a reverse hopping cannot be Hermitized; fail loudly '''
    edge_key = np.array([[0, 0, 0, 1, 2]], dtype=np.int64)
    edge_index = np.array([[0], [1]])
    info, _ = _species_setup(2, np.random.default_rng(0))
    noi, _ = _net_out_info(info)
    with pytest.raises(AssertionError, match='reverse hopping partner'):
        HoppingAssembler([(_FakeKernel(info, noi), None, None)],
                         edge_key, edge_index, np.array([0, 1]))
