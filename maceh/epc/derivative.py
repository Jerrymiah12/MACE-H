import json
from dataclasses import dataclass

import numpy as np
import torch
import h5py

from ..graph import get_graph, get_edge_fea
from .supercell import fold_key
from .store import H5DerivativeStore, canonical_group


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


@dataclass
class DerivativeData:
    r''' real-space Hamiltonian derivatives dH_ij(R)/d tau_{kappa,alpha}(p).
    blocks[(kappa, alpha)][(p, R)] is a dense (norb_tot, norb_tot) unit-cell matrix;
    p labels the cell of the displaced atom relative to the bra atom's cell,
    R the bra->ket cell offset (both in unit-cell lattice units). Units: eV / Angstrom. '''
    n_grid: tuple
    n_uc_atoms: int
    delta: float
    norb_cumsum: np.ndarray
    blocks: dict

    @property
    def norb_tot(self):
        return int(self.norb_cumsum[-1])

    def pairs(self):
        r''' (kappa, alpha) pairs that have a stored derivative group '''
        return list(self.blocks.keys())

    def group(self, kappa, alpha):
        r''' {(p, R): dense (norb_tot, norb_tot)} for one (kappa, alpha); {} if absent.
        Canonically ordered, so the Fourier sum is bit-for-bit reproducible against
        an H5DerivativeStore over the same derivatives. '''
        return canonical_group(self.blocks.get((kappa, alpha), {}))

    def nbytes(self):
        r''' total size of the stored derivative blocks in bytes '''
        return sum(m.nbytes for grp in self.blocks.values() for m in grp.values())


class _FoldPlan:
    r''' cache of fold_key results for one supercell map.

    fold_key depends only on the hopping key and the supercell shape, so over a
    finite-difference sweep it returns the same answer for the same key a few
    hundred times. Parsing the key with json.loads each time is what makes that
    expensive, so memoise per (key, grid) instead. '''

    def __init__(self, smap):
        self.n_grid = smap.n_grid
        self.n_uc_atoms = smap.n_uc_atoms
        self._smap = smap
        self._cache = {}

    def matches(self, smap):
        return smap.n_grid == self.n_grid and smap.n_uc_atoms == self.n_uc_atoms

    def get(self, key_str):
        hit = self._cache.get(key_str)
        if hit is None:
            hit = fold_key(json.loads(key_str), self._smap)
            self._cache[key_str] = hit
        return hit


_FOLD_PLAN = None


def _fold_plan_for(smap):
    global _FOLD_PLAN
    if _FOLD_PLAN is None or not _FOLD_PLAN.matches(smap):
        _FOLD_PLAN = _FoldPlan(smap)
    return _FOLD_PLAN


def hermitize_blocks(H):
    r''' H_ij(R) <- (H_ij(R) + H_ji(-R)^dagger) / 2, matching the symmetrization the
    band-structure postprocessing applies to predicted Hamiltonians (Band.py,
    force_hermiticity=True). Differentiating the symmetrized Hamiltonian keeps
    g(k,q)^dagger = g(k+q,-q). Requires the directed edge set to be closed under
    (R, i, j) -> (-R, j, i), which radius-based graphs guarantee. '''
    out = {}
    for key_str, v in H.items():
        key = json.loads(key_str)
        adj = str([-key[0], -key[1], -key[2], key[4], key[3]])
        assert adj in H, f'missing reverse hopping partner for {key_str}'
        out[key_str] = (np.asarray(v) + np.asarray(H[adj]).conj().T) / 2.0
    return out


def _displacement_targets(atom_indices, smap):
    r''' 0-based unit-cell atom indices to displace: every atom when unspecified,
    otherwise the requested ones with duplicates dropped. A repeated index only
    repeats six forward passes, and collides on the streamed HDF5 dataset name. '''
    if atom_indices is None:
        return list(range(smap.n_uc_atoms))
    assert all(0 <= kappa < smap.n_uc_atoms for kappa in atom_indices), \
        f'atom_indices must be 0-based unit-cell atom indices in [0, {smap.n_uc_atoms})'
    return list(dict.fromkeys(atom_indices))


def finite_difference_pair(predict_fn, positions0, smap, norb_cumsum, delta, kappa, alpha,
                           grad_threshold=1e-10):
    r''' central difference of hopping blocks w.r.t. one (kappa, alpha)
    displacement, folded to unit-cell labels. Returns {(p, R): dense}. Public so
    callers that only need a single direction never materialise a whole
    DerivativeData just to read one group out of it. '''
    assert np.isfinite(delta) and delta > 0, \
        'delta must be a positive finite displacement (Angstrom)'
    norb_cumsum = np.asarray(norb_cumsum)
    norb_tot = int(norb_cumsum[-1])
    pos_plus = positions0.clone()
    pos_plus[kappa, alpha] += delta
    pos_minus = positions0.clone()
    pos_minus[kappa, alpha] -= delta
    H_plus = predict_fn(pos_plus)
    H_minus = predict_fn(pos_minus)
    assert H_plus.keys() == H_minus.keys()
    plan = _fold_plan_for(smap)
    out = {}
    for key_str, hp in H_plus.items():
        d = (np.asarray(hp) - np.asarray(H_minus[key_str])) / (2.0 * delta)
        # explicit raise, not assert: must survive python -O
        if not np.isfinite(d).all():
            raise FloatingPointError(
                f'nonfinite derivative for hopping {key_str} (atom {kappa}, '
                f'direction {"xyz"[alpha]}): the model produced nonfinite output')
        if np.abs(d).max() < grad_threshold:
            continue
        p, R, i, j = plan.get(key_str)
        if (p, R) not in out:
            dtype = np.complex128 if np.iscomplexobj(d) else np.float64
            out[(p, R)] = np.zeros((norb_tot, norb_tot), dtype=dtype)
        out[(p, R)][norb_cumsum[i]:norb_cumsum[i + 1],
                    norb_cumsum[j]:norb_cumsum[j + 1]] = d
    return out


def finite_difference(predict_fn, positions0, smap, norb_cumsum, delta,
                      atom_indices=None, grad_threshold=1e-10):
    r''' central finite differences of predicted hopping blocks w.r.t. displacements
    of home-cell atoms, folded back to unit-cell labels via fold_key '''
    assert np.isfinite(delta) and delta > 0, \
        'delta must be a positive finite displacement (Angstrom)'
    norb_cumsum = np.asarray(norb_cumsum)
    blocks = {}
    for kappa in _displacement_targets(atom_indices, smap):
        for alpha in range(3):
            blocks[(kappa, alpha)] = finite_difference_pair(
                predict_fn, positions0, smap, norb_cumsum, delta, kappa, alpha,
                grad_threshold)
    return DerivativeData(n_grid=smap.n_grid, n_uc_atoms=smap.n_uc_atoms, delta=delta,
                          norb_cumsum=norb_cumsum, blocks=blocks)


def stream_finite_difference(predict_fn, positions0, smap, norb_cumsum, delta, out_path,
                             atom_indices=None, grad_threshold=1e-10):
    r''' central finite differences identical to finite_difference, but each
    (kappa, alpha) block group is written to out_path (HDF5, dH/{kappa}/{xyz}/{[p+R]}
    layout) and released immediately, so peak memory is one group instead of all
    displacements. Returns a read-only H5DerivativeStore over out_path. '''
    assert np.isfinite(delta) and delta > 0, \
        'delta must be a positive finite displacement (Angstrom)'
    norb_cumsum = np.asarray(norb_cumsum)
    with h5py.File(out_path, 'w') as f:
        f['n_grid'] = np.asarray(smap.n_grid, dtype=int)
        f['n_uc_atoms'] = int(smap.n_uc_atoms)
        f['delta'] = float(delta)
        f['norb_cumsum'] = norb_cumsum
        for kappa in _displacement_targets(atom_indices, smap):
            for alpha in range(3):
                out = finite_difference_pair(predict_fn, positions0, smap, norb_cumsum,
                                             delta, kappa, alpha, grad_threshold)
                # require_group so a (kappa, alpha) pair with no surviving blocks
                # still appears in H5DerivativeStore.pairs(), matching
                # DerivativeData.pairs() (blocks[(kappa, alpha)] = {} is still a pair)
                grp = f.require_group(f'dH/{kappa}/{"xyz"[alpha]}')
                for (p, R), m in out.items():
                    grp[str(list(p) + list(R))] = m
                del out
    return H5DerivativeStore(out_path)


def image_free_radius(sc_lattice):
    r''' half the smallest perpendicular thickness of the supercell: the largest
    separation at which a displaced atom cannot yet see its own periodic image.
    Derivatives involving atoms further apart than this wrap around the cell. '''
    inv = np.linalg.inv(np.asarray(sc_lattice, dtype=np.float64))
    return 0.5 * min(1.0 / np.linalg.norm(inv[:, a]) for a in range(3))


def contamination_profile(group, kappa, positions, uc_lattice, sc_lattice, norb_cumsum):
    r''' (distance, max |dH|) for one (kappa, alpha) derivative group, resolved by the
    separation between the displaced atom and the bra atom of each orbital block.

    The physical dH/dtau decays monotonically with that separation. A supercell too
    small for the model's receptive field breaks the decay: beyond the image-free
    radius the displaced atom couples to its own periodic images and |dH| stops
    falling, or rises again. Comparing the two regimes turns that failure into a
    number instead of an unconditional warning about a receptive field that a
    tractable supercell can rarely satisfy.

    positions/uc_lattice describe the unit cell; sc_lattice the displacement supercell. '''
    positions = np.asarray(positions, dtype=np.float64)
    uc_lattice = np.asarray(uc_lattice, dtype=np.float64)
    sc_lattice = np.asarray(sc_lattice, dtype=np.float64)
    inv_sc = np.linalg.inv(sc_lattice)
    norb_cumsum = np.asarray(norb_cumsum)
    n_uc_atoms = len(norb_cumsum) - 1
    dists, vals = [], []
    for (p, R), dense in group.items():
        # displaced atom sits in cell p relative to the bra atom's cell
        origin = positions[kappa] + np.asarray(p, dtype=np.float64) @ uc_lattice
        for i in range(n_uc_atoms):
            block = dense[norb_cumsum[i]:norb_cumsum[i + 1], :]
            if block.size == 0:
                continue
            sep = origin - positions[i]
            frac = sep @ inv_sc
            frac -= np.round(frac)
            dists.append(float(np.linalg.norm(frac @ sc_lattice)))
            vals.append(float(np.abs(block).max()))
    return np.asarray(dists), np.asarray(vals)


def acoustic_sum_rule(deriv):
    r''' max over alpha and R of |sum_{kappa, p} dH(R)|; should vanish for a
    translation-invariant model. Only meaningful when all atoms were displaced. '''
    worst = 0.0
    for alpha in range(3):
        acc = {}
        for kappa in range(deriv.n_uc_atoms):
            for (p, R), dense in deriv.group(kappa, alpha).items():
                acc[R] = acc.get(R, 0) + dense
        for m in acc.values():
            worst = max(worst, float(np.abs(m).max()))
    return worst
