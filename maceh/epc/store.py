import json

import h5py


def canonical_group(blocks):
    r''' derivative blocks in a storage-independent (p, R) order. Float addition is
    not associative, so the Fourier sum in assemble only lands on the same bits from
    either backend if every group() hands its keys back in one canonical order:
    HDF5 iterates links lexicographically by name, where '[0, 0, 0, 10, 0, 0]' sorts
    before '[0, 0, 0, 2, 0, 0]' and a leading '-1' before both, while an in-memory
    dict preserves the insertion order of the predicted hoppings. Sorting at write
    time cannot fix this -- string order is not tuple order -- so both accessors
    sort on read instead. '''
    return dict(sorted(blocks.items()))


class H5DerivativeStore:
    r''' read-only view of real-space Hamiltonian derivatives persisted by
    stream_finite_difference. Presents the same metadata attributes and
    pairs()/group() interface as DerivativeData, so every derivative consumer
    works on an on-disk store without holding all (kappa, alpha) groups in RAM.
    Datasets live at dH/{kappa}/{x|y|z}/{str(list(p) + list(R))}. '''

    def __init__(self, path):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.n_grid = tuple(int(x) for x in f['n_grid'][()])
            self.n_uc_atoms = int(f['n_uc_atoms'][()])
            self.delta = float(f['delta'][()])
            self.norb_cumsum = f['norb_cumsum'][()]

    @property
    def norb_tot(self):
        return int(self.norb_cumsum[-1])

    def pairs(self):
        out = []
        with h5py.File(self.path, 'r') as f:
            if 'dH' not in f:
                return out
            for kappa in f['dH']:
                for a in f[f'dH/{kappa}']:
                    out.append((int(kappa), 'xyz'.index(a)))
        return sorted(out)

    def nbytes(self):
        r''' total size of the stored derivative datasets in bytes, read from HDF5
        metadata only: no block is loaded into memory '''
        total = 0
        with h5py.File(self.path, 'r') as f:
            for kappa in f.get('dH', {}):
                for a in f[f'dH/{kappa}']:
                    total += sum(ds.nbytes for ds in f[f'dH/{kappa}/{a}'].values())
        return total

    def group(self, kappa, alpha):
        out = {}
        with h5py.File(self.path, 'r') as f:
            grp = f.get(f'dH/{kappa}/{"xyz"[alpha]}')
            if grp is None:
                return out
            for name, ds in grp.items():
                key = json.loads(name)               # [px, py, pz, Rx, Ry, Rz]
                p, R = tuple(key[:3]), tuple(key[3:])
                out[(p, R)] = ds[()]
        return canonical_group(out)
