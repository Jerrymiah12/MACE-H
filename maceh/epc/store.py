import json

import h5py


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
        return out
