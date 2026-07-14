import numpy as np
import h5py


def compute_epc_cartesian(deriv, kpts, qpts):
    r''' Cartesian atomic-orbital electron-phonon coupling
    g_ijka(k, q) = sum_p e^{2 pi i q.p} sum_R e^{2 pi i k.R} [dH(p, R)]_ij
    (cell-phase gauge). Returns g of shape (nk, nq, n_displaced, 3, norb, norb);
    phonon-mode contraction and band transformation are left for downstream. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = sorted({kappa for kappa, _ in deriv.blocks.keys()})
    nk, nq = len(kpts), len(qpts)
    norb = deriv.norb_tot
    g = np.zeros((nk, nq, len(displaced), 3, norb, norb), dtype=np.complex128)
    for ikap, kappa in enumerate(displaced):
        for alpha in range(3):
            for (p, R), m in deriv.blocks.get((kappa, alpha), {}).items():
                # phase factors for all (k, q) at once
                phase_q = np.exp(2j * np.pi * (qpts @ np.asarray(p, dtype=np.float64)))
                phase_k = np.exp(2j * np.pi * (kpts @ np.asarray(R, dtype=np.float64)))
                g[:, :, ikap, alpha] += (phase_k[:, None] * phase_q[None, :])[:, :, None, None] * m
    return dict(g=g, atom_indices=np.array(displaced, dtype=int),
                kpoints=kpts, qpoints=qpts)


def write_epc_cartesian_h5(path, results, struct, deriv, attrs):
    norb_per_atom = np.diff(deriv.norb_cumsum)
    orbital_indices = np.repeat(np.arange(deriv.n_uc_atoms), norb_per_atom)
    with h5py.File(path, 'w') as f:
        f['g_real'] = results['g'].real
        f['g_imag'] = results['g'].imag
        f['kpoints'] = results['kpoints']
        f['qpoints'] = results['qpoints']
        f['atomic_numbers'] = np.asarray(struct.numbers, dtype=int)
        f['atom_indices'] = results['atom_indices']
        f['cartesian_directions'] = np.array([b'x', b'y', b'z'])
        f['orbital_indices'] = orbital_indices
        f['lattice'] = struct.lattice
        f['positions'] = struct.positions
        f['supercell_matrix'] = np.diag(deriv.n_grid)
        f['finite_difference_delta'] = deriv.delta
        for k, v in attrs.items():
            f.attrs[k] = v
