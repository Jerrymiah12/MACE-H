import os
import tempfile

import numpy as np
import h5py


def displaced_atoms(deriv):
    return np.array(sorted({kappa for kappa, _ in deriv.pairs()}), dtype=int)


def compute_q_slab(deriv, kpts, q, displaced):
    r''' g for a single q-point: g_ijka(k, q) = sum_p e^{2 pi i q.p} sum_R
    e^{2 pi i k.R} [dH(p, R)]_ij (cell-phase gauge), shape
    (nk, n_displaced, 3, norb, norb) '''
    kpts = np.asarray(kpts, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    norb = deriv.norb_tot
    slab = np.zeros((len(kpts), len(displaced), 3, norb, norb), dtype=np.complex128)
    for ikap, kappa in enumerate(displaced):
        for alpha in range(3):
            for (p, R), m in deriv.group(kappa, alpha).items():
                phase_q = np.exp(2j * np.pi * (q @ np.asarray(p, dtype=np.float64)))
                phase_k = np.exp(2j * np.pi * (kpts @ np.asarray(R, dtype=np.float64)))
                slab[:, ikap, alpha] += (phase_q * phase_k)[:, None, None] * m
    return slab


def compute_epc_cartesian(deriv, kpts, qpts):
    r''' Cartesian atomic-orbital electron-phonon coupling, fully in memory.
    Returns g of shape (nk, nq, n_displaced, 3, norb, norb); phonon-mode
    contraction and band transformation are left for downstream. For large
    k/q grids prefer write_epc_cartesian_h5, which streams one q at a time. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = displaced_atoms(deriv)
    g = np.stack([compute_q_slab(deriv, kpts, q, displaced) for q in qpts], axis=1)
    return dict(g=g, atom_indices=displaced, kpoints=kpts, qpoints=qpts)


def write_epc_cartesian_h5(path, struct, deriv, kpts, qpts, attrs,
                           save_derivatives=False):
    r''' compute g and write epc_cartesian_pred.h5, streaming one q-point at a
    time into chunked g_real/g_imag datasets so peak memory is one q-slab
    instead of the full (nk, nq, ...) tensor. The file is written to a unique
    temporary file next to path and atomically renamed on success, so an
    interrupted, failed or concurrent write never clobbers a previous result
    with a truncated file. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = displaced_atoms(deriv)
    norb = deriv.norb_tot
    shape = (len(kpts), len(qpts), len(displaced), 3, norb, norb)
    size_bytes = 2 * float(np.prod(shape)) * 8
    if save_derivatives:
        size_bytes += sum(m.nbytes for kappa, alpha in deriv.pairs()
                          for m in deriv.group(kappa, alpha).values())
    print(f'Writing g_real/g_imag of shape {shape}'
          f'{" plus dH derivatives" if save_derivatives else ""}'
          f' (~{size_bytes / 1024 ** 3:.2f} GiB on disk)')
    norb_per_atom = np.diff(deriv.norb_cumsum)
    orbital_indices = np.repeat(np.arange(deriv.n_uc_atoms), norb_per_atom)
    # unique temp name in the destination directory: concurrent jobs writing the
    # same path must not truncate or delete each other's in-progress file, and
    # os.replace stays atomic only within one filesystem
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                    prefix=os.path.basename(path) + '.', suffix='.tmp')
    os.close(fd)
    try:
        with h5py.File(tmp_path, 'w') as f:
            g_real = f.create_dataset('g_real', shape=shape, dtype=np.float64, chunks=True)
            g_imag = f.create_dataset('g_imag', shape=shape, dtype=np.float64, chunks=True)
            for iq, q in enumerate(qpts):
                slab = compute_q_slab(deriv, kpts, q, displaced)
                g_real[:, iq] = slab.real
                g_imag[:, iq] = slab.imag
            f['kpoints'] = kpts
            f['qpoints'] = qpts
            f['atomic_numbers'] = np.asarray(struct.numbers, dtype=int)
            f['atom_indices'] = displaced
            f['cartesian_directions'] = np.array([b'x', b'y', b'z'])
            f['orbital_indices'] = orbital_indices
            f['lattice'] = struct.lattice
            f['positions'] = struct.positions
            f['supercell_matrix'] = np.diag(deriv.n_grid)
            f['finite_difference_delta'] = deriv.delta
            for k, v in attrs.items():
                f.attrs[k] = v
            if save_derivatives:
                for kappa, alpha in deriv.pairs():
                    for (p, R), m in deriv.group(kappa, alpha).items():
                        f[f'dH/{kappa}/{"xyz"[alpha]}/{str(list(p) + list(R))}'] = m
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
