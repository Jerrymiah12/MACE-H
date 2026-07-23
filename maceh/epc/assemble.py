import os
import tempfile

import numpy as np
import h5py


def displaced_atoms(deriv):
    return np.array(sorted({kappa for kappa, _ in deriv.pairs()}), dtype=int)


def group_q_slab(blocks, kpts, q, norb):
    r''' contribution of one already-loaded (kappa, alpha) derivative group at a
    single q: sum_p e^{2 pi i q.p} sum_R e^{2 pi i k.R} [dH(p, R)]_ij (cell-phase
    gauge), shape (nk, norb, norb) '''
    q = np.asarray(q, dtype=np.float64)
    acc = np.zeros((len(kpts), norb, norb), dtype=np.complex128)
    for (p, R), m in blocks.items():
        phase_q = np.exp(2j * np.pi * (q @ np.asarray(p, dtype=np.float64)))
        phase_k = np.exp(2j * np.pi * (kpts @ np.asarray(R, dtype=np.float64)))
        acc += (phase_q * phase_k)[:, None, None] * m
    return acc


def compute_epc_cartesian(deriv, kpts, qpts):
    r''' Cartesian atomic-orbital electron-phonon coupling, fully in memory.
    Returns g of shape (nk, nq, n_displaced, 3, norb, norb); phonon-mode
    contraction and band transformation are left for downstream. For large
    k/q grids prefer write_epc_cartesian_h5, which never holds the full tensor. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = displaced_atoms(deriv)
    norb = deriv.norb_tot
    g = np.zeros((len(kpts), len(qpts), len(displaced), 3, norb, norb),
                 dtype=np.complex128)
    # group-major: each derivative group is fetched from the store exactly once and
    # transformed for every q, instead of refetching the whole store per q-point
    for ikap, kappa in enumerate(displaced):
        for alpha in range(3):
            blocks = deriv.group(kappa, alpha)
            for iq, q in enumerate(qpts):
                g[:, iq, ikap, alpha] = group_q_slab(blocks, kpts, q, norb)
    return dict(g=g, atom_indices=displaced, kpoints=kpts, qpoints=qpts)


def write_epc_cartesian_h5(path, struct, deriv, kpts, qpts, attrs,
                           save_derivatives=False):
    r''' compute g and write epc_cartesian_pred.h5. The transform runs group-major:
    each (kappa, alpha) derivative group is fetched from deriv exactly once and
    transformed for every q straight into chunked g_real/g_imag datasets, so peak
    memory is one group plus one (nk, norb, norb) accumulator and an on-disk store
    is read once in total rather than once per q-point. The file is written to a
    unique temporary file next to path and atomically renamed on success, so an
    interrupted, failed or concurrent write never clobbers a previous result with a
    truncated file. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = displaced_atoms(deriv)
    norb = deriv.norb_tot
    shape = (len(kpts), len(qpts), len(displaced), 3, norb, norb)
    size_bytes = 2 * float(np.prod(shape)) * 8
    if save_derivatives:
        size_bytes += deriv.nbytes()   # dataset metadata only, reads no blocks
    print(f'Writing g_real/g_imag of shape {shape}'
          f'{" plus dH derivatives" if save_derivatives else ""}'
          f' (~{size_bytes / 1024 ** 3:.2f} GiB on disk)')
    norb_per_atom = np.diff(deriv.norb_cumsum)
    orbital_indices = np.repeat(np.arange(deriv.n_uc_atoms), norb_per_atom)
    # chunk along k only: every write below fills one whole (nk, norb, norb) hyperslab
    # at fixed (q, kappa, alpha) exactly once, so a chunk spanning q/kappa/alpha would
    # be left partially filled and force HDF5 into read-modify-write on the next write
    nk_chunk = max(1, min(len(kpts), 4 * 1024 ** 2 // max(norb * norb * 8, 1)))
    chunks = (nk_chunk, 1, 1, 1, norb, norb)
    # unique temp name in the destination directory: concurrent jobs writing the
    # same path must not truncate or delete each other's in-progress file, and
    # os.replace stays atomic only within one filesystem
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                    prefix=os.path.basename(path) + '.', suffix='.tmp')
    os.close(fd)
    try:
        with h5py.File(tmp_path, 'w') as f:
            g_real = f.create_dataset('g_real', shape=shape, dtype=np.float64, chunks=chunks)
            g_imag = f.create_dataset('g_imag', shape=shape, dtype=np.float64, chunks=chunks)
            for ikap, kappa in enumerate(displaced):
                for alpha in range(3):
                    blocks = deriv.group(kappa, alpha)
                    for iq, q in enumerate(qpts):
                        slab = group_q_slab(blocks, kpts, q, norb)
                        g_real[:, iq, ikap, alpha] = slab.real
                        g_imag[:, iq, ikap, alpha] = slab.imag
                    if save_derivatives:
                        # copied while the group is already in hand: a second pass over
                        # deriv would double the reads for an on-disk store
                        for (p, R), m in blocks.items():
                            f[f'dH/{kappa}/{"xyz"[alpha]}/{str(list(p) + list(R))}'] = m
                    del blocks
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
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
