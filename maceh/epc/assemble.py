import os
import tempfile

import numpy as np
import h5py

# memory budget for one k-batch of the complex accumulator in write_epc_cartesian_h5
BATCH_BYTES = 16 * 1024 ** 2
# memory budget for the stacked derivative blocks group_q_slab feeds to BLAS
SLAB_STACK_BYTES = 256 * 1024 ** 2


def displaced_atoms(deriv):
    return np.array(sorted({kappa for kappa, _ in deriv.pairs()}), dtype=int)


def group_q_slab(blocks, kpts, q, norb):
    r''' contribution of one already-loaded (kappa, alpha) derivative group at a
    single q: sum_p e^{2 pi i q.p} sum_R e^{2 pi i k.R} [dH(p, R)]_ij (cell-phase
    gauge), shape (nk, norb, norb).

    The sum is a matrix product -- phases (nk, n_blocks) times flattened blocks
    (n_blocks, norb^2) -- so it is handed to BLAS rather than accumulated block by
    block. Doing it termwise costs a full (nk, norb, norb) complex temporary per
    block (tens of MiB each, hundreds of blocks per group), which makes the stage
    memory-bandwidth bound; as one GEMM each block is read once. Real derivative
    blocks stay real all the way into BLAS and only the phases are complex, which
    halves the traffic again.

    Blocks are stacked in the order the mapping yields them and reduced in that
    order for every k, so the result does not depend on how the caller batches k. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    acc = np.zeros((len(kpts), norb, norb), dtype=np.complex128)
    if not blocks or len(kpts) == 0:
        return acc
    acc_flat = acc.reshape(len(kpts), norb * norb)

    items = list(blocks.items())
    per_block = norb * norb * 8
    chunk = max(1, SLAB_STACK_BYTES // max(per_block, 1))
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        ps = np.array([pR[0] for pR, _ in part], dtype=np.float64)
        Rs = np.array([pR[1] for pR, _ in part], dtype=np.float64)
        # (nk, n_part): e^{2 pi i q.p} e^{2 pi i k.R}
        phase = (np.exp(2j * np.pi * (ps @ q))[None, :]
                 * np.exp(2j * np.pi * (kpts @ Rs.T)))
        stack = np.stack([np.asarray(m).reshape(-1) for _, m in part])
        if np.iscomplexobj(stack):
            acc_flat += phase @ stack
        else:
            # real blocks: two real GEMMs instead of one complex one
            acc_flat.real += phase.real @ stack
            acc_flat.imag += phase.imag @ stack
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
    # k is transformed one batch at a time: the complex accumulator and its broadcast
    # temporary are the largest live arrays in stage 2, and a whole (nk, norb, norb)
    # slab reaches hundreds of MiB on realistic grids. Chunk along k to the same batch
    # and along nothing else, so every write covers whole chunks exactly once -- a
    # chunk spanning q/kappa/alpha would be left partly filled and force HDF5 into
    # read-modify-write on the next write.
    nk_batch = max(1, min(len(kpts), BATCH_BYTES // max(norb * norb * 16, 1)))
    chunks = (nk_batch, 1, 1, 1, norb, norb)
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
                        for k0 in range(0, len(kpts), nk_batch):
                            k1 = min(k0 + nk_batch, len(kpts))
                            slab = group_q_slab(blocks, kpts[k0:k1], q, norb)
                            g_real[k0:k1, iq, ikap, alpha] = slab.real
                            g_imag[k0:k1, iq, ikap, alpha] = slab.imag
                            # each k row is independent, so batching is arithmetically
                            # a no-op; dropping the slab keeps the previous batch from
                            # staying alive while the next one is built
                            del slab
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
