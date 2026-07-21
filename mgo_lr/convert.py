"""ABACUS -> DeepH-E3/MACE-H data conversion.

The per-l orbital permutation/sign table is copied verbatim from
DeepH-pack deeph/preprocess/abacus_get_data.py (class OrbAbacus2DeepH).
A silent error here corrupts every matrix — the table is pinned by
tests/test_orbital_reorder.py and must not be re-derived.
"""
import json
import os

import h5py
import numpy as np
from scipy.linalg import block_diag

from .config import atomic_write_text, sha256_file
from .constants import ATOMIC_NUMBERS, RY_TO_EV


def _build_us():
    us = {0: np.eye(1),
          1: np.eye(3)[[1, 2, 0]],
          2: np.eye(5)[[0, 3, 4, 1, 2]],
          3: np.eye(7)}
    minus = {1: [0, 1], 2: [3, 4], 3: [1, 2, 5, 6]}
    for l, rows in minus.items():
        us[l][rows] *= -1.0
    return us


_U_ABACUS2DEEPH = _build_us()


def orbital_u(l):
    if l not in _U_ABACUS2DEEPH:
        raise NotImplementedError(f"only l <= 3 supported, got l={l}")
    return _U_ABACUS2DEEPH[l]


def atom_u(orbital_types_atom):
    """Block-diagonal transform for one atom's full AO set."""
    return block_diag(*[orbital_u(l) for l in orbital_types_atom])


def transform_block(mat, l_left, l_right):
    """U_i @ mat @ U_j.T for an atom-pair block."""
    return atom_u(l_left) @ np.asarray(mat, float) @ atom_u(l_right).T


BLOCK_SKIP_THRESHOLD = 1e-8   # same block-sparsity cutoff as DeepH-pack


def key_str(R, i, j):
    """DeepH-E3 h5 key: JSON list, 1-based atom indices (i, j are 0-based in)."""
    return f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {i + 1}, {j + 1}]"


def parse_key(k):
    v = json.loads(k)
    return (v[0], v[1], v[2], v[3], v[4])


def write_blocks(path, blocks):
    tmp = f"{path}.tmp.{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        for k, v in blocks.items():
            f[k] = np.asarray(v, np.float64)
    os.replace(tmp, path)


def read_blocks(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k], dtype=np.float64)
    return out


def species_orbital_info(cfg, species_list):
    types = [cfg["abacus"]["orbital_types"][s] for s in species_list]
    norb = [sum(2 * l + 1 for l in t) for t in types]
    offsets = np.concatenate([[0], np.cumsum(norb)])
    return types, norb, offsets


def matrices_to_blocks(csr_blocks, dim, cfg, species_list, factor):
    """Slice per-R matrices into atom-pair blocks, apply the orbital
    transform, scale by `factor` (RY_TO_EV for H, 1.0 for S)."""
    types, norb, offsets = species_orbital_info(cfg, species_list)
    if int(offsets[-1]) != dim:
        raise ValueError(f"matrix dimension {dim} != expected {offsets[-1]} "
                         f"from orbital_types for {len(species_list)} atoms")
    n_at = len(species_list)
    out = {}
    for R, m in csr_blocks.items():
        dense = m.toarray()
        if not np.all(np.isfinite(dense)):
            raise ValueError(f"NaN/Inf in matrix block R={R}")
        for i in range(n_at):
            for j in range(n_at):
                blk = dense[offsets[i]:offsets[i + 1],
                            offsets[j]:offsets[j + 1]]
                if np.abs(blk).max() < BLOCK_SKIP_THRESHOLD:
                    continue
                out[key_str(R, i, j)] = factor * transform_block(
                    blk, types[i], types[j])
    return out


def write_structure_files(folder, cell, cart, species, cfg, fermi_ev):
    cell = np.asarray(cell, float)
    types, norb, _ = species_orbital_info(cfg, species)
    np.savetxt(os.path.join(folder, "lat.dat"), cell.T)
    np.savetxt(os.path.join(folder, "rlat.dat"),
               np.linalg.inv(cell) * 2.0 * np.pi)
    np.savetxt(os.path.join(folder, "site_positions.dat"),
               np.asarray(cart, float).T)
    atomic_write_text(os.path.join(folder, "element.dat"),
                      "\n".join(str(ATOMIC_NUMBERS[s]) for s in species) + "\n")
    atomic_write_text(os.path.join(folder, "orbital_types.dat"),
                      "\n".join("  ".join(str(l) for l in t)
                                for t in types) + "\n")
    info = {"nsites": len(species), "isorthogonal": False,
            "isspinful": False, "norbits": int(sum(norb)),
            "fermi_level": fermi_ev if fermi_ev is not None else 0.0}
    atomic_write_text(os.path.join(folder, "info.json"), json.dumps(info))


def collect_dft_stage(cfg, workspace, args):
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell
    if getattr(args, "set_name", None) is None:
        raise SystemExit("collect-dft requires --set pilot|main|large")
    ref = load_reference(workspace)
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    store = SnapshotStore(workspace, args.set_name)
    tau_diag = float(cfg["validation"]["tau_overlap_diag"])
    exit_code, converted, skipped = 0, 0, 0
    for sid in store.list():
        if store.read_status(sid)["state"] == "rejected":
            continue
        if store.state_at_least(sid, "converted") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        out_dir = os.path.join(folder, "OUT.MgO")
        log = os.path.join(out_dir, "running_scf.log")
        if not os.path.exists(log):
            continue                       # DFT not run yet: stay prepared
        from . import abacus_io
        scf = abacus_io.parse_running_scf(log)
        if not scf["converged"]:
            store.reject(sid, "scf_not_converged")
            exit_code = 1
            continue
        h_path = os.path.join(out_dir, cfg["abacus"]["csr_h_filename"])
        s_path = os.path.join(out_dir, cfg["abacus"]["csr_s_filename"])
        if not (os.path.exists(h_path) and os.path.exists(s_path)):
            store.reject(sid, "csr_files_missing")
            exit_code = 1
            continue
        cell, cart, species = abacus_io.parse_stru(
            os.path.join(folder, "STRU"))
        u = np.load(os.path.join(folder, "displacements.npy"))
        if species != sc.species:
            store.reject(sid, "atom_order_changed")
            exit_code = 1
            continue
        if not (np.allclose(cell, sc.cell, atol=1e-8)
                and np.allclose(cart, sc.cart + u, atol=1e-6)):
            store.reject(sid, "geometry_mismatch_vs_reference")
            exit_code = 1
            continue
        try:
            dim_h, h_csr = abacus_io.parse_csr(h_path)
            dim_s, s_csr = abacus_io.parse_csr(s_path)
            if dim_h != dim_s:
                raise ValueError(f"H dim {dim_h} != S dim {dim_s}")
            h_blocks = matrices_to_blocks(h_csr, dim_h, cfg, species,
                                          RY_TO_EV)
            s_blocks = matrices_to_blocks(s_csr, dim_s, cfg, species, 1.0)
        except ValueError as e:
            store.reject(sid, f"matrix_parse_failed: {e}")
            exit_code = 1
            continue
        bad_diag = False
        for i in range(len(species)):
            k = key_str((0, 0, 0), i, i)
            if k not in s_blocks or \
                    np.abs(np.diag(s_blocks[k]) - 1.0).max() > tau_diag:
                bad_diag = True
        if bad_diag:
            store.reject(sid, "pathological_overlap_diagonal")
            exit_code = 1
            continue
        full_path = os.path.join(folder, "hamiltonians_full.h5")
        if os.path.exists(full_path) and not args.force:
            raise SystemExit(f"{sid}: hamiltonians_full.h5 exists; "
                             "refusing to overwrite without --force")
        write_blocks(full_path, h_blocks)
        write_blocks(os.path.join(folder, "overlaps.h5"), s_blocks)
        write_structure_files(folder, cell, cart, species, cfg,
                              scf["fermi_ev"])
        store.write_status(
            sid, "converted", scf_converged=True, etot_ev=scf["etot_ev"],
            csr_files=[cfg["abacus"]["csr_h_filename"],
                       cfg["abacus"]["csr_s_filename"]],
            raw_sha256={os.path.basename(p): sha256_file(p)
                        for p in (h_path, s_path, log)})
        converted += 1
    print(f"{args.set_name}: converted {converted}, skipped {skipped}")
    return exit_code
