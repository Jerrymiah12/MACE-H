"""ABACUS file I/O. Writers here; output parsers live below (Task 9).

Atom-ordering contract: every STRU we write lists species in
cfg["material"]["species"] order with each species' atoms contiguous
(species-major).  ABACUS orders its matrix rows in STRU order, so this
must match Supercell ordering exactly.
"""
import re

import numpy as np

from .config import atomic_write_text
from .constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM


def write_stru(path, cell, cart, species, cfg):
    mat = cfg["material"]
    ab = cfg["abacus"]
    order = mat["species"]
    expected = [s for s in order for _ in range(species.count(s))]
    if list(species) != expected:
        raise ValueError("atoms must be species-major in config species order")
    frac = np.asarray(cart, float) @ np.linalg.inv(np.asarray(cell, float))
    lines = ["ATOMIC_SPECIES"]
    for s in order:
        lines.append(f"{s} {mat['masses'][s]} {ab['pseudopotentials'][s]}")
    lines += ["", "NUMERICAL_ORBITAL"]
    for s in order:
        lines.append(ab["orbitals"][s])
    lines += ["", "LATTICE_CONSTANT", f"{ANGSTROM_TO_BOHR:.15f}",
              "", "LATTICE_VECTORS"]
    for v in np.asarray(cell, float):
        lines.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    lines += ["", "ATOMIC_POSITIONS", "Direct"]
    for s in order:
        idx = [i for i, sp in enumerate(species) if sp == s]
        lines += [s, "0.0", str(len(idx))]
        for i in idx:
            f = frac[i]
            lines.append(f"{f[0]:.12f} {f[1]:.12f} {f[2]:.12f} m 0 0 0")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_input(path, cfg, **overrides):
    ab = cfg["abacus"]
    params = {
        "suffix": "MgO",
        "calculation": "scf",
        "basis_type": "lcao",
        "ntype": len(cfg["material"]["species"]),
        "nspin": 1,
        "symmetry": 0,
        "gamma_only": 0,
        "ecutwfc": ab["ecutwfc"],
        "scf_thr": ab["scf_thr"],
        "scf_nmax": ab["scf_nmax"],
        "smearing_method": ab["smearing_method"],
        "smearing_sigma": ab["smearing_sigma"],
        "pseudo_dir": ab["pseudo_dir"],
        "orbital_dir": ab["orbital_dir"],
    }
    params.update(overrides)
    if int(params["gamma_only"]) != 0:
        raise ValueError("gamma_only must stay 0 (out_mat_hs2 unsupported "
                         "under the gamma-only algorithm)")
    lines = ["INPUT_PARAMETERS"]
    for k, v in params.items():
        lines.append(f"{k:24s}{v}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_kpt(path, mesh):
    m = " ".join(str(int(x)) for x in mesh)
    atomic_write_text(path, f"K_POINTS\n0\nGamma\n{m} 0 0 0\n")


_FLOAT = r"[-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?"


def parse_running_scf(path):
    """Convergence flag, final total energy (eV), Fermi level (eV)."""
    with open(path, errors="replace") as f:
        text = f.read()
    low = text.lower()
    converged = ("charge density convergence is achieved" in low
                 or "convergence has been achieved" in low)
    m = re.search(r"!FINAL_ETOT_IS\s+(" + _FLOAT + r")\s+eV", text)
    etot = float(m.group(1)) if m else None
    if converged and etot is None:
        raise ValueError(f"{path}: converged run without !FINAL_ETOT_IS line")
    mf = re.search(r"EFERMI\s*=?\s*(" + _FLOAT + r")\s*eV", text)
    fermi = float(mf.group(1)) if mf else None
    return {"converged": converged, "etot_ev": etot, "fermi_ev": fermi}


def parse_csr(path):
    """Parse an ABACUS out_mat_hs2 sparse-matrix file.

    Format (ABACUS >= 3.0 prepends a 'STEP: 0' line):
        Matrix Dimension of H(R): <dim>
        Matrix number of H(R): <n>
        Rx Ry Rz nnz
        <nnz values> / <nnz col indices> / <dim+1 row pointers>   (if nnz > 0)
    """
    import scipy.sparse
    with open(path) as f:
        line = f.readline()
        if "Matrix Dimension of" not in line:
            line = f.readline()
            if "Matrix Dimension of" not in line:
                raise ValueError(f"{path}: missing 'Matrix Dimension of' header")
        dim = int(line.split()[-1])
        f.readline()                      # "Matrix number of ..."
        blocks = {}
        for line in f:
            parts = line.split()
            if not parts:
                break
            if len(parts) != 4:
                raise ValueError(f"{path}: malformed R header line: {line!r}")
            R = tuple(int(x) for x in parts[:3])
            nnz = int(parts[3])
            if nnz == 0:
                continue
            vals = np.array(f.readline().split(), dtype=float)
            cols = np.array(f.readline().split(), dtype=int)
            ptr = np.array(f.readline().split(), dtype=int)
            if len(vals) != nnz or len(cols) != nnz or len(ptr) != dim + 1:
                raise ValueError(f"{path}: CSR block {R} lengths inconsistent")
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"{path}: NaN/Inf in CSR block {R}")
            if R in blocks:
                raise ValueError(f"{path}: duplicate R block {R}")
            blocks[R] = scipy.sparse.csr_matrix((vals, cols, ptr),
                                                shape=(dim, dim))
    return dim, blocks


def parse_stru(path):
    """Parse STRU (as written by write_stru, or ABACUS STRU_ION_D).

    Returns (cell (3,3) Å, cart (N,3) Å, species).  Only Direct coordinates
    are supported.
    """
    with open(path) as f:
        lines = [l.strip() for l in f.read().splitlines()]
    i = lines.index("LATTICE_CONSTANT")
    lat_const_bohr = float(lines[i + 1].split()[0])
    j = lines.index("LATTICE_VECTORS")
    cell = np.array([[float(x) for x in lines[j + 1 + r].split()[:3]]
                     for r in range(3)])
    cell = cell * lat_const_bohr * BOHR_TO_ANGSTROM
    k = lines.index("ATOMIC_POSITIONS")
    k += 1
    while not lines[k]:
        k += 1
    if not lines[k].startswith("Direct"):
        raise ValueError(f"{path}: only Direct coordinates supported, "
                         f"got {lines[k]!r}")
    species, frac = [], []
    idx = k + 1
    while idx < len(lines):
        if not lines[idx]:
            idx += 1
            continue
        name = lines[idx].split()[0]
        count = int(lines[idx + 2].split()[0])
        for c in range(count):
            row = lines[idx + 3 + c].split()
            frac.append([float(x) for x in row[:3]])
            species.append(name)
        idx += 3 + count
    return cell, np.asarray(frac, float) @ cell, species


def write_job_script(path, cfg, snapshot_dirs):
    body = [cfg["slurm"]["header"].rstrip(), ""]
    body.append("for d in \\")
    for d in snapshot_dirs:
        body.append(f"    {d} \\")
    body.append("; do")
    body.append(f"    (cd \"$d\" && {cfg['slurm']['abacus_command']} "
                "> abacus.stdout 2>&1)")
    body.append("done")
    atomic_write_text(path, "\n".join(body) + "\n")
