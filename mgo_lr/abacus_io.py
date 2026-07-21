"""ABACUS file I/O. Writers here; output parsers live below (Task 9).

Atom-ordering contract: every STRU we write lists species in
cfg["material"]["species"] order with each species' atoms contiguous
(species-major).  ABACUS orders its matrix rows in STRU order, so this
must match Supercell ordering exactly.
"""
import numpy as np

from .config import atomic_write_text
from .constants import ANGSTROM_TO_BOHR


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
