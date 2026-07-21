import os
import time

import yaml

REQUIRED = [
    "material.name", "material.lattice_constant_guess", "material.species",
    "material.masses",
    "abacus.pseudo_dir", "abacus.orbital_dir", "abacus.pseudopotentials",
    "abacus.orbitals", "abacus.orbital_types", "abacus.ecutwfc",
    "abacus.scf_thr", "abacus.scf_nmax", "abacus.smearing_method",
    "abacus.smearing_sigma", "abacus.kmesh_primitive",
    "abacus.kmesh_supercell", "abacus.gamma_only_algorithm",
    "abacus.csr_h_filename", "abacus.csr_s_filename", "abacus.version",
    "qe.pseudo_dir", "qe.pseudopotentials", "qe.ecutwfc", "qe.kmesh",
    "qe.conv_thr", "qe.tr2_ph", "qe.version", "qe.pw_command", "qe.ph_command",
    "dfpt.zstar_sum_warn", "dfpt.isotropy_warn",
    "reference.ecut_scan", "reference.kmesh_scan",
    "supercells.pilot", "supercells.main", "supercells.large",
    "displacements.seed", "displacements.min_distance",
    "displacements.amplitudes", "displacements.pilot_ladder",
    "displacements.main_composition", "displacements.large_count",
    "lr.ewald_lambda", "lr.reciprocal_tolerance", "lr.imaginary_tolerance",
    "lr.convergence_factor",
    "validation.delta", "validation.tau_eq", "validation.tau_u",
    "validation.tau_translation", "validation.tau_reconstruct",
    "validation.tau_hermiticity", "validation.tau_G",
    "validation.tau_overlap_diag", "validation.tier2_enforce",
    "splits.validation_fraction", "splits.test_fraction",
    "slurm.header", "slurm.abacus_command",
]


def require(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing required config field: {dotted}")
        node = node[part]
    return node


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    for field in REQUIRED:
        require(cfg, field)
    if require(cfg, "abacus.gamma_only_algorithm"):
        raise ValueError(
            "abacus.gamma_only_algorithm must be false: ABACUS gamma-only "
            "algorithm does not support out_mat_hs2")
    return cfg


def atomic_write_text(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_resolved(cfg, workspace, stage):
    logs = os.path.join(workspace, "generation_logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, f"config-{stage}-{time.strftime('%Y%m%d-%H%M%S')}.yaml")
    atomic_write_text(out, yaml.safe_dump(cfg, sort_keys=False))
    return out
