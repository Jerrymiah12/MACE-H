"""Reference-structure stages: ABACUS convergence/relax decks and the
permanent reference artifacts every later stage consumes."""
import json
import math
import os

import numpy as np
import yaml

from . import __version__, abacus_io
from .config import atomic_write_text
from .constants import ATOMIC_NUMBERS
from .structures import rocksalt_primitive


def _write_deck(folder, cell, frac, species, cfg, kmesh, **input_overrides):
    os.makedirs(folder, exist_ok=True)
    abacus_io.write_stru(os.path.join(folder, "STRU"), cell,
                         np.asarray(frac, float) @ np.asarray(cell, float),
                         species, cfg)
    abacus_io.write_input(os.path.join(folder, "INPUT"), cfg,
                          suffix="MgO", **input_overrides)
    abacus_io.write_kpt(os.path.join(folder, "KPT"), kmesh)


def init_reference_stage(cfg, workspace, args):
    a = float(cfg["material"]["lattice_constant_guess"])
    cell, frac, species = rocksalt_primitive(a)
    base = os.path.join(workspace, "reference", "abacus")
    kp = cfg["abacus"]["kmesh_primitive"]
    for e in cfg["reference"]["ecut_scan"]:
        _write_deck(os.path.join(base, f"ecut_{e}"), cell, frac, species,
                    cfg, kp, calculation="scf", ecutwfc=e)
    for mesh in cfg["reference"]["kmesh_scan"]:
        _write_deck(os.path.join(base, f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}"),
                    cell, frac, species, cfg, mesh, calculation="scf")
    _write_deck(os.path.join(base, "cell_relax"), cell, frac, species, cfg,
                kp, calculation="cell-relax", cal_force=1, cal_stress=1,
                relax_nmax=100)
    _write_deck(os.path.join(base, "final_scf"), cell, frac, species, cfg,
                kp, calculation="scf", out_mat_hs2=1)
    print(f"reference decks written under {base}")
    return 0


def lattice_constant_from_cell(cell):
    """Cubic lattice constant from the fcc primitive cell a/2*(011)-type rows."""
    cell = np.asarray(cell, float)
    norms = np.linalg.norm(cell, axis=1)
    if np.abs(norms - norms.mean()).max() > 1e-3 * norms.mean():
        print(f"WARNING: relaxed cell deviates from cubic: |v_i| = {norms}")
    return float(norms.mean() * math.sqrt(2.0))


def _write_cif(path, cell, frac, species):
    cell = np.asarray(cell, float)
    a, b, c = np.linalg.norm(cell, axis=1)

    def ang(u, v):
        return math.degrees(math.acos(
            float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))

    lines = ["data_MgO",
             f"_cell_length_a {a:.8f}", f"_cell_length_b {b:.8f}",
             f"_cell_length_c {c:.8f}",
             f"_cell_angle_alpha {ang(cell[1], cell[2]):.6f}",
             f"_cell_angle_beta {ang(cell[0], cell[2]):.6f}",
             f"_cell_angle_gamma {ang(cell[0], cell[1]):.6f}",
             "_symmetry_space_group_name_H-M 'P 1'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    for i, (s, f) in enumerate(zip(species, np.asarray(frac, float)), 1):
        lines.append(f"{s}{i} {s} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def collect_reference_stage(cfg, workspace, args):
    ref_dir = os.path.join(workspace, "reference")
    base = os.path.join(ref_dir, "abacus")
    override = cfg["material"].get("lattice_constant_relaxed")
    if override is not None:
        a = float(override)
    else:
        relaxed = os.path.join(base, "cell_relax", "OUT.MgO", "STRU_ION_D")
        if not os.path.exists(relaxed):
            raise SystemExit(
                f"no relaxed structure at {relaxed} and no "
                "material.lattice_constant_relaxed override — run the "
                "cell-relax deck first")
        cell_r, _, _ = abacus_io.parse_stru(relaxed)
        a = lattice_constant_from_cell(cell_r)
    cell, frac, species = rocksalt_primitive(a)

    # scan summary (tolerant: report what ran, warn about what did not)
    summary = {"ecut": {}, "kmesh": {}, "warnings": []}
    for e in cfg["reference"]["ecut_scan"]:
        log = os.path.join(base, f"ecut_{e}", "OUT.MgO", "running_scf.log")
        if os.path.exists(log):
            summary["ecut"][str(e)] = abacus_io.parse_running_scf(log)
        else:
            summary["warnings"].append(f"ecut_{e}: output missing")
    for mesh in cfg["reference"]["kmesh_scan"]:
        name = f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}"
        log = os.path.join(base, name, "OUT.MgO", "running_scf.log")
        if os.path.exists(log):
            summary["kmesh"][name] = abacus_io.parse_running_scf(log)
        else:
            summary["warnings"].append(f"{name}: output missing")
    atomic_write_text(os.path.join(ref_dir, "scan_summary.json"),
                      json.dumps(summary, indent=1))

    np.save(os.path.join(ref_dir, "reference_cell.npy"), cell)
    np.save(os.path.join(ref_dir, "reference_positions.npy"), frac)
    np.save(os.path.join(ref_dir, "atomic_numbers.npy"),
            np.array([ATOMIC_NUMBERS[s] for s in species]))
    atomic_write_text(os.path.join(ref_dir, "species_order.json"),
                      json.dumps(species))
    # one line per atom (2-atom primitive cell -> 2 lines)
    ot = cfg["abacus"]["orbital_types"]
    atomic_write_text(os.path.join(ref_dir, "orbital_types.dat"),
                      "\n".join("  ".join(str(l) for l in ot[s])
                                for s in species) + "\n")
    _write_cif(os.path.join(ref_dir, "primitive.cif"), cell, frac, species)
    settings = {"lattice_constant_relaxed": a,
                "abacus": cfg["abacus"], "qe": cfg["qe"],
                "mgo_lr_version": __version__}
    atomic_write_text(os.path.join(ref_dir, "dft_settings.yaml"),
                      yaml.safe_dump(settings, sort_keys=False))
    # regenerate the final high-accuracy SCF deck at the relaxed geometry
    _write_deck(os.path.join(base, "final_scf"), cell, frac, species, cfg,
                cfg["abacus"]["kmesh_primitive"], calculation="scf",
                out_mat_hs2=1)
    print(f"reference collected: a = {a:.6f} Å; rerun final_scf deck if "
          "the lattice constant changed")
    return 0
