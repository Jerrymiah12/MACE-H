"""Quantum ESPRESSO q=0 DFPT: input writers and Z*/eps_inf collection.

Consistency contract with ABACUS (recorded in dft_settings.yaml): same
relaxed lattice vectors and positions, same XC functional, same valence
configurations, same relativistic treatment, same charge and spin state;
prefer the same UPF pseudopotential files in both codes where supported.
MgO is an insulator: occupations = 'fixed' (DFPT requires no smearing).
"""
import json
import os
import re
import shutil

import numpy as np

from .config import atomic_write_text


def write_pw_input(path, cfg, cell, frac, species):
    qe, mat = cfg["qe"], cfg["material"]
    lines = ["&CONTROL", "  calculation = 'scf'", "  prefix = 'mgo'",
             "  outdir = './out'", f"  pseudo_dir = '{qe['pseudo_dir']}'",
             "  tprnfor = .true.", "  tstress = .true.", "/",
             "&SYSTEM", "  ibrav = 0", f"  nat = {len(species)}",
             f"  ntyp = {len(mat['species'])}",
             f"  ecutwfc = {qe['ecutwfc']}", "  occupations = 'fixed'", "/",
             "&ELECTRONS", f"  conv_thr = {qe['conv_thr']}", "/",
             "ATOMIC_SPECIES"]
    for s in mat["species"]:
        lines.append(f"{s} {mat['masses'][s]} {qe['pseudopotentials'][s]}")
    lines.append("CELL_PARAMETERS angstrom")
    for v in np.asarray(cell, float):
        lines.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    lines.append("ATOMIC_POSITIONS crystal")
    for s, f in zip(species, np.asarray(frac, float)):
        lines.append(f"{s}  {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}")
    k = qe["kmesh"]
    lines += ["K_POINTS automatic", f"{k[0]} {k[1]} {k[2]} 0 0 0"]
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_ph_input(path, cfg):
    qe = cfg["qe"]
    lines = ["MgO q=0 DFPT: dielectric tensor and Born effective charges",
             "&INPUTPH", "  prefix = 'mgo'", "  outdir = './out'",
             "  fildyn = 'mgo.dyn'", f"  tr2_ph = {qe['tr2_ph']}",
             "  epsil = .true.", "  trans = .true.", "/",
             "0.0 0.0 0.0"]
    atomic_write_text(path, "\n".join(lines) + "\n")


def init_dfpt_stage(cfg, workspace, args):
    from .snapshot import load_reference
    ref = load_reference(workspace)
    qdir = os.path.join(workspace, "reference", "qe")
    os.makedirs(qdir, exist_ok=True)
    write_pw_input(os.path.join(qdir, "pw.in"), cfg, ref["prim_cell"],
                   ref["frac"], ref["species"])
    write_ph_input(os.path.join(qdir, "ph.in"), cfg)
    job = [cfg["slurm"]["header"].rstrip(), "",
           f"{cfg['qe']['pw_command']} -in pw.in > pw.out 2>&1",
           f"{cfg['qe']['ph_command']} -in ph.in > ph.out 2>&1"]
    atomic_write_text(os.path.join(qdir, "job_qe.sh"), "\n".join(job) + "\n")
    print(f"QE DFPT inputs written to {qdir}")
    return 0
