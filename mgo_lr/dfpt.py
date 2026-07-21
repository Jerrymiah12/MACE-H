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


_FLOAT = r"[-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?"


def _three_floats(line):
    vals = re.findall(_FLOAT, line)
    if len(vals) < 3:
        raise ValueError(f"expected 3 floats in ph.x line: {line!r}")
    return [float(v.replace("D", "E").replace("d", "e")) for v in vals[-3:]]


def parse_ph_output(text):
    """Extract eps_inf and Born charges (d Force / dE block) from ph.x output."""
    lines = text.splitlines()
    eps = None
    for i, line in enumerate(lines):
        if "Dielectric constant in cartesian axis" in line:
            rows = [l for l in lines[i + 1:i + 8] if "(" in l][:3]
            if len(rows) != 3:
                raise ValueError("ph.x output: dielectric block malformed")
            eps = np.array([_three_floats(r) for r in rows])
    idx = None
    for i, line in enumerate(lines):
        if "Effective charges (d Force / dE) in cartesian axis" in line:
            idx = i
    if eps is None or idx is None:
        raise ValueError(
            "ph.x output lacks the dielectric tensor or Born effective "
            "charges — was ph.x run with epsil=.true. and trans=.true.?")
    atom_re = re.compile(r"atom\s+(\d+)\s+(\S+)")
    zstar, labels = [], []
    i = idx + 1
    while i < len(lines):
        if "Effective charges" in lines[i]:
            break                                # next block (d P / du)
        m = atom_re.search(lines[i])
        if m:
            rows = [l for l in lines[i + 1:i + 5] if "(" in l][:3]
            if len(rows) != 3:
                raise ValueError(f"ph.x Born block malformed at atom {m.group(1)}")
            zstar.append([_three_floats(r) for r in rows])
            labels.append(m.group(2))
            i += 4
        else:
            i += 1
    if not zstar:
        raise ValueError("ph.x output: no Born-charge atom blocks found")
    return np.asarray(eps, float), np.asarray(zstar, float), labels


def apply_asr(zstar):
    """Acoustic sum rule: Z~*_k = Z*_k - (1/N) sum_k' Z*_k'."""
    zstar = np.asarray(zstar, float)
    return zstar - zstar.sum(axis=0)[None] / zstar.shape[0]


def collect_dfpt_stage(cfg, workspace, args):
    from .snapshot import load_reference
    ref_dir = os.path.join(workspace, "reference")
    out_path = os.path.join(ref_dir, "qe", "ph.out")
    if not os.path.exists(out_path):
        raise SystemExit(f"ph.x output not found: {out_path}")
    with open(out_path) as f:
        eps, zstar, labels = parse_ph_output(f.read())
    ref = load_reference(workspace)
    hard, warn = [], []
    if list(labels) != list(ref["species"]):
        hard.append(f"atom labels {labels} != species order {ref['species']}")
    if zstar.shape != (len(ref["species"]), 3, 3):
        hard.append(f"Z* shape {zstar.shape} != (2,3,3)")
    if eps.shape != (3, 3):
        hard.append(f"eps shape {eps.shape} != (3,3)")
    else:
        if not np.allclose(eps, eps.T, atol=1e-6):
            warn.append("eps_inf not symmetric")
        if np.linalg.eigvalsh(0.5 * (eps + eps.T)).min() <= 0.0:
            hard.append("eps_inf not positive definite")
    if not hard:
        raw_sum = np.abs(zstar.sum(axis=0)).max()
        if raw_sum > float(cfg["dfpt"]["zstar_sum_warn"]):
            warn.append(f"raw ASR violation max|sum Z*| = {raw_sum:.4f}")
        z_asr = apply_asr(zstar)
        for a, lab in enumerate(labels):
            z = z_asr[a]
            diag = np.diag(z)
            off = float(np.abs(z - np.diag(diag)).max())
            aniso = float(np.abs(diag - diag.mean()).max()
                          / max(abs(diag.mean()), 1e-12))
            if off > float(cfg["dfpt"]["isotropy_warn"]) \
                    or aniso > float(cfg["dfpt"]["isotropy_warn"]):
                warn.append(f"Z*({lab}) anisotropic: off {off:.4f}, "
                            f"rel {aniso:.4f}")
        if not (np.diag(z_asr[0]).mean() > 0.0 > np.diag(z_asr[1]).mean()):
            hard.append("Z* diagonal signs wrong: expected Z*_Mg > 0 > Z*_O")
    checks = {"hard_failures": hard, "warnings": warn}
    atomic_write_text(os.path.join(ref_dir, "dfpt_checks.json"),
                      json.dumps(checks, indent=1))
    if hard:
        raise SystemExit("collect-dfpt hard failures: " + "; ".join(hard))
    np.save(os.path.join(ref_dir, "born_effective_charges.npy"), z_asr)
    np.save(os.path.join(ref_dir, "dielectric_infinity.npy"), eps)
    shutil.copyfile(out_path, os.path.join(ref_dir, "qe_dfpt_output.out"))
    for w in warn:
        print(f"WARNING: {w}")
    print(f"Z*_Mg = {np.diag(z_asr[0]).mean():+.4f}, "
          f"Z*_O = {np.diag(z_asr[1]).mean():+.4f}, "
          f"eps_inf = {np.diag(eps).mean():.4f}")
    return 0


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
