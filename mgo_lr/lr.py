"""Standalone long-range (LR) Hamiltonian processor.

Unit/sign convention (see also constants.py — this module is the ONLY
place the Coulomb prefactor and sign enter):

    Z*  dimensionless (units of e);  u in Å;  d_k = Z~*_k u_k^rel (e·Å)
    phi(G) = -i (4π/Ω) C_COUL [Σ_k G·d_k e^(-iG·R0_k)] / (G·ε∞·G) f_Ewald(G)
    V_LR(G) = LR_SIGN · phi(G)          # electron potential energy, eV
    V_LR(r) = Σ_{G∈𝒢} V_LR(G) e^(+iG·r);   V(G=0) = 0 (fixed gauge)
    f_Ewald(G) = exp(-(G·ε∞·G)/(4Λ²))

Λ is part of the dataset definition: the damped reciprocal-space sum alone
IS the LR definition (no compensating real-space term), so H^LR depends on
Λ by construction.  G-set requirements (inversion symmetry, G=0 excluded,
no duplicates) are hard: the realness of V^LR depends on them.
"""
import json
import os

import numpy as np

from .constants import C_COUL, LR_SIGN
from .convert import key_str, parse_key


def gmax_squared(lam, tol):
    """Bound on G·ε∞·G from the f_Ewald floor `tol`."""
    return 4.0 * float(lam) ** 2 * np.log(1.0 / float(tol))


def reciprocal_set(rec_cell, eps, gmax_sq):
    """Integer combinations of supercell reciprocal vectors inside the
    dielectric ellipsoid G·ε∞·G <= gmax_sq, G=0 excluded.  The symmetric
    cutoff makes the set inversion-symmetric by construction."""
    rec = np.asarray(rec_cell, float)
    eps = np.asarray(eps, float)
    eps_min = float(np.linalg.eigvalsh(0.5 * (eps + eps.T)).min())
    if eps_min <= 0.0:
        raise ValueError("dielectric tensor not positive definite")
    gmax_cart = np.sqrt(gmax_sq / eps_min)
    real = 2.0 * np.pi * np.linalg.inv(rec).T          # rows a_i
    nmax = [int(np.ceil(gmax_cart * np.linalg.norm(a) / (2.0 * np.pi)))
            for a in real]
    ns, gs = [], []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                if n1 == n2 == n3 == 0:
                    continue
                g = np.array([n1, n2, n3], float) @ rec
                if float(g @ eps @ g) <= gmax_sq:
                    ns.append((n1, n2, n3))
                    gs.append(g)
    return np.array(ns, int).reshape(-1, 3), np.array(gs).reshape(-1, 3)


def check_reciprocal_set(n_int):
    tuples = [tuple(int(x) for x in v) for v in np.asarray(n_int).reshape(-1, 3)]
    s = set(tuples)
    rep = {"number_of_vectors": len(tuples),
           "excludes_G_zero": (0, 0, 0) not in s,
           "no_duplicates": len(s) == len(tuples),
           "inversion_symmetric": all((-a, -b, -c) in s for a, b, c in s)}
    rep["ok"] = (rep["excludes_G_zero"] and rep["no_duplicates"]
                 and rep["inversion_symmetric"])
    return rep


def lr_coefficients(g_cart, dipoles, ref_positions, eps, lam, volume):
    """V_LR(G) with the reference-position phase convention (exactly linear
    in u^rel)."""
    g = np.asarray(g_cart, float)
    eps = np.asarray(eps, float)
    geg = np.einsum("ga,ab,gb->g", g, eps, g)
    f_ewald = np.exp(-geg / (4.0 * float(lam) ** 2))
    gd = g @ np.asarray(dipoles, float).T                       # (M,N)
    phases = np.exp(-1j * (g @ np.asarray(ref_positions, float).T))
    s_g = np.sum(gd * phases, axis=1)
    phi = -1j * (4.0 * np.pi / float(volume)) * C_COUL * s_g / geg * f_ewald
    return LR_SIGN * phi


def evaluate_potential(g_cart, coeffs, points):
    ph = np.exp(1j * (np.asarray(points, float) @ np.asarray(g_cart, float).T))
    return ph @ np.asarray(coeffs)


def imaginary_residual(v, delta):
    v = np.asarray(v)
    return float(np.linalg.norm(np.imag(v))
                 / (np.linalg.norm(np.real(v)) + float(delta)))


def minimum_image_displacements(cell, cart, ref_cart):
    """u = cart - ref wrapped to the nearest image (valid for |u| << cell)."""
    cell = np.asarray(cell, float)
    dfrac = (np.asarray(cart, float) - np.asarray(ref_cart, float)) \
        @ np.linalg.inv(cell)
    dfrac -= np.round(dfrac)
    return dfrac @ cell


def assemble_lr_hamiltonian(overlap_blocks, v_atom):
    """H^LR_ij(R) = (V_i + V_j)/2 * S_ij(R) over every stored overlap key.
    Hermiticity is inherited from S."""
    v_atom = np.asarray(v_atom, float)
    out = {}
    for k, s in overlap_blocks.items():
        _, _, _, i, j = parse_key(k)                    # 1-based
        out[k] = 0.5 * (v_atom[i - 1] + v_atom[j - 1]) * np.asarray(s, float)
    return out


def blocks_norm(blocks):
    return float(np.sqrt(sum(float(np.sum(v * v)) for v in blocks.values())))


def blocks_diff_norm(a, b):
    """Frobenius norm of a - b over the union of keys (absent -> zero)."""
    tot = 0.0
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        d = va - vb if va is not None and vb is not None \
            else (va if vb is None else -vb)
        tot += float(np.sum(d * d))
    return float(np.sqrt(tot))


_REFERENCE_DEFINITION_FILES = (
    "reference_cell.npy",
    "reference_positions.npy",
    "atomic_numbers.npy",
    "species_order.json",
    "born_effective_charges.npy",
    "dielectric_infinity.npy",
)


def reference_fingerprints(workspace):
    """Content hashes for every artifact that changes the physical LR labels."""
    from .config import sha256_file

    ref_dir = os.path.join(workspace, "reference")
    missing = [name for name in _REFERENCE_DEFINITION_FILES
               if not os.path.isfile(os.path.join(ref_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"LR reference artifacts missing from {ref_dir}: {missing}")
    return {name: sha256_file(os.path.join(ref_dir, name))
            for name in _REFERENCE_DEFINITION_FILES}


def _lr_definition(cfg, gmax_sq, reference_hashes):
    """Cell-invariant physical LR definition — the workspace compatibility key.

    The reciprocal-vector *count* depends on the supercell size and therefore
    is deliberately NOT part of this key (it is recorded per snapshot instead):
    including it would make one processed set reject every other cell size as a
    "different lr_definition".
    """
    return {"ewald_lambda": float(cfg["lr"]["ewald_lambda"]),
            "reciprocal_cutoff": float(gmax_sq),
            "reciprocal_tolerance": float(cfg["lr"]["reciprocal_tolerance"]),
            "reciprocal_set": {"inversion_symmetric": True,
                               "excludes_G_zero": True,
                               "cutoff_type": "dielectric_ellipsoid"},
            "imaginary_tolerance": float(cfg["lr"]["imaginary_tolerance"]),
            "gauge": "G_zero_equals_zero",
            "sign_convention": "electron_potential_energy",
            "phase_convention": "reference_positions",
            "reference_artifacts_sha256": dict(sorted(reference_hashes.items()))}


def expected_lr_definition(cfg, workspace):
    """The complete, cell-invariant LR identity expected in this workspace."""
    gmax_sq = gmax_squared(cfg["lr"]["ewald_lambda"],
                           cfg["lr"]["reciprocal_tolerance"])
    return _lr_definition(cfg, gmax_sq, reference_fingerprints(workspace))


def require_current_lr_definition(cfg, workspace):
    """Fail before publication if labels no longer match their references."""
    import yaml

    path = os.path.join(workspace, "metadata.yaml")
    if not os.path.isfile(path):
        raise SystemExit(
            "metadata.yaml is missing; run lr-process and validate first")
    with open(path) as handle:
        stored = (yaml.safe_load(handle) or {}).get("lr_definition")
    expected = expected_lr_definition(cfg, workspace)
    if stored != expected:
        raise SystemExit(
            "workspace lr_definition does not match the current LR config and "
            "reference artifacts; rerun in a clean workspace or restore the "
            "original references")
    return stored


def _record_lr_definition(workspace, lr_def):
    import yaml
    from .config import atomic_write_text
    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    stored = data.get("lr_definition")
    if stored is not None and stored != lr_def:
        raise SystemExit(
            "metadata.yaml already records a different lr_definition — "
            "refusing to mix LR definitions in one workspace (change the "
            "workspace or restore the original Λ/cutoff config)")
    units = {"energy": "eV", "length": "angstrom", "charge": "e"}
    if data.get("units") not in (None, units):
        raise SystemExit(
            "metadata.yaml records incompatible units; expected "
            "energy=eV, length=angstrom, charge=e")
    data["lr_definition"] = lr_def
    data["units"] = units
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def _invalidate_lr_outputs(folder, workspace):
    """Remove derived labels/exports after a failed forced LR calculation."""
    marker = os.path.join(folder, "export_metadata.json")
    exported_lr_target = False
    if os.path.isfile(marker):
        try:
            with open(marker) as f:
                exported_lr_target = json.load(f).get("target") in ("lr", "sr")
        except (json.JSONDecodeError, OSError):
            exported_lr_target = True
    target_path = os.path.join(folder, "hamiltonians.h5")
    if os.path.islink(target_path):
        exported_lr_target = (
            os.path.basename(os.readlink(target_path))
            in ("hamiltonians_lr.h5", "hamiltonians_sr.h5"))

    for name in ("hamiltonians_lr.h5", "hamiltonians_sr.h5",
                 "lr_metadata.json", "quality_checks.json"):
        path = os.path.join(folder, name)
        if os.path.lexists(path):
            os.remove(path)

    if exported_lr_target:
        for name in ("hamiltonians.h5", "export_metadata.json"):
            path = os.path.join(folder, name)
            if os.path.lexists(path):
                os.remove(path)

        import yaml
        from .config import atomic_write_text
        meta_path = os.path.join(workspace, "metadata.yaml")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                data = yaml.safe_load(f) or {}
            if data.get("training_target") in ("lr", "sr"):
                data.pop("training_target")
                atomic_write_text(meta_path,
                                  yaml.safe_dump(data, sort_keys=False))


def lr_process_stage(cfg, workspace, args):
    from .config import atomic_write_text
    from .convert import read_blocks, write_blocks
    from .displacements import remove_uniform_translation
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell, reciprocal
    from . import __version__

    if getattr(args, "set_name", None) is None:
        raise SystemExit("lr-process requires --set pilot|main|large")
    ref = load_reference(workspace)
    ref_dir = os.path.join(workspace, "reference")
    born = np.load(os.path.join(ref_dir, "born_effective_charges.npy"))
    eps = np.load(os.path.join(ref_dir, "dielectric_infinity.npy"))
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    lam = float(cfg["lr"]["ewald_lambda"])
    tol = float(cfg["lr"]["reciprocal_tolerance"])
    tau_imag = float(cfg["lr"]["imaginary_tolerance"])
    factor = float(cfg["lr"]["convergence_factor"])
    delta = float(cfg["validation"]["delta"])
    rec = reciprocal(sc.cell)
    volume = abs(float(np.linalg.det(sc.cell)))
    gmax_sq = gmax_squared(lam, tol)
    n_int, g_cart = reciprocal_set(rec, eps, gmax_sq)
    rep = check_reciprocal_set(n_int)
    if not rep["ok"] or rep["number_of_vectors"] == 0:
        raise SystemExit(f"reciprocal set invalid or empty: {rep}")
    n_int2, g2 = reciprocal_set(rec, eps, gmax_sq * factor ** 2)
    lr_def = _lr_definition(cfg, gmax_sq, reference_fingerprints(workspace))
    _record_lr_definition(workspace, lr_def)

    store = SnapshotStore(workspace, args.set_name)
    exit_code, processed, skipped = 0, 0, 0
    for sid in store.list():
        st = store.read_status(sid)
        if st["state"] == "rejected" \
                or not store.state_at_least(sid, "converted"):
            continue
        if store.state_at_least(sid, "lr_done") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        if args.force:
            # Transaction boundary: both symlink and copy exports must be
            # invalidated before rewriting their source labels.  If any later
            # read/calculation raises unexpectedly, the state remains safely
            # below lr_done and no stale derived label can be published.
            _invalidate_lr_outputs(folder, workspace)
            store.write_status(sid, "converted", lr_failed=None,
                               r_imag=None, lr_convergence=None,
                               lr_rerun_pending=True)
        pos = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        u = minimum_image_displacements(sc.cell, pos, sc.cart)
        u_stored = np.load(os.path.join(folder, "displacements.npy"))
        if np.abs(u - u_stored).max() > 1e-6:
            print(f"WARNING {sid}: recomputed u differs from "
                  f"displacements.npy by {np.abs(u - u_stored).max():.2e} Å")
        u_rel = remove_uniform_translation(u)          # processor-level ASR
        dipoles = np.einsum("nab,nb->na", born[sc.basis_index], u_rel)
        coeffs = lr_coefficients(g_cart, dipoles, sc.cart, eps, lam, volume)
        v_c = evaluate_potential(g_cart, coeffs, pos)  # snapshot AO centers
        r_imag = imaginary_residual(v_c, delta)
        if not np.isfinite(r_imag) or r_imag >= tau_imag:
            _invalidate_lr_outputs(folder, workspace)
            atomic_write_text(
                os.path.join(folder, "lr_failure.json"),
                json.dumps({"r_imag": r_imag, "reciprocal_set": rep,
                            "n_vectors": int(len(n_int)),
                            "lr_definition": lr_def}, indent=1))
            store.write_status(sid, "converted",
                               lr_failed=f"imaginary_residual {r_imag:.3e}",
                               r_imag=None, lr_convergence=None,
                               lr_rerun_pending=False)
            exit_code = 1
            continue
        v_atom = np.real(v_c)
        s_blocks = read_blocks(os.path.join(folder, "overlaps.h5"))
        h_full = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        h_lr = assemble_lr_hamiltonian(s_blocks, v_atom)
        coeffs2 = lr_coefficients(g2, dipoles, sc.cart, eps, lam, volume)
        v2 = np.real(evaluate_potential(g2, coeffs2, pos))
        h_lr2 = assemble_lr_hamiltonian(s_blocks, v2)
        conv = blocks_diff_norm(h_lr2, h_lr) / (blocks_norm(h_lr2) + delta)
        h_sr = {}
        for k in set(h_full) | set(h_lr):
            hf = h_full.get(k)
            hl = h_lr.get(k)
            if hf is None:
                hf = np.zeros_like(hl)
            if hl is None:
                hl = np.zeros_like(hf)
            h_sr[k] = hf - hl
        write_blocks(os.path.join(folder, "hamiltonians_lr.h5"), h_lr)
        write_blocks(os.path.join(folder, "hamiltonians_sr.h5"), h_sr)
        atomic_write_text(os.path.join(folder, "lr_metadata.json"),
                          json.dumps({"lr_definition": lr_def,
                                      "reciprocal_set": rep,
                                      "n_vectors": int(len(n_int)),
                                      "r_imag": r_imag,
                                      "lr_convergence": conv,
                                      "code_version": __version__}, indent=1))
        failure_path = os.path.join(folder, "lr_failure.json")
        if os.path.exists(failure_path):
            os.remove(failure_path)
        store.write_status(sid, "lr_done", r_imag=r_imag,
                           lr_convergence=conv, lr_failed=None,
                           lr_rerun_pending=False)
        processed += 1
    print(f"{args.set_name}: lr-processed {processed}, skipped {skipped}")
    return exit_code
