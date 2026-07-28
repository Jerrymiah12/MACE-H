"""Validation battery.

Tier 1 (hard, per snapshot -> rejection + exit 1): structural/numerical and
algebraic checks.  Tier 2 (small-amplitude response): E_sign / E_linear must
decrease with decreasing amplitude within a pattern group; warnings unless
validation.tier2_enforce, in which case violations fail the SET (exit 1)
without rejecting individual snapshots.  Tier 3 lives in locality.py.
"""
import json
import os

import numpy as np

from .config import atomic_write_text, sha256_file
from .constants import ATOMIC_NUMBERS
from .convert import key_str, parse_key, read_blocks, species_orbital_info
from .displacements import remove_uniform_translation
from .lr import blocks_diff_norm, blocks_norm, minimum_image_displacements
from .snapshot import SnapshotStore, load_reference
from .structures import make_supercell

REQUIRED_FILES = ["STRU", "displacements.npy", "displacement_metadata.json",
                  "hamiltonians_full.h5", "overlaps.h5", "hamiltonians_lr.h5",
                  "hamiltonians_sr.h5", "lat.dat", "rlat.dat",
                  "site_positions.dat", "orbital_types.dat", "element.dat",
                  "info.json", "lr_metadata.json"]


def hermiticity_error(blocks):
    """max |H_ij(R) - H_ji(-R)^T|; inf if any block lacks its partner."""
    worst = 0.0
    for k, v in blocks.items():
        r0, r1, r2, i, j = parse_key(k)
        pk = key_str((-r0, -r1, -r2), j - 1, i - 1)
        if pk not in blocks:
            return float("inf")
        worst = max(worst, float(np.abs(v - blocks[pk].T).max()))
    return worst


def check_keys_and_dims(blocks, norb):
    """1-based indices in range; shapes match orbital counts; finite values."""
    n_at = len(norb)
    for k, v in blocks.items():
        r0, r1, r2, i, j = parse_key(k)
        if not (1 <= i <= n_at and 1 <= j <= n_at):
            return f"key {k}: atom index out of range (must be 1..{n_at})"
        if v.shape != (norb[i - 1], norb[j - 1]):
            return (f"key {k}: block shape {v.shape} != "
                    f"({norb[i - 1]}, {norb[j - 1]})")
        if not np.all(np.isfinite(v)):
            return f"key {k}: nan_or_inf"
    return None


def tier1_snapshot(cfg, folder, status, sc, born, ws_lr_def=None):
    val = cfg["validation"]
    delta = float(val["delta"])
    failures, metrics = [], {}
    missing = [f for f in REQUIRED_FILES
               if not os.path.exists(os.path.join(folder, f))]
    if missing:
        return [f"missing_file: {missing}"], metrics

    if not status.get("scf_converged"):
        failures.append("scf_not_converged_in_status")
    out_dir = os.path.join(folder, "OUT.MgO")
    for name, digest in status.get("raw_sha256", {}).items():
        p = os.path.join(out_dir, name)
        if not os.path.exists(p) or sha256_file(p) != digest:
            failures.append(f"raw_dft_modified: {name}")

    types, norb, _ = species_orbital_info(cfg, sc.species)
    h_full = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_lr = read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    h_sr = read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))
    s = read_blocks(os.path.join(folder, "overlaps.h5"))
    for name, blocks in (("full", h_full), ("lr", h_lr), ("sr", h_sr),
                         ("overlap", s)):
        err = check_keys_and_dims(blocks, norb)
        if err:
            failures.append(f"{name}: {err}")
    if failures:
        return failures, metrics

    ot = open(os.path.join(folder, "orbital_types.dat")).read().splitlines()
    expected_ot = ["  ".join(str(l) for l in t) for t in types]
    if [line.strip() for line in ot] != [line.strip() for line in expected_ot]:
        failures.append("orbital_types.dat contents disagree with the "
                        "species-major orbital layout")

    # element.dat: atomic numbers, one per atom, in the fixed species-major
    # order — the adversarial Mg->H swap must be caught here.
    expected_z = [ATOMIC_NUMBERS[s] for s in sc.species]
    try:
        got_z = [int(round(x)) for x in
                 np.atleast_1d(np.loadtxt(os.path.join(folder, "element.dat")))]
    except (ValueError, OSError):
        got_z = None
    if got_z != expected_z:
        failures.append(f"element.dat {got_z} != expected {expected_z}")

    info = json.load(open(os.path.join(folder, "info.json")))
    if (info.get("isspinful") is not False
            or info.get("nsites") != len(sc.species)
            or info.get("norbits") != int(sum(norb))):
        failures.append(f"info.json inconsistent: {info}")
    lat = np.loadtxt(os.path.join(folder, "lat.dat"))
    rlat = np.loadtxt(os.path.join(folder, "rlat.dat"))
    if not np.allclose(rlat.T @ lat, 2.0 * np.pi * np.eye(3), atol=1e-8):
        failures.append("rlat/lat convention violated (need rlat^T lat = 2 pi I)")
    # full lattice agreement with the expected supercell, not merely internal
    # rlat/lat consistency (a jointly-scaled cell would otherwise slip through)
    if not np.allclose(lat.T, sc.cell, atol=1e-6):
        failures.append("lat.dat disagrees with the expected supercell lattice")
    # DFT positions must equal reference geometry + the recorded displacement
    pos = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
    u_stored = np.load(os.path.join(folder, "displacements.npy"))
    pos_err = float(np.abs(
        minimum_image_displacements(sc.cell, pos, sc.cart) - u_stored).max())
    metrics["position_reference_error"] = pos_err
    if pos_err > float(val.get("tau_position", 1e-6)):
        failures.append(f"site_positions vs reference mismatch = {pos_err:.3e}")
    tau_diag = float(val["tau_overlap_diag"])
    for i in range(len(sc.species)):
        k = key_str((0, 0, 0), i, i)
        if k not in s or np.abs(np.diag(s[k]) - 1.0).max() > tau_diag:
            failures.append(f"overlap diagonal pathological at atom {i + 1}")
            break

    for name, blocks in (("full", h_full), ("lr", h_lr), ("sr", h_sr)):
        herm = hermiticity_error(blocks)
        metrics[f"hermiticity_{name}"] = herm
        if herm > float(val["tau_hermiticity"]):
            failures.append(f"hermiticity({name}) = {herm:.3e}")

    total = {}
    for k in set(h_sr) | set(h_lr):
        a, b = h_sr.get(k), h_lr.get(k)
        total[k] = a if b is None else b if a is None else a + b
    rec_err = blocks_diff_norm(total, h_full) / (blocks_norm(h_full) + delta)
    metrics["reconstruction_error"] = rec_err
    if rec_err > float(val["tau_reconstruct"]):
        failures.append(f"reconstruction_error = {rec_err:.3e}")

    lr_meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    rec = lr_meta.get("reciprocal_set", {})
    if not (rec.get("ok") and rec.get("inversion_symmetric")
            and rec.get("excludes_G_zero") and rec.get("no_duplicates")
            and int(rec.get("number_of_vectors", 0)) > 0):
        failures.append(f"reciprocal_set not verifiably sound: {rec}")
    if ws_lr_def is not None and lr_meta.get("lr_definition") != ws_lr_def:
        failures.append("lr_definition disagrees with workspace metadata.yaml")
    metrics["r_imag"] = lr_meta["r_imag"]
    metrics["lr_convergence"] = lr_meta["lr_convergence"]
    if lr_meta["r_imag"] >= float(cfg["lr"]["imaginary_tolerance"]):
        failures.append(f"imaginary_residual = {lr_meta['r_imag']:.3e}")
    if lr_meta["lr_convergence"] >= float(val["tau_G"]):
        failures.append(f"lr_convergence = {lr_meta['lr_convergence']:.3e}")

    dmeta = json.load(open(os.path.join(folder,
                                        "displacement_metadata.json")))
    lr_norm = blocks_norm(h_lr)
    metrics["lr_norm"] = lr_norm
    if dmeta.get("pattern_class") == "equilibrium" \
            and lr_norm > float(val["tau_eq"]):
        failures.append(f"equilibrium |H_LR| = {lr_norm:.3e}")
    if dmeta.get("rigid_translation"):
        u = np.load(os.path.join(folder, "displacements.npy"))
        u_rel = remove_uniform_translation(u)
        max_u = float(np.linalg.norm(u_rel, axis=1).max())
        metrics["translation_max_u_rel"] = max_u
        metrics["translation_max_dipole"] = float(np.abs(np.einsum(
            "nab,nb->na", born[sc.basis_index], u_rel)).max())
        if max_u > float(val["tau_u"]):
            failures.append(f"translation max|u_rel| = {max_u:.3e}")
        if lr_norm > float(val["tau_translation"]):
            failures.append(f"translation |H_LR| = {lr_norm:.3e}")
    return failures, metrics


def tier2_checks(store, cfg, sids):
    """E_sign per ± pair (counted once, from the positive member) and
    E_linear per amplitude-doubling pair; monotonicity violations per
    pattern group."""
    delta = float(cfg["validation"]["delta"])
    metas = {sid: json.load(open(os.path.join(
        store.folder(sid), "displacement_metadata.json"))) for sid in sids}
    cache = {}

    def lr_blocks(sid):
        if sid not in cache:
            cache[sid] = read_blocks(os.path.join(store.folder(sid),
                                                  "hamiltonians_lr.h5"))
        return cache[sid]

    e_sign, e_linear = [], []
    for sid, m in sorted(metas.items()):
        amp = float(m.get("amplitude") or 0.0)
        group = m.get("pattern_group_id")
        partner = m.get("sign_partner_id")
        if partner in metas and amp > 0.0:
            hp, hm = lr_blocks(sid), lr_blocks(partner)
            value = blocks_diff_norm(hp, {k: -v for k, v in hm.items()}) \
                / (blocks_norm(hp) + delta)
            e_sign.append({"group": group, "amplitude": amp,
                           "sids": [sid, partner], "value": value})
        for pid in m.get("amplitude_partner_ids", []):
            if pid not in metas or amp <= 0.0:
                continue
            amp2 = float(metas[pid].get("amplitude") or 0.0)
            if abs(amp2 - 2.0 * amp) < 1e-12:
                h1, h2 = lr_blocks(sid), lr_blocks(pid)
                value = blocks_diff_norm(
                    h2, {k: 2.0 * v for k, v in h1.items()}) \
                    / (2.0 * blocks_norm(h1) + delta)
                e_linear.append({"group": group, "amplitude": amp,
                                 "sids": [sid, pid], "value": value})
    violations = []
    for series, name in ((e_sign, "e_sign"), (e_linear, "e_linear")):
        by_group = {}
        for e in series:
            by_group.setdefault(e["group"], []).append(e)
        for group, entries in sorted(by_group.items()):
            entries.sort(key=lambda e: e["amplitude"])
            for lo, hi in zip(entries, entries[1:]):
                if lo["value"] > hi["value"]:
                    violations.append(
                        f"{name}[{group}]: {lo['value']:.3e} at "
                        f"A={lo['amplitude']} > {hi['value']:.3e} at "
                        f"A={hi['amplitude']} (must decrease with A)")
    return e_sign, e_linear, violations


def validate_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("validate requires --set pilot|main|large")
    ref = load_reference(workspace)
    born = np.load(os.path.join(workspace, "reference",
                                "born_effective_charges.npy"))
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"],
                        cfg["supercells"][args.set_name])
    ws_meta_path = os.path.join(workspace, "metadata.yaml")
    ws_lr_def = None
    if os.path.exists(ws_meta_path):
        import yaml
        with open(ws_meta_path) as f:
            ws_lr_def = (yaml.safe_load(f) or {}).get("lr_definition")
    store = SnapshotStore(workspace, args.set_name)
    exit_code, results = 0, {}
    for sid in store.list():
        st = store.read_status(sid)
        if st["state"] == "rejected" \
                or not store.state_at_least(sid, "lr_done"):
            continue
        failures, metrics = tier1_snapshot(cfg, store.folder(sid), st, sc,
                                           born, ws_lr_def)
        qc = {"tier1": {"failures": failures, "metrics": metrics}}
        atomic_write_text(os.path.join(store.folder(sid),
                                       "quality_checks.json"),
                          json.dumps(qc, indent=1))
        results[sid] = failures
        if failures:
            store.reject(sid, "; ".join(failures))
            exit_code = 1
        elif st["state"] != "validated":
            store.write_status(sid, "validated")
    survivors = [sid for sid in store.list()
                 if store.read_status(sid)["state"] == "validated"]
    e_sign, e_linear, violations = tier2_checks(store, cfg, survivors)
    enforced = bool(cfg["validation"]["tier2_enforce"])
    if enforced and violations:
        exit_code = 1
    summary = {"set": args.set_name,
               "counts": {"validated": len(survivors),
                          "rejected": sum(1 for f in results.values() if f)},
               "tier1": {sid: f for sid, f in sorted(results.items())},
               "tier2": {"e_sign": e_sign, "e_linear": e_linear,
                         "violations": violations, "enforced": enforced}}
    logs = os.path.join(workspace, "generation_logs")
    os.makedirs(logs, exist_ok=True)
    atomic_write_text(os.path.join(logs, f"validation_{args.set_name}.json"),
                      json.dumps(summary, indent=1))
    for v in violations:
        print(f"TIER2 {'FAIL' if enforced else 'WARN'}: {v}")
    print(f"{args.set_name}: validated {len(survivors)}, "
          f"rejected {summary['counts']['rejected']}")
    return exit_code
