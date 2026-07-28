"""Displacement-pattern engine and the gen-structures stage.

u_kappa(R_l) = sum_m A_m * w_m(species_kappa) * e_m * cos(q_m . R_l + phi_m)

R_l is the primitive-cell lattice vector of the atom's home cell
(cell_index @ prim_cell).  q vectors are integer combinations of the
SUPERCELL reciprocal vectors, so every pattern is commensurate by
construction.  Species weights are normalized so max|w| = 1, i.e. the mode
amplitude A is the peak displacement of the most-displaced species
("max_species_weight_1" normalization).

Uniform-translation removal is a plain (deliberately NOT mass-weighted)
mean over atoms.
"""
import hashlib
import json
import os

import numpy as np

from .structures import reciprocal

MODE_NORMALIZATION = "max_species_weight_1"


def _hash_id(prefix, *parts):
    text = json.dumps(parts, sort_keys=True, default=str)
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def _sid(k):
    return f"snapshot_{k:06d}"


def remove_uniform_translation(u):
    """Subtract the plain mean over atoms — deliberately not mass-weighted."""
    u = np.asarray(u, float)
    return u - u.mean(axis=0)


def minimum_distance(cell, cart):
    """Minimum interatomic distance under PBC (exact within +-1 image search)."""
    cell = np.asarray(cell, float)
    cart = np.asarray(cart, float)
    shifts = (np.array(list(np.ndindex(3, 3, 3))) - 1) @ cell
    dmin = np.inf
    for s in shifts:
        d = np.linalg.norm(cart[None, :, :] + s - cart[:, None, :], axis=-1)
        if np.allclose(s, 0.0):
            np.fill_diagonal(d, np.inf)
        dmin = min(dmin, float(d.min()))
    return dmin


def apply_pattern(sc, prim_cell, pattern, global_seed):
    """Displacement field (N,3) in Å for one pattern dict."""
    n_at = len(sc.species)
    if pattern.get("translation") is not None:
        return np.tile(np.asarray(pattern["translation"], float), (n_at, 1))
    if pattern.get("random") is not None:
        r = pattern["random"]
        rng = np.random.default_rng([global_seed, r["index"]])
        u = remove_uniform_translation(rng.standard_normal((n_at, 3)))
        return u * (r["amplitude"] / np.linalg.norm(u, axis=1).max())
    u = np.zeros((n_at, 3))
    rec_super = reciprocal(sc.cell)
    lattice_r = sc.cell_index @ np.asarray(prim_cell, float)
    for mode in pattern["modes"]:
        q = np.asarray(mode["q_int"], float) @ rec_super
        phase = np.cos(lattice_r @ q + mode["phase"])
        w = np.array([mode["species_weights"][s] for s in sc.species])
        u += mode["amplitude"] * (w * phase)[:, None] \
             * np.asarray(mode["polarization"], float)
    return u


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _transverse(qhat):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(qhat, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(qhat, ref))


def _mode(q_int, amplitude, phase, polarization, pol_class, weights):
    return {"q_int": [int(x) for x in q_int], "amplitude": float(amplitude),
            "phase": float(phase),
            "polarization": [float(x) for x in polarization],
            "polarization_class": pol_class,
            "species_weights": {k: float(v) for k, v in weights.items()}}


def _metadata(n, prim_cell, pattern, group_id, seed_index):
    modes = pattern.get("modes", [])
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    if modes:
        q_magnitudes = [float(np.linalg.norm(
            np.asarray(mode["q_int"], float) @ rec_super))
            for mode in modes]
        q_mag = q_magnitudes[0]
        amp = modes[0]["amplitude"]
        phase = modes[0]["phase"]
        pol_class = modes[0]["polarization_class"]
        weights = modes[0]["species_weights"]
    elif pattern.get("random") is not None:
        q_mag, amp, phase, pol_class = 0.0, pattern["random"]["amplitude"], 0.0, "none"
        q_magnitudes = []
        weights = {"Mg": 1.0, "O": 1.0}
    else:
        q_mag, phase, pol_class = 0.0, 0.0, "none"
        q_magnitudes = []
        weights = {"Mg": 0.0, "O": 0.0}
        amp = float(np.linalg.norm(pattern["translation"])) \
            if pattern.get("translation") is not None else 0.0
    ratio_sig = sorted((k, round(v, 8)) for k, v in weights.items())
    # Comparison family: matched |q|, amplitude, phase, normalization,
    # species ratio, supercell — but NOT polarization class, so matched
    # longitudinal/transverse partners share a family.
    family = _hash_id("fam", n, round(q_mag, 8), round(abs(amp), 8),
                      round(phase, 8), MODE_NORMALIZATION, ratio_sig)
    wave_family = None
    if len(modes) == 1:
        # Same controlled pattern except for |q|.  Locality diagnostics use
        # this identity for small-|q| versus large-|q| comparisons.
        wave_family = _hash_id(
            "qfam", n, round(abs(amp), 8), round(phase, 8),
            MODE_NORMALIZATION, ratio_sig, pol_class)
    return {
        "pattern_group_id": group_id,
        "pattern_class": pattern["pattern_class"],
        "comparison_family_id": family,
        "wavevector_family_id": wave_family,
        "mode_normalization": MODE_NORMALIZATION,
        "q_vectors": [m["q_int"] for m in modes],
        "q_magnitude": q_mag,
        "q_magnitudes": q_magnitudes,
        "polarizations": [m["polarization"] for m in modes],
        "polarization_class": pol_class,
        "phases": [m["phase"] for m in modes],
        "phase": phase,
        "amplitudes": [m["amplitude"] for m in modes] or [amp],
        "amplitude": amp,
        "sign_partner_id": None,
        "amplitude_partner_ids": [],
        "rigid_translation": bool(pattern.get("translation") is not None),
        "seed": seed_index,
    }


def build_pilot(cfg, prim_cell):
    """Deterministic Section-5 pilot list with amplitude ladders."""
    n = cfg["supercells"]["pilot"]
    ladder = [float(a) for a in cfg["displacements"]["pilot_ladder"]]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    x = [1.0, 0.0, 0.0]
    # n=1 is supported by tiny synthetic parser fixtures; it has no nonzero
    # folded q, so retain a reciprocal-lattice representative there.  The
    # production pilot is n=2 and always uses centered indices.
    q1 = fold_q([1, 0, 0], n) if n > 1 else [1, 0, 0]
    qhat = _unit(np.asarray(q1, float) @ rec_super)
    bases = [
        ("mg_only_x", [0, 0, 0], x, "none", {"Mg": 1.0, "O": 0.0}),
        ("o_only_x", [0, 0, 0], x, "none", {"Mg": 0.0, "O": 1.0}),
        ("optical_x", [0, 0, 0], x, "none", {"Mg": 1.0, "O": -1.0}),
        ("longitudinal_q", q1, qhat.tolist(), "longitudinal",
         {"Mg": 1.0, "O": -1.0}),
        ("transverse_q", q1, _transverse(qhat).tolist(), "transverse",
         {"Mg": 1.0, "O": -1.0}),
    ]
    plans, k = [], 1

    def add(pattern, group_id):
        nonlocal k
        plan = {"sid": _sid(k), "pattern": pattern,
                "metadata": _metadata(n, prim_cell, pattern, group_id, k)}
        plans.append(plan)
        k += 1
        return plan

    add({"pattern_class": "equilibrium", "modes": []},
        _hash_id("grp", "pilot", n, "equilibrium"))

    def add_signed_series(base, amplitudes):
        name, q_int, pol, pol_class, weights = base
        gid = _hash_id("grp", "pilot", n, name, q_int, pol_class,
                       sorted(weights.items()))
        members = {}
        for amp in amplitudes:
            for sign in (1.0, -1.0):
                plan = add({"pattern_class": name, "modes": [
                    _mode(q_int, sign * amp, 0.0, pol, pol_class, weights)]},
                    gid)
                members[(amp, sign)] = plan
        for (amp, sign), plan in members.items():
            plan["metadata"]["sign_partner_id"] = members[(amp, -sign)]["sid"]
            plan["metadata"]["amplitude_partner_ids"] = [
                members[(a, sign)]["sid"] for a in amplitudes if a != amp]

    if bool(cfg["displacements"]["pilot_expanded"]):
        for base in bases:
            add_signed_series(base, ladder)
        # Matched finite-q probes for the requested controlled |q| trend.
        # The signed q1 ladder above supplies the smallest shell.  These add
        # face- and body-diagonal shells at the same amplitude, phase, species
        # ratio, normalization, and polarization class, bringing the expanded
        # pilot to exactly 50 structures.
        calibration_amp = min(ladder, key=lambda a: abs(a - 0.01))
        for q_raw in ([1, 1, 0], [1, 1, 1]):
            q_int = fold_q(q_raw, n)
            qhat_probe = _unit(np.asarray(q_int, float) @ rec_super)
            for pol, pol_class in (
                    (qhat_probe, "longitudinal"),
                    (_transverse(qhat_probe), "transverse")):
                add({"pattern_class": "wavevector_trend", "modes": [
                    _mode(q_int, calibration_amp, 0.0, pol.tolist(),
                          pol_class, {"Mg": 1.0, "O": -1.0})]},
                    _hash_id("grp", "pilot", n, "wavevector_trend",
                             q_int, pol_class))
    else:
        # Initial approval pilot: one complete optical amplitude ladder, one
        # additional directional sign pair, and matched longitudinal/transverse
        # probes.  This exercises every required behavior in 18 structures;
        # setting pilot_expanded=true enables the 50-structure follow-up.
        by_name = {base[0]: base for base in bases}
        add_signed_series(by_name["optical_x"], ladder)
        calibration_amp = min(ladder, key=lambda a: abs(a - 0.01))
        add_signed_series(by_name["mg_only_x"], [calibration_amp])
        for name in ("longitudinal_q", "transverse_q"):
            base = by_name[name]
            bname, q_int, pol, pol_class, weights = base
            add({"pattern_class": bname, "modes": [
                _mode(q_int, calibration_amp, 0.0, pol, pol_class, weights)]},
                _hash_id("grp", "pilot", n, bname, q_int, pol_class,
                         sorted(weights.items())))

    q2 = fold_q([0, 1, 0], n) if n > 1 else [0, 1, 0]
    q2hat = _unit(np.asarray(q2, float) @ rec_super)
    for i, (a1, a2, ph2) in enumerate([(0.01, 0.005, 0.0),
                                       (0.01, 0.01, np.pi / 3)]):
        add({"pattern_class": "mixed", "modes": [
            _mode(q1, a1, 0.0, qhat.tolist(), "longitudinal",
                  {"Mg": 1.0, "O": -1.0}),
            _mode(q2, a2, ph2, _transverse(q2hat).tolist(), "transverse",
                  {"Mg": 1.0, "O": -1.0})]},
            _hash_id("grp", "pilot", n, "mixed", i))

    for i in range(2):
        add({"pattern_class": "random_local", "modes": [],
             "random": {"index": 1000 + i, "amplitude": 0.01}},
            _hash_id("grp", "pilot", n, "random_local", i))

    add({"pattern_class": "rigid_translation", "modes": [],
         "translation": ((0.02 / np.sqrt(3.0)) * np.ones(3)).tolist()},
        _hash_id("grp", "pilot", n, "rigid_translation"))
    return plans


def fold_q(q, n):
    """Canonicalize integer reciprocal indices into the centered interval
    (-n/2, n/2].  Folding leaves the commensurate displacement field unchanged
    (q·R differs by 2π·integer on every lattice site) but gives the TRUE
    wavevector direction and magnitude: a raw index of n-1 means the -1
    direction, not the +(n-1) direction.  All directions, magnitudes,
    polarizations, metadata, and family identities must be computed from the
    folded vector."""
    n = int(n)
    return [int(((int(c) + n // 2) % n) - n // 2) for c in q]


def _random_q(rng, n, candidates=None):
    if candidates:
        return list(candidates[int(rng.integers(0, len(candidates)))])
    while True:
        q = [int(rng.integers(0, n)) for _ in range(3)]
        if any(q):
            return fold_q(q, n)


def _low_q(rng, n, candidates=None):
    """Components 0 or ±1 (mod n): the longest wavelengths the cell holds."""
    if candidates:
        low = [q for q in candidates if all(abs(int(c)) <= 1 for c in q)]
        if not low:
            raise ValueError("split-specific q pool contains no low-q vectors")
        return list(low[int(rng.integers(0, len(low)))])
    while True:
        q = [int(rng.choice([0, 1, n - 1])) for _ in range(3)]
        if any(q):
            return fold_q(q, n)


def _single_q_pattern(rng, rec_super, q_int, amp, pattern_class):
    qhat = _unit(np.asarray(q_int, float) @ rec_super)
    if rng.random() < 0.5:
        pol, pol_class = qhat, "longitudinal"
    else:
        pol, pol_class = _transverse(qhat), "transverse"
    return {"pattern_class": pattern_class, "modes": [
        _mode(q_int, amp, float(rng.uniform(0.0, 2.0 * np.pi)),
              pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0})]}


def _mixed_pattern(rng, rec_super, n, amps, n_modes, pattern_class,
                   q_candidates=None):
    modes = []
    for _ in range(n_modes):
        q_int = _low_q(rng, n, q_candidates)
        qhat = _unit(np.asarray(q_int, float) @ rec_super)
        if rng.random() < 0.5:
            pol, pol_class = qhat, "longitudinal"
        else:
            pol, pol_class = _transverse(qhat), "transverse"
        modes.append(_mode(q_int, float(rng.choice(amps[:2])),
                           float(rng.uniform(0.0, 2.0 * np.pi)),
                           pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0}))
    return {"pattern_class": pattern_class, "modes": modes}


def _sample_split_hint(rng, cfg):
    """Deterministic approximate train/validation/test allocation."""
    test = float(cfg["splits"]["test_fraction"])
    val = float(cfg["splits"]["validation_fraction"])
    x = float(rng.random())
    if x < test:
        return "test"
    if x < test + val:
        return "validation"
    return "train"


def q_split_pools(n, rec_super, cfg):
    """Partition complete |q| shells before structure generation.

    Mixed-mode structures draw every constituent q from one split-specific
    pool.  This prevents the transitive shared-q graph from collapsing nearly
    all finite-q structures into one holdout component after generation.
    """
    vectors = sorted({tuple(fold_q(q, n)) for q in np.ndindex(n, n, n)}
                     - {(0, 0, 0)})
    shells = {}
    for q in vectors:
        magnitude = round(float(np.linalg.norm(np.asarray(q) @ rec_super)), 10)
        shells.setdefault(magnitude, []).append(q)
    if len(shells) < 3:
        raise ValueError(
            f"main supercell n={n} has only {len(shells)} |q| shells; "
            "cannot make leakage-safe train/validation/test q pools")
    shell_ids = sorted(shells)
    rng = np.random.default_rng([int(cfg["displacements"]["seed"]), 551903])
    rng.shuffle(shell_ids)
    n_shell = len(shell_ids)
    n_test = max(1, int(round(float(cfg["splits"]["test_fraction"]) * n_shell)))
    n_val = max(1, int(round(float(cfg["splits"]["validation_fraction"])
                             * n_shell)))
    if n_test + n_val >= n_shell:
        n_test = n_val = 1
    assignment = {
        "test": shell_ids[:n_test],
        "validation": shell_ids[n_test:n_test + n_val],
        "train": shell_ids[n_test + n_val:],
    }
    return {name: sorted(q for shell in selected for q in shells[shell])
            for name, selected in assignment.items()}


def build_main(cfg, prim_cell):
    """Section-11 composition, one global seed, per-snapshot derived streams
    np.random.default_rng([seed, snapshot_index])."""
    n = cfg["supercells"]["main"]
    comp = cfg["displacements"]["main_composition"]
    amps = [float(a) for a in cfg["displacements"]["amplitudes"]]
    seed = cfg["displacements"]["seed"]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    q_pools = q_split_pools(n, rec_super, cfg)
    plans, k = [], 1

    def add(pattern, group_id, split_hint):
        nonlocal k
        plan = {"sid": _sid(k), "pattern": pattern,
                "metadata": _metadata(n, prim_cell, pattern, group_id, k)}
        plan["metadata"]["split_hint"] = split_hint
        plans.append(plan)
        k += 1
        return plan

    for _ in range(comp["single_q_optical"]):
        rng = np.random.default_rng([seed, k])
        hint = _sample_split_hint(rng, cfg)
        add(_single_q_pattern(rng, rec_super,
                              _random_q(rng, n, q_pools[hint]),
                              float(rng.choice(amps)), "single_q_optical"),
            _hash_id("grp", "main", n, "single_q", k), hint)

    for _ in range(comp["mixed_low_q"]):
        rng = np.random.default_rng([seed, k])
        hint = _sample_split_hint(rng, cfg)
        add(_mixed_pattern(rng, rec_super, n, amps,
                           int(rng.integers(2, 5)), "mixed_low_q",
                           q_pools[hint]),
            _hash_id("grp", "main", n, "mixed_low_q", k), hint)

    for _ in range(comp["random_local"]):
        rng = np.random.default_rng([seed, k])
        hint = _sample_split_hint(rng, cfg)
        add({"pattern_class": "random_local", "modes": [],
             "random": {"index": k, "amplitude": float(rng.choice(amps))}},
            _hash_id("grp", "main", n, "random_local", k), hint)

    for _ in range(comp["sign_paired_calibration"] // 2):
        rng = np.random.default_rng([seed, k])
        hint = _sample_split_hint(rng, cfg)
        gid = _hash_id("grp", "main", n, "sign_pair", k)
        base = _single_q_pattern(rng, rec_super,
                                 _random_q(rng, n, q_pools[hint]),
                                 float(rng.choice(amps)),
                                 "sign_paired_calibration")
        plus = add(base, gid, hint)
        neg = json.loads(json.dumps(base))
        neg["modes"][0]["amplitude"] *= -1.0
        minus = add(neg, gid, hint)
        plus["metadata"]["sign_partner_id"] = minus["sid"]
        minus["metadata"]["sign_partner_id"] = plus["sid"]

    for _ in range(comp["near_equilibrium"]):
        rng = np.random.default_rng([seed, k])
        hint = _sample_split_hint(rng, cfg)
        add({"pattern_class": "near_equilibrium", "modes": [],
             "random": {"index": k, "amplitude": 0.5 * min(amps)}},
            _hash_id("grp", "main", n, "near_equilibrium", k), hint)
    return plans


def build_large(cfg, prim_cell):
    """4x4x4 extrapolation set: small q, longitudinal optical, mixed
    long-wavelength; amplitudes within the main-set range.  Derived seed
    streams use index 900000+k to stay disjoint from the main set."""
    n = cfg["supercells"]["large"]
    count = cfg["displacements"]["large_count"]
    amps = [float(a) for a in cfg["displacements"]["amplitudes"]]
    seed = cfg["displacements"]["seed"]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    plans = []
    for k in range(1, count + 1):
        rng = np.random.default_rng([seed, 900000 + k])
        if rng.random() < 0.5:
            q_int = _low_q(rng, n)
            qhat = _unit(np.asarray(q_int, float) @ rec_super)
            pat = {"pattern_class": "single_q_optical", "modes": [
                _mode(q_int, float(rng.choice(amps[:3])),
                      float(rng.uniform(0.0, 2.0 * np.pi)),
                      qhat.tolist(), "longitudinal", {"Mg": 1.0, "O": -1.0})]}
        else:
            pat = _mixed_pattern(rng, rec_super, n, amps,
                                 int(rng.integers(2, 4)), "mixed_low_q")
        plans.append({"sid": _sid(k), "pattern": pat,
                      "metadata": _metadata(n, prim_cell, pat,
                                            _hash_id("grp", "large", n, k),
                                            900000 + k)})
    return plans


def gen_structures_stage(cfg, workspace, args):
    from . import abacus_io
    from .config import atomic_write_text
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell

    if getattr(args, "set_name", None) is None:
        raise SystemExit("gen-structures requires --set pilot|main|large")
    ref = load_reference(workspace)
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    builders = {"pilot": build_pilot, "main": build_main, "large": build_large}
    plans = builders[args.set_name](cfg, ref["prim_cell"])
    store = SnapshotStore(workspace, args.set_name)
    os.makedirs(store.set_dir, exist_ok=True)
    seed = cfg["displacements"]["seed"]
    min_d = float(cfg["displacements"]["min_distance"])
    written = 0
    for plan in plans:
        sid, folder = plan["sid"], store.folder(plan["sid"])
        if store.is_rejected(sid):
            continue                       # never recreate a rejected snapshot
        if os.path.isdir(folder):
            if not args.force:
                continue
            if any(d.startswith("OUT.") for d in os.listdir(folder)):
                print(f"{sid}: has DFT output; refusing to regenerate")
                continue
        pattern = json.loads(json.dumps(plan["pattern"]))
        meta = dict(plan["metadata"])
        u = apply_pattern(sc, ref["prim_cell"], pattern, seed)
        if pattern["modes"]:
            u = remove_uniform_translation(u)
        attempts = 0
        while minimum_distance(sc.cell, sc.cart + u) < min_d:
            if pattern.get("random") is None:
                raise ValueError(
                    f"{args.set_name}/{sid}: minimum interatomic distance "
                    f"{minimum_distance(sc.cell, sc.cart + u):.3f} Å "
                    f"< {min_d} Å for a deterministic pattern")
            attempts += 1
            if attempts > 100:
                raise ValueError(f"{sid}: no valid random draw in 100 tries")
            pattern["random"]["index"] += 100000
            meta["seed"] = pattern["random"]["index"]
            u = apply_pattern(sc, ref["prim_cell"], pattern, seed)
        os.makedirs(folder, exist_ok=True)
        abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell,
                             sc.cart + u, sc.species, cfg)
        abacus_io.write_input(os.path.join(folder, "INPUT"), cfg,
                              calculation="scf", out_mat_hs2=1, suffix="MgO")
        abacus_io.write_kpt(os.path.join(folder, "KPT"),
                            cfg["abacus"]["kmesh_supercell"][args.set_name])
        np.save(os.path.join(folder, "displacements.npy"), u)
        meta["pattern"] = pattern
        atomic_write_text(os.path.join(folder, "displacement_metadata.json"),
                          json.dumps(meta, indent=1))
        store.write_status(sid, "prepared", set_name=args.set_name)
        written += 1
    abacus_io.write_job_script(
        os.path.join(store.set_dir, "job_abacus.sh"), cfg,
        [p["sid"] for p in plans if not store.is_rejected(p["sid"])])
    print(f"{args.set_name}: wrote {written} snapshots "
          f"({len(plans) - written} skipped)")
    return 0
