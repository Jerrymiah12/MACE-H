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
        q_mag = float(np.linalg.norm(
            np.asarray(modes[0]["q_int"], float) @ rec_super))
        amp = modes[0]["amplitude"]
        phase = modes[0]["phase"]
        pol_class = modes[0]["polarization_class"]
        weights = modes[0]["species_weights"]
    elif pattern.get("random") is not None:
        q_mag, amp, phase, pol_class = 0.0, pattern["random"]["amplitude"], 0.0, "none"
        weights = {"Mg": 1.0, "O": 1.0}
    else:
        q_mag, phase, pol_class = 0.0, 0.0, "none"
        weights = {"Mg": 0.0, "O": 0.0}
        amp = float(np.linalg.norm(pattern["translation"])) \
            if pattern.get("translation") is not None else 0.0
    ratio_sig = sorted((k, round(v, 8)) for k, v in weights.items())
    # Comparison family: matched |q|, amplitude, phase, normalization,
    # species ratio, supercell — but NOT polarization class, so matched
    # longitudinal/transverse partners share a family.
    family = _hash_id("fam", n, round(q_mag, 8), round(abs(amp), 8),
                      round(phase, 8), MODE_NORMALIZATION, ratio_sig)
    return {
        "pattern_group_id": group_id,
        "pattern_class": pattern["pattern_class"],
        "comparison_family_id": family,
        "mode_normalization": MODE_NORMALIZATION,
        "q_vectors": [m["q_int"] for m in modes],
        "q_magnitude": q_mag,
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
    q1 = [1, 0, 0]
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

    for name, q_int, pol, pol_class, weights in bases:
        gid = _hash_id("grp", "pilot", n, name, q_int, pol_class,
                       sorted(weights.items()))
        members = {}
        for amp in ladder:
            for sign in (1.0, -1.0):
                plan = add({"pattern_class": name, "modes": [
                    _mode(q_int, sign * amp, 0.0, pol, pol_class, weights)]},
                    gid)
                members[(amp, sign)] = plan
        for (amp, sign), plan in members.items():
            plan["metadata"]["sign_partner_id"] = members[(amp, -sign)]["sid"]
            plan["metadata"]["amplitude_partner_ids"] = [
                members[(a, sign)]["sid"] for a in ladder if a != amp]

    q2 = [0, 1, 0]
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


def _random_q(rng, n):
    while True:
        q = [int(rng.integers(0, n)) for _ in range(3)]
        if any(q):
            return q


def _low_q(rng, n):
    """Components 0 or ±1 (mod n): the longest wavelengths the cell holds."""
    while True:
        q = [int(rng.choice([0, 1, n - 1])) for _ in range(3)]
        if any(q):
            return q


def _single_q_pattern(rng, rec_super, q_int, amp, pattern_class):
    qhat = _unit(np.asarray(q_int, float) @ rec_super)
    if rng.random() < 0.5:
        pol, pol_class = qhat, "longitudinal"
    else:
        pol, pol_class = _transverse(qhat), "transverse"
    return {"pattern_class": pattern_class, "modes": [
        _mode(q_int, amp, float(rng.uniform(0.0, 2.0 * np.pi)),
              pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0})]}


def _mixed_pattern(rng, rec_super, n, amps, n_modes, pattern_class):
    modes = []
    for _ in range(n_modes):
        q_int = _low_q(rng, n)
        qhat = _unit(np.asarray(q_int, float) @ rec_super)
        if rng.random() < 0.5:
            pol, pol_class = qhat, "longitudinal"
        else:
            pol, pol_class = _transverse(qhat), "transverse"
        modes.append(_mode(q_int, float(rng.choice(amps[:2])),
                           float(rng.uniform(0.0, 2.0 * np.pi)),
                           pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0}))
    return {"pattern_class": pattern_class, "modes": modes}


def build_main(cfg, prim_cell):
    """Section-11 composition, one global seed, per-snapshot derived streams
    np.random.default_rng([seed, snapshot_index])."""
    n = cfg["supercells"]["main"]
    comp = cfg["displacements"]["main_composition"]
    amps = [float(a) for a in cfg["displacements"]["amplitudes"]]
    seed = cfg["displacements"]["seed"]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    plans, k = [], 1

    def add(pattern, group_id):
        nonlocal k
        plan = {"sid": _sid(k), "pattern": pattern,
                "metadata": _metadata(n, prim_cell, pattern, group_id, k)}
        plans.append(plan)
        k += 1
        return plan

    for _ in range(comp["single_q_optical"]):
        rng = np.random.default_rng([seed, k])
        add(_single_q_pattern(rng, rec_super, _random_q(rng, n),
                              float(rng.choice(amps)), "single_q_optical"),
            _hash_id("grp", "main", n, "single_q", k))

    for _ in range(comp["mixed_low_q"]):
        rng = np.random.default_rng([seed, k])
        add(_mixed_pattern(rng, rec_super, n, amps,
                           int(rng.integers(2, 5)), "mixed_low_q"),
            _hash_id("grp", "main", n, "mixed_low_q", k))

    for _ in range(comp["random_local"]):
        rng = np.random.default_rng([seed, k])
        add({"pattern_class": "random_local", "modes": [],
             "random": {"index": k, "amplitude": float(rng.choice(amps))}},
            _hash_id("grp", "main", n, "random_local", k))

    for _ in range(comp["sign_paired_calibration"] // 2):
        rng = np.random.default_rng([seed, k])
        gid = _hash_id("grp", "main", n, "sign_pair", k)
        base = _single_q_pattern(rng, rec_super, _random_q(rng, n),
                                 float(rng.choice(amps)),
                                 "sign_paired_calibration")
        plus = add(base, gid)
        neg = json.loads(json.dumps(base))
        neg["modes"][0]["amplitude"] *= -1.0
        minus = add(neg, gid)
        plus["metadata"]["sign_partner_id"] = minus["sid"]
        minus["metadata"]["sign_partner_id"] = plus["sid"]

    for _ in range(comp["near_equilibrium"]):
        rng = np.random.default_rng([seed, k])
        add({"pattern_class": "near_equilibrium", "modes": [],
             "random": {"index": k, "amplitude": 0.5 * min(amps)}},
            _hash_id("grp", "main", n, "near_equilibrium", k))
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
