import json

import numpy as np
import pytest

from mgo_lr import displacements as dp
from mgo_lr.config import load_config
from mgo_lr.structures import make_supercell, reciprocal, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")
PRIM_CELL, PRIM_FRAC, PRIM_SPECIES = rocksalt_primitive(4.2)


def _sc(n=2):
    return make_supercell(PRIM_CELL, PRIM_FRAC, PRIM_SPECIES, n)


def test_remove_uniform_translation_plain_mean():
    u = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = dp.remove_uniform_translation(u)
    assert np.allclose(out.mean(axis=0), 0.0)
    assert np.allclose(out, [[-1.0, 0, 0], [1.0, 0, 0]])  # NOT mass-weighted


def test_minimum_distance():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    assert abs(dp.minimum_distance(cell, cart) - 0.5) < 1e-12


def test_apply_optical_x():
    sc = _sc()
    pat = {"pattern_class": "optical_x", "modes": [{
        "q_int": [0, 0, 0], "amplitude": 0.01, "phase": 0.0,
        "polarization": [1.0, 0.0, 0.0], "polarization_class": "none",
        "species_weights": {"Mg": 1.0, "O": -1.0}}]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, CFG["displacements"]["seed"])
    assert np.allclose(u[:8], [0.01, 0.0, 0.0])   # all Mg +x
    assert np.allclose(u[8:], [-0.01, 0.0, 0.0])  # all O -x


def test_apply_finite_q_commensurate():
    sc = _sc(2)
    rec_super = reciprocal(sc.cell)
    qhat = np.asarray([1.0, 0, 0]) @ rec_super
    qhat /= np.linalg.norm(qhat)
    pat = {"pattern_class": "longitudinal_q", "modes": [{
        "q_int": [1, 0, 0], "amplitude": 0.01, "phase": 0.0,
        "polarization": qhat.tolist(), "polarization_class": "longitudinal",
        "species_weights": {"Mg": 1.0, "O": 0.0}}]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, 0)
    # q.R_l = 2*pi*(m.c)/n: cells with c1=0 get cos(0)=+1, c1=1 get cos(pi)=-1
    for i in range(8):  # Mg atoms
        expected_sign = 1.0 if sc.cell_index[i][0] == 0 else -1.0
        assert np.allclose(u[i], expected_sign * 0.01 * qhat, atol=1e-12)
    assert np.allclose(u[8:], 0.0)


def test_apply_rigid_translation_and_random():
    sc = _sc()
    pat = {"pattern_class": "rigid_translation", "modes": [],
           "translation": [0.01, 0.02, 0.03]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, 0)
    assert np.allclose(u, [0.01, 0.02, 0.03])
    pat = {"pattern_class": "random_local", "modes": [],
           "random": {"index": 1000, "amplitude": 0.01}}
    u1 = dp.apply_pattern(sc, PRIM_CELL, pat, 7)
    u2 = dp.apply_pattern(sc, PRIM_CELL, pat, 7)
    assert np.allclose(u1, u2)                       # seeded reproducibility
    assert np.allclose(u1.mean(axis=0), 0.0)         # translation removed
    assert abs(np.linalg.norm(u1, axis=1).max() - 0.01) < 1e-12


def test_build_pilot_contents():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    # Initial approval pilot: 1 equilibrium + optical ladder (8) + Mg sign
    # pair (2) + matched L/T (2) + 2 mixed + 2 random + 1 translation.
    assert len(plans) == 18
    assert plans[0]["metadata"]["pattern_class"] == "equilibrium"
    classes = [p["metadata"]["pattern_class"] for p in plans]
    assert classes.count("optical_x") == 8
    assert classes.count("mg_only_x") == 2
    assert classes.count("longitudinal_q") == 1
    assert classes.count("transverse_q") == 1
    assert classes.count("mixed") == 2
    assert classes.count("random_local") == 2
    assert classes.count("rigid_translation") == 1
    # deterministic
    again = dp.build_pilot(CFG, PRIM_CELL)
    assert json.dumps(plans, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_build_expanded_pilot_contents():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["displacements"]["pilot_expanded"] = True
    plans = dp.build_pilot(cfg, PRIM_CELL)
    assert len(plans) == 50
    classes = [p["metadata"]["pattern_class"] for p in plans]
    for name in ("mg_only_x", "o_only_x", "optical_x", "longitudinal_q",
                 "transverse_q"):
        assert classes.count(name) == 8
    assert classes.count("wavevector_trend") == 4
    q_families = {}
    for plan in plans:
        meta = plan["metadata"]
        if meta["wavevector_family_id"]:
            q_families.setdefault(meta["wavevector_family_id"], set()).add(
                round(meta["q_magnitude"], 10))
    assert sum(len(magnitudes) >= 2 for magnitudes in q_families.values()) == 2


def test_pilot_partner_wiring():
    plans = {p["sid"]: p["metadata"] for p in dp.build_pilot(CFG, PRIM_CELL)}
    for sid, meta in plans.items():
        if meta["sign_partner_id"]:
            partner = plans[meta["sign_partner_id"]]
            assert partner["sign_partner_id"] == sid
            assert partner["pattern_group_id"] == meta["pattern_group_id"]
            assert abs(partner["amplitude"] + meta["amplitude"]) < 1e-12
            assert partner["comparison_family_id"] == meta["comparison_family_id"]
        for pid in meta["amplitude_partner_ids"]:
            assert plans[pid]["pattern_group_id"] == meta["pattern_group_id"]


def test_pilot_longitudinal_transverse_share_family():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    lon = [p for p in plans if p["metadata"]["pattern_class"] == "longitudinal_q"
           and abs(p["metadata"]["amplitude"] - 0.01) < 1e-12]
    tra = [p for p in plans if p["metadata"]["pattern_class"] == "transverse_q"
           and abs(p["metadata"]["amplitude"] - 0.01) < 1e-12]
    assert lon and tra
    assert (lon[0]["metadata"]["comparison_family_id"]
            == tra[0]["metadata"]["comparison_family_id"])
    assert lon[0]["metadata"]["polarization_class"] == "longitudinal"
    assert tra[0]["metadata"]["polarization_class"] == "transverse"


def test_metadata_schema():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    keys = {"pattern_group_id", "pattern_class", "comparison_family_id",
            "mode_normalization", "q_vectors", "q_magnitude", "q_magnitudes",
            "polarizations",
            "polarization_class", "phases", "phase", "amplitudes", "amplitude",
            "sign_partner_id", "amplitude_partner_ids", "rigid_translation",
            "seed"}
    for p in plans:
        assert keys <= set(p["metadata"])
        json.dumps(p)  # everything JSON-serializable


def test_build_main_composition_and_seeding():
    plans = dp.build_main(CFG, PRIM_CELL)
    assert len(plans) == 400
    classes = [p["metadata"]["pattern_class"] for p in plans]
    comp = CFG["displacements"]["main_composition"]
    assert classes.count("single_q_optical") == comp["single_q_optical"]
    assert classes.count("mixed_low_q") == comp["mixed_low_q"]
    assert classes.count("random_local") == comp["random_local"]
    assert classes.count("sign_paired_calibration") == comp["sign_paired_calibration"]
    assert classes.count("near_equilibrium") == comp["near_equilibrium"]
    again = dp.build_main(CFG, PRIM_CELL)
    assert json.dumps(plans, sort_keys=True) == json.dumps(again, sort_keys=True)
    # sign pairs wired both ways
    by_sid = {p["sid"]: p["metadata"] for p in plans}
    pairs = [m for m in by_sid.values()
             if m["pattern_class"] == "sign_paired_calibration"]
    assert all(by_sid[m["sign_partner_id"]]["sign_partner_id"] for m in pairs)
    assert {p["metadata"]["split_hint"] for p in plans} == \
        {"train", "validation", "test"}
    subset_qs = {}
    for subset in ("train", "validation", "test"):
        subset_qs[subset] = {
            tuple(q) for p in plans if p["metadata"]["split_hint"] == subset
            for q in p["metadata"]["q_vectors"]}
    assert subset_qs["train"].isdisjoint(subset_qs["validation"])
    assert subset_qs["train"].isdisjoint(subset_qs["test"])
    assert subset_qs["validation"].isdisjoint(subset_qs["test"])


def test_fold_q_centers_indices():
    # a raw index of n-1 is the -1 direction, not +(n-1)
    assert dp.fold_q([2, 1, 0], 3) == [-1, 1, 0]
    assert dp.fold_q([0, 1, 2], 3) == [0, 1, -1]
    assert dp.fold_q([3, 2, 1], 4) == [-1, -2, 1]
    once = dp.fold_q([2, 2, 2], 3)
    assert dp.fold_q(once, 3) == once            # idempotent


def test_generated_q_vectors_are_canonical():
    # P0 regression: _low_q/_random_q returned raw indices (n-1 for -1), so the
    # unfolded vector drove qhat/polarization/q_magnitude in the WRONG Cartesian
    # direction.  Every stored q_int must lie in the centered reciprocal
    # interval so downstream directions and magnitudes are physical.
    for build, key in ((dp.build_main, "main"), (dp.build_large, "large")):
        n = CFG["supercells"][key]
        for p in build(CFG, PRIM_CELL):
            for q in p["metadata"]["q_vectors"]:
                for c in q:
                    assert -(n // 2) <= c <= n // 2, (key, n, q)


def test_longitudinal_polarization_parallel_to_folded_q():
    # a longitudinal single-q mode's polarization must be parallel to the
    # folded wavevector direction, including when the raw draw was n-1.
    n = CFG["supercells"]["large"]
    rec_super = reciprocal(np.asarray(PRIM_CELL, float) * n)
    for p in dp.build_large(CFG, PRIM_CELL):
        for m in p["pattern"]["modes"]:
            if m["polarization_class"] != "longitudinal":
                continue
            qhat = np.asarray(m["q_int"], float) @ rec_super
            qhat = qhat / np.linalg.norm(qhat)
            cos = abs(float(np.dot(qhat, m["polarization"])))
            assert cos > 1.0 - 1e-9, (m["q_int"], cos)


def test_build_large():
    plans = dp.build_large(CFG, PRIM_CELL)
    assert len(plans) == CFG["displacements"]["large_count"]
    for p in plans:
        assert p["metadata"]["pattern_class"] in ("single_q_optical", "mixed_low_q")
        for a in p["metadata"]["amplitudes"]:
            assert abs(a) <= max(CFG["displacements"]["amplitudes"]) + 1e-12
