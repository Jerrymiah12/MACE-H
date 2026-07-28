import json
import os

import numpy as np

from mgo_lr import convert, locality, validate
from mgo_lr.tests.test_convert import Args
from mgo_lr.tests.test_validate import ladder_workspace


def test_block_distance():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(locality.block_distance("[0, 0, 0, 1, 2]", cart, cell) - 1.0) < 1e-12
    assert abs(locality.block_distance("[1, 0, 0, 1, 1]", cart, cell) - 10.0) < 1e-12
    assert abs(locality.block_distance("[-1, 0, 0, 1, 2]", cart, cell) - 9.0) < 1e-12


def test_tail_fractions_ordering():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    blocks = {"[0, 0, 0, 1, 1]": np.array([[1.0]]),      # d = 0
              "[0, 0, 0, 1, 2]": np.array([[0.1]])}      # d = 4
    f = locality.tail_fractions(blocks, cart, cell, [1.0, 5.0])
    total = 1.0 + 0.01
    assert abs(f[0] - 0.01 / total) < 1e-12
    assert f[1] == 0.0
    b = locality.binned_norms(blocks, cart, cell, 1.0)
    assert b[0]["count"] == 1 and abs(b[0]["max"] - 1.0) < 1e-12


def test_odd_response_perfect_match():
    k = "[0, 0, 0, 1, 1]"
    h_p = {k: np.array([[2.0]])}
    h_m = {k: np.array([[-2.0]])}
    out = locality.odd_response(h_p, h_m, {k: np.array([[2.0]])}, 1e-12)
    assert abs(out["cos_theta"] - 1.0) < 1e-9
    assert abs(out["r_lr"] - 1.0) < 1e-9
    out2 = locality.odd_response(h_p, h_m, {k: np.array([[-2.0]])}, 1e-12)
    assert abs(out2["cos_theta"] + 1.0) < 1e-9


def test_locality_equal_tails_is_not_a_pass():
    # P1 regression: identical SR and full tails = NO localization improvement,
    # yet `s <= f + 1e-12` accepted it.  Equality must not PASS.
    f_full = [0.9, 0.7, 0.5, 0.3]
    assert locality.long_range_localizes(f_full, list(f_full), 1e-6, 0.05) is False


def test_locality_measurable_improvement_passes():
    f_full = [0.9, 0.7, 0.5, 0.3]
    f_sr = [0.9, 0.7, 0.20, 0.10]          # >5% lower in the long-distance half
    assert locality.long_range_localizes(f_full, f_sr, 1e-6, 0.05) is True


def test_locality_all_zero_long_range_is_not_a_pass():
    # radii beyond the largest nonzero block: both zero, no evidence -> not PASS
    f_full = [0.5, 0.2, 0.0, 0.0]
    assert locality.long_range_localizes(f_full, list(f_full), 1e-6, 0.05) is False


def test_controlled_q_comparisons_average_each_shell():
    metas = {
        "small-plus": {"wavevector_family_id": "qfam", "q_magnitude": 0.2,
                       "polarization_class": "longitudinal"},
        "small-minus": {"wavevector_family_id": "qfam", "q_magnitude": 0.2,
                        "polarization_class": "longitudinal"},
        "medium": {"wavevector_family_id": "qfam", "q_magnitude": 0.4,
                   "polarization_class": "longitudinal"},
        "large": {"wavevector_family_id": "qfam", "q_magnitude": 0.6,
                  "polarization_class": "longitudinal"},
    }
    norms = {"small-plus": 5.0, "small-minus": 3.0,
             "medium": 2.0, "large": 1.0}
    comparisons = locality.controlled_q_comparisons(metas, norms)
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert [shell["mean_lr_norm"] for shell in comparison["shells"]] == \
        [4.0, 2.0, 1.0]
    assert comparison["small_q_has_stronger_lr"] is True


def test_locality_report_stage(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    assert validate.validate_stage(cfg, ws, Args()) == 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    rep = json.load(open(os.path.join(
        ws, "generation_logs", "locality", "locality_pilot.json")))
    assert rep["n_snapshots"] == 4
    t = rep["tail"]
    assert len(t["radii"]) == len(t["F_full"]) == len(t["F_lr"]) == len(t["F_sr"])
    assert isinstance(t["f_sr_below_f_full"], bool)
    assert all(0.0 <= x <= 1.0 for x in t["F_full"])
    # first radii tail fractions are monotonically non-increasing
    assert all(a >= b - 1e-12 for a, b in zip(t["F_full"], t["F_full"][1:]))
    amps = sorted(round(e["amplitude"], 6) for e in rep["odd_response"])
    assert amps == [0.01, 0.02]               # one entry per ± pair
    for e in rep["odd_response"]:
        assert -1.0 - 1e-9 <= e["cos_theta"] <= 1.0 + 1e-9
        assert e["r_lr"] >= 0.0
    fam = rep["families"]["fam-test"]
    assert len(fam["members"]) == 4
    assert "mean_lr_norm_by_class" in fam


def test_locality_report_empty_set(tmp_path, capsys):
    ws, cfg, store = ladder_workspace(tmp_path)
    # nothing validated yet -> report nothing, still exit 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    assert "no validated" in capsys.readouterr().out
