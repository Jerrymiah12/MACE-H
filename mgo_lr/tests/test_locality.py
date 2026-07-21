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
