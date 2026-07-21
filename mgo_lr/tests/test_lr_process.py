import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import convert, lr
from mgo_lr.config import load_config
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell
from mgo_lr.tests.test_convert import (Args, fabricate_dft, prepared_snapshot,
                                       small_cfg)
from mgo_lr.tests.test_gen_structures import make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")


def lr_cfg():
    cfg = small_cfg()
    cfg["lr"]["ewald_lambda"] = 1.0     # non-empty G set on the 2-atom cell
    return cfg


def add_dfpt_artifacts(ws):
    ref = os.path.join(ws, "reference")
    z = np.array([np.eye(3) * 1.97, np.eye(3) * -1.97])
    np.save(os.path.join(ref, "born_effective_charges.npy"), z)
    np.save(os.path.join(ref, "dielectric_infinity.npy"), np.eye(3) * 3.0)


def converted_snapshot(tmp_path, u=None):
    ws = str(tmp_path)
    cfg = lr_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg, u=u)
    add_dfpt_artifacts(ws)
    fabricate_dft(store.folder(sid), cfg, sc)
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    return ws, cfg, store, sid, sc


def test_lr_process_equilibrium_zero(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)     # u = 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    folder = store.folder(sid)
    h_lr = convert.read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) < 1e-10                # H_LR(u=0) = 0
    h_full = convert.read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_sr = convert.read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))
    assert lr.blocks_diff_norm(
        {k: h_sr.get(k, 0) + h_lr.get(k, 0) * 0 for k in h_sr}, {}) > 0
    # reconstruction: H_SR + H_LR = H_full on the union
    total = {k: h_sr.get(k, np.zeros_like(h_lr.get(k)))
             + h_lr.get(k, np.zeros_like(h_sr.get(k)))
             for k in set(h_sr) | set(h_lr)}
    assert lr.blocks_diff_norm(total, h_full) < 1e-10
    meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    ld = meta["lr_definition"]
    assert ld["gauge"] == "G_zero_equals_zero"
    assert ld["sign_convention"] == "electron_potential_energy"
    assert ld["phase_convention"] == "reference_positions"
    assert ld["reciprocal_set"]["inversion_symmetric"] is True
    assert ld["reciprocal_set"]["number_of_vectors"] > 0
    assert meta["r_imag"] < cfg["lr"]["imaginary_tolerance"]
    assert store.read_status(sid)["state"] == "lr_done"
    ws_meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert ws_meta["lr_definition"]["ewald_lambda"] == 1.0


def test_lr_process_translation_zero(tmp_path):
    u = np.tile([[0.02, 0.01, -0.01]], (2, 1))
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path, u=u)
    lr.lr_process_stage(cfg, ws, Args())
    h_lr = convert.read_blocks(
        os.path.join(store.folder(sid), "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) < 1e-10        # exact zero by construction


def test_lr_process_nonzero_for_optical(tmp_path):
    u = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path, u=u)
    lr.lr_process_stage(cfg, ws, Args())
    folder = store.folder(sid)
    h_lr = convert.read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) > 1e-6
    # hermiticity inherited from S
    for k, v in h_lr.items():
        r0, r1, r2, i, j = convert.parse_key(k)
        pk = convert.key_str((-r0, -r1, -r2), j - 1, i - 1)
        assert np.allclose(v, h_lr[pk].T, atol=1e-10)
    meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    assert meta["lr_convergence"] < cfg["validation"]["tau_G"]


def test_lr_process_idempotent_and_lambda_guard(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)
    lr.lr_process_stage(cfg, ws, Args())
    before = store.read_status(sid)["history"]
    lr.lr_process_stage(cfg, ws, Args())               # skip, no --force
    assert store.read_status(sid)["history"] == before
    cfg2 = copy.deepcopy(cfg)
    cfg2["lr"]["ewald_lambda"] = 0.5
    args = Args()
    args.force = True
    with pytest.raises(SystemExit, match="lr_definition"):
        lr.lr_process_stage(cfg2, ws, args)            # refuses to mix Λ


def test_lr_process_imaginary_gate(tmp_path, monkeypatch):
    ws, cfg, store, sid, sc = converted_snapshot(
        tmp_path, u=np.array([[0.01, 0, 0], [-0.01, 0, 0]]))

    def broken(g, c, pts):
        return np.full(len(np.atleast_2d(pts)), 1.0 + 1.0j)

    monkeypatch.setattr(lr, "evaluate_potential", broken)
    assert lr.lr_process_stage(cfg, ws, Args()) == 1
    folder = store.folder(sid)
    assert os.path.exists(os.path.join(folder, "lr_failure.json"))
    assert not os.path.exists(os.path.join(folder, "hamiltonians_lr.h5"))
    assert not os.path.exists(os.path.join(folder, "hamiltonians_sr.h5"))
    assert "lr_failed" in store.read_status(sid)
