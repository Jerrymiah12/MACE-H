import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import convert, export, lr
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
    assert meta["reciprocal_set"]["number_of_vectors"] > 0   # realized count
    assert meta["r_imag"] < cfg["lr"]["imaginary_tolerance"]
    assert store.read_status(sid)["state"] == "lr_done"
    ws_meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert ws_meta["lr_definition"]["ewald_lambda"] == 1.0
    fingerprints = ws_meta["lr_definition"]["reference_artifacts_sha256"]
    assert set(fingerprints) == set(lr._REFERENCE_DEFINITION_FILES)
    assert all(len(digest) == 64 for digest in fingerprints.values())


def test_workspace_lr_definition_is_cell_size_invariant(tmp_path):
    # P0 regression: the reciprocal-vector count depends on supercell size, so
    # embedding it in the workspace compatibility key makes processing a second
    # set (e.g. main after pilot) abort as a "different lr_definition".  The
    # workspace key must contain only cell-invariant physical parameters; the
    # realized count belongs in per-snapshot provenance instead.
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    stored = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))["lr_definition"]
    assert "number_of_vectors" not in stored["reciprocal_set"]
    meta = json.load(open(os.path.join(store.folder(sid), "lr_metadata.json")))
    assert meta["reciprocal_set"]["number_of_vectors"] > 0
    assert meta["reciprocal_set"]["inversion_symmetric"] is True


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


def test_lr_process_rejects_changed_reference_artifact(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    born_path = os.path.join(ws, "reference", "born_effective_charges.npy")
    born = np.load(born_path)
    born[0, 0, 0] += 0.01
    np.save(born_path, born)
    args = Args()
    args.force = True
    with pytest.raises(SystemExit, match="lr_definition"):
        lr.lr_process_stage(cfg, ws, args)


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


def test_failed_forced_rerun_invalidates_old_labels_and_export(
        tmp_path, monkeypatch):
    ws, cfg, store, sid, sc = converted_snapshot(
        tmp_path, u=np.array([[0.01, 0, 0], [-0.01, 0, 0]]))
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    export_args = Args()
    export_args.target = "sr"
    assert export.export_target_stage(cfg, ws, export_args) == 0

    def broken(g, c, pts):
        return np.full(len(np.atleast_2d(pts)), 1.0 + 1.0j)

    monkeypatch.setattr(lr, "evaluate_potential", broken)
    force_args = Args()
    force_args.force = True
    assert lr.lr_process_stage(cfg, ws, force_args) == 1
    folder = store.folder(sid)
    for name in ("hamiltonians_lr.h5", "hamiltonians_sr.h5",
                 "hamiltonians.h5", "lr_metadata.json",
                 "export_metadata.json", "quality_checks.json"):
        assert not os.path.lexists(os.path.join(folder, name))
    status = store.read_status(sid)
    assert status["state"] == "converted"
    assert status["lr_failed"]
    assert status["r_imag"] is None
    assert status["lr_convergence"] is None
    metadata = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert "training_target" not in metadata


def test_unexpected_forced_rerun_error_leaves_no_stale_labels(
        tmp_path, monkeypatch):
    ws, cfg, store, sid, sc = converted_snapshot(
        tmp_path, u=np.array([[0.01, 0, 0], [-0.01, 0, 0]]))
    assert lr.lr_process_stage(cfg, ws, Args()) == 0

    def crash(g, c, pts):
        raise RuntimeError("synthetic evaluator crash")

    monkeypatch.setattr(lr, "evaluate_potential", crash)
    force_args = Args()
    force_args.force = True
    with pytest.raises(RuntimeError, match="synthetic evaluator crash"):
        lr.lr_process_stage(cfg, ws, force_args)
    folder = store.folder(sid)
    for name in ("hamiltonians_lr.h5", "hamiltonians_sr.h5",
                 "lr_metadata.json", "quality_checks.json"):
        assert not os.path.lexists(os.path.join(folder, name))
    status = store.read_status(sid)
    assert status["state"] == "converted"
    assert status["lr_rerun_pending"] is True
