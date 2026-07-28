import copy
import json
import os

import numpy as np
import pytest

from mgo_lr import abacus_io, convert, lr, validate
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell
from mgo_lr.tests.test_convert import Args, fabricate_dft
from mgo_lr.tests.test_gen_structures import make_fake_reference
from mgo_lr.tests.test_lr_process import add_dfpt_artifacts, lr_cfg


def add_snapshot(ws, cfg, sc, sid, u, meta):
    """Prepared snapshot with fabricated DFT output and explicit metadata."""
    store = SnapshotStore(ws, "pilot")
    folder = store.folder(sid)
    os.makedirs(folder, exist_ok=True)
    abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell, sc.cart + u,
                         sc.species, cfg)
    np.save(os.path.join(folder, "displacements.npy"), u)
    base = {"pattern_class": "single_q_optical", "pattern_group_id": "grp-test",
            "comparison_family_id": "fam-test", "rigid_translation": False,
            "sign_partner_id": None, "amplitude_partner_ids": [],
            "amplitude": 0.0, "polarization_class": "none", "q_magnitude": 0.0}
    base.update(meta)
    with open(os.path.join(folder, "displacement_metadata.json"), "w") as f:
        json.dump(base, f)
    store.write_status(sid, "prepared")
    fabricate_dft(folder, cfg, sc)
    return store


def ladder_workspace(tmp_path):
    """Four optical snapshots ±A / ±2A wired as sign and amplitude partners,
    taken through collect-dft and lr-process."""
    ws = str(tmp_path)
    cfg = lr_cfg()
    cell, frac, species = make_fake_reference(ws)
    # asymmetric basis: with the ideal rocksalt basis the 2-atom ± patterns
    # give an LR potential EVEN in A (E_sign ~ 2), so sign pairing would be
    # meaningless — same degeneracy as in test_lr_core's amplitude test
    frac = np.array([[0.0, 0.0, 0.0], [0.4, 0.55, 0.5]])
    np.save(os.path.join(ws, "reference", "reference_positions.npy"), frac)
    add_dfpt_artifacts(ws)
    sc = make_supercell(cell, frac, species, cfg["supercells"]["pilot"])
    x = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    wiring = [("snapshot_000001", 0.01, "snapshot_000002", ["snapshot_000003"]),
              ("snapshot_000002", -0.01, "snapshot_000001", ["snapshot_000004"]),
              ("snapshot_000003", 0.02, "snapshot_000004", ["snapshot_000001"]),
              ("snapshot_000004", -0.02, "snapshot_000003", ["snapshot_000002"])]
    for sid, amp, partner, amp_partners in wiring:
        add_snapshot(ws, cfg, sc, sid, amp * x,
                     {"amplitude": amp, "sign_partner_id": partner,
                      "amplitude_partner_ids": amp_partners})
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    return ws, cfg, SnapshotStore(ws, "pilot")


def test_validate_passes_clean_set(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    assert validate.validate_stage(cfg, ws, Args()) == 0
    for sid in store.list():
        assert store.read_status(sid)["state"] == "validated"
        qc = json.load(open(os.path.join(store.folder(sid),
                                         "quality_checks.json")))
        assert qc["tier1"]["failures"] == []
        assert (qc["tier1"]["metrics"]["reconstruction_error"]
                < cfg["validation"]["tau_reconstruct"])
    summary = json.load(open(os.path.join(ws, "generation_logs",
                                          "validation_pilot.json")))
    assert summary["counts"]["validated"] == 4
    amp_to_val = {round(abs(e["amplitude"]), 6): e["value"]
                  for e in summary["tier2"]["e_sign"]}
    assert set(amp_to_val) == {0.01, 0.02}
    assert 0.0 <= amp_to_val[0.01] < amp_to_val[0.02]   # decreasing with A
    assert summary["tier2"]["e_linear"]
    assert all(e["value"] >= 0.0 for e in summary["tier2"]["e_linear"])
    assert summary["tier2"]["violations"] == []


def test_validate_equilibrium_and_translation(tmp_path):
    ws = str(tmp_path)
    cfg = lr_cfg()
    cell, frac, species = make_fake_reference(ws)
    add_dfpt_artifacts(ws)
    sc = make_supercell(cell, frac, species, cfg["supercells"]["pilot"])
    add_snapshot(ws, cfg, sc, "snapshot_000001", np.zeros((2, 3)),
                 {"pattern_class": "equilibrium"})
    add_snapshot(ws, cfg, sc, "snapshot_000002",
                 np.tile([[0.02, 0.01, -0.01]], (2, 1)),
                 {"pattern_class": "rigid_translation",
                  "rigid_translation": True})
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    assert validate.validate_stage(cfg, ws, Args()) == 0
    store = SnapshotStore(ws, "pilot")
    qc = json.load(open(os.path.join(store.folder("snapshot_000002"),
                                     "quality_checks.json")))
    m = qc["tier1"]["metrics"]
    assert m["translation_max_u_rel"] < cfg["validation"]["tau_u"]
    assert m["lr_norm"] < cfg["validation"]["tau_translation"]
    qc0 = json.load(open(os.path.join(store.folder("snapshot_000001"),
                                      "quality_checks.json")))
    assert qc0["tier1"]["metrics"]["lr_norm"] < cfg["validation"]["tau_eq"]


def _rejected_reason(ws, sid):
    p = os.path.join(ws, "rejected", f"pilot_{sid}", "status.json")
    return json.load(open(p))["reason"]


def test_validate_rejects_nan(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "hamiltonians_lr.h5")
    blocks = convert.read_blocks(p)
    blocks[next(iter(sorted(blocks)))][0, 0] = np.nan
    convert.write_blocks(p, blocks)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert sid not in store.list()
    assert "nan" in _rejected_reason(ws, sid)


def test_validate_rejects_nan_position(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    path = os.path.join(store.folder(sid), "site_positions.dat")
    positions = np.loadtxt(path)
    positions[0, 0] = np.nan
    np.savetxt(path, positions)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "NaN" in _rejected_reason(ws, sid)


def test_validate_rejects_missing_file(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    os.remove(os.path.join(store.folder(sid), "lat.dat"))
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "missing_file" in _rejected_reason(ws, sid)


def test_validate_rejects_missing_workspace_metadata(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    os.remove(os.path.join(ws, "metadata.yaml"))
    assert validate.validate_stage(cfg, ws, Args()) == 1
    for sid in ("snapshot_000001", "snapshot_000002",
                "snapshot_000003", "snapshot_000004"):
        assert "lr_definition" in _rejected_reason(ws, sid)


def test_validate_rejects_missing_raw_hash_provenance(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    status_path = os.path.join(store.folder(sid), "status.json")
    status = json.load(open(status_path))
    status["raw_sha256"].pop("running_scf.log")
    with open(status_path, "w") as handle:
        json.dump(status, handle)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "raw_dft_sha256" in _rejected_reason(ws, sid)


def test_validate_rejects_modified_raw(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    with open(os.path.join(store.folder(sid), "OUT.MgO",
                           cfg["abacus"]["csr_h_filename"]), "a") as f:
        f.write("# tampered\n")
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "raw_dft_modified" in _rejected_reason(ws, sid)


def test_validate_rejects_broken_reconstruction(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "hamiltonians_sr.h5")
    blocks = convert.read_blocks(p)
    k = convert.key_str((0, 0, 0), 0, 0)     # self-paired: hermiticity survives
    blocks[k] = blocks[k] + 1.0
    convert.write_blocks(p, blocks)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "reconstruction" in _rejected_reason(ws, sid)


def test_validate_rejects_broken_hermiticity_and_rlat(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid_h, sid_r = store.list()[0], store.list()[1]
    p = os.path.join(store.folder(sid_h), "hamiltonians_lr.h5")
    blocks = convert.read_blocks(p)
    k = convert.key_str((1, 0, 0), 0, 1)     # partner (-1,0,0),2,1 untouched
    blocks[k] = blocks[k] + 1.0
    convert.write_blocks(p, blocks)
    rlat_path = os.path.join(store.folder(sid_r), "rlat.dat")
    np.savetxt(rlat_path, 2.0 * np.loadtxt(rlat_path))
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "hermiticity" in _rejected_reason(ws, sid_h)
    assert "rlat" in _rejected_reason(ws, sid_r)


def test_validate_rejects_wrong_element(tmp_path):
    # adversarial repro: element.dat changed from Mg/O to H/H must be rejected,
    # not silently validated.
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    with open(os.path.join(store.folder(sid), "element.dat"), "w") as f:
        f.write("1\n1\n")
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "element" in _rejected_reason(ws, sid)


def test_validate_rejects_broken_reciprocal_set(tmp_path):
    # adversarial repro: a recorded reciprocal set marked not inversion-symmetric
    # (realness of V^LR depends on it) must be rejected.
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "lr_metadata.json")
    meta = json.load(open(p))
    meta["reciprocal_set"]["inversion_symmetric"] = False
    meta["reciprocal_set"]["ok"] = False
    with open(p, "w") as f:
        json.dump(meta, f)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "reciprocal_set" in _rejected_reason(ws, sid)


def test_validate_rejects_lr_definition_mismatch(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "lr_metadata.json")
    meta = json.load(open(p))
    meta["lr_definition"]["ewald_lambda"] = 999.0
    with open(p, "w") as f:
        json.dump(meta, f)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "lr_definition" in _rejected_reason(ws, sid)


def test_validate_rejects_wrong_lattice(tmp_path):
    # a self-consistent but wrong cell (lat and rlat scaled together so
    # rlat^T lat = 2 pi I still holds) must fail full-lattice agreement.
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    folder = store.folder(sid)
    lat_p = os.path.join(folder, "lat.dat")
    rlat_p = os.path.join(folder, "rlat.dat")
    np.savetxt(lat_p, 1.05 * np.loadtxt(lat_p))
    np.savetxt(rlat_p, np.loadtxt(rlat_p) / 1.05)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "lat" in _rejected_reason(ws, sid)


def test_validate_rejects_position_mismatch(tmp_path):
    # DFT positions that do not equal reference + recorded displacement.
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "site_positions.dat")
    pos = np.loadtxt(p)
    pos[0, 0] += 0.3
    np.savetxt(p, pos)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "position" in _rejected_reason(ws, sid)


def test_validate_tier2_enforce_flags_violation(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = "snapshot_000001"                  # the +0.01 member
    folder = store.folder(sid)
    h_lr = {k: 5.0 * v for k, v in convert.read_blocks(
        os.path.join(folder, "hamiltonians_lr.h5")).items()}
    h_full = convert.read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_sr = {k: h_full[k] - h_lr[k] for k in h_full}  # reconstruction stays exact
    convert.write_blocks(os.path.join(folder, "hamiltonians_lr.h5"), h_lr)
    convert.write_blocks(os.path.join(folder, "hamiltonians_sr.h5"), h_sr)
    cfg2 = copy.deepcopy(cfg)
    cfg2["validation"]["tier2_enforce"] = True
    assert validate.validate_stage(cfg2, ws, Args()) == 1
    assert sid in store.list()               # tier-2 never rejects snapshots
    summary = json.load(open(os.path.join(ws, "generation_logs",
                                          "validation_pilot.json")))
    assert summary["tier2"]["violations"]
