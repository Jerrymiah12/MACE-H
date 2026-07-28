import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mgo_lr import displacements as dp
from mgo_lr.config import load_config
from mgo_lr.snapshot import SnapshotStore, load_reference
from mgo_lr.structures import make_supercell, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")


def make_fake_reference(workspace, a=4.2):
    """Create the reference artifacts Task 10 will write in production."""
    ref = os.path.join(workspace, "reference")
    os.makedirs(ref, exist_ok=True)
    cell, frac, species = rocksalt_primitive(a)
    np.save(os.path.join(ref, "reference_cell.npy"), cell)
    np.save(os.path.join(ref, "reference_positions.npy"), frac)
    np.save(os.path.join(ref, "atomic_numbers.npy"), np.array([12, 8]))
    with open(os.path.join(ref, "species_order.json"), "w") as f:
        json.dump(species, f)
    return cell, frac, species


class Args:
    set_name = "pilot"
    force = False


def test_load_reference_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="reference_cell.npy"):
        load_reference(str(tmp_path))


def test_gen_structures_pilot(tmp_path):
    ws = str(tmp_path)
    cell, frac, species = make_fake_reference(ws)
    assert dp.gen_structures_stage(CFG, ws, Args()) == 0
    store = SnapshotStore(ws, "pilot")
    sids = store.list()
    assert len(sids) == 46
    sc = make_supercell(cell, frac, species, CFG["supercells"]["pilot"])
    for sid in sids:
        folder = store.folder(sid)
        for name in ("STRU", "INPUT", "KPT", "displacements.npy",
                     "displacement_metadata.json", "status.json"):
            assert os.path.exists(os.path.join(folder, name)), (sid, name)
        assert store.read_status(sid)["state"] == "prepared"
        u = np.load(os.path.join(folder, "displacements.npy"))
        assert u.shape == (16, 3)
        d = dp.minimum_distance(sc.cell, sc.cart + u)
        assert d >= CFG["displacements"]["min_distance"]
    # equilibrium snapshot has zero displacements
    u0 = np.load(os.path.join(store.folder(sids[0]), "displacements.npy"))
    assert np.allclose(u0, 0.0)
    text = open(os.path.join(store.folder(sids[0]), "INPUT")).read()
    assert "out_mat_hs2" in text and "gamma_only" in text
    meta = json.load(open(os.path.join(store.folder(sids[1]),
                                       "displacement_metadata.json")))
    assert "pattern_group_id" in meta and "pattern" in meta
    assert os.path.exists(os.path.join(store.set_dir, "job_abacus.sh"))


def test_gen_structures_idempotent(tmp_path):
    ws = str(tmp_path)
    make_fake_reference(ws)
    dp.gen_structures_stage(CFG, ws, Args())
    store = SnapshotStore(ws, "pilot")
    sid = store.list()[0]
    before = store.read_status(sid)["history"]
    dp.gen_structures_stage(CFG, ws, Args())          # no --force: skip all
    assert store.read_status(sid)["history"] == before


def test_gen_structures_force_protects_dft_output(tmp_path):
    ws = str(tmp_path)
    make_fake_reference(ws)
    dp.gen_structures_stage(CFG, ws, Args())
    store = SnapshotStore(ws, "pilot")
    sid = store.list()[0]
    os.makedirs(os.path.join(store.folder(sid), "OUT.MgO"))
    args = Args()
    args.force = True
    dp.gen_structures_stage(CFG, ws, args)
    # snapshot with DFT output untouched; others regenerated
    assert len(store.read_status(sid)["history"]) == 1
    other = store.list()[1]
    assert len(store.read_status(other)["history"]) == 2


def test_rejected_snapshot_not_regenerated(tmp_path):
    # P1 regression: rejecting a snapshot moves it to rejected/<set>_<sid>, but
    # gen-structures only checked the active set dir, so the rejected id
    # reappeared in `prepared` state on the next run (and a later re-rejection
    # would collide with the existing rejected destination).
    ws = str(tmp_path)
    make_fake_reference(ws)
    dp.gen_structures_stage(CFG, ws, Args())
    store = SnapshotStore(ws, "pilot")
    sid = store.list()[0]
    store.reject(sid, "synthetic")
    assert sid not in store.list()                       # moved out
    assert store.is_rejected(sid)
    dp.gen_structures_stage(CFG, ws, Args())             # rerun
    assert sid not in store.list()                       # must stay rejected
    assert not os.path.isdir(store.folder(sid))


def test_cli_requires_reference(tmp_path):
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "gen-structures",
                        "--workspace", str(tmp_path), "--set", "pilot"],
                       capture_output=True, text=True)
    assert r.returncode != 0
