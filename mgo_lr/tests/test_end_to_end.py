import json
import os
import subprocess
import sys

import numpy as np
import yaml

from mgo_lr import convert, dfpt, export, locality, lr, organize, reference, validate
from mgo_lr import displacements as dp
from mgo_lr.snapshot import SnapshotStore, load_reference
from mgo_lr.structures import make_supercell
from mgo_lr.tests.test_convert import fabricate_dft
from mgo_lr.tests.test_dfpt_collect import PH_OUT
from mgo_lr.tests.test_lr_process import lr_cfg


class Args:
    set_name = "pilot"
    force = False
    target = None


def test_full_pipeline(tmp_path):
    ws = str(tmp_path)
    cfg = lr_cfg()
    cfg["material"]["lattice_constant_relaxed"] = 4.2
    a = Args()

    # reference + DFPT round-trip (synthetic outputs)
    assert reference.init_reference_stage(cfg, ws, a) == 0
    assert reference.collect_reference_stage(cfg, ws, a) == 0
    assert dfpt.init_dfpt_stage(cfg, ws, a) == 0
    with open(os.path.join(ws, "reference", "qe", "ph.out"), "w") as f:
        f.write(PH_OUT)
    assert dfpt.collect_dfpt_stage(cfg, ws, a) == 0

    # structures
    assert dp.gen_structures_stage(cfg, ws, a) == 0
    store = SnapshotStore(ws, "pilot")
    sids = store.list()
    assert len(sids) == 46

    # fabricate DFT for the first 10 snapshots (equilibrium + ladder pairs);
    # one common seed so every snapshot shares the same synthetic H/S
    ref = load_reference(ws)
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"],
                        cfg["supercells"]["pilot"])
    done = sids[:10]
    for sid in done:
        fabricate_dft(store.folder(sid), cfg, sc, seed=0)

    assert convert.collect_dft_stage(cfg, ws, a) == 0
    states = {sid: store.read_status(sid)["state"] for sid in store.list()}
    assert all(states[sid] == "converted" for sid in done)
    assert all(states[sid] == "prepared" for sid in sids[10:])

    assert lr.lr_process_stage(cfg, ws, a) == 0
    assert validate.validate_stage(cfg, ws, a) == 0
    validated = [sid for sid in store.list()
                 if store.read_status(sid)["state"] == "validated"]
    assert sorted(validated) == sorted(done)

    assert locality.locality_report_stage(cfg, ws, a) == 0
    assert os.path.exists(os.path.join(ws, "generation_logs", "locality",
                                       "locality_pilot.json"))
    assert organize.organize_stage(cfg, ws, a) == 0        # empty main OK
    splits = json.load(open(os.path.join(ws, "splits.json")))
    assert splits["pilot"] == sorted(done)

    a2 = Args()
    a2.target = "sr"
    assert export.export_target_stage(cfg, ws, a2) == 0
    for sid in done:
        f = store.folder(sid)
        got = convert.read_blocks(os.path.join(f, "hamiltonians.h5"))
        want = convert.read_blocks(os.path.join(f, "hamiltonians_sr.h5"))
        assert lr.blocks_diff_norm(got, want) == 0.0
    meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert meta["training_target"] == "sr"
    assert meta["lr_definition"]["sign_convention"] == \
        "electron_potential_energy"

    # CLI smoke test
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "status",
                        "--workspace", ws], capture_output=True, text=True)
    assert r.returncode == 0
    assert "pilot" in r.stdout and "validated=10" in r.stdout
