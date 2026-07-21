import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import convert, export, lr
from mgo_lr.config import sha256_file
from mgo_lr.tests.test_convert import Args
from mgo_lr.tests.test_lr_process import converted_snapshot
from mgo_lr.tests.test_validate import ladder_workspace


def _args(target):
    a = Args()
    a.target = target
    return a


def test_export_lr_then_switch_to_sr(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    folders = [store.folder(sid) for sid in store.list()]
    before = {f: {n: sha256_file(os.path.join(f, n))
                  for n in export.SOURCES.values()} for f in folders}
    assert export.export_target_stage(cfg, ws, _args("lr")) == 0
    for f in folders:
        got = convert.read_blocks(os.path.join(f, "hamiltonians.h5"))
        want = convert.read_blocks(os.path.join(f, "hamiltonians_lr.h5"))
        assert lr.blocks_diff_norm(got, want) == 0.0
        marker = json.load(open(os.path.join(f, "export_metadata.json")))
        assert marker["target"] == "lr"
    assert export.export_target_stage(cfg, ws, _args("sr")) == 0
    for f in folders:
        got = convert.read_blocks(os.path.join(f, "hamiltonians.h5"))
        want = convert.read_blocks(os.path.join(f, "hamiltonians_sr.h5"))
        assert lr.blocks_diff_norm(got, want) == 0.0
        # sources untouched by both exports
        for n, digest in before[f].items():
            assert sha256_file(os.path.join(f, n)) == digest
    meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert meta["training_target"] == "sr"


def test_export_full_from_converted_only(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)   # no lr-process
    assert export.export_target_stage(cfg, ws, _args("full")) == 0
    f = store.folder(sid)
    assert os.path.exists(os.path.join(f, "hamiltonians.h5"))
    # lr/sr export skips converted-only snapshots instead of failing
    assert export.export_target_stage(cfg, ws, _args("sr")) == 0
    got = convert.read_blocks(os.path.join(f, "hamiltonians.h5"))
    want = convert.read_blocks(os.path.join(f, "hamiltonians_full.h5"))
    assert lr.blocks_diff_norm(got, want) == 0.0             # unchanged


def test_export_refuses_foreign_file(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    folder = store.folder(sid)
    target = os.path.join(folder, "hamiltonians.h5")
    with open(target, "w") as f:
        f.write("precious hand-made data")
    with pytest.raises(SystemExit, match="refusing"):
        export.export_target_stage(cfg, ws, _args("lr"))
    assert open(target).read() == "precious hand-made data"


def test_export_requires_target(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    with pytest.raises(SystemExit, match="target"):
        export.export_target_stage(cfg, ws, Args())
