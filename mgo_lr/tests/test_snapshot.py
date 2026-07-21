import os

import pytest

from mgo_lr.snapshot import STATES, SnapshotStore, set_dir_name, status_stage


def _mk(store, sid):
    os.makedirs(store.folder(sid), exist_ok=True)
    store.write_status(sid, "prepared")


def test_set_dir_name():
    assert set_dir_name("pilot") == "pilot"
    assert set_dir_name("main") == "main"
    assert set_dir_name("large") == "test_large_cell"


def test_status_roundtrip_and_history(tmp_path):
    store = SnapshotStore(str(tmp_path), "pilot")
    _mk(store, "snapshot_000001")
    store.write_status("snapshot_000001", "dft_done", note="ok")
    st = store.read_status("snapshot_000001")
    assert st["state"] == "dft_done"
    assert st["note"] == "ok"
    assert [h["state"] for h in st["history"]] == ["prepared", "dft_done"]


def test_state_at_least(tmp_path):
    store = SnapshotStore(str(tmp_path), "pilot")
    _mk(store, "snapshot_000001")
    store.write_status("snapshot_000001", "converted")
    assert store.state_at_least("snapshot_000001", "dft_done")
    assert not store.state_at_least("snapshot_000001", "lr_done")


def test_invalid_state_rejected(tmp_path):
    store = SnapshotStore(str(tmp_path), "pilot")
    _mk(store, "snapshot_000001")
    with pytest.raises(ValueError):
        store.write_status("snapshot_000001", "bogus")


def test_reject_moves_folder(tmp_path):
    store = SnapshotStore(str(tmp_path), "main")
    _mk(store, "snapshot_000007")
    store.reject("snapshot_000007", "scf_not_converged")
    assert store.list() == []
    dest = tmp_path / "rejected" / "main_snapshot_000007"
    assert dest.is_dir()
    import json
    st = json.load(open(dest / "status.json"))
    assert st["state"] == "rejected" and st["reason"] == "scf_not_converged"


def test_status_stage_runs(tmp_path, capsys):
    store = SnapshotStore(str(tmp_path), "pilot")
    _mk(store, "snapshot_000001")
    class A: pass
    assert status_stage({}, str(tmp_path), A()) == 0
    assert "pilot" in capsys.readouterr().out
