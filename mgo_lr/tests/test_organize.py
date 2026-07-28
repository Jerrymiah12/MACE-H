import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import organize
from mgo_lr.config import load_config, sha256_file
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.tests.test_convert import Args

CFG = load_config("mgo_lr/configs/mgo.yaml")


def test_grouped_split_integrity_and_determinism():
    groups = {f"g{i:02d}": [f"snapshot_{2*i+1:06d}", f"snapshot_{2*i+2:06d}"]
              for i in range(10)}
    s1 = organize.grouped_split(groups, 0.2, 0.2, 42)
    s2 = organize.grouped_split(dict(reversed(list(groups.items()))),
                                0.2, 0.2, 42)
    assert s1 == s2                                        # deterministic
    all_sids = sorted(sid for m in groups.values() for sid in m)
    got = sorted(s1["train"] + s1["validation"] + s1["test"])
    assert got == all_sids                                 # partition
    for members in groups.values():                        # groups intact
        subsets = {k for k in s1 if set(members) & set(s1[k])}
        assert len(subsets) == 1
    assert len(s1["test"]) >= 0.2 * 20                     # filled to target
    assert len(s1["validation"]) >= 0.2 * 20
    s3 = organize.grouped_split(groups, 0.2, 0.2, 43)
    assert s3 != s1                                        # seed-dependent


def test_holdout_groups_unions_shared_qvector():
    metas = {
        "s1": {"pattern_group_id": "g1", "q_vectors": [[1, 0, 0]]},
        "s2": {"pattern_group_id": "g2", "q_vectors": [[1, 0, 0]]},   # shares q
        "s3": {"pattern_group_id": "g3", "q_vectors": [[0, 1, 0]]},
        "s4": {"pattern_group_id": "g4", "q_vectors": [[-1, 0, 0]]},  # -q of s1
        "s5": {"pattern_group_id": "g5", "q_vectors": []},            # q-less
    }
    groups = organize.holdout_groups(metas)
    g_of = {sid: gk for gk, sids in groups.items() for sid in sids}
    assert g_of["s1"] == g_of["s2"] == g_of["s4"]     # same ±q shell
    assert g_of["s3"] != g_of["s1"]
    assert g_of["s5"] != g_of["s1"]                    # q-less stands alone
    assert sorted(sid for sids in groups.values() for sid in sids) == \
        ["s1", "s2", "s3", "s4", "s5"]                 # partition


def test_split_has_no_qvector_leakage():
    # P1 regression: build_main gives each snapshot a unique pattern_group_id,
    # so grouping by it scattered shared q-vectors across train/val/test.  A
    # q-vector-family holdout must keep every ±q shell inside one subset.
    qpool = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    metas = {f"snapshot_{i + 1:06d}": {"pattern_group_id": f"grp-{i:03d}",
                                       "q_vectors": [qpool[i % len(qpool)]]}
             for i in range(24)}
    groups = organize.holdout_groups(metas)
    splits = organize.grouped_split(groups, 0.25, 0.25, 7)
    subset_qs = {name: {organize._canonical_q(q)
                        for sid in sids for q in metas[sid]["q_vectors"]}
                 for name, sids in splits.items()}
    subsets = ["train", "validation", "test"]
    for a in range(len(subsets)):
        for b in range(a + 1, len(subsets)):
            assert subset_qs[subsets[a]].isdisjoint(subset_qs[subsets[b]]), \
                (subsets[a], subsets[b])


def _mk_main(ws, n_groups=4, per_group=2):
    store = SnapshotStore(ws, "main")
    k = 1
    for g in range(n_groups):
        for _ in range(per_group):
            sid = f"snapshot_{k:06d}"
            os.makedirs(store.folder(sid))
            with open(os.path.join(store.folder(sid),
                                   "displacement_metadata.json"), "w") as f:
                json.dump({"pattern_group_id": f"grp-{g:02d}"}, f)
            store.write_status(sid, "prepared")
            store.write_status(sid, "validated")
            k += 1
    return store


def _cfg_with_local_files(tmp_path):
    cfg = copy.deepcopy(CFG)
    pdir = tmp_path / "pseudo"
    pdir.mkdir()
    (pdir / cfg["abacus"]["pseudopotentials"]["Mg"]).write_text("MG PSEUDO")
    cfg["abacus"]["pseudo_dir"] = str(pdir)          # only Mg present
    cfg["qe"]["pseudo_dir"] = str(pdir)
    return cfg, pdir


def test_organize_stage(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    _mk_main(ws)
    cfg, pdir = _cfg_with_local_files(tmp_path)
    # pre-existing lr_definition must survive the merge
    with open(os.path.join(ws, "metadata.yaml"), "w") as f:
        yaml.safe_dump({"lr_definition": {"ewald_lambda": 0.35}}, f)
    assert organize.organize_stage(cfg, ws, Args()) == 0
    splits = json.load(open(os.path.join(ws, "splits.json")))
    main = splits["main"]
    assert sorted(main["train"] + main["validation"] + main["test"]) == \
        [f"snapshot_{k:06d}" for k in range(1, 9)]
    assert splits["large_test"] == [] and splits["pilot"] == []
    # candidate dirs: json listing + symlinks resolving into main/
    for subset, dirname in (("validation", "validation_candidates"),
                            ("test", "test_candidates")):
        listing = json.load(open(os.path.join(ws, dirname,
                                              "candidates.json")))
        assert listing == main[subset]
        for sid in main[subset]:
            link = os.path.join(ws, dirname, sid)
            assert os.path.islink(link)
            assert os.path.isdir(os.path.realpath(link))
    meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert meta["lr_definition"]["ewald_lambda"] == 0.35   # preserved
    assert meta["units"] == {"energy": "eV", "length": "angstrom",
                             "charge": "e"}
    prov = meta["provenance"]
    mg = cfg["abacus"]["pseudopotentials"]["Mg"]
    assert prov["abacus"]["pseudopotentials"][mg] == \
        sha256_file(str(pdir / mg))
    o = cfg["abacus"]["pseudopotentials"]["O"]
    assert prov["abacus"]["pseudopotentials"][o] is None   # missing file
    assert any(o in p for p in prov["missing_files"])
    assert meta["code_versions"]["abacus"] == str(cfg["abacus"]["version"])
    assert meta["splits"]["main"] == {k: len(v) for k, v in main.items()}
    # rerun is idempotent (same seed -> same splits, symlinks refreshed)
    assert organize.organize_stage(cfg, ws, Args()) == 0
    assert json.load(open(os.path.join(ws, "splits.json")))["main"] == main


def test_organize_refuses_foreign_candidate_entries(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(os.path.join(ws, "validation_candidates", "not_a_link"))
    _mk_main(ws)
    cfg, _ = _cfg_with_local_files(tmp_path)
    with pytest.raises(SystemExit, match="not a symlink"):
        organize.organize_stage(cfg, ws, Args())
