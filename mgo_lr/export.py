"""export-target: materialize hamiltonians.h5 (the file the maceh loader
reads, see maceh/graph.py) from the selected label source.

The three source files hamiltonians_{full,lr,sr}.h5 are never modified or
renamed.  hamiltonians.h5 is only ever (re)written when it was produced by
this stage (symlink into SOURCES, or export_metadata.json marker) — a
foreign hamiltonians.h5 is never clobbered.
"""
import json
import os
import shutil

import yaml

from . import __version__
from .config import atomic_write_text
from .snapshot import SnapshotStore

SOURCES = {"full": "hamiltonians_full.h5",
           "lr": "hamiltonians_lr.h5",
           "sr": "hamiltonians_sr.h5"}
TARGET_NAME = "hamiltonians.h5"
MARKER = "export_metadata.json"


def _safe_to_replace(folder):
    t = os.path.join(folder, TARGET_NAME)
    if not os.path.lexists(t):
        return True
    if os.path.islink(t) \
            and os.path.basename(os.readlink(t)) in SOURCES.values():
        return True
    return os.path.exists(os.path.join(folder, MARKER))


def export_snapshot(folder, target):
    src = SOURCES[target]
    src_path = os.path.join(folder, src)
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    if not _safe_to_replace(folder):
        raise SystemExit(
            f"{os.path.join(folder, TARGET_NAME)} exists and was not "
            "written by export-target — refusing to clobber it")
    t = os.path.join(folder, TARGET_NAME)
    if os.path.lexists(t):
        os.remove(t)
    try:
        os.symlink(src, t)
        method = "symlink"
    except OSError:
        tmp = f"{t}.tmp.{os.getpid()}"
        shutil.copyfile(src_path, tmp)
        os.replace(tmp, t)
        method = "copy"
    atomic_write_text(os.path.join(folder, MARKER),
                      json.dumps({"target": target, "source": src,
                                  "method": method,
                                  "code_version": __version__}))
    return method


def export_target_stage(cfg, workspace, args):
    target = getattr(args, "target", None)
    if target not in SOURCES:
        raise SystemExit("export-target requires --target full|lr|sr")
    min_state = "converted" if target == "full" else "lr_done"
    n = 0
    for set_name in ("pilot", "main", "large"):
        store = SnapshotStore(workspace, set_name)
        for sid in store.list():
            if store.read_status(sid)["state"] == "rejected":
                continue
            if not store.state_at_least(sid, min_state):
                continue
            export_snapshot(store.folder(sid), target)
            n += 1
    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data["training_target"] = target
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    print(f"exported {TARGET_NAME} <- {SOURCES[target]} "
          f"for {n} snapshots (target recorded in metadata.yaml)")
    return 0
