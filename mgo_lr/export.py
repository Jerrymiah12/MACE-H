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
from .lr import require_current_lr_definition
from .snapshot import SnapshotStore

SOURCES = {"full": "hamiltonians_full.h5",
           "lr": "hamiltonians_lr.h5",
           "sr": "hamiltonians_sr.h5"}
TARGET_NAME = "hamiltonians.h5"
MARKER = "export_metadata.json"


def _current_export_target(folder):
    """The target of an export WE produced in this folder, or None.

    Used to detect a stale export of a *different* target left behind when a
    snapshot is skipped — that, not a never-exported snapshot, is what makes a
    dataset mixed.
    """
    marker = os.path.join(folder, MARKER)
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                return json.load(f).get("target")
        except (json.JSONDecodeError, OSError):
            return None
    t = os.path.join(folder, TARGET_NAME)
    if os.path.islink(t):
        base = os.path.basename(os.readlink(t))
        for name, src in SOURCES.items():
            if src == base:
                return name
    return None


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
    src = SOURCES[target]

    # Export is all-or-nothing: verify every converted snapshot BEFORE changing
    # any file.  Foreign targets and snapshots that have not reached the
    # requested label state are caught up front, so failure changes nothing.
    eligible, stale, foreign, incomplete = [], [], [], []
    for set_name in ("pilot", "main", "large"):
        store = SnapshotStore(workspace, set_name)
        for sid in store.list():
            if store.read_status(sid)["state"] == "rejected":
                continue
            folder = store.folder(sid)
            # A converted folder already satisfies the MACE-H structure-file
            # discovery contract, so it is part of the export scope.  Silently
            # skipping it would leave an unloadable/mixed dataset.
            if not store.state_at_least(sid, "converted"):
                continue
            ready = (store.state_at_least(sid, min_state)
                     and os.path.exists(os.path.join(folder, src)))
            if ready and _safe_to_replace(folder):
                eligible.append(folder)
            elif ready:
                foreign.append(os.path.join(folder, TARGET_NAME))
            else:
                current = _current_export_target(folder)
                incomplete.append(
                    f"{set_name}/{sid} (state "
                    f"{store.read_status(sid)['state']}, missing {src})")
                if current not in (None, target):
                    stale.append(f"{set_name}/{sid} (currently {current})")

    if foreign:
        raise SystemExit(
            f"{foreign[0]} exists and was not written by export-target — "
            "refusing to clobber it (no files changed)")
    if stale:
        raise SystemExit(
            f"export-target {target} is all-or-nothing but "
            f"{len(stale)} snapshot(s) still carry a different export target "
            "and cannot advance; refusing to publish a mixed dataset (no files "
            "changed):\n  " + "\n  ".join(stale))
    if incomplete:
        raise SystemExit(
            f"export-target {target} is all-or-nothing but "
            f"{len(incomplete)} converted snapshot(s) are not ready; refusing "
            "to publish an incomplete dataset (no files changed):\n  "
            + "\n  ".join(incomplete))
    if not eligible:
        raise SystemExit(f"export-target {target}: no eligible snapshots")
    if target in ("lr", "sr"):
        require_current_lr_definition(cfg, workspace)

    for folder in eligible:
        export_snapshot(folder, target)
    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data["training_target"] = target
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    print(f"exported {TARGET_NAME} <- {src} for {len(eligible)} snapshots "
          "(target recorded in metadata.yaml)")
    return 0
