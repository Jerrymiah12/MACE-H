import json
import os
import shutil
import time
from collections import Counter

from .config import atomic_write_text

STATES = ["prepared", "dft_done", "converted", "lr_done", "validated", "rejected"]
_SET_DIRS = {"pilot": "pilot", "main": "main", "large": "test_large_cell"}


def set_dir_name(set_name):
    return _SET_DIRS[set_name]


class SnapshotStore:
    def __init__(self, workspace, set_name):
        if set_name not in _SET_DIRS:
            raise ValueError(f"unknown set: {set_name}")
        self.workspace = workspace
        self.set_name = set_name
        self.set_dir = os.path.join(workspace, set_dir_name(set_name))
        self.rejected_dir = os.path.join(workspace, "rejected")

    def folder(self, sid):
        return os.path.join(self.set_dir, sid)

    def list(self):
        if not os.path.isdir(self.set_dir):
            return []
        return sorted(d for d in os.listdir(self.set_dir)
                      if d.startswith("snapshot_")
                      and os.path.isdir(os.path.join(self.set_dir, d)))

    def read_status(self, sid):
        with open(os.path.join(self.folder(sid), "status.json")) as f:
            return json.load(f)

    def write_status(self, sid, state, **extra):
        if state not in STATES:
            raise ValueError(f"invalid state: {state}")
        path = os.path.join(self.folder(sid), "status.json")
        cur = {}
        if os.path.exists(path):
            with open(path) as f:
                cur = json.load(f)
        hist = cur.get("history", [])
        hist.append({"state": state,
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        cur.update(extra)
        cur["state"] = state
        cur["history"] = hist
        atomic_write_text(path, json.dumps(cur, indent=1))

    def state_at_least(self, sid, state):
        s = self.read_status(sid)["state"]
        if s == "rejected":
            return True   # never reprocess rejected snapshots
        return STATES.index(s) >= STATES.index(state)

    def reject(self, sid, reason):
        self.write_status(sid, "rejected", reason=reason)
        os.makedirs(self.rejected_dir, exist_ok=True)
        shutil.move(self.folder(sid),
                    os.path.join(self.rejected_dir,
                                 f"{self.set_name}_{sid}"))


def load_reference(workspace):
    """Load the permanent reference artifacts written by collect-reference."""
    import numpy as np
    ref_dir = os.path.join(workspace, "reference")
    needed = ["reference_cell.npy", "reference_positions.npy",
              "atomic_numbers.npy", "species_order.json"]
    missing = [f for f in needed
               if not os.path.exists(os.path.join(ref_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"reference artifacts missing from {ref_dir}: {missing} — run "
            "init-reference / collect-reference first")
    with open(os.path.join(ref_dir, "species_order.json")) as f:
        species = json.load(f)
    return {"prim_cell": np.load(os.path.join(ref_dir, "reference_cell.npy")),
            "frac": np.load(os.path.join(ref_dir, "reference_positions.npy")),
            "atomic_numbers": np.load(os.path.join(ref_dir,
                                                   "atomic_numbers.npy")),
            "species": species}


def status_stage(cfg, workspace, args):
    for set_name in ("pilot", "main", "large"):
        store = SnapshotStore(workspace, set_name)
        counts = Counter(store.read_status(sid)["state"]
                         for sid in store.list())
        line = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"
        print(f"{set_name:6s} ({set_dir_name(set_name)}): {line}")
    rej = os.path.join(workspace, "rejected")
    n_rej = len(os.listdir(rej)) if os.path.isdir(rej) else 0
    print(f"rejected: {n_rej}")
    return 0
