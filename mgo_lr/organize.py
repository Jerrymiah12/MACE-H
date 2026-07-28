"""Final dataset organization: grouped leakage-safe splits, candidate
directories, and the top-level metadata.yaml provenance record.

Snapshots are never split individually: all members of one
pattern_group_id (sign partners, amplitude ladders, phase families, mode
mixtures) land in the same subset.  The 4x4x4 set stays entirely separate
as the large-cell extrapolation set.
"""
import json
import os

import numpy as np
import yaml

from . import __version__
from .config import atomic_write_text, sha256_file
from .displacements import MODE_NORMALIZATION
from .snapshot import SnapshotStore, set_dir_name


def _canonical_q(q):
    """Sign-canonical integer q so q and -q share one holdout unit."""
    q = tuple(int(x) for x in q)
    for c in q:
        if c > 0:
            return q
        if c < 0:
            return tuple(-x for x in q)
    return q                              # all-zero


def holdout_groups(metas):
    """Atomic holdout units: union any snapshots that share a q-vector
    (sign-canonicalized, so a ±q shell is one unit) or a pattern_group_id
    (sign/amplitude partners).  Because whole units are assigned to a single
    subset, no q-vector or q-shell can straddle train/val/test.  Snapshots with
    no q-vector (equilibrium, random_local, near_equilibrium) leak nothing and
    are grouped by pattern_group_id alone.
    """
    parent = {sid: sid for sid in metas}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    buckets = {}
    for sid, m in metas.items():
        for q in m.get("q_vectors") or []:
            buckets.setdefault(("q", _canonical_q(q)), []).append(sid)
        pg = m.get("pattern_group_id")
        if pg is not None:
            buckets.setdefault(("pg", pg), []).append(sid)
    for members in buckets.values():
        for other in members[1:]:
            union(members[0], other)

    groups = {}
    for sid in metas:
        groups.setdefault(find(sid), []).append(sid)
    return {root: sorted(sids) for root, sids in groups.items()}


def grouped_split(groups, val_frac, test_frac, seed):
    gids = sorted(groups)
    rng = np.random.default_rng([int(seed), 777001])
    rng.shuffle(gids)
    total = sum(len(groups[g]) for g in gids)
    out = {"train": [], "validation": [], "test": []}
    n_test = n_val = 0
    for g in gids:
        members = sorted(groups[g])
        if n_test < test_frac * total:
            out["test"] += members
            n_test += len(members)
        elif n_val < val_frac * total:
            out["validation"] += members
            n_val += len(members)
        else:
            out["train"] += members
    return {k: sorted(v) for k, v in out.items()}


def _validated(store):
    return [sid for sid in store.list()
            if store.read_status(sid)["state"] == "validated"]


def _hash_files(base_dir, names):
    out, missing = {}, []
    for _, name in sorted(names.items()):
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            out[name] = sha256_file(p)
        else:
            out[name] = None
            missing.append(p)
    return out, missing


def _fill_candidates(workspace, dirname, sids):
    d = os.path.join(workspace, dirname)
    os.makedirs(d, exist_ok=True)
    for entry in os.listdir(d):
        p = os.path.join(d, entry)
        if os.path.islink(p) or entry == "candidates.json":
            os.remove(p)
        else:
            raise SystemExit(f"{p}: not a symlink written by organize — "
                             "refusing to touch")
    atomic_write_text(os.path.join(d, "candidates.json"), json.dumps(sids))
    for sid in sids:
        try:
            os.symlink(os.path.join("..", set_dir_name("main"), sid),
                       os.path.join(d, sid))
        except OSError as e:
            print(f"WARNING: symlink {sid} failed ({e}); "
                  "candidates.json remains authoritative")
            break


def organize_stage(cfg, workspace, args):
    seed = cfg["displacements"]["seed"]
    stores = {name: SnapshotStore(workspace, name)
              for name in ("pilot", "main", "large")}
    metas = {}
    for sid in _validated(stores["main"]):
        with open(os.path.join(stores["main"].folder(sid),
                               "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
    groups = holdout_groups(metas)
    splits = grouped_split(groups,
                           float(cfg["splits"]["validation_fraction"]),
                           float(cfg["splits"]["test_fraction"]), seed)
    doc = {"seed": seed,
           "validation_fraction": float(cfg["splits"]["validation_fraction"]),
           "test_fraction": float(cfg["splits"]["test_fraction"]),
           "grouping": "q_vector_family (shared ±q shells + pattern groups)",
           "main": splits,
           "pilot": sorted(_validated(stores["pilot"])),
           "large_test": sorted(_validated(stores["large"]))}
    atomic_write_text(os.path.join(workspace, "splits.json"),
                      json.dumps(doc, indent=1))
    _fill_candidates(workspace, "validation_candidates", splits["validation"])
    _fill_candidates(workspace, "test_candidates", splits["test"])

    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    ab, qe = cfg["abacus"], cfg["qe"]
    ab_pp, m1 = _hash_files(ab["pseudo_dir"], ab["pseudopotentials"])
    ab_orb, m2 = _hash_files(ab["orbital_dir"], ab["orbitals"])
    qe_pp, m3 = _hash_files(qe["pseudo_dir"], qe["pseudopotentials"])
    data.update({
        "material": cfg["material"]["name"],
        "units": {"energy": "eV", "length": "angstrom", "charge": "e"},
        "atom_ordering": "species_major_cell_minor (np.ndindex)",
        "mode_normalization": MODE_NORMALIZATION,
        "supercells": {k: int(v) for k, v in cfg["supercells"].items()},
        "displacement_seed": int(seed),
        "code_versions": {"mgo_lr": __version__,
                          "abacus": str(ab["version"]),
                          "quantum_espresso": str(qe["version"])},
        "dft_settings": {"abacus": ab, "qe": qe},
        "splits": {"main": {k: len(v) for k, v in splits.items()},
                   "pilot": len(doc["pilot"]),
                   "large_test": len(doc["large_test"])},
        "provenance": {
            "abacus": {"pseudopotentials": ab_pp, "orbitals": ab_orb},
            "quantum_espresso": {"pseudopotentials": qe_pp},
            "missing_files": m1 + m2 + m3},
    })
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    for p in m1 + m2 + m3:
        print(f"WARNING: provenance file not found locally: {p}")
    print(f"organize: main train/val/test = "
          f"{len(splits['train'])}/{len(splits['validation'])}/"
          f"{len(splits['test'])}, pilot = {len(doc['pilot'])}, "
          f"large_test = {len(doc['large_test'])}")
    return 0
