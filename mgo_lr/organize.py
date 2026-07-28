"""Final dataset organization: grouped leakage-safe splits, candidate
directories, and the top-level metadata.yaml provenance record.

Snapshots are never split individually: all members of one
pattern_group_id (sign partners, amplitude ladders, phase families, mode
mixtures) land in the same subset.  The 4x4x4 set stays entirely separate
as the large-cell extrapolation set.
"""
import json
import os
import shutil

import numpy as np
import yaml

from . import __version__
from .config import atomic_write_text, sha256_file
from .displacements import MODE_NORMALIZATION
from .lr import require_current_lr_definition
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


def split_from_hints(metas):
    """Materialize generator-assigned leakage-safe split identities."""
    allowed = ("train", "validation", "test")
    missing = [sid for sid, meta in metas.items()
               if meta.get("split_hint") not in allowed]
    if missing:
        raise ValueError(
            f"{len(missing)} main snapshots lack a valid split_hint; "
            "regenerate the main structures with the current pipeline")
    out = {name: sorted(sid for sid, meta in metas.items()
                        if meta["split_hint"] == name)
           for name in allowed}

    # Pattern groups and exact ±q families may never cross subsets.
    seen_pattern, seen_q, seen_q_shell = {}, {}, {}
    for subset, sids in out.items():
        for sid in sids:
            meta = metas[sid]
            pg = meta.get("pattern_group_id")
            if pg in seen_pattern and seen_pattern[pg] != subset:
                raise ValueError(f"pattern group {pg} crosses "
                                 f"{seen_pattern[pg]}/{subset}")
            seen_pattern[pg] = subset
            for q in meta.get("q_vectors") or []:
                cq = _canonical_q(q)
                if cq in seen_q and seen_q[cq] != subset:
                    raise ValueError(f"q family {cq} crosses "
                                     f"{seen_q[cq]}/{subset}")
                seen_q[cq] = subset
            for magnitude in meta.get("q_magnitudes") or []:
                shell = round(float(magnitude), 10)
                if shell in seen_q_shell and seen_q_shell[shell] != subset:
                    raise ValueError(f"q shell |q|={shell} crosses "
                                     f"{seen_q_shell[shell]}/{subset}")
                seen_q_shell[shell] = subset
    return out


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


_LOADER_FILES = (
    "element.dat", "orbital_types.dat", "lat.dat", "rlat.dat",
    "site_positions.dat", "info.json", "hamiltonians.h5", "overlaps.h5",
    "displacement_metadata.json", "quality_checks.json",
)


def _fill_loader_view(workspace, subset, sids):
    """Real directories containing file symlinks, which os.walk can consume.

    MACE-H does not follow directory symlinks.  Candidate directories retain
    their lightweight directory links for human inspection, while these views
    give the loader an immediately traversable train/validation/test root.
    """
    root = os.path.join(workspace, "loader_splits", subset)
    marker = os.path.join(root, ".mgo_lr_loader_view")
    if os.path.exists(root):
        if not os.path.isfile(marker):
            raise SystemExit(f"{root}: not a loader view written by organize; "
                             "refusing to replace it")
        shutil.rmtree(root)
    os.makedirs(root)
    atomic_write_text(marker, "generated by mgo_lr organize\n")
    for sid in sids:
        view = os.path.join(root, sid)
        os.makedirs(view)
        source = os.path.join(workspace, set_dir_name("main"), sid)
        for name in _LOADER_FILES:
            os.symlink(os.path.relpath(os.path.join(source, name), view),
                       os.path.join(view, name))


def organize_stage(cfg, workspace, args):
    seed = cfg["displacements"]["seed"]
    stores = {name: SnapshotStore(workspace, name)
              for name in ("pilot", "main", "large")}
    metas = {}
    for sid in _validated(stores["main"]):
        with open(os.path.join(stores["main"].folder(sid),
                               "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
    if metas:
        try:
            splits = split_from_hints(metas)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        splits = {"train": [], "validation": [], "test": []}

    # Provenance is a hard preflight: do not publish splits/metadata with null
    # hashes, because those records cannot identify a reproducible dataset.
    ab, qe = cfg["abacus"], cfg["qe"]
    ab_pp, m1 = _hash_files(ab["pseudo_dir"], ab["pseudopotentials"])
    ab_orb, m2 = _hash_files(ab["orbital_dir"], ab["orbitals"])
    qe_pp, m3 = _hash_files(qe["pseudo_dir"], qe["pseudopotentials"])
    missing_provenance = m1 + m2 + m3
    if missing_provenance:
        raise SystemExit(
            "organize requires complete pseudopotential/orbital provenance; "
            "missing:\n  " + "\n  ".join(missing_provenance))
    require_current_lr_definition(cfg, workspace)

    # Loader views bind hamiltonians.h5 (produced by export-target) and the
    # other per-snapshot files by symlink.  Verify every one exists BEFORE any
    # write, so a run never publishes a view with dangling links; this makes
    # export-target a prerequisite of organize.
    main_dir = os.path.join(workspace, set_dir_name("main"))
    dangling = [f"{sid}/{name}"
                for sid in splits["train"] + splits["validation"]
                + splits["test"]
                for name in _LOADER_FILES
                if not os.path.exists(os.path.join(main_dir, sid, name))]
    if dangling:
        raise SystemExit(
            "organize builds MACE-H loader views that symlink hamiltonians.h5 "
            "and other per-snapshot files; run collect-dft, validate, and "
            "export-target before organize so nothing dangles. Missing:\n  "
            + "\n  ".join(dangling))

    doc = {"seed": seed,
           "validation_fraction": float(cfg["splits"]["validation_fraction"]),
           "test_fraction": float(cfg["splits"]["test_fraction"]),
           "grouping": "generation-time split-specific q shells + pattern groups",
           "main": splits,
           "pilot": sorted(_validated(stores["pilot"])),
           "large_test": sorted(_validated(stores["large"]))}
    atomic_write_text(os.path.join(workspace, "splits.json"),
                      json.dumps(doc, indent=1))
    _fill_candidates(workspace, "validation_candidates", splits["validation"])
    _fill_candidates(workspace, "test_candidates", splits["test"])
    for subset, sids in splits.items():
        _fill_loader_view(workspace, subset, sids)

    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update({
        "material": cfg["material"]["name"],
        "exchange_correlation": cfg["material"]["xc_functional"],
        "units": {"energy": "eV", "length": "angstrom", "charge": "e"},
        "atom_ordering": "species_major_cell_minor (np.ndindex)",
        "mode_normalization": MODE_NORMALIZATION,
        "supercells": {k: int(v) for k, v in cfg["supercells"].items()},
        "supercell_matrices": {
            k: [[int(v), 0, 0], [0, int(v), 0], [0, 0, int(v)]]
            for k, v in cfg["supercells"].items()},
        "displacement_seed": int(seed),
        "code_versions": {"mgo_lr": __version__,
                          "abacus": str(ab["version"]),
                          "quantum_espresso": str(qe["version"])},
        "dft_settings": {"abacus": ab, "qe": qe},
        "splits": {"main": {k: len(v) for k, v in splits.items()},
                   "pilot": len(doc["pilot"]),
                   "large_test": len(doc["large_test"])},
        "loader_split_roots": {
            name: os.path.join("loader_splits", name) for name in splits},
        "provenance": {
            "abacus": {"pseudopotentials": ab_pp, "orbitals": ab_orb},
            "quantum_espresso": {"pseudopotentials": qe_pp},
            "missing_files": []},
    })
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    print(f"organize: main train/val/test = "
          f"{len(splits['train'])}/{len(splits['validation'])}/"
          f"{len(splits['test'])}, pilot = {len(doc['pilot'])}, "
          f"large_test = {len(doc['large_test'])}")
    return 0
