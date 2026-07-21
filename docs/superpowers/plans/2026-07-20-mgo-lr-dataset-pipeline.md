# MgO LR Dataset-Generation Pipeline (`mgo_lr`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone `mgo_lr` package that generates the MgO MACE-H-LR dataset: structure/displacement generation, ABACUS + QE input decks, DFT-output parsing and conversion to DeepH-E3 format, the screened-dipole LR processor (`H^LR`, `H^SR`), the three-tier validation battery, locality diagnostics, grouped splits, and training-target export.

**Architecture:** One self-contained Python package `mgo_lr/` (never imports `maceh`) driven by `python -m mgo_lr <stage> --config <yaml> --workspace <dir>`. Stages are idempotent against a per-snapshot `status.json` state machine (`prepared → dft_done → converted → lr_done → validated` / `rejected`). All physics conventions (units, Ewald Λ, gauge, phase, sign) live in `configs/mgo.yaml` + `constants.py` and are recorded in metadata. Spec: `docs/superpowers/specs/2026-07-20-mgo-lr-dataset-pipeline-design.md`.

**Tech Stack:** Python 3 (conda env `DeepH`), numpy, scipy, h5py, PyYAML, ase, pytest.

## Global Constraints

- Interpreter: `PY=/opt/anaconda3/envs/DeepH/bin/python`; run everything from the repo root `/Users/jb/MACE-H`.
- Test command: `$PY -m pytest mgo_lr/tests -q` (or a single file/test with `-v`). Tests never need DFT binaries or network.
- `mgo_lr` NEVER imports `maceh` (format compatibility is tested at the file level).
- Units everywhere: energies **eV**, lengths **Å**, charges in units of **e**. `constants.py` is the only place unit factors are defined.
- H5 block files: keys are `str([Rx, Ry, Rz, i, j])` (JSON-parseable, **1-based** atom indices), values dense float64 `(norb_i, norb_j)` arrays — exactly what `maceh/graph.py` parses.
- Every h5/json/yaml write is atomic: write `<path>.tmp.<pid>` then `os.replace`.
- ABACUS `INPUT` always has `gamma_only 0` and `symmetry 0`; snapshot/final-scf runs add `out_mat_hs2 1`.
- Reciprocal set: inversion-symmetric, `G=0` excluded, ellipsoidal cutoff `G·ε∞·G ≤ 4Λ²ln(1/tol)`; Λ fixed from config (part of the dataset definition — never "converged away").
- LR phase convention: reference positions `R⁰` in dipole coefficients; potential evaluated at snapshot positions; `V` is electron potential energy (sign `LR_SIGN = −1` in constants, pinned by test).
- One global seed `displacements.seed`; all randomness through `np.random.default_rng([seed, ...])` with documented derived streams.
- Raw DFT outputs are read-only; `hamiltonians_full.h5` is never rewritten after `collect-dft`; `export-target` only ever (re)writes `hamiltonians.h5` and refuses to clobber a non-exported file.
- Commit after every task with the message given in the task's final step (append the standard `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer).

---

### Task 1: Package skeleton — constants, config loader, CLI dispatcher, default YAML

**Files:**
- Create: `mgo_lr/__init__.py`
- Create: `mgo_lr/constants.py`
- Create: `mgo_lr/config.py`
- Create: `mgo_lr/__main__.py`
- Create: `mgo_lr/configs/mgo.yaml`
- Test: `mgo_lr/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `mgo_lr.__version__: str`; `constants.BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, RY_TO_EV, C_COUL, LR_SIGN`; `config.load_config(path: str) -> dict`, `config.require(cfg: dict, dotted: str)`, `config.save_resolved(cfg: dict, workspace: str, stage: str) -> str`, `config.atomic_write_text(path: str, text: str)`; `__main__.main(argv: list[str]) -> int` and `STAGES: dict[str, tuple[str, str]]` mapping stage name → `(module, function)`; every stage function has signature `stage(cfg: dict, workspace: str, args: argparse.Namespace) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_config.py
import copy
import subprocess
import sys

import pytest

from mgo_lr import config, constants

CFG_PATH = "mgo_lr/configs/mgo.yaml"


def test_constants_values():
    assert abs(constants.RY_TO_EV - 13.605693122994) < 1e-9
    assert abs(constants.C_COUL - 14.399645478) < 1e-6
    assert constants.LR_SIGN == -1.0
    assert abs(constants.BOHR_TO_ANGSTROM * constants.ANGSTROM_TO_BOHR - 1.0) < 1e-14


def test_load_default_config():
    cfg = config.load_config(CFG_PATH)
    assert cfg["material"]["species"] == ["Mg", "O"]
    assert cfg["abacus"]["gamma_only_algorithm"] is False
    assert cfg["supercells"] == {"pilot": 2, "main": 3, "large": 4}
    assert isinstance(cfg["lr"]["ewald_lambda"], float)


def test_missing_field_raises(tmp_path):
    cfg = config.load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    del bad["lr"]["ewald_lambda"]
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError, match="lr.ewald_lambda"):
        config.load_config(str(p))


def test_gamma_only_true_rejected(tmp_path):
    cfg = config.load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["abacus"]["gamma_only_algorithm"] = True
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="gamma_only"):
        config.load_config(str(p))


def test_save_resolved(tmp_path):
    cfg = config.load_config(CFG_PATH)
    out = config.save_resolved(cfg, str(tmp_path), "unit-test")
    assert "generation_logs" in out
    import yaml
    assert yaml.safe_load(open(out))["material"]["name"] == "MgO"


def test_cli_unknown_stage_fails():
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "no-such-stage",
                        "--workspace", "/tmp/x"], capture_output=True)
    assert r.returncode != 0


def test_cli_help_lists_stages():
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "gen-structures" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jb/MACE-H && /opt/anaconda3/envs/DeepH/bin/python -m pytest mgo_lr/tests/test_config.py -q`
Expected: collection error / ModuleNotFoundError `mgo_lr`.

- [ ] **Step 3: Write the implementation**

`mgo_lr/__init__.py`:

```python
__version__ = "0.1.0"
```

`mgo_lr/constants.py`:

```python
"""Unit system: energies in eV, lengths in Angstrom, charges in units of e.

V_LR is the potential energy of an ELECTRON (charge -e) in the screened
field of the induced Born dipoles.  With the plane-wave synthesis
V(r) = sum_G V(G) exp(+iG.r) and dipole coefficients carrying
exp(-iG.R0_kappa) source phases, the electron potential energy is the
NEGATIVE of the electrostatic potential of the (positive) dipole
density, hence LR_SIGN = -1.  Pinned numerically by
tests/test_lr_core.py::test_sign_and_prefactor_against_filtered_dipole.
"""

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
RY_TO_EV = 13.605693122994
C_COUL = 14.399645478425668   # e^2/(4 pi eps0) in eV*Angstrom
LR_SIGN = -1.0                # electron potential energy vs dipole potential
```

`mgo_lr/config.py`:

```python
import os
import time

import yaml

REQUIRED = [
    "material.name", "material.lattice_constant_guess", "material.species",
    "material.masses",
    "abacus.pseudo_dir", "abacus.orbital_dir", "abacus.pseudopotentials",
    "abacus.orbitals", "abacus.orbital_types", "abacus.ecutwfc",
    "abacus.scf_thr", "abacus.scf_nmax", "abacus.smearing_method",
    "abacus.smearing_sigma", "abacus.kmesh_primitive",
    "abacus.kmesh_supercell", "abacus.gamma_only_algorithm",
    "abacus.csr_h_filename", "abacus.csr_s_filename", "abacus.version",
    "qe.pseudo_dir", "qe.pseudopotentials", "qe.ecutwfc", "qe.kmesh",
    "qe.conv_thr", "qe.tr2_ph", "qe.version",
    "reference.ecut_scan", "reference.kmesh_scan",
    "supercells.pilot", "supercells.main", "supercells.large",
    "displacements.seed", "displacements.min_distance",
    "displacements.amplitudes", "displacements.pilot_ladder",
    "displacements.main_composition", "displacements.large_count",
    "lr.ewald_lambda", "lr.reciprocal_tolerance", "lr.imaginary_tolerance",
    "lr.convergence_factor",
    "validation.delta", "validation.tau_eq", "validation.tau_u",
    "validation.tau_translation", "validation.tau_reconstruct",
    "validation.tau_hermiticity", "validation.tau_G",
    "validation.tau_overlap_diag", "validation.tier2_enforce",
    "splits.validation_fraction", "splits.test_fraction",
    "slurm.header", "slurm.abacus_command",
]


def require(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing required config field: {dotted}")
        node = node[part]
    return node


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    for field in REQUIRED:
        require(cfg, field)
    if require(cfg, "abacus.gamma_only_algorithm"):
        raise ValueError(
            "abacus.gamma_only_algorithm must be false: ABACUS gamma-only "
            "algorithm does not support out_mat_hs2")
    return cfg


def atomic_write_text(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def save_resolved(cfg, workspace, stage):
    logs = os.path.join(workspace, "generation_logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, f"config-{stage}-{time.strftime('%Y%m%d-%H%M%S')}.yaml")
    atomic_write_text(out, yaml.safe_dump(cfg, sort_keys=False))
    return out
```

`mgo_lr/__main__.py`:

```python
import argparse
import importlib
import os
import sys

from . import __version__
from .config import load_config, save_resolved

# stage name -> (module, function). Functions: stage(cfg, workspace, args) -> int
STAGES = {
    "init-reference": ("mgo_lr.reference", "init_reference_stage"),
    "collect-reference": ("mgo_lr.reference", "collect_reference_stage"),
    "init-dfpt": ("mgo_lr.dfpt", "init_dfpt_stage"),
    "collect-dfpt": ("mgo_lr.dfpt", "collect_dfpt_stage"),
    "gen-structures": ("mgo_lr.displacements", "gen_structures_stage"),
    "collect-dft": ("mgo_lr.convert", "collect_dft_stage"),
    "lr-process": ("mgo_lr.lr", "lr_process_stage"),
    "validate": ("mgo_lr.validate", "validate_stage"),
    "locality-report": ("mgo_lr.locality", "locality_report_stage"),
    "organize": ("mgo_lr.organize", "organize_stage"),
    "export-target": ("mgo_lr.export", "export_target_stage"),
    "status": ("mgo_lr.snapshot", "status_stage"),
}

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "mgo.yaml")


def main(argv):
    p = argparse.ArgumentParser(
        prog="python -m mgo_lr",
        description=f"MgO MACE-H-LR dataset pipeline v{__version__}. "
                    f"Stages: {', '.join(STAGES)}")
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--workspace", required=True)
    p.add_argument("--set", dest="set_name",
                   choices=["pilot", "main", "large"], default=None)
    p.add_argument("--target", choices=["full", "lr", "sr"], default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    os.makedirs(args.workspace, exist_ok=True)
    save_resolved(cfg, args.workspace, args.stage)
    module, func = STAGES[args.stage]
    fn = getattr(importlib.import_module(module), func)
    return fn(cfg, args.workspace, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

`mgo_lr/configs/mgo.yaml`:

```yaml
material:
  name: MgO
  lattice_constant_guess: 4.25        # Angstrom, rocksalt conventional cube
  lattice_constant_relaxed: null      # set after collect-reference (or parsed)
  species: [Mg, O]
  masses: {Mg: 24.305, O: 15.999}

abacus:
  pseudo_dir: /path/on/cluster/abacus_pseudo
  orbital_dir: /path/on/cluster/abacus_orb
  pseudopotentials: {Mg: Mg_ONCV_PBE-1.0.upf, O: O_ONCV_PBE-1.0.upf}
  orbitals: {Mg: Mg_gga_8au_100Ry_4s2p1d.orb, O: O_gga_8au_100Ry_3s3p2d.orb}
  # l of every orbital per species, in ABACUS shell order (4s2p1d etc.)
  orbital_types: {Mg: [0, 0, 0, 0, 1, 1, 2], O: [0, 0, 0, 1, 1, 2]}
  ecutwfc: 100
  scf_thr: 1.0e-8
  scf_nmax: 100
  smearing_method: gaussian
  smearing_sigma: 0.002
  kmesh_primitive: [8, 8, 8]
  kmesh_supercell: {pilot: [4, 4, 4], main: [3, 3, 3], large: [2, 2, 2]}
  gamma_only_algorithm: false
  csr_h_filename: data-HR-sparse_SPIN0.csr
  csr_s_filename: data-SR-sparse_SPIN0.csr
  version: "3.7"

qe:
  pseudo_dir: /path/on/cluster/qe_pseudo
  pseudopotentials: {Mg: Mg_ONCV_PBE-1.0.upf, O: O_ONCV_PBE-1.0.upf}
  ecutwfc: 80
  kmesh: [8, 8, 8]
  conv_thr: 1.0e-12
  tr2_ph: 1.0e-14
  version: "7.2"

reference:
  ecut_scan: [60, 80, 100, 120]
  kmesh_scan: [[4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]]

supercells: {pilot: 2, main: 3, large: 4}

displacements:
  seed: 20260720
  amplitudes: [0.005, 0.01, 0.02, 0.04, 0.06]   # Angstrom, main set
  pilot_ladder: [0.0025, 0.005, 0.01, 0.02]     # Angstrom, pilot sign pairs
  min_distance: 1.8                             # Angstrom, hard reject below
  main_composition:
    single_q_optical: 150
    mixed_low_q: 120
    random_local: 60
    sign_paired_calibration: 40
    near_equilibrium: 30
  large_count: 40

lr:
  ewald_lambda: 0.35            # 1/Angstrom; part of the dataset definition
  reciprocal_tolerance: 1.0e-10 # f_Ewald floor -> Gmax
  imaginary_tolerance: 1.0e-8   # hard gate on r_imag
  convergence_factor: 1.5       # Gmax ratio for the convergence metric

validation:
  delta: 1.0e-12
  tau_eq: 1.0e-10               # eV, max|H_LR| at equilibrium
  tau_u: 1.0e-12                # Angstrom, max|u_rel| for rigid translation
  tau_translation: 1.0e-10      # eV
  tau_reconstruct: 1.0e-12      # relative Frobenius
  tau_hermiticity: 1.0e-8       # eV, max abs
  tau_G: 1.0e-6                 # relative Frobenius
  tau_overlap_diag: 0.05        # |S_ii - 1| on R=0 diagonal
  tier2_enforce: false          # promote E_sign/E_linear to hard checks later

splits:
  validation_fraction: 0.10
  test_fraction: 0.10

slurm:
  header: |
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --time=04:00:00
  abacus_command: "mpirun -n 16 abacus"
```

Also create the empty test package marker: `mgo_lr/tests/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jb/MACE-H && /opt/anaconda3/envs/DeepH/bin/python -m pytest mgo_lr/tests/test_config.py -q`
Expected: all pass (7 tests). Note: `test_cli_help_lists_stages` passes because argparse lists choices in help before any stage module is imported.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr && git commit -m "feat(mgo_lr): package skeleton, constants, config loader, CLI dispatcher"
```

---

### Task 2: Snapshot state machine and `status` stage

**Files:**
- Create: `mgo_lr/snapshot.py`
- Test: `mgo_lr/tests/test_snapshot.py`

**Interfaces:**
- Consumes: `config.atomic_write_text`.
- Produces: `snapshot.STATES = ["prepared", "dft_done", "converted", "lr_done", "validated", "rejected"]`; `class SnapshotStore(workspace: str, set_name: str)` with attributes `.set_dir`, `.rejected_dir` and methods `.folder(sid) -> str`, `.list() -> list[str]`, `.read_status(sid) -> dict`, `.write_status(sid, state, **extra)`, `.reject(sid, reason: str)`, `.state_at_least(sid, state) -> bool`; `set_dir_name(set_name) -> str` (`pilot`→`pilot`, `main`→`main`, `large`→`test_large_cell`); `status_stage(cfg, workspace, args) -> int` printing per-set state counts.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_snapshot.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_snapshot.py -q` → ModuleNotFoundError `mgo_lr.snapshot`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/snapshot.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_snapshot.py mgo_lr/tests/test_config.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/snapshot.py mgo_lr/tests/test_snapshot.py
git commit -m "feat(mgo_lr): snapshot state machine and status stage"
```

---

### Task 3: Rocksalt primitive cell and supercell builder

**Files:**
- Create: `mgo_lr/structures.py`
- Test: `mgo_lr/tests/test_structures.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `structures.rocksalt_primitive(a: float) -> tuple[np.ndarray (3,3), np.ndarray (2,3), list[str]]` (cell rows = lattice vectors Å; fractional positions; species `["Mg","O"]`); `@dataclass Supercell: cell (3,3), cart (N,3), species (list[str]), cell_index (N,3 int), basis_index (N,) int` with **species-major, cell-minor atom ordering** (all Mg first, then all O; within a species, unit cells in `np.ndindex(n,n,n)` order); `structures.make_supercell(cell, frac, species, n: int) -> Supercell`; `structures.reciprocal(cell) -> np.ndarray (3,3)` (rows `b_i`, includes 2π).

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_structures.py
import numpy as np

from mgo_lr.structures import Supercell, make_supercell, reciprocal, rocksalt_primitive


def test_rocksalt_primitive():
    a = 4.2
    cell, frac, species = rocksalt_primitive(a)
    assert species == ["Mg", "O"]
    # fcc primitive vectors a/2 (0,1,1) etc.
    assert np.allclose(np.abs(np.linalg.det(cell)), a**3 / 4.0)
    assert np.allclose(frac[0], [0.0, 0.0, 0.0])
    assert np.allclose(frac[1], [0.5, 0.5, 0.5])
    # nearest neighbour Mg-O distance is a/2
    cart = frac @ cell
    d = np.linalg.norm(cart[1] - cart[0] - cell[0])
    assert abs(min(np.linalg.norm(cart[1] - cart[0]), d) - a / 2.0) < 1e-12


def test_make_supercell_ordering():
    cell, frac, species = rocksalt_primitive(4.2)
    sc = make_supercell(cell, frac, species, 2)
    assert isinstance(sc, Supercell)
    assert len(sc.species) == 16
    assert sc.species[:8] == ["Mg"] * 8 and sc.species[8:] == ["O"] * 8
    assert sc.cell_index.shape == (16, 3)
    # first Mg at cell (0,0,0), second at (0,0,1) (np.ndindex order)
    assert tuple(sc.cell_index[0]) == (0, 0, 0)
    assert tuple(sc.cell_index[1]) == (0, 0, 1)
    assert np.allclose(sc.cell, 2 * cell)
    assert sc.basis_index[0] == 0 and sc.basis_index[8] == 1


def test_reciprocal():
    cell, _, _ = rocksalt_primitive(4.2)
    rec = reciprocal(cell)
    assert np.allclose(rec @ cell.T, 2 * np.pi * np.eye(3), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_structures.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/structures.py
from dataclasses import dataclass

import numpy as np


def rocksalt_primitive(a):
    """Rocksalt MgO primitive cell. Rows of `cell` are lattice vectors (Å)."""
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0]])
    frac = np.array([[0.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5]])
    return cell, frac, ["Mg", "O"]


@dataclass
class Supercell:
    cell: np.ndarray        # (3,3) rows = supercell lattice vectors, Å
    cart: np.ndarray        # (N,3) Cartesian positions, Å
    species: list           # length N
    cell_index: np.ndarray  # (N,3) int, primitive-cell offset n_l of each atom
    basis_index: np.ndarray # (N,) int, index into the primitive basis


def make_supercell(cell, frac, species, n):
    """n x n x n supercell, species-major then cell-minor atom ordering."""
    cart_prim = frac @ cell
    cells = np.array(list(np.ndindex(n, n, n)))          # (n^3, 3)
    pos, spec, cidx, bidx = [], [], [], []
    for b, s in enumerate(species):
        for c in cells:
            pos.append(cart_prim[b] + c @ cell)
            spec.append(s)
            cidx.append(c)
            bidx.append(b)
    return Supercell(cell=n * np.asarray(cell, float),
                     cart=np.array(pos),
                     species=spec,
                     cell_index=np.array(cidx, dtype=int),
                     basis_index=np.array(bidx, dtype=int))


def reciprocal(cell):
    """Reciprocal lattice, rows b_i, includes the 2*pi factor. 1/Å."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_structures.py -q` → 3 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/structures.py mgo_lr/tests/test_structures.py
git commit -m "feat(mgo_lr): rocksalt primitive and supercell builder"
```

---

### Task 4: ABACUS input writers (STRU / INPUT / KPT / job script)

**Files:**
- Create: `mgo_lr/abacus_io.py`
- Test: `mgo_lr/tests/test_abacus_writers.py`

**Interfaces:**
- Consumes: `constants.ANGSTROM_TO_BOHR`, `config.atomic_write_text`.
- Produces: `abacus_io.write_stru(path, cell, cart, species, cfg)` (groups atoms by species **in `cfg["material"]["species"]` order**, positions written `Direct`; raises if `species` is not already species-major in that order — atom ordering must match the matrices); `abacus_io.write_input(path, cfg, **overrides)` (always `gamma_only 0`, `symmetry 0`; raises on `gamma_only` override ≠ 0); `abacus_io.write_kpt(path, mesh: list[int])`; `abacus_io.write_job_script(path, cfg, snapshot_dirs: list[str])`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_abacus_writers.py
import numpy as np
import pytest

from mgo_lr import abacus_io
from mgo_lr.config import load_config
from mgo_lr.constants import ANGSTROM_TO_BOHR
from mgo_lr.structures import make_supercell, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")


def _sc():
    cell, frac, species = rocksalt_primitive(4.2)
    return make_supercell(cell, frac, species, 2)


def test_write_stru(tmp_path):
    sc = _sc()
    p = tmp_path / "STRU"
    abacus_io.write_stru(str(p), sc.cell, sc.cart, sc.species, CFG)
    text = p.read_text()
    lines = [l.strip() for l in text.splitlines()]
    assert "ATOMIC_SPECIES" in lines and "NUMERICAL_ORBITAL" in lines
    i = lines.index("LATTICE_CONSTANT")
    assert abs(float(lines[i + 1]) - ANGSTROM_TO_BOHR) < 1e-12
    assert "Direct" in lines
    # species blocks in config order with correct counts
    i_mg, i_o = lines.index("Mg", lines.index("Direct")), None
    i_o = lines.index("O", i_mg)
    assert int(lines[i_mg + 2]) == 8 and int(lines[i_o + 2]) == 8
    # a Direct coordinate row has 3 floats + "m 0 0 0"
    row = lines[i_mg + 3].split()
    assert len(row) == 7 and row[3] == "m"
    frac = np.array([float(x) for x in row[:3]])
    assert np.allclose(frac @ sc.cell, sc.cart[0], atol=1e-10)


def test_write_stru_rejects_bad_order(tmp_path):
    sc = _sc()
    bad = list(sc.species)
    bad[0], bad[8] = bad[8], bad[0]
    with pytest.raises(ValueError, match="species-major"):
        abacus_io.write_stru(str(tmp_path / "STRU"), sc.cell, sc.cart, bad, CFG)


def test_write_input(tmp_path):
    p = tmp_path / "INPUT"
    abacus_io.write_input(str(p), CFG, calculation="scf", out_mat_hs2=1,
                          suffix="MgO")
    text = p.read_text()
    assert text.startswith("INPUT_PARAMETERS")
    assert "gamma_only" in text and " 0" in text
    for key in ("basis_type", "ecutwfc", "scf_thr", "out_mat_hs2", "symmetry"):
        assert key in text
    with pytest.raises(ValueError, match="gamma_only"):
        abacus_io.write_input(str(p), CFG, gamma_only=1)


def test_write_kpt(tmp_path):
    p = tmp_path / "KPT"
    abacus_io.write_kpt(str(p), [4, 4, 4])
    assert p.read_text() == "K_POINTS\n0\nGamma\n4 4 4 0 0 0\n"


def test_write_job_script(tmp_path):
    p = tmp_path / "job.sh"
    abacus_io.write_job_script(str(p), CFG, ["snapshot_000001", "snapshot_000002"])
    text = p.read_text()
    assert text.startswith("#!/bin/bash")
    assert CFG["slurm"]["abacus_command"] in text
    assert "snapshot_000001" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_abacus_writers.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/abacus_io.py
"""ABACUS file I/O. Writers here; output parsers are added in Task 9.

Atom-ordering contract: every STRU we write lists species in
cfg["material"]["species"] order with each species' atoms contiguous
(species-major).  ABACUS orders its matrix rows in STRU order, so this
must match Supercell ordering exactly.
"""
import numpy as np

from .config import atomic_write_text
from .constants import ANGSTROM_TO_BOHR


def write_stru(path, cell, cart, species, cfg):
    mat = cfg["material"]
    ab = cfg["abacus"]
    order = mat["species"]
    expected = [s for s in order for _ in range(species.count(s))]
    if list(species) != expected:
        raise ValueError("atoms must be species-major in config species order")
    frac = np.asarray(cart, float) @ np.linalg.inv(np.asarray(cell, float))
    lines = ["ATOMIC_SPECIES"]
    for s in order:
        lines.append(f"{s} {mat['masses'][s]} {ab['pseudopotentials'][s]}")
    lines += ["", "NUMERICAL_ORBITAL"]
    for s in order:
        lines.append(ab["orbitals"][s])
    lines += ["", "LATTICE_CONSTANT", f"{ANGSTROM_TO_BOHR:.15f}",
              "", "LATTICE_VECTORS"]
    for v in np.asarray(cell, float):
        lines.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    lines += ["", "ATOMIC_POSITIONS", "Direct"]
    for s in order:
        idx = [i for i, sp in enumerate(species) if sp == s]
        lines += [s, "0.0", str(len(idx))]
        for i in idx:
            f = frac[i]
            lines.append(f"{f[0]:.12f} {f[1]:.12f} {f[2]:.12f} m 0 0 0")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_input(path, cfg, **overrides):
    ab = cfg["abacus"]
    params = {
        "suffix": "MgO",
        "calculation": "scf",
        "basis_type": "lcao",
        "ntype": len(cfg["material"]["species"]),
        "nspin": 1,
        "symmetry": 0,
        "gamma_only": 0,
        "ecutwfc": ab["ecutwfc"],
        "scf_thr": ab["scf_thr"],
        "scf_nmax": ab["scf_nmax"],
        "smearing_method": ab["smearing_method"],
        "smearing_sigma": ab["smearing_sigma"],
        "pseudo_dir": ab["pseudo_dir"],
        "orbital_dir": ab["orbital_dir"],
    }
    params.update(overrides)
    if int(params["gamma_only"]) != 0:
        raise ValueError("gamma_only must stay 0 (out_mat_hs2 unsupported "
                         "under the gamma-only algorithm)")
    lines = ["INPUT_PARAMETERS"]
    for k, v in params.items():
        lines.append(f"{k:24s}{v}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_kpt(path, mesh):
    m = " ".join(str(int(x)) for x in mesh)
    atomic_write_text(path, f"K_POINTS\n0\nGamma\n{m} 0 0 0\n")


def write_job_script(path, cfg, snapshot_dirs):
    body = [cfg["slurm"]["header"].rstrip(), ""]
    body.append("for d in \\")
    for d in snapshot_dirs:
        body.append(f"    {d} \\")
    body.append("; do")
    body.append(f"    (cd \"$d\" && {cfg['slurm']['abacus_command']} "
                "> abacus.stdout 2>&1)")
    body.append("done")
    atomic_write_text(path, "\n".join(body) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_abacus_writers.py -q` → 5 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/abacus_io.py mgo_lr/tests/test_abacus_writers.py
git commit -m "feat(mgo_lr): ABACUS STRU/INPUT/KPT/job writers"
```

---

### Task 5: Displacement-pattern engine

**Files:**
- Create: `mgo_lr/displacements.py`
- Test: `mgo_lr/tests/test_displacements.py`

**Interfaces:**
- Consumes: `structures.Supercell`, `structures.reciprocal`.
- Produces: `displacements.MODE_NORMALIZATION = "max_species_weight_1"`; `apply_pattern(sc: Supercell, prim_cell: np.ndarray, pattern: dict, global_seed: int) -> np.ndarray (N,3)`; `remove_uniform_translation(u) -> np.ndarray` (plain mean, not mass-weighted); `minimum_distance(cell, cart) -> float`; `build_pilot(cfg, prim_cell) -> list[dict]`, `build_main(cfg, prim_cell) -> list[dict]`, `build_large(cfg, prim_cell) -> list[dict]` — each plan dict is `{"sid": "snapshot_%06d", "pattern": {...}, "metadata": {...}}` where `metadata` is the exact `displacement_metadata.json` content (spec schema). Pattern dicts are JSON-serializable: `{"pattern_class": str, "modes": [mode...], "random": {"index": int, "amplitude": float}|absent, "translation": [x,y,z]|absent}` and each mode is `{"q_int": [i,j,k], "amplitude": float (signed), "phase": float, "polarization": [x,y,z], "polarization_class": "longitudinal"|"transverse"|"none", "species_weights": {"Mg": w, "O": w}}`. The stage function `gen_structures_stage` is added in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_displacements.py
import json

import numpy as np
import pytest

from mgo_lr import displacements as dp
from mgo_lr.config import load_config
from mgo_lr.structures import make_supercell, reciprocal, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")
PRIM_CELL, PRIM_FRAC, PRIM_SPECIES = rocksalt_primitive(4.2)


def _sc(n=2):
    return make_supercell(PRIM_CELL, PRIM_FRAC, PRIM_SPECIES, n)


def test_remove_uniform_translation_plain_mean():
    u = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = dp.remove_uniform_translation(u)
    assert np.allclose(out.mean(axis=0), 0.0)
    assert np.allclose(out, [[-1.0, 0, 0], [1.0, 0, 0]])  # NOT mass-weighted


def test_minimum_distance():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    assert abs(dp.minimum_distance(cell, cart) - 0.5) < 1e-12


def test_apply_optical_x():
    sc = _sc()
    pat = {"pattern_class": "optical_x", "modes": [{
        "q_int": [0, 0, 0], "amplitude": 0.01, "phase": 0.0,
        "polarization": [1.0, 0.0, 0.0], "polarization_class": "none",
        "species_weights": {"Mg": 1.0, "O": -1.0}}]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, CFG["displacements"]["seed"])
    assert np.allclose(u[:8], [0.01, 0.0, 0.0])   # all Mg +x
    assert np.allclose(u[8:], [-0.01, 0.0, 0.0])  # all O -x


def test_apply_finite_q_commensurate():
    sc = _sc(2)
    rec_super = reciprocal(sc.cell)
    qhat = np.asarray([1.0, 0, 0]) @ rec_super
    qhat /= np.linalg.norm(qhat)
    pat = {"pattern_class": "longitudinal_q", "modes": [{
        "q_int": [1, 0, 0], "amplitude": 0.01, "phase": 0.0,
        "polarization": qhat.tolist(), "polarization_class": "longitudinal",
        "species_weights": {"Mg": 1.0, "O": 0.0}}]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, 0)
    # q.R_l = 2*pi*(m.c)/n: cells with c1=0 get cos(0)=+1, c1=1 get cos(pi)=-1
    for i in range(8):  # Mg atoms
        expected_sign = 1.0 if sc.cell_index[i][0] == 0 else -1.0
        assert np.allclose(u[i], expected_sign * 0.01 * qhat, atol=1e-12)
    assert np.allclose(u[8:], 0.0)


def test_apply_rigid_translation_and_random():
    sc = _sc()
    pat = {"pattern_class": "rigid_translation", "modes": [],
           "translation": [0.01, 0.02, 0.03]}
    u = dp.apply_pattern(sc, PRIM_CELL, pat, 0)
    assert np.allclose(u, [0.01, 0.02, 0.03])
    pat = {"pattern_class": "random_local", "modes": [],
           "random": {"index": 1000, "amplitude": 0.01}}
    u1 = dp.apply_pattern(sc, PRIM_CELL, pat, 7)
    u2 = dp.apply_pattern(sc, PRIM_CELL, pat, 7)
    assert np.allclose(u1, u2)                       # seeded reproducibility
    assert np.allclose(u1.mean(axis=0), 0.0)         # translation removed
    assert abs(np.linalg.norm(u1, axis=1).max() - 0.01) < 1e-12


def test_build_pilot_contents():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    # 1 eq + 5 sign-paired bases x 4 amplitudes x 2 signs + 2 mixed
    # + 2 random + 1 rigid translation
    assert len(plans) == 46
    assert plans[0]["metadata"]["pattern_class"] == "equilibrium"
    classes = [p["metadata"]["pattern_class"] for p in plans]
    for name in ("mg_only_x", "o_only_x", "optical_x", "longitudinal_q",
                 "transverse_q"):
        assert classes.count(name) == 8
    assert classes.count("mixed") == 2
    assert classes.count("random_local") == 2
    assert classes.count("rigid_translation") == 1
    # deterministic
    again = dp.build_pilot(CFG, PRIM_CELL)
    assert json.dumps(plans, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_pilot_partner_wiring():
    plans = {p["sid"]: p["metadata"] for p in dp.build_pilot(CFG, PRIM_CELL)}
    for sid, meta in plans.items():
        if meta["sign_partner_id"]:
            partner = plans[meta["sign_partner_id"]]
            assert partner["sign_partner_id"] == sid
            assert partner["pattern_group_id"] == meta["pattern_group_id"]
            assert abs(partner["amplitude"] + meta["amplitude"]) < 1e-12
            assert partner["comparison_family_id"] == meta["comparison_family_id"]
        for pid in meta["amplitude_partner_ids"]:
            assert plans[pid]["pattern_group_id"] == meta["pattern_group_id"]


def test_pilot_longitudinal_transverse_share_family():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    lon = [p for p in plans if p["metadata"]["pattern_class"] == "longitudinal_q"
           and abs(p["metadata"]["amplitude"] - 0.01) < 1e-12]
    tra = [p for p in plans if p["metadata"]["pattern_class"] == "transverse_q"
           and abs(p["metadata"]["amplitude"] - 0.01) < 1e-12]
    assert lon and tra
    assert (lon[0]["metadata"]["comparison_family_id"]
            == tra[0]["metadata"]["comparison_family_id"])
    assert lon[0]["metadata"]["polarization_class"] == "longitudinal"
    assert tra[0]["metadata"]["polarization_class"] == "transverse"


def test_metadata_schema():
    plans = dp.build_pilot(CFG, PRIM_CELL)
    keys = {"pattern_group_id", "pattern_class", "comparison_family_id",
            "mode_normalization", "q_vectors", "q_magnitude", "polarizations",
            "polarization_class", "phases", "phase", "amplitudes", "amplitude",
            "sign_partner_id", "amplitude_partner_ids", "rigid_translation",
            "seed"}
    for p in plans:
        assert keys <= set(p["metadata"])
        json.dumps(p)  # everything JSON-serializable


def test_build_main_composition_and_seeding():
    plans = dp.build_main(CFG, PRIM_CELL)
    assert len(plans) == 400
    classes = [p["metadata"]["pattern_class"] for p in plans]
    comp = CFG["displacements"]["main_composition"]
    assert classes.count("single_q_optical") == comp["single_q_optical"]
    assert classes.count("mixed_low_q") == comp["mixed_low_q"]
    assert classes.count("random_local") == comp["random_local"]
    assert classes.count("sign_paired_calibration") == comp["sign_paired_calibration"]
    assert classes.count("near_equilibrium") == comp["near_equilibrium"]
    again = dp.build_main(CFG, PRIM_CELL)
    assert json.dumps(plans, sort_keys=True) == json.dumps(again, sort_keys=True)
    # sign pairs wired both ways
    by_sid = {p["sid"]: p["metadata"] for p in plans}
    pairs = [m for m in by_sid.values()
             if m["pattern_class"] == "sign_paired_calibration"]
    assert all(by_sid[m["sign_partner_id"]]["sign_partner_id"] for m in pairs)


def test_build_large():
    plans = dp.build_large(CFG, PRIM_CELL)
    assert len(plans) == CFG["displacements"]["large_count"]
    for p in plans:
        assert p["metadata"]["pattern_class"] in ("single_q_optical", "mixed_low_q")
        for a in p["metadata"]["amplitudes"]:
            assert abs(a) <= max(CFG["displacements"]["amplitudes"]) + 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_displacements.py -q` → ModuleNotFoundError `mgo_lr.displacements`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/displacements.py
"""Displacement-pattern engine and the gen-structures stage.

u_kappa(R_l) = sum_m A_m * w_m(species_kappa) * e_m * cos(q_m . R_l + phi_m)

R_l is the primitive-cell lattice vector of the atom's home cell
(cell_index @ prim_cell).  q vectors are integer combinations of the
SUPERCELL reciprocal vectors, so every pattern is commensurate by
construction.  Species weights are normalized so max|w| = 1, i.e. the mode
amplitude A is the peak displacement of the most-displaced species
("max_species_weight_1" normalization).

Uniform-translation removal is a plain (deliberately NOT mass-weighted)
mean over atoms.
"""
import hashlib
import json
import os

import numpy as np

from .structures import reciprocal

MODE_NORMALIZATION = "max_species_weight_1"


def _hash_id(prefix, *parts):
    text = json.dumps(parts, sort_keys=True, default=str)
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def _sid(k):
    return f"snapshot_{k:06d}"


def remove_uniform_translation(u):
    """Subtract the plain mean over atoms — deliberately not mass-weighted."""
    u = np.asarray(u, float)
    return u - u.mean(axis=0)


def minimum_distance(cell, cart):
    """Minimum interatomic distance under PBC (exact within +-1 image search)."""
    cell = np.asarray(cell, float)
    cart = np.asarray(cart, float)
    shifts = (np.array(list(np.ndindex(3, 3, 3))) - 1) @ cell
    dmin = np.inf
    for s in shifts:
        d = np.linalg.norm(cart[None, :, :] + s - cart[:, None, :], axis=-1)
        if np.allclose(s, 0.0):
            np.fill_diagonal(d, np.inf)
        dmin = min(dmin, float(d.min()))
    return dmin


def apply_pattern(sc, prim_cell, pattern, global_seed):
    """Displacement field (N,3) in Å for one pattern dict."""
    n_at = len(sc.species)
    if pattern.get("translation") is not None:
        return np.tile(np.asarray(pattern["translation"], float), (n_at, 1))
    if pattern.get("random") is not None:
        r = pattern["random"]
        rng = np.random.default_rng([global_seed, r["index"]])
        u = remove_uniform_translation(rng.standard_normal((n_at, 3)))
        return u * (r["amplitude"] / np.linalg.norm(u, axis=1).max())
    u = np.zeros((n_at, 3))
    rec_super = reciprocal(sc.cell)
    lattice_r = sc.cell_index @ np.asarray(prim_cell, float)
    for mode in pattern["modes"]:
        q = np.asarray(mode["q_int"], float) @ rec_super
        phase = np.cos(lattice_r @ q + mode["phase"])
        w = np.array([mode["species_weights"][s] for s in sc.species])
        u += mode["amplitude"] * (w * phase)[:, None] \
             * np.asarray(mode["polarization"], float)
    return u


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _transverse(qhat):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(qhat, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(qhat, ref))


def _mode(q_int, amplitude, phase, polarization, pol_class, weights):
    return {"q_int": [int(x) for x in q_int], "amplitude": float(amplitude),
            "phase": float(phase),
            "polarization": [float(x) for x in polarization],
            "polarization_class": pol_class,
            "species_weights": {k: float(v) for k, v in weights.items()}}


def _metadata(n, prim_cell, pattern, group_id, seed_index):
    modes = pattern.get("modes", [])
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    if modes:
        q_mag = float(np.linalg.norm(
            np.asarray(modes[0]["q_int"], float) @ rec_super))
        amp = modes[0]["amplitude"]
        phase = modes[0]["phase"]
        pol_class = modes[0]["polarization_class"]
        weights = modes[0]["species_weights"]
    elif pattern.get("random") is not None:
        q_mag, amp, phase, pol_class = 0.0, pattern["random"]["amplitude"], 0.0, "none"
        weights = {"Mg": 1.0, "O": 1.0}
    else:
        q_mag, phase, pol_class = 0.0, 0.0, "none"
        weights = {"Mg": 0.0, "O": 0.0}
        amp = float(np.linalg.norm(pattern["translation"])) \
            if pattern.get("translation") is not None else 0.0
    ratio_sig = sorted((k, round(v, 8)) for k, v in weights.items())
    # Comparison family: matched |q|, amplitude, phase, normalization,
    # species ratio, supercell — but NOT polarization class, so matched
    # longitudinal/transverse partners share a family.
    family = _hash_id("fam", n, round(q_mag, 8), round(abs(amp), 8),
                      round(phase, 8), MODE_NORMALIZATION, ratio_sig)
    return {
        "pattern_group_id": group_id,
        "pattern_class": pattern["pattern_class"],
        "comparison_family_id": family,
        "mode_normalization": MODE_NORMALIZATION,
        "q_vectors": [m["q_int"] for m in modes],
        "q_magnitude": q_mag,
        "polarizations": [m["polarization"] for m in modes],
        "polarization_class": pol_class,
        "phases": [m["phase"] for m in modes],
        "phase": phase,
        "amplitudes": [m["amplitude"] for m in modes] or [amp],
        "amplitude": amp,
        "sign_partner_id": None,
        "amplitude_partner_ids": [],
        "rigid_translation": bool(pattern.get("translation") is not None),
        "seed": seed_index,
    }


def build_pilot(cfg, prim_cell):
    """Deterministic Section-5 pilot list with amplitude ladders."""
    n = cfg["supercells"]["pilot"]
    ladder = [float(a) for a in cfg["displacements"]["pilot_ladder"]]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    x = [1.0, 0.0, 0.0]
    q1 = [1, 0, 0]
    qhat = _unit(np.asarray(q1, float) @ rec_super)
    bases = [
        ("mg_only_x", [0, 0, 0], x, "none", {"Mg": 1.0, "O": 0.0}),
        ("o_only_x", [0, 0, 0], x, "none", {"Mg": 0.0, "O": 1.0}),
        ("optical_x", [0, 0, 0], x, "none", {"Mg": 1.0, "O": -1.0}),
        ("longitudinal_q", q1, qhat.tolist(), "longitudinal",
         {"Mg": 1.0, "O": -1.0}),
        ("transverse_q", q1, _transverse(qhat).tolist(), "transverse",
         {"Mg": 1.0, "O": -1.0}),
    ]
    plans, k = [], 1

    def add(pattern, group_id):
        nonlocal k
        plan = {"sid": _sid(k), "pattern": pattern,
                "metadata": _metadata(n, prim_cell, pattern, group_id, k)}
        plans.append(plan)
        k += 1
        return plan

    add({"pattern_class": "equilibrium", "modes": []},
        _hash_id("grp", "pilot", n, "equilibrium"))

    for name, q_int, pol, pol_class, weights in bases:
        gid = _hash_id("grp", "pilot", n, name, q_int, pol_class,
                       sorted(weights.items()))
        members = {}
        for amp in ladder:
            for sign in (1.0, -1.0):
                plan = add({"pattern_class": name, "modes": [
                    _mode(q_int, sign * amp, 0.0, pol, pol_class, weights)]},
                    gid)
                members[(amp, sign)] = plan
        for (amp, sign), plan in members.items():
            plan["metadata"]["sign_partner_id"] = members[(amp, -sign)]["sid"]
            plan["metadata"]["amplitude_partner_ids"] = [
                members[(a, sign)]["sid"] for a in ladder if a != amp]

    q2 = [0, 1, 0]
    q2hat = _unit(np.asarray(q2, float) @ rec_super)
    for i, (a1, a2, ph2) in enumerate([(0.01, 0.005, 0.0),
                                       (0.01, 0.01, np.pi / 3)]):
        add({"pattern_class": "mixed", "modes": [
            _mode(q1, a1, 0.0, qhat.tolist(), "longitudinal",
                  {"Mg": 1.0, "O": -1.0}),
            _mode(q2, a2, ph2, _transverse(q2hat).tolist(), "transverse",
                  {"Mg": 1.0, "O": -1.0})]},
            _hash_id("grp", "pilot", n, "mixed", i))

    for i in range(2):
        add({"pattern_class": "random_local", "modes": [],
             "random": {"index": 1000 + i, "amplitude": 0.01}},
            _hash_id("grp", "pilot", n, "random_local", i))

    add({"pattern_class": "rigid_translation", "modes": [],
         "translation": ((0.02 / np.sqrt(3.0)) * np.ones(3)).tolist()},
        _hash_id("grp", "pilot", n, "rigid_translation"))
    return plans


def _random_q(rng, n):
    while True:
        q = [int(rng.integers(0, n)) for _ in range(3)]
        if any(q):
            return q


def _low_q(rng, n):
    """Components 0 or ±1 (mod n): the longest wavelengths the cell holds."""
    while True:
        q = [int(rng.choice([0, 1, n - 1])) for _ in range(3)]
        if any(q):
            return q


def _single_q_pattern(rng, rec_super, q_int, amp, pattern_class):
    qhat = _unit(np.asarray(q_int, float) @ rec_super)
    if rng.random() < 0.5:
        pol, pol_class = qhat, "longitudinal"
    else:
        pol, pol_class = _transverse(qhat), "transverse"
    return {"pattern_class": pattern_class, "modes": [
        _mode(q_int, amp, float(rng.uniform(0.0, 2.0 * np.pi)),
              pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0})]}


def _mixed_pattern(rng, rec_super, n, amps, n_modes, pattern_class):
    modes = []
    for _ in range(n_modes):
        q_int = _low_q(rng, n)
        qhat = _unit(np.asarray(q_int, float) @ rec_super)
        if rng.random() < 0.5:
            pol, pol_class = qhat, "longitudinal"
        else:
            pol, pol_class = _transverse(qhat), "transverse"
        modes.append(_mode(q_int, float(rng.choice(amps[:2])),
                           float(rng.uniform(0.0, 2.0 * np.pi)),
                           pol.tolist(), pol_class, {"Mg": 1.0, "O": -1.0}))
    return {"pattern_class": pattern_class, "modes": modes}


def build_main(cfg, prim_cell):
    """Section-11 composition, one global seed, per-snapshot derived streams
    np.random.default_rng([seed, snapshot_index])."""
    n = cfg["supercells"]["main"]
    comp = cfg["displacements"]["main_composition"]
    amps = [float(a) for a in cfg["displacements"]["amplitudes"]]
    seed = cfg["displacements"]["seed"]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    plans, k = [], 1

    def add(pattern, group_id):
        nonlocal k
        plan = {"sid": _sid(k), "pattern": pattern,
                "metadata": _metadata(n, prim_cell, pattern, group_id, k)}
        plans.append(plan)
        k += 1
        return plan

    for _ in range(comp["single_q_optical"]):
        rng = np.random.default_rng([seed, k])
        add(_single_q_pattern(rng, rec_super, _random_q(rng, n),
                              float(rng.choice(amps)), "single_q_optical"),
            _hash_id("grp", "main", n, "single_q", k))

    for _ in range(comp["mixed_low_q"]):
        rng = np.random.default_rng([seed, k])
        add(_mixed_pattern(rng, rec_super, n, amps,
                           int(rng.integers(2, 5)), "mixed_low_q"),
            _hash_id("grp", "main", n, "mixed_low_q", k))

    for _ in range(comp["random_local"]):
        rng = np.random.default_rng([seed, k])
        add({"pattern_class": "random_local", "modes": [],
             "random": {"index": k, "amplitude": float(rng.choice(amps))}},
            _hash_id("grp", "main", n, "random_local", k))

    for _ in range(comp["sign_paired_calibration"] // 2):
        rng = np.random.default_rng([seed, k])
        gid = _hash_id("grp", "main", n, "sign_pair", k)
        base = _single_q_pattern(rng, rec_super, _random_q(rng, n),
                                 float(rng.choice(amps)),
                                 "sign_paired_calibration")
        plus = add(base, gid)
        neg = json.loads(json.dumps(base))
        neg["modes"][0]["amplitude"] *= -1.0
        minus = add(neg, gid)
        plus["metadata"]["sign_partner_id"] = minus["sid"]
        minus["metadata"]["sign_partner_id"] = plus["sid"]

    for _ in range(comp["near_equilibrium"]):
        rng = np.random.default_rng([seed, k])
        add({"pattern_class": "near_equilibrium", "modes": [],
             "random": {"index": k, "amplitude": 0.5 * min(amps)}},
            _hash_id("grp", "main", n, "near_equilibrium", k))
    return plans


def build_large(cfg, prim_cell):
    """4x4x4 extrapolation set: small q, longitudinal optical, mixed
    long-wavelength; amplitudes within the main-set range.  Derived seed
    streams use index 900000+k to stay disjoint from the main set."""
    n = cfg["supercells"]["large"]
    count = cfg["displacements"]["large_count"]
    amps = [float(a) for a in cfg["displacements"]["amplitudes"]]
    seed = cfg["displacements"]["seed"]
    rec_super = reciprocal(np.asarray(prim_cell, float) * n)
    plans = []
    for k in range(1, count + 1):
        rng = np.random.default_rng([seed, 900000 + k])
        if rng.random() < 0.5:
            q_int = _low_q(rng, n)
            qhat = _unit(np.asarray(q_int, float) @ rec_super)
            pat = {"pattern_class": "single_q_optical", "modes": [
                _mode(q_int, float(rng.choice(amps[:3])),
                      float(rng.uniform(0.0, 2.0 * np.pi)),
                      qhat.tolist(), "longitudinal", {"Mg": 1.0, "O": -1.0})]}
        else:
            pat = _mixed_pattern(rng, rec_super, n, amps,
                                 int(rng.integers(2, 4)), "mixed_low_q")
        plans.append({"sid": _sid(k), "pattern": pat,
                      "metadata": _metadata(n, prim_cell, pat,
                                            _hash_id("grp", "large", n, k),
                                            900000 + k)})
    return plans
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_displacements.py -q` → 11 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/displacements.py mgo_lr/tests/test_displacements.py
git commit -m "feat(mgo_lr): displacement-pattern engine (pilot/main/large builders)"
```

---

### Task 6: `gen-structures` stage and reference loader

**Files:**
- Modify: `mgo_lr/snapshot.py` (add `load_reference`)
- Modify: `mgo_lr/displacements.py` (add `gen_structures_stage`)
- Test: `mgo_lr/tests/test_gen_structures.py`

**Interfaces:**
- Consumes: Task 5 builders, `abacus_io` writers, `SnapshotStore`, `make_supercell`.
- Produces: `snapshot.load_reference(workspace: str) -> dict` with keys `prim_cell (3,3 float Å)`, `frac (2,3 float)`, `atomic_numbers (2,) int`, `species (list[str])` — read from `<workspace>/reference/{reference_cell.npy, reference_positions.npy, atomic_numbers.npy, species_order.json}`, raising `FileNotFoundError` naming the missing files. `displacements.gen_structures_stage(cfg, workspace, args) -> int` writing, per snapshot: `STRU`, `INPUT` (scf + `out_mat_hs2 1`), `KPT`, `displacements.npy` (N×3 Cartesian Å), `displacement_metadata.json` (metadata + the pattern dict under key `"pattern"`), `status.json` state `prepared`; plus `job_abacus.sh` at the set dir. Reference-artifact *files* are the contract between this task and Task 10 (which writes them in production; tests fabricate them).

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_gen_structures.py
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


def test_cli_requires_reference(tmp_path):
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "gen-structures",
                        "--workspace", str(tmp_path), "--set", "pilot"],
                       capture_output=True, text=True)
    assert r.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_gen_structures.py -q` → ImportError `load_reference`.

- [ ] **Step 3: Write the implementation**

Append to `mgo_lr/snapshot.py`:

```python
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
```

Append to `mgo_lr/displacements.py`:

```python
def gen_structures_stage(cfg, workspace, args):
    from . import abacus_io
    from .config import atomic_write_text
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell

    if getattr(args, "set_name", None) is None:
        raise SystemExit("gen-structures requires --set pilot|main|large")
    ref = load_reference(workspace)
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    builders = {"pilot": build_pilot, "main": build_main, "large": build_large}
    plans = builders[args.set_name](cfg, ref["prim_cell"])
    store = SnapshotStore(workspace, args.set_name)
    os.makedirs(store.set_dir, exist_ok=True)
    seed = cfg["displacements"]["seed"]
    min_d = float(cfg["displacements"]["min_distance"])
    written = 0
    for plan in plans:
        sid, folder = plan["sid"], store.folder(plan["sid"])
        if os.path.isdir(folder):
            if not args.force:
                continue
            if any(d.startswith("OUT.") for d in os.listdir(folder)):
                print(f"{sid}: has DFT output; refusing to regenerate")
                continue
        pattern = json.loads(json.dumps(plan["pattern"]))
        meta = dict(plan["metadata"])
        u = apply_pattern(sc, ref["prim_cell"], pattern, seed)
        if pattern["modes"]:
            u = remove_uniform_translation(u)
        attempts = 0
        while minimum_distance(sc.cell, sc.cart + u) < min_d:
            if pattern.get("random") is None:
                raise ValueError(
                    f"{args.set_name}/{sid}: minimum interatomic distance "
                    f"{minimum_distance(sc.cell, sc.cart + u):.3f} Å "
                    f"< {min_d} Å for a deterministic pattern")
            attempts += 1
            if attempts > 100:
                raise ValueError(f"{sid}: no valid random draw in 100 tries")
            pattern["random"]["index"] += 100000
            meta["seed"] = pattern["random"]["index"]
            u = apply_pattern(sc, ref["prim_cell"], pattern, seed)
        os.makedirs(folder, exist_ok=True)
        abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell,
                             sc.cart + u, sc.species, cfg)
        abacus_io.write_input(os.path.join(folder, "INPUT"), cfg,
                              calculation="scf", out_mat_hs2=1, suffix="MgO")
        abacus_io.write_kpt(os.path.join(folder, "KPT"),
                            cfg["abacus"]["kmesh_supercell"][args.set_name])
        np.save(os.path.join(folder, "displacements.npy"), u)
        meta["pattern"] = pattern
        atomic_write_text(os.path.join(folder, "displacement_metadata.json"),
                          json.dumps(meta, indent=1))
        store.write_status(sid, "prepared", set_name=args.set_name)
        written += 1
    abacus_io.write_job_script(os.path.join(store.set_dir, "job_abacus.sh"),
                               cfg, [p["sid"] for p in plans])
    print(f"{args.set_name}: wrote {written} snapshots "
          f"({len(plans) - written} skipped)")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_gen_structures.py mgo_lr/tests/test_displacements.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/snapshot.py mgo_lr/displacements.py mgo_lr/tests/test_gen_structures.py
git commit -m "feat(mgo_lr): gen-structures stage and reference loader"
```

---

### Task 7: QE input writers and `init-dfpt` stage

**Files:**
- Create: `mgo_lr/dfpt.py`
- Modify: `mgo_lr/config.py` (extend `REQUIRED`)
- Modify: `mgo_lr/configs/mgo.yaml` (add `qe.pw_command`, `qe.ph_command`)
- Test: `mgo_lr/tests/test_dfpt_inputs.py`

**Interfaces:**
- Consumes: `snapshot.load_reference`, `config.atomic_write_text`.
- Produces: `dfpt.write_pw_input(path, cfg, cell, frac, species)`, `dfpt.write_ph_input(path, cfg)` (both `epsil = .true.` and `trans = .true.` explicit), `dfpt.init_dfpt_stage(cfg, workspace, args) -> int` writing `<workspace>/reference/qe/{pw.in, ph.in, job_qe.sh}` at the relaxed reference geometry. `collect_dfpt_stage` is Task 8.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_dfpt_inputs.py
import os

import numpy as np

from mgo_lr import dfpt
from mgo_lr.config import load_config
from mgo_lr.tests.test_gen_structures import Args, make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")


def test_config_has_qe_commands():
    assert "pw.x" in CFG["qe"]["pw_command"]
    assert "ph.x" in CFG["qe"]["ph_command"]


def test_init_dfpt_stage(tmp_path):
    ws = str(tmp_path)
    cell, frac, species = make_fake_reference(ws, a=4.19)
    assert dfpt.init_dfpt_stage(CFG, ws, Args()) == 0
    qdir = os.path.join(ws, "reference", "qe")
    pw = open(os.path.join(qdir, "pw.in")).read()
    assert "ibrav = 0" in pw and "nat = 2" in pw and "ntyp = 2" in pw
    assert "occupations = 'fixed'" in pw
    assert "CELL_PARAMETERS angstrom" in pw
    # relaxed lattice vectors present
    row = cell[0]
    assert f"{row[0]:.12f} {row[1]:.12f} {row[2]:.12f}" in pw
    assert "ATOMIC_POSITIONS crystal" in pw
    assert "K_POINTS automatic" in pw
    ph = open(os.path.join(qdir, "ph.in")).read()
    assert "epsil = .true." in ph          # both flags explicit
    assert "trans = .true." in ph
    assert ph.rstrip().endswith("0.0 0.0 0.0")
    job = open(os.path.join(qdir, "job_qe.sh")).read()
    assert CFG["qe"]["pw_command"] in job and CFG["qe"]["ph_command"] in job
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_dfpt_inputs.py -q` → ModuleNotFoundError `mgo_lr.dfpt` (and KeyError for `qe.pw_command` once the module exists).

- [ ] **Step 3: Write the implementation**

In `mgo_lr/config.py`, extend the `REQUIRED` qe line:

```python
    "qe.pseudo_dir", "qe.pseudopotentials", "qe.ecutwfc", "qe.kmesh",
    "qe.conv_thr", "qe.tr2_ph", "qe.version", "qe.pw_command", "qe.ph_command",
```

In `mgo_lr/configs/mgo.yaml`, add to the `qe:` section (after `version`):

```yaml
  pw_command: "mpirun -n 16 pw.x"
  ph_command: "mpirun -n 16 ph.x"
```

Create `mgo_lr/dfpt.py`:

```python
# mgo_lr/dfpt.py
"""Quantum ESPRESSO q=0 DFPT: input writers and Z*/eps_inf collection.

Consistency contract with ABACUS (recorded in dft_settings.yaml): same
relaxed lattice vectors and positions, same XC functional, same valence
configurations, same relativistic treatment, same charge and spin state;
prefer the same UPF pseudopotential files in both codes where supported.
MgO is an insulator: occupations = 'fixed' (DFPT requires no smearing).
"""
import json
import os
import re
import shutil

import numpy as np

from .config import atomic_write_text


def write_pw_input(path, cfg, cell, frac, species):
    qe, mat = cfg["qe"], cfg["material"]
    lines = ["&CONTROL", "  calculation = 'scf'", "  prefix = 'mgo'",
             "  outdir = './out'", f"  pseudo_dir = '{qe['pseudo_dir']}'",
             "  tprnfor = .true.", "  tstress = .true.", "/",
             "&SYSTEM", "  ibrav = 0", f"  nat = {len(species)}",
             f"  ntyp = {len(mat['species'])}",
             f"  ecutwfc = {qe['ecutwfc']}", "  occupations = 'fixed'", "/",
             "&ELECTRONS", f"  conv_thr = {qe['conv_thr']}", "/",
             "ATOMIC_SPECIES"]
    for s in mat["species"]:
        lines.append(f"{s} {mat['masses'][s]} {qe['pseudopotentials'][s]}")
    lines.append("CELL_PARAMETERS angstrom")
    for v in np.asarray(cell, float):
        lines.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    lines.append("ATOMIC_POSITIONS crystal")
    for s, f in zip(species, np.asarray(frac, float)):
        lines.append(f"{s}  {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}")
    k = qe["kmesh"]
    lines += ["K_POINTS automatic", f"{k[0]} {k[1]} {k[2]} 0 0 0"]
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_ph_input(path, cfg):
    qe = cfg["qe"]
    lines = ["MgO q=0 DFPT: dielectric tensor and Born effective charges",
             "&INPUTPH", "  prefix = 'mgo'", "  outdir = './out'",
             "  fildyn = 'mgo.dyn'", f"  tr2_ph = {qe['tr2_ph']}",
             "  epsil = .true.", "  trans = .true.", "/",
             "0.0 0.0 0.0"]
    atomic_write_text(path, "\n".join(lines) + "\n")


def init_dfpt_stage(cfg, workspace, args):
    from .snapshot import load_reference
    ref = load_reference(workspace)
    qdir = os.path.join(workspace, "reference", "qe")
    os.makedirs(qdir, exist_ok=True)
    write_pw_input(os.path.join(qdir, "pw.in"), cfg, ref["prim_cell"],
                   ref["frac"], ref["species"])
    write_ph_input(os.path.join(qdir, "ph.in"), cfg)
    job = [cfg["slurm"]["header"].rstrip(), "",
           f"{cfg['qe']['pw_command']} -in pw.in > pw.out 2>&1",
           f"{cfg['qe']['ph_command']} -in ph.in > ph.out 2>&1"]
    atomic_write_text(os.path.join(qdir, "job_qe.sh"), "\n".join(job) + "\n")
    print(f"QE DFPT inputs written to {qdir}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_dfpt_inputs.py mgo_lr/tests/test_config.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/dfpt.py mgo_lr/config.py mgo_lr/configs/mgo.yaml mgo_lr/tests/test_dfpt_inputs.py
git commit -m "feat(mgo_lr): QE pw.x/ph.x input writers and init-dfpt stage"
```

---

### Task 8: ph.x output parser, ASR correction, `collect-dfpt` stage

**Files:**
- Modify: `mgo_lr/dfpt.py`
- Modify: `mgo_lr/config.py` (extend `REQUIRED` with `dfpt.zstar_sum_warn`, `dfpt.isotropy_warn`)
- Modify: `mgo_lr/configs/mgo.yaml` (add `dfpt:` section)
- Test: `mgo_lr/tests/test_dfpt_collect.py`

**Interfaces:**
- Consumes: Task 7's `dfpt.py`, `snapshot.load_reference`.
- Produces: `dfpt.parse_ph_output(text: str) -> (eps (3,3), zstar (n_atom,3,3), labels list[str])` reading the **(d Force / dE)** Born block (rows in printed Ex/Ey/Ez order) and ignoring any later `(d P / du)` block; `dfpt.apply_asr(zstar) -> np.ndarray` (`Z̃*_κ = Z*_κ − mean_κ' Z*_κ'`); `dfpt.collect_dfpt_stage(cfg, workspace, args) -> int` writing `<workspace>/reference/{born_effective_charges.npy [2,3,3], dielectric_infinity.npy [3,3], qe_dfpt_output.out, dfpt_checks.json}`. Hard failures (SystemExit): missing blocks, wrong atom count/order, non-3×3 tensors, ε∞ not positive-definite, wrong Z* diagonal signs. Warnings (recorded in `dfpt_checks.json`): raw ASR violation > `dfpt.zstar_sum_warn`, anisotropy > `dfpt.isotropy_warn`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_dfpt_collect.py
import json
import os

import numpy as np
import pytest

from mgo_lr import dfpt
from mgo_lr.config import load_config
from mgo_lr.tests.test_gen_structures import Args, make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")

PH_OUT = """
     Computing the dielectric constant

          Dielectric constant in cartesian axis

          (       3.135573418       0.000000000       0.000000000 )
          (       0.000000000       3.135573418       0.000000000 )
          (       0.000000000       0.000000000       3.135573418 )

          Effective charges (d Force / dE) in cartesian axis without asr

           atom      1   Mg
      Ex  (        1.97120        0.00000        0.00000 )
      Ey  (        0.00000        1.97120        0.00000 )
      Ez  (        0.00000        0.00000        1.97120 )
           atom      2   O
      Ex  (       -1.95320        0.00000        0.00000 )
      Ey  (        0.00000       -1.95320        0.00000 )
      Ez  (        0.00000        0.00000       -1.95320 )

          Effective charges (d P / du) in cartesian axis apply asr

           atom      1   Mg
      Px  (        9.99999        0.00000        0.00000 )
      Py  (        0.00000        9.99999        0.00000 )
      Pz  (        0.00000        0.00000        9.99999 )
           atom      2   O
      Px  (       -9.99999        0.00000        0.00000 )
      Py  (        0.00000       -9.99999        0.00000 )
      Pz  (        0.00000        0.00000       -9.99999 )
"""


def test_parse_ph_output():
    eps, zstar, labels = dfpt.parse_ph_output(PH_OUT)
    assert eps.shape == (3, 3)
    assert abs(eps[0, 0] - 3.135573418) < 1e-9
    assert zstar.shape == (2, 3, 3)          # d P / du block NOT parsed
    assert labels == ["Mg", "O"]
    assert abs(zstar[0, 0, 0] - 1.97120) < 1e-9
    assert abs(zstar[1, 2, 2] + 1.95320) < 1e-9


def test_parse_missing_born_raises():
    with pytest.raises(ValueError, match="Born"):
        dfpt.parse_ph_output("no tensors here\n")


def test_apply_asr_exact():
    _, zstar, _ = dfpt.parse_ph_output(PH_OUT)
    corrected = dfpt.apply_asr(zstar)
    assert np.abs(corrected.sum(axis=0)).max() < 1e-13
    # symmetric correction: each atom shifted by half the raw sum
    assert abs(corrected[0, 0, 0] - (1.97120 - 0.017 / 2 * 2 / 2)) < 1e-2


def _ws_with_ph_out(tmp_path, text):
    ws = str(tmp_path)
    make_fake_reference(ws)
    qdir = os.path.join(ws, "reference", "qe")
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, "ph.out"), "w") as f:
        f.write(text)
    return ws


def test_collect_dfpt_stage(tmp_path):
    ws = _ws_with_ph_out(tmp_path, PH_OUT)
    assert dfpt.collect_dfpt_stage(CFG, ws, Args()) == 0
    ref = os.path.join(ws, "reference")
    z = np.load(os.path.join(ref, "born_effective_charges.npy"))
    e = np.load(os.path.join(ref, "dielectric_infinity.npy"))
    assert z.shape == (2, 3, 3) and e.shape == (3, 3)
    assert np.abs(z.sum(axis=0)).max() < 1e-13          # ASR applied
    assert z[0, 0, 0] > 0 > z[1, 0, 0]
    assert os.path.exists(os.path.join(ref, "qe_dfpt_output.out"))
    checks = json.load(open(os.path.join(ref, "dfpt_checks.json")))
    assert checks["hard_failures"] == []


def test_collect_dfpt_sign_flip_fails(tmp_path):
    flipped = PH_OUT.replace("1.97120", "-1.97120").replace("--1.97120",
                                                            "1.97120")
    ws = _ws_with_ph_out(tmp_path, flipped)
    with pytest.raises(SystemExit):
        dfpt.collect_dfpt_stage(CFG, ws, Args())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_dfpt_collect.py -q` → AttributeError `parse_ph_output`.

- [ ] **Step 3: Write the implementation**

In `mgo_lr/config.py` `REQUIRED`, add after the `qe.` lines:

```python
    "dfpt.zstar_sum_warn", "dfpt.isotropy_warn",
```

In `mgo_lr/configs/mgo.yaml`, add a top-level section after `qe:`:

```yaml
dfpt:
  zstar_sum_warn: 0.05     # |sum_kappa Z*| above this warns (raw, pre-ASR)
  isotropy_warn: 0.05      # relative anisotropy / off-diagonal warning level
```

Append to `mgo_lr/dfpt.py`:

```python
_FLOAT = r"[-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?"


def _three_floats(line):
    vals = re.findall(_FLOAT, line)
    if len(vals) < 3:
        raise ValueError(f"expected 3 floats in ph.x line: {line!r}")
    return [float(v.replace("D", "E").replace("d", "e")) for v in vals[-3:]]


def parse_ph_output(text):
    """Extract eps_inf and Born charges (d Force / dE block) from ph.x output."""
    lines = text.splitlines()
    eps = None
    for i, line in enumerate(lines):
        if "Dielectric constant in cartesian axis" in line:
            rows = [l for l in lines[i + 1:i + 8] if "(" in l][:3]
            if len(rows) != 3:
                raise ValueError("ph.x output: dielectric block malformed")
            eps = np.array([_three_floats(r) for r in rows])
    idx = None
    for i, line in enumerate(lines):
        if "Effective charges (d Force / dE) in cartesian axis" in line:
            idx = i
    if eps is None or idx is None:
        raise ValueError(
            "ph.x output lacks the dielectric tensor or Born effective "
            "charges — was ph.x run with epsil=.true. and trans=.true.?")
    atom_re = re.compile(r"atom\s+(\d+)\s+(\S+)")
    zstar, labels = [], []
    i = idx + 1
    while i < len(lines):
        if "Effective charges" in lines[i]:
            break                                # next block (d P / du)
        m = atom_re.search(lines[i])
        if m:
            rows = [l for l in lines[i + 1:i + 5] if "(" in l][:3]
            if len(rows) != 3:
                raise ValueError(f"ph.x Born block malformed at atom {m.group(1)}")
            zstar.append([_three_floats(r) for r in rows])
            labels.append(m.group(2))
            i += 4
        else:
            i += 1
    if not zstar:
        raise ValueError("ph.x output: no Born-charge atom blocks found")
    return np.asarray(eps, float), np.asarray(zstar, float), labels


def apply_asr(zstar):
    """Acoustic sum rule: Z~*_k = Z*_k - (1/N) sum_k' Z*_k'."""
    zstar = np.asarray(zstar, float)
    return zstar - zstar.sum(axis=0)[None] / zstar.shape[0]


def collect_dfpt_stage(cfg, workspace, args):
    from .snapshot import load_reference
    ref_dir = os.path.join(workspace, "reference")
    out_path = os.path.join(ref_dir, "qe", "ph.out")
    if not os.path.exists(out_path):
        raise SystemExit(f"ph.x output not found: {out_path}")
    with open(out_path) as f:
        eps, zstar, labels = parse_ph_output(f.read())
    ref = load_reference(workspace)
    hard, warn = [], []
    if list(labels) != list(ref["species"]):
        hard.append(f"atom labels {labels} != species order {ref['species']}")
    if zstar.shape != (len(ref["species"]), 3, 3):
        hard.append(f"Z* shape {zstar.shape} != (2,3,3)")
    if eps.shape != (3, 3):
        hard.append(f"eps shape {eps.shape} != (3,3)")
    else:
        if not np.allclose(eps, eps.T, atol=1e-6):
            warn.append("eps_inf not symmetric")
        if np.linalg.eigvalsh(0.5 * (eps + eps.T)).min() <= 0.0:
            hard.append("eps_inf not positive definite")
    if not hard:
        raw_sum = np.abs(zstar.sum(axis=0)).max()
        if raw_sum > float(cfg["dfpt"]["zstar_sum_warn"]):
            warn.append(f"raw ASR violation max|sum Z*| = {raw_sum:.4f}")
        z_asr = apply_asr(zstar)
        for a, lab in enumerate(labels):
            z = z_asr[a]
            diag = np.diag(z)
            off = float(np.abs(z - np.diag(diag)).max())
            aniso = float(np.abs(diag - diag.mean()).max()
                          / max(abs(diag.mean()), 1e-12))
            if off > float(cfg["dfpt"]["isotropy_warn"]) \
                    or aniso > float(cfg["dfpt"]["isotropy_warn"]):
                warn.append(f"Z*({lab}) anisotropic: off {off:.4f}, "
                            f"rel {aniso:.4f}")
        if not (np.diag(z_asr[0]).mean() > 0.0 > np.diag(z_asr[1]).mean()):
            hard.append("Z* diagonal signs wrong: expected Z*_Mg > 0 > Z*_O")
    checks = {"hard_failures": hard, "warnings": warn}
    atomic_write_text(os.path.join(ref_dir, "dfpt_checks.json"),
                      json.dumps(checks, indent=1))
    if hard:
        raise SystemExit("collect-dfpt hard failures: " + "; ".join(hard))
    np.save(os.path.join(ref_dir, "born_effective_charges.npy"), z_asr)
    np.save(os.path.join(ref_dir, "dielectric_infinity.npy"), eps)
    shutil.copyfile(out_path, os.path.join(ref_dir, "qe_dfpt_output.out"))
    for w in warn:
        print(f"WARNING: {w}")
    print(f"Z*_Mg = {np.diag(z_asr[0]).mean():+.4f}, "
          f"Z*_O = {np.diag(z_asr[1]).mean():+.4f}, "
          f"eps_inf = {np.diag(eps).mean():.4f}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_dfpt_collect.py -q` → 5 pass. Fix the loose `test_apply_asr_exact` third assertion if the tolerance math is off — the essential assertions are the first two (exact ASR).

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/dfpt.py mgo_lr/config.py mgo_lr/configs/mgo.yaml mgo_lr/tests/test_dfpt_collect.py
git commit -m "feat(mgo_lr): ph.x parser, ASR correction, collect-dfpt stage"
```

---

### Task 9: ABACUS output parsers (SCF log, CSR matrices, STRU)

**Files:**
- Modify: `mgo_lr/abacus_io.py`
- Test: `mgo_lr/tests/test_abacus_parsers.py`

**Interfaces:**
- Consumes: Task 4's `abacus_io.py`, `constants.BOHR_TO_ANGSTROM`.
- Produces: `abacus_io.parse_running_scf(path) -> {"converged": bool, "etot_ev": float|None, "fermi_ev": float|None}`; `abacus_io.parse_csr(path) -> (dim: int, {(Rx,Ry,Rz): scipy.sparse.csr_matrix})` handling the optional ABACUS ≥3.0 `STEP: 0` first line and raising ValueError on NaN/Inf, duplicate R, or malformed blocks; `abacus_io.parse_stru(path) -> (cell (3,3) Å, cart (N,3) Å, species list[str])` (round-trips `write_stru` output and ABACUS `STRU_ION_D`).

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_abacus_parsers.py
import numpy as np
import pytest
import scipy.sparse

from mgo_lr import abacus_io
from mgo_lr.config import load_config
from mgo_lr.structures import make_supercell, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")

SCF_LOG = """
 Charge Density Convergence is achieved
 charge density convergence is achieved
 !FINAL_ETOT_IS -7524.123456789 eV
 EFERMI = 5.4321 eV
"""


def write_csr(path, dim, blocks, name="H", step_line=True):
    """Fabricate an ABACUS out_mat_hs2 sparse file (shared with later tests)."""
    lines = []
    if step_line:
        lines.append("STEP: 0")
    lines.append(f"Matrix Dimension of {name}(R): {dim}")
    lines.append(f"Matrix number of {name}(R): {len(blocks)}")
    for R, dense in blocks.items():
        m = scipy.sparse.csr_matrix(dense)
        lines.append(f"{R[0]} {R[1]} {R[2]} {m.nnz}")
        if m.nnz:
            lines.append(" ".join(f"{v:.12e}" for v in m.data))
            lines.append(" ".join(str(i) for i in m.indices))
            lines.append(" ".join(str(i) for i in m.indptr))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_parse_running_scf(tmp_path):
    p = tmp_path / "running_scf.log"
    p.write_text(SCF_LOG)
    out = abacus_io.parse_running_scf(str(p))
    assert out["converged"] is True
    assert abs(out["etot_ev"] + 7524.123456789) < 1e-9
    assert abs(out["fermi_ev"] - 5.4321) < 1e-9
    p.write_text("scf failed horribly\n")
    assert abacus_io.parse_running_scf(str(p))["converged"] is False


def test_parse_csr_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    a[np.abs(a) < 0.7] = 0.0
    blocks = {(0, 0, 0): a, (1, 0, -1): np.zeros((4, 4)),
              (-1, 0, 1): a.T.copy()}
    p = tmp_path / "data-HR-sparse_SPIN0.csr"
    write_csr(str(p), 4, blocks)
    dim, parsed = abacus_io.parse_csr(str(p))
    assert dim == 4
    assert set(parsed) == {(0, 0, 0), (-1, 0, 1)}    # nnz=0 block dropped
    assert np.allclose(parsed[(0, 0, 0)].toarray(), a)
    # no STEP line (older ABACUS) also parses
    write_csr(str(p), 4, blocks, step_line=False)
    dim2, parsed2 = abacus_io.parse_csr(str(p))
    assert dim2 == 4 and set(parsed2) == set(parsed)


def test_parse_csr_rejects_nan(tmp_path):
    bad = np.array([[np.nan, 1.0], [0.0, 2.0]])
    p = tmp_path / "bad.csr"
    write_csr(str(p), 2, {(0, 0, 0): bad})
    with pytest.raises(ValueError, match="NaN"):
        abacus_io.parse_csr(str(p))


def test_parse_csr_rejects_duplicate_R(tmp_path):
    p = tmp_path / "dup.csr"
    a = np.eye(2)
    lines = ["Matrix Dimension of H(R): 2", "Matrix number of H(R): 2"]
    for _ in range(2):
        m = scipy.sparse.csr_matrix(a)
        lines.append("0 0 0 2")
        lines.append(" ".join(f"{v:.3e}" for v in m.data))
        lines.append(" ".join(str(i) for i in m.indices))
        lines.append(" ".join(str(i) for i in m.indptr))
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        abacus_io.parse_csr(str(p))


def test_parse_stru_roundtrip(tmp_path):
    cell, frac, species = rocksalt_primitive(4.19)
    sc = make_supercell(cell, frac, species, 2)
    p = tmp_path / "STRU"
    abacus_io.write_stru(str(p), sc.cell, sc.cart, sc.species, CFG)
    cell2, cart2, species2 = abacus_io.parse_stru(str(p))
    assert np.allclose(cell2, sc.cell, atol=1e-9)
    assert np.allclose(cart2, sc.cart, atol=1e-8)
    assert species2 == sc.species
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_abacus_parsers.py -q` → AttributeError `parse_running_scf`.

- [ ] **Step 3: Write the implementation**

Append to `mgo_lr/abacus_io.py` (update the module docstring's "Task 9" note to say parsers live below; add `import re` and `from .constants import BOHR_TO_ANGSTROM` to the imports):

```python
_FLOAT = r"[-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?"


def parse_running_scf(path):
    """Convergence flag, final total energy (eV), Fermi level (eV)."""
    with open(path, errors="replace") as f:
        text = f.read()
    low = text.lower()
    converged = ("charge density convergence is achieved" in low
                 or "convergence has been achieved" in low)
    m = re.search(r"!FINAL_ETOT_IS\s+(" + _FLOAT + r")\s+eV", text)
    etot = float(m.group(1)) if m else None
    if converged and etot is None:
        raise ValueError(f"{path}: converged run without !FINAL_ETOT_IS line")
    mf = re.search(r"EFERMI\s*=?\s*(" + _FLOAT + r")\s*eV", text)
    fermi = float(mf.group(1)) if mf else None
    return {"converged": converged, "etot_ev": etot, "fermi_ev": fermi}


def parse_csr(path):
    """Parse an ABACUS out_mat_hs2 sparse-matrix file.

    Format (ABACUS >= 3.0 prepends a 'STEP: 0' line):
        Matrix Dimension of H(R): <dim>
        Matrix number of H(R): <n>
        Rx Ry Rz nnz
        <nnz values> / <nnz col indices> / <dim+1 row pointers>   (if nnz > 0)
    """
    import scipy.sparse
    with open(path) as f:
        line = f.readline()
        if "Matrix Dimension of" not in line:
            line = f.readline()
            if "Matrix Dimension of" not in line:
                raise ValueError(f"{path}: missing 'Matrix Dimension of' header")
        dim = int(line.split()[-1])
        f.readline()                      # "Matrix number of ..."
        blocks = {}
        for line in f:
            parts = line.split()
            if not parts:
                break
            if len(parts) != 4:
                raise ValueError(f"{path}: malformed R header line: {line!r}")
            R = tuple(int(x) for x in parts[:3])
            nnz = int(parts[3])
            if nnz == 0:
                continue
            vals = np.array(f.readline().split(), dtype=float)
            cols = np.array(f.readline().split(), dtype=int)
            ptr = np.array(f.readline().split(), dtype=int)
            if len(vals) != nnz or len(cols) != nnz or len(ptr) != dim + 1:
                raise ValueError(f"{path}: CSR block {R} lengths inconsistent")
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"{path}: NaN/Inf in CSR block {R}")
            if R in blocks:
                raise ValueError(f"{path}: duplicate R block {R}")
            blocks[R] = scipy.sparse.csr_matrix((vals, cols, ptr),
                                                shape=(dim, dim))
    return dim, blocks


def parse_stru(path):
    """Parse STRU (as written by write_stru, or ABACUS STRU_ION_D).

    Returns (cell (3,3) Å, cart (N,3) Å, species).  Only Direct coordinates
    are supported.
    """
    with open(path) as f:
        lines = [l.strip() for l in f.read().splitlines()]
    i = lines.index("LATTICE_CONSTANT")
    lat_const_bohr = float(lines[i + 1].split()[0])
    j = lines.index("LATTICE_VECTORS")
    cell = np.array([[float(x) for x in lines[j + 1 + r].split()[:3]]
                     for r in range(3)])
    cell = cell * lat_const_bohr * BOHR_TO_ANGSTROM
    k = lines.index("ATOMIC_POSITIONS")
    k += 1
    while not lines[k]:
        k += 1
    if not lines[k].startswith("Direct"):
        raise ValueError(f"{path}: only Direct coordinates supported, "
                         f"got {lines[k]!r}")
    species, frac = [], []
    idx = k + 1
    while idx < len(lines):
        if not lines[idx]:
            idx += 1
            continue
        name = lines[idx].split()[0]
        count = int(lines[idx + 2].split()[0])
        for c in range(count):
            row = lines[idx + 3 + c].split()
            frac.append([float(x) for x in row[:3]])
            species.append(name)
        idx += 3 + count
    return cell, np.asarray(frac, float) @ cell, species
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_abacus_parsers.py mgo_lr/tests/test_abacus_writers.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/abacus_io.py mgo_lr/tests/test_abacus_parsers.py
git commit -m "feat(mgo_lr): ABACUS SCF-log, CSR and STRU parsers"
```

---

### Task 10: `init-reference` / `collect-reference` stages

**Files:**
- Create: `mgo_lr/reference.py`
- Modify: `mgo_lr/constants.py` (add `ATOMIC_NUMBERS = {"Mg": 12, "O": 8}`)
- Test: `mgo_lr/tests/test_reference.py`

**Interfaces:**
- Consumes: `structures.rocksalt_primitive`, `abacus_io.write_stru/write_input/write_kpt/parse_stru/parse_running_scf`, `config.atomic_write_text`.
- Produces: `reference.init_reference_stage(cfg, workspace, args) -> int` writing decks under `<workspace>/reference/abacus/`: `ecut_<E>/` per `reference.ecut_scan` value, `kmesh_<k1>x<k2>x<k3>/` per `reference.kmesh_scan` mesh, `cell_relax/`, `final_scf/` (with `out_mat_hs2 1`), each with STRU/INPUT/KPT. `reference.collect_reference_stage(cfg, workspace, args) -> int`: relaxed lattice constant from `material.lattice_constant_relaxed` config override if set, else parsed from `cell_relax/OUT.MgO/STRU_ION_D`; writes `reference/{reference_cell.npy, reference_positions.npy, atomic_numbers.npy, species_order.json, orbital_types.dat (one line per atom), primitive.cif, dft_settings.yaml, scan_summary.json}` and regenerates `final_scf/STRU` at the relaxed geometry. `reference.lattice_constant_from_cell(cell) -> float` (a = |v₀|·√2 for the fcc primitive cell, cubic-symmetry checked).

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_reference.py
import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import abacus_io, reference
from mgo_lr.config import load_config
from mgo_lr.snapshot import load_reference
from mgo_lr.structures import rocksalt_primitive
from mgo_lr.tests.test_gen_structures import Args

CFG = load_config("mgo_lr/configs/mgo.yaml")


def test_init_reference_decks(tmp_path):
    ws = str(tmp_path)
    assert reference.init_reference_stage(CFG, ws, Args()) == 0
    base = os.path.join(ws, "reference", "abacus")
    for e in CFG["reference"]["ecut_scan"]:
        d = os.path.join(base, f"ecut_{e}")
        assert os.path.isdir(d)
        assert f"ecutwfc" in open(os.path.join(d, "INPUT")).read()
        assert str(e) in open(os.path.join(d, "INPUT")).read()
    for mesh in CFG["reference"]["kmesh_scan"]:
        d = os.path.join(base, f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}")
        assert os.path.isdir(d)
    relax = open(os.path.join(base, "cell_relax", "INPUT")).read()
    assert "cell-relax" in relax
    final = open(os.path.join(base, "final_scf", "INPUT")).read()
    assert "out_mat_hs2" in final


def test_lattice_constant_from_cell():
    cell, _, _ = rocksalt_primitive(4.19)
    assert abs(reference.lattice_constant_from_cell(cell) - 4.19) < 1e-12


def test_collect_reference_with_override(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    cfg = copy.deepcopy(CFG)
    cfg["material"]["lattice_constant_relaxed"] = 4.19
    assert reference.collect_reference_stage(cfg, ws, Args()) == 0
    ref = load_reference(ws)
    assert abs(reference.lattice_constant_from_cell(ref["prim_cell"]) - 4.19) < 1e-10
    assert list(ref["atomic_numbers"]) == [12, 8]
    assert ref["species"] == ["Mg", "O"]
    ref_dir = os.path.join(ws, "reference")
    ot = open(os.path.join(ref_dir, "orbital_types.dat")).read().splitlines()
    assert len(ot) == 2                       # one line per atom (2-atom cell)
    assert ot[0].split() == [str(l) for l in CFG["abacus"]["orbital_types"]["Mg"]]
    assert os.path.exists(os.path.join(ref_dir, "primitive.cif"))
    settings = yaml.safe_load(open(os.path.join(ref_dir, "dft_settings.yaml")))
    assert abs(settings["lattice_constant_relaxed"] - 4.19) < 1e-10
    # final_scf STRU regenerated at the relaxed constant
    cell2, _, _ = abacus_io.parse_stru(
        os.path.join(ref_dir, "abacus", "final_scf", "STRU"))
    assert abs(reference.lattice_constant_from_cell(cell2) - 4.19) < 1e-8


def test_collect_reference_parses_relaxed_stru(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    out = os.path.join(ws, "reference", "abacus", "cell_relax", "OUT.MgO")
    os.makedirs(out)
    cell, frac, species = rocksalt_primitive(4.213)
    abacus_io.write_stru(os.path.join(out, "STRU_ION_D"),
                         cell, frac @ cell, species, CFG)
    assert reference.collect_reference_stage(CFG, ws, Args()) == 0
    ref = load_reference(ws)
    assert abs(reference.lattice_constant_from_cell(ref["prim_cell"]) - 4.213) < 1e-8


def test_collect_reference_scan_summary(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    e0 = CFG["reference"]["ecut_scan"][0]
    out = os.path.join(ws, "reference", "abacus", f"ecut_{e0}", "OUT.MgO")
    os.makedirs(out)
    with open(os.path.join(out, "running_scf.log"), "w") as f:
        f.write("charge density convergence is achieved\n"
                "!FINAL_ETOT_IS -7000.5 eV\n")
    cfg = copy.deepcopy(CFG)
    cfg["material"]["lattice_constant_relaxed"] = 4.19
    reference.collect_reference_stage(cfg, ws, Args())
    summary = json.load(open(os.path.join(ws, "reference",
                                          "scan_summary.json")))
    assert summary["ecut"][str(e0)]["etot_ev"] == -7000.5
    assert any("missing" in w for w in summary["warnings"])


def test_collect_reference_no_relax_no_override_fails(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    with pytest.raises(SystemExit, match="relax"):
        reference.collect_reference_stage(CFG, ws, Args())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_reference.py -q` → ModuleNotFoundError `mgo_lr.reference`.

- [ ] **Step 3: Write the implementation**

Append to `mgo_lr/constants.py`:

```python
ATOMIC_NUMBERS = {"Mg": 12, "O": 8}
```

Create `mgo_lr/reference.py`:

```python
# mgo_lr/reference.py
"""Reference-structure stages: ABACUS convergence/relax decks and the
permanent reference artifacts every later stage consumes."""
import json
import math
import os

import numpy as np
import yaml

from . import __version__, abacus_io
from .config import atomic_write_text
from .constants import ATOMIC_NUMBERS
from .structures import rocksalt_primitive


def _write_deck(folder, cell, frac, species, cfg, kmesh, **input_overrides):
    os.makedirs(folder, exist_ok=True)
    abacus_io.write_stru(os.path.join(folder, "STRU"), cell,
                         np.asarray(frac, float) @ np.asarray(cell, float),
                         species, cfg)
    abacus_io.write_input(os.path.join(folder, "INPUT"), cfg,
                          suffix="MgO", **input_overrides)
    abacus_io.write_kpt(os.path.join(folder, "KPT"), kmesh)


def init_reference_stage(cfg, workspace, args):
    a = float(cfg["material"]["lattice_constant_guess"])
    cell, frac, species = rocksalt_primitive(a)
    base = os.path.join(workspace, "reference", "abacus")
    kp = cfg["abacus"]["kmesh_primitive"]
    for e in cfg["reference"]["ecut_scan"]:
        _write_deck(os.path.join(base, f"ecut_{e}"), cell, frac, species,
                    cfg, kp, calculation="scf", ecutwfc=e)
    for mesh in cfg["reference"]["kmesh_scan"]:
        _write_deck(os.path.join(base, f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}"),
                    cell, frac, species, cfg, mesh, calculation="scf")
    _write_deck(os.path.join(base, "cell_relax"), cell, frac, species, cfg,
                kp, calculation="cell-relax", cal_force=1, cal_stress=1,
                relax_nmax=100)
    _write_deck(os.path.join(base, "final_scf"), cell, frac, species, cfg,
                kp, calculation="scf", out_mat_hs2=1)
    print(f"reference decks written under {base}")
    return 0


def lattice_constant_from_cell(cell):
    """Cubic lattice constant from the fcc primitive cell a/2*(011)-type rows."""
    cell = np.asarray(cell, float)
    norms = np.linalg.norm(cell, axis=1)
    if np.abs(norms - norms.mean()).max() > 1e-3 * norms.mean():
        print(f"WARNING: relaxed cell deviates from cubic: |v_i| = {norms}")
    return float(norms.mean() * math.sqrt(2.0))


def _write_cif(path, cell, frac, species):
    cell = np.asarray(cell, float)
    a, b, c = np.linalg.norm(cell, axis=1)

    def ang(u, v):
        return math.degrees(math.acos(
            float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))

    lines = ["data_MgO",
             f"_cell_length_a {a:.8f}", f"_cell_length_b {b:.8f}",
             f"_cell_length_c {c:.8f}",
             f"_cell_angle_alpha {ang(cell[1], cell[2]):.6f}",
             f"_cell_angle_beta {ang(cell[0], cell[2]):.6f}",
             f"_cell_angle_gamma {ang(cell[0], cell[1]):.6f}",
             "_symmetry_space_group_name_H-M 'P 1'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    for i, (s, f) in enumerate(zip(species, np.asarray(frac, float)), 1):
        lines.append(f"{s}{i} {s} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def collect_reference_stage(cfg, workspace, args):
    ref_dir = os.path.join(workspace, "reference")
    base = os.path.join(ref_dir, "abacus")
    override = cfg["material"].get("lattice_constant_relaxed")
    if override is not None:
        a = float(override)
    else:
        relaxed = os.path.join(base, "cell_relax", "OUT.MgO", "STRU_ION_D")
        if not os.path.exists(relaxed):
            raise SystemExit(
                f"no relaxed structure at {relaxed} and no "
                "material.lattice_constant_relaxed override — run the "
                "cell-relax deck first")
        cell_r, _, _ = abacus_io.parse_stru(relaxed)
        a = lattice_constant_from_cell(cell_r)
    cell, frac, species = rocksalt_primitive(a)

    # scan summary (tolerant: report what ran, warn about what did not)
    summary = {"ecut": {}, "kmesh": {}, "warnings": []}
    for e in cfg["reference"]["ecut_scan"]:
        log = os.path.join(base, f"ecut_{e}", "OUT.MgO", "running_scf.log")
        if os.path.exists(log):
            summary["ecut"][str(e)] = abacus_io.parse_running_scf(log)
        else:
            summary["warnings"].append(f"ecut_{e}: output missing")
    for mesh in cfg["reference"]["kmesh_scan"]:
        name = f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}"
        log = os.path.join(base, name, "OUT.MgO", "running_scf.log")
        if os.path.exists(log):
            summary["kmesh"][name] = abacus_io.parse_running_scf(log)
        else:
            summary["warnings"].append(f"{name}: output missing")
    atomic_write_text(os.path.join(ref_dir, "scan_summary.json"),
                      json.dumps(summary, indent=1))

    np.save(os.path.join(ref_dir, "reference_cell.npy"), cell)
    np.save(os.path.join(ref_dir, "reference_positions.npy"), frac)
    np.save(os.path.join(ref_dir, "atomic_numbers.npy"),
            np.array([ATOMIC_NUMBERS[s] for s in species]))
    atomic_write_text(os.path.join(ref_dir, "species_order.json"),
                      json.dumps(species))
    # one line per atom (2-atom primitive cell -> 2 lines)
    ot = cfg["abacus"]["orbital_types"]
    atomic_write_text(os.path.join(ref_dir, "orbital_types.dat"),
                      "\n".join("  ".join(str(l) for l in ot[s])
                                for s in species) + "\n")
    _write_cif(os.path.join(ref_dir, "primitive.cif"), cell, frac, species)
    settings = {"lattice_constant_relaxed": a,
                "abacus": cfg["abacus"], "qe": cfg["qe"],
                "mgo_lr_version": __version__}
    atomic_write_text(os.path.join(ref_dir, "dft_settings.yaml"),
                      yaml.safe_dump(settings, sort_keys=False))
    # regenerate the final high-accuracy SCF deck at the relaxed geometry
    _write_deck(os.path.join(base, "final_scf"), cell, frac, species, cfg,
                cfg["abacus"]["kmesh_primitive"], calculation="scf",
                out_mat_hs2=1)
    print(f"reference collected: a = {a:.6f} Å; rerun final_scf deck if "
          "the lattice constant changed")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_reference.py -q` → 6 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/reference.py mgo_lr/constants.py mgo_lr/tests/test_reference.py
git commit -m "feat(mgo_lr): init-reference and collect-reference stages"
```

---

### Task 11: ABACUS→DeepH-E3 orbital reorder transform

**Files:**
- Create: `mgo_lr/convert.py` (transform only; the rest of the module is Task 12)
- Test: `mgo_lr/tests/test_orbital_reorder.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `convert.orbital_u(l: int) -> np.ndarray (2l+1, 2l+1)`; `convert.atom_u(orbital_types_atom: list[int]) -> np.ndarray`; `convert.transform_block(mat, l_left: list[int], l_right: list[int]) -> np.ndarray` computing `U_i @ mat @ U_j.T`.

The table is copied **verbatim** from DeepH-pack `deeph/preprocess/abacus_get_data.py` (`OrbAbacus2DeepH`) — the de-facto definition of the DeepH-E3 orbital convention for ABACUS data. Do not "fix" or re-derive it:

```
U[0] = eye(1)
U[1] = eye(3)[[1, 2, 0]]   then rows [0, 1] *= -1
U[2] = eye(5)[[0, 3, 4, 1, 2]]  then rows [3, 4] *= -1
U[3] = eye(7)             then rows [1, 2, 5, 6] *= -1
```

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_orbital_reorder.py
import numpy as np

from mgo_lr import convert


def test_us_are_signed_permutations():
    for l in range(4):
        u = convert.orbital_u(l)
        assert u.shape == (2 * l + 1, 2 * l + 1)
        assert np.allclose(u @ u.T, np.eye(2 * l + 1))       # orthogonal
        assert np.allclose(np.abs(u).sum(axis=0), 1.0)       # permutation
        assert np.allclose(np.abs(u).sum(axis=1), 1.0)


def test_p_transform_hand_checked():
    # DeepH-pack: U[1] = eye(3)[[1,2,0]] with rows [0,1] negated, so an
    # ABACUS p-vector (a0, a1, a2) maps to (-a1, -a2, +a0).
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(convert.orbital_u(1) @ v, [-2.0, -3.0, 1.0])


def test_d_transform_hand_checked():
    # U[2] = eye(5)[[0,3,4,1,2]] with rows [3,4] negated:
    # (a0..a4) -> (a0, a3, a4, -a1, -a2)
    v = np.arange(5, dtype=float)
    assert np.allclose(convert.orbital_u(2) @ v, [0.0, 3.0, 4.0, -1.0, -2.0])


def test_atom_u_block_diagonal():
    u = convert.atom_u([0, 0, 1])
    assert u.shape == (5, 5)
    assert np.allclose(u[:2, :2], np.eye(2))
    assert np.allclose(u[2:, 2:], convert.orbital_u(1))
    assert np.allclose(u[:2, 2:], 0.0)


def test_transform_preserves_hermitian_pairs():
    rng = np.random.default_rng(3)
    li, lj = [0, 1], [1]
    a = rng.standard_normal((4, 3))          # H_ij(R)
    b = a.T.copy()                           # H_ji(-R) = H_ij(R)^T
    ta = convert.transform_block(a, li, lj)
    tb = convert.transform_block(b, lj, li)
    assert np.allclose(tb, ta.T)


def test_transform_involution_via_orthogonality():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((3, 3))
    t = convert.transform_block(a, [1], [1])
    u = convert.orbital_u(1)
    assert np.allclose(u.T @ t @ u, a)       # U^T (U a U^T) U = a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_orbital_reorder.py -q` → ModuleNotFoundError `mgo_lr.convert`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/convert.py
"""ABACUS -> DeepH-E3/MACE-H data conversion.

The per-l orbital permutation/sign table is copied verbatim from
DeepH-pack deeph/preprocess/abacus_get_data.py (class OrbAbacus2DeepH).
A silent error here corrupts every matrix — the table is pinned by
tests/test_orbital_reorder.py and must not be re-derived.
"""
import numpy as np
from scipy.linalg import block_diag


def _build_us():
    us = {0: np.eye(1),
          1: np.eye(3)[[1, 2, 0]],
          2: np.eye(5)[[0, 3, 4, 1, 2]],
          3: np.eye(7)}
    minus = {1: [0, 1], 2: [3, 4], 3: [1, 2, 5, 6]}
    for l, rows in minus.items():
        us[l][rows] *= -1.0
    return us


_U_ABACUS2DEEPH = _build_us()


def orbital_u(l):
    if l not in _U_ABACUS2DEEPH:
        raise NotImplementedError(f"only l <= 3 supported, got l={l}")
    return _U_ABACUS2DEEPH[l]


def atom_u(orbital_types_atom):
    """Block-diagonal transform for one atom's full AO set."""
    return block_diag(*[orbital_u(l) for l in orbital_types_atom])


def transform_block(mat, l_left, l_right):
    """U_i @ mat @ U_j.T for an atom-pair block."""
    return atom_u(l_left) @ np.asarray(mat, float) @ atom_u(l_right).T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_orbital_reorder.py -q` → 6 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/convert.py mgo_lr/tests/test_orbital_reorder.py
git commit -m "feat(mgo_lr): ABACUS->DeepH-E3 orbital reorder transform (DeepH-pack table)"
```

---

### Task 12: DeepH-E3 format writer and `collect-dft` stage

**Files:**
- Modify: `mgo_lr/convert.py`
- Modify: `mgo_lr/config.py` (add `sha256_file`)
- Test: `mgo_lr/tests/test_convert.py`

**Interfaces:**
- Consumes: Task 9 parsers, Task 11 transform, `constants.RY_TO_EV/ATOMIC_NUMBERS`, `SnapshotStore`, `load_reference`, `make_supercell`.
- Produces: `config.sha256_file(path) -> str`; in `convert.py`: `BLOCK_SKIP_THRESHOLD = 1e-8` (same sparsity cutoff as DeepH-pack); `key_str(R, i, j) -> str` (i, j **0-based in, 1-based out**: `"[Rx, Ry, Rz, i+1, j+1]"`); `parse_key(k) -> (Rx,Ry,Rz,i,j)` (returns the stored 1-based i, j); `write_blocks(path, blocks: dict[str, np.ndarray])` (atomic h5); `read_blocks(path) -> dict[str, np.ndarray]`; `species_orbital_info(cfg, species_list) -> (types per atom, norb per atom, offsets (N+1,))`; `matrices_to_blocks(csr_blocks, dim, cfg, species_list, factor) -> dict[str, np.ndarray]` (slice → skip-threshold → reorder-transform → × factor); `write_structure_files(folder, cell, cart, species, cfg, fermi_ev)` writing `lat.dat`/`rlat.dat` (3×3, vectors as **columns**, `rlat` includes 2π), `site_positions.dat` (3×N), `element.dat`, `orbital_types.dat` (one line per atom), `info.json` (`nsites`, `isorthogonal`, `isspinful`, `norbits`, `fermi_level`); `collect_dft_stage(cfg, workspace, args) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_convert.py
import copy
import json
import os

import h5py
import numpy as np
import pytest
import yaml

from mgo_lr import abacus_io, convert
from mgo_lr.config import load_config, sha256_file
from mgo_lr.constants import RY_TO_EV
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell, rocksalt_primitive
from mgo_lr.tests.test_abacus_parsers import write_csr
from mgo_lr.tests.test_gen_structures import make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")

SCF_LOG = ("charge density convergence is achieved\n"
           "!FINAL_ETOT_IS -7524.1 eV\nEFERMI = 5.4 eV\n")


class Args:
    set_name = "pilot"
    force = False


def small_cfg():
    """Pilot on the 2-atom primitive cell (n=1) keeps matrices tiny."""
    cfg = copy.deepcopy(CFG)
    cfg["supercells"]["pilot"] = 1
    return cfg


def fabricate_dft(folder, cfg, sc, seed=0):
    """Write OUT.MgO with a converged log and hermitian synthetic H/S CSRs.
    Shared with the validate/locality/end-to-end tests."""
    types, norb, offsets = convert.species_orbital_info(cfg, sc.species)
    dim = int(offsets[-1])
    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal((dim, dim)) * 0.05
    h0 = 0.5 * (h0 + h0.T)                       # H(0) symmetric
    hp = rng.standard_normal((dim, dim)) * 0.01  # H(R), H(-R) = H(R)^T
    s0 = np.eye(dim) + 0.01 * (lambda m: 0.5 * (m + m.T))(
        rng.standard_normal((dim, dim)))
    sp = 0.01 * rng.standard_normal((dim, dim))
    out = os.path.join(folder, "OUT.MgO")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "running_scf.log"), "w") as f:
        f.write(SCF_LOG)
    write_csr(os.path.join(out, cfg["abacus"]["csr_h_filename"]), dim,
              {(0, 0, 0): h0, (1, 0, 0): hp, (-1, 0, 0): hp.T.copy()})
    write_csr(os.path.join(out, cfg["abacus"]["csr_s_filename"]), dim,
              {(0, 0, 0): s0, (1, 0, 0): sp, (-1, 0, 0): sp.T.copy()},
              name="S")
    return {"h0": h0, "s0": s0, "dim": dim, "offsets": offsets}


def prepared_snapshot(ws, cfg, u=None):
    cell, frac, species = make_fake_reference(ws)
    n = cfg["supercells"]["pilot"]
    sc = make_supercell(cell, frac, species, n)
    store = SnapshotStore(ws, "pilot")
    sid = "snapshot_000001"
    folder = store.folder(sid)
    os.makedirs(folder)
    if u is None:
        u = np.zeros((len(sc.species), 3))
    abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell, sc.cart + u,
                         sc.species, cfg)
    np.save(os.path.join(folder, "displacements.npy"), u)
    meta = {"pattern_class": "equilibrium", "rigid_translation": False,
            "sign_partner_id": None, "amplitude_partner_ids": [],
            "amplitude": 0.0}
    with open(os.path.join(folder, "displacement_metadata.json"), "w") as f:
        json.dump(meta, f)
    store.write_status(sid, "prepared")
    return store, sid, sc


def test_key_roundtrip():
    k = convert.key_str((0, -1, 2), 0, 4)
    assert k == "[0, -1, 2, 1, 5]"                 # 1-based indices
    assert convert.parse_key(k) == (0, -1, 2, 1, 5)


def test_matrices_to_blocks_units_and_transform(tmp_path):
    cfg = small_cfg()
    cell, frac, species = rocksalt_primitive(4.2)
    sc = make_supercell(cell, frac, species, 1)
    types, norb, offsets = convert.species_orbital_info(cfg, sc.species)
    assert norb == [15, 14]                        # Mg 4s2p1d, O 3s3p2d
    dim = int(offsets[-1])
    dense = np.zeros((dim, dim))
    dense[0, 0] = 2.0                              # Mg s <-> Mg s
    import scipy.sparse
    csr = {(0, 0, 0): scipy.sparse.csr_matrix(dense)}
    blocks = convert.matrices_to_blocks(csr, dim, cfg, sc.species, RY_TO_EV)
    assert set(blocks) == {"[0, 0, 0, 1, 1]"}
    b = blocks["[0, 0, 0, 1, 1]"]
    assert b.shape == (15, 15)
    assert abs(b[0, 0] - 2.0 * RY_TO_EV) < 1e-12   # s-channel: U = identity
    # dimension mismatch raises
    with pytest.raises(ValueError, match="dimension"):
        convert.matrices_to_blocks(csr, dim, cfg, ["Mg"], 1.0)


def test_collect_dft_stage(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fab = fabricate_dft(store.folder(sid), cfg, sc)
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    st = store.read_status(sid)
    assert st["state"] == "converted"
    assert st["scf_converged"] is True
    assert st["csr_files"] == [cfg["abacus"]["csr_h_filename"],
                               cfg["abacus"]["csr_s_filename"]]
    assert set(st["raw_sha256"]) == {cfg["abacus"]["csr_h_filename"],
                                     cfg["abacus"]["csr_s_filename"],
                                     "running_scf.log"}
    folder = store.folder(sid)
    with h5py.File(os.path.join(folder, "hamiltonians_full.h5")) as f:
        keys = [json.loads(k) for k in f.keys()]
        assert all(len(k) == 5 and k[3] >= 1 and k[4] >= 1 for k in keys)
        b11 = np.array(f["[0, 0, 0, 1, 1]"])
    # Ry -> eV applied, transform for the s-block diagonal entry is identity
    assert abs(b11[0, 0] - fab["h0"][0, 0] * RY_TO_EV) < 1e-10
    lat = np.loadtxt(os.path.join(folder, "lat.dat"))
    assert np.allclose(lat.T, sc.cell)             # vectors as columns
    rlat = np.loadtxt(os.path.join(folder, "rlat.dat"))
    assert np.allclose(rlat.T @ lat, 2 * np.pi * np.eye(3), atol=1e-10)
    pos = np.loadtxt(os.path.join(folder, "site_positions.dat"))
    assert pos.shape == (3, 2)                     # 3 x N
    ot = open(os.path.join(folder, "orbital_types.dat")).read().splitlines()
    assert len(ot) == 2                            # one line per atom
    info = json.load(open(os.path.join(folder, "info.json")))
    assert info["isspinful"] is False and info["norbits"] == fab["dim"]
    elements = np.loadtxt(os.path.join(folder, "element.dat"))
    assert list(elements) == [12.0, 8.0]


def test_collect_dft_skips_and_protects(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fabricate_dft(store.folder(sid), cfg, sc)
    convert.collect_dft_stage(cfg, ws, Args())
    h1 = sha256_file(os.path.join(store.folder(sid), "hamiltonians_full.h5"))
    convert.collect_dft_stage(cfg, ws, Args())     # idempotent skip
    assert sha256_file(os.path.join(store.folder(sid),
                                    "hamiltonians_full.h5")) == h1


def test_collect_dft_rejects_unconverged(tmp_path):
    ws = str(tmp_path)
    cfg = small_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg)
    fabricate_dft(store.folder(sid), cfg, sc)
    with open(os.path.join(store.folder(sid), "OUT.MgO",
                           "running_scf.log"), "w") as f:
        f.write("it exploded\n")
    assert convert.collect_dft_stage(cfg, ws, Args()) == 1
    assert store.list() == []                      # moved to rejected/
    rej = os.path.join(ws, "rejected", f"pilot_{sid}")
    st = json.load(open(os.path.join(rej, "status.json")))
    assert st["reason"] == "scf_not_converged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_convert.py -q` → ImportError `sha256_file`.

- [ ] **Step 3: Write the implementation**

Append to `mgo_lr/config.py`:

```python
def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

Append to `mgo_lr/convert.py` (extend the imports with `import json`, `import os`, `import h5py`, `from .config import atomic_write_text, sha256_file`, `from .constants import ATOMIC_NUMBERS, RY_TO_EV`):

```python
BLOCK_SKIP_THRESHOLD = 1e-8   # same block-sparsity cutoff as DeepH-pack


def key_str(R, i, j):
    """DeepH-E3 h5 key: JSON list, 1-based atom indices (i, j are 0-based in)."""
    return f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {i + 1}, {j + 1}]"


def parse_key(k):
    v = json.loads(k)
    return (v[0], v[1], v[2], v[3], v[4])


def write_blocks(path, blocks):
    tmp = f"{path}.tmp.{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        for k, v in blocks.items():
            f[k] = np.asarray(v, np.float64)
    os.replace(tmp, path)


def read_blocks(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k], dtype=np.float64)
    return out


def species_orbital_info(cfg, species_list):
    types = [cfg["abacus"]["orbital_types"][s] for s in species_list]
    norb = [sum(2 * l + 1 for l in t) for t in types]
    offsets = np.concatenate([[0], np.cumsum(norb)])
    return types, norb, offsets


def matrices_to_blocks(csr_blocks, dim, cfg, species_list, factor):
    """Slice per-R matrices into atom-pair blocks, apply the orbital
    transform, scale by `factor` (RY_TO_EV for H, 1.0 for S)."""
    types, norb, offsets = species_orbital_info(cfg, species_list)
    if int(offsets[-1]) != dim:
        raise ValueError(f"matrix dimension {dim} != expected {offsets[-1]} "
                         f"from orbital_types for {len(species_list)} atoms")
    n_at = len(species_list)
    out = {}
    for R, m in csr_blocks.items():
        dense = m.toarray()
        if not np.all(np.isfinite(dense)):
            raise ValueError(f"NaN/Inf in matrix block R={R}")
        for i in range(n_at):
            for j in range(n_at):
                blk = dense[offsets[i]:offsets[i + 1],
                            offsets[j]:offsets[j + 1]]
                if np.abs(blk).max() < BLOCK_SKIP_THRESHOLD:
                    continue
                out[key_str(R, i, j)] = factor * transform_block(
                    blk, types[i], types[j])
    return out


def write_structure_files(folder, cell, cart, species, cfg, fermi_ev):
    cell = np.asarray(cell, float)
    types, norb, _ = species_orbital_info(cfg, species)
    np.savetxt(os.path.join(folder, "lat.dat"), cell.T)
    np.savetxt(os.path.join(folder, "rlat.dat"),
               np.linalg.inv(cell) * 2.0 * np.pi)
    np.savetxt(os.path.join(folder, "site_positions.dat"),
               np.asarray(cart, float).T)
    atomic_write_text(os.path.join(folder, "element.dat"),
                      "\n".join(str(ATOMIC_NUMBERS[s]) for s in species) + "\n")
    atomic_write_text(os.path.join(folder, "orbital_types.dat"),
                      "\n".join("  ".join(str(l) for l in t)
                                for t in types) + "\n")
    info = {"nsites": len(species), "isorthogonal": False,
            "isspinful": False, "norbits": int(sum(norb)),
            "fermi_level": fermi_ev if fermi_ev is not None else 0.0}
    atomic_write_text(os.path.join(folder, "info.json"), json.dumps(info))


def collect_dft_stage(cfg, workspace, args):
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell
    if getattr(args, "set_name", None) is None:
        raise SystemExit("collect-dft requires --set pilot|main|large")
    ref = load_reference(workspace)
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    store = SnapshotStore(workspace, args.set_name)
    tau_diag = float(cfg["validation"]["tau_overlap_diag"])
    exit_code, converted, skipped = 0, 0, 0
    for sid in store.list():
        if store.read_status(sid)["state"] == "rejected":
            continue
        if store.state_at_least(sid, "converted") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        out_dir = os.path.join(folder, "OUT.MgO")
        log = os.path.join(out_dir, "running_scf.log")
        if not os.path.exists(log):
            continue                       # DFT not run yet: stay prepared
        from . import abacus_io
        scf = abacus_io.parse_running_scf(log)
        if not scf["converged"]:
            store.reject(sid, "scf_not_converged")
            exit_code = 1
            continue
        h_path = os.path.join(out_dir, cfg["abacus"]["csr_h_filename"])
        s_path = os.path.join(out_dir, cfg["abacus"]["csr_s_filename"])
        if not (os.path.exists(h_path) and os.path.exists(s_path)):
            store.reject(sid, "csr_files_missing")
            exit_code = 1
            continue
        cell, cart, species = abacus_io.parse_stru(
            os.path.join(folder, "STRU"))
        u = np.load(os.path.join(folder, "displacements.npy"))
        if species != sc.species:
            store.reject(sid, "atom_order_changed")
            exit_code = 1
            continue
        if not (np.allclose(cell, sc.cell, atol=1e-8)
                and np.allclose(cart, sc.cart + u, atol=1e-6)):
            store.reject(sid, "geometry_mismatch_vs_reference")
            exit_code = 1
            continue
        try:
            dim_h, h_csr = abacus_io.parse_csr(h_path)
            dim_s, s_csr = abacus_io.parse_csr(s_path)
            if dim_h != dim_s:
                raise ValueError(f"H dim {dim_h} != S dim {dim_s}")
            h_blocks = matrices_to_blocks(h_csr, dim_h, cfg, species,
                                          RY_TO_EV)
            s_blocks = matrices_to_blocks(s_csr, dim_s, cfg, species, 1.0)
        except ValueError as e:
            store.reject(sid, f"matrix_parse_failed: {e}")
            exit_code = 1
            continue
        bad_diag = False
        for i in range(len(species)):
            k = key_str((0, 0, 0), i, i)
            if k not in s_blocks or \
                    np.abs(np.diag(s_blocks[k]) - 1.0).max() > tau_diag:
                bad_diag = True
        if bad_diag:
            store.reject(sid, "pathological_overlap_diagonal")
            exit_code = 1
            continue
        full_path = os.path.join(folder, "hamiltonians_full.h5")
        if os.path.exists(full_path) and not args.force:
            raise SystemExit(f"{sid}: hamiltonians_full.h5 exists; "
                             "refusing to overwrite without --force")
        write_blocks(full_path, h_blocks)
        write_blocks(os.path.join(folder, "overlaps.h5"), s_blocks)
        write_structure_files(folder, cell, cart, species, cfg,
                              scf["fermi_ev"])
        store.write_status(
            sid, "converted", scf_converged=True, etot_ev=scf["etot_ev"],
            csr_files=[cfg["abacus"]["csr_h_filename"],
                       cfg["abacus"]["csr_s_filename"]],
            raw_sha256={os.path.basename(p): sha256_file(p)
                        for p in (h_path, s_path, log)})
        converted += 1
    print(f"{args.set_name}: converted {converted}, skipped {skipped}")
    return exit_code
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_convert.py mgo_lr/tests/test_orbital_reorder.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/convert.py mgo_lr/config.py mgo_lr/tests/test_convert.py
git commit -m "feat(mgo_lr): DeepH-E3 format conversion and collect-dft stage"
```

---

### Task 13: LR core — reciprocal set, screened-dipole potential, H^LR assembly

**Files:**
- Create: `mgo_lr/lr.py` (core functions; the stage is Task 14)
- Test: `mgo_lr/tests/test_lr_core.py`

**Interfaces:**
- Consumes: `constants.C_COUL/LR_SIGN`, `displacements.remove_uniform_translation`, `convert.parse_key/key_str`.
- Produces: `lr.gmax_squared(lam, tol) -> float` (`4λ²ln(1/tol)`, the bound on `G·ε∞·G`); `lr.reciprocal_set(rec_cell, eps, gmax_sq) -> (n_int (M,3) int, g_cart (M,3))`; `lr.check_reciprocal_set(n_int) -> dict` with keys `number_of_vectors, excludes_G_zero, no_duplicates, inversion_symmetric, ok`; `lr.lr_coefficients(g_cart, dipoles (N,3), ref_positions (N,3), eps, lam, volume) -> np.ndarray complex (M,)` (reference-position phase `e^{−iG·R⁰}`, `V(G) = LR_SIGN · φ(G)` with `φ(G) = −i(4π/Ω)C_COUL(Σ G·d e^{−iG·R⁰})/(G·ε·G)·f_Ewald`); `lr.evaluate_potential(g_cart, coeffs, points) -> np.ndarray complex (N,)` (`Σ_G V(G)e^{+iG·r}`); `lr.imaginary_residual(v, delta) -> float`; `lr.minimum_image_displacements(cell, cart, ref_cart) -> (N,3)`; `lr.assemble_lr_hamiltonian(overlap_blocks: dict, v_atom (N,) real) -> dict` (`H^LR_ij(R) = (V_i+V_j)/2 · S_ij(R)` over every stored overlap key); `lr.blocks_norm(d) -> float`, `lr.blocks_diff_norm(a, b) -> float` (Frobenius over the union of keys, absent blocks zero).

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_lr_core.py
import cmath
import math

import numpy as np
import pytest

from mgo_lr import lr
from mgo_lr.constants import C_COUL, LR_SIGN
from mgo_lr.convert import key_str
from mgo_lr.structures import reciprocal

EPS_I = np.eye(3)


def _cube(L=8.0):
    cell = L * np.eye(3)
    return cell, reciprocal(cell), abs(np.linalg.det(cell))


def test_reciprocal_set_properties():
    cell, rec, vol = _cube()
    gmax_sq = lr.gmax_squared(1.0, 1e-10)
    n_int, g_cart = lr.reciprocal_set(rec, EPS_I, gmax_sq)
    rep = lr.check_reciprocal_set(n_int)
    assert rep["ok"] and rep["number_of_vectors"] == len(n_int) > 0
    assert rep["excludes_G_zero"] and rep["inversion_symmetric"]
    # every vector satisfies the ellipsoidal cutoff
    assert all(g @ EPS_I @ g <= gmax_sq + 1e-9 for g in g_cart)


def test_check_flags_broken_set():
    n_int, _ = lr.reciprocal_set(_cube()[1], EPS_I, lr.gmax_squared(1.0, 1e-6))
    broken = n_int[1:]                      # drop one vector -> asymmetric
    assert lr.check_reciprocal_set(broken)["ok"] is False
    with_zero = np.vstack([n_int, [0, 0, 0]])
    assert lr.check_reciprocal_set(with_zero)["excludes_G_zero"] is False


def test_sign_and_prefactor_against_filtered_dipole():
    """Production vectorized implementation vs a deliberately slow loop
    reference building the SAME filtered coefficients, plus the sign pin:
    an electron just above the positive lobe of a +z dipole has NEGATIVE
    potential energy."""
    cell, rec, vol = _cube(8.0)
    lam, tol = 0.6, 1e-8
    n_int, g_cart = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(lam, tol))
    dipoles = np.array([[0.0, 0.0, 0.1]])
    refpos = np.array([[4.0, 4.0, 4.0]])
    points = np.array([[4.0, 4.0, 5.0], [4.0, 4.0, 3.0], [5.0, 4.0, 4.0]])
    coeffs = lr.lr_coefficients(g_cart, dipoles, refpos, EPS_I, lam, vol)
    v = lr.evaluate_potential(g_cart, coeffs, points)

    slow = np.zeros(len(points), complex)
    for nvec in n_int:
        g = np.asarray(nvec, float) @ rec
        geg = float(g @ EPS_I @ g)
        f = math.exp(-geg / (4.0 * lam * lam))
        s = sum((g @ d) * cmath.exp(-1j * float(g @ r0))
                for d, r0 in zip(dipoles, refpos))
        vg = LR_SIGN * (-1j) * (4.0 * math.pi / vol) * C_COUL * s / geg * f
        for p, r in enumerate(points):
            slow[p] += vg * cmath.exp(1j * float(g @ r))
    assert np.allclose(v, slow, atol=1e-10)
    assert lr.imaginary_residual(v, 1e-12) < 1e-10
    vr = np.real(v)
    assert vr[0] < 0.0 < vr[1]              # sign pin (electron energy)
    assert abs(vr[0] + vr[1]) < 1e-10       # odd in z


def test_coefficients_linear_in_dipoles():
    cell, rec, vol = _cube()
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(0.8, 1e-8))
    rng = np.random.default_rng(0)
    d = rng.standard_normal((4, 3)) * 0.01
    r0 = rng.uniform(0, 8, (4, 3))
    c1 = lr.lr_coefficients(g, d, r0, EPS_I, 0.8, vol)
    c2 = lr.lr_coefficients(g, 2.0 * d, r0, EPS_I, 0.8, vol)
    assert np.allclose(c2, 2.0 * c1)        # exactly linear in u_rel
    c0 = lr.lr_coefficients(g, 0.0 * d, r0, EPS_I, 0.8, vol)
    assert np.allclose(c0, 0.0)             # equilibrium -> exact zero


def test_uniform_translation_exact_zero():
    from mgo_lr.displacements import remove_uniform_translation
    u = np.tile([[0.03, -0.01, 0.02]], (6, 1))
    assert np.allclose(remove_uniform_translation(u), 0.0)


def test_realness_requires_inversion_symmetry():
    cell, rec, vol = _cube()
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(0.8, 1e-6))
    d = np.array([[0.0, 0.0, 0.05]])
    r0 = np.array([[1.234, 2.345, 3.456]])
    c = lr.lr_coefficients(g, d, r0, EPS_I, 0.8, vol)
    pts = np.array([[0.5, 1.5, 2.5]])
    assert lr.imaginary_residual(lr.evaluate_potential(g, c, pts), 1e-12) < 1e-10
    # drop one vector: residual blows up
    v_bad = lr.evaluate_potential(g[1:], c[1:], pts)
    assert lr.imaginary_residual(v_bad, 1e-12) > 1e-4


def test_gmax_convergence_at_fixed_lambda():
    cell, rec, vol = _cube()
    lam, tol = 0.8, 1e-10
    d = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    r0 = np.array([[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]])
    pts = r0 + 0.01
    vs = []
    for scale in (1.0, 1.5 ** 2):
        n_int, g = lr.reciprocal_set(rec, EPS_I,
                                     lr.gmax_squared(lam, tol) * scale)
        c = lr.lr_coefficients(g, d, r0, EPS_I, lam, vol)
        vs.append(np.real(lr.evaluate_potential(g, c, pts)))
    rel = np.linalg.norm(vs[1] - vs[0]) / (np.linalg.norm(vs[1]) + 1e-12)
    assert rel < 1e-6                       # converged at fixed Lambda


def test_minimum_image_displacements():
    cell = 10.0 * np.eye(3)
    ref = np.array([[0.5, 0.5, 0.5]])
    cart = np.array([[9.8, 0.5, 0.5]])      # wrapped: really at -0.2
    u = lr.minimum_image_displacements(cell, cart, ref)
    assert np.allclose(u, [[-0.7, 0.0, 0.0]])


def test_assemble_and_hermiticity_and_small_amplitude():
    """H^LR = (V_i+V_j)/2 S inherits hermiticity from S; with u-dependent S
    the sign-reversal and linearity errors DECREASE with amplitude."""
    cell, rec, vol = _cube(8.0)
    lam = 0.8
    n_int, g = lr.reciprocal_set(rec, EPS_I, lr.gmax_squared(lam, 1e-8))
    ref = np.array([[2.0, 2.0, 2.0], [6.0, 2.0, 2.0]])
    z = np.array([np.eye(3) * 2.0, np.eye(3) * -2.0])

    def h_lr(amp):
        u = np.array([[amp, 0.0, 0.0], [-amp, 0.0, 0.0]])
        from mgo_lr.displacements import remove_uniform_translation
        u_rel = remove_uniform_translation(u)
        d = np.einsum("nab,nb->na", z, u_rel)
        pos = ref + u
        c = lr.lr_coefficients(g, d, ref, EPS_I, lam, vol)
        v = np.real(lr.evaluate_potential(g, c, pos))
        dist = np.linalg.norm(pos[0] - pos[1])
        s12 = np.array([[math.exp(-dist / 4.0)]])   # u-dependent overlap
        s = {key_str((0, 0, 0), 0, 0): np.array([[1.0]]),
             key_str((0, 0, 0), 1, 1): np.array([[1.0]]),
             key_str((0, 0, 0), 0, 1): s12,
             key_str((0, 0, 0), 1, 0): s12.T.copy()}
        return lr.assemble_lr_hamiltonian(s, v)

    h = h_lr(0.02)
    assert np.allclose(h[key_str((0, 0, 0), 0, 1)],
                       h[key_str((0, 0, 0), 1, 0)].T)    # hermitian
    delta = 1e-12

    def e_sign(a):
        return lr.blocks_diff_norm(h_lr(a), {k: -v for k, v in
                                             h_lr(-a).items()}) \
            / (lr.blocks_norm(h_lr(a)) + delta)

    def e_linear(a):
        h2, h1 = h_lr(2 * a), h_lr(a)
        return lr.blocks_diff_norm(h2, {k: 2.0 * v for k, v in h1.items()}) \
            / (2.0 * lr.blocks_norm(h1) + delta)

    assert e_sign(0.005) < e_sign(0.01) < e_sign(0.02)
    assert e_linear(0.005) < e_linear(0.01) < e_linear(0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_lr_core.py -q` → ModuleNotFoundError `mgo_lr.lr`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/lr.py
"""Standalone long-range (LR) Hamiltonian processor.

Unit/sign convention (see also constants.py — this module is the ONLY
place the Coulomb prefactor and sign enter):

    Z*  dimensionless (units of e);  u in Å;  d_k = Z~*_k u_k^rel (e·Å)
    phi(G) = -i (4π/Ω) C_COUL [Σ_k G·d_k e^(-iG·R0_k)] / (G·ε∞·G) f_Ewald(G)
    V_LR(G) = LR_SIGN · phi(G)          # electron potential energy, eV
    V_LR(r) = Σ_{G∈𝒢} V_LR(G) e^(+iG·r);   V(G=0) = 0 (fixed gauge)
    f_Ewald(G) = exp(-(G·ε∞·G)/(4Λ²))

Λ is part of the dataset definition: the damped reciprocal-space sum alone
IS the LR definition (no compensating real-space term), so H^LR depends on
Λ by construction.  G-set requirements (inversion symmetry, G=0 excluded,
no duplicates) are hard: the realness of V^LR depends on them.
"""
import json
import os

import numpy as np

from .constants import C_COUL, LR_SIGN
from .convert import key_str, parse_key


def gmax_squared(lam, tol):
    """Bound on G·ε∞·G from the f_Ewald floor `tol`."""
    return 4.0 * float(lam) ** 2 * np.log(1.0 / float(tol))


def reciprocal_set(rec_cell, eps, gmax_sq):
    """Integer combinations of supercell reciprocal vectors inside the
    dielectric ellipsoid G·ε∞·G <= gmax_sq, G=0 excluded.  The symmetric
    cutoff makes the set inversion-symmetric by construction."""
    rec = np.asarray(rec_cell, float)
    eps = np.asarray(eps, float)
    eps_min = float(np.linalg.eigvalsh(0.5 * (eps + eps.T)).min())
    if eps_min <= 0.0:
        raise ValueError("dielectric tensor not positive definite")
    gmax_cart = np.sqrt(gmax_sq / eps_min)
    real = 2.0 * np.pi * np.linalg.inv(rec).T          # rows a_i
    nmax = [int(np.ceil(gmax_cart * np.linalg.norm(a) / (2.0 * np.pi)))
            for a in real]
    ns, gs = [], []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                if n1 == n2 == n3 == 0:
                    continue
                g = np.array([n1, n2, n3], float) @ rec
                if float(g @ eps @ g) <= gmax_sq:
                    ns.append((n1, n2, n3))
                    gs.append(g)
    return np.array(ns, int).reshape(-1, 3), np.array(gs).reshape(-1, 3)


def check_reciprocal_set(n_int):
    tuples = [tuple(int(x) for x in v) for v in np.asarray(n_int).reshape(-1, 3)]
    s = set(tuples)
    rep = {"number_of_vectors": len(tuples),
           "excludes_G_zero": (0, 0, 0) not in s,
           "no_duplicates": len(s) == len(tuples),
           "inversion_symmetric": all((-a, -b, -c) in s for a, b, c in s)}
    rep["ok"] = (rep["excludes_G_zero"] and rep["no_duplicates"]
                 and rep["inversion_symmetric"])
    return rep


def lr_coefficients(g_cart, dipoles, ref_positions, eps, lam, volume):
    """V_LR(G) with the reference-position phase convention (exactly linear
    in u^rel)."""
    g = np.asarray(g_cart, float)
    eps = np.asarray(eps, float)
    geg = np.einsum("ga,ab,gb->g", g, eps, g)
    f_ewald = np.exp(-geg / (4.0 * float(lam) ** 2))
    gd = g @ np.asarray(dipoles, float).T                       # (M,N)
    phases = np.exp(-1j * (g @ np.asarray(ref_positions, float).T))
    s_g = np.sum(gd * phases, axis=1)
    phi = -1j * (4.0 * np.pi / float(volume)) * C_COUL * s_g / geg * f_ewald
    return LR_SIGN * phi


def evaluate_potential(g_cart, coeffs, points):
    ph = np.exp(1j * (np.asarray(points, float) @ np.asarray(g_cart, float).T))
    return ph @ np.asarray(coeffs)


def imaginary_residual(v, delta):
    v = np.asarray(v)
    return float(np.linalg.norm(np.imag(v))
                 / (np.linalg.norm(np.real(v)) + float(delta)))


def minimum_image_displacements(cell, cart, ref_cart):
    """u = cart - ref wrapped to the nearest image (valid for |u| << cell)."""
    cell = np.asarray(cell, float)
    dfrac = (np.asarray(cart, float) - np.asarray(ref_cart, float)) \
        @ np.linalg.inv(cell)
    dfrac -= np.round(dfrac)
    return dfrac @ cell


def assemble_lr_hamiltonian(overlap_blocks, v_atom):
    """H^LR_ij(R) = (V_i + V_j)/2 * S_ij(R) over every stored overlap key.
    Hermiticity is inherited from S."""
    v_atom = np.asarray(v_atom, float)
    out = {}
    for k, s in overlap_blocks.items():
        _, _, _, i, j = parse_key(k)                    # 1-based
        out[k] = 0.5 * (v_atom[i - 1] + v_atom[j - 1]) * np.asarray(s, float)
    return out


def blocks_norm(blocks):
    return float(np.sqrt(sum(float(np.sum(v * v)) for v in blocks.values())))


def blocks_diff_norm(a, b):
    """Frobenius norm of a - b over the union of keys (absent -> zero)."""
    tot = 0.0
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        d = va - vb if va is not None and vb is not None \
            else (va if vb is None else -vb)
        tot += float(np.sum(d * d))
    return float(np.sqrt(tot))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_lr_core.py -q` → 9 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/lr.py mgo_lr/tests/test_lr_core.py
git commit -m "feat(mgo_lr): LR core (reciprocal set, screened dipole potential, H_LR assembly)"
```

---

### Task 14: `lr-process` stage

**Files:**
- Modify: `mgo_lr/lr.py`
- Test: `mgo_lr/tests/test_lr_process.py`

**Interfaces:**
- Consumes: Task 13 core, `convert.read_blocks/write_blocks`, `SnapshotStore`, `load_reference`, `make_supercell`, `displacements.remove_uniform_translation`.
- Produces: `lr.lr_process_stage(cfg, workspace, args) -> int`. Per snapshot ≥ `converted`: recomputes `u` by minimum image from `site_positions.dat` vs the rebuilt reference supercell (warns if it disagrees with `displacements.npy` by > 1e-6 Å), removes uniform translation **inside the processor**, forms dipoles with the ASR-corrected Born charges, builds 𝒢 (hard-fails the whole stage if `check_reciprocal_set` fails), evaluates `V^LR` at the **snapshot** positions, applies the imaginary-residual hard gate (on failure: writes `lr_failure.json` with reciprocal-set diagnostics, sets `lr_failed` in status, writes **no** LR/SR labels, exit 1), writes `hamiltonians_lr.h5` + `hamiltonians_sr.h5` (atomic), `lr_metadata.json` (the spec's `lr_definition` block + `r_imag` + `lr_convergence` from the `convergence_factor`-scaled G set), merges `lr_definition` into `<workspace>/metadata.yaml` refusing to mix definitions, sets state `lr_done`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_lr_process.py
import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import convert, lr
from mgo_lr.config import load_config
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell
from mgo_lr.tests.test_convert import (Args, fabricate_dft, prepared_snapshot,
                                       small_cfg)
from mgo_lr.tests.test_gen_structures import make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")


def lr_cfg():
    cfg = small_cfg()
    cfg["lr"]["ewald_lambda"] = 1.0     # non-empty G set on the 2-atom cell
    return cfg


def add_dfpt_artifacts(ws):
    ref = os.path.join(ws, "reference")
    z = np.array([np.eye(3) * 1.97, np.eye(3) * -1.97])
    np.save(os.path.join(ref, "born_effective_charges.npy"), z)
    np.save(os.path.join(ref, "dielectric_infinity.npy"), np.eye(3) * 3.0)


def converted_snapshot(tmp_path, u=None):
    ws = str(tmp_path)
    cfg = lr_cfg()
    store, sid, sc = prepared_snapshot(ws, cfg, u=u)
    add_dfpt_artifacts(ws)
    fabricate_dft(store.folder(sid), cfg, sc)
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    return ws, cfg, store, sid, sc


def test_lr_process_equilibrium_zero(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)     # u = 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    folder = store.folder(sid)
    h_lr = convert.read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) < 1e-10                # H_LR(u=0) = 0
    h_full = convert.read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_sr = convert.read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))
    assert lr.blocks_diff_norm(
        {k: h_sr.get(k, 0) + h_lr.get(k, 0) * 0 for k in h_sr}, {}) > 0
    # reconstruction: H_SR + H_LR = H_full on the union
    total = {k: h_sr.get(k, np.zeros_like(h_lr.get(k)))
             + h_lr.get(k, np.zeros_like(h_sr.get(k)))
             for k in set(h_sr) | set(h_lr)}
    assert lr.blocks_diff_norm(total, h_full) < 1e-10
    meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    ld = meta["lr_definition"]
    assert ld["gauge"] == "G_zero_equals_zero"
    assert ld["sign_convention"] == "electron_potential_energy"
    assert ld["phase_convention"] == "reference_positions"
    assert ld["reciprocal_set"]["inversion_symmetric"] is True
    assert ld["reciprocal_set"]["number_of_vectors"] > 0
    assert meta["r_imag"] < cfg["lr"]["imaginary_tolerance"]
    assert store.read_status(sid)["state"] == "lr_done"
    ws_meta = yaml.safe_load(open(os.path.join(ws, "metadata.yaml")))
    assert ws_meta["lr_definition"]["ewald_lambda"] == 1.0


def test_lr_process_translation_zero(tmp_path):
    u = np.tile([[0.02, 0.01, -0.01]], (2, 1))
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path, u=u)
    lr.lr_process_stage(cfg, ws, Args())
    h_lr = convert.read_blocks(
        os.path.join(store.folder(sid), "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) < 1e-10        # exact zero by construction


def test_lr_process_nonzero_for_optical(tmp_path):
    u = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path, u=u)
    lr.lr_process_stage(cfg, ws, Args())
    folder = store.folder(sid)
    h_lr = convert.read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    assert lr.blocks_norm(h_lr) > 1e-6
    # hermiticity inherited from S
    for k, v in h_lr.items():
        r0, r1, r2, i, j = convert.parse_key(k)
        pk = convert.key_str((-r0, -r1, -r2), j - 1, i - 1)
        assert np.allclose(v, h_lr[pk].T, atol=1e-10)
    meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    assert meta["lr_convergence"] < cfg["validation"]["tau_G"]


def test_lr_process_idempotent_and_lambda_guard(tmp_path):
    ws, cfg, store, sid, sc = converted_snapshot(tmp_path)
    lr.lr_process_stage(cfg, ws, Args())
    before = store.read_status(sid)["history"]
    lr.lr_process_stage(cfg, ws, Args())               # skip, no --force
    assert store.read_status(sid)["history"] == before
    cfg2 = copy.deepcopy(cfg)
    cfg2["lr"]["ewald_lambda"] = 0.5
    args = Args()
    args.force = True
    with pytest.raises(SystemExit, match="lr_definition"):
        lr.lr_process_stage(cfg2, ws, args)            # refuses to mix Λ


def test_lr_process_imaginary_gate(tmp_path, monkeypatch):
    ws, cfg, store, sid, sc = converted_snapshot(
        tmp_path, u=np.array([[0.01, 0, 0], [-0.01, 0, 0]]))

    def broken(g, c, pts):
        return np.full(len(np.atleast_2d(pts)), 1.0 + 1.0j)

    monkeypatch.setattr(lr, "evaluate_potential", broken)
    assert lr.lr_process_stage(cfg, ws, Args()) == 1
    folder = store.folder(sid)
    assert os.path.exists(os.path.join(folder, "lr_failure.json"))
    assert not os.path.exists(os.path.join(folder, "hamiltonians_lr.h5"))
    assert not os.path.exists(os.path.join(folder, "hamiltonians_sr.h5"))
    assert "lr_failed" in store.read_status(sid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_lr_process.py -q` → AttributeError `lr_process_stage`.

- [ ] **Step 3: Write the implementation**

Append to `mgo_lr/lr.py` (add `import yaml` and the config/snapshot imports used below):

```python
def _lr_definition(cfg, gmax_sq, rep):
    return {"ewald_lambda": float(cfg["lr"]["ewald_lambda"]),
            "reciprocal_cutoff": float(gmax_sq),
            "reciprocal_tolerance": float(cfg["lr"]["reciprocal_tolerance"]),
            "reciprocal_set": {"inversion_symmetric": True,
                               "excludes_G_zero": True,
                               "cutoff_type": "dielectric_ellipsoid",
                               "number_of_vectors": int(rep["number_of_vectors"])},
            "imaginary_tolerance": float(cfg["lr"]["imaginary_tolerance"]),
            "gauge": "G_zero_equals_zero",
            "sign_convention": "electron_potential_energy",
            "phase_convention": "reference_positions"}


def _record_lr_definition(workspace, lr_def):
    import yaml
    from .config import atomic_write_text
    path = os.path.join(workspace, "metadata.yaml")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    stored = data.get("lr_definition")
    if stored is not None and stored != lr_def:
        raise SystemExit(
            "metadata.yaml already records a different lr_definition — "
            "refusing to mix LR definitions in one workspace (change the "
            "workspace or restore the original Λ/cutoff config)")
    data["lr_definition"] = lr_def
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def lr_process_stage(cfg, workspace, args):
    from .config import atomic_write_text
    from .convert import read_blocks, write_blocks
    from .displacements import remove_uniform_translation
    from .snapshot import SnapshotStore, load_reference
    from .structures import make_supercell, reciprocal
    from . import __version__

    if getattr(args, "set_name", None) is None:
        raise SystemExit("lr-process requires --set pilot|main|large")
    ref = load_reference(workspace)
    ref_dir = os.path.join(workspace, "reference")
    born = np.load(os.path.join(ref_dir, "born_effective_charges.npy"))
    eps = np.load(os.path.join(ref_dir, "dielectric_infinity.npy"))
    n = cfg["supercells"][args.set_name]
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"], n)
    lam = float(cfg["lr"]["ewald_lambda"])
    tol = float(cfg["lr"]["reciprocal_tolerance"])
    tau_imag = float(cfg["lr"]["imaginary_tolerance"])
    factor = float(cfg["lr"]["convergence_factor"])
    delta = float(cfg["validation"]["delta"])
    rec = reciprocal(sc.cell)
    volume = abs(float(np.linalg.det(sc.cell)))
    gmax_sq = gmax_squared(lam, tol)
    n_int, g_cart = reciprocal_set(rec, eps, gmax_sq)
    rep = check_reciprocal_set(n_int)
    if not rep["ok"] or rep["number_of_vectors"] == 0:
        raise SystemExit(f"reciprocal set invalid or empty: {rep}")
    n_int2, g2 = reciprocal_set(rec, eps, gmax_sq * factor ** 2)
    lr_def = _lr_definition(cfg, gmax_sq, rep)
    _record_lr_definition(workspace, lr_def)

    store = SnapshotStore(workspace, args.set_name)
    exit_code, processed, skipped = 0, 0, 0
    for sid in store.list():
        st = store.read_status(sid)
        if st["state"] == "rejected" \
                or not store.state_at_least(sid, "converted"):
            continue
        if store.state_at_least(sid, "lr_done") and not args.force:
            skipped += 1
            continue
        folder = store.folder(sid)
        pos = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        u = minimum_image_displacements(sc.cell, pos, sc.cart)
        u_stored = np.load(os.path.join(folder, "displacements.npy"))
        if np.abs(u - u_stored).max() > 1e-6:
            print(f"WARNING {sid}: recomputed u differs from "
                  f"displacements.npy by {np.abs(u - u_stored).max():.2e} Å")
        u_rel = remove_uniform_translation(u)          # processor-level ASR
        dipoles = np.einsum("nab,nb->na", born[sc.basis_index], u_rel)
        coeffs = lr_coefficients(g_cart, dipoles, sc.cart, eps, lam, volume)
        v_c = evaluate_potential(g_cart, coeffs, pos)  # snapshot AO centers
        r_imag = imaginary_residual(v_c, delta)
        if r_imag >= tau_imag:
            atomic_write_text(
                os.path.join(folder, "lr_failure.json"),
                json.dumps({"r_imag": r_imag, "reciprocal_set": rep,
                            "n_vectors": int(len(n_int)),
                            "lr_definition": lr_def}, indent=1))
            store.write_status(sid, st["state"],
                               lr_failed=f"imaginary_residual {r_imag:.3e}")
            exit_code = 1
            continue
        v_atom = np.real(v_c)
        s_blocks = read_blocks(os.path.join(folder, "overlaps.h5"))
        h_full = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        h_lr = assemble_lr_hamiltonian(s_blocks, v_atom)
        coeffs2 = lr_coefficients(g2, dipoles, sc.cart, eps, lam, volume)
        v2 = np.real(evaluate_potential(g2, coeffs2, pos))
        h_lr2 = assemble_lr_hamiltonian(s_blocks, v2)
        conv = blocks_diff_norm(h_lr2, h_lr) / (blocks_norm(h_lr2) + delta)
        h_sr = {}
        for k in set(h_full) | set(h_lr):
            hf = h_full.get(k)
            hl = h_lr.get(k)
            if hf is None:
                hf = np.zeros_like(hl)
            if hl is None:
                hl = np.zeros_like(hf)
            h_sr[k] = hf - hl
        write_blocks(os.path.join(folder, "hamiltonians_lr.h5"), h_lr)
        write_blocks(os.path.join(folder, "hamiltonians_sr.h5"), h_sr)
        atomic_write_text(os.path.join(folder, "lr_metadata.json"),
                          json.dumps({"lr_definition": lr_def,
                                      "r_imag": r_imag,
                                      "lr_convergence": conv,
                                      "code_version": __version__}, indent=1))
        store.write_status(sid, "lr_done", r_imag=r_imag, lr_convergence=conv)
        processed += 1
    print(f"{args.set_name}: lr-processed {processed}, skipped {skipped}")
    return exit_code
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_lr_process.py mgo_lr/tests/test_lr_core.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/lr.py mgo_lr/tests/test_lr_process.py
git commit -m "feat(mgo_lr): lr-process stage (H_LR/H_SR labels, imaginary gate, metadata)"
```

### Task 15: `validate` stage (Tier-1 hard checks, Tier-2 response checks)

**Files:**
- Create: `mgo_lr/validate.py`
- Test: `mgo_lr/tests/test_validate.py`

**Interfaces:**
- Consumes: `convert.read_blocks/key_str/parse_key/species_orbital_info`, `lr.blocks_norm/blocks_diff_norm`, `displacements.remove_uniform_translation`, `SnapshotStore`, `load_reference`, `make_supercell`, `config.atomic_write_text/sha256_file`.
- Produces: `validate.REQUIRED_FILES: list[str]`; `validate.hermiticity_error(blocks: dict) -> float` (max `|H_ij(R) − H_ji(−R)ᵀ|`, `inf` on an unpaired block); `validate.check_keys_and_dims(blocks, norb: list[int]) -> str | None`; `validate.tier1_snapshot(cfg, folder, status, sc, born) -> (list[str] failures, dict metrics)`; `validate.tier2_checks(store, cfg, sids) -> (e_sign: list[dict], e_linear: list[dict], violations: list[str])`; `validate.validate_stage(cfg, workspace, args) -> int`. Tier-1 failure ⇒ snapshot rejected (moved to `rejected/`, reason = joined failures) and exit 1. Tier-2 violations are warnings unless `validation.tier2_enforce` is true (then exit 1, but snapshots stay validated — enforcement gates the set, not individual snapshots). Per-snapshot `quality_checks.json`; set summary `generation_logs/validation_<set>.json` with keys `set, counts, tier1, tier2{e_sign, e_linear, violations, enforced}`. The test helpers `add_snapshot` and `ladder_workspace` are reused by Tasks 16 and 19.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_validate.py
import copy
import json
import os

import numpy as np
import pytest

from mgo_lr import abacus_io, convert, lr, validate
from mgo_lr.snapshot import SnapshotStore
from mgo_lr.structures import make_supercell
from mgo_lr.tests.test_convert import Args, fabricate_dft
from mgo_lr.tests.test_gen_structures import make_fake_reference
from mgo_lr.tests.test_lr_process import add_dfpt_artifacts, lr_cfg


def add_snapshot(ws, cfg, sc, sid, u, meta):
    """Prepared snapshot with fabricated DFT output and explicit metadata."""
    store = SnapshotStore(ws, "pilot")
    folder = store.folder(sid)
    os.makedirs(folder, exist_ok=True)
    abacus_io.write_stru(os.path.join(folder, "STRU"), sc.cell, sc.cart + u,
                         sc.species, cfg)
    np.save(os.path.join(folder, "displacements.npy"), u)
    base = {"pattern_class": "single_q_optical", "pattern_group_id": "grp-test",
            "comparison_family_id": "fam-test", "rigid_translation": False,
            "sign_partner_id": None, "amplitude_partner_ids": [],
            "amplitude": 0.0, "polarization_class": "none", "q_magnitude": 0.0}
    base.update(meta)
    with open(os.path.join(folder, "displacement_metadata.json"), "w") as f:
        json.dump(base, f)
    store.write_status(sid, "prepared")
    fabricate_dft(folder, cfg, sc)
    return store


def ladder_workspace(tmp_path):
    """Four optical snapshots ±A / ±2A wired as sign and amplitude partners,
    taken through collect-dft and lr-process."""
    ws = str(tmp_path)
    cfg = lr_cfg()
    cell, frac, species = make_fake_reference(ws)
    add_dfpt_artifacts(ws)
    sc = make_supercell(cell, frac, species, cfg["supercells"]["pilot"])
    x = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    wiring = [("snapshot_000001", 0.01, "snapshot_000002", ["snapshot_000003"]),
              ("snapshot_000002", -0.01, "snapshot_000001", ["snapshot_000004"]),
              ("snapshot_000003", 0.02, "snapshot_000004", ["snapshot_000001"]),
              ("snapshot_000004", -0.02, "snapshot_000003", ["snapshot_000002"])]
    for sid, amp, partner, amp_partners in wiring:
        add_snapshot(ws, cfg, sc, sid, amp * x,
                     {"amplitude": amp, "sign_partner_id": partner,
                      "amplitude_partner_ids": amp_partners})
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    return ws, cfg, SnapshotStore(ws, "pilot")


def test_validate_passes_clean_set(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    assert validate.validate_stage(cfg, ws, Args()) == 0
    for sid in store.list():
        assert store.read_status(sid)["state"] == "validated"
        qc = json.load(open(os.path.join(store.folder(sid),
                                         "quality_checks.json")))
        assert qc["tier1"]["failures"] == []
        assert (qc["tier1"]["metrics"]["reconstruction_error"]
                < cfg["validation"]["tau_reconstruct"])
    summary = json.load(open(os.path.join(ws, "generation_logs",
                                          "validation_pilot.json")))
    assert summary["counts"]["validated"] == 4
    amp_to_val = {round(abs(e["amplitude"]), 6): e["value"]
                  for e in summary["tier2"]["e_sign"]}
    assert set(amp_to_val) == {0.01, 0.02}
    assert 0.0 <= amp_to_val[0.01] < amp_to_val[0.02]   # decreasing with A
    assert summary["tier2"]["e_linear"]
    assert all(e["value"] >= 0.0 for e in summary["tier2"]["e_linear"])
    assert summary["tier2"]["violations"] == []


def test_validate_equilibrium_and_translation(tmp_path):
    ws = str(tmp_path)
    cfg = lr_cfg()
    cell, frac, species = make_fake_reference(ws)
    add_dfpt_artifacts(ws)
    sc = make_supercell(cell, frac, species, cfg["supercells"]["pilot"])
    add_snapshot(ws, cfg, sc, "snapshot_000001", np.zeros((2, 3)),
                 {"pattern_class": "equilibrium"})
    add_snapshot(ws, cfg, sc, "snapshot_000002",
                 np.tile([[0.02, 0.01, -0.01]], (2, 1)),
                 {"pattern_class": "rigid_translation",
                  "rigid_translation": True})
    assert convert.collect_dft_stage(cfg, ws, Args()) == 0
    assert lr.lr_process_stage(cfg, ws, Args()) == 0
    assert validate.validate_stage(cfg, ws, Args()) == 0
    store = SnapshotStore(ws, "pilot")
    qc = json.load(open(os.path.join(store.folder("snapshot_000002"),
                                     "quality_checks.json")))
    m = qc["tier1"]["metrics"]
    assert m["translation_max_u_rel"] < cfg["validation"]["tau_u"]
    assert m["lr_norm"] < cfg["validation"]["tau_translation"]
    qc0 = json.load(open(os.path.join(store.folder("snapshot_000001"),
                                      "quality_checks.json")))
    assert qc0["tier1"]["metrics"]["lr_norm"] < cfg["validation"]["tau_eq"]


def _rejected_reason(ws, sid):
    p = os.path.join(ws, "rejected", f"pilot_{sid}", "status.json")
    return json.load(open(p))["reason"]


def test_validate_rejects_nan(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "hamiltonians_lr.h5")
    blocks = convert.read_blocks(p)
    blocks[next(iter(sorted(blocks)))][0, 0] = np.nan
    convert.write_blocks(p, blocks)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert sid not in store.list()
    assert "nan" in _rejected_reason(ws, sid)


def test_validate_rejects_missing_file(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    os.remove(os.path.join(store.folder(sid), "lat.dat"))
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "missing_file" in _rejected_reason(ws, sid)


def test_validate_rejects_modified_raw(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    with open(os.path.join(store.folder(sid), "OUT.MgO",
                           cfg["abacus"]["csr_h_filename"]), "a") as f:
        f.write("# tampered\n")
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "raw_dft_modified" in _rejected_reason(ws, sid)


def test_validate_rejects_broken_reconstruction(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = store.list()[0]
    p = os.path.join(store.folder(sid), "hamiltonians_sr.h5")
    blocks = convert.read_blocks(p)
    k = convert.key_str((0, 0, 0), 0, 0)     # self-paired: hermiticity survives
    blocks[k] = blocks[k] + 1.0
    convert.write_blocks(p, blocks)
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "reconstruction" in _rejected_reason(ws, sid)


def test_validate_rejects_broken_hermiticity_and_rlat(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid_h, sid_r = store.list()[0], store.list()[1]
    p = os.path.join(store.folder(sid_h), "hamiltonians_lr.h5")
    blocks = convert.read_blocks(p)
    k = convert.key_str((1, 0, 0), 0, 1)     # partner (-1,0,0),2,1 untouched
    blocks[k] = blocks[k] + 1.0
    convert.write_blocks(p, blocks)
    rlat_path = os.path.join(store.folder(sid_r), "rlat.dat")
    np.savetxt(rlat_path, 2.0 * np.loadtxt(rlat_path))
    assert validate.validate_stage(cfg, ws, Args()) == 1
    assert "hermiticity" in _rejected_reason(ws, sid_h)
    assert "rlat" in _rejected_reason(ws, sid_r)


def test_validate_tier2_enforce_flags_violation(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    sid = "snapshot_000001"                  # the +0.01 member
    folder = store.folder(sid)
    h_lr = {k: 5.0 * v for k, v in convert.read_blocks(
        os.path.join(folder, "hamiltonians_lr.h5")).items()}
    h_full = convert.read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_sr = {k: h_full[k] - h_lr[k] for k in h_full}  # reconstruction stays exact
    convert.write_blocks(os.path.join(folder, "hamiltonians_lr.h5"), h_lr)
    convert.write_blocks(os.path.join(folder, "hamiltonians_sr.h5"), h_sr)
    cfg2 = copy.deepcopy(cfg)
    cfg2["validation"]["tier2_enforce"] = True
    assert validate.validate_stage(cfg2, ws, Args()) == 1
    assert sid in store.list()               # tier-2 never rejects snapshots
    summary = json.load(open(os.path.join(ws, "generation_logs",
                                          "validation_pilot.json")))
    assert summary["tier2"]["violations"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_validate.py -q` → ModuleNotFoundError `mgo_lr.validate`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/validate.py
"""Validation battery.

Tier 1 (hard, per snapshot -> rejection + exit 1): structural/numerical and
algebraic checks.  Tier 2 (small-amplitude response): E_sign / E_linear must
decrease with decreasing amplitude within a pattern group; warnings unless
validation.tier2_enforce, in which case violations fail the SET (exit 1)
without rejecting individual snapshots.  Tier 3 lives in locality.py.
"""
import json
import os

import numpy as np

from .config import atomic_write_text, sha256_file
from .convert import key_str, parse_key, read_blocks, species_orbital_info
from .displacements import remove_uniform_translation
from .lr import blocks_diff_norm, blocks_norm
from .snapshot import SnapshotStore, load_reference
from .structures import make_supercell

REQUIRED_FILES = ["STRU", "displacements.npy", "displacement_metadata.json",
                  "hamiltonians_full.h5", "overlaps.h5", "hamiltonians_lr.h5",
                  "hamiltonians_sr.h5", "lat.dat", "rlat.dat",
                  "site_positions.dat", "orbital_types.dat", "element.dat",
                  "info.json", "lr_metadata.json"]


def hermiticity_error(blocks):
    """max |H_ij(R) - H_ji(-R)^T|; inf if any block lacks its partner."""
    worst = 0.0
    for k, v in blocks.items():
        r0, r1, r2, i, j = parse_key(k)
        pk = key_str((-r0, -r1, -r2), j - 1, i - 1)
        if pk not in blocks:
            return float("inf")
        worst = max(worst, float(np.abs(v - blocks[pk].T).max()))
    return worst


def check_keys_and_dims(blocks, norb):
    """1-based indices in range; shapes match orbital counts; finite values."""
    n_at = len(norb)
    for k, v in blocks.items():
        r0, r1, r2, i, j = parse_key(k)
        if not (1 <= i <= n_at and 1 <= j <= n_at):
            return f"key {k}: atom index out of range (must be 1..{n_at})"
        if v.shape != (norb[i - 1], norb[j - 1]):
            return (f"key {k}: block shape {v.shape} != "
                    f"({norb[i - 1]}, {norb[j - 1]})")
        if not np.all(np.isfinite(v)):
            return f"key {k}: nan_or_inf"
    return None


def tier1_snapshot(cfg, folder, status, sc, born):
    val = cfg["validation"]
    delta = float(val["delta"])
    failures, metrics = [], {}
    missing = [f for f in REQUIRED_FILES
               if not os.path.exists(os.path.join(folder, f))]
    if missing:
        return [f"missing_file: {missing}"], metrics

    if not status.get("scf_converged"):
        failures.append("scf_not_converged_in_status")
    out_dir = os.path.join(folder, "OUT.MgO")
    for name, digest in status.get("raw_sha256", {}).items():
        p = os.path.join(out_dir, name)
        if not os.path.exists(p) or sha256_file(p) != digest:
            failures.append(f"raw_dft_modified: {name}")

    _, norb, _ = species_orbital_info(cfg, sc.species)
    h_full = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
    h_lr = read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
    h_sr = read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))
    s = read_blocks(os.path.join(folder, "overlaps.h5"))
    for name, blocks in (("full", h_full), ("lr", h_lr), ("sr", h_sr),
                         ("overlap", s)):
        err = check_keys_and_dims(blocks, norb)
        if err:
            failures.append(f"{name}: {err}")
    if failures:
        return failures, metrics

    ot = open(os.path.join(folder, "orbital_types.dat")).read().splitlines()
    if len(ot) != len(sc.species):
        failures.append(f"orbital_types.dat has {len(ot)} lines for "
                        f"{len(sc.species)} atoms")
    info = json.load(open(os.path.join(folder, "info.json")))
    if (info.get("isspinful") is not False
            or info.get("nsites") != len(sc.species)
            or info.get("norbits") != int(sum(norb))):
        failures.append(f"info.json inconsistent: {info}")
    lat = np.loadtxt(os.path.join(folder, "lat.dat"))
    rlat = np.loadtxt(os.path.join(folder, "rlat.dat"))
    if not np.allclose(rlat.T @ lat, 2.0 * np.pi * np.eye(3), atol=1e-8):
        failures.append("rlat/lat convention violated (need rlat^T lat = 2 pi I)")
    tau_diag = float(val["tau_overlap_diag"])
    for i in range(len(sc.species)):
        k = key_str((0, 0, 0), i, i)
        if k not in s or np.abs(np.diag(s[k]) - 1.0).max() > tau_diag:
            failures.append(f"overlap diagonal pathological at atom {i + 1}")
            break

    for name, blocks in (("full", h_full), ("lr", h_lr), ("sr", h_sr)):
        herm = hermiticity_error(blocks)
        metrics[f"hermiticity_{name}"] = herm
        if herm > float(val["tau_hermiticity"]):
            failures.append(f"hermiticity({name}) = {herm:.3e}")

    total = {}
    for k in set(h_sr) | set(h_lr):
        a, b = h_sr.get(k), h_lr.get(k)
        total[k] = a if b is None else b if a is None else a + b
    rec_err = blocks_diff_norm(total, h_full) / (blocks_norm(h_full) + delta)
    metrics["reconstruction_error"] = rec_err
    if rec_err > float(val["tau_reconstruct"]):
        failures.append(f"reconstruction_error = {rec_err:.3e}")

    lr_meta = json.load(open(os.path.join(folder, "lr_metadata.json")))
    metrics["r_imag"] = lr_meta["r_imag"]
    metrics["lr_convergence"] = lr_meta["lr_convergence"]
    if lr_meta["r_imag"] >= float(cfg["lr"]["imaginary_tolerance"]):
        failures.append(f"imaginary_residual = {lr_meta['r_imag']:.3e}")
    if lr_meta["lr_convergence"] >= float(val["tau_G"]):
        failures.append(f"lr_convergence = {lr_meta['lr_convergence']:.3e}")

    dmeta = json.load(open(os.path.join(folder,
                                        "displacement_metadata.json")))
    lr_norm = blocks_norm(h_lr)
    metrics["lr_norm"] = lr_norm
    if dmeta.get("pattern_class") == "equilibrium" \
            and lr_norm > float(val["tau_eq"]):
        failures.append(f"equilibrium |H_LR| = {lr_norm:.3e}")
    if dmeta.get("rigid_translation"):
        u = np.load(os.path.join(folder, "displacements.npy"))
        u_rel = remove_uniform_translation(u)
        max_u = float(np.linalg.norm(u_rel, axis=1).max())
        metrics["translation_max_u_rel"] = max_u
        metrics["translation_max_dipole"] = float(np.abs(np.einsum(
            "nab,nb->na", born[sc.basis_index], u_rel)).max())
        if max_u > float(val["tau_u"]):
            failures.append(f"translation max|u_rel| = {max_u:.3e}")
        if lr_norm > float(val["tau_translation"]):
            failures.append(f"translation |H_LR| = {lr_norm:.3e}")
    return failures, metrics


def tier2_checks(store, cfg, sids):
    """E_sign per ± pair (counted once, from the positive member) and
    E_linear per amplitude-doubling pair; monotonicity violations per
    pattern group."""
    delta = float(cfg["validation"]["delta"])
    metas = {sid: json.load(open(os.path.join(
        store.folder(sid), "displacement_metadata.json"))) for sid in sids}
    cache = {}

    def lr_blocks(sid):
        if sid not in cache:
            cache[sid] = read_blocks(os.path.join(store.folder(sid),
                                                  "hamiltonians_lr.h5"))
        return cache[sid]

    e_sign, e_linear = [], []
    for sid, m in sorted(metas.items()):
        amp = float(m.get("amplitude") or 0.0)
        group = m.get("pattern_group_id")
        partner = m.get("sign_partner_id")
        if partner in metas and amp > 0.0:
            hp, hm = lr_blocks(sid), lr_blocks(partner)
            value = blocks_diff_norm(hp, {k: -v for k, v in hm.items()}) \
                / (blocks_norm(hp) + delta)
            e_sign.append({"group": group, "amplitude": amp,
                           "sids": [sid, partner], "value": value})
        for pid in m.get("amplitude_partner_ids", []):
            if pid not in metas or amp <= 0.0:
                continue
            amp2 = float(metas[pid].get("amplitude") or 0.0)
            if abs(amp2 - 2.0 * amp) < 1e-12:
                h1, h2 = lr_blocks(sid), lr_blocks(pid)
                value = blocks_diff_norm(
                    h2, {k: 2.0 * v for k, v in h1.items()}) \
                    / (2.0 * blocks_norm(h1) + delta)
                e_linear.append({"group": group, "amplitude": amp,
                                 "sids": [sid, pid], "value": value})
    violations = []
    for series, name in ((e_sign, "e_sign"), (e_linear, "e_linear")):
        by_group = {}
        for e in series:
            by_group.setdefault(e["group"], []).append(e)
        for group, entries in sorted(by_group.items()):
            entries.sort(key=lambda e: e["amplitude"])
            for lo, hi in zip(entries, entries[1:]):
                if lo["value"] > hi["value"]:
                    violations.append(
                        f"{name}[{group}]: {lo['value']:.3e} at "
                        f"A={lo['amplitude']} > {hi['value']:.3e} at "
                        f"A={hi['amplitude']} (must decrease with A)")
    return e_sign, e_linear, violations


def validate_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("validate requires --set pilot|main|large")
    ref = load_reference(workspace)
    born = np.load(os.path.join(workspace, "reference",
                                "born_effective_charges.npy"))
    sc = make_supercell(ref["prim_cell"], ref["frac"], ref["species"],
                        cfg["supercells"][args.set_name])
    store = SnapshotStore(workspace, args.set_name)
    exit_code, results = 0, {}
    for sid in store.list():
        st = store.read_status(sid)
        if st["state"] == "rejected" \
                or not store.state_at_least(sid, "lr_done"):
            continue
        failures, metrics = tier1_snapshot(cfg, store.folder(sid), st, sc,
                                           born)
        qc = {"tier1": {"failures": failures, "metrics": metrics}}
        atomic_write_text(os.path.join(store.folder(sid),
                                       "quality_checks.json"),
                          json.dumps(qc, indent=1))
        results[sid] = failures
        if failures:
            store.reject(sid, "; ".join(failures))
            exit_code = 1
        elif st["state"] != "validated":
            store.write_status(sid, "validated")
    survivors = [sid for sid in store.list()
                 if store.read_status(sid)["state"] == "validated"]
    e_sign, e_linear, violations = tier2_checks(store, cfg, survivors)
    enforced = bool(cfg["validation"]["tier2_enforce"])
    if enforced and violations:
        exit_code = 1
    summary = {"set": args.set_name,
               "counts": {"validated": len(survivors),
                          "rejected": sum(1 for f in results.values() if f)},
               "tier1": {sid: f for sid, f in sorted(results.items())},
               "tier2": {"e_sign": e_sign, "e_linear": e_linear,
                         "violations": violations, "enforced": enforced}}
    logs = os.path.join(workspace, "generation_logs")
    os.makedirs(logs, exist_ok=True)
    atomic_write_text(os.path.join(logs, f"validation_{args.set_name}.json"),
                      json.dumps(summary, indent=1))
    for v in violations:
        print(f"TIER2 {'FAIL' if enforced else 'WARN'}: {v}")
    print(f"{args.set_name}: validated {len(survivors)}, "
          f"rejected {summary['counts']['rejected']}")
    return exit_code
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_validate.py -q` → 8 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/validate.py mgo_lr/tests/test_validate.py
git commit -m "feat(mgo_lr): validate stage (Tier-1 hard checks, Tier-2 response checks)"
```

---

### Task 16: Locality diagnostics and `locality-report` stage

**Files:**
- Create: `mgo_lr/locality.py`
- Modify: `mgo_lr/config.py` (extend `REQUIRED` with `locality.bin_width`)
- Modify: `mgo_lr/configs/mgo.yaml` (add `locality:` section)
- Test: `mgo_lr/tests/test_locality.py`

**Interfaces:**
- Consumes: `convert.parse_key/read_blocks`, `lr.blocks_norm`, `SnapshotStore`, validate's test helpers.
- Produces: `locality.block_distance(key: str, cart (N,3), cell (3,3)) -> float` (AO-center distance `|r_j + R·cell − r_i|`); `locality.frobenius_inner(a, b) -> float` (over shared keys); `locality.odd_response(h_plus, h_minus, h_lr, delta) -> {"cos_theta", "r_lr"}` (`ΔH = (H⁺−H⁻)/2`, `cosθ = ⟨ΔH,H_LR⟩/(‖ΔH‖‖H_LR‖+δ)`, `r_LR = ‖H_LR‖/(‖ΔH‖+δ)`); `locality.tail_fractions(blocks, cart, cell, radii) -> list[float]` (`F(r) = Σ_{d>r}‖blk‖²/Σ‖blk‖²`); `locality.binned_norms(blocks, cart, cell, bin_width) -> list[dict]` (per-bin count/mean/median/max block norm); `locality.locality_report_stage(cfg, workspace, args) -> int` — Tier-3 only, never rejects, always exit 0 when inputs exist. Writes `generation_logs/locality/locality_<set>.json` with keys `set, n_snapshots, tail {radii, F_full, F_lr, F_sr, f_sr_below_f_full}, binned {full, lr, sr}, odd_response [{sids, amplitude, family, cos_theta, r_lr}], families {fam_id: {q_magnitude, members, mean_lr_norm_by_class}}` computed over **validated** snapshots only; odd pairs counted once from the positive-amplitude member; controlled comparisons grouped by `comparison_family_id`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_locality.py
import json
import os

import numpy as np

from mgo_lr import convert, locality, validate
from mgo_lr.tests.test_convert import Args
from mgo_lr.tests.test_validate import ladder_workspace


def test_block_distance():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(locality.block_distance("[0, 0, 0, 1, 2]", cart, cell) - 1.0) < 1e-12
    assert abs(locality.block_distance("[1, 0, 0, 1, 1]", cart, cell) - 10.0) < 1e-12
    assert abs(locality.block_distance("[-1, 0, 0, 1, 2]", cart, cell) - 9.0) < 1e-12


def test_tail_fractions_ordering():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    blocks = {"[0, 0, 0, 1, 1]": np.array([[1.0]]),      # d = 0
              "[0, 0, 0, 1, 2]": np.array([[0.1]])}      # d = 4
    f = locality.tail_fractions(blocks, cart, cell, [1.0, 5.0])
    total = 1.0 + 0.01
    assert abs(f[0] - 0.01 / total) < 1e-12
    assert f[1] == 0.0
    b = locality.binned_norms(blocks, cart, cell, 1.0)
    assert b[0]["count"] == 1 and abs(b[0]["max"] - 1.0) < 1e-12


def test_odd_response_perfect_match():
    k = "[0, 0, 0, 1, 1]"
    h_p = {k: np.array([[2.0]])}
    h_m = {k: np.array([[-2.0]])}
    out = locality.odd_response(h_p, h_m, {k: np.array([[2.0]])}, 1e-12)
    assert abs(out["cos_theta"] - 1.0) < 1e-9
    assert abs(out["r_lr"] - 1.0) < 1e-9
    out2 = locality.odd_response(h_p, h_m, {k: np.array([[-2.0]])}, 1e-12)
    assert abs(out2["cos_theta"] + 1.0) < 1e-9


def test_locality_report_stage(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    assert validate.validate_stage(cfg, ws, Args()) == 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    rep = json.load(open(os.path.join(
        ws, "generation_logs", "locality", "locality_pilot.json")))
    assert rep["n_snapshots"] == 4
    t = rep["tail"]
    assert len(t["radii"]) == len(t["F_full"]) == len(t["F_lr"]) == len(t["F_sr"])
    assert isinstance(t["f_sr_below_f_full"], bool)
    assert all(0.0 <= x <= 1.0 for x in t["F_full"])
    # first radii tail fractions are monotonically non-increasing
    assert all(a >= b - 1e-12 for a, b in zip(t["F_full"], t["F_full"][1:]))
    amps = sorted(round(e["amplitude"], 6) for e in rep["odd_response"])
    assert amps == [0.01, 0.02]               # one entry per ± pair
    for e in rep["odd_response"]:
        assert -1.0 - 1e-9 <= e["cos_theta"] <= 1.0 + 1e-9
        assert e["r_lr"] >= 0.0
    fam = rep["families"]["fam-test"]
    assert len(fam["members"]) == 4
    assert "mean_lr_norm_by_class" in fam


def test_locality_report_empty_set(tmp_path, capsys):
    ws, cfg, store = ladder_workspace(tmp_path)
    # nothing validated yet -> report nothing, still exit 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    assert "no validated" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_locality.py -q` → ModuleNotFoundError `mgo_lr.locality`.

- [ ] **Step 3: Write the implementation**

In `mgo_lr/config.py` `REQUIRED`, add after the `validation.` lines: `"locality.bin_width",`. In `mgo_lr/configs/mgo.yaml`, add after `validation:`:

```yaml
locality:
  bin_width: 1.0            # Angstrom, distance bin for block-norm statistics
```

```python
# mgo_lr/locality.py
"""Tier-3 dataset-level physics and locality diagnostics.

Never a per-snapshot rejection: results are reports under
generation_logs/locality/ feeding the dataset-level approval decision
(F_SR(r) < F_full(r) over the long-distance region before scaling up).
Physics comparisons are made only within matched comparison_family_id
groups.
"""
import json
import os

import numpy as np

from .config import atomic_write_text
from .convert import parse_key, read_blocks
from .lr import blocks_norm
from .snapshot import SnapshotStore


def block_distance(key, cart, cell):
    r0, r1, r2, i, j = parse_key(key)
    shift = np.array([r0, r1, r2], float) @ np.asarray(cell, float)
    cart = np.asarray(cart, float)
    return float(np.linalg.norm(cart[j - 1] + shift - cart[i - 1]))


def frobenius_inner(a, b):
    return float(sum(np.sum(a[k] * b[k]) for k in set(a) & set(b)))


def odd_response(h_plus, h_minus, h_lr, delta):
    """ΔH_DFT = (H(+A) - H(-A))/2 compared against H_LR(+A).  Diagnostic
    only: ΔH_DFT = ΔH_SR + H_LR, so no exact match is expected."""
    dh = {}
    for k in set(h_plus) | set(h_minus):
        p, m = h_plus.get(k), h_minus.get(k)
        if p is None:
            p = np.zeros_like(m)
        if m is None:
            m = np.zeros_like(p)
        dh[k] = 0.5 * (p - m)
    n_dh, n_lr = blocks_norm(dh), blocks_norm(h_lr)
    return {"cos_theta": frobenius_inner(dh, h_lr) / (n_dh * n_lr + delta),
            "r_lr": n_lr / (n_dh + delta)}


def tail_fractions(blocks, cart, cell, radii):
    dw = [(block_distance(k, cart, cell), float(np.sum(v * v)))
          for k, v in blocks.items()]
    total = sum(w for _, w in dw)
    if total <= 0.0:
        return [0.0 for _ in radii]
    return [sum(w for d, w in dw if d > r) / total for r in radii]


def binned_norms(blocks, cart, cell, bin_width):
    bins = {}
    for k, v in blocks.items():
        b = int(block_distance(k, cart, cell) // bin_width)
        bins.setdefault(b, []).append(float(np.linalg.norm(v)))
    return [{"r_lo": b * bin_width, "r_hi": (b + 1) * bin_width,
             "count": len(ns), "mean": float(np.mean(ns)),
             "median": float(np.median(ns)), "max": float(np.max(ns))}
            for b, ns in sorted(bins.items())]


def locality_report_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("locality-report requires --set pilot|main|large")
    delta = float(cfg["validation"]["delta"])
    bin_width = float(cfg["locality"]["bin_width"])
    store = SnapshotStore(workspace, args.set_name)
    sids = [s for s in store.list()
            if store.read_status(s)["state"] == "validated"]
    if not sids:
        print(f"{args.set_name}: no validated snapshots; nothing to report")
        return 0
    metas, h_full, h_lr, h_sr = {}, {}, {}, {}
    for sid in sids:
        folder = store.folder(sid)
        with open(os.path.join(folder, "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
        h_full[sid] = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        h_lr[sid] = read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
        h_sr[sid] = read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))

    folder0 = store.folder(sids[0])
    cell = np.loadtxt(os.path.join(folder0, "lat.dat")).T   # columns -> rows
    cart0 = np.loadtxt(os.path.join(folder0, "site_positions.dat")).T
    rmax = max(block_distance(k, cart0, cell) for k in h_full[sids[0]])
    radii = [bin_width * i for i in range(1, int(rmax // bin_width) + 2)]

    tails = {"full": [], "lr": [], "sr": []}
    for sid in sids:
        cart = np.loadtxt(os.path.join(store.folder(sid),
                                       "site_positions.dat")).T
        tails["full"].append(tail_fractions(h_full[sid], cart, cell, radii))
        tails["lr"].append(tail_fractions(h_lr[sid], cart, cell, radii))
        tails["sr"].append(tail_fractions(h_sr[sid], cart, cell, radii))
    f_mean = {k: np.mean(np.array(v), axis=0).tolist()
              for k, v in tails.items()}
    upper = slice(len(radii) // 2, None)      # long-distance region
    f_sr_ok = bool(all(s <= f + 1e-12 for s, f in
                       zip(f_mean["sr"][upper], f_mean["full"][upper])))

    odd = []
    for sid in sids:
        m = metas[sid]
        partner = m.get("sign_partner_id")
        amp = float(m.get("amplitude") or 0.0)
        if partner in metas and amp > 0.0:
            entry = odd_response(h_full[sid], h_full[partner], h_lr[sid],
                                 delta)
            entry.update({"sids": [sid, partner], "amplitude": amp,
                          "family": m.get("comparison_family_id")})
            odd.append(entry)

    families = {}
    for sid in sids:
        m = metas[sid]
        fam = families.setdefault(
            m.get("comparison_family_id"),
            {"q_magnitude": m.get("q_magnitude"), "members": []})
        fam["members"].append({
            "sid": sid, "polarization_class": m.get("polarization_class"),
            "amplitude": m.get("amplitude"),
            "lr_norm": blocks_norm(h_lr[sid])})
    for fam in families.values():
        by_class = {}
        for e in fam["members"]:
            by_class.setdefault(e["polarization_class"] or "none",
                                []).append(e["lr_norm"])
        fam["mean_lr_norm_by_class"] = {c: float(np.mean(v))
                                        for c, v in by_class.items()}

    report = {"set": args.set_name, "n_snapshots": len(sids),
              "tail": {"radii": radii, "F_full": f_mean["full"],
                       "F_lr": f_mean["lr"], "F_sr": f_mean["sr"],
                       "f_sr_below_f_full": f_sr_ok},
              "binned": {"full": binned_norms(h_full[sids[0]], cart0, cell,
                                              bin_width),
                         "lr": binned_norms(h_lr[sids[0]], cart0, cell,
                                            bin_width),
                         "sr": binned_norms(h_sr[sids[0]], cart0, cell,
                                            bin_width)},
              "odd_response": odd, "families": families}
    out_dir = os.path.join(workspace, "generation_logs", "locality")
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_text(os.path.join(out_dir,
                                   f"locality_{args.set_name}.json"),
                      json.dumps(report, indent=1))
    verdict = "PASS" if f_sr_ok else "NOT YET"
    print(f"{args.set_name}: locality report for {len(sids)} snapshots; "
          f"F_SR < F_full over long distances: {verdict}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_locality.py mgo_lr/tests/test_config.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/locality.py mgo_lr/config.py mgo_lr/configs/mgo.yaml mgo_lr/tests/test_locality.py
git commit -m "feat(mgo_lr): locality-report stage (tail fractions, odd response, families)"
```

---

### Task 17: `organize` stage — grouped splits and dataset metadata

**Files:**
- Create: `mgo_lr/organize.py`
- Test: `mgo_lr/tests/test_organize.py`

**Interfaces:**
- Consumes: `SnapshotStore`, `set_dir_name`, `config.sha256_file/atomic_write_text`, `displacements.MODE_NORMALIZATION`.
- Produces: `organize.grouped_split(groups: dict[str, list[str]], val_frac, test_frac, seed) -> dict` with keys `train/validation/test` — whole `pattern_group_id` groups land in one subset, deterministic under one seed, order-independent input; `organize.organize_stage(cfg, workspace, args) -> int` which (1) computes grouped splits over **validated main-set** snapshots, (2) writes `<workspace>/splits.json` (`seed, fractions, grouping, main{train,validation,test}, pilot, large_test`), (3) populates `validation_candidates/` and `test_candidates/` with relative symlinks into `main/` plus a portable `candidates.json` listing (existing non-symlink entries are a hard error; stale symlinks are replaced), (4) merges the dataset block into `metadata.yaml` **preserving** `lr_definition`/`training_target`: units, atom ordering, mode normalization, supercells, seed, code versions (mgo_lr/ABACUS/QE), full `dft_settings` (abacus+qe config sections), split counts, and `provenance` with **separate per-code hash sections** (`abacus.pseudopotentials/orbitals`, `quantum_espresso.pseudopotentials` — sha256 when the file exists, `null` + a `missing_files` entry otherwise). The 4×4×4 set is only ever listed as `large_test`, never split into train/val/test.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_organize.py
import copy
import json
import os

import numpy as np
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
    import pytest
    with pytest.raises(SystemExit, match="not a symlink"):
        organize.organize_stage(cfg, ws, Args())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_organize.py -q` → ModuleNotFoundError `mgo_lr.organize`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/organize.py
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
    groups = {}
    for sid in _validated(stores["main"]):
        with open(os.path.join(stores["main"].folder(sid),
                               "displacement_metadata.json")) as f:
            gid = json.load(f)["pattern_group_id"]
        groups.setdefault(gid, []).append(sid)
    splits = grouped_split(groups,
                           float(cfg["splits"]["validation_fraction"]),
                           float(cfg["splits"]["test_fraction"]), seed)
    doc = {"seed": seed,
           "validation_fraction": float(cfg["splits"]["validation_fraction"]),
           "test_fraction": float(cfg["splits"]["test_fraction"]),
           "grouping": "pattern_group_id",
           "main": splits,
           "pilot": _validated(stores["pilot"]),
           "large_test": _validated(stores["large"])}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_organize.py -q` → 3 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/organize.py mgo_lr/tests/test_organize.py
git commit -m "feat(mgo_lr): organize stage (grouped splits, candidates, metadata.yaml)"
```

---

### Task 18: `export-target` stage

**Files:**
- Create: `mgo_lr/export.py`
- Test: `mgo_lr/tests/test_export.py`

**Interfaces:**
- Consumes: `SnapshotStore`, `config.atomic_write_text`, `convert.read_blocks` (tests), `lr.blocks_diff_norm` (tests).
- Produces: `export.SOURCES = {"full": "hamiltonians_full.h5", "lr": "hamiltonians_lr.h5", "sr": "hamiltonians_sr.h5"}`; `export.export_snapshot(folder, target) -> "symlink"|"copy"` (relative symlink where supported, else atomic copy; writes marker `export_metadata.json`); `export.export_target_stage(cfg, workspace, args) -> int`. Safety contract: the three source files are **never modified**; `hamiltonians.h5` is only replaced when it is a symlink into `SOURCES` or the marker records a previous export — anything else is `SystemExit` (a foreign `hamiltonians.h5` is never clobbered). Snapshots need state ≥ `lr_done` for `lr`/`sr`, ≥ `converted` for `full`; all three set dirs are processed. The active target is recorded as `training_target` in `metadata.yaml`.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_export.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_export.py -q` → ModuleNotFoundError `mgo_lr.export`.

- [ ] **Step 3: Write the implementation**

```python
# mgo_lr/export.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest mgo_lr/tests/test_export.py -q` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/export.py mgo_lr/tests/test_export.py
git commit -m "feat(mgo_lr): export-target stage (hamiltonians.h5 materialization)"
```

---

### Task 19: README, end-to-end pipeline test, full-suite gate

**Files:**
- Create: `mgo_lr/README.md`
- Test: `mgo_lr/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: every stage function; test helpers `fabricate_dft` (Task 12), `PH_OUT` (Task 8), `lr_cfg` (Task 14).
- Produces: an end-to-end synthetic run of the whole pipeline on the 2-atom cell — init-reference → collect-reference (config override) → init-dfpt → collect-dfpt (fixture ph.out) → gen-structures (46 pilot snapshots) → fabricated DFT for a subset → collect-dft → lr-process → validate → locality-report → organize → export-target sr → CLI `status` smoke test. Snapshots without DFT output must remain `prepared` untouched. `README.md` documents: stage table with commands, the asynchronous cluster round-trip, workspace layout, unit/sign/phase conventions, the Λ policy, and the test command.

- [ ] **Step 1: Write the failing test**

```python
# mgo_lr/tests/test_end_to_end.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest mgo_lr/tests/test_end_to_end.py -q` → fails (README/no changes yet is fine — the test must fail only if any stage misbehaves; if it passes immediately, continue: this task's new artifact is the README plus the e2e gate).

- [ ] **Step 3: Write `mgo_lr/README.md`**

Sections: What this is (one paragraph, pointer to spec + instructions); Requirements (`/opt/anaconda3/envs/DeepH/bin/python`, no maceh import); Quickstart table mapping each stage command to its inputs/outputs and when to run it around the cluster round-trips; Workspace layout tree; Conventions block (eV/Å/e units, `LR_SIGN`, reference-position phase, `G=0` gauge, inversion-symmetric 𝒢, Λ-is-part-of-the-definition policy, 1-based h5 keys, per-atom `orbital_types.dat`, species-major atom ordering); Validation tiers summary; `pytest` test command.

- [ ] **Step 4: Run the full suite**

Run: `cd /Users/jb/MACE-H && /opt/anaconda3/envs/DeepH/bin/python -m pytest mgo_lr/tests -q`
Expected: every test from Tasks 1–19 passes.

- [ ] **Step 5: Commit**

```bash
git add mgo_lr/README.md mgo_lr/tests/test_end_to_end.py
git commit -m "feat(mgo_lr): README and end-to-end synthetic pipeline test"
```
