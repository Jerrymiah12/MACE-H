# Electron-Phonon Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute electron-phonon coupling matrix elements g_mnν(k,q) from finite-difference first derivatives of the MACE-H-predicted real-space Hamiltonian, saved to a per-structure `epc_pred.h5`.

**Architecture:** New package `maceh/epc/` with pure-function stages: supercell/index math (`supercell.py`), finite-difference derivatives on a fixed graph (`derivative.py`), Bloch sums + generalized eigenproblem (`electron.py`), phonopy interface (`phonon.py`), g contraction + HDF5 writer (`assemble.py`), and an orchestrator (`run.py`) driven by a new CLI `deephe3-epc.py` and `EPCConfig`. Spec: `docs/superpowers/specs/2026-07-13-electron-phonon-coupling-design.md`.

**Tech Stack:** PyTorch + torch_geometric + e3nn (existing model stack), numpy, scipy (`eigh`), h5py, phonopy (new dependency).

## Global Constraints

- Python for all commands: `/opt/anaconda3/envs/DeepH/bin/python` (has torch 2.x, torch_geometric, e3nn, scipy, h5py). Task 1 installs `phonopy` and `pytest` into it.
- Run pytest from the repo root `/Users/jb/MACE-H` so `maceh` is importable: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/... -v`
- Units everywhere: energies eV, lengths Å, masses amu; g comes out in eV.
- H5 hopping keys are `str([Rx, Ry, Rz, i, j])` with **1-based** i, j; all in-memory atom indices are **0-based**.
- Bloch gauge is the cell-phase convention: `M(k) = Σ_R exp(2πi k·R) M_R` (matches the existing Julia/Band.py postprocessing).
- The training pipeline and model code are unchanged. The only edit outside `maceh/epc/`, `maceh/parse_configs.py`, `maceh/default_configs/`, and top-level script/docs is a one-line addition in `maceh/graph.py` (attach `edge_key` to the `data_folder=None` Data).
- Supercell atom ordering is cell-major: supercell index `= cell_lin(p) * n_uc_atoms + i` with `cell_lin(p) = (p1*n2 + p2)*n3 + p3`; home cell p=(0,0,0) atoms come first.
- `import numpy as np`, `import torch` conventions as in the existing codebase; comment style matches the repo (sparse, explanatory only where needed).

---

### Task 1: Supercell math (`maceh/epc/supercell.py`)

**Files:**
- Create: `maceh/epc/__init__.py` (empty file)
- Create: `maceh/epc/supercell.py`
- Test: `tests/test_supercell.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Tasks 2, 3, 8):
  - `Structure` namedtuple: `positions` np.ndarray (N,3) cartesian Å float64, `lattice` np.ndarray (3,3) **rows are lattice vectors**, `numbers` np.ndarray (N,) int atomic numbers.
  - `load_structure(structure_dir: str) -> Structure`
  - `class SupercellMap(n_grid: tuple[int,int,int], n_uc_atoms: int)` with `.n_grid`, `.n_uc_atoms`, `.cells` (list of 3-tuples, cell-major order), `.n_cells`, `.cell_lin(p) -> int`, `.sc_index(i, p) -> int`, `.uc_of(sc_i) -> (i, p_tuple)`
  - `build_supercell(struct: Structure, n_grid) -> (Structure, SupercellMap)`
  - `fold_key(key: list[int], smap: SupercellMap) -> (p: tuple, R: tuple, i: int, j: int)` — key is `[Rx,Ry,Rz,I,J]` 1-based supercell; returns p reduced mod n_grid, R exact in unit-cell lattice units, i/j 0-based unit-cell atoms.
  - `uniform_grid(n_grid) -> np.ndarray (n1*n2*n3, 3)` fractional coordinates, cell-major order.

- [ ] **Step 1: Install test/runtime deps into the DeepH env**

```bash
/opt/anaconda3/envs/DeepH/bin/pip install pytest phonopy
```

Expected: both install without error (phonopy pulls its own pure-python deps).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_supercell.py`:

```python
import numpy as np
import pytest

from maceh.epc.supercell import (Structure, SupercellMap, build_supercell,
                                 fold_key, uniform_grid, load_structure)


def make_uc():
    return Structure(positions=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                     lattice=4.0 * np.eye(3),
                     numbers=np.array([79, 79]))


def test_supercell_map_roundtrip():
    smap = SupercellMap((2, 3, 1), n_uc_atoms=2)
    assert smap.n_cells == 6
    assert smap.cells[0] == (0, 0, 0)
    for i in range(2):
        for p in smap.cells:
            sc = smap.sc_index(i, p)
            assert smap.uc_of(sc) == (i, p)
    # home cell atoms come first
    assert smap.sc_index(0, (0, 0, 0)) == 0
    assert smap.sc_index(1, (0, 0, 0)) == 1


def test_build_supercell():
    sc, smap = build_supercell(make_uc(), (2, 1, 1))
    assert sc.positions.shape == (4, 3)
    assert np.allclose(sc.lattice, np.diag([8.0, 4.0, 4.0]))
    assert np.array_equal(sc.numbers, [79, 79, 79, 79])
    # atom 1 in cell (1,0,0) sits at uc position + a1
    idx = smap.sc_index(1, (1, 0, 0))
    assert np.allclose(sc.positions[idx], [6.0, 2.0, 2.0])


def test_fold_key():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=2)
    # bra atom: uc atom 1 in cell (1,0,0)  -> sc index 3 -> 1-based 4
    # ket atom: uc atom 0 in cell (0,0,0)  -> sc index 0 -> 1-based 1
    # supercell image shift R' = (1, 0, 0)
    p, R, i, j = fold_key([1, 0, 0, 4, 1], smap)
    assert (i, j) == (1, 0)
    # R = p_j + n*R' - p_i = (0 + 2*1 - 1, 0, 0)
    assert R == (1, 0, 0)
    # p = -p_i mod n = (-1) % 2 = 1
    assert p == (1, 0, 0)


def test_uniform_grid():
    g = uniform_grid((2, 1, 2))
    assert g.shape == (4, 3)
    assert np.allclose(g[0], [0, 0, 0])
    assert np.allclose(g[1], [0, 0, 0.5])
    assert np.allclose(g[2], [0.5, 0, 0])


def test_load_structure(tmp_path):
    # files use the DeepH convention: columns are atoms / lattice vectors
    np.savetxt(tmp_path / 'site_positions.dat', np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]))
    np.savetxt(tmp_path / 'element.dat', np.array([79.0, 79.0]))
    np.savetxt(tmp_path / 'lat.dat', 4.0 * np.eye(3))
    s = load_structure(str(tmp_path))
    assert s.positions.shape == (2, 3)
    assert np.allclose(s.positions[1], [2.0, 2.0, 2.0])
    assert np.allclose(s.lattice, 4.0 * np.eye(3))
    assert np.array_equal(s.numbers, [79, 79])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_supercell.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc'`

- [ ] **Step 4: Write the implementation**

Create empty `maceh/epc/__init__.py`, then `maceh/epc/supercell.py`:

```python
import os
import itertools
from collections import namedtuple

import numpy as np

# positions: (N, 3) cartesian Angstrom; lattice: (3, 3) rows are lattice vectors;
# numbers: (N,) atomic numbers
Structure = namedtuple('Structure', ['positions', 'lattice', 'numbers'])


def load_structure(structure_dir):
    positions = np.loadtxt(os.path.join(structure_dir, 'site_positions.dat')).T
    numbers = np.loadtxt(os.path.join(structure_dir, 'element.dat'))
    lattice = np.loadtxt(os.path.join(structure_dir, 'lat.dat')).T
    if numbers.ndim == 0:
        numbers = numbers[None]
        positions = positions[None, :]
    return Structure(positions.astype(np.float64), lattice.astype(np.float64),
                     numbers.astype(int))


class SupercellMap:
    r''' cell-major ordering: supercell atom index = cell_lin(p) * n_uc_atoms + i '''

    def __init__(self, n_grid, n_uc_atoms):
        self.n_grid = tuple(int(n) for n in n_grid)
        self.n_uc_atoms = int(n_uc_atoms)
        self.cells = list(itertools.product(range(self.n_grid[0]),
                                            range(self.n_grid[1]),
                                            range(self.n_grid[2])))

    @property
    def n_cells(self):
        return len(self.cells)

    def cell_lin(self, p):
        return (p[0] * self.n_grid[1] + p[1]) * self.n_grid[2] + p[2]

    def sc_index(self, i, p):
        return self.cell_lin(p) * self.n_uc_atoms + i

    def uc_of(self, sc_i):
        return sc_i % self.n_uc_atoms, self.cells[sc_i // self.n_uc_atoms]


def build_supercell(struct, n_grid):
    smap = SupercellMap(n_grid, len(struct.numbers))
    sc_lattice = struct.lattice * np.array(smap.n_grid, dtype=np.float64)[:, None]
    positions, numbers = [], []
    for p in smap.cells:
        shift = np.array(p, dtype=np.float64) @ struct.lattice
        positions.append(struct.positions + shift)
        numbers.append(struct.numbers)
    return Structure(np.concatenate(positions), sc_lattice, np.concatenate(numbers)), smap


def fold_key(key, smap):
    r''' fold supercell hopping key [Rx, Ry, Rz, I, J] (I, J 1-based; R in supercell
    lattice units) into unit-cell labels, for displacement of a home-cell atom.
    Returns (p, R, i, j): p = cell of the displaced atom relative to the bra atom's
    cell (reduced mod n_grid, exact for q commensurate with the grid); R = bra->ket
    offset in unit-cell lattice units; i, j = 0-based unit-cell atom indices '''
    n = smap.n_grid
    i, p_i = smap.uc_of(key[3] - 1)
    j, p_j = smap.uc_of(key[4] - 1)
    R = tuple(p_j[a] + n[a] * key[a] - p_i[a] for a in range(3))
    p = tuple((-p_i[a]) % n[a] for a in range(3))
    return p, R, i, j


def uniform_grid(n_grid):
    return np.array([[p[0] / n_grid[0], p[1] / n_grid[1], p[2] / n_grid[2]]
                     for p in itertools.product(range(n_grid[0]), range(n_grid[1]),
                                                range(n_grid[2]))])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_supercell.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add maceh/epc/__init__.py maceh/epc/supercell.py tests/test_supercell.py
git commit -m "feat(epc): supercell construction and index folding for EPC derivatives"
```

---

### Task 2: Supercell graph construction (`maceh/epc/derivative.py`, part 1)

**Files:**
- Modify: `maceh/graph.py:391-393` (attach `edge_key` in the `data_folder=None` branch)
- Create: `maceh/epc/derivative.py`
- Test: `tests/test_derivative_graph.py`

**Interfaces:**
- Consumes: `Structure` from Task 1; `maceh.graph.get_graph`, `maceh.graph.get_edge_fea` (existing).
- Produces (used by Task 8):
  - `build_supercell_graph(struct: Structure, radius: float, default_dtype_torch) -> torch_geometric.data.Data` with fields `x` (atomic numbers, int64), `edge_index`, `edge_attr` = `[dist, dx, dy, dz]`, `edge_key` (`[Rx,Ry,Rz,i,j]` 1-based), `pos`, `lattice` (shape (1,3,3)). Raises `AssertionError` if the recomputed `edge_attr` does not match graph construction (the spec's startup self-check).

- [ ] **Step 1: Write the failing test**

Create `tests/test_derivative_graph.py`:

```python
import numpy as np
import torch
import pytest

from maceh.epc.supercell import Structure
from maceh.epc.derivative import build_supercell_graph
from maceh.graph import get_edge_fea


def test_build_supercell_graph():
    torch.set_default_dtype(torch.float64)
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                       lattice=4.0 * np.eye(3),
                       numbers=np.array([79, 79]))
    data = build_supercell_graph(struct, radius=4.5, default_dtype_torch=torch.float64)

    assert data.x.dtype == torch.int64
    assert torch.equal(data.x, torch.tensor([79, 79]))
    assert data.edge_key.shape[1] == 5
    assert data.edge_key.shape[0] == data.edge_index.shape[1] == data.edge_attr.shape[0]
    # keys are 1-based
    assert data.edge_key[:, 3].min() >= 1 and data.edge_key[:, 4].min() >= 1
    # the self-check inside build_supercell_graph already asserted consistency;
    # verify the recomputation contract explicitly too
    recomputed = get_edge_fea(data.pos, data.lattice[0], torch.float64, data.edge_key)
    assert torch.allclose(recomputed, data.edge_attr, atol=1e-10)
    # directed graph: for every (i->j, R) there is (j->i, -R)
    keys = set(map(tuple, data.edge_key.tolist()))
    for (r1, r2, r3, i, j) in keys:
        assert (-r1, -r2, -r3, j, i) in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_derivative_graph.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_supercell_graph'`

- [ ] **Step 3: Modify `maceh/graph.py` to attach edge_key**

In `maceh/graph.py`, the final `else` branch of `get_graph` (around line 391) currently reads:

```python
    else:
        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id,
                    pos=cart_coords.type(default_dtype_torch), lattice=lattice.unsqueeze(0), **kwargs)
```

Change it to (both `create_from_DFT` branches define `edge_key`, so this is always available):

```python
    else:
        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id,
                    pos=cart_coords.type(default_dtype_torch), lattice=lattice.unsqueeze(0),
                    edge_key=edge_key, **kwargs)
```

- [ ] **Step 4: Write `build_supercell_graph`**

Create `maceh/epc/derivative.py`:

```python
import json
from dataclasses import dataclass

import numpy as np
import torch

from ..graph import get_graph, get_edge_fea
from .supercell import fold_key


def build_supercell_graph(struct, radius, default_dtype_torch):
    r''' radius-based graph for an in-memory structure. Returns a Data object with
    x = atomic numbers, edge_attr = [dist, dx, dy, dz] and edge_key = [Rx,Ry,Rz,i,j]
    (1-based). Asserts that edge_attr can be recomputed exactly from positions and
    edge_key -- the invariant the finite-difference driver relies on. '''
    assert radius > 0, 'EPC graph construction requires an explicit cutoff radius'
    lattice = torch.tensor(struct.lattice, dtype=default_dtype_torch)
    cart_coords = torch.tensor(struct.positions, dtype=default_dtype_torch)
    frac_coords = cart_coords @ torch.linalg.inv(lattice)
    numbers = torch.tensor(struct.numbers, dtype=torch.int64)
    data = get_graph(cart_coords, frac_coords, numbers, stru_id='epc_supercell',
                     r=radius, max_num_nbr=0, edge_Aij=False, lattice=lattice,
                     default_dtype_torch=default_dtype_torch, data_folder=None,
                     target_file_name='overlaps.h5', inference=True, only_ij=False,
                     create_from_DFT=False)
    recomputed = get_edge_fea(data.pos, data.lattice[0], default_dtype_torch, data.edge_key)
    assert torch.allclose(recomputed, data.edge_attr, atol=1e-7), \
        'edge_attr recomputed from positions does not match graph construction'
    return data
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_derivative_graph.py tests/test_supercell.py -v`
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add maceh/graph.py maceh/epc/derivative.py tests/test_derivative_graph.py
git commit -m "feat(epc): radius-based supercell graph with edge_attr recomputation invariant"
```

---

### Task 3: Finite-difference driver (`maceh/epc/derivative.py`, part 2)

**Files:**
- Modify: `maceh/epc/derivative.py`
- Test: `tests/test_derivative_fd.py`

**Interfaces:**
- Consumes: `SupercellMap`, `fold_key` from Task 1.
- Produces (used by Tasks 6, 8):
  - `@dataclass DerivativeData`: fields `n_grid: tuple`, `n_uc_atoms: int`, `delta: float`, `norb_cumsum: np.ndarray` ((n_uc_atoms+1,)), `blocks: dict` mapping `(kappa, alpha)` → `{(p_tuple, R_tuple): np.ndarray (norb_tot, norb_tot)}`; property `norb_tot`.
  - `finite_difference(predict_fn, positions0: torch.Tensor, smap, norb_cumsum, delta, atom_indices=None, grad_threshold=1e-10) -> DerivativeData` where `predict_fn(positions) -> {str([Rx,Ry,Rz,I,J]): np.ndarray}` predicts supercell hopping blocks at given positions on a **fixed** graph.
  - `acoustic_sum_rule(deriv: DerivativeData) -> float` — max |Σ_{κ,p} dH| over α and R.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_derivative_fd.py`. The stub model returns, for each hopping key, a 1×1 block equal to the edge length — its exact derivative w.r.t. any atom position is known analytically, so this validates the FD arithmetic, the fold-back wiring, and thresholding at once. Chain: 1 atom per cell along x, supercell 2×1×1.

```python
import numpy as np
import torch
import pytest

from maceh.epc.supercell import SupercellMap
from maceh.epc.derivative import DerivativeData, finite_difference, acoustic_sum_rule

A = 3.0  # unit-cell lattice constant along x
L_SC = np.diag([2 * A, 10.0, 10.0])  # 2x1x1 supercell lattice (rows)
# supercell hopping keys [R'x, R'y, R'z, I, J], 1-based, directed, incl. onsite
KEYS = [[0, 0, 0, 1, 1], [0, 0, 0, 2, 2],
        [0, 0, 0, 1, 2], [0, 0, 0, 2, 1],
        [-1, 0, 0, 1, 2], [1, 0, 0, 2, 1],
        [1, 0, 0, 1, 2], [-1, 0, 0, 2, 1],
        [1, 0, 0, 1, 1], [-1, 0, 0, 1, 1],
        [1, 0, 0, 2, 2], [-1, 0, 0, 2, 2]]


def edge_vec(pos, key):
    R = np.array(key[:3], dtype=np.float64)
    return pos[key[4] - 1] + R @ L_SC - pos[key[3] - 1]


def predict_fn(positions):
    pos = positions.detach().numpy()
    return {str(k): np.array([[np.linalg.norm(edge_vec(pos, k))]]) for k in KEYS}


def analytic_deriv(pos, key, kappa_sc, alpha):
    # d|v|/d tau_{kappa,alpha} where v = r_J + R'.L - r_I; displacing a supercell
    # atom moves it in ALL supercell images simultaneously
    v = edge_vec(pos, key)
    n = np.linalg.norm(v)
    d = 0.0
    if key[4] - 1 == kappa_sc:
        d += v[alpha] / n
    if key[3] - 1 == kappa_sc:
        d -= v[alpha] / n
    return d


def test_finite_difference_matches_analytic():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    norb_cumsum = np.array([0, 1])
    deriv = finite_difference(predict_fn, pos0, smap, norb_cumsum, delta=1e-4,
                              grad_threshold=1e-12)
    assert deriv.n_uc_atoms == 1 and deriv.norb_tot == 1
    # displacing uc atom 0 displaces supercell atom 0 (home cell)
    pos_np = pos0.numpy()
    for alpha in range(3):
        found = deriv.blocks[(0, alpha)]
        for key in KEYS:
            expected = analytic_deriv(pos_np, key, kappa_sc=0, alpha=alpha)
            from maceh.epc.supercell import fold_key
            p, R, i, j = fold_key(key, smap)
            got = found.get((p, R), np.zeros((1, 1)))[i, j]
            assert got == pytest.approx(expected, abs=1e-6), (key, alpha)


def test_grad_threshold_drops_far_blocks():
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    deriv = finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=1e-4,
                              grad_threshold=1e30)
    assert all(len(v) == 0 for v in deriv.blocks.values())


def test_acoustic_sum_rule_zero_for_translation_invariant_model():
    # the stub depends only on relative positions, so the sum rule is exact
    smap = SupercellMap((2, 1, 1), n_uc_atoms=1)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0]], dtype=torch.float64)
    deriv = finite_difference(predict_fn, pos0, smap, np.array([0, 1]), delta=1e-4,
                              grad_threshold=1e-12)
    assert acoustic_sum_rule(deriv) < 1e-6
```

Note: with 1 atom per unit cell, displacing uc atom 0 moves only supercell atom 0 (index 0 = home cell); the atom in cell (1,0,0) is supercell atom 1 and stays put. `analytic_deriv` handles edges where both endpoints are atom 0 (contributions cancel).

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_derivative_fd.py -v`
Expected: FAIL with `ImportError: cannot import name 'DerivativeData'`

- [ ] **Step 3: Implement finite_difference in `maceh/epc/derivative.py`**

Append to `maceh/epc/derivative.py`:

```python
@dataclass
class DerivativeData:
    r''' real-space Hamiltonian derivatives dH_ij(R)/d tau_{kappa,alpha}(p).
    blocks[(kappa, alpha)][(p, R)] is a dense (norb_tot, norb_tot) unit-cell matrix;
    p labels the cell of the displaced atom relative to the bra atom's cell,
    R the bra->ket cell offset (both in unit-cell lattice units). Units: eV / Angstrom. '''
    n_grid: tuple
    n_uc_atoms: int
    delta: float
    norb_cumsum: np.ndarray
    blocks: dict

    @property
    def norb_tot(self):
        return int(self.norb_cumsum[-1])


def finite_difference(predict_fn, positions0, smap, norb_cumsum, delta,
                      atom_indices=None, grad_threshold=1e-10):
    r''' central finite differences of predicted hopping blocks w.r.t. displacements
    of home-cell atoms, folded back to unit-cell labels via fold_key '''
    if atom_indices is None:
        atom_indices = list(range(smap.n_uc_atoms))
    norb_cumsum = np.asarray(norb_cumsum)
    norb_tot = int(norb_cumsum[-1])
    blocks = {}
    for kappa in atom_indices:
        for alpha in range(3):
            pos_plus = positions0.clone()
            pos_plus[kappa, alpha] += delta
            pos_minus = positions0.clone()
            pos_minus[kappa, alpha] -= delta
            H_plus = predict_fn(pos_plus)
            H_minus = predict_fn(pos_minus)
            assert H_plus.keys() == H_minus.keys()
            out = {}
            for key_str, hp in H_plus.items():
                d = (np.asarray(hp) - np.asarray(H_minus[key_str])) / (2.0 * delta)
                if np.abs(d).max() < grad_threshold:
                    continue
                p, R, i, j = fold_key(json.loads(key_str), smap)
                if (p, R) not in out:
                    dtype = np.complex128 if np.iscomplexobj(d) else np.float64
                    out[(p, R)] = np.zeros((norb_tot, norb_tot), dtype=dtype)
                out[(p, R)][norb_cumsum[i]:norb_cumsum[i + 1],
                            norb_cumsum[j]:norb_cumsum[j + 1]] = d
            blocks[(kappa, alpha)] = out
    return DerivativeData(n_grid=smap.n_grid, n_uc_atoms=smap.n_uc_atoms, delta=delta,
                          norb_cumsum=norb_cumsum, blocks=blocks)


def acoustic_sum_rule(deriv):
    r''' max over alpha and R of |sum_{kappa, p} dH(R)|; should vanish for a
    translation-invariant model. Only meaningful when all atoms were displaced. '''
    worst = 0.0
    for alpha in range(3):
        acc = {}
        for kappa in range(deriv.n_uc_atoms):
            for (p, R), dense in deriv.blocks.get((kappa, alpha), {}).items():
                acc[R] = acc.get(R, 0) + dense
        for m in acc.values():
            worst = max(worst, float(np.abs(m).max()))
    return worst
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_derivative_fd.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add maceh/epc/derivative.py tests/test_derivative_fd.py
git commit -m "feat(epc): finite-difference Hamiltonian derivatives with fold-back and sum-rule diagnostic"
```

---

### Task 4: Electronic structure (`maceh/epc/electron.py`)

**Files:**
- Create: `maceh/epc/electron.py`
- Test: `tests/test_electron.py`

**Interfaces:**
- Consumes: `maceh.graph.load_orbital_types` (existing).
- Produces (used by Tasks 6, 8):
  - `load_orbital_slices(orbital_types_path, spinful=False) -> (norb_list: list[int], norb_cumsum: np.ndarray)` (counts doubled when spinful)
  - `load_blocks_h5(path, force_hermiticity=True) -> {str_key: np.ndarray}`
  - `blocks_to_dense(blocks, norb_cumsum, spin_expand=False) -> {R_tuple: np.ndarray (norb_tot, norb_tot) complex128}` — `spin_expand=True` block-diagonally doubles each block (for spin-independent overlaps in spinful mode)
  - `bloch_sum(dense_by_R, k_frac) -> np.ndarray` — `Σ_R exp(2πi k·R) M_R`
  - `class ElectronSolver(H_by_R, S_by_R)` with `.solve(k_frac) -> (eps: np.ndarray, C: np.ndarray)` (generalized `scipy.linalg.eigh`, cached per k mod 1)
  - `band_window(solver, k_list, fermi_energy, half_width) -> (b_lo: int, b_hi: int)` — global band index range covering all states inside the window at any listed k; raises AssertionError if empty

- [ ] **Step 1: Write the failing tests**

Create `tests/test_electron.py` (1-orbital 1D chain, ε(k) = −2cos(2πk)):

```python
import json

import numpy as np
import h5py
import pytest

from maceh.epc.electron import (load_orbital_slices, load_blocks_h5, blocks_to_dense,
                                bloch_sum, ElectronSolver, band_window)


def chain_blocks():
    return {'[0, 0, 0, 1, 1]': np.array([[0.0]]),
            '[1, 0, 0, 1, 1]': np.array([[-1.0]]),
            '[-1, 0, 0, 1, 1]': np.array([[-1.0]])}


def overlap_blocks():
    return {'[0, 0, 0, 1, 1]': np.array([[1.0]]),
            '[1, 0, 0, 1, 1]': np.array([[0.0]]),
            '[-1, 0, 0, 1, 1]': np.array([[0.0]])}


def test_load_orbital_slices(tmp_path):
    # two atoms: s+p (1+3 orbitals) and s (1 orbital)
    (tmp_path / 'orbital_types.dat').write_text('0 1\n0\n')
    norb, cumsum = load_orbital_slices(str(tmp_path / 'orbital_types.dat'))
    assert norb == [4, 1]
    assert list(cumsum) == [0, 4, 5]
    norb2, cumsum2 = load_orbital_slices(str(tmp_path / 'orbital_types.dat'), spinful=True)
    assert norb2 == [8, 2]


def test_load_blocks_h5_hermitizes(tmp_path):
    path = str(tmp_path / 'h.h5')
    with h5py.File(path, 'w') as f:
        f['[0, 0, 0, 1, 1]'] = np.array([[1.0]])
        f['[1, 0, 0, 1, 1]'] = np.array([[-1.2]])
        f['[-1, 0, 0, 1, 1]'] = np.array([[-0.8]])
    blocks = load_blocks_h5(path)
    assert blocks['[1, 0, 0, 1, 1]'][0, 0] == pytest.approx(-1.0)
    assert blocks['[-1, 0, 0, 1, 1]'][0, 0] == pytest.approx(-1.0)


def test_bloch_sum_and_solver():
    H = blocks_to_dense(chain_blocks(), np.array([0, 1]))
    S = blocks_to_dense(overlap_blocks(), np.array([0, 1]))
    assert bloch_sum(H, [0.0, 0.0, 0.0])[0, 0] == pytest.approx(-2.0)
    assert bloch_sum(H, [0.5, 0.0, 0.0])[0, 0] == pytest.approx(2.0)
    solver = ElectronSolver(H, S)
    eps, C = solver.solve([0.25, 0.0, 0.0])
    assert eps[0] == pytest.approx(-2.0 * np.cos(2 * np.pi * 0.25), abs=1e-12)
    assert abs(C[0, 0]) == pytest.approx(1.0)
    # cache respects periodicity in k
    eps2, _ = solver.solve([1.25, 0.0, 0.0])
    assert eps2[0] == eps[0]


def test_spin_expand():
    S = blocks_to_dense(overlap_blocks(), np.array([0, 2]), spin_expand=True)
    assert S[(0, 0, 0)].shape == (2, 2)
    assert np.allclose(S[(0, 0, 0)], np.eye(2))


def test_band_window():
    H = blocks_to_dense(chain_blocks(), np.array([0, 1]))
    S = blocks_to_dense(overlap_blocks(), np.array([0, 1]))
    solver = ElectronSolver(H, S)
    ks = [[k, 0, 0] for k in np.linspace(0, 0.5, 6)]
    b_lo, b_hi = band_window(solver, ks, fermi_energy=0.0, half_width=3.0)
    assert (b_lo, b_hi) == (0, 1)
    with pytest.raises(AssertionError):
        band_window(solver, ks, fermi_energy=100.0, half_width=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_electron.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc.electron'`

- [ ] **Step 3: Implement `maceh/epc/electron.py`**

```python
import json

import numpy as np
import h5py
from scipy.linalg import eigh

from ..graph import load_orbital_types


def load_orbital_slices(orbital_types_path, spinful=False):
    norb_list = load_orbital_types(orbital_types_path)
    if spinful:
        norb_list = [2 * n for n in norb_list]
    return norb_list, np.concatenate([[0], np.cumsum(norb_list)])


def load_blocks_h5(path, force_hermiticity=True):
    with h5py.File(path, 'r') as f:
        blocks = {k: np.array(v) for k, v in f.items()}
    if force_hermiticity:
        out = {}
        for k, v in blocks.items():
            key = json.loads(k)
            k_adj = str([-key[0], -key[1], -key[2], key[4], key[3]])
            out[k] = (v + blocks[k_adj].conj().T) / 2.0
        blocks = out
    return blocks


def blocks_to_dense(blocks, norb_cumsum, spin_expand=False):
    norb_cumsum = np.asarray(norb_cumsum)
    norb_tot = int(norb_cumsum[-1])
    dense_by_R = {}
    for key_str, b in blocks.items():
        key = json.loads(key_str)
        R, i, j = tuple(key[:3]), key[3] - 1, key[4] - 1
        if spin_expand:
            b = np.block([[b, np.zeros_like(b)], [np.zeros_like(b), b]])
        if R not in dense_by_R:
            dense_by_R[R] = np.zeros((norb_tot, norb_tot), dtype=np.complex128)
        dense_by_R[R][norb_cumsum[i]:norb_cumsum[i + 1],
                      norb_cumsum[j]:norb_cumsum[j + 1]] = b
    return dense_by_R


def bloch_sum(dense_by_R, k_frac):
    r''' M(k) = sum_R exp(2 pi i k.R) M_R (cell-phase gauge, matching the Julia
    band-structure postprocessing) '''
    k = np.asarray(k_frac, dtype=np.float64)
    out = 0
    for R, m in dense_by_R.items():
        out = out + np.exp(2j * np.pi * (k @ np.asarray(R, dtype=np.float64))) * m
    return out


class ElectronSolver:

    def __init__(self, H_by_R, S_by_R):
        self.H_by_R = H_by_R
        self.S_by_R = S_by_R
        self._cache = {}

    def solve(self, k_frac):
        key = tuple(np.round(np.asarray(k_frac, dtype=np.float64) % 1.0, 8))
        if key not in self._cache:
            Hk = bloch_sum(self.H_by_R, k_frac)
            Sk = bloch_sum(self.S_by_R, k_frac)
            Hk = (Hk + Hk.conj().T) / 2.0
            Sk = (Sk + Sk.conj().T) / 2.0
            self._cache[key] = eigh(Hk, Sk)
        return self._cache[key]


def band_window(solver, k_list, fermi_energy, half_width):
    b_lo, b_hi = None, None
    for k in k_list:
        eps, _ = solver.solve(k)
        idx = np.where(np.abs(eps - fermi_energy) <= half_width)[0]
        if idx.size == 0:
            continue
        b_lo = int(idx[0]) if b_lo is None else min(b_lo, int(idx[0]))
        b_hi = int(idx[-1]) + 1 if b_hi is None else max(b_hi, int(idx[-1]) + 1)
    assert b_lo is not None, 'no electronic states inside the energy window'
    return b_lo, b_hi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_electron.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add maceh/epc/electron.py tests/test_electron.py
git commit -m "feat(epc): Bloch sums, generalized eigensolver and Fermi-window selection"
```

---

### Task 5: Phonon interface (`maceh/epc/phonon.py`)

**Files:**
- Create: `maceh/epc/phonon.py`
- Test: `tests/test_phonon.py`

**Interfaces:**
- Consumes: phonopy (installed in Task 1).
- Produces (used by Tasks 6, 8):
  - `THZ_TO_EV = 4.135667696e-3`
  - `class PhononData(phonopy_obj)` with `.natoms`, `.masses` (np (N,), amu), `.frac_coords` (np (N,3)), `.modes(q_frac) -> (omega_ev: np (3N,), evec: np (N, 3, 3N) complex)`; classmethod `.from_directory(phonopy_dir)` loading `phonopy.yaml` (+ `FORCE_CONSTANTS` if present). Negative `omega_ev` marks imaginary modes. Eigenvectors are converted from phonopy's atom-position phase convention to the cell-phase gauge: `e_cell(κ) = e_phonopy(κ) · exp(2πi q·τ_frac,κ)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phonon.py`. A 1-atom cell with diagonal force constants `Φ = c·I` and identity supercell gives three degenerate modes with `ν_THz = VaspToTHz · sqrt(c/m)`; the gauge conversion is tested by placing the atom off-origin:

```python
import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToTHz

from maceh.epc.phonon import PhononData, THZ_TO_EV


def make_phonon(scaled_position):
    cell = PhonopyAtoms(symbols=['Au'], cell=4.0 * np.eye(3),
                        scaled_positions=[scaled_position])
    ph = Phonopy(cell, supercell_matrix=np.eye(3, dtype=int))
    c = 5.0  # eV / Angstrom^2
    ph.force_constants = c * np.eye(3)[None, None, :, :]
    return ph, c


def test_frequencies():
    ph, c = make_phonon([0.0, 0.0, 0.0])
    data = PhononData(ph)
    assert data.natoms == 1
    mass = data.masses[0]
    omega, evec = data.modes([0.0, 0.0, 0.0])
    expected = VaspToTHz * np.sqrt(c / mass) * THZ_TO_EV
    assert np.allclose(omega, expected, rtol=1e-6)
    assert evec.shape == (1, 3, 3)


def test_gauge_conversion():
    # same dynamics, atom moved off-origin: cell-phase eigenvector must carry
    # the extra factor exp(2 pi i q . tau)
    q = [0.5, 0.0, 0.0]
    tau = [0.25, 0.0, 0.0]
    ph, _ = make_phonon(tau)
    data = PhononData(ph)
    _, evec_cell = data.modes(q)
    freqs, evec_raw = ph.get_frequencies_with_eigenvectors(q)
    expected = evec_raw.reshape(1, 3, 3) * np.exp(2j * np.pi * np.dot(tau, q))
    assert np.allclose(evec_cell, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_phonon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc.phonon'`

- [ ] **Step 3: Implement `maceh/epc/phonon.py`**

```python
import os

import numpy as np

THZ_TO_EV = 4.135667696e-3  # h * 1 THz in eV


class PhononData:
    r''' phonon frequencies and eigenvectors from phonopy, converted to the
    cell-phase gauge used by the electronic Bloch sums '''

    def __init__(self, phonopy_obj):
        self._ph = phonopy_obj
        prim = phonopy_obj.primitive
        self.masses = np.asarray(prim.masses, dtype=np.float64)  # amu
        self.frac_coords = np.asarray(prim.scaled_positions, dtype=np.float64)
        self.natoms = len(self.masses)

    @classmethod
    def from_directory(cls, phonopy_dir):
        import phonopy
        kwargs = {}
        fc_path = os.path.join(phonopy_dir, 'FORCE_CONSTANTS')
        if os.path.isfile(fc_path):
            kwargs['force_constants_filename'] = fc_path
        return cls(phonopy.load(os.path.join(phonopy_dir, 'phonopy.yaml'), **kwargs))

    def modes(self, q_frac):
        r''' returns (omega_ev (3N,), evec (N, 3, 3N) complex) at fractional q.
        Negative omega marks imaginary (soft) modes. Phonopy's dynamical matrix
        uses atom-position phases exp(iq.r(l kappa)); the electronic side uses
        cell phases exp(iq.p), so eigenvectors are converted with
        e_cell(kappa) = e_phonopy(kappa) * exp(2 pi i q . tau_kappa). '''
        q = np.asarray(q_frac, dtype=np.float64)
        freqs_thz, evecs = self._ph.get_frequencies_with_eigenvectors(q)
        omega_ev = np.asarray(freqs_thz) * THZ_TO_EV
        phase = np.exp(2j * np.pi * (self.frac_coords @ q))
        evec = evecs.reshape(self.natoms, 3, 3 * self.natoms) * phase[:, None, None]
        return omega_ev, evec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_phonon.py -v`
Expected: 2 PASSED. If `test_gauge_conversion` fails on the phase factor sign/presence, check the installed phonopy's dynamical-matrix convention (`phonopy/harmonic/dynamical_matrix.py`) and adjust the conversion so that `e_cell = e_phonopy * exp(+2πi q·τ)` holds for the *atom-position* convention; if the installed phonopy already uses cell phases (some builds expose `dynamical_matrix_decimals`/'auto' conventions), the conversion factor must be dropped — update both code and test to the verified convention and record it in the docstring.

- [ ] **Step 5: Commit**

```bash
git add maceh/epc/phonon.py tests/test_phonon.py
git commit -m "feat(epc): phonopy interface with cell-phase gauge conversion"
```

---

### Task 6: EPC assembly and output (`maceh/epc/assemble.py`)

**Files:**
- Create: `maceh/epc/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `DerivativeData` (Task 3), `ElectronSolver`, `bloch_sum`, `band_window` (Task 4), a phonon object duck-typing `PhononData` (Task 5: `.natoms`, `.masses`, `.modes(q)`).
- Produces (used by Task 8):
  - `HBAR_JS`, `AMU_KG`, `EV_J` constants; `zero_point_length(mass_amu, omega_ev) -> float` (Å)
  - `compute_epc(deriv, solver, phonons, kpts, qpts, fermi_energy, energy_window, omega_tol=1e-5) -> dict` with keys `g` (complex128, (nq, 3N, nk, nb, nb)), `omega_q` ((nq, 3N)), `soft_mode_mask` (bool (nq, 3N)), `eps_k` ((nk, nb)), `eps_kq` ((nq, nk, nb)), `band_range` ((2,) = [b_lo, b_hi)), `kpts`, `qpts`
  - `write_epc_h5(path, results: dict, attrs: dict)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assemble.py` (1 atom, 1 orbital, S = 1, flat band at 0; dH constant per direction; stub phonons with ω = 1 eV and cartesian eigenvectors — every factor of g is then hand-computable):

```python
import numpy as np
import h5py
import pytest

from maceh.epc.derivative import DerivativeData
from maceh.epc.electron import ElectronSolver
from maceh.epc.assemble import (compute_epc, write_epc_h5, zero_point_length,
                                HBAR_JS, AMU_KG, EV_J)


class StubPhonons:
    natoms = 1
    masses = np.array([2.0])  # amu

    def __init__(self, omega=1.0):
        self.omega = omega

    def modes(self, q_frac):
        evec = np.zeros((1, 3, 3), dtype=complex)
        evec[0, :, :] = np.eye(3)  # mode nu polarized along axis nu
        return np.full(3, self.omega), evec


def make_inputs():
    solver = ElectronSolver({(0, 0, 0): np.array([[0.0]], dtype=complex)},
                            {(0, 0, 0): np.array([[1.0]], dtype=complex)})
    d = [1.0, 2.0, 3.0]
    blocks = {(0, a): {((0, 0, 0), (0, 0, 0)): np.array([[d[a]]])} for a in range(3)}
    deriv = DerivativeData(n_grid=(1, 1, 1), n_uc_atoms=1, delta=0.01,
                           norb_cumsum=np.array([0, 1]), blocks=blocks)
    return deriv, solver, d


def test_zero_point_length():
    l = zero_point_length(2.0, 1.0)
    expected = HBAR_JS / np.sqrt(2.0 * 2.0 * AMU_KG * 1.0 * EV_J) * 1e10
    assert l == pytest.approx(expected)
    assert 0.01 < l < 1.0  # sanity: sub-Angstrom zero-point length


def test_compute_epc_single_site():
    deriv, solver, d = make_inputs()
    res = compute_epc(deriv, solver, StubPhonons(), kpts=[[0, 0, 0]], qpts=[[0, 0, 0]],
                      fermi_energy=0.0, energy_window=1.0)
    assert res['g'].shape == (1, 3, 1, 1, 1)
    l = zero_point_length(2.0, 1.0)
    for nu in range(3):
        assert res['g'][0, nu, 0, 0, 0] == pytest.approx(l * d[nu])
    assert res['band_range'].tolist() == [0, 1]
    assert not res['soft_mode_mask'].any()
    assert res['eps_k'][0, 0] == pytest.approx(0.0)


def test_soft_modes_are_zeroed():
    deriv, solver, _ = make_inputs()
    res = compute_epc(deriv, solver, StubPhonons(omega=0.0), kpts=[[0, 0, 0]],
                      qpts=[[0, 0, 0]], fermi_energy=0.0, energy_window=1.0)
    assert res['soft_mode_mask'].all()
    assert np.all(res['g'] == 0)


def test_write_epc_h5(tmp_path):
    deriv, solver, _ = make_inputs()
    res = compute_epc(deriv, solver, StubPhonons(), kpts=[[0, 0, 0]], qpts=[[0, 0, 0]],
                      fermi_energy=0.0, energy_window=1.0)
    path = str(tmp_path / 'epc_pred.h5')
    write_epc_h5(path, res, {'fermi_energy': 0.0, 'delta': 0.01})
    with h5py.File(path, 'r') as f:
        assert f['g'].shape == (1, 3, 1, 1, 1)
        assert f.attrs['fermi_energy'] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc.assemble'`

- [ ] **Step 3: Implement `maceh/epc/assemble.py`**

```python
import numpy as np
import h5py

from .electron import bloch_sum, band_window

HBAR_JS = 1.054571817e-34
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19


def zero_point_length(mass_amu, omega_ev):
    r''' sqrt(hbar / (2 M omega)) in Angstrom, with M in amu and hbar*omega in eV '''
    return HBAR_JS / np.sqrt(2.0 * mass_amu * AMU_KG * omega_ev * EV_J) * 1e10


def compute_epc(deriv, solver, phonons, kpts, qpts, fermi_energy, energy_window,
                omega_tol=1e-5):
    r''' g_{m n nu}(k, q) = sum_{kappa alpha} sqrt(hbar / 2 M_kappa omega_{q nu})
    e_{kappa alpha, nu}(q) <psi_{m,k+q}| dH/d tau_{kappa alpha}(q) |psi_{n,k}>,
    with dH(k, q) = sum_p e^{2 pi i q.p} sum_R e^{2 pi i k.R} dH(p, R).
    Modes with omega <= omega_tol (eV) are masked and their g set to zero. '''
    kpts = [np.asarray(k, dtype=np.float64) for k in kpts]
    qpts = [np.asarray(q, dtype=np.float64) for q in qpts]
    all_k = kpts + [k + q for k in kpts for q in qpts]
    b_lo, b_hi = band_window(solver, all_k, fermi_energy, energy_window)
    nb = b_hi - b_lo
    nk, nq, nmodes = len(kpts), len(qpts), 3 * phonons.natoms

    g = np.zeros((nq, nmodes, nk, nb, nb), dtype=np.complex128)
    omega_q = np.zeros((nq, nmodes))
    soft_mode_mask = np.zeros((nq, nmodes), dtype=bool)
    eps_k = np.zeros((nk, nb))
    eps_kq = np.zeros((nq, nk, nb))

    for iq, q in enumerate(qpts):
        omega, evec = phonons.modes(q)
        omega_q[iq] = omega
        soft_mode_mask[iq] = omega <= omega_tol
        pref = np.zeros((phonons.natoms, nmodes))
        live = ~soft_mode_mask[iq]
        for kappa in range(phonons.natoms):
            pref[kappa, live] = zero_point_length(phonons.masses[kappa], omega[live])
        # sum over displaced-atom cells p first (q-dependent, k-independent)
        dHq = {}
        for (kappa, alpha), by_pR in deriv.blocks.items():
            acc = {}
            for (p, R), m in by_pR.items():
                phase = np.exp(2j * np.pi * (q @ np.asarray(p, dtype=np.float64)))
                acc[R] = acc.get(R, 0) + phase * m
            dHq[(kappa, alpha)] = acc
        for ik, k in enumerate(kpts):
            eps_n, C_n = solver.solve(k)
            eps_m, C_m = solver.solve(k + q)
            eps_k[ik] = eps_n[b_lo:b_hi]
            eps_kq[iq, ik] = eps_m[b_lo:b_hi]
            Cn = C_n[:, b_lo:b_hi]
            Cm = C_m[:, b_lo:b_hi]
            for (kappa, alpha), by_R in dHq.items():
                M = Cm.conj().T @ bloch_sum(by_R, k) @ Cn
                weight = (pref[kappa] * evec[kappa, alpha]).reshape(-1, 1, 1)
                g[iq, :, ik] += weight * M[None, :, :]

    return dict(g=g, omega_q=omega_q, soft_mode_mask=soft_mode_mask, eps_k=eps_k,
                eps_kq=eps_kq, band_range=np.array([b_lo, b_hi]),
                kpts=np.asarray(kpts), qpts=np.asarray(qpts))


def write_epc_h5(path, results, attrs):
    with h5py.File(path, 'w') as f:
        for name, arr in results.items():
            f[name] = arr
        for k, v in attrs.items():
            f.attrs[k] = v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_assemble.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add maceh/epc/assemble.py tests/test_assemble.py
git commit -m "feat(epc): g_mn,nu(k,q) contraction with phonon weighting and epc_pred.h5 writer"
```

---

### Task 7: EPC config (`maceh/parse_configs.py` + `maceh/default_configs/epc_default.ini`)

**Files:**
- Modify: `maceh/parse_configs.py` (append `EPCConfig` after `EvalConfig`, ~line 292)
- Create: `maceh/default_configs/epc_default.ini`
- Test: `tests/test_epc_config.py`

**Interfaces:**
- Consumes: `BaseConfig`, `EvalConfig` (existing).
- Produces (used by Task 8): `EPCConfig(config_file)` — all `EvalConfig` attributes (`model_dir`, `device`, `torch_dtype`, `out_dir`, `target`, `inference`, plus `[data]` incl. `radius`) and new attributes `structure_dir: str`, `q_grid: tuple[int,int,int]`, `k_grid: tuple`, `delta: float`, `phonopy_dir: str`, `fermi_energy: float`, `energy_window: float`, `grad_threshold: float`, `omega_tol: float`, `atom_indices: list[int] | None`, `save_derivatives: bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_epc_config.py`:

```python
import os

import pytest

from maceh.parse_configs import EPCConfig


def write_config(tmp_path):
    out_dir = tmp_path / 'out'
    cfg = f"""
[basic]
device = cpu
dtype = double
trained_model_dir = {tmp_path}
output_dir = {out_dir}

[data]
radius = 7.2

[epc]
structure_dir = {tmp_path}
q_grid = 2 2 2
k_grid = 4 4 4
delta = 0.02
phonopy_dir = {tmp_path}
fermi_energy = -1.5
energy_window = 1.0
atom_indices = 0 2
"""
    path = tmp_path / 'epc.ini'
    path.write_text(cfg)
    return str(path)


def test_epc_config_parses(tmp_path):
    config = EPCConfig(write_config(tmp_path))
    assert config.q_grid == (2, 2, 2)
    assert config.k_grid == (4, 4, 4)
    assert config.delta == pytest.approx(0.02)
    assert config.fermi_energy == pytest.approx(-1.5)
    assert config.energy_window == pytest.approx(1.0)
    assert config.atom_indices == [0, 2]
    assert config.radius == pytest.approx(7.2)
    # defaults from epc_default.ini
    assert config.grad_threshold == pytest.approx(1e-10)
    assert config.omega_tol == pytest.approx(1e-5)
    assert config.save_derivatives is False
    assert config.inference is True


def test_epc_config_requires_radius(tmp_path):
    path = write_config(tmp_path)
    text = open(path).read().replace('radius = 7.2', 'radius = -1')
    open(path, 'w').write(text)
    with pytest.raises(AssertionError):
        EPCConfig(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_epc_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'EPCConfig'`

- [ ] **Step 3: Create `maceh/default_configs/epc_default.ini`**

```ini
; DO NOT MODIFY THIS CONFIG FILE HERE!
; This is the default config file for deephe3-epc.py. If you want to create your
; own config, please first create a copy somewhere else.

[basic]

; same meanings as in eval_default.ini; inference must stay True for EPC.

device = cpu
dtype = double
trained_model_dir =
output_dir =
target = hamiltonian
inference = True
test_only = False

[data]

; Only `radius` is used by the EPC tool: it is the cutoff radius (Angstrom) for
; building the displacement-supercell graph and must be > 0. It should match the
; hopping range of the DFT data the model was trained on.

graph_dir =
DFT_data_dir =
processed_data_dir =
save_graph_dir =
target_data = hamiltonian
dataset_name =
get_overlap = True
radius = -1

[epc]

; structure_dir     string   Processed-data folder of ONE structure, containing lat.dat,
;                            element.dat, site_positions.dat, orbital_types.dat, info.json,
;                            overlaps.h5 and hamiltonians_pred.h5 (run deephe3-eval.py first).
; q_grid            3 ints   Uniform q-grid; also fixes the displacement supercell size.
; k_grid            3 ints   Uniform k-grid for the electronic states.
; delta             float    Finite-difference displacement step (Angstrom).
; phonopy_dir       string   Directory containing phonopy.yaml (+ FORCE_CONSTANTS) for the
;                            same unit cell.
; fermi_energy      float    Fermi level (eV) of the structure.
; energy_window     float    Half-width (eV) around fermi_energy; only states inside are stored.
; grad_threshold    float    Derivative blocks with max|dH| below this (eV/Angstrom) are dropped.
; omega_tol         float    Modes with energy below this (eV) are masked; their g is set to 0.
; atom_indices      ints     Optional 0-based unit-cell atoms to displace (blank = all).
;                            The acoustic sum rule diagnostic only runs when blank.
; save_derivatives  bool     Also store the raw dH/dR blocks in epc_pred.h5 (large!).

structure_dir =
q_grid = 1 1 1
k_grid = 1 1 1
delta = 0.01
phonopy_dir =
fermi_energy = 0.0
energy_window = 2.0
grad_threshold = 1e-10
omega_tol = 1e-5
atom_indices =
save_derivatives = False
```

- [ ] **Step 4: Append `EPCConfig` to `maceh/parse_configs.py`**

After the `EvalConfig` class (file ends around line 292), add:

```python
class EPCConfig(EvalConfig):
    def __init__(self, config_file):
        BaseConfig.__init__(self)
        epc_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'default_configs/epc_default.ini')
        print(f'Loading EPC config from: {config_file}')
        self.get_config(config_file, config_file_default=epc_default)

        self.get_basic_section()
        self.get_data_section()
        self.get_epc_section()

    def get_epc_section(self):
        self.structure_dir = self._config.get('epc', 'structure_dir')
        self.q_grid = tuple(int(x) for x in self._config.get('epc', 'q_grid').split())
        self.k_grid = tuple(int(x) for x in self._config.get('epc', 'k_grid').split())
        assert len(self.q_grid) == 3 and len(self.k_grid) == 3
        self.delta = self._config.getfloat('epc', 'delta')
        self.phonopy_dir = self._config.get('epc', 'phonopy_dir')
        self.fermi_energy = self._config.getfloat('epc', 'fermi_energy')
        self.energy_window = self._config.getfloat('epc', 'energy_window')
        self.grad_threshold = self._config.getfloat('epc', 'grad_threshold')
        self.omega_tol = self._config.getfloat('epc', 'omega_tol')
        ai = self._config.get('epc', 'atom_indices')
        self.atom_indices = [int(x) for x in ai.split()] if ai.strip() else None
        self.save_derivatives = self._config.getboolean('epc', 'save_derivatives')
        assert self.inference, 'EPC requires inference = True'
        assert self.radius > 0, 'EPC requires [data] radius > 0 (supercell graph cutoff)'
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_epc_config.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add maceh/parse_configs.py maceh/default_configs/epc_default.ini tests/test_epc_config.py
git commit -m "feat(epc): EPCConfig and default config for deephe3-epc.py"
```

---

### Task 8: Orchestrator, CLI, docs (`maceh/epc/run.py`, `deephe3-epc.py`)

**Files:**
- Create: `maceh/epc/run.py`
- Create: `deephe3-epc.py`
- Modify: `README.md` (add EPC section under Usage)
- Modify: `environment.yml` (add `phonopy` to the pip dependencies; read the file first and match its formatting)
- Test: `tests/test_run_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7; `DeepHE3Kernel`, `NetOutInfo` from `maceh/kernel.py`; `Collater`, `get_edge_fea` from `maceh/graph.py`.
- Produces: `run_epc(config_path: str, debug: bool = False)`; helpers `load_model_contexts(config) -> list[(kernel, net, construct_kernel)]` and `make_predict_fn(contexts, data, config, debug=False) -> callable` (the `predict_fn` consumed by `finite_difference`).

- [ ] **Step 1: Write the failing smoke test**

Full `run_epc` needs a trained checkpoint, which CI does not have; the smoke test covers importability and CLI wiring (real-model verification is the manual step at the end).

Create `tests/test_run_smoke.py`:

```python
import subprocess
import sys


def test_run_module_imports():
    from maceh.epc.run import run_epc, load_model_contexts, make_predict_fn
    assert callable(run_epc)


def test_cli_help():
    out = subprocess.run([sys.executable, 'deephe3-epc.py', '--help'],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert 'electron-phonon' in out.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_run_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc.run'`

- [ ] **Step 3: Implement `maceh/epc/run.py`**

```python
import os
import json
import time
import warnings

import numpy as np
import torch
import h5py

from ..kernel import DeepHE3Kernel, NetOutInfo
from ..graph import Collater, get_edge_fea
from ..parse_configs import EPCConfig
from .supercell import load_structure, build_supercell, uniform_grid
from .derivative import build_supercell_graph, finite_difference, acoustic_sum_rule
from .electron import load_orbital_slices, load_blocks_h5, blocks_to_dense, ElectronSolver
from .phonon import PhononData
from .assemble import compute_epc, write_epc_h5


def load_model_contexts(config):
    r''' one (kernel, net, construct_kernel) per trained model, mirroring
    DeepHE3Kernel.eval; multiple models each predict a subset of targets and
    their blocks are merged by update_hopping '''
    contexts = []
    for model_path in DeepHE3Kernel.find_model(config.model_dir):
        kernel = DeepHE3Kernel()
        kernel.eval_config = config
        kernel.load_config(train_config_path=os.path.join(model_path, 'src/train.ini'))
        kernel.dataset_info = NetOutInfo.from_json(os.path.join(model_path, 'src')).dataset_info
        kernel.config_set_target()
        construct_kernel = kernel.register_constructor(device=config.device)
        net = kernel.load_model(os.path.join(model_path, 'src'), device=config.device)
        checkpoint = torch.load(os.path.join(model_path, 'best_model.pkl'), map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        contexts.append((kernel, net, construct_kernel))
    return contexts


def make_predict_fn(contexts, data, config, debug=False):
    r''' returns predict_fn(positions) -> {str([Rx,Ry,Rz,I,J]): np.ndarray} on the
    fixed supercell graph; only edge_attr is recomputed from the positions '''
    dtype = torch.get_default_dtype()
    collate = Collater()

    def predict_fn(positions):
        data.edge_attr = get_edge_fea(positions, data.lattice[0], dtype, data.edge_key)
        batch = collate([data])
        H = {}
        for kernel, net, construct_kernel in contexts:
            with torch.no_grad():
                _, output_edge = net(batch.to(device=config.device))
            H_pred = construct_kernel.get_H(output_edge).cpu().numpy()
            kernel.update_hopping(H, H_pred, batch.x.cpu(), batch.edge_index.cpu(),
                                  batch.edge_key.cpu(), debug=debug)
        return H

    return predict_fn


def run_epc(config_path, debug=False):
    config = EPCConfig(config_path)
    torch.set_default_dtype(config.torch_dtype)
    if config.torch_dtype != torch.float64:
        warnings.warn('finite differences with float32 are noisy; '
                      'dtype = double is strongly recommended for EPC')

    struct = load_structure(config.structure_dir)
    with open(os.path.join(config.structure_dir, 'info.json')) as f:
        spinful = json.load(f)['isspinful']
    norb_list, norb_cumsum = load_orbital_slices(
        os.path.join(config.structure_dir, 'orbital_types.dat'), spinful=spinful)

    print('\n------- Loading trained model(s) -------')
    contexts = load_model_contexts(config)
    kernel0 = contexts[0][0]
    assert kernel0.dataset_info.spinful == spinful, \
        'model spinful does not match structure info.json'

    print('\n------- Stage 1: finite-difference Hamiltonian derivatives -------')
    sc_struct, smap = build_supercell(struct, config.q_grid)
    receptive = (kernel0.train_config.num_blocks + 1) * kernel0.train_config.cutoff_radius
    inv_lat = np.linalg.inv(sc_struct.lattice)
    for a in range(3):
        thickness = 1.0 / np.linalg.norm(inv_lat[:, a])
        if thickness < 2 * receptive:
            warnings.warn(f'supercell thickness along axis {a} ({thickness:.2f} A) is below '
                          f'twice the model receptive field ({receptive:.2f} A); periodic '
                          f'images of displaced atoms may contaminate dH/dR. '
                          f'Consider a denser q_grid.')

    data = build_supercell_graph(sc_struct, config.radius, torch.get_default_dtype())
    data.x = kernel0.dataset_info.Z_to_index[data.x]
    assert torch.all(data.x >= 0), 'structure contains elements unknown to the model'
    predict_fn = make_predict_fn(contexts, data, config, debug=debug)
    positions0 = data.pos.clone()

    n_displaced = len(config.atom_indices) if config.atom_indices else smap.n_uc_atoms
    begin = time.time()
    deriv = finite_difference(predict_fn, positions0, smap, norb_cumsum, config.delta,
                              atom_indices=config.atom_indices,
                              grad_threshold=config.grad_threshold)
    print(f'Finished {6 * n_displaced} forward passes on the supercell, '
          f'cost {time.time() - begin:.2f} seconds.')

    # delta-convergence report on the first displaced atom
    probe = [config.atom_indices[0]] if config.atom_indices else [0]
    deriv_half = finite_difference(predict_fn, positions0, smap, norb_cumsum,
                                   config.delta / 2, atom_indices=probe,
                                   grad_threshold=config.grad_threshold)
    dev = 0.0
    for alpha in range(3):
        full = deriv.blocks[(probe[0], alpha)]
        for pR, m in deriv_half.blocks[(probe[0], alpha)].items():
            if pR in full:
                dev = max(dev, float(np.abs(full[pR] - m).max()))
    print(f'delta-convergence: max |dH(delta) - dH(delta/2)| = {dev:.3e} eV/A '
          f'(delta = {config.delta} A)')
    if config.atom_indices is None:
        print(f'acoustic sum rule violation: {acoustic_sum_rule(deriv):.3e} eV/A')

    print('\n------- Stage 2: electron-phonon coupling assembly -------')
    H_by_R = blocks_to_dense(
        load_blocks_h5(os.path.join(config.structure_dir, 'hamiltonians_pred.h5')),
        norb_cumsum)
    S_by_R = blocks_to_dense(
        load_blocks_h5(os.path.join(config.structure_dir, 'overlaps.h5')),
        norb_cumsum, spin_expand=spinful)
    solver = ElectronSolver(H_by_R, S_by_R)
    phonons = PhononData.from_directory(config.phonopy_dir)
    assert phonons.natoms == len(struct.numbers), \
        'phonopy primitive cell does not match the structure'
    frac = struct.positions @ np.linalg.inv(struct.lattice)
    if not np.allclose(np.mod(frac, 1.0), np.mod(phonons.frac_coords, 1.0), atol=1e-3):
        warnings.warn('phonopy atom positions differ from the structure; '
                      'check that phonopy.yaml belongs to this unit cell')

    results = compute_epc(deriv, solver, phonons,
                          kpts=uniform_grid(config.k_grid), qpts=uniform_grid(config.q_grid),
                          fermi_energy=config.fermi_energy,
                          energy_window=config.energy_window,
                          omega_tol=config.omega_tol)

    stru_id = os.path.basename(os.path.normpath(config.structure_dir))
    out_dir = os.path.join(config.out_dir, stru_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'epc_pred.h5')
    write_epc_h5(out_path, results, dict(
        fermi_energy=config.fermi_energy, energy_window=config.energy_window,
        delta=config.delta, spinful=spinful, model_dir=config.model_dir,
        units='g, eps, omega in eV; dH in eV/Angstrom; masses in amu',
        note='dS/dtau corrections are neglected (model predicts H only)',
        date=time.strftime('%Y-%m-%d %H:%M:%S')))
    if config.save_derivatives:
        with h5py.File(out_path, 'a') as f:
            for (kappa, alpha), by_pR in deriv.blocks.items():
                for (p, R), m in by_pR.items():
                    f[f'dH/{kappa}/{"xyz"[alpha]}/{str(list(p) + list(R))}'] = m
    print(f'\nEPC written to "{out_path}"')
```

- [ ] **Step 4: Create `deephe3-epc.py`**

```python
#!/usr/bin/env python
# ===================================================================== #
# Electron-phonon coupling from a trained model via finite differences  #
# ===================================================================== #

# Usage: python <path-to-this-file>/deephe3-epc.py <your_config>.ini [-n NUM_THREADS] [--debug]
# Default config file is maceh/default_configs/epc_default.ini

import os
import argparse

parser = argparse.ArgumentParser(
    description='Compute electron-phonon coupling g_mn,nu(k,q) from finite-difference '
                'derivatives of the predicted Hamiltonian')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for EPC calculation')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
parser.add_argument('--debug', action='store_true',
                    help='Fill unpredicted matrix elements with 0 instead of throwing error.')
args = parser.parse_args()

if args.n is not None:
    os.environ["OMP_NUM_THREADS"] = f"{args.n}"
    os.environ["MKL_NUM_THREADS"] = f"{args.n}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{args.n}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{args.n}"
    import torch
    torch.set_num_threads(args.n)

from maceh.epc.run import run_epc
run_epc(args.config, debug=args.debug)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_run_smoke.py -v`
Expected: 2 PASSED (note: `test_cli_help` runs under `sys.executable`; run pytest with the DeepH python so the subprocess also has the deps).

- [ ] **Step 6: Update README.md and environment.yml**

In `README.md`, after the "Model inference" section, add:

```markdown
### Electron-phonon coupling

Given a trained model, the predicted `hamiltonians_pred.h5` (from `deephe3-eval.py`), the
overlap matrices `overlaps.h5` (preprocess with `get_overlap = True`), and a phonopy
calculation (`phonopy.yaml` + `FORCE_CONSTANTS`) for the same unit cell, you can compute
electron-phonon coupling matrix elements

g_mnv(k, q) = sum_ka sqrt(hbar / 2 M_k w_qv) e_kav(q) <psi_m,k+q| dH/dtau_ka(q) |psi_n,k>

on uniform k/q grids with

```
${python_path} ./deephe3-epc.py ./configs/epc.ini
```

The Hamiltonian derivatives dH/dtau are obtained by central finite differences of the
model prediction on a supercell commensurate with the q-grid (the supercell graph is
built once; only the edge vectors are recomputed for each displacement). Electronic
states come from diagonalizing the generalized problem H(k)C = S(k)Ce within an energy
window around the Fermi level; phonon frequencies and eigenvectors are read through the
phonopy API. Results are written to `<output_dir>/<stru_id>/epc_pred.h5`.

Note: in the non-orthogonal NAO basis the rigorous coupling also contains dS/dtau
correction terms. The model predicts H only, so these terms are neglected here.
```

In `environment.yml`, add `phonopy` to the pip dependency list (read the file first; keep its formatting).

- [ ] **Step 7: Run the full test suite**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add maceh/epc/run.py deephe3-epc.py tests/test_run_smoke.py README.md environment.yml
git commit -m "feat(epc): deephe3-epc.py CLI, run orchestrator, docs and phonopy dependency"
```

---

### Manual verification (requires a trained model — cannot run in CI)

After all tasks: with a trained model directory, a structure processed with `get_overlap = True`, a prior `deephe3-eval.py` run, and phonopy output, run:

```bash
/opt/anaconda3/envs/DeepH/bin/python deephe3-epc.py configs/epc.ini | tee sh/log_epc.txt
```

Check the printed diagnostics: (1) the edge_attr self-check passes silently (an AssertionError means convention drift); (2) delta-convergence deviation is small (≲1e-3 eV/Å); (3) the acoustic sum rule violation is small compared to typical |dH| values; (4) at q = Γ, `g` for acoustic modes is masked. The q=0 fold-back cross-check from the spec: run once with `q_grid = 1 1 1` and once with e.g. `2 1 1`, and compare `g` at the shared q=Γ, k=Γ point — they must agree to FD accuracy.
