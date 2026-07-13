# Cartesian AO Electron-Phonon Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the Cartesian atomic-orbital electron-phonon coupling g_ijκα(k,q) = [∂H(k,q)/∂τ_κα]_ij from finite-difference first derivatives of the MACE-H-predicted real-space Hamiltonian, saved to a per-structure `epc_cartesian_pred.h5`. Phonon-mode contraction and band-basis transformation are explicitly postponed (no phonopy, no diagonalization, no overlaps.h5).

**Architecture:** New package `maceh/epc/` with pure-function stages: supercell/index math (`supercell.py`), finite-difference derivatives on a fixed graph (`derivative.py`), double Fourier transform + HDF5 writer (`assemble.py`), and an orchestrator (`run.py`) driven by a new CLI `deephe3-epc.py` and `EPCConfig`. Spec: `docs/superpowers/specs/2026-07-13-electron-phonon-coupling-design.md`.

**Tech Stack:** PyTorch + torch_geometric + e3nn (existing model stack), numpy, h5py.

## Global Constraints

- Python for all commands: `/opt/anaconda3/envs/DeepH/bin/python` (has torch 2.x, torch_geometric, e3nn, scipy, h5py). Task 1 installs `pytest` into it.
- Run pytest from the repo root `/Users/jb/MACE-H` so `maceh` is importable: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/... -v`
- Units: energies eV, lengths Å; g comes out in eV/Å. k and q are fractional coordinates.
- H5 hopping keys are `str([Rx, Ry, Rz, i, j])` with **1-based** i, j; all in-memory atom indices are **0-based**.
- Bloch gauge is the cell-phase convention: `g_ijκα(k,q) = Σ_p exp(2πi q·p) Σ_R exp(2πi k·R) [∂H(p,R)]_ij` (matches the existing Julia/Band.py postprocessing).
- The training pipeline and model code are unchanged. The only edit outside `maceh/epc/`, `maceh/parse_configs.py`, `maceh/default_configs/`, and top-level script/docs is a one-line addition in `maceh/graph.py` (attach `edge_key` to the `data_folder=None` Data).
- Supercell atom ordering is cell-major: supercell index `= cell_lin(p) * n_uc_atoms + i` with `cell_lin(p) = (p1*n2 + p2)*n3 + p3`; home cell p=(0,0,0) atoms come first.
- Comment style matches the repo (sparse, `r''' ... '''` docstrings where used).

---

### Task 1: Supercell math (`maceh/epc/supercell.py`)

**Files:**
- Create: `maceh/epc/__init__.py` (empty file)
- Create: `maceh/epc/supercell.py`
- Test: `tests/test_supercell.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Tasks 2, 3, 4, 5):
  - `Structure` namedtuple: `positions` np.ndarray (N,3) cartesian Å float64, `lattice` np.ndarray (3,3) **rows are lattice vectors**, `numbers` np.ndarray (N,) int atomic numbers.
  - `load_structure(structure_dir: str) -> Structure`
  - `class SupercellMap(n_grid: tuple[int,int,int], n_uc_atoms: int)` with `.n_grid`, `.n_uc_atoms`, `.cells` (list of 3-tuples, cell-major order), `.n_cells`, `.cell_lin(p) -> int`, `.sc_index(i, p) -> int`, `.uc_of(sc_i) -> (i, p_tuple)`
  - `build_supercell(struct: Structure, n_grid) -> (Structure, SupercellMap)`
  - `fold_key(key: list[int], smap: SupercellMap) -> (p: tuple, R: tuple, i: int, j: int)` — key is `[Rx,Ry,Rz,I,J]` 1-based supercell; returns p reduced mod n_grid, R exact in unit-cell lattice units, i/j 0-based unit-cell atoms.
  - `uniform_grid(n_grid) -> np.ndarray (n1*n2*n3, 3)` fractional coordinates, cell-major order.

- [ ] **Step 1: Install test deps into the DeepH env**

```bash
/opt/anaconda3/envs/DeepH/bin/pip install pytest
```

Expected: installs without error.

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
- Produces (used by Task 6):
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
- Produces (used by Tasks 4, 6):
  - `@dataclass DerivativeData`: fields `n_grid: tuple`, `n_uc_atoms: int`, `delta: float`, `norb_cumsum: np.ndarray` ((n_uc_atoms+1,)), `blocks: dict` mapping `(kappa, alpha)` → `{(p_tuple, R_tuple): np.ndarray (norb_tot, norb_tot)}`; property `norb_tot`.
  - `finite_difference(predict_fn, positions0: torch.Tensor, smap, norb_cumsum, delta, atom_indices=None, grad_threshold=1e-10) -> DerivativeData` where `predict_fn(positions) -> {str([Rx,Ry,Rz,I,J]): np.ndarray}` predicts supercell hopping blocks at given positions on a **fixed** graph.
  - `acoustic_sum_rule(deriv: DerivativeData) -> float` — max |Σ_{κ,p} dH| over α and R.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_derivative_fd.py`. The stub model returns, for each hopping key, a 1×1 block equal to the edge length — its exact derivative w.r.t. any atom position is known analytically, so this validates the FD arithmetic, the fold-back wiring, and thresholding at once. Chain: 1 atom per cell along x, supercell 2×1×1.

```python
import numpy as np
import torch
import pytest

from maceh.epc.supercell import SupercellMap, fold_key
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
    # atom moves only that atom (its supercell periodic images are other sc atoms)
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

Note: with 1 atom per unit cell in a 2×1×1 supercell, displacing uc atom 0 moves only supercell atom 0 (index 0 = home cell); the atom in cell (1,0,0) is supercell atom 1 and stays put. `analytic_deriv` handles edges where both endpoints are atom 0 (contributions cancel).

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

### Task 4: Fourier transform and output (`maceh/epc/assemble.py`)

**Files:**
- Create: `maceh/epc/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `DerivativeData` (Task 3), `Structure` (Task 1).
- Produces (used by Task 6):
  - `compute_epc_cartesian(deriv: DerivativeData, kpts, qpts) -> dict` with keys:
    - `g` complex128 `(nk, nq, n_displaced, 3, norb_tot, norb_tot)` — `g_ijκα(k,q) = Σ_p e^{2πi q·p} Σ_R e^{2πi k·R} [∂H(p,R)]_ij`
    - `atom_indices` np int `(n_displaced,)` — sorted 0-based displaced atoms (derived from `deriv.blocks` keys)
    - `kpoints` `(nk, 3)`, `qpoints` `(nq, 3)` fractional
  - `write_epc_cartesian_h5(path, results: dict, struct: Structure, deriv: DerivativeData, attrs: dict)` — writes the spec's `epc_cartesian_pred.h5` layout: datasets `g_real`, `g_imag`, `kpoints`, `qpoints`, `atomic_numbers`, `atom_indices`, `cartesian_directions` (bytes 'x','y','z'), `orbital_indices` ((norb_tot,) atom owning each AO), `lattice`, `positions`, `supercell_matrix` (diag(n_grid)), `finite_difference_delta`; file attrs from `attrs` (must include `units`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assemble.py` (hand-computable phase sums: 1 atom, 1 orbital, two real-space derivative entries):

```python
import numpy as np
import h5py
import pytest

from maceh.epc.supercell import Structure
from maceh.epc.derivative import DerivativeData
from maceh.epc.assemble import compute_epc_cartesian, write_epc_cartesian_h5


def make_deriv():
    # blocks only for (kappa=0, alpha=0); alpha=1,2 empty
    blocks = {(0, 0): {((0, 0, 0), (0, 0, 0)): np.array([[1.0]]),
                       ((1, 0, 0), (2, 0, 0)): np.array([[0.5]])},
              (0, 1): {}, (0, 2): {}}
    return DerivativeData(n_grid=(2, 1, 1), n_uc_atoms=1, delta=0.01,
                          norb_cumsum=np.array([0, 1]), blocks=blocks)


def test_compute_epc_cartesian_phases():
    deriv = make_deriv()
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    qpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    res = compute_epc_cartesian(deriv, kpts, qpts)
    g = res['g']
    assert g.shape == (2, 2, 1, 3, 1, 1)
    assert np.array_equal(res['atom_indices'], [0])
    # k=0, q=0: 1 + 0.5
    assert g[0, 0, 0, 0, 0, 0] == pytest.approx(1.5)
    # k=0, q=(1/2,0,0): 1 + 0.5 * exp(2pi i * 0.5 * 1) = 1 - 0.5
    assert g[0, 1, 0, 0, 0, 0] == pytest.approx(0.5)
    # k=(1/2,0,0), q=0: 1 + 0.5 * exp(2pi i * 0.5 * 2) = 1.5
    assert g[1, 0, 0, 0, 0, 0] == pytest.approx(1.5)
    # k=(1/2,0,0), q=(1/2,0,0): 1 + 0.5 * (-1) * (1) = 0.5
    assert g[1, 1, 0, 0, 0, 0] == pytest.approx(0.5)
    # untouched directions are zero
    assert np.all(g[:, :, :, 1:, :, :] == 0)


def test_write_epc_cartesian_h5(tmp_path):
    deriv = make_deriv()
    struct = Structure(positions=np.array([[0.0, 0.0, 0.0]]),
                       lattice=3.0 * np.eye(3),
                       numbers=np.array([79]))
    kpts = np.array([[0.0, 0.0, 0.0]])
    qpts = np.array([[0.0, 0.0, 0.0]])
    res = compute_epc_cartesian(deriv, kpts, qpts)
    path = str(tmp_path / 'epc_cartesian_pred.h5')
    write_epc_cartesian_h5(path, res, struct, deriv,
                           {'units': 'g in eV/Angstrom', 'spinful': False})
    with h5py.File(path, 'r') as f:
        assert f['g_real'].shape == (1, 1, 1, 3, 1, 1)
        assert f['g_imag'].shape == (1, 1, 1, 3, 1, 1)
        assert f['g_real'][0, 0, 0, 0, 0, 0] == pytest.approx(1.5)
        assert np.array_equal(f['atomic_numbers'][()], [79])
        assert np.array_equal(f['atom_indices'][()], [0])
        assert [d.decode() for d in f['cartesian_directions'][()]] == ['x', 'y', 'z']
        assert np.array_equal(f['orbital_indices'][()], [0])
        assert np.allclose(f['lattice'][()], 3.0 * np.eye(3))
        assert np.allclose(f['supercell_matrix'][()], np.diag([2, 1, 1]))
        assert f['finite_difference_delta'][()] == pytest.approx(0.01)
        assert f.attrs['units'] == 'g in eV/Angstrom'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'maceh.epc.assemble'`

- [ ] **Step 3: Implement `maceh/epc/assemble.py`**

```python
import numpy as np
import h5py


def compute_epc_cartesian(deriv, kpts, qpts):
    r''' Cartesian atomic-orbital electron-phonon coupling
    g_ijka(k, q) = sum_p e^{2 pi i q.p} sum_R e^{2 pi i k.R} [dH(p, R)]_ij
    (cell-phase gauge). Returns g of shape (nk, nq, n_displaced, 3, norb, norb);
    phonon-mode contraction and band transformation are left for downstream. '''
    kpts = np.asarray(kpts, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    displaced = sorted({kappa for kappa, _ in deriv.blocks.keys()})
    nk, nq = len(kpts), len(qpts)
    norb = deriv.norb_tot
    g = np.zeros((nk, nq, len(displaced), 3, norb, norb), dtype=np.complex128)
    for ikap, kappa in enumerate(displaced):
        for alpha in range(3):
            for (p, R), m in deriv.blocks.get((kappa, alpha), {}).items():
                # phase factors for all (k, q) at once
                phase_q = np.exp(2j * np.pi * (qpts @ np.asarray(p, dtype=np.float64)))
                phase_k = np.exp(2j * np.pi * (kpts @ np.asarray(R, dtype=np.float64)))
                g[:, :, ikap, alpha] += (phase_k[:, None] * phase_q[None, :])[:, :, None, None] * m
    return dict(g=g, atom_indices=np.array(displaced, dtype=int),
                kpoints=kpts, qpoints=qpts)


def write_epc_cartesian_h5(path, results, struct, deriv, attrs):
    norb_per_atom = np.diff(deriv.norb_cumsum)
    orbital_indices = np.repeat(np.arange(deriv.n_uc_atoms), norb_per_atom)
    with h5py.File(path, 'w') as f:
        f['g_real'] = results['g'].real
        f['g_imag'] = results['g'].imag
        f['kpoints'] = results['kpoints']
        f['qpoints'] = results['qpoints']
        f['atomic_numbers'] = np.asarray(struct.numbers, dtype=int)
        f['atom_indices'] = results['atom_indices']
        f['cartesian_directions'] = np.array([b'x', b'y', b'z'])
        f['orbital_indices'] = orbital_indices
        f['lattice'] = struct.lattice
        f['positions'] = struct.positions
        f['supercell_matrix'] = np.diag(deriv.n_grid)
        f['finite_difference_delta'] = deriv.delta
        for k, v in attrs.items():
            f.attrs[k] = v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/test_assemble.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add maceh/epc/assemble.py tests/test_assemble.py
git commit -m "feat(epc): Fourier transform to Cartesian AO g_ij,ka(k,q) and h5 writer"
```

---

### Task 5: EPC config (`maceh/parse_configs.py` + `maceh/default_configs/epc_default.ini`)

**Files:**
- Modify: `maceh/parse_configs.py` (append `EPCConfig` after `EvalConfig`, ~line 292)
- Create: `maceh/default_configs/epc_default.ini`
- Test: `tests/test_epc_config.py`

**Interfaces:**
- Consumes: `BaseConfig`, `EvalConfig` (existing).
- Produces (used by Task 6): `EPCConfig(config_file)` — all `EvalConfig` attributes (`model_dir`, `device`, `torch_dtype`, `out_dir`, `target`, `inference`, plus `[data]` incl. `radius`) and new attributes `structure_dir: str`, `q_grid: tuple[int,int,int]`, `k_grid: tuple`, `delta: float`, `grad_threshold: float`, `atom_indices: list[int] | None`, `save_derivatives: bool`.

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
    assert config.atom_indices == [0, 2]
    assert config.radius == pytest.approx(7.2)
    # defaults from epc_default.ini
    assert config.grad_threshold == pytest.approx(1e-10)
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

; structure_dir     string   Folder with lat.dat, element.dat, site_positions.dat of ONE
;                            structure (DeepH processed-data conventions). Orbital counts
;                            and spinful-ness come from the trained model's dataset_info.
; q_grid            3 ints   Uniform q-grid; also fixes the displacement supercell size.
; k_grid            3 ints   Uniform k-grid.
; delta             float    Finite-difference displacement step (Angstrom).
; grad_threshold    float    Derivative blocks with max|dH| below this (eV/Angstrom) are dropped.
; atom_indices      ints     Optional 0-based unit-cell atoms to displace (blank = all).
;                            The acoustic sum rule diagnostic only runs when blank.
; save_derivatives  bool     Also store the raw dH/dR blocks in epc_cartesian_pred.h5 (large!).

structure_dir =
q_grid = 1 1 1
k_grid = 1 1 1
delta = 0.01
grad_threshold = 1e-10
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
        self.grad_threshold = self._config.getfloat('epc', 'grad_threshold')
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

### Task 6: Orchestrator, CLI, docs (`maceh/epc/run.py`, `deephe3-epc.py`)

**Files:**
- Create: `maceh/epc/run.py`
- Create: `deephe3-epc.py`
- Modify: `README.md` (add EPC section under Usage)
- Test: `tests/test_run_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `DeepHE3Kernel`, `NetOutInfo` from `maceh/kernel.py`; `Collater`, `get_edge_fea` from `maceh/graph.py`.
- Produces: `run_epc(config_path: str, debug: bool = False)`; helpers `load_model_contexts(config) -> list[(kernel, net, construct_kernel)]`, `make_predict_fn(contexts, data, config, debug=False) -> callable` (the `predict_fn` consumed by `finite_difference`), and `atom_norb_from_model(dataset_info, numbers) -> np.ndarray norb_cumsum` (per-unit-cell-atom orbital counts derived from the model's dataset_info, doubled when spinful).

- [ ] **Step 1: Write the failing smoke test**

Full `run_epc` needs a trained checkpoint, which CI does not have; the smoke test covers importability, the orbital-count helper, and CLI wiring (real-model verification is the manual step at the end).

Create `tests/test_run_smoke.py`:

```python
import subprocess
import sys

import numpy as np
import torch


def test_run_module_imports():
    from maceh.epc.run import run_epc, load_model_contexts, make_predict_fn
    assert callable(run_epc)


def test_atom_norb_from_model():
    from maceh.epc.run import atom_norb_from_model
    from maceh.kernel import DatasetInfo
    # species 0 = Z 6 with s+p (4 orbitals), species 1 = Z 79 with s (1 orbital)
    info = DatasetInfo(spinful=False, index_to_Z=[6, 79], orbital_types=[[0, 1], [0]])
    cumsum = atom_norb_from_model(info, np.array([79, 6, 6]))
    assert list(cumsum) == [0, 1, 5, 9]
    info_sp = DatasetInfo(spinful=True, index_to_Z=[6, 79], orbital_types=[[0, 1], [0]])
    cumsum_sp = atom_norb_from_model(info_sp, np.array([79, 6]))
    assert list(cumsum_sp) == [0, 2, 10]


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
from .assemble import compute_epc_cartesian, write_epc_cartesian_h5


def atom_norb_from_model(dataset_info, numbers):
    r''' per-atom orbital counts (doubled when spinful) derived from the trained
    model's dataset_info; returns the cumulative-sum slice boundaries '''
    norb_per_species = [sum(2 * l + 1 for l in types) for types in dataset_info.orbital_types]
    factor = 2 if dataset_info.spinful else 1
    norb = []
    for Z in numbers:
        species = int(dataset_info.Z_to_index[int(Z)])
        assert species >= 0, f'element Z={Z} unknown to the model'
        norb.append(factor * norb_per_species[species])
    return np.concatenate([[0], np.cumsum(norb)])


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

    print('\n------- Loading trained model(s) -------')
    contexts = load_model_contexts(config)
    kernel0 = contexts[0][0]
    norb_cumsum = atom_norb_from_model(kernel0.dataset_info, struct.numbers)

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

    print('\n------- Stage 2: Fourier transform to g_ij,ka(k, q) -------')
    results = compute_epc_cartesian(deriv, kpts=uniform_grid(config.k_grid),
                                    qpts=uniform_grid(config.q_grid))

    stru_id = os.path.basename(os.path.normpath(config.structure_dir))
    out_dir = os.path.join(config.out_dir, stru_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'epc_cartesian_pred.h5')
    write_epc_cartesian_h5(out_path, results, struct, deriv, dict(
        units='g in eV/Angstrom; k, q fractional; lattice, positions in Angstrom',
        spinful=kernel0.dataset_info.spinful, delta=config.delta,
        model_dir=config.model_dir,
        note='Cartesian AO coupling g_ij,ka(k,q) = [dH(k,q)/dtau_ka]_ij; phonon-mode '
             'contraction and band transformation (incl. possible dS/dtau handling) '
             'are left for downstream postprocessing',
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
    description='Compute the Cartesian AO electron-phonon coupling g_ij,ka(k,q) from '
                'finite-difference derivatives of the predicted Hamiltonian')
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
Expected: 3 PASSED (note: `test_cli_help` runs `sys.executable`, so run pytest with the DeepH python; `--help` exits before the heavy `maceh` import because `parse_args()` is called first).

- [ ] **Step 6: Update README.md**

In `README.md`, after the "Model inference" section, add:

```markdown
### Electron-phonon coupling (Cartesian AO basis)

Given a trained model and a structure (lat.dat, element.dat, site_positions.dat), you can
compute the Cartesian atomic-orbital electron-phonon coupling

g_ij,ka(k, q) = [dH(k, q) / dtau_ka]_ij

on uniform k/q grids with

```
${python_path} ./deephe3-epc.py ./configs/epc.ini
```

The Hamiltonian derivatives dH/dtau are obtained by central finite differences of the
model prediction on a supercell commensurate with the q-grid (the supercell graph is
built once; only the edge vectors are recomputed for each displacement), folded back to
cell-resolved dH_ij(R)/dtau_ka(p) and Fourier transformed. Results are written to
`<output_dir>/<stru_id>/epc_cartesian_pred.h5` (complex g stored as g_real/g_imag with
shape [nk, nq, natoms, 3, norb, norb], plus grids and structure metadata).

This output is the electronic perturbation before contraction with phonon eigenvectors
and before transformation to band eigenstates; those steps (which need phonon data,
electronic eigenvectors, and in the non-orthogonal NAO basis possibly dS/dtau handling)
are downstream postprocessing.
```

- [ ] **Step 7: Run the full test suite**

Run: `/opt/anaconda3/envs/DeepH/bin/python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add maceh/epc/run.py deephe3-epc.py tests/test_run_smoke.py README.md
git commit -m "feat(epc): deephe3-epc.py CLI, run orchestrator and docs for Cartesian AO EPC"
```

---

### Manual verification (requires a trained model — cannot run in CI)

After all tasks: with a trained model directory and a structure folder, run:

```bash
/opt/anaconda3/envs/DeepH/bin/python deephe3-epc.py configs/epc.ini | tee sh/log_epc.txt
```

Check the printed diagnostics: (1) the edge_attr self-check passes silently (an AssertionError means convention drift); (2) delta-convergence deviation is small (≲1e-3 eV/Å); (3) the acoustic sum rule violation is small compared to typical |dH| values. The q=0 fold-back cross-check from the spec: run once with `q_grid = 1 1 1` and once with e.g. `2 1 1`, and compare `g` at the shared k=Γ, q=Γ point — they must agree to FD accuracy.
