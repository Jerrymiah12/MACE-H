# Design: Cartesian AO electron-phonon coupling from predicted Hamiltonian derivatives

**Date:** 2026-07-13 (revised same day: stop before phonon-mode and band-state contractions)
**Status:** Approved

## Goal

Add an inference-side tool that takes a trained MACE-H model and, per structure,
computes the **Cartesian, atomic-orbital electron-phonon coupling**

g_ijκα(k, q) = [∂H(k, q) / ∂τ_κα]_ij

from finite-difference first derivatives of the predicted real-space Hamiltonian,
where i, j are atomic-orbital indices, κ the displaced atom, α ∈ {x, y, z} the
Cartesian direction, k the electronic wavevector and q the perturbation
wavevector. Results are stored per structure in `epc_cartesian_pred.h5`.

This is the electronic perturbation **before** contraction with phonon
eigenvectors and **before** transformation into band eigenstates. The downstream
conversion (postponed) is

g_mnν(k, q) = Σ_ijκα C*_im(k+q) g_ijκα(k, q) C_jn(k) √(ħ / 2 M_κ ω_qν) e_κα,ν(q),

which will need phonon data, electronic eigenvectors, and (in the non-orthogonal
AO basis, depending on convention) ∂S/∂τ handling.

## Explicitly out of scope (postponed)

- Phonopy force constants, phonon frequencies ω_qν, eigenvectors e_κα,ν(q)
- Electronic diagonalization, band indices m/n, Fermi-level energy window
- Use of `overlaps.h5` / the generalized eigenvalue problem
- Final g_mnν(k, q)

Orbital and overlap conventions are preserved (AO ordering matches
`hamiltonians_pred.h5` / `overlaps.h5` block conventions) so the later
transformation can be added downstream.

## Pipeline

```
input crystal structure (unit cell)
   ↓ build q-commensurate supercell (n1×n2×n3 = q_grid)
   ↓ displace atom κ by ±δ along α (home cell); graph built once,
     only edge_attr recomputed per displacement
   ↓ trained MACE-H model forward passes (6N per structure)
   ↓ central finite difference:  ∂H/∂τ ≈ [H(τ+δ) − H(τ−δ)] / 2δ
   ↓ fold supercell blocks → cell-resolved ∂H_ij(R)/∂τ_κα(p)
   ↓ double Fourier transform:
     g_ijκα(k, q) = Σ_p e^{2πi q·p} Σ_R e^{2πi k·R} [∂H(R)/∂τ_κα(p)]_ij
   ↓ epc_cartesian_pred.h5
```

## Stage 1 — Hamiltonian derivatives (model side)

Unchanged from the original design:

1. Build the n1×n2×n3 supercell (commensurate q-grid makes Fourier sums over p
   exact). Supercell atom ordering is cell-major; home-cell atoms first.
2. Build the graph **once** (radius-based, cutoff = `[data] radius`), reusing
   `edge_index`/`edge_key` across displacements; recompute only `edge_attr`
   analytically via `get_edge_fea` (edge vector = `pos[j] + R_key @ lattice −
   pos[i]`). Startup self-check asserts the recomputed unperturbed `edge_attr`
   matches graph construction.
3. Central differences with δ = 0.01 Å default; 6N supercell forward passes.
   Predicted blocks are Hermitized (H_ij(R) ← (H_ij(R) + H_ji(−R)†)/2, matching
   the band-structure postprocessing) before differencing, so g is the
   derivative of the same Hamiltonian used for bands and satisfies
   g(k,q)† = g(k+q,−q). EPC inference precision is decoupled from the recorded
   training dtype: float32 checkpoints are promoted to the EPC dtype
   (double by default) at load time.
4. Fold back by translational invariance: supercell block between cells
   (p_i, p_j) with κ displaced in the home cell → ∂H_{i(0), j(p_j − p_i)} /
   ∂τ_{κ(−p_i), α} (p reduced mod n_grid; exact for commensurate q).
5. Warn if any supercell thickness is below twice the model receptive field
   ((num_blocks + 1) × cutoff_radius).
6. Derivative blocks below `grad_threshold` are dropped (model locality makes
   far blocks exactly zero).

## Stage 2 — Fourier transform to g_ijκα(k, q)

Cell-phase gauge (matching the existing Julia/Band.py postprocessing):
for each displaced atom κ, direction α, q-point and k-point,

g_ijκα(k, q) = Σ_p e^{2πi q·p} Σ_R e^{2πi k·R} [∂H(p, R)]_ij,

evaluated on uniform k- and q-grids from the config (q-grid = supercell grid).
Output is dense complex, shape `[nk, nq, n_displaced_atoms, 3, norb, norb]`,
stored as separate real and imaginary arrays. The writer streams one q-point
slab at a time into chunked HDF5 datasets, so peak memory is bounded by
`nk × n_displaced × 3 × norb²` rather than the full tensor; the estimated
on-disk size is printed before writing.

## Inputs

- Trained model directory (same as `deephe3-eval.py`).
- Structure files in `structure_dir`: `lat.dat`, `element.dat`,
  `site_positions.dat` (DeepH processed-data conventions). Orbital counts per
  atom and spinful-ness come from the model's `dataset_info.json` — no
  Hamiltonian/overlap h5 files are needed at this stage.
- Config: q-grid / k-grid, δ, cutoff radius, optional displaced-atom subset.

## Interface

- CLI `deephe3-epc.py <config>.ini [-n N] [--debug]`, mirroring `deephe3-eval.py`.
- New `EPCConfig` (extends `EvalConfig`) with `[epc]` section:
  `structure_dir`, `q_grid`, `k_grid`, `delta` (default 0.01), `grad_threshold`
  (default 1e-10), `atom_indices` (optional 0-based subset), `save_derivatives`
  (bool). `[data] radius` must be > 0.
- Package `maceh/epc/`: `supercell.py` (supercell + index folding),
  `derivative.py` (graph + FD driver), `assemble.py` (Fourier transform +
  writer), `run.py` (orchestrator).

## Output — `epc_cartesian_pred.h5` per structure

```
epc_cartesian_pred.h5
├── g_real, g_imag          [nk, nq, natoms_displaced, 3, norb, norb]
├── kpoints, qpoints        fractional, uniform grids
├── atomic_numbers          (N,) unit-cell atoms
├── atom_indices            (natoms_displaced,) 0-based displaced atoms
├── cartesian_directions    ['x', 'y', 'z']
├── orbital_indices         (norb,) atom index owning each AO row/column
├── lattice                 (3,3) rows are unit-cell lattice vectors, Å
├── positions               (N,3) cartesian Å
├── supercell_matrix        diag(q_grid)
├── finite_difference_delta δ in Å
└── attrs: units ('g in eV/Angstrom'), spinful, model_dir, date
```

Raw real-space dH blocks are stored only with `save_derivatives = True`.

## Verification

1. `edge_attr` recomputation self-check — hard assert at startup.
2. δ-convergence report (first displaced atom, δ vs δ/2) — printed.
3. Acoustic sum-rule diagnostic (Σ_κp ∂H) — printed when all atoms displaced.
4. q=0 fold-back cross-check (manual): g at the shared Γ point must agree
   between a 1×1×1 and a larger q_grid run to FD accuracy.

## Testing

- Unit tests for supercell index maps and `fold_key` on small lattices.
- `edge_attr` recomputation test against `maceh/graph.py` on a tiny structure.
- FD driver tested with a stub `predict_fn` whose analytic derivative is known.
- Fourier transform tested against hand-computed phase sums.
- No trained checkpoint required in CI; real-model check is manual.

## Known limitation (documented, not solved)

Output stays in the AO basis. The later band transformation in the
non-orthogonal NAO basis may require ∂S/∂τ handling depending on convention;
the model predicts H only. Stated in the README.
