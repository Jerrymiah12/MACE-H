# Design: Electron-phonon coupling from predicted Hamiltonian derivatives

**Date:** 2026-07-13
**Status:** Approved

## Goal

Add an inference-side tool that takes a trained MACE-H model and, per structure,
computes the electron-phonon coupling matrix elements

g_mnν(k, q) = Σ_κα √(ħ / 2 M_κ ω_qν) e_κα,ν(q) ⟨ψ_m,k+q | ∂H/∂τ_κα(q) | ψ_n,k⟩

from finite-difference first derivatives of the predicted real-space Hamiltonian,
and stores them in a separate per-structure HDF5 file `epc_pred.h5` (next to the
existing `hamiltonians_pred.h5`). The Hamiltonian training and data pipeline are
unchanged.

## Inputs

- Trained model directory (same as `deephe3-eval.py`).
- Unit-cell structure via the existing processed-data / graph pipeline.
- Overlap matrices `overlaps.h5` from the existing `get_olp` preprocessing
  (needed because the NAO basis is non-orthogonal).
- Phonopy results for the same unit cell: `phonopy.yaml` + `FORCE_CONSTANTS`
  (or `force_constants.hdf5`), read through the phonopy Python API.
- Uniform q-grid (n1 × n2 × n3) and k-grid, Fermi energy, and an energy window
  around it — all from the config.

## Architecture

New package `maceh/epc/` plus a CLI script `deephe3-epc.py <config>.ini`:

- `maceh/epc/supercell.py` — build the n1×n2×n3 supercell from the unit cell;
  index maps unit-cell atom (κ, cell p) ↔ supercell atom.
- `maceh/epc/derivative.py` — finite-difference driver: graph construction reuse,
  analytic `edge_attr` recomputation, central differences, fold-back to
  unit-cell labels ∂H_ij(R)/∂τ_κα(p).
- `maceh/epc/electron.py` — H(k)/S(k) Bloch sums, generalized eigenproblem,
  Fermi-window state selection.
- `maceh/epc/phonon.py` — phonopy interface: ω_qν and e_κα,ν(q) on the q-grid.
- `maceh/epc/assemble.py` — contraction into g_mnν(k, q) and the `epc_pred.h5`
  writer.
- `deephe3-epc.py` — CLI mirroring `deephe3-eval.py`; new kernel entry point
  `DeepHE3Kernel.epc(config)` that reuses eval's config/model/graph loading.

## Stage 1 — Hamiltonian derivatives (model side)

For q ≠ 0 the per-cell derivative ∂H/∂τ_κα(p) is required, so displacements are
done in a supercell commensurate with the q-grid (frozen-phonon style):

1. Build the n1×n2×n3 supercell; the q-grid being commensurate makes all
   Fourier sums over p exact.
2. Build the graph **once** from the unperturbed supercell.
3. For each unit-cell atom κ (N atoms) and direction α ∈ {x, y, z}: displace the
   home-cell copy of κ by ±δ (default δ = 0.01 Å), recompute only `edge_attr`,
   run the model, construct hopping blocks, central-difference:
   dH = (H⁺ − H⁻) / 2δ. 6N supercell forward passes total.
4. Fold back by translational invariance: a supercell derivative block between
   atoms in cells (p_i, p_j) with κ displaced in the home cell equals
   ∂H_{i(0), j(p_j − p_i)} / ∂τ_{κ(−p_i), α}.

### Fixed edge set, recomputed geometry

The neighbor list, `edge_index`, and `edge_key` are reused across displacements;
only `edge_attr` is recomputed analytically. Convention (from `maceh/graph.py`):
`edge_key` rows are `[Rx, Ry, Rz, i, j]` (1-based i, j) with
`nn_coords = pos[j] + R_key @ lattice`, so the displaced edge vector is
`v = (pos'[j] + R_key @ lattice) − pos'[i]` and `edge_attr = [|v|, v_x, v_y, v_z]`
(stored order; `Net.forward` applies its own `[0, 2, 3, 1]` reorder). This keeps
block bookkeeping identical across displacements so blocks subtract cleanly, and
is exact because δ ≪ cutoff.

**Startup self-check:** recompute `edge_attr` for the unperturbed supercell from
`pos`, `lattice`, `edge_key` and assert it matches the stored `edge_attr` to
tight tolerance.

Other stage-1 notes:

- `only_ij` output is completed with the existing ijji conversion; the
  hermiticity relation H_ji(−R) = H_ij(R)† is preserved by differentiation.
- The model's shift/scale buffers are position-independent, so FD through the
  full `forward` is automatically correct.
- Warn if any supercell dimension is smaller than the model receptive field
  (num_blocks × r_max) — periodic images of the displaced atom would then
  contaminate the derivative.

## Stage 2 — EPC assembly (post-processing side)

1. Predict unit-cell H_R blocks with the same model (existing eval path);
   read S_R from `overlaps.h5`.
2. Bloch sums: H(k) = Σ_R e^{ik·R} H_R, S(k) likewise. Solve the generalized
   eigenproblem H(k) C = S(k) C ε with `scipy.linalg.eigh` for every k and k+q
   needed by the grids.
3. Phonons: phonopy API gives ω_qν and eigenvectors e_κα,ν(q) at each q-grid
   point; atomic masses come from phonopy.
4. Perturbation Bloch sum: for each (q, ν),
   δ_qν H(k) = Σ_κα √(ħ/2M_κω_qν) e_κα,ν(q) Σ_p e^{iq·p} [Bloch-summed
   ∂H/∂τ_κα(p)](k), then g_mnν(k, q) = ⟨ψ_m,k+q| δ_qν H |ψ_n,k⟩ with the
   S-metric bras/kets from step 2.
5. State filter: keep only bands with ε within `fermi_energy ± energy_window`;
   store their global band indices per k.
6. Soft modes (ω ≤ small tolerance, e.g. acoustic at Γ): the √(ħ/2Mω) factor
   diverges — set those g to 0 and record a mask, matching standard practice.

Units: H and ε in eV, positions in Å, masses in amu; g in eV using standard
CODATA constants for ħ.

## Config

New `[epc]` section (defaults added in `maceh/default_configs`); model and
structure keys reuse the eval config:

- `q_grid` — e.g. `4 4 4` (also fixes the supercell size)
- `k_grid` — e.g. `8 8 8`
- `delta` — FD step in Å, default `0.01`
- `phonopy_dir` — directory containing `phonopy.yaml` and force constants
- `fermi_energy` — eV
- `energy_window` — eV, half-width around `fermi_energy`, default `2.0`
- `overlap_h5_path` — path to `overlaps.h5` (default: alongside processed data)
- `out_dir` — output directory
- `save_derivatives` — bool, default false; also dump dH/dR blocks for debugging

## Output — `epc_pred.h5` per structure

- `g` — complex, shape `[nq, nmodes, nk, nbands_w, nbands_w]`
- `band_indices` — per-k global indices of the windowed bands
- `eps_k` — band energies of windowed states, eV
- `omega_q` — phonon frequencies `[nq, nmodes]`
- `kpts`, `qpts` — fractional grid coordinates
- `soft_mode_mask` — modes where g was zeroed
- attrs: `fermi_energy`, `energy_window`, `delta`, units, model path, date

dH/dR is not stored by default (`save_derivatives` opts in).

## Verification

1. `edge_attr` recomputation self-check — hard assert at startup.
2. δ-convergence report: recompute one atom's derivatives with δ/2 and print the
   max relative deviation (report only).
3. Acoustic sum-rule diagnostic: Σ_κp ∂H_ij(R)/∂τ_κα(p) should be small for a
   translationally invariant model; print the max norm.
4. q=0 cross-check: g at q=0 assembled through the supercell fold-back must
   match a direct unit-cell FD calculation (validates fold-back and gauge).

## Testing

- Unit tests for the supercell index maps and fold-back relation on a tiny
  synthetic lattice.
- Unit test for `edge_attr` recomputation against `maceh/graph.py` output.
- Unit tests for Bloch-sum/gauge conventions and the g contraction using a stub
  "model" with an analytic H(positions), so no trained checkpoint is needed.
- Hermiticity checks: g_mnν(k, q) = g*_nmν(k+q, −q) on test data.

## Known limitation (documented, not solved)

In the non-orthogonal NAO basis the rigorous EPC also contains ∂S/∂τ
corrections. The model predicts H only, so these terms are neglected; this is
stated in the README/tool documentation.
