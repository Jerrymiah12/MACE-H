# mgo_lr — MgO MACE-H-LR dataset-generation pipeline

Standalone tooling that generates the MgO training dataset for MACE-H-LR:
displacement structures, ABACUS/Quantum ESPRESSO input decks, DFT-output
conversion to the DeepH-E3/MACE-H format, the screened-dipole long-range
processor producing `H^LR` and `H^SR = H^full − H^LR` labels, a three-tier
validation battery, locality diagnostics, grouped leakage-safe splits, and
training-target export.

Design spec: `docs/superpowers/specs/2026-07-20-mgo-lr-dataset-pipeline-design.md`.
Source instructions and reviews: `instructions/`.

`mgo_lr` never imports `maceh`. Compatibility with the maceh loader is a
file-format contract (`maceh/graph.py` and its recursive directory discovery)
pinned by tests.

## Requirements

- `/opt/anaconda3/envs/DeepH/bin/python` (numpy, scipy, h5py, PyYAML, pytest).
- DFT (ABACUS SCF, QE DFPT) runs externally on the cluster; this package only
  prepares inputs and consumes outputs, so every stage is separately
  invokable.

## Usage

```
python -m mgo_lr <stage> --config mgo_lr/configs/mgo.yaml --workspace <dir> [--set pilot|main|large] [--target full|lr|sr] [--force]
```

Stages, in the order of one full round-trip:

| Stage | What it does | Run after |
|---|---|---|
| `init-reference` | ABACUS decks: ecut/k-mesh scans, cell-relax, final SCF | — |
| `collect-reference` | relaxed lattice constant + permanent reference artifacts | reference ABACUS runs |
| `init-dfpt` | QE `pw.in` / `ph.in` (`epsil`+`trans` explicit) at the reference geometry | collect-reference |
| `collect-dfpt` | parse ph.x → ASR-corrected `Z*`, `ε∞` (+ sanity checks) | QE runs |
| `gen-structures --set …` | supercells + displacement patterns + ABACUS decks + metadata | collect-reference |
| `collect-dft --set …` | parse SCF logs + CSR matrices → `hamiltonians_full.h5`, `overlaps.h5`, DeepH-E3 structure files | snapshot ABACUS runs |
| `lr-process --set …` | screened-dipole `V^LR` → `hamiltonians_lr.h5`, `hamiltonians_sr.h5` | collect-dft, collect-dfpt |
| `validate --set …` | Tier-1 hard checks (reject) + Tier-2 response checks | lr-process |
| `locality-report --set …` | Tier-3 diagnostics: tail fractions, odd response, family comparisons | validate |
| `export-target --target …` | materialize `hamiltonians.h5` for the maceh loader | lr-process |
| `organize` | grouped splits, candidate dirs, `loader_splits/` views, `metadata.yaml` provenance | validate, export-target |
| `status` | per-set state counts | any time |

Snapshots advance through `prepared → dft_done → converted → lr_done →
validated` (or `rejected`, moved to `rejected/` with a machine-readable
reason). Stages are idempotent: already-processed snapshots are skipped
without `--force`, and raw DFT outputs are never modified.

The default `pilot_expanded: false` generates the 18-snapshot initial
approval pilot. Set it to `true` in a separate workspace to generate the
50-snapshot follow-up, including matched finite-|q| trend probes.

## Workspace layout

```
<workspace>/
├── metadata.yaml            # lr_definition, provenance, training_target
├── splits.json              # grouped, leakage-safe splits
├── reference/               # permanent reference artifacts + abacus/ + qe/
├── pilot/                   # snapshot_000001/ …
├── main/
├── test_large_cell/         # 4x4x4 extrapolation set, never mixed into main
├── validation_candidates/   # symlinks + candidates.json
├── test_candidates/
├── loader_splits/            # real train/validation/test dirs with file links
│   ├── train/
│   ├── validation/
│   └── test/
├── rejected/
└── generation_logs/         # resolved configs, validation + locality reports
```

Use the roots recorded under `loader_split_roots` in `metadata.yaml` when
building MACE-H graphs. They contain real snapshot directories because the
loader's `os.walk` traversal does not follow directory symlinks. Run
`export-target` before loading so each view's `hamiltonians.h5` link resolves.

## Conventions (fixed for the dataset; recorded in metadata)

- Units: energies **eV**, lengths **Å**, charges in units of **e**
  (`constants.py` is the only place unit factors live).
- h5 keys: JSON `[Rx, Ry, Rz, i, j]`, **1-based** atom indices, dense
  float64 `(norb_i, norb_j)` blocks; `orbital_types.dat` one line per atom;
  `lat.dat`/`rlat.dat` store vectors as columns, `rlat` includes 2π.
- Atom ordering is species-major (all Mg, then all O), cells in
  `np.ndindex` order; STRU writers enforce it.
- LR definition: damped reciprocal-space screened-dipole sum, `V(G=0)=0`
  gauge, reference-position phase (`e^{−iG·R⁰}`), electron-potential-energy
  sign (`LR_SIGN = −1`, pinned by test), inversion-symmetric G set with a
  dielectric-ellipsoid cutoff. **Ewald Λ is part of the dataset definition**
  — it is never "converged away", and `lr-process` refuses to mix two LR
  definitions in one workspace. The identity also includes SHA-256 hashes of
  the reference cell, positions, species/atomic-number mapping, Born charges,
  and dielectric tensor. Validation, organization, and LR/SR export reject a
  stale identity.
- ABACUS `INPUT` always has `gamma_only 0` and `symmetry 0` (the gamma-only
  algorithm does not support `out_mat_hs2`).
- One global seed (`displacements.seed`); all randomness flows through
  `np.random.default_rng([seed, …])` derived streams.

## Validation tiers

1. **Tier 1 (hard, per snapshot):** file inventory, NaN/Inf, dimensions,
   integer key format, atom/orbital mapping, reference-position agreement,
   units, rlat/lat convention, overlap diagonal, complete raw-output SHA-256,
   hermiticity, `H^SR + H^LR = H^full` reconstruction, reciprocal-set
   symmetry, imaginary residual, G-sum convergence, equilibrium/translation
   exact zeros. Failure ⇒ snapshot rejected, nonzero exit.
2. **Tier 2 (small-amplitude response):** `E_sign(A)` and `E_linear(A)`
   must decrease with decreasing A within each pattern group. Warnings
   until `validation.tier2_enforce: true` (then the set fails, individual
   snapshots are kept).
3. **Tier 3 (dataset-level, `locality-report`):** odd displacement response
   vs `H^LR`, longitudinal/transverse and |q| trends within matched
   comparison families, locality tail fractions — `F_SR(r) < F_full(r)`
   over long distances is the dataset-level approval requirement.

## Publication safeguards

- Main-set q shells are assigned to train/validation/test before patterns are
  generated. Pattern groups, exact ±q families, and complete |q| shells cannot
  cross subsets. Existing main structures without `split_hint` metadata must
  be regenerated.
- `organize` requires every configured ABACUS/QE pseudopotential and ABACUS
  orbital file to exist locally, hashes them, and refuses null provenance.
- LR/SR export covers every converted snapshot and is all-or-nothing. Missing
  labels, foreign targets, stale reference identities, or mixed prior exports
  fail during preflight before any target file or workspace metadata changes.
- A failed forced LR rerun removes old LR/SR labels, validation output, and any
  LR/SR training export before returning the snapshot to `converted`.

## Tests

```
/opt/anaconda3/envs/DeepH/bin/python -m pytest mgo_lr/tests -q
```

Synthetic fixtures only — no DFT binaries or network required.

Before production use, archive at least one small real ABACUS/QE pilot output
as a parser compatibility fixture and run the 18-snapshot pilot end to end on
the target cluster software versions.
