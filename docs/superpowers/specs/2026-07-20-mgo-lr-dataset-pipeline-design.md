# MgO LR Dataset-Generation Pipeline (`mgo_lr`) — Design

**Date:** 2026-07-20
**Branch:** `mgo-lr-dataset` (off `master`)
**Source instructions:** `instructions/MgO_MACE-H-LR_Dataset_Generation_Plan.md`

## Goal

Implement the full tooling for generating the MgO training dataset for MACE-H-LR:
structure/displacement generation, ABACUS and Quantum ESPRESSO input preparation,
output parsing and conversion to the DeepH-E3/MACE-H data format, the standalone
long-range (LR) Hamiltonian processor producing `H^LR` and `H^SR = H^full − H^LR`
labels, the Section-10 validation battery, and final dataset organization.

DFT itself (ABACUS SCF, QE DFPT) runs externally on the user's cluster. This code
prepares inputs and consumes outputs; every pipeline stage is separately invokable
to match that asynchronous round-trip.

## Non-goals

- No changes to the `maceh` package; `mgo_lr` never imports it (the plan requires
  the LR processor stay separate from model code during dataset development).
- No per-snapshot DFPT: equilibrium `Z*` and `ε∞` are fixed material parameters
  for every snapshot in dataset version 1.
- No model training or graph building — the pipeline ends at processed snapshot
  folders plus dataset metadata.

## Architecture

A self-contained Python package in a new top-level folder, driven by one CLI:

```
mgo_lr/
├── __init__.py          # version string (recorded in all metadata)
├── __main__.py          # python -m mgo_lr <stage> --config ... --workspace ...
├── config.py            # YAML config load/validate; provenance snapshot
├── constants.py         # unit conversions, Coulomb constant, physical constants
├── reference.py         # rocksalt primitive cell; ABACUS reference decks; collect
├── dfpt.py              # QE pw.x/ph.x input writers; Z*, eps_inf parser + ASR
├── displacements.py     # pilot/main/large-cell displacement patterns
├── abacus_io.py         # STRU/INPUT/KPT writers; running_scf.log + CSR parsers
├── convert.py           # ABACUS matrices -> DeepH-E3 h5 format (+ orbital reorder)
├── lr.py                # screened-dipole potential; H_LR / H_SR processor
├── validate.py          # Section-10 validation battery; quality_checks.json
├── organize.py          # final tree, splits, metadata.yaml
├── snapshot.py          # snapshot state machine (status.json), workspace layout
├── configs/
│   └── mgo.yaml         # the single source of truth for all conventions
├── tests/               # pytest; synthetic fixtures, no DFT required
└── README.md            # stage-by-stage usage
```

Runtime: `/opt/anaconda3/envs/DeepH/bin/python`. Dependencies: numpy, scipy,
h5py, PyYAML, ase (all present in that env).

## Workspace

Generated data lives in a workspace directory passed per invocation
(`--workspace <dir>`), never inside the repo. Layout follows Section 13:

```
<workspace>/
├── metadata.yaml
├── reference/            # primitive.cif, reference_cell.npy, reference_positions.npy,
│                         # atomic_numbers.npy, species_order.json, orbital_types.dat,
│                         # born_effective_charges.npy, dielectric_infinity.npy,
│                         # dft_settings.yaml, qe_dfpt_output.out, abacus decks
├── pilot/                # snapshot_000001/ ...
├── main/
├── test_large_cell/
├── validation_candidates/
├── test_candidates/
├── rejected/             # failed snapshots moved here with reason
└── generation_logs/
```

Each snapshot folder progresses through states recorded in its `status.json`:
`prepared → dft_done → converted → lr_done → validated`, or `rejected` at any
point with a machine-readable reason. Stages are idempotent: they skip snapshots
already past their state unless `--force`, and never overwrite raw DFT output.

## Stages

### `init-reference` / `collect-reference`

`init-reference` builds the ideal rocksalt MgO two-atom primitive cell (lattice
guess from config) and writes ABACUS decks under `reference/abacus/`:
ecutwfc scan, k-mesh scan, `cell-relax`, and a final high-accuracy `scf` with
`out_mat_hs2 1`. `collect-reference` parses the runs, extracts the relaxed
lattice constant (config override allowed), and writes the permanent reference
artifacts listed above. `orbital_types.dat` is derived from the configured
numerical-atomic-orbital files' angular-momentum channels.

### `init-dfpt` / `collect-dfpt`

`init-dfpt` writes QE inputs at the reference geometry: a converged `pw.x` SCF
input and a `q=0` `ph.x` input with `epsil = .true.`. `collect-dfpt` parses the
ph.x output for the dielectric tensor and Born effective charges, then:

- applies the acoustic-sum-rule correction `Z̃*_κ = Z*_κ − (1/N) Σ_κ' Z*_κ'`,
- checks `Z*_Mg + Z*_O ≈ 0`, near-isotropy of `Z*_κ` and `ε∞` (tolerances in
  config; warnings vs. hard failures distinguished),
- saves `born_effective_charges.npy` `[2,3,3]`, `dielectric_infinity.npy`
  `[3,3]`, and a copy of the raw ph.x output.

### `gen-structures --set pilot|main|large`

Builds supercells of the reference primitive cell — 2×2×2 (16 atoms, pilot),
3×3×3 (54 atoms, main), 4×4×4 (128 atoms, large-cell test) — and displacement
patterns per Sections 5–6:

- General pattern: `u_κ(R_l) = Σ_m A_m e_{κm} cos(q_m·R_l + φ_m)` with
  commensurate `q`, longitudinal/transverse polarization, per-species relative
  amplitude/sign, 1–4 combined modes.
- **Pilot set:** the deterministic Section-5 list — equilibrium, ±x Mg-only,
  ±x O-only, opposite Mg/O optical, longitudinal finite-q, transverse finite-q,
  two mixed, two random-local, one rigid translation — at amplitudes
  0.005/0.01/0.02 Å. Sign pairs and amplitude pairs record explicit partner
  snapshot IDs in metadata so validation can find them.
- **Main set:** the Section-11 composition table (150 single-q optical /
  120 mixed low-q / 60 random local / 40 sign-paired calibration /
  30 near-equilibrium ≈ 400 snapshots), sampled with one global config seed and
  per-snapshot derived seeds (`seed + index`) for reproducibility.
- **Large set:** 30–50 structures favoring small q, longitudinal optical, and
  mixed long-wavelength modes within main-set amplitudes.
- Center-of-mass displacement removed from every structure except those flagged
  `rigid_translation: true`; minimum interatomic distance checked before
  acceptance.

Per snapshot it writes: ABACUS `STRU`/`INPUT`/`KPT` (fixed cell, static SCF,
`out_mat_hs2 1`), `displacements.npy` (N×3, Cartesian Å, vs. reference by
minimum image), `displacement_metadata.json` (pattern class, modes, amplitudes,
q, phases, seed, flags, partner IDs), `status.json`, plus a batch slurm job
template with a config-editable header.

### `collect-dft --set …`

For each prepared snapshot with DFT output present:

1. Parse `running_scf.log` for SCF convergence; reject non-converged runs.
2. Locate and parse the ABACUS sparse-matrix files (default
   `data-HR-sparse_SPIN0.csr` / `data-SR-sparse_SPIN0.csr`; names are
   config-driven and recorded in `status.json` since they vary by ABACUS
   version) into per-R sparse matrices.
3. Reject on: dimension mismatch vs. reference orbital count, NaN/Inf entries,
   pathological overlap diagonal, changed atom ordering.
4. Convert to the DeepH-E3/MACE-H format: `hamiltonians_full.h5` and
   `overlaps.h5` keyed by JSON `[Rx, Ry, Rz, i, j]` with **1-based** atom
   indices and dense `(norb_i, norb_j)` float64 blocks; `lat.dat` / `rlat.dat`
   (3×3, vectors as columns; `rlat` includes the 2π factor),
   `site_positions.dat` (3×N, Cartesian Å),
   `orbital_types.dat`, `element.dat`, `info.json` (`{"isspinful": false}`) —
   matching exactly what `maceh/graph.py` parses.
5. Units at this boundary: Ry → eV, Bohr → Å (DeepH ABACUS convention).
6. Apply the ABACUS → DeepH-E3 orbital ordering/sign transformation within each
   l channel — a fixed per-l permutation/sign table, isolated in one function
   and unit-tested, since a silent error here corrupts every matrix.

`hamiltonians_full.h5` is never overwritten by later stages.

### `lr-process --set …`

The standalone LR processor (Section 9), all in eV/Å units
(Coulomb constant e²/4πε₀ = 14.399645 eV·Å):

1. Displacements `u_κ` from minimum-image differences vs. the reference mapping.
2. Induced dipoles `p_κ = e Z̃*_κ u_κ` with the ASR-corrected equilibrium Born
   charges.
3. Screened periodic dipole potential on the supercell reciprocal lattice:
   `φ^LR(G) = (4πi/Ω) [Σ_κ G·p_κ e^{−iG·R_κ}] / (G·ε∞·G) · f_Ewald(G)` for
   `G ≠ 0`; the `G = 0` component is zero (fixed gauge, recorded in metadata).
   `f_Ewald(G) = exp(−(G·ε∞·G)/(4Λ²))` with one fixed damping Λ for the whole
   dataset (config; recorded in metadata). The G-sum runs over an ellipsoidal
   shell defined by a tolerance on `f_Ewald`.
4. φ evaluated at the snapshot atom positions (the actual AO centers); the AO →
   atom mapping comes from `orbital_types.dat` + `element.dat`. The result is
   real up to roundoff (asserted, then the real part is taken).
5. `H^LR_ij(R) = −e (φ_i + φ_j)/2 · S_ij(R)` blockwise over every stored
   overlap key — hermiticity is inherited from S. `H^SR = H^full − H^LR` on the
   union of key sets, absent blocks treated as zero.
6. Write `hamiltonians_lr.h5` and `hamiltonians_sr.h5` atomically (temp file +
   rename); record Λ, gauge, G-cutoff, and code version in the snapshot
   metadata.

### `validate --set …`

The Section-10 battery over processed snapshots, tolerances from config,
results per snapshot in `quality_checks.json` plus a set-level summary in
`generation_logs/`; nonzero exit code if any check fails:

- **Equilibrium:** `‖H^LR(u=0)‖ ≈ 0`.
- **Sign reversal:** `H^LR(−u) ≈ −H^LR(u)` via recorded partner IDs.
- **Linearity:** `H^LR(2A) ≈ 2 H^LR(A)` for recorded amplitude pairs.
- **Rigid translation:** LR correction ≈ 0 for flagged structures (exact after
  ASR, up to numerics).
- **Hermiticity:** `H_ij(R) = [H_ji(−R)]*` for full, LR, and SR.
- **Reconstruction:** `H^SR + H^LR = H^full` to floating-point accuracy.
- **Consistency:** orbital counts and index maps vs. reference, identical
  orbital sets across structures, unit and lattice-convention checks, atom
  ordering unchanged.

### `organize`

Assembles the final Section-13 tree, splits `validation_candidates/` and
`test_candidates/` from the main set (config fractions, seeded), and writes the
top-level `metadata.yaml` with every provenance field the plan lists: XC
functional, pseudopotential and orbital file names **and sha256 hashes**, basis
spec, k-mesh, cutoffs, SCF tolerance, units, reference ID, supercell matrices,
atom-ordering convention, potential gauge, Ewald Λ, random seed, ABACUS/QE
versions (user-supplied in config), and `mgo_lr` code version.

### `status`

Scans a workspace and prints per-set counts by state, including rejects with
reasons — the quick health check between cluster round-trips.

## Configuration

`mgo_lr/configs/mgo.yaml` is the single source of truth: lattice-constant
guess, pseudopotential/orbital file names (paths valid on the cluster), ecut,
k-point density, SCF tolerance, supercell definitions, amplitude lists,
composition table, global seed, Ewald Λ and G-tolerance, minimum-distance
threshold, validation tolerances, CSR filenames, slurm header. Stages refuse to
run if required config fields are missing. A copy of the resolved config is
stored in `generation_logs/` on every invocation (provenance).

## Error handling

- Parsers fail loudly with file and line context; every matrix ingestion checks
  for NaN/Inf (consistent with the recent EPC-code conventions in this repo).
- Rejection is a first-class outcome: the snapshot moves to `rejected/` with a
  machine-readable reason in `status.json`; it is never silently dropped.
- All h5 writes are atomic (unique temp file + rename).
- Raw DFT outputs are read-only inputs; no stage modifies them.

## Testing

pytest under `mgo_lr/tests/`, run with the DeepH env python; no DFT anywhere:

- **Displacements:** COM removal, exact sign-pair negation, commensurability,
  seeded reproducibility, min-distance rejection, pilot list contents.
- **ABACUS I/O:** STRU/INPUT/KPT writers round-trip against a reference parse;
  CSR parser against a small handcrafted fixture; convergence-log parsing.
- **Conversion:** h5 key/format contract matches `maceh/graph.py`'s parsing
  (format-level test, no maceh import); orbital-reorder transform is a valid
  signed permutation and matches a hand-checked reference case.
- **LR physics:** single dipole in a large cell vs. direct real-space screened
  dipole sum (isotropic ε∞); exact linearity and antisymmetry in u; hermiticity;
  rigid-translation zero after ASR; Λ-independence of the converged G-sum
  (two Λ values, tight tolerance).
- **Validation suite:** exercised on synthetic snapshot folders with fabricated
  H/S designed to pass and to fail each check.

## Immediate workflow after implementation

1. `init-reference` → run ABACUS decks on cluster → `collect-reference`.
2. `init-dfpt` → run QE on cluster → `collect-dfpt`.
3. `gen-structures --set pilot` → run ABACUS → `collect-dft` → `lr-process` →
   `validate`.
4. Only after the pilot passes: `gen-structures --set main` (and `--set large`),
   same cycle, then `organize`.
