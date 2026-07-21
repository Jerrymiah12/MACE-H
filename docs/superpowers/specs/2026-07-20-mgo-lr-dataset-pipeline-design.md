# MgO LR Dataset-Generation Pipeline (`mgo_lr`) — Design

**Date:** 2026-07-20 (revision 2, after final external review)
**Branch:** `mgo-lr-dataset` (off `master`)
**Source instructions:** `instructions/MgO_MACE-H-LR_Dataset_Generation_Plan.md`
**Reviews:** `instructions/MgO_LR_Dataset_Plan_Review.md` (10 items) and
`instructions/Final_Review_Updated_MgO_LR_Dataset_Plan (1).md` (4 adjustments)
— all incorporated.

## Goal

Implement the full tooling for generating the MgO training dataset for MACE-H-LR:
structure/displacement generation, ABACUS and Quantum ESPRESSO input preparation,
output parsing and conversion to the DeepH-E3/MACE-H data format, the standalone
long-range (LR) Hamiltonian processor producing `H^LR` and `H^SR = H^full − H^LR`
labels, the validation battery (hard structural/numerical/algebraic checks,
small-amplitude response checks, and dataset-level physics diagnostics),
locality diagnostics, training-target export, and final dataset organization.

DFT itself (ABACUS SCF, QE DFPT) runs externally on the user's cluster. This code
prepares inputs and consumes outputs; every pipeline stage is separately invokable
to match that asynchronous round-trip.

## Non-goals

- No changes to the `maceh` package; `mgo_lr` never imports it (the plan requires
  the LR processor stay separate from model code during dataset development).
- No per-snapshot DFPT: equilibrium `Z*` and `ε∞` are fixed material parameters
  for every snapshot in dataset version 1.
- No model training or graph building — the pipeline ends at processed snapshot
  folders plus dataset metadata (with `hamiltonians.h5` exported ready for the
  maceh loader).

## Architecture

A self-contained Python package in a new top-level folder, driven by one CLI:

```
mgo_lr/
├── __init__.py          # version string (recorded in all metadata)
├── __main__.py          # python -m mgo_lr <stage> --config ... --workspace ...
├── config.py            # YAML config load/validate; provenance snapshot
├── constants.py         # unit conversions, Coulomb constant, sign conventions
├── reference.py         # rocksalt primitive cell; ABACUS reference decks; collect
├── dfpt.py              # QE pw.x/ph.x input writers; Z*, eps_inf parser + ASR
├── displacements.py     # pilot/main/large-cell displacement patterns + groups
├── abacus_io.py         # STRU/INPUT/KPT writers; running_scf.log + CSR parsers
├── convert.py           # ABACUS matrices -> DeepH-E3 h5 format (+ orbital reorder)
├── lr.py                # screened-dipole potential; H_LR / H_SR processor
├── validate.py          # hard checks + small-amplitude checks; quality_checks.json
├── locality.py          # dataset-level physics + locality diagnostics
├── organize.py          # final tree, grouped splits, metadata.yaml
├── export.py            # export-target: hamiltonians.h5 from full/lr/sr
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
├── test_large_cell/      # 4x4x4 extrapolation set, never mixed into main
├── validation_candidates/
├── test_candidates/
├── rejected/             # failed snapshots moved here with reason
└── generation_logs/      # incl. locality/ reports and resolved-config copies
```

Each snapshot folder progresses through states recorded in its `status.json`:
`prepared → dft_done → converted → lr_done → validated`, or `rejected` at any
point with a machine-readable reason. Stages are idempotent: they skip snapshots
already past their state unless `--force`, and never overwrite raw DFT output.

## Stages

```
python -m mgo_lr init-reference | collect-reference
python -m mgo_lr init-dfpt | collect-dfpt
python -m mgo_lr gen-structures | collect-dft | lr-process | validate
                 | locality-report   --set pilot|main|large
python -m mgo_lr organize
python -m mgo_lr export-target --target full|lr|sr
python -m mgo_lr status
```

### `init-reference` / `collect-reference`

`init-reference` builds the ideal rocksalt MgO two-atom primitive cell (lattice
guess from config) and writes ABACUS decks under `reference/abacus/`:
ecutwfc scan, k-mesh scan, `cell-relax`, and a final high-accuracy `scf` with
`out_mat_hs2 1`. `collect-reference` parses the runs, extracts the relaxed
lattice constant (config override allowed), and writes the permanent reference
artifacts listed above. `orbital_types.dat` is derived from the configured
numerical-atomic-orbital files' angular-momentum channels and written **one
line per atom** (a 16-atom supercell has 16 lines; each atom gets its species'
l-channel line), matching `maceh/graph.py::load_orbital_types`.

### `init-dfpt` / `collect-dfpt`

`init-dfpt` writes QE inputs at the reference geometry: a converged `pw.x` SCF
input and a `q=0` `ph.x` input with **both flags explicit**:

```
epsil = .true.
trans = .true.
```

QE and ABACUS setups must stay consistent: same relaxed lattice vectors and
positions, same XC functional, same valence configurations, same relativistic
treatment, same charge and spin state; prefer the same UPF pseudopotential
files in both codes where supported.

`collect-dfpt` parses the ph.x output and:

- verifies the output contains Born effective charges, the electronic
  dielectric tensor, the expected Mg+O atom count, and 3×3 Cartesian tensors,
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
  two mixed, two random-local, one rigid translation — with **amplitude
  ladders** `0.0025 / 0.005 / 0.01 / 0.02 Å` on the sign-paired patterns so the
  small-displacement limit can be probed (not just A vs. 2A).
- **Main set:** the Section-11 composition table (150 single-q optical /
  120 mixed low-q / 60 random local / 40 sign-paired calibration /
  30 near-equilibrium ≈ 400 snapshots), sampled with one global config seed and
  per-snapshot derived seeds (`seed + index`) for reproducibility.
- **Large set:** 30–50 structures favoring small q, longitudinal optical, and
  mixed long-wavelength modes within main-set amplitudes.
- **Uniform-translation removal** (plain mean over atoms — deliberately not
  mass-weighted, and named accordingly throughout the code) applied to every
  structure except those flagged `rigid_translation: true`; minimum
  interatomic distance checked before acceptance.

Per snapshot it writes: ABACUS `STRU`/`INPUT`/`KPT` (fixed cell, static SCF,
`out_mat_hs2 1`, and **`gamma_only 0` always explicit** — ABACUS's gamma-only
algorithm does not support `out_mat_hs2`, so the general k-point algorithm is
forced even for Γ-point runs; config field `abacus.gamma_only_algorithm:
false`), `displacements.npy` (N×3, Cartesian Å, vs. reference by minimum
image), `displacement_metadata.json`, `status.json`, plus a batch slurm job
template with a config-editable header.

`displacement_metadata.json` carries both the **pattern-group identity** used
for leakage-safe splitting and the **comparison-family identity** used for
matched physics diagnostics:

```json
{
  "pattern_group_id": "...",
  "pattern_class": "...",
  "comparison_family_id": "...",
  "mode_normalization": "...",
  "q_vectors": [], "q_magnitude": 0.0,
  "polarizations": [], "polarization_class": "longitudinal",
  "phases": [], "phase": 0.0,
  "amplitudes": [], "amplitude": 0.01,
  "sign_partner_id": "...",
  "amplitude_partner_ids": [],
  "rigid_translation": false,
  "seed": 0
}
```

Snapshots sharing the same displacement family — same base q, polarization,
longitudinal/transverse class, phase family, mode mixture, or sign/amplitude
partnership — share one `pattern_group_id`. Longitudinal/transverse and
small-q/large-q physics comparisons are only made within a
`comparison_family_id`, i.e. between structures matched in `|q|`, amplitude,
phase, mode normalization, species displacement ratio, and supercell.

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
   `orbital_types.dat` (one line per atom; the converter verifies line count ==
   atom count and that each line's orbital count matches the corresponding
   block dimensions), `element.dat`, `info.json` (`{"isspinful": false}`) —
   matching exactly what `maceh/graph.py` parses.
5. Units at this boundary: Ry → eV, Bohr → Å (DeepH ABACUS convention).
6. Apply the ABACUS → DeepH-E3 orbital ordering/sign transformation within each
   l channel — a fixed per-l permutation/sign table, isolated in one function
   and unit-tested, since a silent error here corrupts every matrix.

`hamiltonians_full.h5` is never overwritten by later stages.

### `lr-process --set …`

The standalone LR processor (Section 9 as amended by both reviews).

**Unit and sign convention** (documented in `constants.py`, `lr.py`, and
`metadata.yaml`; there is exactly one place where the Coulomb prefactor and
sign enter):

```
Z*    dimensionless (units of electron charge)
u     Å
d_κ   = Z̃*_κ · u_κ^rel        (Å)
V_LR  electron potential energy (eV)
H_LR  eV
S     dimensionless
C_Coul = e²/4πε₀ = 14.399645 eV·Å
```

Steps:

1. Displacements `u_κ` from minimum-image differences vs. the reference
   mapping.
2. **Uniform-translation removal inside the processor** (independent of what
   structure generation did): `u_κ^rel = u_κ − (1/N) Σ_κ' u_κ'`. This makes
   `H^LR` exactly zero for pure rigid translations by construction — the
   Born-charge ASR alone does *not* guarantee this at finite G, because the
   species- and position-dependent phase factors in
   `Σ_κ G·Z*_κ a e^{−iG·R_κ}` do not cancel.
3. Dipole moments `d_κ = Z̃*_κ u_κ^rel` with the ASR-corrected equilibrium Born
   charges (no factor of e here).
4. **Reciprocal set 𝒢:** integer triplets `(n1,n2,n3)` on the supercell
   reciprocal lattice, `(0,0,0)` excluded, accepted by the symmetric
   ellipsoidal cutoff `G·ε∞·G ≤ G_max²` (equivalently a tolerance on
   `f_Ewald`). The construction guarantees and a runtime check verifies:
   **inversion symmetry** (`G ∈ 𝒢 ⇒ −G ∈ 𝒢`), `G = 0 ∉ 𝒢`, and no duplicate
   vectors. These are hard requirements — the realness of `V^LR` depends on
   them.
5. Screened periodic potential, directly as electron potential energy, with
   the **reference-position phase convention**:
   `V^LR(G) = C_Coul · (4πi/Ω) · [Σ_κ G·d_κ e^{−iG·R⁰_κ}] / (G·ε∞·G) · f_Ewald(G)`
   for `G ∈ 𝒢`; the `G = 0` component is zero (fixed gauge). Reference
   positions `R⁰_κ` in the dipole-source phase make `V^LR` exactly linear in
   `u^rel`. No further factor of e is applied anywhere. The overall sign
   follows from the Fourier convention and the electron-energy definition; it
   is pinned by a dedicated unit test and confirmed against ±displacement DFT
   pairs during pilot validation.
   `f_Ewald(G) = exp(−(G·ε∞·G)/(4Λ²))` with one fixed damping Λ for the whole
   dataset. **Λ is part of the dataset definition**, not a numerical
   convergence knob: the damped reciprocal-space sum alone *is* the LR
   definition (there is no compensating real-space term), so `H^LR` depends on
   Λ by construction. During pilot development several Λ values may be
   compared to pick the one giving the most localized `H^SR`; after that, one
   value is fixed and recorded.
6. `V^LR(r) = Σ_{G∈𝒢} V^LR(G) e^{iG·r}` evaluated at the **snapshot** atom
   positions (the actual AO centers); the AO → atom mapping comes from
   `orbital_types.dat` + `element.dat`. **Imaginary-residual hard check:**
   `r_imag = ‖Im V^LR‖₂ / (‖Re V^LR‖₂ + δ) < τ_imag`. If it fails, the LR
   calculation is marked failed, reciprocal-set diagnostics are saved, and no
   LR/SR labels are written — the imaginary part is never silently discarded.
   Only after the check passes is the real part taken.
7. `H^LR_ij(R) = (V_i + V_j)/2 · S_ij(R)` blockwise over every stored overlap
   key — hermiticity is inherited from S. `H^SR = H^full − H^LR` on the union
   of key sets, absent blocks treated as zero.
   Note: because the potential is evaluated at snapshot AO centers and
   projected with the snapshot overlap `S(u)`, `H^LR(u)` is linear in `u` only
   to first order — second-order corrections remain even with the
   reference-position phase convention. Sign reversal and linearity are
   therefore **small-amplitude convergence properties**, not exact identities
   (see `validate`).
8. Write `hamiltonians_lr.h5` and `hamiltonians_sr.h5` atomically (temp file +
   rename); record the LR definition in snapshot metadata and `metadata.yaml`:

```yaml
lr_definition:
  ewald_lambda: ...
  reciprocal_cutoff: ...
  reciprocal_tolerance: ...
  reciprocal_set:
    inversion_symmetric: true
    excludes_G_zero: true
    cutoff_type: dielectric_ellipsoid
    number_of_vectors: ...
  imaginary_tolerance: ...
  gauge: G_zero_equals_zero
  sign_convention: electron_potential_energy
  phase_convention: reference_positions
```

### `validate --set …`

Results per snapshot in `quality_checks.json` plus a set-level summary in
`generation_logs/`. Checks are split into three tiers; only the first tier
causes nonzero exit / snapshot rejection.

**Tier 1 — hard checks** (nonzero exit, snapshot rejection):

- *Structural/numerical:* NaN/Inf anywhere; matrix block dimensions;
  missing required files; atom ordering / reference mapping; orbital counts;
  `orbital_types.dat` one line per atom; 1-based key indices; lattice column
  convention; 2π in `rlat`; eV/Å units metadata; valid overlap matrices;
  raw DFT outputs unmodified; atomic HDF5 writes; missing provenance;
  failed SCF convergence.
- *Algebraic:* equilibrium `‖H^LR(u=0)‖_F < τ_eq`; uniform translation
  (uniform displacement detected, `max_κ ‖u_κ^rel‖ < τ_u`, dipoles ≈ 0,
  `‖H^LR‖_F < τ_translation`); hermiticity `H_ij(R) = [H_ji(−R)]*` for full,
  LR, and SR; reconstruction
  `‖H^SR + H^LR − H^full‖_F / (‖H^full‖_F + δ) < τ_reconstruct`;
  reciprocal-sum convergence at fixed Λ
  (`‖H^LR(G_max,2) − H^LR(G_max,1)‖_F / (‖H^LR(G_max,2)‖_F + δ) < τ_G`;
  Λ-independence is **not** tested); reciprocal-set inversion symmetry;
  imaginary residual `r_imag < τ_imag`; unpaired reciprocal vectors.

**Tier 2 — small-amplitude response checks** (start as warnings during early
pilot development; promoted to hard checks once tolerance ranges are
established from pilot data):

- Sign-reversal convergence:
  `E_sign(A) = ‖H^LR(−A) + H^LR(A)‖_F / (‖H^LR(A)‖_F + δ)` evaluated along the
  amplitude ladder (A, A/2, A/4 …) must **decrease with decreasing A**.
- Linearity convergence:
  `E_linear(A) = ‖H^LR(2A) − 2H^LR(A)‖_F / (2‖H^LR(A)‖_F + δ)` must likewise
  decrease as A → 0.

**Tier 3 — physics diagnostics** (never per-snapshot rejections; stored as
warnings and dataset-level reports via `locality-report`): longitudinal vs.
transverse LR strength, small-|q| trend, LR magnitude vs. the odd DFT
response, locality improvement, Λ usefulness. These depend on amplitude,
phase, normalization, and Λ, so they are evaluated only within matched
`comparison_family_id` groups and feed the dataset-level approval decision,
not individual snapshot acceptance.

### `locality-report --set …`

Dataset-level diagnostics (never per-snapshot rejection), saved under
`generation_logs/locality/`:

- **Odd displacement response:** for recorded ± pairs,
  `ΔH_DFT(A) = (H^full(+A) − H^full(−A))/2`, compared against `H^LR(A)` via
  `cos θ = ⟨ΔH_DFT, H^LR⟩_F / (‖ΔH_DFT‖_F ‖H^LR‖_F)` and
  `r_LR = ‖H^LR‖_F / (‖ΔH_DFT‖_F + δ)`. Diagnostic only (ΔH_DFT = ΔH^SR +
  H^LR, so no exact match expected); checks sign, scale, small-A behavior.
- **Controlled comparisons:** longitudinal vs. transverse and small-|q| vs.
  large-|q| trends computed only within matched `comparison_family_id` groups
  (same |q|, amplitude, phase, mode normalization, species ratio, supercell).
- **Locality tail:** block norms of `H^full`, `H^LR`, `H^SR` binned by
  AO-center distance (mean/median/max norm, nonzero-block counts) and the
  cumulative tail fraction
  `F_X(r) = Σ_{d>r} ‖H^X_ij‖_F² / Σ ‖H^X_ij‖_F²` for X ∈ {full, LR, SR}.
  `F_SR(r) < F_full(r)` over the relevant long-distance region is a
  **dataset-level approval requirement before generating the full dataset**,
  not a per-snapshot check.

### `organize`

Assembles the final Section-13 tree and performs **grouped, leakage-safe
splits**: snapshots are never split individually — all members of one
`pattern_group_id` (sign partners, amplitude ladders, phase families, mode
mixtures) land in the same subset. Held-out subsets are formed from complete
q-vectors/q-shells, selected polarization families, selected mode mixtures,
and some high-amplitude groups. The 4×4×4 set stays entirely separate as the
large-cell extrapolation set.

Writes the top-level `metadata.yaml` with every provenance field the plan
lists — XC functional, basis spec, k-mesh, cutoffs, SCF tolerance, units,
reference ID, supercell matrices, atom-ordering convention, the `lr_definition`
block, random seed, ABACUS/QE versions (user-supplied in config), `mgo_lr`
code version — with **separate hash sections** per code:

```yaml
provenance:
  abacus:
    pseudopotentials: {name: sha256, ...}
    orbitals: {name: sha256, ...}
  quantum_espresso:
    pseudopotentials: {name: sha256, ...}
```

### `export-target --target full|lr|sr`

The maceh loader (`maceh/data.py`) reads the training target from
`hamiltonians.h5`. This stage materializes `hamiltonians.h5` from the selected
source (`hamiltonians_full.h5` / `hamiltonians_lr.h5` / `hamiltonians_sr.h5`)
per snapshot, via symlink where supported, else atomic copy. The three source
files are never overwritten or renamed; keys stay in the `[Rx, Ry, Rz, i, j]`
1-based format. The selected target is recorded in `metadata.yaml`. Re-running
with a different target replaces only `hamiltonians.h5`.

### `status`

Scans a workspace and prints per-set counts by state, including rejects with
reasons — the quick health check between cluster round-trips.

## Configuration

`mgo_lr/configs/mgo.yaml` is the single source of truth: lattice-constant
guess, pseudopotential/orbital file names (paths valid on the cluster), ecut,
k-point density, SCF tolerance, `abacus.gamma_only_algorithm: false`, supercell
definitions, amplitude ladders, composition table, global seed, Ewald Λ,
G-tolerance and imaginary tolerance, minimum-distance threshold, validation
tolerances per tier, CSR filenames, slurm header, ABACUS/QE version strings.
Stages refuse to run if required config fields are missing. A copy of the
resolved config is stored in `generation_logs/` on every invocation
(provenance).

## Error handling

- Parsers fail loudly with file and line context; every matrix ingestion checks
  for NaN/Inf (consistent with the recent EPC-code conventions in this repo).
- Rejection is a first-class outcome: the snapshot moves to `rejected/` with a
  machine-readable reason in `status.json`; it is never silently dropped.
- A failed imaginary-residual check aborts label writing entirely and saves
  reciprocal-set diagnostics.
- All h5 writes are atomic (unique temp file + rename).
- Raw DFT outputs are read-only inputs; no stage modifies them.

## Testing

pytest under `mgo_lr/tests/`, run with the DeepH env python; no DFT anywhere:

- **Displacements:** uniform-translation removal, sign-pair generation,
  amplitude-ladder generation, commensurability, seeded reproducibility,
  min-distance rejection, pilot list contents, pattern-group assignment,
  comparison-family assignment.
- **ABACUS I/O:** STRU/INPUT/KPT writers round-trip against a reference parse
  (including explicit `gamma_only 0`); CSR parser against a small handcrafted
  fixture; convergence-log parsing; version-dependent CSR filename handling.
- **Conversion:** h5 key/format contract matches `maceh/graph.py`'s parsing
  (format-level test, no maceh import); 1-based indices; per-atom
  `orbital_types.dat` verification; orbital-count consistency; orbital-reorder
  transform is a valid signed permutation and matches a hand-checked reference
  case; lattice/reciprocal conventions; eV/Å conversion.
- **LR physics:**
  - **Filtered periodic dipole reference test** — the production implementation
    against a deliberately slow reference that builds the *same* filtered
    reciprocal coefficients `V^LR(G)` on a small 𝒢 and evaluates
    `Σ_G V^LR(G) e^{iG·r}` directly. No comparison against an *unfiltered*
    real-space dipole sum — that is a different decomposition.
  - Reciprocal-set properties: inversion symmetry, `G=0` exclusion, no
    duplicates; imaginary residual below tolerance on valid sets and hard
    failure on a deliberately broken (asymmetric) set.
  - Reference-position phase convention; exact linearity of `V^LR` in `u^rel`
    (potential level, fixed evaluation points); prefactor and sign convention
    pinned numerically.
  - Small-amplitude convergence of `E_sign(A)` and `E_linear(A)` on synthetic
    snapshots with u-dependent S; uniform-translation exact zero; equilibrium
    zero; hermiticity; G_max convergence at fixed Λ — no Λ-independence test.
- **Validation suite + locality:** synthetic snapshot folders designed to pass
  every hard check and to fail each one individually (reconstruction,
  hermiticity, inversion symmetry, imaginary residual, atom mapping, orbital
  dimensions, unit metadata), plus cases that trigger Tier-3 physics warnings
  *without* hard rejection; `F_SR < F_full` locality computation.
- **Export:** `hamiltonians.h5` matches the selected source; source files
  unchanged; target switching; metadata records the active target.

## Immediate workflow after implementation

1. **Implement the pipeline** (all stages above) with synthetic tests passing
   before any DFT.
2. **Initial pilot (12–20 structures):** equilibrium, pure translation, ±
   pairs, amplitude ladders, longitudinal and transverse finite-q, mixed
   modes, random local distortions. Run ABACUS reference + QE DFPT + pilot
   ABACUS on the cluster, then `collect-reference`, `collect-dfpt`,
   `collect-dft`, `lr-process`, `validate`, `locality-report`.
3. **Approve the LR definition** before scaling: units and overall sign
   confirmed against DFT ± pairs; reference-position phase convention
   verified; reciprocal set inversion-symmetric with negligible imaginary
   residual; reciprocal sum converged; translation removal working;
   hermiticity and reconstruction passing; `E_sign`/`E_linear` decreasing
   toward zero with amplitude; LR response physically sensible vs. DFT;
   `H^SR` measurably more localized than `H^full`.
4. **Select and freeze Λ:** compare candidate Λ values on the pilot — stable
   reciprocal convergence, sensible LR scale, greatest useful `H^SR`
   localization, numerically stable labels — then freeze it as part of the
   dataset definition.
5. **Expand the pilot to ~50 structures** to exercise more q vectors,
   polarizations, amplitudes, mixed modes, reruns, rejection handling, grouped
   splitting, and dataset-level physics trends.
6. **Main dataset** (~400, 3×3×3) and **large-cell set** (30–50, 4×4×4), then
   `organize` and `export-target --target sr`.
