# MgO LR Pipeline Implementation Review

**Review date:** 2026-07-21  
**Reviewed branch:** `mgo-lr-pipeline`  
**Compared against:**

- `instructions/Final_Review_Updated_MgO_LR_Dataset_Plan (1).md`
- `docs/superpowers/specs/2026-07-20-mgo-lr-dataset-pipeline-design.md`
- `instructions/MgO_LR_Dataset_Plan_Review.md`
- `instructions/MgO_MACE-H-LR_Dataset_Generation_Plan.md`

## Verdict

The `mgo-lr-pipeline` branch is not ready for production dataset generation.

The core long-range mathematics and the synthetic happy path look sound, but several cross-stage, data-integrity, and physics-critical defects remain. These should be corrected before spending compute time on the pilot, main, or large-cell DFT datasets.

## Resolution status (updated 2026-07-28)

All findings below were remediated on `mgo-lr-pipeline` under test-driven
development. The original review text is preserved unchanged as a record.

| Finding | Status | Commit / notes |
|---|---|---|
| P0 — pilot blocks main/large processing | Fixed | `d37d43a` — cell-dependent reciprocal count removed from the workspace LR key; realized count moved to per-snapshot metadata |
| P0 — wrapped q-vectors misinterpreted | Fixed | `d37d43a` — `fold_q` canonicalises every generated q-vector to the centered interval |
| P1 — export can publish a mixed dataset | Fixed | `d37d43a` — all-or-nothing atomic pre-scan; strengthened to reject any incomplete converted scope in `c6f359e` |
| P1 — Tier-1 validation omits hard checks | Fixed | `d37d43a` + `c6f359e` — element/orbital/lattice/position/reciprocal-set/lr-definition/units/finiteness checks |
| P1 — split does not hold out q-families | Fixed | `d37d43a` (post-hoc union-find) → `c6f359e` (generation-time |q|-shell pools + `split_from_hints` cross-checks) |
| P1 — locality approval false-pass + memory | Fixed | `d37d43a` — measurable-improvement criterion; streaming one snapshot at a time |
| P1 — rejected snapshots regenerated | Fixed | `d37d43a` — `is_rejected` guard in gen-structures |
| P2 — initial pilot too large | Fixed | `c6f359e` — `pilot_expanded` config: 18-structure approval pilot (default) vs 50 |
| Provenance treated as optional | Fixed | `c6f359e` — hard preflight in organize; missing hashes abort |

Additional hardening in `c6f359e`: reference-artifact fingerprints bound into
the LR definition (`require_current_lr_definition`), transactional `--force`
reprocess, and `loader_splits/` views (with a preflight that makes
export-target a prerequisite of organize so no link dangles).

Test suite: 101 → 131 passing, synthetic fixtures only. The real-DFT
prerequisite in *Recommended disposition* still stands.

## Findings

### P0 — Pilot processing prevents main and large-cell processing

The workspace-wide LR definition includes `reciprocal_set.number_of_vectors`, which depends on the supercell size. The pipeline then compares the entire stored definition for exact equality before processing another set.

Relevant implementation:

- [`mgo_lr/lr.py`](../mgo_lr/lr.py#L129) includes the reciprocal-vector count in `_lr_definition()`.
- [`mgo_lr/lr.py`](../mgo_lr/lr.py#L151) rejects any definition that does not exactly equal the stored dictionary.
- [`mgo_lr/lr.py`](../mgo_lr/lr.py#L175) constructs the reciprocal set using the selected set's supercell.

A representative numerical check using a valid isotropic dielectric tensor produced:

| Set size | Reciprocal vectors |
|---:|---:|
| 2×2×2 pilot | 14 |
| 3×3×3 main | 58 |
| 4×4×4 large | 168 |

Consequently, after processing the pilot, processing the main set aborts with the message that `metadata.yaml` records a different LR definition, even though the physical LR parameters have not changed.

The workspace-level compatibility key should contain only invariant physical parameters. Cell-dependent values such as the reciprocal-vector count should be stored per set or per snapshot and excluded from the compatibility comparison.

### P0 — Wrapped wavevectors are interpreted incorrectly

`_low_q()` represents a negative reciprocal index as `n - 1`, but downstream code uses that raw positive integer to calculate the Cartesian wavevector, polarization, and reported magnitude.

Relevant implementation:

- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L208) samples components from `0`, `1`, and `n - 1`.
- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L223) converts the unwrapped integer vector directly to Cartesian coordinates.
- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L97) uses the same unwrapped representation for `q_magnitude` metadata.

For a 3×3×3 cell, the discrete vector `[2, 1, 0]` should be folded to `[-1, 1, 0]`. In a numerical check, the raw and folded Cartesian directions had cosine `-0.426`, and their magnitudes also differed.

This means that nominally longitudinal modes can use an incorrect physical direction, while reported `q_magnitude` values and small-|q| classifications can be wrong. It affects the main and large-cell structure generators and undermines the controlled wavevector diagnostics.

Integer q-vectors should be canonicalized into the centered reciprocal interval before calculating directions, magnitudes, polarizations, metadata, or family identities.

### P1 — Export can publish a mixed dataset under one target

`export-target` silently skips snapshots that have not reached the state required for the requested target, but it always changes the workspace-level `training_target` metadata.

Relevant implementation:

- [`mgo_lr/export.py`](../mgo_lr/export.py#L63) skips ineligible snapshots and updates metadata unconditionally.
- [`mgo_lr/tests/test_export.py`](../mgo_lr/tests/test_export.py#L45) explicitly accepts a case where an SR export processes zero snapshots and leaves an earlier full target unchanged.

For example:

1. Export `full` while snapshots are converted.
2. Request `sr` before every snapshot reaches `lr_done`.
3. Converted-only snapshots keep their existing full `hamiltonians.h5`.
4. Eligible snapshots receive SR targets.
5. `metadata.yaml` declares `training_target: sr` for the workspace.

The result is a mixed-label dataset advertised as a uniform SR dataset.

Export should be all-or-nothing for the selected scope. It should fail before changing any files if required sources are missing, or explicitly record per-snapshot targets and refuse to publish a global target until all intended snapshots agree.

### P1 — Tier-1 validation omits required hard checks

The validator checks file presence, matrix dimensions, finite values, overlap diagonals, Hermiticity, reconstruction, imaginary residual, and reciprocal convergence. It does not implement several hard checks required by the plan.

Missing or incomplete checks include:

- `element.dat` contents and atom ordering
- Snapshot positions versus the reference mapping
- `orbital_types.dat` contents, beyond its number of lines
- Energy and length unit metadata
- Required provenance and raw-file hashes
- LR-definition agreement with the configuration and workspace metadata
- Recorded reciprocal-set inversion symmetry, G=0 exclusion, and duplicate status
- Full lattice agreement with the expected supercell, rather than only checking that `lat` and `rlat` are mutually consistent

Relevant implementation:

- [`mgo_lr/validate.py`](../mgo_lr/validate.py#L55) contains the Tier-1 implementation.
- [`mgo_lr/validate.py`](../mgo_lr/validate.py#L85) checks only the number of orbital-type lines.
- [`mgo_lr/validate.py`](../mgo_lr/validate.py#L120) reads only `r_imag` and `lr_convergence` from LR metadata.

An adversarial reproduction changed `element.dat` from Mg/O to H/H and changed the recorded reciprocal set to `inversion_symmetric: false`. Validation still returned exit code 0 and marked the snapshot `validated`.

Provenance is also treated as optional during organization: missing pseudopotential or orbital files produce warnings and `null` hashes rather than the hard failure required by the reviewed plan.

### P1 — The split does not implement the planned q-family holdout

The split keeps explicit sign pairs together, but most main-set snapshots receive a unique `pattern_group_id`. The organizer then randomly shuffles these groups without holding out complete q-vectors, q-shells, polarization families, or mode families.

Relevant implementation:

- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L267) creates mostly per-snapshot main-set group IDs.
- [`mgo_lr/organize.py`](../mgo_lr/organize.py#L21) performs a simple shuffled grouped split.

For the default 400-snapshot main plan:

- 380 groups were generated.
- 16 q-vectors appeared in both train and test.
- 19 q-vectors appeared in both train and validation.

This does not meet the design requirement that held-out subsets be formed from complete q-vectors or q-shells and selected polarization or mode families. It also weakens extrapolation claims because closely related wavevector structures appear on both sides of the split.

### P1 — Locality approval can falsely pass and scales poorly

The locality verdict accepts equality:

```python
s <= f + 1e-12
```

Therefore, a dataset where `H^SR` has exactly the same tail as `H^full` can receive a `PASS`. Equality at radii beyond the largest nonzero block also contributes to the verdict. This contradicts the requirement for a measurable localization improvement.

Relevant implementation:

- [`mgo_lr/locality.py`](../mgo_lr/locality.py#L102) defines the permissive approval criterion over an arbitrary upper half of the radius grid.
- [`mgo_lr/locality.py`](../mgo_lr/locality.py#L140) calculates binned statistics only for the first validated snapshot.

The report also does not implement the requested controlled small-|q| versus large-|q| trend. It groups members by `comparison_family_id` and reports mean LR norms by polarization class, but does not construct matched wavevector comparisons.

The implementation is also not memory efficient for the planned dataset size. It loads `H^full`, `H^LR`, and `H^SR` for every validated snapshot into memory simultaneously:

- [`mgo_lr/locality.py`](../mgo_lr/locality.py#L78)

For approximately 400 main structures, this can require many gigabytes of memory. The locality calculations should stream one snapshot or one matched family at a time, retaining only aggregate statistics and the limited blocks needed for pair comparisons.

### P1 — Rejected snapshots are regenerated

Rejection moves a snapshot folder out of its set and into `rejected/`. Structure generation only checks the active set directory, so a later generation run sees the rejected snapshot ID as missing and recreates it in `prepared` state.

Relevant implementation:

- [`mgo_lr/snapshot.py`](../mgo_lr/snapshot.py#L62) moves rejected snapshots out of the active set.
- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L353) checks only whether the active snapshot folder exists.

The reproduced sequence was:

```text
generate pilot -> reject snapshot_000001 -> rerun gen-structures
```

The rerun reported one newly written snapshot, and `snapshot_000001` returned in `prepared` state. This contradicts the stated state-machine rule that rejected snapshots are never reprocessed. A subsequent rejection can also collide with the existing rejected destination.

Generation should maintain a persistent rejection registry or check the set-qualified rejected path before recreating any planned snapshot ID.

### P2 — The initial pilot contains 46 structures

The reviewed plan calls for an initial pilot of approximately 12–20 structures before expanding it to roughly 50. The implementation immediately generates 46 structures:

- One equilibrium structure
- Five pattern bases × four amplitudes × two signs = 40 structures
- Two mixed structures
- Two random-local structures
- One rigid translation

Relevant implementation and test:

- [`mgo_lr/displacements.py`](../mgo_lr/displacements.py#L141)
- [`mgo_lr/tests/test_displacements.py`](../mgo_lr/tests/test_displacements.py#L74)

This is close to the intended expanded pilot, but it substantially increases external DFT cost before the initial LR definition has been approved. The workflow should distinguish the initial 12–20 snapshot pilot from the later expanded pilot.

## What is implemented well

The following parts align well with the reviewed design:

- Reference-position phase convention in the LR source term
- Processor-level uniform-translation removal
- Acoustic-sum-rule correction for Born effective charges
- Inversion-symmetric reciprocal-set construction
- G=0 exclusion
- Imaginary-residual hard gate before taking the real part
- Fixed-Λ reciprocal convergence calculation
- Separate full, LR, and SR Hamiltonian files
- Hermiticity and reconstruction calculations
- Atomic HDF5 writes
- One-based DeepH-E3/MACE-H matrix keys
- Per-atom orbital metadata
- Explicit `gamma_only 0` and `symmetry 0` ABACUS settings
- Deterministic seeded displacement generation
- Synthetic end-to-end pipeline coverage

## Verification performed

### Full test suite

```text
/opt/anaconda3/envs/DeepH/bin/python -m pytest mgo_lr/tests -q
```

Result:

```text
101 passed in 6.68s
```

### Additional checks

- Python bytecode compilation completed successfully.
- `git diff --check master...mgo-lr-pipeline` completed successfully.
- Reciprocal-vector counts were compared across the three configured supercell sizes.
- Wrapped and centered q-vector directions and magnitudes were compared numerically.
- Validation was tested against deliberately corrupted element and reciprocal-set metadata.
- Default grouped splits were checked for q-vector overlap.
- Rejected-snapshot regeneration was reproduced.

### Limitations

No production ABACUS or Quantum ESPRESSO output was available during this review. Parser compatibility and the complete scientific workflow were therefore tested only with synthetic fixtures. A small real-output fixture should be added before trusting the pipeline on cluster-generated data.

## Recommended disposition

**Original:** Do not generate the main or large-cell dataset from this branch yet.

At minimum, fix the two P0 issues and all data-integrity P1 issues, add regression tests for the reproduced failures, and run a real 12–20 snapshot pilot through ABACUS, QE, conversion, LR processing, validation, locality analysis, organization, export, and MACE-H loading before approving the LR definition or scaling to the full dataset.

**Update (2026-07-28):** The code defects (all P0/P1 + P2) are fixed with
regression tests (`d37d43a`, `c6f359e`). The remaining gate is unchanged: a
real 12–20 snapshot pilot through the full ABACUS→QE→convert→LR→validate→
locality→organize→export→MACE-H chain before approving the LR definition or
scaling. One design choice to confirm before that run: the split holds out
whole |q| shells, so at n=3 each subset sees a small disjoint set of |q|
magnitudes (train 2, validation/test 1 each).

## Repository state during review

No source files were changed as part of the original review. The unrelated untracked archive `bulk_gold_data.tar.gz` was left untouched.
