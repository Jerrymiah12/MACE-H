# Final Review of the Updated MgO LR Dataset-Generation Pipeline

## Overall Assessment

This updated version is **much stronger and nearly ready to implement**.

It now includes:

- MACE-H target export
- Grouped, leakage-safe splitting
- Explicit unit and sign conventions
- Uniform-translation removal
- Fixed-\(\Lambda\) reciprocal convergence testing
- DFT-facing locality diagnostics
- Per-atom orbital metadata
- Explicit ABACUS and Quantum ESPRESSO settings
- Strong provenance and testing

### Rating

\[
\boxed{9.5/10}
\]

Before implementation, make the following four adjustments.

---

# 1. Correct the Linearity and Sign-Reversal Assumptions

The current plan constructs the LR potential using displacement-dependent quantities, including:

- Snapshot atomic positions in the reciprocal-space phase factors
- Snapshot AO centers
- Snapshot overlap matrices

This means the final LR Hamiltonian is not exactly linear in displacement.

For example:

\[
e^{-i\mathbf G\cdot(\mathbf R_\kappa^0+\mathbf u_\kappa)}
\]

depends nonlinearly on \(\mathbf u_\kappa\).

The overlap matrix also changes with the structure:

\[
S_{ij}
=
S_{ij}(\mathbf u).
\]

Therefore:

\[
H^{\mathrm{LR}}(\mathbf u)
=
V^{\mathrm{LR}}(\mathbf u)
S(\mathbf u)
\]

will not generally satisfy exact relations such as:

\[
H^{\mathrm{LR}}(2\mathbf u)
=
2H^{\mathrm{LR}}(\mathbf u)
\]

or:

\[
H^{\mathrm{LR}}(-\mathbf u)
=
-H^{\mathrm{LR}}(\mathbf u).
\]

Second-order corrections will generally remain.

## Recommended Phase Convention

Use the reference atomic positions in the dipole-source phase:

\[
V^{\mathrm{LR}}(\mathbf G)
=
C_{\mathrm{Coul}}
\frac{4\pi i}{\Omega}
\frac{
\sum_\kappa
\mathbf G\cdot\mathbf d_\kappa
e^{-i\mathbf G\cdot\mathbf R_\kappa^0}
}{
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
}
f_{\mathrm{Ewald}}(\mathbf G).
\]

Here:

\[
\mathbf d_\kappa
=
\widetilde Z_\kappa^*
\mathbf u_\kappa^{\mathrm{rel}}.
\]

The resulting potential can still be evaluated at the current AO centers, and the current overlap matrix can still be used for the AO projection.

However, the validation should test **small-displacement first-order behavior**, not exact linearity or exact antisymmetry.

## Replace the Existing Validation Language

Replace:

```text
exact sign reversal
exact linearity
```

with:

```text
approximate sign reversal in the small-displacement limit
approximate linearity in the small-displacement limit
error decreases as displacement amplitude approaches zero
```

## Sign-Reversal Error

Define:

\[
E_{\mathrm{sign}}(A)
=
\frac{
\left\|
H^{\mathrm{LR}}(-A)
+
H^{\mathrm{LR}}(A)
\right\|_F
}{
\left\|
H^{\mathrm{LR}}(A)
\right\|_F
+
\delta
}.
\]

Evaluate this for:

\[
A,\quad \frac{A}{2},\quad \frac{A}{4}.
\]

The expected behavior is:

\[
E_{\mathrm{sign}}\left(\frac{A}{2}\right)
<
E_{\mathrm{sign}}(A),
\]

and:

\[
E_{\mathrm{sign}}\left(\frac{A}{4}\right)
<
E_{\mathrm{sign}}\left(\frac{A}{2}\right).
\]

## Linearity Error

Define:

\[
E_{\mathrm{linear}}(A)
=
\frac{
\left\|
H^{\mathrm{LR}}(2A)
-
2H^{\mathrm{LR}}(A)
\right\|_F
}{
2\left\|
H^{\mathrm{LR}}(A)
\right\|_F
+
\delta
}.
\]

This error should also decrease as the displacement amplitude becomes smaller.

## Recommended Pilot Amplitudes

Include amplitude ladders such as:

\[
0.0025,\ 0.005,\ 0.01,\ 0.02\ \text{\AA}.
\]

These provide a better test of the linear-response limit than only comparing one amplitude with twice that amplitude.

---

# 2. Separate Hard Validation Failures from Physics Diagnostics

The updated plan currently says that validation returns a nonzero exit code when any validation check fails.

That is appropriate for structural, numerical, and algebraic failures.

However, some of the listed physics expectations should not be treated as hard per-snapshot rejection conditions.

Examples include:

- Longitudinal patterns should produce a stronger LR response than transverse patterns
- Smaller-\(|\mathbf q|\) patterns should show greater LR importance
- \(H^{\mathrm{SR}}\) should be more localized than \(H^{\mathrm{full}}\)
- The LR magnitude should be sensible relative to the odd DFT response

These are useful dataset-level diagnostics, but they can depend on:

- Displacement amplitude
- Phase
- Mode normalization
- Mixed longitudinal and transverse character
- AO block selection
- Reciprocal-space sampling
- The selected value of \(\Lambda\)

## Hard Failures

The following should produce a nonzero exit code or snapshot rejection:

- NaN or infinity values
- Incorrect matrix dimensions
- Missing required files
- Failed reconstruction
- Broken Hermiticity
- Incorrect atom mapping
- Incorrect orbital count
- Incorrect indexing convention
- Nonconverged reciprocal-space sum
- Incorrect equilibrium behavior
- Incorrect uniform-translation behavior
- Invalid units
- Missing provenance
- Large imaginary residual in the LR potential
- Unpaired reciprocal vectors
- Failed SCF convergence

## Physics Warnings or Dataset-Level Approval Checks

The following should be stored as warnings or dataset-level reports:

- Longitudinal-versus-transverse trend
- Small-\(|\mathbf q|\) trend
- Magnitude relative to \(\Delta H_{\mathrm{DFT}}\)
- Locality improvement
- Selected-\(\Lambda\) usefulness
- Cosine similarity with the odd DFT response
- Relative LR contribution

These checks should not automatically reject a single snapshot unless a clearly defined physical inconsistency is found.

## Matched Comparisons

Only compare longitudinal and transverse structures when they are matched in:

\[
A,\quad
|\mathbf q|,\quad
\varphi,\quad
\text{mode normalization}.
\]

Similarly, compare different values of \(|\mathbf q|\) only within the same controlled pattern family.

Recommended metadata fields:

```json
{
  "comparison_family_id": "...",
  "mode_normalization": "...",
  "q_magnitude": 0.0,
  "polarization_class": "longitudinal",
  "amplitude": 0.01,
  "phase": 0.0
}
```

---

# 3. Correct the Real-Space Reference Test

The current testing plan proposes comparing a single dipole in a large cell with a direct real-space screened dipole sum.

The LR definition used by the project is not the complete unsplit dipole potential.

It is a Gaussian-filtered reciprocal-space component:

\[
V^{\mathrm{LR}}(\mathbf G)
\propto
f_{\mathrm{Ewald}}(\mathbf G).
\]

Therefore, an unfiltered isolated-space or periodic real-space dipole sum will not generally equal the selected LR contribution.

## Incorrect Reference Comparison

Do not compare:

\[
V^{\mathrm{LR}}_{\mathrm{filtered}}
\]

with an unfiltered direct-space expression such as:

\[
V_{\mathrm{dipole}}(\mathbf r)
\propto
\frac{
\mathbf p\cdot\mathbf r
}{
r^3
}.
\]

These represent different decompositions.

## Valid Reference-Test Options

Use one of the following.

### Option 1: Direct Inverse Discrete Fourier Transform

Construct the same reciprocal coefficients:

\[
V^{\mathrm{LR}}(\mathbf G)
\]

using a small reciprocal set, then evaluate:

\[
V^{\mathrm{LR}}(\mathbf r)
=
\sum_{\mathbf G\neq0}
V^{\mathrm{LR}}(\mathbf G)
e^{i\mathbf G\cdot\mathbf r}
\]

using a simple, slow reference implementation.

Compare that result with the optimized LR implementation.

### Option 2: Analytically Matched Filtered Real-Space Expression

Use a real-space expression derived from the same Gaussian Ewald filter.

The real-space formula must correspond exactly to:

\[
f_{\mathrm{Ewald}}(\mathbf G)
=
\exp\left[
-\frac{
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
}{
4\Lambda^2
}
\right].
\]

### Option 3: High-Accuracy Reciprocal Reference

Use a very large, inversion-symmetric reciprocal set with a tight tolerance as the reference.

Compare the production reciprocal implementation against this high-accuracy result.

## Rename the Test

Use:

```text
filtered periodic dipole reference test
```

instead of:

```text
single dipole versus direct real-space dipole sum
```

This ensures the reference and production implementations represent the same mathematical LR definition.

---

# 4. Require an Inversion-Symmetric Reciprocal Set

The plan states that the evaluated LR potential should be real up to numerical roundoff.

For this to hold numerically, the reciprocal set must be inversion symmetric.

Explicitly require:

\[
\mathbf G\in\mathcal G
\quad\Longrightarrow\quad
-\mathbf G\in\mathcal G.
\]

Also require:

\[
\mathbf G=0
\notin
\mathcal G.
\]

## Reciprocal-Set Construction

A safe procedure is:

1. Generate integer reciprocal indices:

\[
(n_1,n_2,n_3).
\]

2. Exclude:

\[
(0,0,0).
\]

3. Calculate:

\[
\mathbf G
=
n_1\mathbf b_1
+
n_2\mathbf b_2
+
n_3\mathbf b_3.
\]

4. Apply the ellipsoidal cutoff using a symmetric scalar condition such as:

\[
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
\le
G_{\max}^2.
\]

5. Verify that every accepted integer triplet has its negative triplet.

## Required Tests

Add:

```text
G_set_is_inversion_symmetric
G_zero_is_excluded
duplicate_G_vectors_absent
maximum_imaginary_residual_below_tolerance
```

## Imaginary-Residual Check

Do not automatically discard a significant imaginary component by taking only the real part.

Calculate:

\[
r_{\mathrm{imag}}
=
\frac{
\left\|
\operatorname{Im}
V^{\mathrm{LR}}
\right\|_2
}{
\left\|
\operatorname{Re}
V^{\mathrm{LR}}
\right\|_2
+
\delta
}.
\]

Require:

\[
r_{\mathrm{imag}}
<
\tau_{\mathrm{imag}}.
\]

If this condition fails:

- Mark the LR calculation as failed
- Save the reciprocal-set diagnostics
- Do not write accepted LR and SR labels
- Do not silently take the real part

Taking the real part is acceptable only after the imaginary component is confirmed to be at numerical roundoff.

## Recommended Metadata

```yaml
lr_definition:
  reciprocal_set:
    inversion_symmetric: true
    excludes_G_zero: true
    cutoff_type: dielectric_ellipsoid
    number_of_vectors: ...
  imaginary_tolerance: ...
```

---

# Revised Validation Structure

## Hard Validation Stage

The `validate` command should perform the following hard checks.

### Structural Checks

- Atom ordering unchanged
- Reference mapping correct
- Orbital counts correct
- One `orbital_types.dat` line per atom
- Matrix block dimensions correct
- One-based matrix-key indices
- Correct lattice-vector convention
- Correct reciprocal-lattice \(2\pi\) convention
- Correct energy and length units
- Valid overlap matrices
- No NaN or infinity values
- Raw DFT outputs unmodified
- Atomic HDF5 writes successful

### Algebraic Checks

#### Equilibrium

\[
\left\|
H^{\mathrm{LR}}(\mathbf u=0)
\right\|_F
<
\tau_{\mathrm{eq}}.
\]

#### Uniform Translation

After translation removal:

\[
\max_\kappa
\left\|
\mathbf u_\kappa^{\mathrm{rel}}
\right\|
<
\tau_u
\]

and:

\[
\left\|
H^{\mathrm{LR}}
\right\|_F
<
\tau_{\mathrm{translation}}.
\]

#### Hermiticity

For all real-space blocks:

\[
H_{ij}(\mathbf R)
=
\left[
H_{ji}(-\mathbf R)
\right]^*.
\]

Check this for:

\[
H^{\mathrm{full}},
\qquad
H^{\mathrm{LR}},
\qquad
H^{\mathrm{SR}}.
\]

#### Reconstruction

\[
\frac{
\left\|
H^{\mathrm{SR}}
+
H^{\mathrm{LR}}
-
H^{\mathrm{full}}
\right\|_F
}{
\left\|
H^{\mathrm{full}}
\right\|_F
+
\delta
}
<
\tau_{\mathrm{reconstruct}}.
\]

#### Reciprocal Convergence

At fixed \(\Lambda\):

\[
\frac{
\left\|
H^{\mathrm{LR}}(G_{\max,2})
-
H^{\mathrm{LR}}(G_{\max,1})
\right\|_F
}{
\left\|
H^{\mathrm{LR}}(G_{\max,2})
\right\|_F
+
\delta
}
<
\tau_G.
\]

#### Reciprocal Inversion Symmetry

Verify:

\[
\forall\mathbf G\in\mathcal G,
\quad
-\mathbf G\in\mathcal G.
\]

#### Imaginary Residual

\[
r_{\mathrm{imag}}
<
\tau_{\mathrm{imag}}.
\]

## Small-Amplitude Response Checks

Treat sign reversal and linearity as small-amplitude convergence checks.

### Sign-Reversal Convergence

\[
E_{\mathrm{sign}}(A)
=
\frac{
\left\|
H^{\mathrm{LR}}(-A)
+
H^{\mathrm{LR}}(A)
\right\|_F
}{
\left\|
H^{\mathrm{LR}}(A)
\right\|_F+\delta
}.
\]

Check that the error decreases with amplitude.

### Linearity Convergence

\[
E_{\mathrm{linear}}(A)
=
\frac{
\left\|
H^{\mathrm{LR}}(2A)
-
2H^{\mathrm{LR}}(A)
\right\|_F
}{
2\left\|
H^{\mathrm{LR}}(A)
\right\|_F+\delta
}.
\]

Check that the error decreases as \(A\rightarrow0\).

These may begin as warnings during early pilot development and become hard checks after expected tolerance ranges are established.

---

# Revised Locality and Physics Report

The `locality-report` command should produce dataset-level diagnostics rather than rejecting individual snapshots.

## Odd DFT Response

For sign-paired structures:

\[
\Delta H_{\mathrm{DFT}}(A)
=
\frac{
H^{\mathrm{full}}(+A)
-
H^{\mathrm{full}}(-A)
}{2}.
\]

Compare with:

\[
H^{\mathrm{LR}}(A).
\]

Calculate:

\[
\cos\theta
=
\frac{
\left\langle
\Delta H_{\mathrm{DFT}},
H^{\mathrm{LR}}
\right\rangle_F
}{
\left\|
\Delta H_{\mathrm{DFT}}
\right\|_F
\left\|
H^{\mathrm{LR}}
\right\|_F
}.
\]

Also calculate:

\[
r_{\mathrm{LR}}
=
\frac{
\left\|
H^{\mathrm{LR}}
\right\|_F
}{
\left\|
\Delta H_{\mathrm{DFT}}
\right\|_F
+
\delta
}.
\]

These are diagnostics and are not expected to equal one.

## Controlled Polarization Comparisons

Compare longitudinal and transverse structures only when matched in:

- \(|\mathbf q|\)
- Amplitude
- Phase
- Mode normalization
- Species displacement ratio
- Supercell

## Controlled Wavevector Comparisons

Compare small and large values of \(|\mathbf q|\) only within the same pattern family.

## Locality Tail

For:

\[
X\in
\{
\mathrm{full},
\mathrm{LR},
\mathrm{SR}
\},
\]

calculate:

\[
F_X(r)
=
\frac{
\sum_{d_{ij}>r}
\left\|
H^X_{ij}
\right\|_F^2
}{
\sum_{ij}
\left\|
H^X_{ij}
\right\|_F^2
}.
\]

The desired result is:

\[
F_{\mathrm{SR}}(r)
<
F_{\mathrm{full}}(r)
\]

over the relevant long-distance range.

This should be treated as a dataset-level approval requirement before generating the full dataset.

---

# Revised Testing Section

## Displacement Tests

- Uniform-translation removal
- Sign-pair generation
- Amplitude-ladder generation
- Commensurability
- Seeded reproducibility
- Minimum-distance rejection
- Pilot-list contents
- Pattern-group assignment
- Comparison-family assignment

## ABACUS I/O Tests

- `STRU` writer
- `INPUT` writer
- `KPT` writer
- Explicit `gamma_only 0`
- CSR parser
- Convergence-log parser
- Version-dependent CSR filename handling

## Conversion Tests

- DeepH-E3 HDF5 key format
- One-based atom indices
- One `orbital_types.dat` line per atom
- Orbital-count consistency
- Signed orbital permutation
- Lattice and reciprocal-lattice conventions
- eV and Å conversion

## LR Tests

- Uniform-translation removal inside the LR processor
- Reference-position phase convention
- Filtered periodic dipole reference test
- Inversion-symmetric reciprocal set
- Exclusion of \(\mathbf G=0\)
- Reciprocal convergence at fixed \(\Lambda\)
- Imaginary residual below tolerance
- Hermiticity
- Equilibrium zero
- Small-amplitude sign-reversal convergence
- Small-amplitude linearity convergence
- Prefactor and sign convention

## Validation Tests

Use synthetic snapshots designed to:

- Pass every hard check
- Fail reconstruction
- Fail Hermiticity
- Fail reciprocal inversion symmetry
- Fail imaginary-residual tolerance
- Fail atom mapping
- Fail orbital dimensions
- Fail unit metadata
- Trigger physics warnings without hard rejection

## Export Tests

Verify that:

- `hamiltonians.h5` matches the selected source
- Source Hamiltonian files remain unchanged
- Switching targets works
- Metadata records the active target

---

# Updated Immediate Workflow

## 1. Implement the Pipeline

Implement all modules with synthetic tests before any production DFT calculations.

## 2. Generate the Initial Pilot

Generate:

```text
12–20 pilot structures
```

Include:

- Equilibrium
- Pure translation
- Positive and negative displacement pairs
- Amplitude ladders
- Longitudinal finite-\(q\) patterns
- Transverse finite-\(q\) patterns
- Mixed modes
- Random local distortions

## 3. Run External Calculations

Run:

- ABACUS reference calculations
- QE equilibrium DFPT
- ABACUS pilot snapshot calculations

## 4. Process the Pilot

Run:

```bash
python -m mgo_lr collect-reference
python -m mgo_lr collect-dfpt
python -m mgo_lr collect-dft --set pilot
python -m mgo_lr lr-process --set pilot
python -m mgo_lr validate --set pilot
python -m mgo_lr locality-report --set pilot
```

## 5. Approve the LR Definition

Before scaling up, confirm:

- Charge and energy units are correct
- Overall sign is correct
- Reference-position phase convention is correct
- Reciprocal set is inversion symmetric
- Imaginary residual is negligible
- Reciprocal sum is converged
- Translation removal works
- Hermiticity passes
- Reconstruction passes
- Small-amplitude errors decrease toward zero
- LR response is physically sensible relative to DFT
- \(H^{\mathrm{SR}}\) is more localized than \(H^{\mathrm{full}}\)

## 6. Select and Freeze \(\Lambda\)

Compare candidate values of \(\Lambda\) using the pilot dataset.

Choose the value that produces:

- Stable reciprocal convergence
- Physically sensible LR scale
- Greatest useful localization improvement in \(H^{\mathrm{SR}}\)
- Numerically stable matrix labels

Then freeze it as part of the dataset definition.

## 7. Expand the Pilot

Expand to approximately:

```text
50 pilot structures
```

Use the expanded pilot to test:

- More \(\mathbf q\) values
- More polarizations
- More amplitudes
- More mixed modes
- Rerun behavior
- Snapshot rejection
- Grouped splitting
- Dataset-level physics trends

## 8. Generate the Main Dataset

Only after all pilot checks pass, generate:

```text
approximately 400 main structures
```

Then generate:

```text
30–50 large-cell test structures
```

## 9. Organize and Export

Run:

```bash
python -m mgo_lr organize
python -m mgo_lr export-target --target sr
```

---

# Final Assessment

The revised architecture is excellent.

Its strongest features are:

- Standalone dataset package
- External-cluster workflow
- Idempotent processing states
- Immutable raw DFT outputs
- Explicit MACE-H target export
- Fixed and documented unit convention
- Grouped leakage-safe splitting
- DFT-facing physics diagnostics
- Dedicated locality analysis
- Separate large-cell extrapolation set
- Strong provenance
- Comprehensive synthetic tests

After the four adjustments in this review, the plan is ready for implementation and for the initial:

\[
\boxed{12\text{–}20\text{ snapshot pilot}}
\]

The full dataset should only be generated after the pilot confirms:

\[
H^{\mathrm{LR}}
\]

has the correct numerical and physical behavior, and:

\[
H^{\mathrm{SR}}
=
H^{\mathrm{full}}
-
H^{\mathrm{LR}}
\]

is measurably more localized than the original full Hamiltonian.
