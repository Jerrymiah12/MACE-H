# Review of the MgO LR Dataset-Generation Pipeline

## Overall Assessment

**Yes, the plan is strong and well organized.**

The staged CLI, idempotent snapshot state machine, provenance tracking, version-aware ABACUS parsing, immutable raw outputs, and synthetic tests are all good software-engineering choices.

The scope is also clear:

- Generate MgO reference and displaced structures
- Prepare ABACUS and Quantum ESPRESSO inputs
- Consume externally generated DFT and DFPT outputs
- Convert the Hamiltonian and overlap matrices to the DeepH-E3/MACE-H format
- Construct \(H^{\mathrm{LR}}\)
- Construct the residual Hamiltonian

\[
H^{\mathrm{SR}}
=
H^{\mathrm{full}}
-
H^{\mathrm{LR}}
\]

- Validate the resulting dataset
- Stop before model training

The plan should **not launch the full 400-snapshot dataset yet**. Several technical changes should be made and tested on the pilot dataset first.

---

# Required Changes Before Full Dataset Generation

## 1. Add a MACE-H Target Export Step

The plan correctly preserves three separate Hamiltonian files:

```text
hamiltonians_full.h5
hamiltonians_lr.h5
hamiltonians_sr.h5
```

This is good because the original DFT Hamiltonian and the derived LR and SR matrices should remain separately available.

However, the current MACE-H data loader expects the active training target to be named:

```text
hamiltonians.h5
```

The stored matrix keys should remain in the format:

```text
[Rx, Ry, Rz, i, j]
```

with one-based atom indices, matching the DeepH-E3/MACE-H convention.

Add an export command such as:

```bash
python -m mgo_lr export-target \
    --workspace <workspace> \
    --set main \
    --target sr
```

The export step should create:

```text
hamiltonians.h5
```

from:

```text
hamiltonians_sr.h5
```

Use either an atomic copy or a symbolic link when supported.

The original files should never be overwritten:

```text
hamiltonians_full.h5
hamiltonians_lr.h5
hamiltonians_sr.h5
```

Recommended behavior:

```text
export-target --target full
    hamiltonians.h5 -> hamiltonians_full.h5

export-target --target lr
    hamiltonians.h5 -> hamiltonians_lr.h5

export-target --target sr
    hamiltonians.h5 -> hamiltonians_sr.h5
```

The selected target should be recorded in the dataset metadata.

---

## 2. Correct the Ewald-\(\Lambda\) Validation

The plan currently defines a Gaussian-damped reciprocal-space contribution:

\[
f_{\mathrm{Ewald}}(\mathbf G)
=
\exp\left[
-\frac{
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
}{
4\Lambda^2
}
\right]
\]

and proposes testing whether \(H^{\mathrm{LR}}\) is independent of \(\Lambda\).

That is not the correct validation for the selected decomposition.

The Ewald parameter controls how the total interaction is divided into short-range and long-range parts. The complete real-space plus reciprocal-space Ewald sum should be independent of the splitting parameter, but either component by itself generally depends on the chosen value.

Because the project intentionally uses the damped reciprocal-space contribution as the definition of \(H^{\mathrm{LR}}\), \(\Lambda\) is part of the dataset definition.

### Replace the old test

Do not require:

\[
H^{\mathrm{LR}}(\Lambda_1)
\approx
H^{\mathrm{LR}}(\Lambda_2).
\]

Instead, use the following test at fixed \(\Lambda\):

1. Select one value of \(\Lambda\).
2. Increase the reciprocal-space cutoff.
3. Recalculate \(H^{\mathrm{LR}}\).
4. Verify that the matrix converges numerically.

For example:

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
\right\|_F+\delta
}
<
\tau_G.
\]

Record the following in the metadata:

```yaml
lr_definition:
  ewald_lambda: ...
  reciprocal_cutoff: ...
  reciprocal_tolerance: ...
  gauge: G_zero_equals_zero
```

During pilot development, several values of \(\Lambda\) may be compared to determine which one produces the most localized residual Hamiltonian. Once selected, use one fixed value throughout the complete dataset.

---

## 3. Correct the Rigid-Translation Treatment

The plan removes the mean displacement from most generated structures but leaves the displacement intact for structures labeled as rigid translations.

It then assumes that the Born-charge acoustic sum rule alone guarantees:

\[
H^{\mathrm{LR}}\approx0.
\]

That is not generally guaranteed for all finite reciprocal vectors because the atomic phase factors remain species- and position-dependent:

\[
\sum_\kappa
\mathbf G\cdot Z_\kappa^*\mathbf a
e^{-i\mathbf G\cdot\mathbf R_\kappa}.
\]

Even when:

\[
\sum_\kappa Z_\kappa^*=0,
\]

the finite-\(\mathbf G\) expression does not necessarily vanish from that condition alone.

### Required solution

Always remove uniform translation before constructing the induced dipoles:

\[
\overline{\mathbf u}
=
\frac{1}{N}
\sum_{\kappa=1}^{N}
\mathbf u_\kappa,
\]

\[
\mathbf u_\kappa^{\mathrm{rel}}
=
\mathbf u_\kappa
-
\overline{\mathbf u}.
\]

Then calculate:

\[
\mathbf p_\kappa
=
eZ_\kappa^*\mathbf u_\kappa^{\mathrm{rel}}.
\]

For a pure rigid translation:

\[
\mathbf u_\kappa=\mathbf a
\]

for all atoms, which gives:

\[
\mathbf u_\kappa^{\mathrm{rel}}=0
\]

and therefore:

\[
H^{\mathrm{LR}}\approx0.
\]

The validation test should check:

```text
uniform displacement detected
relative displacement approximately zero
induced dipoles approximately zero
H_LR approximately zero
```

Call this operation **uniform-translation removal**, not center-of-mass removal, unless the code deliberately uses mass-weighted averaging.

---

## 4. Clarify the Charge, Potential, and Energy Units

The current plan combines:

- The Coulomb constant

\[
\frac{e^2}{4\pi\epsilon_0}
=
14.399645\ \mathrm{eV\,\AA}
\]

- Induced dipoles written as

\[
\mathbf p=eZ^*\mathbf u
\]

- Hamiltonian projection written as

\[
H^{\mathrm{LR}}=-e\phi S.
\]

This can accidentally introduce an extra factor of \(e\), depending on whether \(\phi\) is stored as an electric potential or as the potential energy of an electron.

### Recommended internal convention

Use:

```text
Z*       dimensionless, expressed in units of electron charge
u        Å
d        Z* @ u, in Å
V_LR     electron potential energy, in eV
H_LR     eV
S        dimensionless
```

Define:

\[
\mathbf d_\kappa
=
Z_\kappa^*\mathbf u_\kappa^{\mathrm{rel}}.
\]

Use the Coulomb prefactor containing \(e^2\) directly when constructing the electron potential energy:

\[
V^{\mathrm{LR}}(\mathbf G)
=
C_{\mathrm{Coul}}
\frac{4\pi i}{\Omega}
\frac{
\sum_\kappa
\mathbf G\cdot\mathbf d_\kappa
e^{-i\mathbf G\cdot\mathbf R_\kappa}
}{
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
}
f_{\mathrm{Ewald}}(\mathbf G),
\]

where:

\[
C_{\mathrm{Coul}}
=
14.399645\ \mathrm{eV\,\AA}.
\]

Then project directly:

\[
H_{ij}^{\mathrm{LR}}
=
\frac{
V_i^{\mathrm{LR}}
+
V_j^{\mathrm{LR}}
}{2}
S_{ij}.
\]

Do not multiply by another explicit factor of \(e\).

The exact overall sign depends on the Fourier-transform convention and whether \(V^{\mathrm{LR}}\) is defined as the electrostatic potential or the electron potential energy.

Document the convention in:

```text
constants.py
lr.py
metadata.yaml
```

The sign should be checked against positive and negative DFT displacement pairs during pilot validation.

---

## 5. Use Grouped Dataset Splits

The current plan proposes seeded validation and test splits from the main dataset.

Do not randomly split individual snapshots.

Random snapshot splitting could place closely related structures in different subsets, including:

- \(+A\) and \(-A\) displacement pairs
- \(A\) and \(2A\) amplitude pairs
- Different amplitudes of the same mode
- Different phases of the same base pattern
- Nearly identical mixed-mode structures

This would create data leakage.

### Add a pattern group identifier

Each generated snapshot should contain:

```json
{
  "pattern_group_id": "...",
  "pattern_class": "...",
  "q_vectors": [],
  "polarizations": [],
  "phases": [],
  "amplitudes": [],
  "sign_partner_id": "...",
  "amplitude_partner_ids": []
}
```

The same `pattern_group_id` should be assigned to structures sharing the same underlying displacement family.

Group structures using:

- Base \(\mathbf q\)
- Polarization direction
- Longitudinal or transverse classification
- Phase family
- Mode mixture
- Sign partners
- Amplitude partners

All snapshots in one group must remain in the same split.

### Recommended split strategy

Use the main \(3\times3\times3\) dataset for grouped train, validation, and test subsets.

Hold out:

- Complete \(\mathbf q\) vectors
- Complete \(\mathbf q\)-shells
- Selected polarization families
- Selected mode mixtures
- Some high-amplitude groups

Keep the complete \(4\times4\times4\) dataset separate as the large-cell extrapolation set.

---

## 6. Add DFT-Facing Physics Validation

The existing validation battery is strong for detecting implementation errors:

- Equilibrium zero
- Sign reversal
- Linearity
- Translation invariance
- Hermiticity
- Exact reconstruction
- Orbital and unit consistency

However, those tests could still pass if the LR correction had the wrong physical scale or sign.

Add validation against the DFT Hamiltonian response.

### Odd displacement response

For a positive and negative displacement pair, calculate:

\[
\Delta H_{\mathrm{DFT}}(A)
=
\frac{
H^{\mathrm{full}}(+A)
-
H^{\mathrm{full}}(-A)
}{2}.
\]

Compare it with:

\[
H^{\mathrm{LR}}(A).
\]

These matrices should not be expected to match exactly because:

\[
\Delta H_{\mathrm{DFT}}
=
\Delta H^{\mathrm{SR}}
+
H^{\mathrm{LR}}.
\]

The comparison should still confirm:

- Correct sign behavior
- Sensible magnitude
- Linear behavior for small \(A\)
- Stronger LR response for longitudinal polar modes
- Increasing LR importance as \(|\mathbf q|\) decreases

Recommended diagnostics:

\[
\cos\theta
=
\frac{
\langle
\Delta H_{\mathrm{DFT}},
H^{\mathrm{LR}}
\rangle_F
}{
\|\Delta H_{\mathrm{DFT}}\|_F
\|H^{\mathrm{LR}}\|_F
}.
\]

Also calculate:

\[
r_{\mathrm{LR}}
=
\frac{
\|H^{\mathrm{LR}}\|_F
}{
\|\Delta H_{\mathrm{DFT}}\|_F+\delta
}.
\]

These metrics are diagnostic only and do not need to equal one.

### Locality diagnostic

The purpose of the decomposition is to produce a more local residual Hamiltonian.

Measure:

\[
\left\|
H^{\mathrm{full}}_{ij}(\mathbf R)
\right\|,
\]

\[
\left\|
H^{\mathrm{LR}}_{ij}(\mathbf R)
\right\|,
\]

and:

\[
\left\|
H^{\mathrm{SR}}_{ij}(\mathbf R)
\right\|
\]

as functions of the distance between the two AO centers.

Create distance bins and calculate:

```text
mean block norm
median block norm
maximum block norm
number of nonzero blocks
cumulative norm outside radius r
```

A useful cumulative diagnostic is:

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
},
\]

where:

\[
X\in
\{
\mathrm{full},
\mathrm{LR},
\mathrm{SR}
\}.
\]

The decomposition is useful only if:

\[
F_{\mathrm{SR}}(r)
<
F_{\mathrm{full}}(r)
\]

over the relevant long-distance region.

Add a validation command such as:

```bash
python -m mgo_lr locality-report \
    --workspace <workspace> \
    --set pilot
```

Save the results under:

```text
generation_logs/locality/
```

---

# Smaller Technical Corrections

## 7. Explicitly Disable the ABACUS Gamma-Only Algorithm

ABACUS does not provide `out_mat_hs2` under its gamma-only algorithm.

Even when using only the physical \(\Gamma\)-point, explicitly configure the calculation to use the general k-point algorithm supported by the sparse matrix output.

Add a configuration field such as:

```yaml
abacus:
  gamma_only_algorithm: false
```

The generated `INPUT` file should explicitly reflect the selected ABACUS version and supported keyword.

The parser should continue to support configurable CSR filenames because ABACUS output names vary by version.

---

## 8. Write `orbital_types.dat` Per Atom

`orbital_types.dat` should contain one line for every atom in the structure, not one line per chemical species.

For example, a 16-atom supercell should contain 16 lines.

If Mg and O have different orbital configurations, each atom receives the line corresponding to its species.

The converter should verify:

```text
number of orbital_types.dat lines == number of atoms
```

It should also verify that the orbital count derived from each line matches the dimensions of the corresponding Hamiltonian blocks.

---

## 9. Set Both QE DFPT Flags Explicitly

Use:

```text
epsil = .true.
trans = .true.
```

Even when `trans` defaults to true, explicitly setting both values improves reproducibility and makes the intended calculation clear.

The parser should verify that the output contains:

- Born effective charges
- Electronic dielectric tensor
- The expected Mg and O atom count
- The expected Cartesian tensor dimensions

---

## 10. Keep QE and ABACUS Setups Consistent

Use the same:

- Relaxed lattice vectors
- Relaxed atomic positions
- Exchange-correlation functional
- Valence configurations
- Relativistic treatment
- Charge state
- Spin treatment

Prefer using the same UPF pseudopotential files in QE and ABACUS when both codes support them.

Record separate hashes for:

```text
ABACUS pseudopotentials
ABACUS numerical orbital files
QE pseudopotentials
```

Store them in:

```yaml
provenance:
  abacus:
    pseudopotentials:
    orbitals:
  quantum_espresso:
    pseudopotentials:
```

---

# Revised Validation Battery

The final validation suite should include the following checks.

## Algebraic Checks

### Equilibrium

\[
\left\|
H^{\mathrm{LR}}(\mathbf u=0)
\right\|_F
<
\tau_{\mathrm{eq}}.
\]

### Sign Reversal

\[
\frac{
\left\|
H^{\mathrm{LR}}(-\mathbf u)
+
H^{\mathrm{LR}}(\mathbf u)
\right\|_F
}{
\left\|
H^{\mathrm{LR}}(\mathbf u)
\right\|_F+\delta
}
<
\tau_{\mathrm{sign}}.
\]

### Linearity

\[
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
}
<
\tau_{\mathrm{linear}}.
\]

### Uniform Translation

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

### Hermiticity

For every real-space block:

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

### Reconstruction

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
\right\|_F+\delta
}
<
\tau_{\mathrm{reconstruct}}.
\]

### Reciprocal-Sum Convergence

At fixed \(\Lambda\):

\[
H^{\mathrm{LR}}(G_{\max})
\]

must converge as \(G_{\max}\) increases.

Do not test independence from \(\Lambda\).

---

## Structural and Format Checks

Verify:

- Atom ordering is unchanged
- Reference mapping is correct
- Orbital counts match
- `orbital_types.dat` has one line per atom
- Matrix block dimensions are correct
- All matrix keys use one-based atom indices
- Lattice vectors use the expected column convention
- Reciprocal lattice contains the \(2\pi\) factor
- Energy units are eV
- Length units are Å
- Overlap matrices are finite and numerically reasonable
- No NaN or infinity values are present
- Raw DFT outputs remain unchanged
- Derived HDF5 files are written atomically

---

## Physics Checks

Verify:

- The LR response has the expected sign under displacement reversal
- The LR magnitude is sensible relative to the odd DFT response
- Longitudinal polar patterns have stronger LR contributions than comparable transverse patterns
- Smaller-\(|\mathbf q|\) patterns show greater LR importance
- The SR residual is more localized than the full Hamiltonian
- The selected \(\Lambda\) gives a stable and useful decomposition

---

# Recommended Updated CLI

```text
python -m mgo_lr init-reference
python -m mgo_lr collect-reference

python -m mgo_lr init-dfpt
python -m mgo_lr collect-dfpt

python -m mgo_lr gen-structures --set pilot
python -m mgo_lr collect-dft --set pilot
python -m mgo_lr lr-process --set pilot
python -m mgo_lr validate --set pilot
python -m mgo_lr locality-report --set pilot

python -m mgo_lr gen-structures --set main
python -m mgo_lr collect-dft --set main
python -m mgo_lr lr-process --set main
python -m mgo_lr validate --set main

python -m mgo_lr gen-structures --set large
python -m mgo_lr collect-dft --set large
python -m mgo_lr lr-process --set large
python -m mgo_lr validate --set large

python -m mgo_lr organize
python -m mgo_lr export-target --target sr
python -m mgo_lr status
```

---

# Recommended Immediate Workflow

## Phase 1: Implement the Pipeline

Implement:

- Configuration validation
- Workspace state machine
- Reference input generation
- DFPT input generation and parser
- Displacement generation
- ABACUS input and output handling
- DeepH-E3 conversion
- LR processor
- Validation battery
- Locality diagnostics
- Target export
- Dataset organization

Use synthetic tests before running DFT.

## Phase 2: Generate the Pilot Dataset

Generate only:

```text
12–20 pilot structures
```

Run:

1. ABACUS reference calculations
2. QE equilibrium DFPT
3. Pilot ABACUS calculations
4. Matrix conversion
5. LR processing
6. Algebraic validation
7. Physics validation
8. Locality diagnostics

## Phase 3: Approve the LR Definition

Before scaling up, confirm:

- Unit convention is correct
- Sign convention is correct
- Reciprocal sum is converged
- Translation removal works
- Hermiticity passes
- Reconstruction passes
- The LR response behaves sensibly relative to DFT
- The SR residual is more local than the full Hamiltonian

## Phase 4: Expand the Pilot

Increase to approximately:

```text
50 pilot structures
```

Use this expanded pilot to test:

- More \(\mathbf q\) vectors
- More polarization directions
- More displacement amplitudes
- Mixed-mode patterns
- Reproducible reruns
- Rejection handling
- Grouped splitting logic

## Phase 5: Generate the Main Dataset

Only after all pilot checks pass, generate:

```text
approximately 400 main structures
```

Then generate:

```text
30–50 large-cell test structures
```

---

# Final Verdict

The plan is strong and close to implementation-ready.

### Rating as written

\[
\boxed{8.5/10}
\]

Its strongest parts are:

- Clear package architecture
- Separate invokable stages
- Idempotent snapshot state machine
- External-cluster workflow
- Strong provenance tracking
- Version-aware ABACUS parsing
- Immutable raw outputs
- Atomic HDF5 writes
- Synthetic test coverage
- Clear separation from the MACE-H model code

Before generating the complete dataset, correct:

1. MACE-H target filename export
2. Ewald-\(\Lambda\) validation
3. Uniform-translation handling
4. Charge and energy unit conventions
5. Grouped train/validation/test splitting
6. DFT-facing physics validation
7. Locality diagnostics
8. ABACUS gamma-only settings
9. Per-atom `orbital_types.dat`
10. Explicit QE DFPT flags

After these changes, the plan is ready for the **12–20 structure pilot dataset**.

The full 400-snapshot generation should begin only after the pilot confirms that:

\[
H^{\mathrm{LR}}
\]

has the correct sign and scale, and:

\[
H^{\mathrm{SR}}
=
H^{\mathrm{full}}
-
H^{\mathrm{LR}}
\]

is measurably more localized than the original full Hamiltonian.
