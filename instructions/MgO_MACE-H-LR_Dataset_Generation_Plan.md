# MgO Dataset Generation Plan for MACE-H-LR

## Should the dataset be generated before coding the LR part?

Do **not** generate the entire dataset before implementing and validating the long-range processing code.

Use this order:

\[
\boxed{
\text{Define the data format}
\rightarrow
\text{Generate 12–20 pilot structures}
\rightarrow
\text{Build the LR processor}
\rightarrow
\text{Validate the labels}
\rightarrow
\text{Generate the full dataset}
}
\]

You can generate the following raw data before the LR code is complete:

\[
\left\{
\mathbf R,\,
H^{\mathrm{DFT}},\,
S,\,
Z^*,\,
\epsilon_\infty
\right\}.
\]

However, the LR code is needed before creating the final dataset labels:

\[
H^{\mathrm{LR}}
\]

and

\[
H^{\mathrm{SR}}
=
H^{\mathrm{DFT}}-H^{\mathrm{LR}}.
\]

Generating hundreds of DFT calculations before validating the LR formulation could waste computing time if the supercell size, displacement patterns, orbital metadata, or matrix conventions need to change.

---

# MgO Dataset-Generation Instructions

## 1. Fix the Computational Setup

Use one consistent DFT setup for every MgO structure:

- **DFT code:** ABACUS
- **Basis:** LCAO
- **Exchange-correlation functional:** PBE
- Same Mg and O pseudopotentials
- Same numerical atomic orbitals
- Same orbital ordering
- Same k-point density
- Same energy cutoff
- Same SCF tolerance
- Non-spin-polarized calculations
- Fixed cell for the first dataset

Use settings equivalent to:

```text
calculation  scf
basis_type  lcao
out_mat_hs2 1
```

Do not change the basis, pseudopotentials, or orbital files after dataset generation begins.

---

## 2. Create the Equilibrium MgO Reference

Use rocksalt MgO with a two-atom primitive cell.

Complete the following steps:

1. Converge the basis, cutoff, and k-point density.
2. Relax the lattice constant.
3. Relax the atomic positions.
4. Perform a final high-accuracy static calculation.
5. Save the equilibrium structure as the permanent reference.

Store:

```text
reference/
├── primitive.cif
├── reference_cell.npy
├── reference_positions.npy
├── atomic_numbers.npy
├── species_order.json
├── orbital_types.dat
└── dft_settings.yaml
```

Every distorted structure must retain the same atom ordering and map back to this reference.

For snapshot \(s\), define:

\[
\mathbf u_{\kappa}^{(s)}
=
\mathbf R_{\kappa}^{(s)}
-
\mathbf R_{\kappa}^{0}.
\]

Use minimum-image or unwrapped fractional-coordinate differences so that atoms crossing periodic boundaries do not appear to have extremely large displacements.

---

## 3. Generate \(Z^*\) and \(\epsilon_\infty\)

Perform one separate DFPT calculation at the equilibrium MgO geometry.

Using Quantum ESPRESSO:

1. Run a converged `pw.x` ground-state calculation.
2. Run a \(q=0\) `ph.x` calculation.
3. Enable the dielectric and Born-charge calculation:

```text
epsil = .true.
```

Save:

```text
born_effective_charges.npy    # Shape: [2, 3, 3]
dielectric_infinity.npy       # Shape: [3, 3]
qe_dfpt_output.out
```

Check:

\[
Z_{\mathrm{Mg}}^*+Z_{\mathrm O}^*\approx0,
\]

\[
Z_{\kappa}^*\approx z_\kappa^*I,
\]

and

\[
\epsilon_\infty
\approx
\epsilon_{\infty}^{\mathrm{scalar}}I.
\]

Correct small acoustic-sum-rule errors with:

\[
\widetilde Z_\kappa^*
=
Z_\kappa^*
-
\frac{1}{N}
\sum_{\kappa'}Z_{\kappa'}^*.
\]

For the first MgO-only dataset, use these equilibrium tensors as fixed material parameters for every snapshot.

Do not calculate \(Z^*\) and \(\epsilon_\infty\) for every distorted structure in version 1.

---

## 4. Define the Supercells

Use three dataset levels.

### Development Cell

Use a \(2\times2\times2\) primitive-cell supercell:

\[
2\times2\times2\times2
=
16\text{ atoms}.
\]

Use this cell for debugging and pilot calculations.

### Main Cell

Use a \(3\times3\times3\) primitive-cell supercell:

\[
3\times3\times3\times2
=
54\text{ atoms}.
\]

Use this cell for most of the final dataset.

### Large-Cell Test

Use a \(4\times4\times4\) primitive-cell supercell:

\[
4\times4\times4\times2
=
128\text{ atoms}.
\]

Keep these structures separate from the main dataset.

---

## 5. Generate the First Pilot Structures

Before producing hundreds of structures, generate approximately **12–20 pilot snapshots** in the \(2\times2\times2\) cell.

Include:

- Equilibrium structure
- Mg-only displacement in \(+x\)
- Mg-only displacement in \(-x\)
- O-only displacement in \(+x\)
- O-only displacement in \(-x\)
- Opposite Mg/O optical displacement
- Longitudinal finite-\(q\) pattern
- Transverse finite-\(q\) pattern
- Two mixed-mode structures
- Two random local-displacement structures
- One rigid-translation test structure

Use small amplitudes initially:

\[
0.005,\ 0.01,\ 0.02\ \text{\AA}.
\]

Positive and negative displacement pairs are important because they allow the sign behavior of the LR contribution to be checked.

---

## 6. Generate Finite-Wavevector Displacement Patterns

The dataset should contain collective polar distortions, not only independent random atomic noise.

For unit cell \(l\), use patterns such as:

\[
\mathbf u_{\mathrm{Mg}}(\mathbf R_l)
=
A\widehat{\mathbf e}
\cos(\mathbf q\cdot\mathbf R_l+\varphi),
\]

\[
\mathbf u_{\mathrm O}(\mathbf R_l)
=
-A\widehat{\mathbf e}
\cos(\mathbf q\cdot\mathbf R_l+\varphi).
\]

Vary:

- Commensurate wavevector \(\mathbf q\)
- Polarization direction \(\widehat{\mathbf e}\)
- Longitudinal versus transverse orientation
- Phase \(\varphi\)
- Amplitude \(A\)
- Relative Mg/O amplitudes
- Number of simultaneously combined modes

Use amplitudes such as:

\[
0.005,\ 0.01,\ 0.02,\ 0.04,\ 0.06\ \text{\AA}.
\]

For mixed structures, combine two to four patterns:

\[
\mathbf u_\kappa
=
\sum_{m=1}^{M}
A_m\mathbf e_{\kappa m}
\cos(\mathbf q_m\cdot\mathbf R+\varphi_m).
\]

Remove accidental rigid translation:

\[
\mathbf u_\kappa
\leftarrow
\mathbf u_\kappa
-
\frac{1}{N}
\sum_{\kappa'}\mathbf u_{\kappa'}.
\]

Keep a small number of deliberately rigidly translated structures for invariance testing.

---

## 7. Run Static DFT Calculations

For every accepted snapshot:

1. Write the displaced ABACUS structure.
2. Keep the cell fixed.
3. Run a static LCAO SCF calculation.
4. Output \(H(\mathbf R)\) and \(S(\mathbf R)\).
5. Save the full calculation output and convergence status.

Save the raw output before conversion:

```text
snapshot_000001/
├── structure/
├── INPUT
├── KPT
├── pseudopotential_links/
├── orbital_links/
├── running_scf.log
├── H_R.csr
├── S_R.csr
└── status.json
```

The exact CSR filenames may depend on the ABACUS version, so record them in `status.json`.

Reject a structure when:

- SCF does not converge
- Atoms become unphysically close
- Matrix dimensions change
- Orbital ordering changes
- Output matrices contain NaNs or infinities
- The overlap matrix becomes numerically problematic

---

## 8. Convert the Data into MACE-H-Compatible Format

Convert each successful calculation into the DeepH-E3/MACE-H data representation.

Each processed structure should contain files equivalent to:

```text
snapshot_000001/
├── hamiltonians_full.h5
├── overlaps.h5
├── lat.dat
├── rlat.dat
├── site_positions.dat
├── orbital_types.dat
├── element.dat
├── info.json
├── displacements.npy
└── displacement_metadata.json
```

At this stage, call the DFT Hamiltonian file:

```text
hamiltonians_full.h5
```

Do not overwrite it when generating the LR and SR matrices.

---

## 9. Implement the Minimum LR Dataset Processor

After the first 12–20 raw structures are working, implement the standalone LR preprocessing script.

The script must read:

```text
reference positions
current positions
cell
atom mapping
Z*
epsilon_infinity
H_full
overlap matrix
AO-to-atom mapping
AO centers
```

For each atom, calculate the induced dipole:

\[
\mathbf p_\kappa
=
eZ_\kappa^*\mathbf u_\kappa.
\]

Construct the screened periodic dipole potential in reciprocal space:

\[
\phi^{\mathrm{LR}}(\mathbf G)
=
\frac{4\pi i}{\Omega}
\frac{
\sum_\kappa
\mathbf G\cdot\mathbf p_\kappa
e^{-i\mathbf G\cdot\mathbf R_\kappa}
}{
\mathbf G\cdot\epsilon_\infty\cdot\mathbf G
}
f_{\mathrm{Ewald}}(\mathbf G),
\qquad
\mathbf G\neq0.
\]

Important conventions:

- Set the \(\mathbf G=0\) component to zero.
- Use one fixed Ewald damping parameter for the entire dataset.
- Save the damping parameter in the metadata.
- Use one consistent unit system.
- Use the reference atom mapping when calculating displacements.

For the first implementation, use the symmetric potential-times-overlap approximation:

\[
H_{ij}^{\mathrm{LR}}
\approx
-e
\frac{
\phi^{\mathrm{LR}}(\mathbf r_i)
+
\phi^{\mathrm{LR}}(\mathbf r_j)
}{2}
S_{ij}.
\]

Then calculate:

\[
H^{\mathrm{SR}}
=
H^{\mathrm{full}}
-
H^{\mathrm{LR}}.
\]

Save:

```text
hamiltonians_full.h5
hamiltonians_lr.h5
hamiltonians_sr.h5
```

Keep the LR processor separate from the MACE-H model code during the initial dataset-development stage.

---

## 10. Validate the Pilot Dataset

Do not scale up until the pilot dataset passes the following tests.

### Equilibrium Test

\[
H^{\mathrm{LR}}(\mathbf u=0)
\approx
0.
\]

### Displacement-Reversal Test

\[
H^{\mathrm{LR}}(-\mathbf u)
\approx
-H^{\mathrm{LR}}(\mathbf u).
\]

### Linear-Amplitude Test

For sufficiently small \(A\):

\[
H^{\mathrm{LR}}(2A)
\approx
2H^{\mathrm{LR}}(A).
\]

### Rigid-Translation Test

The LR correction should not change when every atom receives the same translation, apart from numerical precision and the selected potential gauge.

### Hermiticity Test

For the stored real-space blocks, verify the appropriate Hermitian relation:

\[
H^{\mathrm{LR}}_{ij}(\mathbf R)
=
\left[
H^{\mathrm{LR}}_{ji}(-\mathbf R)
\right]^*.
\]

Check the same condition for the full and residual Hamiltonians.

### Reconstruction Test

Verify numerically:

\[
H^{\mathrm{SR}}
+
H^{\mathrm{LR}}
=
H^{\mathrm{full}}.
\]

### Supercell and Orbital Consistency

Confirm that:

- Every matrix index maps to the correct atom and orbital
- The same orbitals are present in every structure
- All lengths and energies use consistent units
- The lattice-vector conventions match between ABACUS and the LR code
- Atom ordering is unchanged
- The reference-to-snapshot mapping is correct

---

## 11. Generate the Main Dataset

After the pilot passes, generate approximately **400 main snapshots** using the \(3\times3\times3\) cell.

Recommended composition:

| Structure Type | Count |
|---|---:|
| Single finite-\(q\) optical patterns | 150 |
| Mixed low-\(q\) polar patterns | 120 |
| Local random distortions | 60 |
| Directional and sign-paired calibration structures | 40 |
| Near-equilibrium low-amplitude structures | 30 |

For every structure:

1. Generate the displacement pattern.
2. Check minimum interatomic distances.
3. Run ABACUS.
4. Check SCF convergence.
5. Extract \(H^{\mathrm{full}}\) and \(S\).
6. Run the LR processor.
7. Save \(H^{\mathrm{LR}}\).
8. Save \(H^{\mathrm{SR}}\).
9. Run automatic quality checks.
10. Record whether the structure passed or was rejected.

---

## 12. Generate the Large-Cell Dataset

Generate approximately **30–50 structures** using the \(4\times4\times4\) cell.

Favor:

- Small wavevectors
- Longitudinal optical patterns
- Mixed long-wavelength modes
- Amplitudes within the range used in the main dataset

Store these structures separately:

```text
test_large_cell/
```

Do not mix them into the initial main dataset.

---

## 13. Organize the Final Dataset

```text
MgO_MACE_H_LR/
├── metadata.yaml
├── reference/
│   ├── primitive.cif
│   ├── reference_cell.npy
│   ├── reference_positions.npy
│   ├── born_effective_charges.npy
│   ├── dielectric_infinity.npy
│   ├── dft_settings.yaml
│   └── qe_dfpt_output.out
├── pilot/
├── main/
├── validation_candidates/
├── test_candidates/
├── test_large_cell/
├── rejected/
└── generation_logs/
```

Each accepted structure should contain:

```text
structure.cif
displacements.npy
displacement_metadata.json
hamiltonians_full.h5
hamiltonians_lr.h5
hamiltonians_sr.h5
overlaps.h5
lat.dat
rlat.dat
site_positions.dat
orbital_types.dat
element.dat
info.json
calculation_status.json
quality_checks.json
```

The main metadata file should record:

```text
exchange-correlation functional
pseudopotential names and hashes
orbital file names and hashes
basis specification
k-point mesh or density
energy cutoff
SCF tolerance
length units
energy units
reference structure ID
supercell matrix
atom-ordering convention
potential gauge convention
Ewald damping parameter
random seed
ABACUS version
Quantum ESPRESSO version
preprocessing-code version
```

---

# Recommended Immediate Workflow

1. Complete one converged equilibrium ABACUS MgO calculation.
2. Obtain equilibrium \(Z^*\) and \(\epsilon_\infty\).
3. Generate 12 sign-paired and finite-\(q\) pilot structures.
4. Extract the full Hamiltonian and overlap matrices.
5. Build the standalone LR preprocessing code.
6. Validate \(H^{\mathrm{LR}}\) and \(H^{\mathrm{SR}}\).
7. Expand the pilot dataset to approximately 50 structures.
8. Generate the 400-structure main dataset.
9. Generate the separate 30–50 structure large-cell dataset.

The full dataset should only be generated after the pilot structures and LR labels pass all consistency checks.
