"""Unit system: energies in eV, lengths in Angstrom, charges in units of e.

V_LR is the potential energy of an ELECTRON (charge -e) in the screened
field of the induced Born dipoles.  With the plane-wave synthesis
V(r) = sum_G V(G) exp(+iG.r) and dipole coefficients carrying
exp(-iG.R0_kappa) source phases, the electron potential energy is the
NEGATIVE of the electrostatic potential of the (positive) dipole
density, hence LR_SIGN = -1.  Pinned numerically by
tests/test_lr_core.py::test_sign_and_prefactor_against_filtered_dipole.
"""

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
RY_TO_EV = 13.605693122994
C_COUL = 14.399645478425668   # e^2/(4 pi eps0) in eV*Angstrom
LR_SIGN = -1.0                # electron potential energy vs dipole potential
ATOMIC_NUMBERS = {"Mg": 12, "O": 8}
