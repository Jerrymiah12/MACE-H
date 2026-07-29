# EPC figures — bulk Au, `run_Aims_45xp63ba`

Model `MACE-H/train/Au_bulk/2026-07-20_16-13-39`, test structure (unseen during
training).

## Headline number

**Predicted vs. DFT Hamiltonian: 0.218 % L2 relative error** over all 1,180,494
matrix elements in 14,574 blocks.

| metric | value |
|---|---|
| L2 relative error | **0.218 %** |
| MAE | 2.847e-04 eV |
| RMSE | 6.033e-04 eV (0.0061 % of full scale) |
| max abs error | 1.884e-02 eV |
| bias | −7.4e-07 eV (essentially unbiased) |
| peak abs H | 9.96 eV |

## This is NOT a bound on EPC accuracy

An earlier version of this file called the Hamiltonian error "a real bound on EPC
accuracy". **That was wrong and has been removed.** Differentiation is an unbounded
operation: two functions can agree everywhere to 0.218 % and still have arbitrarily
different derivatives.

Concretely, a central difference amplifies any *position-dependent* part of the
model's error by 1/(2δ). If the model−DFT error were uncorrelated between the +δ
and −δ evaluations, an RMSE of 6.03e-04 eV at δ = 1e-4 Å would produce roughly
4 eV/Å of noise in dH — about 80× the RMS |dH| itself, i.e. the EPC would be pure
noise. Finite-differencing an ML Hamiltonian works only because that error is
strongly *correlated* across the two evaluations, which is a statement about the
**smoothness of the error surface in configuration space**, not about its size.

What has actually been established:

- the **model** is smooth in configuration space — a 14-point δ sweep gives clean
  δ² convergence from 1e-3 down to 5e-6 (see the project-level analysis);
- the **model matches DFT at one geometry** — the 0.218 % above.

What has *not* been established, and is what actually controls EPC accuracy: that
the model−DFT **error** varies smoothly as atoms move. Nothing in this dataset
probes that. Only DFT on displaced geometries would.

## Why there is no predicted-vs-DFT EPC figure

EPC is a derivative of the Hamiltonian, so a DFT reference requires DFT runs on
displaced geometries (±δ for every atom and Cartesian direction — 384 SCF
calculations per structure). `bulk_gold_data` contains 190 training and 10 test
structures that are **independent MD snapshots**: consecutive test structures
differ by ~4 Å RMS in atomic position, not by a small displacement. They cannot be
differenced.

Figure 02 therefore compares the **Hamiltonian**. It is a necessary condition for
EPC accuracy and a useful sanity check, but per the section above it is not
sufficient and not a bound. Do not present it as EPC validation.

## Figures

| file | what it shows |
|---|---|
| `01_epc_heatmap.png` | Orbital-resolved \|g_ij\| for atom 0 displaced along x, at k = q = Γ; 576×576 (64 atoms × 9 orbitals), log colour. Caption states the hard floor (9.5e-06 eV/Å, 10th percentile), what fraction of elements render at it, the true smallest nonzero element, and that there are **no exact zeros** so no zero-handling convention applies. |
| `02_pred_vs_dft.png` | Parity plot of predicted vs DFT Hamiltonian, with a ±0.02 eV inset (82.6 % of elements) where the scatter width is the actual error, plus a residual-vs-magnitude panel. **Hamiltonian, not EPC.** |
| `03_error_histogram.png` | Distribution of (predicted − DFT), log count. The x-range is the 99.999th percentile of \|error\|; the 12 elements (0.00102 %) outside it are **excluded, not clipped into the edge bins**, and the quoted statistics are over the full set. |
| `04_q_path.png` | ‖g(Γ, q)‖_F along Γ–X–W–L–Γ–K. ‖·‖_F is the Frobenius norm over the AO index pair (i,j) over all 576×576 orbital pairs. Because that norm scales with orbital count, curves are plotted as fractional deviation from their own mean in **ppm** (size-intensive, comparable across systems); means are 26.5–27.3 eV/Å, i.e. an RMS matrix element of ~0.046 eV/Å. Total modulation 150–330 ppm. Exact on the 2×2×2 q-grid; between those 8 points it is band-limited interpolation from p ∈ {0,1}³. |
| `05_crystal_vectors.png` | Per atom, v_{a,α} = Tr Re[g(k=Γ, q=Γ)_aa] = Σ_{i∈a} Σ_{R,p} ∂H_ii(R)/∂τ_{a,α}(p), trace over the 9 AO diagonal elements of atom a. This is the **Γ-point Bloch diagonal block**, so it sums over R and includes the atom coupling to its own periodic images — it is *not* the bare R = 0 on-site term. Real part; the imaginary trace is **exactly 0**, as Hermiticity requires. Model is **non-spinful** (`spinful = False`), so no spin channels. Evaluated **after Hermitization** — `predict_fn` symmetrises H before differencing. Magnitudes span 0.80–11.06 eV/Å. |

## Provenance

- Figures 01 and 04: **q_grid 2×2×2, δ = 1e-4** run
  (`~/.local/share/maceh-epc/runs/q222_d1e4/`), 512-atom supercell, 8 q-points.
- Figure 05: original **64-atom q = Γ** run (`epc_output/`) — the only one with all
  64 atoms displaced.
- Figures 02 and 03: a fresh single forward pass on the unit cell compared with
  `MACE-H/bulk_gold_data/test/run_Aims_45xp63ba/hamiltonians.h5`.

δ matters: the earlier δ = 0.01 runs carry 0.85 % L2 error in g, roughly four times
the model's own Hamiltonian error against DFT.

## Regenerating

Scripts live in this session's scratchpad, not the repo: `plotdata2.py` (collects
everything into `plots.npz`), then `make_plots.py`. Copy them into the repo if these
figures need to be reproducible long-term.
