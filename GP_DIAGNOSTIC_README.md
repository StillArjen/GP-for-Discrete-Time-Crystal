# A generator that resolves the Dicke parity branches

The general-generator extension does allow a useful construction. The rotation
generator must be Hermitian; the oscillator demonstration uses the number operator
rather than the annihilation operator itself. The new code evaluates a specified
unitary-orbit GP on the joint spin-cavity state at each stroboscopic time.

For N atoms in the symmetric spin sector, choose the following fixed generator:

$$
j=N/2,\qquad u=\frac{1}{4N},\qquad v=\sqrt{1-u^2},\qquad
Q_{xy}=jI+uJ_x+vJ_y.
$$

The spin operators act as identity on the cavity. Since the transverse axis has
unit length, Q has spectrum 0,1,...,N, each repeated over the cavity space. Its
full rotation therefore closes. The angle is fixed by N before scanning detuning
and loss; it is not fitted to a phase diagram.

For a nondegenerate density spectrum the amplitude is

$$
Z_Q[\rho]=\sum_k p_k\exp\left(i2\pi\langle\psi_k|Q|\psi_k\rangle\right),
\qquad \Gamma_Q=\arg Z_Q.
$$

At degeneracies, `gp_geometry.py` instead uses the projector formula from the
supplement. It retains the complex amplitude and treats the phase as undefined
at a zero. Numerical grouping tolerances are exposed and recorded.

The Dicke parity transforms this generator as

$$
\Pi Q_{xy}\Pi^\dagger=NI-Q_{xy},\qquad
Z_Q[\Pi\rho\Pi^\dagger]=Z_Q[\rho]^*.
$$

The second identity also holds with degenerate density eigenspaces. Thus the
imaginary part is parity odd. For an ideal branch exchange after each period,

$$
\rho((m+1)T)=\Pi\rho(mT)\Pi^\dagger
\quad\Longrightarrow\quad
\operatorname{Im}Z_Q((m+1)T)=-\operatorname{Im}Z_Q(mT).
$$

This removes the exact symmetry obstruction of the number generator. It is a
sensitivity statement, not a theorem that every parity-broken state gives a
nonzero imaginary part. Phasor cancellation or a real geometric amplitude can
still make this generator insensitive at particular states.

An exact calibration illustrates why the small tilt was chosen. For N>=2,

$$
\rho_f=f|J_x=j\rangle\langle J_x=j|
 +(1-f)|J_x=-j\rangle\langle J_x=-j|,
\qquad
Z_Q[\rho_f]=\frac{(-1)^N}{\sqrt2}\left[1+i(2f-1)\right].
$$

Consequently, the reported toy-normalized response

$$
g_Q=\sqrt2(-1)^N\operatorname{Im}Z_Q
$$

equals the branch imbalance in this family, including the degenerate equal
mixture. The same result holds for the illustrative joint states with cavity
coherent factors of opposite signs. This family is an analytic check, not an
assumption about the actual Dicke density matrix. The normalization is not a
universal bound for arbitrary states.

Using Im(Z)=|Z| sin(Gamma) retains the phase information while avoiding branch-cut
jumps and suppressing ill-defined bare phases at vanishing visibility. The GP
itself reverses sign modulo 2pi under parity. The saved complex amplitudes allow
either the phase or the visibility-weighted phase response to be examined.

The code also evaluates a second construction:

$$
Q_\pm=jI+vJ_z\pm uJ_x,\qquad
S=\operatorname{Im}(Z_{Q_+}Z_{Q_-}^*).
$$

Parity exchanges the two generators, so S is odd and also equals the imbalance
in the calibration family. This provides a comparison with the single-generator
response. Cavity number is retained as a control.

## What the calculation measures

Each GP is for an auxiliary loop at a fixed physical time, not the geometric
phase of the dissipative time trajectory. A spin-only rotation conjugates both
the atomic-frequency term and the interaction spin direction, while leaving the
cavity jump operator unchanged. To realize the covariant family dynamically,
the initial spin state must also be rotated. This is an added control protocol;
varying the square-wave timing alone does not implement it.

The diagnostic is the alternating component of the geometric signal at the
drive strobe. The scan reports

$$
A_Q=\left|\frac1M\sum_{m\in W}(-1)^m g_Q(mT)\right|
$$

over an even late-time window W. The corresponding magnetization amplitude is
computed over exactly the same samples. A finite-window Fourier amplitude can
contain leakage from a changing envelope, so the saved full trajectories and
the demodulated traces should also be inspected.

A parity-symmetric Floquet stationary state has real Z_Q and zero odd response.
The scientifically relevant target here is finite-time subharmonic persistence.
These small-system maps do not establish an infinite-time DTC phase, rigidity
under arbitrary perturbations, or thermodynamic transition boundaries. A new GP
sensor is also not automatically an independent measure of order: near a
nondegenerate parity-symmetric state its leading odd response can track a
transverse-spin expectation.

## Supplied-code audit and implementation

- The imported helper for the full-loop sweep was missing. `_phase_eta_exact_core.py`
  is a documented reconstruction from the manuscript, not recovered original code.
  Its `one_point` signature matches the supplied driver.
- The supplied loop routine computes a 2pi squeezing-phase loop, a half number
  rotation with a parity endpoint. Its negative exponent is consistent with its
  reversed Hamiltonian phase convention. The manuscript uses a positive full
  number rotation, corresponding to a 4pi squeezing-phase loop. These are distinct
  protocols; mixed-state half-loop angles cannot simply be doubled to obtain the
  full-loop result.
- That routine also hardcodes positive U/2 and zero one-photon pumping. The
  manuscript benchmark uses U=-1 multiplying the full Kerr term and all three
  rates equal to 0.1. Those scans should not be expected to agree numerically.
- The new geometry core handles degenerate eigenspaces, rejects materially invalid
  density matrices, retains complex Z, and exposes small numerical corrections.
  It does not silently clip arbitrary solver errors.
- Both supplied sweep files remain unchanged. Their large default grids were not
  launched. `kerr_gp_check.py` is the configurable small benchmark.

The Dicke Hamiltonian uses cavity frequency 1-epsilon, atomic frequency 1+epsilon,
period 2pi, coupling on in the first half-period, and cavity loss kappa D[a]. The
initial state is vacuum times a spin coherent state along +x. These conventions
agree with the authors' [exact solver](https://github.com/nguye66h/qed-dtc/blob/main/src/exact.py).
Propagation applies two sparse exponential actions, so no integration step can
skip the square-wave switching time. The quantum state is kept in the symmetric
spin sector and a finite cavity Fock basis.

## Running the checks

The task-local Windows environment is `.venv`. For a new environment, install
`requirements-gp.txt`; `requirements-gp-lock.txt` records the tested versions.

```powershell
.\.venv\Scripts\python.exe -m unittest test_gp_geometry test_gp_models -v
.\.venv\Scripts\python.exe kerr_gp_check.py --cutoff 50 --delta -15 30 31 --eta 0.001 5 25 --jobs 4 --out results/kerr_check
.\.venv\Scripts\python.exe plot_gp_checks.py kerr results/kerr_check.npz
.\.venv\Scripts\python.exe dicke_gp_scan.py --atoms 2 --cutoff 32 --check-cutoff 48 --periods 60 --eps -0.056 0.01 --kappas 0.05 --jobs 2 --out results/dicke_two_points
.\.venv\Scripts\python.exe plot_gp_checks.py dicke results/dicke_check.npz --traces-data results/dicke_cutoff48_check.npz
```

For arbitrary generators:

```python
import numpy as np
from gp_geometry import geometric_amplitude
Z = geometric_amplitude(rho, Q)
visibility = abs(Z)
phase = np.angle(Z) if visibility > 1e-6 else np.nan
```

For repeated snapshots, `prepare_generator` caches each generator's eigensystem;
`prepare_state` shares a density eigendecomposition between several generators.
`dicke_gp_scan.py` exposes photon cutoff, physical parameters, observation window,
and density-degeneracy tolerances. NPZ files contain every simulated strobe.

## Numerical and figure checks

The test suite includes discrete parallel transport independent of the closed
formula, basis changes inside degenerate eigenspaces, half-loop endpoint factors,
random-state parity identities, analytic branch mixtures, zero-coupling spin
precession, and an independent segmented ODE comparison of Dicke propagation.

The Kerr benchmark uses the same 25-by-31 grid at Fock dimensions 50 and 70.
Maximum changes are 1.03e-8 in photon number and 3.72e-9 in complex full-loop Z.
The maximum raw steady-state residual is below 1.4e-13. The bright-state region
and geometric structure are visible; this is a numerical finite-system check,
not a new determination of critical exponents or phase-transition order.

The Dicke pilot uses two atoms, 60 drive periods, and a 9-by-4 detuning-loss grid
at Fock dimension 32. The same fixed Q is used at every point. Both spin-based
geometric responses show the central region of enhanced subharmonic persistence;
the paired contrast tracks the magnetization map more closely than the single
transverse generator. Their amplitudes need not match because their nonlinear
state dependence and calibration differ.

Eight selected points were rerun at Fock dimension 48. At loss 0.05, the last-20-
sample amplitudes from that larger-cutoff run are:

| Detuning | Magnetization | Single-generator response | Paired contrast |
| --- | ---: | ---: | ---: |
| -0.056 | 0.0328709 | 0.0154788 | 0.0418028 |
| 0.01 | 0.1674770 | 0.2023723 | 0.1779748 |

For the detuning 0.01 case, increasing dimension 32 to 48 changes the entire
single-generator response trajectory by at most 1.23e-6 and its late-window
amplitude by 1.88e-9. Its paired contrast changes by at most 2.25e-7. For the
-0.056 case, those full-trajectory changes are below 8.95e-6 and 2.42e-6.
The delivered trace figure uses these dimension-48 calculations.

The full grid remains a pilot, not a uniformly cutoff-converged phase diagram.
Nine of its 36 points have a top-three-Fock-level population above 1e-5 at a
sampled half period and are marked with crosses in the map. A small tail alone
also does not prove convergence. In the worst sampled corner (detuning 0.13,
loss 0.02), dimension 32 to 48 changes the full single-generator trajectory by
0.0305 or less, and dimension 48 still has a tail as large as 3.52e-4. That
corner requires further cutoff refinement before quantitative interpretation.
All 36 points remain visible; none were removed from the map. The selected
comparisons are saved in `results/dicke_selected_cutoff_comparison.json`.

All 16 geometry and model tests passed. A zero-coupling resonant control has
nonzero static geometric signals but a vanishing alternating component, so
the diagnostic does not equate a static GP with period doubling.

Figures are quantitative grids from deterministic model calculations, with no
statistical replicates or fitted phase labels. No input measurements were supplied
or excluded. Grid cells use their actual simulated values without interpolation.
PNG is a 300-dpi preview; editable SVG and PDF are also saved. The figure preflight
warnings about absent TIFF and a 600-dpi raster are accepted because these are
exploratory outputs with vector originals, not a journal submission package.
