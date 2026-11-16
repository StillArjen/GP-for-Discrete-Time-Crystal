# Extending the GP diagnostic to the open Dicke model and its time crystal

Working notes, November 2026. These notes test whether each ingredient of the
oscillator paper (*The Geometric Phase as a Diagnostic for Driven-Dissipative
Oscillators*) carries over to the setup of Nguyen & Reichman (arXiv:2610.27589):
the periodically switched open Dicke model, its superradiant transition, and its
finite-$N$ discrete time crystal (DTC). All numbers are exact Lindblad
calculations in the symmetric spin sector, with $\kappa=0.05$ and
$\epsilon=0.01$ unless stated. Earlier results are in
[FLOQUET_GP_NOTES.md](FLOQUET_GP_NOTES.md).

## 0. Summary

| Oscillator paper | Dicke / DTC counterpart | Status |
|---|---|---|
| U(1) generator $\hat n$; drive phase $\phi$ | excitation number $\hat N_{\rm exc}=a^\dagger a+J_z+N/2$; phase $\theta$ of the counter-rotating coupling | exact covariance, verified to $10^{-16}$ |
| Single-state reduction, Eq. (4) | $Z_{\rm full}=\sum_k\lambda_k e^{2\pi i\langle N_{\rm exc}\rangle_k}$, $Z_{\rm half}=\sum_k\lambda_k\pi_k e^{i\pi\langle N_{\rm exc}\rangle_k}$ | equals the tracked loop GP to $10^{-5}$ |
| Weak drive: $\Gamma\propto E^2$ from coherences (S28, S31) | normal phase: $\Gamma\propto\lambda^2$ from $\Delta N_{\rm exc}=\pm2$ coherences | formula agrees to 0.1% at $\lambda=0.05$ |
| Kerr bright phase and its DPTs | superradiant transition at $\lambda_c\approx0.5$ | $\lvert\Delta_B\rvert:0\to\pi$, sharpening with $N$; tracks the odd Liouvillian gap |
| Half loop with parity endpoint (S14) | same structure: $R(\pi)=\Pi$ | parity-resolved, it is a $\mathbb Z_2$ order parameter |
| (no counterpart) | DTC: Floquet micromotion loop | $\lvert\Delta_A\rvert\to\pi$ when branches swap ($\mu_1\to-1$), $\to0$ when they don't ($\mu_1\to+1$) |
| (no counterpart) | finite-N DTC regimes (Nguyen & Reichman) | exact spectrum reproduces the non-monotonic $N$ dependence; the GP sector coherence orders the regimes along detuning but does not predict the $N$ dependence |

**In one sentence:** the oscillator construction extends *exactly* to the
open Dicke model through the counter-rotating phase. It diagnoses the
superradiant ($\mathbb Z_2$) transition the same way it diagnoses the Kerr
DPTs. The time crystal needs, in addition, the GP of the physical one-period
loop, which tells branch swapping (DTC) apart from mere branch existence.

## 1. Exact single-state reduction for the Dicke model

Split the Dicke coupling into co- and counter-rotating parts and give the latter
a phase $\theta$:

$$
H_\theta(t)=\omega_c a^\dagger a+\omega J_z+\frac{\lambda(t)}{\sqrt N}\Big[(aJ_++a^\dagger J_-)+(e^{i\theta}aJ_-+e^{-i\theta}a^\dagger J_+)\Big],
$$

which is the model of Nguyen & Reichman at $\theta=0$. With
$R(\varphi)=e^{-i\varphi\hat N_{\rm exc}}$ one has $RaR^\dagger=e^{i\varphi}a$ and
$RJ_-R^\dagger=e^{i\varphi}J_-$. Hence

$$
R(\varphi)\,H_\theta(t)\,R^\dagger(\varphi)=H_{\theta+2\varphi}(t),\qquad D[e^{i\varphi}a]=D[a].
$$

The Liouvillian is therefore covariant at every $t$. Steady states and Floquet
steady states obey $\rho_\theta=R(\theta/2)\,\rho_0\,R^\dagger(\theta/2)$. This is
the structure of the two-photon-driven Kerr resonator, with
$\hat n\to\hat N_{\rm exc}$ and the squeezing phase replaced by $\theta$:

* the counter-rotating terms change $N_{\rm exc}$ by $\pm2$, like the squeezing
  drive changes $n$ by $\pm2$;
* both break U(1) down to the parity $\Pi=e^{i\pi\hat N_{\rm exc}}$.

Equation (S8) of the supplement then reduces exactly to

$$
\theta:0\to4\pi:\quad Z_{\rm full}=\sum_k\lambda_k e^{2\pi i\langle N_{\rm exc}\rangle_k},
\qquad
\theta:0\to2\pi:\quad Z_{\rm half}=\sum_k\lambda_k\,\pi_k\,e^{i\pi\langle N_{\rm exc}\rangle_k}=Z_{\rm even}+Z_{\rm odd}.
$$

Here $\pi_k=\pm1$ is the parity of the $k$-th eigenvector of $\rho$. Degenerate
blocks use the supplement's projector formula. We report the relative sector
phase $\Delta_B=\arg(Z_{\rm even}Z_{\rm odd}^*)$ and the pairing
$C=-2\,\mathrm{Re}(Z_{\rm even}Z_{\rm odd}^*)/(|Z_{\rm even}|^2+|Z_{\rm odd}|^2)$.

**Checks** (in `test_floquet_gp.py`):

* $\theta=0$ reproduces the original Liouvillian to $10^{-15}$.
* The steady state at $\theta=1.3$ equals the rotated one to $10^{-10}$.
* The single-state formula equals the kinematic GP of the explicitly tracked
  orbit to $10^{-5}$, for both the half and the full loop.

**Physical loop.** In cavity-assisted Raman realizations of the Dicke model the
co- and counter-rotating couplings come from separate Raman channels
(Dimer *et al.*, PRA **75**, 013804, 2007). $\theta$ is then their relative laser
phase, so winding $\theta$ is a drive-phase winding exactly as in the
oscillator paper. These platforms also lack the $A^2$ term that would remove
the transition.

## 2. Weak coupling: the counterpart of the synchronization section

In the normal phase the counter-rotating terms create coherences between
$N_{\rm exc}$ sectors that differ by 2. The supplement's general expansion
applies with $n\to N_{\rm exc}$. Split
$\rho=\rho_0+\delta\rho$, where $\rho_0$ is block diagonal in $N_{\rm exc}$, with
eigenvectors $|k\rangle$ and weights $p_k$:

$$
\Gamma_{\rm full}\simeq2\pi\sum_{v<w}(w-v)\sum_{k\in v,\;l\in w}\frac{|\langle k|\delta\rho|l\rangle|^2}{p_k-p_l}.
$$

Results (`results/static_weak_coupling.json`):

| $\lambda$ | exact $\Gamma/\pi$ ($N=2$) | formula | ratio | exact $\Gamma/\pi$ ($N=4$) | ratio |
|---|---|---|---|---|---|
| 0.05 | 0.00252 | 0.00252 | 1.001 | 0.00252 | 1.001 |
| 0.10 | 0.01025 | 0.01029 | 1.004 | 0.01035 | 1.005 |
| 0.20 | 0.04424 | 0.04496 | 1.016 | 0.04605 | 1.020 |
| 0.30 | 0.11309 | 0.11795 | 1.043 | 0.12580 | 1.055 |
| 0.45 | 0.33911 | 0.39006 | 1.150 | 0.46641 | 1.145 |

The GP is quadratic in the coupling, nearly independent of $N$ at weak coupling,
and the formula fails toward $\lambda_c$. This is the same pattern as the
synchronization formula near the Hopf threshold.

## 3. The superradiant transition

Setting: undriven model, coupling always on, $N=2,4,6$, Fock dimension $12+4N$;
`results/static_superradiant.json`, figure `results/extension_superradiant.png`.
Residuals are $\le10^{-13}$ and the top-three Fock populations $\le3\times10^{-5}$.

* **Half-loop sector phase.** $|\Delta_B|$ switches from $\approx0$ to $\pi$.
  The switch sharpens and moves toward the mean-field $\lambda_c=0.500$ as $N$
  grows: $|\Delta_B|=\pi/2$ at $\lambda\approx0.87$, $0.73$ and $0.68$ for
  $N=2,4,6$. At $N=6$ it jumps from $0.25\pi$ to $0.95\pi$ between
  $\lambda=0.65$ and $0.80$. Over the same range $\langle n\rangle/N$ varies
  smoothly.
* **Spectral counterpart.** The switch coincides with the collapse of the
  slowest parity-odd Liouvillian rate, i.e. the rate at which the two
  superradiant branches mix. At $N=6$ that rate falls from $4\times10^{-2}$ at
  $\lambda=0.65$ to $5\times10^{-3}$ at $0.70$ and $5\times10^{-4}$ at $0.80$.
  This is the finite-size precursor of the DPT, and the single-state GP follows it.
* **Interference.** The full-loop visibility $|Z_{\rm full}|$ has a minimum at
  the same crossover: 0.23 ($N=2$, $\lambda\approx0.87$), 0.21 ($N=4$, 0.75),
  0.44 ($N=6$, 0.70). It recovers to $\approx1$ deeper in the superradiant phase.
  This is the paper's phasor-cancellation mechanism. $|Z_{\rm half}|\to0$ there,
  because the even and odd sector amplitudes cancel.
* **Ideal-pair limit.** Deep in the superradiant phase the steady state is an
  incoherent mixture of two parity-related branches. Even and odd eigenvectors
  pair up with equal weights and equal $\langle N_{\rm exc}\rangle$, so
  $Z_{\rm odd}=-Z_{\rm even}$, exactly as for the Kerr cat states:
  $\pi[1-2N_B/\sinh2N_B]$.

## 4. The time crystal: which loop sees what

For the switched drive two loops are available:

* **Loop B**, the counter-rotating phase. Covariance holds at every $t$, so the
  single-state formula applies to $\rho_F(t)$ at any time in the period. It
  detects whether $\mathbb Z_2$ branches *exist*.
* **Loop A**, the micromotion $t\mapsto\rho_F(t)$ over one period (see
  FLOQUET_GP_NOTES.md). It is not a unitary orbit, so it needs eigenvector
  tracking. It detects whether one period *swaps* the branches.

**Control experiment** (`results/dtc_control_*.json`, figure
`results/extension_dtc_control.png`; $N=2$). Lengthen the free segment from
$T_{\rm off}=\pi$ to $2\pi$. At $\epsilon\approx0$ the free evolution over $\pi$
is $\approx e^{-i\pi\hat N_{\rm exc}}=\Pi$, which swaps the branches; over
$2\pi$ it is the identity, which keeps them.

| $\lambda_0$ | $T_{\rm off}=\pi$: $\mu_1$ | $\lvert\Delta_A\rvert/\pi$ | $\lvert\Delta_B\rvert/\pi$ | $T_{\rm off}=2\pi$: $\mu_1$ | $\lvert\Delta_A\rvert/\pi$ | $\lvert\Delta_B\rvert/\pi$ |
|---|---|---|---|---|---|---|
| 0.4 | $0.37+0.84i$ | 0.03 | 0.04 | $-0.36-0.81i$ | 0.05 | 0.04 |
| 0.8 | $-0.70+0.13i$ | 0.31 | 0.27 | $+0.811$ | 0.67 | 0.22 |
| 1.0 | $-0.9916$ | 0.80 | 0.78 | $+0.9888$ | 0.37 | 0.78 |
| 1.2 | $-0.9985$ | 0.964 | 0.956 | $+0.9953$ | 0.064 | 0.958 |
| 1.4 | $-0.9991$ | **0.997** | 0.995 | $+0.9970$ | **0.004** | 0.996 |

$\Delta_B$ is evaluated at the end of the coupled segment. Deep in the ordered
regime the pair-resolved $P_{\rm pair}$ is $+0.995$ for $T_{\rm off}=\pi$ and
$-0.997$ for $T_{\rm off}=2\pi$.

Conclusion: in the ordered limit $e^{i\Delta_A}\to\operatorname{sgn}\mu_1$.
The micromotion GP reads the sign of the leading parity-odd Floquet eigenvalue,
i.e. whether time translation is broken. $\Delta_B$ is blind to that sign.
A DTC requires both $|\Delta_B|\to\pi$ (branches) and $|\Delta_A|\to\pi$ (swap).

## 5. Finite-N regimes

Data: `results/floquet_atoms_eps0p09.json`, `results/floquet_atoms_eps0p01.json`
and the detuning scans; figure `results/extension_finite_N.png`.

**The exact spectrum reproduces the non-monotonic N dependence.** In the
transient window ($\epsilon=0.09$, Fock dimension $20+4N$, top-three populations
$\le2\times10^{-5}$) the DTC decay rate $-\ln|\mu_1|$ per period is

| $N$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $\epsilon=0.09$ | 0.115 | 0.033 | 0.055 | 0.084 | 0.111 | 0.091 |
| $\epsilon=0.01$ | 0.303 (complex $\mu_1$) | 0.0085 | 0.0022 | 0.0018 | 0.0015 | 0.0010 |

The rate at $\epsilon=0.09$ falls, rises and falls again. That is the exact-method
"island" of Nguyen & Reichman (Fig. 4b, $N<20$, $\epsilon\in[0.05,0.11]$),
obtained here directly from the Floquet spectrum rather than from fits. At
$\epsilon=0.01$ the rate falls monotonically, as in their pre-thermal window.

**Micromotion relative phase.** $|\Delta_A|/\pi$ at $\epsilon=0.09$ is
0.55, 0.999, 0.85, 0.88, 0.97, 0.98 for $N=1,\dots,6$. The pair-resolved
$P_{\rm pair}$ is $-0.01$, 0.78, 0.83, 0.74, 0.92, 0.96. The branches are
swapped geometrically throughout the range where the decay rate rises. The
non-monotonic decay is therefore not a loss of branch exchange. This is
consistent with decay by mixing between well-defined branches, which matches
the chaotic-decorrelation mechanism of Nguyen & Reichman. At $N\le6$ this is
suggestive, not a proof.

**Sector coherence.** Define
$S=(|Z_{\rm even}|+|Z_{\rm odd}|)/(w_{\rm even}+w_{\rm odd})$. $S=1$ when all
eigenvectors of a sector acquire the same micromotion holonomy; $S$ is small when
their holonomies are spread.

* **Along detuning at fixed $N$, S tracks the lifetime.** The Spearman
  correlation between $S$ and $\ln(-\ln|\mu_1|)$, over points with real negative
  $\mu_1$, is $-0.80$ at $N=2$ (28 points) and $-0.95$ at $N=4$ (13 points).
* **S orders the three regimes at $N=4$.** It is $\ge0.84$ in the pre-thermal
  window ($|\epsilon|\le0.05$), 0.41–0.74 in the transient window
  ($0.075\le|\epsilon|\le0.1$), and $\le0.42$ for $|\epsilon|\ge0.125$, where
  period doubling is absent or very weak.
* **S does not track the N dependence.** At $\epsilon=0.09$ it decreases from 0.75
  to 0.34 for $N=2\to6$, with a bump at $N=5$; its correlation with the rate is
  $+0.03$. At $\epsilon=0.01$ it stays at $\approx0.98$.

**Verdict for this section.** The GP describes the *structure* of the finite-N
Floquet steady state: whether branches exist ($\Delta_B$), whether they are
swapped ($\Delta_A$), and how uniformly the micromotion holonomy is spread inside
each branch ($S$). Along detuning, the last of these organizes the
pre-thermal / transient / unstable regimes. None of them predicts the
non-monotonic N dependence of the lifetime. That remains a property of the
spectrum ($\mu_1$), not of a single-state geometric quantity. Larger $N$ would
need an approximate method.

## 6. What does not carry over, and open problems

* **No single-state reduction for the time loop.** Loop A is a nonunitary path;
  its GP needs $\rho_F(t)$ along the period. Only loop B reduces to a single
  state.
* **Size.** Exact numerics stop at $N\approx6$ here; a direct steady-state solve
  at $N=8$ ran over 10 minutes and 3.6 GB without finishing. The non-monotonic
  regime of Nguyen & Reichman ($N\sim20$–$500$) needs an approximate method,
  e.g. a Gaussian two-branch model or a TWA-sampled branch decomposition.
* **Sum-level ambiguity.** As found earlier, $\Delta_A$ can be accidentally
  near $\pi$ when the sector amplitudes cancel. Report $P_{\rm pair}$ and
  $|Z_s|/w_s$ alongside it.
* **Measurement.** For loop B, the paper's Ramsey scheme would need a
  conditional rotation generated by $\hat N_{\rm exc}$, i.e. equal dispersive
  shifts of the ancilla from cavity photons and from atomic excitations. Not
  worked out here.
* **Parameters.** Only $\kappa=0.05$ and $\epsilon\approx0.01$, plus the
  detuning scans of FLOQUET_GP_NOTES.md.

## 7. Reproducing

```bash
python dicke_static_scan.py --atoms 2 4 6 --couplings 0.1 1.4 27 --kappa 0.05 --epsilon 0.01 --cutoff 12 --cutoff-per-atom 4 --jobs 4 --out results/static_superradiant
python dicke_weak_coupling_check.py --out results/static_weak_coupling
python dicke_floquet_scan.py --atoms 2 --cutoff 32 --param coupling --list 0.4 0.6 0.8 1.0 1.2 1.4 --t-off 1 --jobs 4 --out results/dtc_control_toff1
python dicke_floquet_scan.py --atoms 2 --cutoff 32 --param coupling --list 0.4 0.6 0.8 1.0 1.2 1.4 --t-off 2 --jobs 4 --out results/dtc_control_toff2
python dicke_floquet_scan.py --param atoms --list 1 2 3 4 5 6 --cutoff 20 --cutoff-per-atom 4 --epsilon 0.09 --jobs 4 --out results/floquet_atoms_eps0p09
python plot_extension.py --static results/static_superradiant.json --weak results/static_weak_coupling.json --control results/dtc_control_toff1.json results/dtc_control_toff2.json --transient results/floquet_atoms_eps0p01.json results/floquet_atoms_eps0p09.json --out results/extension
```
