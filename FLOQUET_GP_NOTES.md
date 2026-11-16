# Can the GP construction say something new about the Dicke DTC?

Working notes, November 2026. All numbers are from exact Lindblad calculations in the
symmetric spin sector (κ = 0.05, λ₀ = 1, T = 2π unless stated); the scripts and
commands that produce them are listed at the end.

Figures: `results/floquet_gp.png` (coupling, detuning and atom-number scans),
`results/floquet_gp_phasors.png` (sector amplitudes in the complex plane),
`results/floquet_gp_late_time.png` (stroboscopic signals vs the Floquet rate),
`results/floquet_gp_kerr.png` (Kerr half loop). Editable SVG and PDF copies sit
alongside.

## Short answer

* **The stroboscopic, fixed-generator GP (`Q_xy`, paired `Q_±`) is not a source of
  new physics.** It is a nonlinear parity-odd order parameter. At late times it
  decays at exactly the rate of the leading parity-odd Floquet eigenvalue μ₁, the
  same rate as the magnetization. It vanishes identically in the Floquet steady state,
  and its loop is synthetic, with no physical covariance behind it. It is a valid
  consistency check, but it would be hard to defend as a new diagnostic.
* **A physical loop does give something new.** The Floquet steady state ρ_F(t) is
  T-periodic, so its one-period micromotion is a closed loop of nonequilibrium steady
  states. Equivalently, it is the family obtained by winding the *temporal* phase of
  the drive, the analogue of the drive-phase winding in the oscillator paper. Its
  kinematic mixed-state GP splits exactly into parity sectors. In the period-doubled
  regime the two sector amplitudes become antiparallel: a π relative geometric phase,
  or "geometric π pairing". This shows up in a parity-symmetric state in which every
  one-time order parameter is zero.
  * The relative phase is ≈ 0 where there is no period doubling. At N = 2 that means
    λ₀ ≤ 0.7: below and somewhat above the superradiant threshold λ_c ≈ 0.5, where μ₁
    is complex.
  * It rises to π as the coupling or the atom number grows (0.80π → 0.972π → 0.990π →
    0.995π for N = 2…5).
  * It does *not* simply track the DTC lifetime.
  * The near-π value is robust only when checked pair by pair (equal weights and a π
    holonomy difference for matched eigenvectors). At N = 4 the sum-level phase gives
    accidental near-π values where the sector amplitudes have cancelled.
* **The same structure is already present in the oscillator paper.** The 2π
  squeezing-phase "half loop" of Eq. (S14) is the lab-frame micromotion over one pump
  period of the two-photon-driven Kerr resonator, a period-doubling oscillator.
  Resolved by parity, the ideal-cat value of the relative phase is π[1 − 2N_B/sinh 2N_B].
  The exact value is ≈ π throughout the bright phase and 0.05–0.13π on the vacuum-like
  side.
* **Prior work.** The closed-system counterpart is known: in unitary Floquet DTCs the
  π quasienergy splitting of parity partners lies in the geometric phase
  (Schindler & Bukov, *Geometric Floquet theory*, arXiv:2410.07029; known here only
  from its abstract and search summaries, since arxiv.org was unreachable from this
  session). A limited search
  found no mixed-state or open-system version. The natural positioning is therefore
  "geometric π pairing for open, mixed-state time crystals", not a new GP diagnostic
  of magnetization dynamics.

## 1. What the stroboscopic generator construction measures

Let E = e^{πL_off} e^{πL_on} be the one-period map. It commutes with the parity
superoperator 𝒫(ρ) = ΠρΠ. Its unique fixed point ρ_F is parity even, so
Im Z_Q[ρ_F] = 0 exactly for any generator with ΠQΠ = N − Q. At late times

    ρ(mT) = ρ_F + c μ₁^m r₁ + (faster modes),   𝒫(r₁) = −r₁, μ₁ ≈ −1.

Every parity-odd functional F (magnetization, Im Z_Q, the paired contrast S) is
therefore asymptotically F'[ρ_F](r₁) c μ₁^m. All such signals share the decay rate
−ln|μ₁| and differ only by a constant prefactor.

Numerical check (N = 2, Fock dimension 32; `results/late_time_eps_*.json`,
`results/floquet_gp_late_time.png`):

| ε | −ln\|μ₁\| (Floquet) | σ_x fit | Im Z_xy fit | S fit | late ratio g/σ_x |
|---|---|---|---|---|---|
| −0.056 | 0.043301 | 0.043301 | 0.04330 (m 150–200) | 0.04329 (m 150–200) | −0.473 |
| 0.01 | 0.008462 | 0.008462 | 0.008457 (m 500–700) | 0.008448 (m 500–700) | +1.390 |

At ε = 0.01 the GP signals reach the asymptotic rate only after ~400 periods. Between
m = 100 and 300 their apparent rates are 5–7% (Im Z_xy) and 15–19% (S) lower. The reason is that ρ_F has
nearly degenerate even/odd eigenpairs, so Im Z_Q stays nonlinear in the odd component
until that component is smaller than the pair splitting. Consequences:

* the scatter (up to ~20%) between rates fitted from the supplied 60-period data is
  consistent with this, and
* a lifetime fitted from a finite window of a GP signal is *biased* relative to the
  Floquet lifetime. It is not a cleaner measure than the magnetization.

The sign flip of the transverse GP signal at ε = −0.056 in
`results/dicke_check_traces.png` is just a negative prefactor (−0.473).

Further remarks:

* **The loop is synthetic.** Nothing in the Dicke Liouvillian is covariant under
  e^{−iφQ_xy}, so Z_Q is a chosen nonlinear functional of ρ(mT), not the GP of a
  physical family of states. The original README already notes this.
* **The calibration is model-specific.** The u = 1/(4N) normalization is calibrated
  on ±x spin-coherent branches. Real branch states are entangled spin–cavity mixed
  states, so g_Q is not a universal imbalance measure.

## 2. A physical loop: micromotion of the Floquet steady state

**Loop.** ρ_F(t), 0 ≤ t ≤ T, with ρ_F(T) = ρ_F(0). Equivalently, ρ_F^{(τ)} is the
Floquet steady state of the square wave shifted by τ. Winding τ over one period
generates a closed loop of nonequilibrium steady states, exactly as winding the
drive phase does in the oscillator paper. The loop is not a unitary orbit. The
eigenvalues change along it, so there is no single-state reduction, and Tong et al.'s
kinematic formula (main text Eq. 2) must be evaluated along the whole period.

**Parity split (exact).** H(t), the dissipator and ρ_F commute with Π, so ρ_F(t) is
block diagonal in parity for all t, and

    Z_T = Z_even + Z_odd,   Z_s = Σ_{k∈s} λ_k e^{iγ_k}.

Degeneracies between sectors are harmless: the block parallel-transport condition is
parity diagonal. Reported quantities:

* Δ = arg(Z_even Z_odd*), the relative sector GP;
* C = −2 Re(Z_even Z_odd*)/(|Z_even|² + |Z_odd|²) ∈ [−1, 1]. C is weight aware:
  it is ≈ 0 if one sector carries negligible weight, and 1 iff Z_even = −Z_odd;
* the loop visibility |Z_T|.

**Ideal period-doubled pair (proposition).** Suppose:

* ρ_F(t) = ½[ρ_A(t) + Πρ_A(t)Π], with the supports of ρ_A and Πρ_AΠ orthogonal along
  the loop;
* one period exchanges the branches, ρ_A(T) = Πρ_A(0)Π;
* there is no inter-branch connection, ⟨Πa_k|∂_t a_k⟩ = 0.

Then every even eigenvector |a_k⟩ + Π|a_k⟩ has an odd partner |a_k⟩ − Π|a_k⟩ of equal
weight. One period maps the pair to ±e^{iβ_k}(…), and both carry the same transport
phase. Hence Z_odd = −Z_even, Δ = π and Z_T = 0, while the doubled loop has
Z_2T ≠ 0.

The π is a Z₂ holonomy: the two branches are exchanged after one period, a Möbius
structure. Deviations from π measure branch overlap and inter-branch tunnelling along
the micromotion, both finite-size effects. In the oscillator paper's language, the
time crystal turns one physical drive period into the parity half loop of Eq. (S14).

**Numerics.** `micromotion_gp` in `dicke_floquet.py` tracks eigenvectors inside each
parity block by maximum-overlap assignment and uses the same discrete transport
estimator as `test_gp_geometry`. Checks:

* Doubling the time resolution (400 → 800 steps per half period) changes Δ by
  < 5×10⁻⁴ rad over the whole N = 2 detuning scan.
* Fock dimension 24 vs 32 agrees to 10⁻⁷.
* ρ_F closes after one period to 10⁻¹⁵.

## 3. Results

### Coupling scan (N = 2, ε = 0.01; `results/floquet_coupling_N2.json`)

| λ₀ | μ₁ | \|Δ\|/π | C | \|Z_T\| |
|---|---|---|---|---|
| 0.1–0.7 | complex (no period doubling) | 0.00–0.08 | −0.01 → −0.73 | 0.94–1.00 |
| 0.8 | −0.701 − 0.131i | 0.31 | −0.53 | 0.83 |
| 0.9 | −0.9589 | 0.62 | 0.35 | 0.56 |
| 1.0 | −0.99157 | 0.80 | 0.80 | 0.31 |
| 1.2 | −0.99848 | 0.964 | 0.99 | 0.056 |
| 1.4 | −0.99912 | 0.997 | 1.00 | 0.004 |

### Atom-number scan (ε = 0.01, Fock dimension 16 + 4N; `results/floquet_atoms_eps0p01.json`)

| N | μ₁ | \|Δ\|/π | \|Z_T\| |
|---|---|---|---|
| 1 | −0.704 + 0.223i (no DTC) | 0.37 | 0.82 |
| 2 | −0.99157 | 0.796 | 0.31 |
| 3 | −0.99784 | 0.972 | 0.045 |
| 4 | −0.99822 | 0.990 | 0.016 |
| 5 | −0.99851 | 0.995 | 0.0076 |
| 6 | −0.99898 | 0.995 | 0.0080 |

The deficit π − |Δ| falls by ~20× from N = 2 to N = 4 and then saturates near
0.005π. Whether this residual is set by κ, by the drive, or is a genuine finite-N
plateau is open.

### Detuning scans (`results/floquet_eps_N2.json`, `results/floquet_eps_N4.json`)

**N = 4, −0.2 ≤ ε ≤ 0.25 (Fock dimension 36).** μ₁ is real and negative for
−0.1 ≤ ε ≤ 0.175 (and again at 0.25). The decay rate is smallest at ε = 0
(0.0012 per period) and grows toward both edges.

* Inside −0.025 ≤ ε ≤ 0.1: |Δ|/π = 0.94–0.995, C ≥ 0.98, |Z_T| ≤ 0.09.
* Below the optimum, Δ falls together with the DTC: 0.78π (ε = −0.05), 0.59π (−0.075),
  0.31π (−0.1), and 0.10π at −0.125, where μ₁ is complex.
* At ε = 0.075 and 0.1, Δ is still ≈ π although the decay is 40–100× faster than at
  ε = 0. This repeats the N = 2 finding below.
* **Accidental near-π values.** At ε = −0.15 and −0.175 there is no period doubling
  (μ₁ complex), yet |Δ| ≈ 0.89–0.90π. In both cases the sector amplitudes have
  collapsed (|Z_s| ≈ 0.12–0.18 against weights ≈ 0.5), so the sum-level phase is
  not meaningful there.

**Pair-resolved check** (`results/floquet_eps_N4_pairs.json`). The proposition
predicts more than Δ = π: matched even/odd eigenvectors should have *equal weights*
and *holonomies differing by π*. Ranking the six heaviest eigenvectors of each sector
by weight, define

    P_pair = Σ_k min(λ_k^e, λ_k^o)(−cos(γ_k^e − γ_k^o)) / Σ_k ½(λ_k^e + λ_k^o).

| ε | μ₁ | \|Δ\|/π (sum) | P_pair | weight in top 6 pairs |
|---|---|---|---|---|
| −0.175 | 0.320 + 0.671i | 0.90 | +0.35 | 0.80 |
| −0.15 | −0.670 + 0.237i | 0.89 | −0.04 | 0.85 |
| −0.10 | −0.810 | 0.31 | −0.53 | 0.94 |
| 0 | −0.9988 | 0.994 | **+0.987** | 1.00 |
| 0.075 | −0.956 | 0.995 | +0.75 | 0.93 |
| 0.125 | −0.831 | 0.75 | +0.55 | 0.69 |
| 0.20 | −0.655 + 0.048i | 0.58 | +0.51 | 0.38 |

What the pair data show:

* **Deep DTC (ε = 0).** One eigenvector pair carries 98.7% of the weight, with
  λ_e = 0.4939, λ_o = 0.4929 and relative holonomy 0.993π. This is the ideal-pair
  structure, realised almost exactly.
* **Large positive ε.** The weights pair exactly (|asymmetry| ≤ 0.04), so two
  orthogonal branches exist, but only the leading pairs are π-paired geometrically.
  P_pair then falls together with the DTC strength.
* **Negative ε.** Weight asymmetries are 2–16% and the pair holonomies are scattered.
  The sum-level near-π at ε = −0.15 is accidental (P_pair ≈ 0). A residual +0.35
  remains at −0.175.

Conclusion: the robust statement is pair-resolved. The sum-level Δ should always be
reported together with the sector coherence |Z_s|/w_s and the pair data.

**Δ is not a function of the lifetime.** At N = 2, μ₁ is real and negative over the
whole window −0.12 ≤ ε ≤ 0.15. The decay rate is minimal at ε ≈ 0.02
(0.0082 per period) and rises on both sides. Over the same window |Δ|/π stays
between 0.61 and 1. It reaches π at ε ≈ 0.09, where the decay is 4× faster than at
the optimum, and is 0.75π at ε ≈ −0.04.

So Δ measures how cleanly the two branches are separated and exchanged during the
micromotion, not how quickly they mix. A hypothesis worth testing: Δ ≈ π with fast
decay indicates decay by switching between well-separated branches. Nguyen & Reichman
attribute the transient regime around ε ≈ 0.09 to chaotic decorrelation, which fits
this picture. Δ < π would then indicate decay by branch overlap. At N = 2 this is
only suggestive.

### Kerr resonator, one pump period (`results/kerr_parity_half_loop.json`)

Setting: η/|U| = 3.5, γ₁ = γ₂ = κ = 0.1|U|, Fock dimension 50. The parity split
reproduces Z_half to 2×10⁻¹⁰.

| region | \|Δ\|/π | C | \|Z_half\| |
|---|---|---|---|
| low-detuning vacuum-like (Δ/\|U\| ≤ −5) | 0.05–0.13 | ≈ −0.8 | 0.88–0.99 |
| bright (0 ≤ Δ/\|U\| ≤ 25) | 0.98–0.9999 | 0.99–1.00 | ≤ 0.07 |
| first-order region (Δ/\|U\| = 27.5 → 30) | 0.89 → 0.41 | 0.92 → −0.26 | 0.14 → 0.50 |

The ideal-cat value π[1 − 2N_B/sinh 2N_B], with moment-matched N_B, gives 0.989π at
Δ = 0 (exact: 0.982π). It approaches π exponentially in the bright-state photon
number. The 4π loop chosen in the paper averages this structure away; the parity-
resolved half loop isolates it.

## 4. What is new and what is not

* **Known.** DTC ⇔ μ₁ ≈ −1 for the Floquet Liouvillian (Gong, Hamazaki & Ueda, and
  others). The π splitting of parity partners in closed Floquet DTCs is geometric
  (Schindler & Bukov, arXiv:2410.07029). Finite-N decay channels of the open Dicke
  DTC are studied in Nguyen & Reichman, arXiv:2610.27589.
* **Apparently new (limited search; please check further).**
  * A mixed-state, open-system geometric π pairing, read from the parity-symmetric
    Floquet steady state, with an exact ideal-pair limit.
  * A demonstration that it switches on with period doubling in the open Dicke
    model and converges to π with N.
  * The observation that the Kerr half loop of Eq. (S14) is the same object for a
    continuously driven period-doubling oscillator.
  * The finding that the geometric pairing and the lifetime are independent finite-
    size characteristics.
* **Not new.** The stroboscopic fixed-generator GP signals as lifetime or order
  diagnostics.

## 5. Caveats and open problems

* Only exact calculations, so N ≤ 6 here. Truncated Wigner gives no density-matrix
  eigensystem. Large N needs an approximate branch model, e.g. Gaussian states
  around the period-2 mean-field orbit.
* The deficit π − Δ saturates near 0.005π for N = 5–6 at κ = 0.05; its origin is
  untested.
* Measuring the micromotion GP needs ρ_F at many times within a period, or an
  interferometric scheme for a non-unitary loop. For the Kerr half loop the paper's
  Ramsey scheme already applies, because the loop is a unitary number rotation and
  the dispersive interaction provides it.
* The sum-level Δ can be accidentally near π when |Z_s| ≪ w_s (N = 4, ε ≤ −0.15).
  Use the pair-resolved P_pair, or a better-founded pair-matching rule, as the
  primary quantity in any paper.
* Eigenvector tracking assumes no exact crossings inside a parity block along the
  loop. Eigenvectors below weight 10⁻⁹ are not tracked; their total weight is
  reported.
* κ dependence and other DTC models have not yet been scanned.

## 6. Suggested next steps

1. Two-branch analytic model. Compute Δ − π from branch overlap and inter-branch
   connection. For the Kerr case this is already closed form.
2. N up to 8–10 at a few detunings, including the transient window
   0.05 ≤ ε ≤ 0.11 where Nguyen & Reichman report non-monotonic N dependence.
3. Combine (|μ₁|, Δ) as a two-parameter classification of finite-N DTC decay, and
   test the switching-versus-overlap hypothesis against quantum-trajectory
   statistics.
4. Paper framing: "drive-phase winding → loop of steady states" extends from phase
   locking (oscillator paper) to time-translation symmetry breaking (Floquet loop),
   with the Kerr half loop as the bridge.

## Reproducing

```bash
python dicke_floquet_scan.py --atoms 2 --cutoff 24 --param epsilon --values -0.12 0.15 28 --check-steps 800 --jobs 4 --out results/floquet_eps_N2
python dicke_floquet_scan.py --atoms 2 --cutoff 32 --param coupling --list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 --jobs 4 --out results/floquet_coupling_N2
python dicke_floquet_scan.py --param atoms --list 1 2 3 4 5 6 --cutoff 16 --cutoff-per-atom 4 --jobs 4 --out results/floquet_atoms_eps0p01
python dicke_floquet_scan.py --atoms 4 --cutoff 36 --param epsilon --values -0.2 0.25 19 --jobs 4 --out results/floquet_eps_N4
python dicke_floquet_scan.py --atoms 4 --cutoff 36 --param epsilon --list -0.175 -0.15 -0.1 0 0.075 0.125 0.2 --check-steps 800 --jobs 4 --out results/floquet_eps_N4_pairs
python dicke_late_time_check.py --eps -0.056 --periods 200 --fit-start 60 --out results/late_time_eps_m0p056
python dicke_late_time_check.py --eps 0.01 --periods 700 --fit-start 500 --out results/late_time_eps0p01
python kerr_parity_half_loop.py --eta 3.5 --delta -15 30 91 --out results/kerr_parity_half_loop
python plot_floquet_gp.py --coupling results/floquet_coupling_N2.json --detuning results/floquet_eps_N2.json results/floquet_eps_N4.json --atoms results/floquet_atoms_eps0p01.json --kerr results/kerr_parity_half_loop.json --late results/late_time_eps_m0p056.json results/late_time_eps0p01.json --out results/floquet_gp
```
