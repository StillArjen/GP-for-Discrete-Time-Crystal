# GP for Discrete Time Crystals

Python simulations and geometric-phase diagnostics for Kerr and driven Dicke
models. See [the scientific notes](GP_DIAGNOSTIC_README.md) for conventions,
interpretation, numerical checks, and limitations of the supplied results.

[GP_DTC_EXTENSION.md](GP_DTC_EXTENSION.md) extends the diagnostic to the open Dicke
model: an exact single-state reduction through the counter-rotating coupling phase
(generator a†a + J_z + N/2), the weak-coupling formula, the superradiant transition,
and a control showing which loop detects the time crystal.

[FLOQUET_GP_NOTES.md](FLOQUET_GP_NOTES.md) assesses whether the construction yields
new DTC physics. In short: the stroboscopic fixed-generator GP decays at the Floquet
rate of the magnetization. The kinematic GP of the Floquet steady state's
micromotion loop instead shows a parity-sector π pairing (geometric period
doubling), which also appears in the Kerr half loop.

## Repository layout

- Root Python files: geometry routines, model simulations, plotting, and tests.
- `docs/papers/`: the three supplied reference PDFs.
- `results/`: supplied numerical datasets, diagnostics, and figures, plus the
  Floquet analysis outputs (`floquet_*`, `late_time_*`, `kerr_parity_half_loop*`) and
  the extension outputs (`static_*`, `dtc_control_*`, `extension_*`).
- `dicke_floquet.py`, `dicke_floquet_scan.py`: Floquet-Liouvillian spectrum and the
  parity-resolved micromotion GP; `dicke_late_time_check.py`: stroboscopic signals
  vs the Floquet rate; `kerr_parity_half_loop.py`: parity-resolved Kerr half loop;
  `plot_floquet_gp.py`: their figures; `test_floquet_gp.py`: their tests.
- `dicke_static_scan.py`, `dicke_weak_coupling_check.py`, `plot_extension.py`: the
  undriven (superradiant) scans, the weak-coupling check, and the extension figures.
- `tmp/`: ignored scratch files; use this for new smoke-test outputs.
- `requirements-gp.txt`: dependency constraints supplied with the project.
- `requirements-gp-lock.txt`: original dependency snapshot from the archive.

The original upload is preserved in Git history. Its contents have been extracted;
temporary PDF previews and old smoke outputs remain local and are not committed.
Supplied results are historical artifacts, not outputs regenerated during setup.

## Setup

Use Python 3.12 and an isolated virtual environment. On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gp-lock.txt
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
export XDG_CACHE_HOME="$PWD/tmp/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest test_gp_geometry test_gp_models test_floquet_gp -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The original
scientific notes include Windows examples for the larger calculations.

## Small simulations

Run commands from the repository root with the virtual environment active:

```bash
python kerr_gp_check.py --cutoff 8 --delta -1 1 3 --eta 0.1 0.3 2 --jobs 1 --out tmp/onboarding/kerr
python plot_gp_checks.py kerr tmp/onboarding/kerr.npz
python dicke_gp_scan.py --atoms 2 --cutoff 8 --check-cutoff 10 --periods 6 --window 4 --eps -0.01 0.01 --kappas 0.05 --jobs 1 --out tmp/onboarding/dicke
python plot_gp_checks.py dicke tmp/onboarding/dicke.npz
```

These small runs exercise the simulation and plotting paths; they do not establish
physical convergence. Large sweeps in `phase_eta-exact.py` and `phase_loop_eta.py`
are optional and can be expensive. Choose cutoffs and parameter grids deliberately.
Use a new output prefix to avoid overwriting the supplied `results/` files.

No server, API key, or GPU is required for this workflow.
