# GP for Discrete Time Crystals

Python simulations and geometric-phase diagnostics for Kerr and driven Dicke
models. See [the scientific notes](GP_DIAGNOSTIC_README.md) for conventions,
interpretation, numerical checks, and limitations of the supplied results.

## Repository layout

- Root Python files: geometry routines, model simulations, plotting, and tests.
- `docs/papers/`: the three supplied reference PDFs.
- `results/`: supplied numerical datasets, diagnostics, and figures.
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
python -m unittest test_gp_geometry test_gp_models -v
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
