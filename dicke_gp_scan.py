"""Finite-time geometric probes of the periodically driven open Dicke model.

This program evaluates *synthetic unitary-loop* geometric amplitudes of exact
spin-cavity density matrices at a fixed Floquet strobe. It does not identify a
time-crystal phase from a threshold. In particular, late-window amplitudes are
finite-time measurements, not infinite-time phase labels.

Example (one point):
  python dicke_gp_scan.py --atoms 2 --cutoff 16 --periods 60 \
      --eps 0.01 --kappas 0.05 --out output/dicke_gp/point

Example (grid with a photon-cutoff comparison):
  python dicke_gp_scan.py --atoms 2 --cutoff 16 --check-cutoff 24 \
      --periods 60 --eps -0.056 0.01 0.051 0.11 \
      --kappas 0.02 0.05 0.1 --jobs 4 --out output/dicke_gp/grid

The coupling is on for the first half of T=2*pi and off for the second.
Frequencies are omega_c=1-epsilon and omega_s=1+epsilon. Photon loss is
kappa*D[a], so the collapse operator is sqrt(kappa)*a.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

# Do not multiply BLAS threads by the number of independent grid workers.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from joblib import Parallel, delayed, parallel_config
import qutip as qt
from scipy.sparse.linalg import expm_multiply

from gp_geometry import (
    prepare_generator,
    prepare_state,
    geometric_amplitude_from_spectrum,
)


def _validate(atoms: int, cutoff: int, periods: int, kappa: float) -> None:
    if isinstance(atoms, bool) or atoms < 1 or int(atoms) != atoms:
        raise ValueError("atoms must be a positive integer")
    if cutoff < 3 or int(cutoff) != cutoff:
        raise ValueError("cutoff must be an integer >= 3")
    if periods < 2 or int(periods) != periods:
        raise ValueError("periods must be an integer >= 2")
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")


def coherent_subharmonic_amplitude(values: np.ndarray, window: int = 20) -> float:
    """Absolute alternating mean over an even number of final strobe samples.

    An even window cancels a constant offset exactly. This is not a fitted
    decay rate or a criterion for a phase transition.
    """
    values = np.asarray(values)
    count = min(int(window), len(values) - 1)
    count -= count % 2
    if count < 2:
        raise ValueError("At least two late-time samples are required")
    indices = np.arange(len(values) - count, len(values))
    return float(abs(np.mean((-1.0) ** indices * values[-count:])))


def simulate_point(
    atoms: int = 2,
    cutoff: int = 16,
    epsilon: float = 0.01,
    kappa: float = 0.05,
    coupling: float = 1.0,
    periods: int = 60,
    *,
    include_wide: bool = False,
    tail_levels: int = 3,
    window: int = 20,
    degeneracy_atol: float = 1e-10,
    degeneracy_rtol: float = 1e-8,
) -> dict[str, Any]:
    """Propagate one point exactly in a truncated symmetric spin-cavity space.

    Return strobe arrays, complex geometric amplitudes, and numerical quality
    diagnostics. No photon-cutoff convergence is implied by a single run.
    ``g_xy`` is Im(Z_xy); ``toy_scaled_g_xy`` is sqrt(2)*(-1)**N*g_xy.
    The latter normalization is motivated by the explicit two-branch toy
    family and does not impose a bound for arbitrary density matrices.
    """
    _validate(atoms, cutoff, periods, kappa)
    if not all(np.isfinite(v) for v in (epsilon, kappa, coupling)):
        raise ValueError("All physical parameters must be finite")
    if not 1 <= tail_levels < cutoff:
        raise ValueError("tail_levels must be between 1 and cutoff-1")
    if window < 2:
        raise ValueError("window must be at least 2")

    started = time.perf_counter()
    j = atoms / 2.0
    spin_dim = atoms + 1
    dim = cutoff * spin_dim
    eye_spin = qt.qeye(spin_dim)
    eye_cavity = qt.qeye(cutoff)
    identity = qt.qeye([cutoff, spin_dim])
    a = qt.tensor(qt.destroy(cutoff), eye_spin)
    n = a.dag() * a
    jx = qt.tensor(eye_cavity, qt.jmat(j, "x"))
    jy = qt.tensor(eye_cavity, qt.jmat(j, "y"))
    jz = qt.tensor(eye_cavity, qt.jmat(j, "z"))
    h_off = (1.0 - epsilon) * n + (1.0 + epsilon) * jz
    h_on = h_off + 2.0 * coupling / np.sqrt(atoms) * (a + a.dag()) * jx
    collapse = [np.sqrt(kappa) * a] if kappa else []

    # The two exponentials handle the switching discontinuity explicitly.
    # Vectorization is column-major, as in QuTiP's operator_to_vector.
    half_period = np.pi
    half_maps = []
    for h in (h_on, h_off):
        generator = qt.liouvillian(h, collapse).data.as_scipy().tocsc()
        scaled_generator = half_period * generator
        half_maps.append((scaled_generator, scaled_generator.diagonal().sum()))

    u = 1.0 / (4.0 * atoms)
    v = np.sqrt(1.0 - u * u)
    generator_operators = {
        "n": n,
        "plus": j * identity + v * jz + u * jx,
        "minus": j * identity + v * jz - u * jx,
        "xy": j * identity + u * jx + v * jy,
    }
    if include_wide:
        generator_operators.update({
            "wide_plus": j * identity + (jz + jx) / np.sqrt(2.0),
            "wide_minus": j * identity + (jz - jx) / np.sqrt(2.0),
        })
    prepared_generators = {
        name: prepare_generator(operator.full())
        for name, operator in generator_operators.items()
    }

    initial_ket = qt.tensor(qt.basis(cutoff, 0), qt.spin_coherent(j, np.pi / 2, 0.0))
    density_vector = qt.ket2dm(initial_ket).full().reshape(-1, order="F")
    count = periods + 1
    result: dict[str, Any] = {
        "m": np.arange(count, dtype=int),
        "t": 2.0 * np.pi * np.arange(count),
        "sigma_x": np.empty(count),
        "nbar": np.empty(count),
        "tail_population": np.empty(count),
        "trace_error": np.empty(count),
        "hermiticity_error": np.empty(count),
        "min_eigenvalue": np.empty(count),
        "degeneracy_l1_correction": np.empty(count),
        "max_degeneracy_spread": np.empty(count),
    }
    for name in prepared_generators:
        result["Z_" + name] = np.empty(count, dtype=complex)

    dense_jx = jx.full()
    photon_numbers = np.repeat(np.arange(cutoff), spin_dim)
    tail_start = (cutoff - tail_levels) * spin_dim
    max_halfstep_tail = 0.0
    max_halfstep_trace_error = 0.0
    max_halfstep_nbar = 0.0

    def inspect_halfstep(vector: np.ndarray) -> None:
        nonlocal max_halfstep_tail, max_halfstep_trace_error, max_halfstep_nbar
        diagonal = vector.reshape((dim, dim), order="F").diagonal()
        max_halfstep_tail = max(max_halfstep_tail, float(np.real(diagonal[tail_start:].sum())))
        max_halfstep_trace_error = max(max_halfstep_trace_error, float(abs(diagonal.sum() - 1.0)))
        max_halfstep_nbar = max(max_halfstep_nbar, float(np.real(photon_numbers @ diagonal)))

    def record(index: int, vector: np.ndarray) -> None:
        rho = vector.reshape((dim, dim), order="F")
        diagonal = rho.diagonal()
        result["sigma_x"][index] = float(np.real(np.einsum("ij,ji->", dense_jx, rho)) / j)
        result["nbar"][index] = float(np.real(photon_numbers @ diagonal))
        result["tail_population"][index] = float(np.real(diagonal[tail_start:].sum()))
        result["trace_error"][index] = abs(np.trace(rho) - 1.0)
        result["hermiticity_error"][index] = np.linalg.norm(rho - rho.conj().T, ord="fro")
        state = prepare_state(rho, degeneracy_atol=degeneracy_atol,
                              degeneracy_rtol=degeneracy_rtol)
        result["min_eigenvalue"][index] = state.raw_min_eigenvalue
        result["degeneracy_l1_correction"][index] = state.degeneracy_l1_correction
        result["max_degeneracy_spread"][index] = state.max_degeneracy_spread
        for name, generator in prepared_generators.items():
            result["Z_" + name][index] = geometric_amplitude_from_spectrum(state, generator)

    inspect_halfstep(density_vector)
    record(0, density_vector)
    for m in range(1, count):
        for generator, trace_generator in half_maps:
            density_vector = expm_multiply(generator, density_vector, traceA=trace_generator)
            inspect_halfstep(density_vector)
        record(m, density_vector)

    result["S"] = np.imag(result["Z_plus"] * result["Z_minus"].conj())
    result["delta_Z"] = result["Z_plus"] - result["Z_minus"]
    result["g_xy"] = result["Z_xy"].imag.copy()
    result["visibility_xy"] = np.abs(result["Z_xy"])
    result["phase_xy"] = np.where(result["visibility_xy"] > 1e-6,
                                  np.angle(result["Z_xy"]), np.nan)
    result["toy_scaled_g_xy"] = np.sqrt(2.0) * (-1.0) ** atoms * result["g_xy"]
    if include_wide:
        result["S_wide"] = np.imag(result["Z_wide_plus"] * result["Z_wide_minus"].conj())
        result["delta_Z_wide"] = result["Z_wide_plus"] - result["Z_wide_minus"]

    amplitude_names = ["sigma_x", "S", "g_xy", "toy_scaled_g_xy", "Z_n"]
    if include_wide:
        amplitude_names.append("S_wide")
    amplitudes = {
        "late_amplitude_" + name: coherent_subharmonic_amplitude(result[name], window)
        for name in amplitude_names
    }
    result["diagnostics"] = {
        "atoms": atoms,
        "cutoff": cutoff,
        "epsilon": epsilon,
        "kappa": kappa,
        "coupling": coupling,
        "periods": periods,
        "dimension": dim,
        "sin_tilt": u,
        "tail_levels": tail_levels,
        "max_tail_population_at_half_period_samples": max_halfstep_tail,
        "max_nbar_at_half_period_samples": max_halfstep_nbar,
        "max_trace_error_at_half_period_samples": max_halfstep_trace_error,
        "max_hermiticity_error_at_strobes": float(np.max(result["hermiticity_error"])),
        "minimum_density_eigenvalue_at_strobes": float(np.min(result["min_eigenvalue"])),
        "degeneracy_atol": degeneracy_atol,
        "degeneracy_rtol": degeneracy_rtol,
        "maximum_degeneracy_l1_correction": float(np.max(result["degeneracy_l1_correction"])),
        "maximum_degeneracy_spread": float(np.max(result["max_degeneracy_spread"])),
        "window_samples": min(window, periods) - min(window, periods) % 2,
        "minimum_geometric_visibility_xy": float(result["visibility_xy"].min()),
        "phase_node_threshold": 1e-6,
        "runtime_seconds": time.perf_counter() - started,
        **amplitudes,
    }
    return result


def run_grid(
    *, atoms: int, cutoff: int, epsilons: list[float], kappas: list[float],
    coupling: float, periods: int, jobs: int = 1, include_wide: bool = False,
    tail_levels: int = 3, window: int = 20,
    degeneracy_atol: float = 1e-10, degeneracy_rtol: float = 1e-8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return arrays ordered [kappa, epsilon, strobe] and JSON-safe metadata."""
    if not 1 <= jobs <= 4:
        raise ValueError("jobs must be in [1, 4]")
    common = dict(atoms=atoms, cutoff=cutoff, coupling=coupling, periods=periods,
                  include_wide=include_wide, tail_levels=tail_levels, window=window,
                  degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol)
    points = [(eps, kap) for kap in kappas for eps in epsilons]
    if jobs == 1:
        results = [simulate_point(epsilon=eps, kappa=kap, **common) for eps, kap in points]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=jobs)(
                delayed(simulate_point)(epsilon=eps, kappa=kap, **common)
                for eps, kap in points
            )
    shape = (len(kappas), len(epsilons))
    arrays = {
        "epsilon_values": np.asarray(epsilons),
        "kappa_values": np.asarray(kappas),
        "m": results[0]["m"],
        "t": results[0]["t"],
    }
    for key, value in results[0].items():
        if isinstance(value, np.ndarray) and key not in ("m", "t"):
            arrays[key] = np.stack([result[key] for result in results]).reshape(shape + value.shape)
    for key in results[0]["diagnostics"]:
        if key.startswith("late_amplitude_"):
            arrays[key] = np.asarray([result["diagnostics"][key] for result in results]).reshape(shape)
    metadata = {
        "model": "Periodically modulated open Dicke model, symmetric j=N/2 sector",
        "array_axes": ["kappa", "epsilon", "strobe"],
        "frequency_convention": "omega_c=1-epsilon, omega_s=1+epsilon",
        "period": 2.0 * np.pi,
        "switching": "coupling on for first half-period, off for second half-period",
        "loss_convention": "kappa D[a], collapse=sqrt(kappa)*a",
        "initial_state": "cavity vacuum tensor spin coherent state along +x",
        "geometric_loop": "rho(phi)=exp(-i phi Q) rho exp(+i phi Q), phi from 0 to 2pi",
        "generator_xy": "Q_xy=j+u Jx+sqrt(1-u^2) Jy, u=1/(4N)",
        "generators_pair": "Q_plus/minus=j+sqrt(1-u^2) Jz +/- u Jx, u=1/(4N)",
        "interpretation": "Finite-time subharmonic persistence only; no phase labels or thermodynamic-limit inference",
        "cutoff_note": "Half-period tail samples are a warning diagnostic, not a proof of convergence; compare cutoffs",
        "toy_scaled_g_xy_note": "sqrt(2)*(-1)^N Im Z_xy; toy-family normalization, not a universal bound",
        "parameters": {**common, "epsilons": epsilons, "kappas": kappas, "jobs": jobs},
        "points": [result["diagnostics"] for result in results],
    }
    return arrays, metadata


def save_grid(prefix: Path | str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    prefix = Path(prefix)
    if prefix.suffix in (".npz", ".json"):
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atoms", type=int, default=2)
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--periods", type=int, default=60)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.05])
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--wide-tilt", action="store_true")
    parser.add_argument("--tail-levels", type=int, default=3)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--degeneracy-atol", type=float, default=1e-10)
    parser.add_argument("--degeneracy-rtol", type=float, default=1e-8)
    parser.add_argument("--check-cutoff", type=int, default=None,
                        help="also run this larger cutoff and save direct trajectory differences")
    parser.add_argument("--out", type=Path, default=Path("output/dicke_gp/scan"))
    args = parser.parse_args()
    if args.check_cutoff is not None and args.check_cutoff <= args.cutoff:
        parser.error("--check-cutoff must exceed --cutoff")
    options = dict(atoms=args.atoms, cutoff=args.cutoff, epsilons=args.eps, kappas=args.kappas,
                   coupling=args.coupling, periods=args.periods, jobs=args.jobs,
                   include_wide=args.wide_tilt, tail_levels=args.tail_levels, window=args.window,
                   degeneracy_atol=args.degeneracy_atol, degeneracy_rtol=args.degeneracy_rtol)
    arrays, metadata = run_grid(**options)
    prefix = args.out.with_suffix("") if args.out.suffix in (".npz", ".json") else args.out
    save_grid(prefix, arrays, metadata)
    print(f"Saved {prefix}.npz and {prefix}.json", flush=True)
    for point in metadata["points"]:
        print(json.dumps(point), flush=True)
    if args.check_cutoff is not None:
        higher, high_metadata = run_grid(**{**options, "cutoff": args.check_cutoff})
        high_prefix = Path(str(prefix) + f"_cutoff{args.check_cutoff}")
        save_grid(high_prefix, higher, high_metadata)
        comparison = {
            "cutoffs": [args.cutoff, args.check_cutoff],
            "axes": ["kappa", "epsilon"],
            "max_absolute_difference_over_strobes": {
                name: np.max(np.abs(arrays[name] - higher[name]), axis=-1).tolist()
                for name in ("sigma_x", "S", "g_xy", "toy_scaled_g_xy", "Z_n", "Z_xy", "nbar")
            },
            "note": "Numerical comparisons only; choose convergence tolerances for the claimed result",
        }
        Path(str(prefix) + "_cutoff_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
