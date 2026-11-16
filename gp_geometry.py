"""Geometric amplitudes of closed, isospectral density-matrix orbits.

The path is rho(phi) = exp(-i phi Q) rho exp(+i phi Q).  For an integer
spectrum Q and a full 2*pi rotation, the amplitude is

    Z = sum_a p_a Tr[exp(i 2*pi P_a Q P_a)]

where a labels distinct density eigenvalues and each trace is over the
corresponding eigenspace.  This is the degeneracy-safe prescription in the
GP supplement.  It is a geometric functional of a specified state and loop;
it does not by itself certify synchronization or time-crystalline order.

All arrays can also be supplied as QuTiP Qobj objects.  No QuTiP dependency
is needed by this module.  A prepared state and prepared generators can be
reused to avoid repeated eigendecompositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value.full() if hasattr(value, "full") else value,
                        dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite entries")
    return result


def _check_nonnegative_tolerances(**values) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class StateSpectrum:
    """Validated density eigensystem and the numerical degeneracy partition.

    Blocks whose eigenvalue spread is within the specified tolerance are
    treated as exactly degenerate and assigned their mean eigenvalue. This
    is a numerical resolution choice, not a theorem about near degeneracy.
    ``max_degeneracy_spread`` and ``degeneracy_l1_correction`` quantify it;
    vary the tolerances when a result depends on nearly degenerate spectra.
    No positive-eigenvalue cutoff is applied.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    blocks: tuple[tuple[int, int], ...]
    block_weights: np.ndarray
    raw_min_eigenvalue: float
    trace_error: float
    hermiticity_error: float
    clipped_negative_mass: float
    max_degeneracy_spread: float
    degeneracy_l1_correction: float


@dataclass(frozen=True)
class PreparedGenerator:
    """Validated Hermitian generator, including its eigensystem."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def prepare_state(
    rho,
    *,
    degeneracy_atol: float = 1e-10,
    degeneracy_rtol: float = 1e-8,
    hermiticity_tol: float = 1e-8,
    trace_tol: float = 1e-8,
    positivity_tol: float = 1e-8,
) -> StateSpectrum:
    """Validate and diagonalize a density matrix, correcting only roundoff.

    Significant non-Hermiticity, trace error, or negative populations raise
    ValueError instead of being silently hidden by symmetrization/clipping.
    The Hermiticity tolerance is an absolute Frobenius-norm tolerance.
    """
    _check_nonnegative_tolerances(
        degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol,
        hermiticity_tol=hermiticity_tol, trace_tol=trace_tol,
        positivity_tol=positivity_tol,
    )
    matrix = _matrix(rho, "rho")
    hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, "fro"))
    if hermiticity_error > hermiticity_tol:
        raise ValueError(f"rho is not Hermitian: error={hermiticity_error:.3g}")
    trace_error = float(abs(np.trace(matrix) - 1.0))
    if trace_error > trace_tol:
        raise ValueError(f"rho is not normalized: trace error={trace_error:.3g}")
    matrix = (matrix + matrix.conj().T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum = float(eigenvalues[0])
    if minimum < -positivity_tol:
        raise ValueError(f"rho is not positive: minimum eigenvalue={minimum:.3g}")
    negative_mass = float(-eigenvalues[eigenvalues < 0.0].sum())
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues /= eigenvalues.sum()

    blocks = []
    start = 0
    for stop in range(1, len(eigenvalues)):
        # Compare to the first member: avoid chaining a sequence of small
        # gaps into a block with a much larger total spectral spread.
        tolerance = degeneracy_atol + degeneracy_rtol * max(
            abs(eigenvalues[start]), abs(eigenvalues[stop]))
        if eigenvalues[stop] - eigenvalues[start] > tolerance:
            blocks.append((start, stop))
            start = stop
    blocks.append((start, len(eigenvalues)))
    weights = np.array([eigenvalues[a:b].mean() for a, b in blocks])
    max_spread = max(float(eigenvalues[b - 1] - eigenvalues[a]) for a, b in blocks)
    correction = sum(float(np.abs(eigenvalues[a:b] - weight).sum())
                     for (a, b), weight in zip(blocks, weights))
    return StateSpectrum(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        blocks=tuple(blocks), block_weights=weights,
        raw_min_eigenvalue=minimum, trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        clipped_negative_mass=negative_mass,
        max_degeneracy_spread=max_spread,
        degeneracy_l1_correction=correction,
    )


def prepare_generator(Q, *, hermiticity_tol: float = 1e-10) -> PreparedGenerator:
    """Require a Hermitian Q and cache its spectral decomposition."""
    _check_nonnegative_tolerances(hermiticity_tol=hermiticity_tol)
    matrix = _matrix(Q, "Q")
    error = float(np.linalg.norm(matrix - matrix.conj().T, "fro"))
    scale = max(1.0, float(np.linalg.norm(matrix, "fro")))
    if error > hermiticity_tol * scale:
        raise ValueError(f"Q must be Hermitian: relative error={error / scale:.3g}")
    matrix = (matrix + matrix.conj().T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return PreparedGenerator(matrix, eigenvalues, eigenvectors)


def geometric_amplitude_from_spectrum(
    state: StateSpectrum,
    Q,
    *,
    angle: float = 2 * np.pi,
    require_integer_spectrum: bool | None = None,
    integer_tol: float = 1e-8,
    closure_tol: float = 1e-8,
) -> complex:
    """Return the complex amplitude, retaining the orbit's endpoint factor.

    For an arbitrary closed orbit the expression evaluated is
    Tr[rho exp(-i angle Q) exp(i angle D_rho(Q))].  Closure is required.
    For angle=+/-2*pi the integer-spectrum condition is required by default;
    pass require_integer_spectrum=False to allow another closed orbit.
    A phase is undefined when the returned amplitude vanishes.
    """
    _check_nonnegative_tolerances(integer_tol=integer_tol, closure_tol=closure_tol)
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    generator = Q if isinstance(Q, PreparedGenerator) else prepare_generator(Q)
    if generator.matrix.shape != state.eigenvectors.shape:
        raise ValueError("rho and Q dimensions differ")
    if require_integer_spectrum is None:
        require_integer_spectrum = bool(np.isclose(abs(angle), 2 * np.pi,
                                                  atol=1e-12, rtol=0))
    integer_error = np.max(np.abs(generator.eigenvalues -
                                  np.rint(generator.eigenvalues)))
    if require_integer_spectrum and integer_error > integer_tol:
        raise ValueError(f"Q does not have integer spectrum: error={integer_error:.3g}")

    endpoint_phases = np.exp(-1j * angle * generator.eigenvalues)
    identity_endpoint = bool(np.max(np.abs(endpoint_phases - 1)) <= 1e-12)
    endpoint_in_state_basis = None
    if not identity_endpoint:
        overlap = state.eigenvectors.conj().T @ generator.eigenvectors
        endpoint_in_state_basis = (overlap * endpoint_phases) @ overlap.conj().T
        p = state.eigenvalues
        closure_error = float(np.linalg.norm(
            endpoint_in_state_basis * (p[None, :] - p[:, None]), "fro"))
        if closure_error > closure_tol:
            raise ValueError(f"state orbit is not closed: error={closure_error:.3g}")

    # One dense basis transformation is much faster than a separate matrix-
    # vector product for every nondegenerate eigenvector in a large sweep.
    projected_generator = (state.eigenvectors.conj().T @ generator.matrix
                           @ state.eigenvectors)
    amplitude = 0.0j
    for (start, stop), weight in zip(state.blocks, state.block_weights):
        if weight == 0.0:
            continue  # A precisely null eigenspace has zero contribution.
        projected = projected_generator[start:stop, start:stop]
        projected = (projected + projected.conj().T) * 0.5
        if identity_endpoint:
            charges = np.linalg.eigvalsh(projected)
            amplitude += weight * np.exp(1j * angle * charges).sum()
        else:
            charges, basis = np.linalg.eigh(projected)
            endpoint_block = endpoint_in_state_basis[start:stop, start:stop]
            endpoint_diagonal = np.diag(basis.conj().T @ endpoint_block @ basis)
            amplitude += weight * np.dot(endpoint_diagonal,
                                         np.exp(1j * angle * charges))
    return complex(amplitude)


def geometric_amplitude(
    rho, Q, *, angle: float = 2 * np.pi,
    degeneracy_atol: float = 1e-10, degeneracy_rtol: float = 1e-8,
    hermiticity_tol: float = 1e-8, trace_tol: float = 1e-8,
    positivity_tol: float = 1e-8,
    require_integer_spectrum: bool | None = None,
    integer_tol: float = 1e-8, closure_tol: float = 1e-8,
) -> complex:
    """Convenience wrapper: validate rho, then compute one orbit amplitude."""
    state = prepare_state(
        rho, degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol,
        hermiticity_tol=hermiticity_tol, trace_tol=trace_tol,
        positivity_tol=positivity_tol,
    )
    return geometric_amplitude_from_spectrum(
        state, Q, angle=angle, require_integer_spectrum=require_integer_spectrum,
        integer_tol=integer_tol, closure_tol=closure_tol,
    )


def amplitudes_for_generators(
    rho_or_spectrum, generators: Mapping[str, object], *, angle: float = 2 * np.pi,
    state_options: dict | None = None, **amplitude_options,
) -> dict[str, complex]:
    """Evaluate many fixed generators using a single density eigensystem."""
    state = (rho_or_spectrum if isinstance(rho_or_spectrum, StateSpectrum)
             else prepare_state(rho_or_spectrum, **(state_options or {})))
    return {name: geometric_amplitude_from_spectrum(
        state, generator, angle=angle, **amplitude_options)
        for name, generator in generators.items()}
