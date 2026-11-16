"""A newly implemented core for ``phase_eta-exact.py``.

This is a reconstruction from ``main_clean.pdf``, not a recovered copy of the
missing original helper.  It uses the manuscript Hamiltonian convention

    H = Delta*n + U*a.dag()**2*a**2
        + E*(exp(+i*phi)*a + exp(-i*phi)*a.dag())
        + eta*(exp(+i*theta)*a**2 + exp(-i*theta)*a.dag()**2).

The returned geometric amplitude is for the positively oriented full number
rotation R(varphi)=exp(-i*varphi*n), varphi from 0 to 2*pi.  With pure squeezing
this is theta from 0 to 4*pi.  With both drives present, their phases must be
wound together: phi -> phi+varphi and theta -> theta+2*varphi.

``N`` follows the existing driver and denotes the Fock-space dimension, not
the atom number of the Dicke model.  No large scan runs on module import.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qutip import Qobj, destroy, expect, steadystate

from gp_geometry import geometric_amplitude


def build_kerr_model(
    Delta: float,
    eta: float,
    N: int,
    U: float,
    E: float,
    phi: float,
    theta: float,
    gamma_1: float,
    gamma_2: float,
    kappa: float,
) -> tuple[Qobj, list[Qobj], Qobj]:
    """Return Hamiltonian, nonzero collapse operators, and number operator.

    The Kerr coefficient is ``U``, not ``U/2``.  Collapse rates use the
    convention D[c]rho = c*rho*c.dag() - {c.dag()*c,rho}/2.
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer Fock-space dimension >= 2")
    parameters = (Delta, eta, U, E, phi, theta, gamma_1, gamma_2, kappa)
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("All model parameters must be finite real numbers")
    if min(gamma_1, gamma_2, kappa) < 0:
        raise ValueError("Dissipation and pumping rates must be nonnegative")

    a = destroy(int(N))
    n = a.dag() * a
    aa = a * a
    H = (
        Delta * n
        + U * aa.dag() * aa
        + E * (np.exp(1j * phi) * a + np.exp(-1j * phi) * a.dag())
        + eta * (np.exp(1j * theta) * aa + np.exp(-1j * theta) * aa.dag())
    )
    c_ops = [
        np.sqrt(rate) * operator
        for rate, operator in ((gamma_1, a.dag()), (gamma_2, aa), (kappa, a))
        if rate > 0
    ]
    return H, c_ops, n


def steady_state_point(
    Delta: float,
    eta: float,
    N: int,
    U: float,
    E: float,
    phi: float,
    theta: float,
    gamma_1: float,
    gamma_2: float,
    kappa: float,
    *,
    solver_options: Mapping[str, Any] | None = None,
) -> Qobj:
    """Return one steady-state density matrix without repairing its spectrum.

    ``solver_options`` is passed to ``qutip.steadystate``.  A successful solve
    alone does not establish photon-cutoff convergence; compare larger N and
    inspect ``steady_state_diagnostics`` before interpreting a scan.
    """
    H, c_ops, _ = build_kerr_model(
        Delta, eta, N, U, E, phi, theta, gamma_1, gamma_2, kappa
    )
    if not c_ops:
        raise ValueError("A unique steady state is not specified without dissipation")
    return steadystate(H, c_ops, **dict(solver_options or {}))


def steady_state_diagnostics(
    ss: Qobj,
    H: Qobj,
    c_ops: list[Qobj],
    *,
    tail_levels: int = 5,
) -> dict[str, float | int]:
    """Report raw solver quality and Fock-tail weight without clipping.

    The residual is the absolute Frobenius norm of the master-equation right
    hand side, in the model's inverse-time units.  Small tail weight is useful
    evidence, but does not replace a calculation at a larger photon cutoff.
    """
    if tail_levels < 1:
        raise ValueError("tail_levels must be positive")
    rho = ss.full()
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("ss must be a density operator")
    hermitian_rho = 0.5 * (rho + rho.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian_rho)
    derivative = -1j * (H * ss - ss * H)
    for collapse in c_ops:
        cc = collapse.dag() * collapse
        derivative += collapse * ss * collapse.dag() - 0.5 * (cc * ss + ss * cc)
    dimension = rho.shape[0]
    tail_count = min(int(tail_levels), dimension)
    return {
        "fock_dimension": int(dimension),
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_frobenius": float(np.linalg.norm(rho - rho.conj().T)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalue_weight": float(-eigenvalues[eigenvalues < 0].sum()),
        "liouvillian_residual_frobenius": float(np.linalg.norm(derivative.full())),
        "tail_levels": int(tail_count),
        "top_fock_tail_probability": float(np.trace(rho[-tail_count:, -tail_count:]).real),
    }


def one_point(
    Delta: float,
    eta: float,
    N: int,
    U: float,
    E: float,
    phi: float,
    theta: float,
    gamma_1: float,
    gamma_2: float,
    kappa: float,
) -> tuple[float, complex]:
    """Return ``(mean_photon_number, full_loop_geometric_amplitude)``.

    This positional signature matches the supplied ``phase_eta-exact.py``.
    The complex amplitude retains visibility and permits callers to mark its
    phase undefined at a node, rather than treating ``angle(0)`` as a result.
    """
    ss = steady_state_point(
        Delta, eta, N, U, E, phi, theta, gamma_1, gamma_2, kappa
    )
    n = destroy(int(N)).dag() * destroy(int(N))
    return float(np.real(expect(n, ss))), complex(geometric_amplitude(ss, n))
