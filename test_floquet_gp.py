"""Checks for the Floquet spectrum and the parity-resolved loop GP.

Run: python -m unittest -v test_floquet_gp
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import unittest

import numpy as np
from scipy.linalg import expm

from dicke_floquet import (build_model, excitation_loop_amplitudes, floquet_spectrum,
                           micromotion_gp, one_period, sector_loop_amplitudes,
                           static_steady_state)
from gp_geometry import geometric_amplitude
from kerr_parity_half_loop import sector_half_loop


def random_parity_symmetric_state(rng, dim):
    parity = (-1.0) ** np.arange(dim)
    rho = np.zeros((dim, dim), dtype=complex)
    weights = rng.uniform(0.2, 1.0, size=dim)
    weights /= weights.sum()
    start = 0
    for idx in (np.flatnonzero(parity > 0), np.flatnonzero(parity < 0)):
        block = rng.normal(size=(len(idx), len(idx))) + 1j * rng.normal(size=(len(idx), len(idx)))
        basis = np.linalg.qr(block)[0]
        w = weights[start:start + len(idx)]
        start += len(idx)
        rho[np.ix_(idx, idx)] = (basis * w) @ basis.conj().T
    return rho, parity


def number_orbit(rho, stop, steps):
    n = np.arange(rho.shape[0])
    for phi in np.linspace(0.0, stop, steps + 1):
        phase = np.exp(-1j * phi * n)
        yield (phase[:, None] * rho) * phase.conj()[None, :]


def coherent(alpha, dim):
    n = np.arange(dim)
    log_fact = np.cumsum(np.log(np.maximum(n, 1)))
    amp = np.exp(-abs(alpha) ** 2 / 2 + n * np.log(abs(alpha)) - 0.5 * log_fact)
    return amp * np.exp(1j * n * np.angle(alpha))


class SectorLoopAmplitudeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2611)

    def test_full_number_loop_matches_closed_formula(self):
        rho, parity = random_parity_symmetric_state(self.rng, 8)
        result = sector_loop_amplitudes(number_orbit(rho, 2 * np.pi, 4000), parity)
        n = np.diag(np.arange(8.0))
        self.assertLess(abs(result["Z_T"] - geometric_amplitude(rho, n)), 2e-5)

    def test_half_number_loop_matches_parity_endpoint_formula(self):
        rho, parity = random_parity_symmetric_state(self.rng, 8)
        result = sector_loop_amplitudes(number_orbit(rho, np.pi, 4000), parity)
        n = np.diag(np.arange(8.0))
        self.assertLess(abs(result["Z_T"] - geometric_amplitude(rho, n, angle=np.pi)), 2e-5)
        # Independent closed form used by kerr_parity_half_loop.py, sector by sector.
        closed = sector_half_loop(rho)
        self.assertLess(abs(result["Z_even"] - closed["even"]), 2e-5)
        self.assertLess(abs(result["Z_odd"] - closed["odd"]), 2e-5)

    def test_exchanged_cat_pair_has_pi_relative_phase(self):
        # rho = (|a><a| + |-a><-a|)/2; the half rotation exchanges the branches.
        dim, alpha = 48, 1.6
        plus, minus = coherent(alpha, dim), coherent(-alpha, dim)
        rho = 0.5 * (np.outer(plus, plus.conj()) + np.outer(minus, minus.conj()))
        parity = (-1.0) ** np.arange(dim)
        result = sector_loop_amplitudes(number_orbit(rho, np.pi, 3000), parity)
        NB = alpha ** 2
        expected = np.pi * (1 - 2 * NB / np.sinh(2 * NB))
        self.assertAlmostEqual(abs(result["relative_sector_phase"]), expected, places=5)
        w_plus, w_minus = (1 + np.exp(-2 * NB)) / 2, (1 - np.exp(-2 * NB)) / 2
        n_plus, n_minus = NB * np.tanh(NB), NB / np.tanh(NB)
        Z = w_plus * np.exp(1j * np.pi * n_plus) - w_minus * np.exp(1j * np.pi * n_minus)
        self.assertLess(abs(result["Z_T"] - Z), 2e-5)

    def test_unexchanged_state_has_no_pi_pairing(self):
        # Full 2*pi rotation of the same cat mixture returns each branch to itself.
        dim, alpha = 48, 1.6
        plus, minus = coherent(alpha, dim), coherent(-alpha, dim)
        rho = 0.5 * (np.outer(plus, plus.conj()) + np.outer(minus, minus.conj()))
        parity = (-1.0) ** np.arange(dim)
        result = sector_loop_amplitudes(number_orbit(rho, 2 * np.pi, 6000), parity)
        NB = alpha ** 2
        expected = 2 * np.pi * (NB / np.tanh(NB) - NB * np.tanh(NB))
        self.assertAlmostEqual(abs(np.angle(np.exp(1j * (result["relative_sector_phase"] + expected)))),
                               0.0, places=4)
        # 2*pi*(n_- - n_+) = 0.385 rad here: small, and nowhere near pi.
        self.assertLess(abs(result["relative_sector_phase"]), np.pi / 4)


class CounterRotatingPhaseLoopTests(unittest.TestCase):
    """The theta-winding loop of the Dicke model and its single-state reduction."""

    def test_theta_zero_is_the_dicke_coupling(self):
        import qutip as qt
        cutoff, atoms, eps, g = 8, 2, 0.03, 0.9
        model = build_model(atoms=atoms, cutoff=cutoff, epsilon=eps, kappa=0.1, coupling=g)
        a = qt.tensor(qt.destroy(cutoff), qt.qeye(atoms + 1))
        jx = qt.tensor(qt.qeye(cutoff), qt.jmat(atoms / 2, "x"))
        jz = qt.tensor(qt.qeye(cutoff), qt.jmat(atoms / 2, "z"))
        h = (1 - eps) * a.dag() * a + (1 + eps) * jz + 2 * g / np.sqrt(atoms) * (a + a.dag()) * jx
        L = qt.liouvillian(h, [np.sqrt(0.1) * a]).data.as_scipy()
        self.assertLess(abs(L - model.L_on).max(), 1e-12)

    def test_theta_winding_rotates_the_steady_state(self):
        theta = 1.3
        base = build_model(atoms=2, cutoff=12, epsilon=0.01, kappa=0.2, coupling=0.8)
        wound = build_model(atoms=2, cutoff=12, epsilon=0.01, kappa=0.2, coupling=0.8, theta=theta)
        rotation = np.exp(-0.5j * theta * np.diag(base.ops["nexc"]))
        rho, rho_theta = static_steady_state(base), static_steady_state(wound)
        expected = (rotation[:, None] * rho) * rotation.conj()[None, :]
        self.assertLess(np.abs(expected - rho_theta).max(), 1e-10)

    def test_single_state_formula_equals_tracked_loop(self):
        model = build_model(atoms=2, cutoff=12, epsilon=0.01, kappa=0.2, coupling=0.8)
        rho = static_steady_state(model)
        nexc = np.diag(model.ops["nexc"]).real
        closed = excitation_loop_amplitudes(model, rho)
        for stop, key in ((np.pi, "half"), (2 * np.pi, "full")):
            def orbit():
                for phi in np.linspace(0.0, stop, 3001):
                    phase = np.exp(-1j * phi * nexc)
                    yield (phase[:, None] * rho) * phase.conj()[None, :]
            tracked = sector_loop_amplitudes(orbit(), model.parity, weight_floor=1e-12)
            if key == "half":
                self.assertLess(abs(tracked["Z_even"] - closed["Z_even"]), 1e-5)
                self.assertLess(abs(tracked["Z_odd"] - closed["Z_odd"]), 1e-5)
            else:
                self.assertLess(abs(tracked["Z_T"] - closed["Z_full"]), 1e-5)

    def test_free_segment_duration_is_respected(self):
        default = build_model(atoms=1, cutoff=6, epsilon=0.02, kappa=0.1)
        longer = build_model(atoms=1, cutoff=6, epsilon=0.02, kappa=0.1, t_off=2 * np.pi)
        vec = np.zeros(default.dim ** 2, dtype=complex)
        vec[0] = 1.0
        once = one_period(default, vec)
        manual = expm(np.pi * default.L_off.toarray()) @ (expm(np.pi * default.L_on.toarray()) @ vec)
        self.assertLess(np.abs(once - manual).max(), 1e-10)
        twice_off = expm(2 * np.pi * default.L_off.toarray()) @ (expm(np.pi * default.L_on.toarray()) @ vec)
        self.assertLess(np.abs(one_period(longer, vec) - twice_off).max(), 1e-10)


class FloquetSpectrumTests(unittest.TestCase):
    def test_sector_restricted_spectrum_matches_dense_map(self):
        model = build_model(atoms=1, cutoff=6, epsilon=0.02, kappa=0.1, coupling=1.0)
        dense = expm(np.pi * model.L_off.toarray()) @ expm(np.pi * model.L_on.toarray())
        even, odd = model.sector_index["even"], model.sector_index["odd"]
        # The one-period map does not couple the two superoperator parity sectors.
        self.assertLess(np.linalg.norm(dense[np.ix_(even, odd)]), 1e-12)
        self.assertLess(np.linalg.norm(dense[np.ix_(odd, even)]), 1e-12)
        for sector, idx in (("even", even), ("odd", odd)):
            reference = np.linalg.eigvals(dense[np.ix_(idx, idx)])
            reference = reference[np.argsort(-np.abs(reference))]
            computed, _ = floquet_spectrum(model, sector, k=3)
            self.assertLess(abs(computed[0] - reference[0]), 1e-9)   # leading mode is unique here
            # Compare by modulus: a complex-conjugate pair has no preferred member.
            np.testing.assert_allclose(np.abs(computed), np.abs(reference[:3]), atol=1e-9)
        self.assertLess(abs(floquet_spectrum(model, "even", k=1)[0][0] - 1.0), 1e-10)

    def test_micromotion_loop_closes_and_converges(self):
        model = build_model(atoms=2, cutoff=12, epsilon=0.01, kappa=0.05, coupling=1.0)
        _, vectors = floquet_spectrum(model, "even", k=2)
        rho = vectors[:, 0].reshape((model.dim, model.dim), order="F")
        rho = rho / np.trace(rho)
        rho = 0.5 * (rho + rho.conj().T)
        vec = rho.reshape(-1, order="F")
        coarse = micromotion_gp(model, vec, steps_per_half=200)
        fine = micromotion_gp(model, vec, steps_per_half=400)
        self.assertLess(coarse["closure_error"], 1e-10)
        self.assertLess(abs(np.angle(np.exp(1j * (coarse["relative_sector_phase"]
                                                  - fine["relative_sector_phase"])))), 2e-3)
        self.assertAlmostEqual(coarse["weight_even"] + coarse["weight_odd"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
