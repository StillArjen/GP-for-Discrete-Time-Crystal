"""Independent numerical checks for the generic geometric construction.

Run: python -m unittest -v test_gp_geometry
"""

import unittest
from dataclasses import replace

import numpy as np
from gp_geometry import (
    amplitudes_for_generators, geometric_amplitude,
    geometric_amplitude_from_spectrum, prepare_generator, prepare_state,
)


def random_unitary(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension))
    return np.linalg.qr(matrix)[0]


def spin_operators(atoms):
    j = atoms / 2
    m = np.arange(j, -j-1, -1)
    raising = np.zeros((atoms+1, atoms+1), dtype=complex)
    for col in range(1, atoms+1):
        raising[col-1, col] = np.sqrt(j*(j+1) - m[col]*(m[col]+1))
    return ((raising + raising.conj().T)/2,
            (raising - raising.conj().T)/(2j), np.diag(m))


class GeometricAmplitudeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(6483)

    def test_discrete_parallel_transport_matches_closed_formula(self):
        """Independent Bargmann-product evaluation along the entire loop."""
        basis = random_unitary(self.rng, 4)
        weights = np.array([0.05, 0.15, 0.30, 0.50])
        rho = (basis * weights) @ basis.conj().T
        Q = np.diag([0, 1, 2, 4])
        expected = geometric_amplitude(rho, Q)
        phases = np.linspace(0.0, 2 * np.pi, 8193)
        observed = 0.0j
        for k, p in enumerate(weights):
            path = np.exp(-1j * phases[:, None] * np.diag(Q)) * basis[:, k]
            backward_overlaps = np.einsum("ij,ij->i", path[1:].conj(), path[:-1])
            transport = np.prod(backward_overlaps / np.abs(backward_overlaps))
            observed += p * np.vdot(path[0], path[-1]) * transport
        self.assertLess(abs(expected - observed), 2e-6)

    def test_degenerate_block_is_invariant_to_eigenbasis_rotation(self):
        basis = random_unitary(self.rng, 4)
        rho = (basis * [0.1, 0.1, 0.4, 0.4]) @ basis.conj().T
        Q = np.diag([0.0, 1.0, 3.0, 5.0])
        state = prepare_state(rho)
        original = geometric_amplitude_from_spectrum(state, Q)
        rotated_vectors = state.eigenvectors.copy()
        for start, stop in state.blocks:
            rotated_vectors[:, start:stop] = (
                rotated_vectors[:, start:stop] @ random_unitary(self.rng, stop-start))
        rotated = replace(state, eigenvectors=rotated_vectors)
        self.assertLess(abs(original - geometric_amplitude_from_spectrum(rotated, Q)), 2e-13)

    def test_maximally_mixed_constant_path_has_unit_amplitude(self):
        unitary = random_unitary(self.rng, 5)
        Q = (unitary * np.arange(5)) @ unitary.conj().T
        self.assertAlmostEqual(abs(geometric_amplitude(np.eye(5)/5, Q) - 1), 0, places=12)
        # Arbitrary noninteger generator also gives a constant path at any angle.
        self.assertLess(abs(geometric_amplitude(np.eye(5)/5, Q / 3,
                                              angle=0.731) - 1), 2e-13)

    def test_endpoint_factor_for_half_loop(self):
        # The state has only even-parity support, but is not a number eigenstate.
        psi = np.array([np.sqrt(0.7), 0, np.sqrt(0.3)], dtype=complex)
        rho = np.outer(psi, psi.conj())
        Q = np.diag([0, 1, 2])
        self.assertLess(abs(geometric_amplitude(rho, Q, angle=-np.pi) -
                            np.exp(-1j * np.pi * 0.6)), 2e-13)
        odd = np.array([0, np.sqrt(0.65), 0, np.sqrt(0.35)], dtype=complex)
        rho_odd = np.outer(odd, odd.conj())
        self.assertLess(abs(geometric_amplitude(rho_odd, np.diag(range(4)), angle=np.pi)
                            - (-np.exp(1j*np.pi*1.7))), 2e-13)

    def test_invalid_inputs_are_rejected(self):
        rho = np.diag([0.7, 0.3])
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            geometric_amplitude(rho, np.array([[0, 1], [0, 0]]))
        with self.assertRaisesRegex(ValueError, "positive"):
            geometric_amplitude(np.diag([1.1, -0.1]), np.diag([0, 1]))
        with self.assertRaisesRegex(ValueError, "normalized"):
            geometric_amplitude(rho * 2, np.diag([0, 1]))
        with self.assertRaisesRegex(ValueError, "integer"):
            geometric_amplitude(rho, np.diag([0, 0.3]))
        state = np.ones((2, 2)) / 2
        with self.assertRaisesRegex(ValueError, "not closed"):
            geometric_amplitude(state, np.diag([0, 1]), angle=np.pi)

    def test_parity_even_generator_and_tilted_spin_generator(self):
        # A spin-1/2 pair pointing along opposite x directions.
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sz = np.diag([1.0, -1.0])
        rho_plus = (np.eye(2) + sx) / 2
        rho_minus = (np.eye(2) - sx) / 2
        Q_z = (np.eye(2) + sz) / 2
        Q_tilt = (np.eye(2) + (sx + sz)/np.sqrt(2)) / 2
        self.assertLess(abs(geometric_amplitude(rho_plus, Q_z) -
                            geometric_amplitude(rho_minus, Q_z)), 2e-13)
        self.assertGreater(abs(geometric_amplitude(rho_plus, Q_tilt) -
                               geometric_amplitude(rho_minus, Q_tilt)), 1)

    def test_exact_branch_mixture_blindness_including_equal_weights(self):
        # Finite-number analogue of |+alpha> and |-alpha>, with orthogonal spins.
        cavity_plus = np.array([np.sqrt(0.7), np.sqrt(0.3)], dtype=complex)
        cavity_minus = cavity_plus * [1, -1]
        spin_plus = np.array([1, 1])/np.sqrt(2)
        spin_minus = np.array([1, -1])/np.sqrt(2)
        plus, minus = np.kron(cavity_plus, spin_plus), np.kron(cavity_minus, spin_minus)
        Q = np.kron(np.diag([0, 1]), np.eye(2))
        for fraction in [0.0, 0.2, 0.5, 0.8, 1.0]:
            rho = fraction*np.outer(plus, plus.conj()) + (1-fraction)*np.outer(minus, minus.conj())
            self.assertLess(abs(geometric_amplitude(rho, Q) - np.exp(2j*np.pi*0.3)), 2e-13)

    def test_reuse_matches_convenience_api(self):
        basis = random_unitary(self.rng, 3)
        rho = (basis * [0.2, 0.3, 0.5]) @ basis.conj().T
        Q = np.diag([0, 1, 2])
        state, generator = prepare_state(rho), prepare_generator(Q)
        computed = amplitudes_for_generators(state, {"n": generator})
        self.assertLess(abs(computed["n"] - geometric_amplitude(rho, Q)), 1e-14)

    def test_paired_tilted_generator_recovers_branch_imbalance(self):
        # An independently known result on an orthogonal spin-branch mixture.
        # For sin(theta)=1/(4*N), Im[Z_+ conj(Z_-)] = 2*f-1 exactly.
        for atoms in [2, 3, 4]:
            j = atoms / 2
            m = np.arange(j, -j-1, -1)
            raising = np.zeros((atoms+1, atoms+1), dtype=complex)
            for col in range(1, atoms+1):
                raising[col-1, col] = np.sqrt(j*(j+1) - m[col]*(m[col]+1))
            Jx, Jz = (raising + raising.conj().T)/2, np.diag(m)
            charges, spin_basis = np.linalg.eigh(Jx)
            minus, plus = spin_basis[:, 0], spin_basis[:, -1]
            sine = 1/(4*atoms)
            common = j*np.eye(atoms+1) + np.sqrt(1-sine*sine)*Jz
            generators = {"plus": prepare_generator(common+sine*Jx),
                          "minus": prepare_generator(common-sine*Jx)}
            for fraction in [0.0, 0.13, 0.5, 0.82, 1.0]:
                rho = (fraction*np.outer(plus, plus.conj()) +
                       (1-fraction)*np.outer(minus, minus.conj()))
                z = amplitudes_for_generators(rho, generators)
                signal = np.imag(z["plus"]*np.conj(z["minus"]))
                self.assertAlmostEqual(signal, 2*fraction-1, places=11)

    def test_single_transverse_generator_recovers_branch_imbalance(self):
        # Both transverse spin components reverse under parity.  The small
        # fixed tilt sets the two pure branch phases to N*pi +/- pi/4.
        for atoms in [2, 3, 4]:
            j = atoms / 2
            Jx, Jy, _ = spin_operators(atoms)
            _, basis = np.linalg.eigh(Jx)
            minus, plus = basis[:, 0], basis[:, -1]
            u = 1/(4*atoms)
            Q = prepare_generator(j*np.eye(atoms+1) + u*Jx + np.sqrt(1-u*u)*Jy)
            for fraction in [0.0, 0.13, 0.5, 0.82, 1.0]:
                rho = (fraction*np.outer(plus, plus.conj()) +
                       (1-fraction)*np.outer(minus, minus.conj()))
                Z = geometric_amplitude(rho, Q)
                signal = np.sqrt(2)*(-1)**atoms*Z.imag
                self.assertAlmostEqual(signal, 2*fraction-1, places=11)

    def test_single_transverse_generator_is_parity_conjugated(self):
        for atoms in [2, 3, 4]:
            j = atoms / 2
            Jx, Jy, Jz = spin_operators(atoms)
            u = 1/(4*atoms)
            Q = j*np.eye(atoms+1) + u*Jx + np.sqrt(1-u*u)*Jy
            parity = np.diag(np.exp(1j*np.pi*(np.diag(Jz)+j)))
            self.assertLess(np.linalg.norm(parity@Q@parity.conj().T -
                                           (atoms*np.eye(atoms+1)-Q)), 2e-13)
            basis = random_unitary(self.rng, atoms+1)
            probabilities = np.arange(1, atoms+2, dtype=float)
            probabilities /= probabilities.sum()
            rho = (basis*probabilities)@basis.conj().T
            Z = geometric_amplitude(rho, Q)
            Z_parity = geometric_amplitude(parity@rho@parity.conj().T, Q)
            self.assertLess(abs(Z_parity - Z.conjugate()), 2e-12)

    def test_roundoff_diagnostics_are_recorded(self):
        state = prepare_state(np.diag([-1e-12, 0.3, 0.7 + 1e-12]))
        self.assertLess(state.raw_min_eigenvalue, 0)
        self.assertAlmostEqual(state.clipped_negative_mass, 1e-12)
        self.assertAlmostEqual(state.eigenvalues.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
