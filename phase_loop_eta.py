import os
import time
import logging
import gc

import numpy as np
from numpy import linalg as la
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from qutip import destroy, steadystate, expect

# ============================================================
# Minimal eta-Delta sweep: save only <a^dag a> and GP to .npz
# Axis convention:
#   x-axis: Delta_list
#   y-axis: eta_list
# Array shape:
#   (len(eta_list), len(Delta_list))
# ============================================================

# ---------------- User parameters ----------------
N = 50

gamma_1 = 0
gamma_2 = 0.1
kappa = 0.1

# Pure squeeze-drive scan. Keep E = 0 for the exact squeeze GP formula.
E = 0.0
phi = 0.0
theta0 = 0.0

Delta_list = np.linspace(-35, 5, 151)
eta_list = np.linspace(0.0, 5, 151)

N_JOBS = 32
OUT_FILE = f"eta_delta_nbar_gp_N={N}_g1={gamma_1}_g2={gamma_2}_kappa={kappa}.npz"

# ---------------- Logging ----------------
log_file = os.environ.get("PBS_JOBID", "local") + "_eta_delta_npz.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Avoid BLAS oversubscription when using joblib workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def sq_QSL_TI_phi_theta(N, Delta, E, phi, eta, theta, gamma_1, gamma_2, kappa):
    a = destroy(N)
    n = a.dag() * a
    U= 1
    H = (
        Delta * n + U/2* a.dag() * a.dag() * a * a
        + E * (np.exp(-1j * phi) * a + np.exp(1j * phi) * a.dag())
        + eta * (np.exp(-1j * theta) * a * a + np.exp(1j * theta) * a.dag() * a.dag())
    )

    c_ops = [
        np.sqrt(gamma_1) * a.dag(),
        np.sqrt(gamma_2) * a * a,
        np.sqrt(kappa) * a,
    ]

    ss = steadystate(H, c_ops)
    return H, c_ops, ss


def gp_exact_squeeze_from_ss(ss, N):
    rho = ss.full()

    # Suppress tiny numerical non-Hermiticity
    rho = 0.5 * (rho + rho.conj().T)

    p, V = la.eigh(rho)
    p = np.real_if_close(p).real

    # Optional but recommended: remove tiny numerical negative probabilities
    p = np.clip(p, 0.0, None)
    if np.sum(p) > 0:
        p = p / np.sum(p)

    n_vals = np.arange(N, dtype=float)
    n_op = np.diag(n_vals)
    parity_op = np.diag((-1.0) ** n_vals)

    n_exp = np.einsum("ik,ij,jk->k", V.conj(), n_op, V).real
    parity_exp = np.einsum("ik,ij,jk->k", V.conj(), parity_op, V).real

    Z = np.sum(p * parity_exp * np.exp(-1j * np.pi * n_exp))

    return float(np.angle(Z))


def one_point(Delta, eta):
    a = destroy(N)
    _, _, ss = sq_QSL_TI_phi_theta(
        N=N,
        Delta=Delta,
        E=E,
        phi=phi,
        eta=eta,
        theta=theta0,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        kappa=kappa,
    )

    nbar = float(np.real(expect(a.dag() * a, ss)))
    gp = gp_exact_squeeze_from_ss(ss, N)
    return nbar, gp


def main():
    t0 = time.time()
    log.info("Starting eta-Delta sweep: %d eta × %d Delta = %d points, N_JOBS=%d",
             len(eta_list), len(Delta_list), len(eta_list) * len(Delta_list), N_JOBS)
    log.info("Parameters: N=%d, gamma_1=%g, gamma_2=%g, kappa=%g, E=%g",
             N, gamma_1, gamma_2, kappa, E)

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(one_point)(Delta, eta)
        for eta in eta_list
        for Delta in Delta_list
    )

    arr = np.array(results, dtype=float).reshape(len(eta_list), len(Delta_list), 2)
    nbar_array = arr[:, :, 0]
    gp_array = arr[:, :, 1]

    np.savez(
        OUT_FILE,
        nbar_array=nbar_array,
        gp_array=gp_array,
        Delta_list=Delta_list,
        eta_list=eta_list,
        N=N,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        kappa=kappa,
        E=E,
        phi=phi,
        theta0=theta0,
    )

    log.info("Saved %s", OUT_FILE)
    log.info("nbar_array shape = %s, gp_array shape = %s", nbar_array.shape, gp_array.shape)
    log.info("Total wall time: %.1f s", time.time() - t0)

    del results, arr, nbar_array, gp_array
    get_reusable_executor().shutdown(wait=True)
    gc.collect()


if __name__ == "__main__":
    main()
