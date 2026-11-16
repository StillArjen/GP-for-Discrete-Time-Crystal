import os

# Put these before numpy/scipy imports to avoid BLAS oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from _phase_eta_exact_core import one_point


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- User parameters ----------------
N = 100

gamma_1 = 0.1       # incoherent one-photon pumping D[a^dag]
gamma_2 = 0.1       # two-photon loss D[a^2]
kappa = 0.1         # one-photon loss D[a]
U = -1

E = 0.0
phi = 0.0
theta0 = 0.0

Delta_list = np.linspace(-15, 30, 251)
eta_list = np.linspace(0.001, 5, 251)

N_JOBS = -1

OUT_FILE = os.path.join(
    SCRIPT_DIR,
    f"eta_delta_nbar_gp_N={N}_U={U}_g1={gamma_1}_g2={gamma_2}_kappa={kappa}.npz"
)


def main():
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(one_point)(
            Delta, eta, N, U, E, phi, theta0,
            gamma_1, gamma_2, kappa,
        )
        for eta in eta_list
        for Delta in Delta_list
    )

    shape = (len(eta_list), len(Delta_list))
    nbar_array = np.array([r[0] for r in results], dtype=float).reshape(shape)
    Z_array = np.array([r[1] for r in results], dtype=np.complex128).reshape(shape)
    gp_array = np.angle(Z_array)
    Z_abs_array = np.abs(Z_array)

    np.savez(
        OUT_FILE,
        nbar_array=nbar_array,
        gp_array=gp_array,
        Z_array=Z_array,
        Z_abs_array=Z_abs_array,
        Delta_list=Delta_list,
        eta_list=eta_list,
        N=N,
        U=U,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        kappa=kappa,
        E=E,
        phi=phi,
        theta0=theta0,
    )

    del results, nbar_array, gp_array, Z_array, Z_abs_array
    get_reusable_executor().shutdown(wait=True)
    gc.collect()


if __name__ == "__main__":
    main()
