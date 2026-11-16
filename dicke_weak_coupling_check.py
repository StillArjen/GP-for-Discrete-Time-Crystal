"""Weak-coupling check: exact arg Z_full of the N_exc loop vs the perturbative formula.

  python dicke_weak_coupling_check.py --out results/static_weak_coupling
"""
import argparse
import json
import os
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

from dicke_floquet import (build_model, excitation_loop_amplitudes, perturbative_excitation_gp,
                           static_steady_state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=int, nargs="+", default=[2, 4])
    p.add_argument("--couplings", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for N in a.atoms:
        for g in a.couplings:
            m = build_model(atoms=N, cutoff=12 + 4 * N, epsilon=a.epsilon, kappa=a.kappa, coupling=g)
            rho = static_steady_state(m)
            exact = float(np.angle(excitation_loop_amplitudes(m, rho)["Z_full"]))
            pert = perturbative_excitation_gp(m, rho)
            rows.append({"atoms": N, "coupling": g, "exact_gamma": exact, "perturbative_gamma": pert})
            print(f"N={N} g={g:.2f} exact/pi={exact/np.pi:+.5f} perturbative/pi={pert/np.pi:+.5f} ratio={pert/exact:.4f}",
                  flush=True)
    out = a.out if a.out.suffix == ".json" else Path(str(a.out) + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"epsilon": a.epsilon, "kappa": a.kappa, "points": rows}, indent=1))


if __name__ == "__main__":
    main()
