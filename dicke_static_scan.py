"""Geometric phase across the superradiant transition of the undriven open Dicke model.

The coupling is always on.  For each (N, lambda) the exact steady state is
computed, and the GP of the counter-rotating-phase loop is evaluated with the
exact single-state reduction (generator N_exc = n + Jz + N/2, see
dicke_floquet.excitation_loop_amplitudes).  The slowest Liouvillian rates of the
parity-odd and parity-even sectors are recorded as the spectral reference.

  python dicke_static_scan.py --atoms 2 4 6 --couplings 0.1 1.4 27 --kappa 0.05 \
      --epsilon 0.01 --cutoff 12 --cutoff-per-atom 4 --jobs 4 --out results/static_superradiant
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from joblib import Parallel, delayed

from dicke_floquet import (build_model, excitation_loop_amplitudes, static_slowest_rates,
                           static_steady_state)


def json_path(prefix: Path) -> Path:
    return prefix if prefix.suffix == ".json" else Path(str(prefix) + ".json")


def cplx(z):
    return [float(np.real(z)), float(np.imag(z))]


def run_point(atoms, cutoff, coupling, epsilon, kappa, rates):
    started = time.perf_counter()
    model = build_model(atoms=atoms, cutoff=cutoff, epsilon=epsilon, kappa=kappa, coupling=coupling)
    rho = static_steady_state(model)
    residual = float(np.max(np.abs(model.L_on @ rho.reshape(-1, order="F"))))
    ex = excitation_loop_amplitudes(model, rho)
    j = atoms / 2
    point = {
        "atoms": atoms, "cutoff": cutoff, "coupling": coupling, "epsilon": epsilon, "kappa": kappa,
        "nbar_per_atom": float(np.real(np.trace(model.ops["n"] @ rho))) / atoms,
        "jz_over_j": float(np.real(np.trace(model.ops["jz"] @ rho))) / j,
        "min_eigenvalue": float(np.linalg.eigvalsh(rho)[0]),
        "tail_top3": float(np.real(np.diag(rho)[-3 * (atoms + 1):].sum())),
        "residual": residual,
        "weight_even": ex["weight_even"], "weight_odd": ex["weight_odd"],
        "Z_even": cplx(ex["Z_even"]), "Z_odd": cplx(ex["Z_odd"]), "Z_half": cplx(ex["Z_half"]),
        "Z_full": cplx(ex["Z_full"]),
        "relative_sector_phase": ex["relative_sector_phase"], "pairing": ex["pairing"],
    }
    if rates:
        try:
            r = static_slowest_rates(model, k=2)
            point["odd_slowest"] = cplx(r["odd"][0])
            point["even_slowest_nonzero"] = cplx(r["even"][1]) if len(r["even"]) > 1 else [np.nan, np.nan]
        except Exception as error:  # record, do not abort the scan
            point["rates_error"] = repr(error)
    point["runtime_seconds"] = time.perf_counter() - started
    return point


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--couplings", type=float, nargs=3, metavar=("START", "STOP", "NUM"),
                   default=[0.1, 1.4, 27])
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--cutoff", type=int, default=12)
    p.add_argument("--cutoff-per-atom", type=float, default=4.0)
    p.add_argument("--no-rates", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    couplings = np.linspace(a.couplings[0], a.couplings[1], int(a.couplings[2]))
    tasks = [(N, int(round(a.cutoff + a.cutoff_per_atom * N)), float(g)) for N in a.atoms for g in couplings]
    points = Parallel(n_jobs=a.jobs)(delayed(run_point)(N, c, g, a.epsilon, a.kappa, not a.no_rates)
                                     for N, c, g in tasks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json_path(a.out).write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()},
                                            "points": points}, indent=1), encoding="utf-8")
    for q in points:
        line = (f"N={q['atoms']} cut={q['cutoff']} g={q['coupling']:.3f} n/N={q['nbar_per_atom']:.3f} "
                f"|D|/pi={abs(q['relative_sector_phase'])/np.pi:.3f} C={q['pairing']:+.3f} "
                f"|Zh|={np.hypot(*q['Z_half']):.3f} argZf/pi={np.angle(complex(*q['Z_full']))/np.pi:+.3f} "
                f"tail={q['tail_top3']:.1e} res={q['residual']:.1e}")
        if "odd_slowest" in q:
            line += f" odd_rate={-q['odd_slowest'][0]:.2e}"
        print(line + f" t={q['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
