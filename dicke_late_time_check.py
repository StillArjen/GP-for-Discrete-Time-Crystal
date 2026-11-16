"""Check that stroboscopic parity-odd GP signals decay at the Floquet rate.

For each detuning, propagate the exact state for many periods (using
dicke_gp_scan.simulate_point), compute the leading parity-odd eigenvalue mu_1
of the one-period map, and compare -ln|mu_1| with log-linear fits of the
demodulated magnetization, single-generator GP signal and paired contrast.

  python dicke_late_time_check.py --eps -0.056 0.01 --periods 200 --out results/late_time_check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from joblib import Parallel, delayed

from dicke_floquet import build_model, floquet_spectrum
from dicke_gp_scan import simulate_point

SIGNALS = ("sigma_x", "toy_scaled_g_xy", "S")


def json_path(prefix: Path) -> Path:
    """Append .json without treating dots in the prefix (e.g. eps0.01) as a suffix."""
    return prefix if prefix.suffix == ".json" else Path(str(prefix) + ".json")


def run(eps, atoms, cutoff, kappa, periods, fit_start):
    traj = simulate_point(atoms=atoms, cutoff=cutoff, epsilon=eps, kappa=kappa, periods=periods)
    model = build_model(atoms=atoms, cutoff=cutoff, epsilon=eps, kappa=kappa)
    odd, _ = floquet_spectrum(model, "odd", k=3)
    m = traj["m"]
    fits = {}
    for key in SIGNALS:
        y = np.abs(traj[key][fit_start:])
        fits[key] = float(-np.polyfit(m[fit_start:], np.log(y), 1)[0]) if np.all(y > 0) else None
    late = slice(periods - 20, periods + 1)
    return {
        "epsilon": eps, "atoms": atoms, "cutoff": cutoff, "kappa": kappa, "periods": periods,
        "fit_start": fit_start,
        "odd_eigenvalues": [[float(v.real), float(v.imag)] for v in odd],
        "floquet_rate": float(-np.log(abs(odd[0]))),
        "fitted_rates": fits,
        "late_ratio_g_over_sigma": float(np.mean(traj["toy_scaled_g_xy"][late] / traj["sigma_x"][late])),
        "late_ratio_S_over_sigma": float(np.mean(traj["S"][late] / traj["sigma_x"][late])),
        "max_tail": traj["diagnostics"]["max_tail_population_at_half_period_samples"],
        "m": m.tolist(), **{key: traj[key].tolist() for key in SIGNALS},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, nargs="+", default=[-0.056, 0.01])
    p.add_argument("--atoms", type=int, default=2)
    p.add_argument("--cutoff", type=int, default=32)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--periods", type=int, default=200)
    p.add_argument("--fit-start", type=int, default=60)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = Parallel(n_jobs=a.jobs)(delayed(run)(e, a.atoms, a.cutoff, a.kappa, a.periods, a.fit_start)
                                   for e in a.eps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json_path(a.out).write_text(json.dumps(rows), encoding="utf-8")
    for r in rows:
        print(f"eps={r['epsilon']:+.3f} floquet={r['floquet_rate']:.6f} fitted="
              + " ".join(f"{k}:{v:.6f}" for k, v in r["fitted_rates"].items())
              + f" g/sx={r['late_ratio_g_over_sigma']:+.4f} S/sx={r['late_ratio_S_over_sigma']:+.4f}"
              f" tail={r['max_tail']:.1e}")


if __name__ == "__main__":
    main()
