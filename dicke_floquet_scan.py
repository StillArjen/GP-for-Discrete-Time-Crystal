"""Scan the Floquet spectrum and micromotion GP of the open Dicke model.

Example:
  python dicke_floquet_scan.py --atoms 2 --cutoff 24 --param epsilon \
      --values -0.12 0.15 28 --kappa 0.05 --jobs 4 --out results/floquet_eps_N2

``--t-off`` sets the free segment in units of pi (2 gives a period of 3 pi,
which returns the branches instead of swapping them; a control experiment).
``--param`` selects the scanned parameter (epsilon, coupling, kappa or atoms);
the others are fixed.  ``--values a b n`` is a linspace; ``--list`` gives
explicit values.  For an atoms scan, ``--cutoff-per-atom`` adds photon levels
in proportion to N.  Each point records the leading parity-even and
parity-odd eigenvalues of the one-period map and the parity-resolved
kinematic GP of the Floquet steady state's micromotion loop, the pair-resolved
score P_pair, and the counter-rotating-phase (N_exc) loop GP of the Floquet
steady state at the strobe and at the end of the coupled segment.
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

from scipy.sparse.linalg import expm_multiply

from dicke_floquet import (build_model, excitation_loop_amplitudes, floquet_spectrum,
                           micromotion_gp, strobe_observables)


def json_path(prefix: Path) -> Path:
    """Append .json without treating dots in the prefix (e.g. eps0.01) as a suffix."""
    return prefix if prefix.suffix == ".json" else Path(str(prefix) + ".json")


def pair_pairing(top_even, top_odd):
    """P_pair: weight-ranked even/odd pairs, min weight x (-cos of holonomy difference)."""
    te, to = np.asarray(top_even), np.asarray(top_odd)
    k = min(len(te), len(to))
    te, to = te[:k], to[:k]
    num = np.sum(np.minimum(te[:, 0], to[:, 0]) * (-np.cos(te[:, 1] - to[:, 1])))
    return float(num / np.sum(0.5 * (te[:, 0] + to[:, 0]))), float(np.sum(te[:, 0] + to[:, 0]))


def excitation_summary(prefix, ex):
    return {f"{prefix}_relative_sector_phase": ex["relative_sector_phase"],
            f"{prefix}_pairing": ex["pairing"],
            f"{prefix}_Z_even": [ex["Z_even"].real, ex["Z_even"].imag],
            f"{prefix}_Z_odd": [ex["Z_odd"].real, ex["Z_odd"].imag],
            f"{prefix}_Z_full": [ex["Z_full"].real, ex["Z_full"].imag]}


def run_point(atoms, cutoff, epsilon, kappa, coupling, steps_per_half, check_steps,
              t_off_over_pi=1.0):
    started = time.perf_counter()
    model = build_model(atoms=atoms, cutoff=cutoff, epsilon=epsilon, kappa=kappa,
                        coupling=coupling, t_off=t_off_over_pi * np.pi)
    even_vals, even_vecs = floquet_spectrum(model, "even", k=3)
    odd_vals, _ = floquet_spectrum(model, "odd", k=3)
    rho = even_vecs[:, 0].reshape((model.dim, model.dim), order="F")
    rho = rho / np.trace(rho)
    rho = 0.5 * (rho + rho.conj().T)
    rho_vec = rho.reshape(-1, order="F")
    obs = strobe_observables(model, rho_vec)
    tail = float(np.real(np.diag(rho)[-3 * (atoms + 1):].sum()))
    gp = micromotion_gp(model, rho_vec, steps_per_half=steps_per_half)
    # Exact counter-rotating-phase loop (single-state formula) at the strobe and
    # at the end of the coupled segment.
    rho_on = np.zeros_like(rho_vec)
    even_idx = model.sector_index["even"]
    rho_on[even_idx] = expm_multiply(model.t_on * model.sector_L["even"][0], rho_vec[even_idx])
    exc = {**excitation_summary("exc_strobe", excitation_loop_amplitudes(model, rho_vec)),
           **excitation_summary("exc_on", excitation_loop_amplitudes(model, rho_on))}
    p_pair, top_weight = pair_pairing(gp["top_even"], gp["top_odd"])
    point = {
        "atoms": atoms, "cutoff": cutoff, "epsilon": epsilon, "kappa": kappa,
        "coupling": coupling, "t_off_over_pi": t_off_over_pi,
        "P_pair": p_pair, "top_pair_weight": top_weight, **exc,
        "even_eigenvalues": [[float(v.real), float(v.imag)] for v in even_vals],
        "odd_eigenvalues": [[float(v.real), float(v.imag)] for v in odd_vals],
        "mu1": [float(odd_vals[0].real), float(odd_vals[0].imag)],
        "dtc_rate_per_period": float(-np.log(abs(odd_vals[0]))),
        "even_gap_rate_per_period": float(-np.log(abs(even_vals[1]))),
        "steady_tail_population_top3": tail,
        **{"steady_" + k: v for k, v in obs.items()},
        **{k: v for k, v in gp.items() if not k.startswith("phasors")
           and not isinstance(v, complex)},
        **{k: [v.real, v.imag] for k, v in gp.items() if isinstance(v, complex)},
    }
    if check_steps:
        fine = micromotion_gp(model, rho_vec, steps_per_half=check_steps)
        point["relative_sector_phase_fine"] = fine["relative_sector_phase"]
        point["pi_pairing_fine"] = fine["pi_pairing"]
        point["Z_T_fine"] = [fine["Z_T"].real, fine["Z_T"].imag]
    point["runtime_seconds"] = time.perf_counter() - started
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atoms", type=int, default=2)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--cutoff-per-atom", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--t-off", type=float, default=1.0,
                        help="duration of the free segment in units of pi (default 1: T = 2 pi)")
    parser.add_argument("--param", choices=["epsilon", "coupling", "kappa", "atoms"],
                        default="epsilon")
    parser.add_argument("--values", type=float, nargs=3, metavar=("START", "STOP", "NUM"))
    parser.add_argument("--list", type=float, nargs="+")
    parser.add_argument("--steps-per-half", type=int, default=400)
    parser.add_argument("--check-steps", type=int, default=0,
                        help="also evaluate the GP with this many steps per half period")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.list:
        values = list(args.list)
    elif args.values:
        values = list(np.linspace(args.values[0], args.values[1], int(args.values[2])))
    else:
        parser.error("give --values or --list")

    def config(value):
        cfg = dict(atoms=args.atoms, cutoff=args.cutoff, epsilon=args.epsilon,
                   kappa=args.kappa, coupling=args.coupling)
        cfg[args.param] = int(value) if args.param == "atoms" else float(value)
        if args.cutoff_per_atom:
            cfg["cutoff"] = int(round(args.cutoff + args.cutoff_per_atom * cfg["atoms"]))
        return cfg

    points = Parallel(n_jobs=args.jobs)(
        delayed(run_point)(**config(v), steps_per_half=args.steps_per_half,
                           check_steps=args.check_steps, t_off_over_pi=args.t_off) for v in values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"scanned": args.param, "values": values, "args": {k: str(v) for k, v in vars(args).items()},
            "points": points}
    json_path(args.out).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for p in points:
        print(f"{args.param}={p[args.param]:<8.4g} cutoff={p['cutoff']:3d} mu1={p['mu1'][0]:+.5f}{p['mu1'][1]:+.5f}i "
              f"rate={p['dtc_rate_per_period']:.5f} dphi/pi={p['relative_sector_phase']/np.pi:+.4f} "
              f"P={p['pi_pairing']:+.4f} |ZT|={np.hypot(*p['Z_T']):.4f} |Ze|={np.hypot(*p['Z_even']):.3f} "
              f"|Zo|={np.hypot(*p['Z_odd']):.3f} tail={p['steady_tail_population_top3']:.1e} "
              f"ov={min(p['min_step_overlap_even'], p['min_step_overlap_odd']):.4f} "
              f"Ppair={p['P_pair']:+.3f} excD_on/pi={abs(p['exc_on_relative_sector_phase'])/np.pi:.3f} "
              f"t={p['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
