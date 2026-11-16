"""Parity-resolved half-loop GP of the two-photon-driven Kerr resonator.

In the lab frame the squeezing drive oscillates at 2*omega and the steady state
is rho_lab(t) = R(omega t) rho_ss R(omega t)^dagger with R(phi) = exp(-i phi n).
Over one pump period (phi: 0 -> pi) this is the half number rotation, i.e. the
2*pi squeezing-phase loop of the supplement, Eq. (S14):

    Z_half = sum_j lambda_j pi_j exp(i pi <n>_j) = Z_even + Z_odd .

The bright lobes |alpha>, |-alpha> are exchanged by R(pi): the lab-frame field
oscillates at half the pump frequency (period doubling).  For the ideal
incoherent cat mixture of Eq. (S50) the relative sector phase is
    |arg(Z_even / Z_odd)| = pi [1 - 2 N_B / sinh(2 N_B)],
which approaches pi exponentially in the bright-state photon number, the same
pi pairing found for the micromotion loop of the Dicke DTC.

  python kerr_parity_half_loop.py --eta 3.5 --delta -15 30 91 --out results/kerr_parity_half_loop
"""
import argparse
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import qutip
from joblib import Parallel, delayed

from _phase_eta_exact_core import build_kerr_model
from gp_geometry import geometric_amplitude


def json_path(prefix: Path) -> Path:
    """Append .json without treating dots in the prefix (e.g. eps0.01) as a suffix."""
    return prefix if prefix.suffix == ".json" else Path(str(prefix) + ".json")


def sector_half_loop(rho: np.ndarray):
    dim = rho.shape[0]
    n = np.arange(dim)
    out = {}
    for name, idx, sign in (("even", n[n % 2 == 0], 1.0), ("odd", n[n % 2 == 1], -1.0)):
        block = rho[np.ix_(idx, idx)]
        values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
        number = np.einsum("ik,i,ik->k", vectors.conj(), idx.astype(float), vectors).real
        out[name] = complex(sign * np.sum(values * np.exp(1j * np.pi * number)))
        out["weight_" + name] = float(values.sum())
    return out


def point(delta, eta, cutoff, U=-1.0, rates=0.1):
    H, jumps, n_op = build_kerr_model(delta, eta, cutoff, U, 0.0, 0.0, 0.0, rates, rates, rates)
    rho_q = qutip.steadystate(H, jumps)
    rho = rho_q.full()
    a = qutip.destroy(cutoff)
    nbar = float(qutip.expect(n_op, rho_q))
    g2num = float(qutip.expect(a.dag() ** 2 * a ** 2, rho_q))
    sectors = sector_half_loop(rho)
    z_total = geometric_amplitude(rho, n_op.full(), angle=np.pi)
    ze, zo = sectors["even"], sectors["odd"]
    NB = g2num / nbar if nbar > 0 else 0.0
    return {
        "delta": delta, "eta": eta, "nbar": nbar, "N_B": NB, "p_B_eff": nbar ** 2 / g2num if g2num else 0.0,
        "Z_half": [z_total.real, z_total.imag], "Z_even": [ze.real, ze.imag], "Z_odd": [zo.real, zo.imag],
        "sum_check": abs(z_total - (ze + zo)),
        "weight_even": sectors["weight_even"], "weight_odd": sectors["weight_odd"],
        "relative_sector_phase": float(np.angle(ze * np.conj(zo))),
        "pairing": float(-2 * np.real(ze * np.conj(zo)) / (abs(ze) ** 2 + abs(zo) ** 2)),
        "ideal_cat_abs_dphi": float(np.pi * (1 - 2 * NB / np.sinh(2 * NB))) if NB > 1e-8 else 0.0,
        "top_tail": float(np.real(np.trace(rho[-5:, -5:]))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eta", type=float, default=3.5)
    p.add_argument("--delta", type=float, nargs=3, default=[-15.0, 30.0, 91])
    p.add_argument("--cutoff", type=int, default=50)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    deltas = np.linspace(a.delta[0], a.delta[1], int(a.delta[2]))
    rows = Parallel(n_jobs=a.jobs)(delayed(point)(float(d), a.eta, a.cutoff) for d in deltas)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json_path(a.out).write_text(json.dumps({"cutoff": a.cutoff, "U": -1.0, "rates": 0.1,
                                                      "points": rows}, indent=1), encoding="utf-8")
    print(f"max |Z_half - (Z_even + Z_odd)| = {max(r['sum_check'] for r in rows):.2e}; "
          f"max top tail = {max(r['top_tail'] for r in rows):.1e}")
    for r in rows[::5]:
        print(f"Delta={r['delta']:+6.2f} nbar={r['nbar']:6.2f} N_B={r['N_B']:5.2f} w_odd={r['weight_odd']:.3f} "
              f"|dphi|/pi={abs(r['relative_sector_phase'])/np.pi:.4f} ideal={r['ideal_cat_abs_dphi']/np.pi:.4f} "
              f"C={r['pairing']:+.3f} |Z|={np.hypot(*r['Z_half']):.4f}")


if __name__ == "__main__":
    main()
