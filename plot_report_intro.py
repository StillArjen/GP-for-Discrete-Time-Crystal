"""Introductory figure for the report: what the discrete time crystal looks like.

Exact dynamics of the switched open Dicke model (N = 4, eps = 0.01, kappa = 0.05,
lambda0 = 1, Fock dimension 32).  Panels: the drive protocol, the stroboscopic
magnetization, the cavity Husimi function at the end of two consecutive coupled
segments (the lobe changes side every period), and the same for the parity-
symmetric Floquet steady state (both lobes present).

  python plot_report_intro.py --out results/report_dtc_intro
"""
import argparse
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.sparse.linalg import expm_multiply

from dicke_floquet import build_model, floquet_spectrum

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b2b", "#8a8986"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "legend.frameon": False})


def cavity_q(model, vec, grid):
    rho = qt.Qobj(vec.reshape((model.dim, model.dim), order="F"),
                  dims=[[model.cutoff, model.atoms + 1], [model.cutoff, model.atoms + 1]])
    rho = 0.5 * (rho + rho.dag())
    return qt.qfunc(rho.ptrace(0), grid, grid, g=2)   # axes are Re(alpha), Im(alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=4)
    p.add_argument("--cutoff", type=int, default=32)
    p.add_argument("--periods", type=int, default=60)
    p.add_argument("--snapshot", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    model = build_model(atoms=a.atoms, cutoff=a.cutoff, epsilon=0.01, kappa=0.05, coupling=1.0)
    j = a.atoms / 2
    psi0 = qt.tensor(qt.basis(a.cutoff, 0), qt.spin_coherent(j, np.pi / 2, 0.0))
    vec = qt.ket2dm(psi0).full().reshape(-1, order="F")
    jx = model.ops["jx"]
    sigma_x, after_on = [], {}
    for m in range(a.periods + 1):
        rho = vec.reshape((model.dim, model.dim), order="F")
        sigma_x.append(float(np.real(np.einsum("ij,ji->", jx, rho))) / j)
        on = expm_multiply(model.t_on * model.L_on, vec)
        if m in (a.snapshot, a.snapshot + 1):
            after_on[m] = on
        vec = expm_multiply(model.t_off * model.L_off, on)

    _, even_vecs = floquet_spectrum(model, "even", k=2)
    rho_f = even_vecs[:, 0].reshape((model.dim, model.dim), order="F")
    rho_f = rho_f / np.trace(rho_f)
    floquet_on = expm_multiply(model.t_on * model.L_on, rho_f.reshape(-1, order="F"))

    grid = np.linspace(-3.6, 3.6, 161)
    fig = plt.figure(figsize=(7.4, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax0 = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 3 * 2 * np.pi, 1200)
    ax0.plot(t / (2 * np.pi), np.where((t % (2 * np.pi)) < np.pi, 1.0, 0.0), color=INK, lw=1.8)
    for k in range(3):
        ax0.axvspan(k, k + 0.5, color=BLUE, alpha=0.08, lw=0)
    ax0.text(0.25, 1.12, "on", ha="center", fontsize=7, color=INK)
    ax0.text(0.75, 0.12, "off\n≈ Π", ha="center", fontsize=7, color=INK)
    ax0.set(xlabel="t / T", ylabel="λ(t) / λ₀", ylim=(-0.1, 1.35), yticks=[0, 1],
            title="a  Drive: coupling switched on and off")

    ax1 = fig.add_subplot(gs[0, 1:])
    m = np.arange(a.periods + 1)
    ax1.plot(m, sigma_x, color=BLUE, lw=1.2, marker="o", ms=2.5)
    ax1.axhline(0, color=MUTED, lw=0.6)
    for k, col in ((a.snapshot, ORANGE), (a.snapshot + 1, ORANGE)):
        ax1.axvline(k, color=col, lw=0.8, ls=":")
    ax1.set(xlabel="drive period m", ylabel="⟨σ_x(mT)⟩",
            title=f"b  Stroboscopic magnetization: sign flips every period (N = {a.atoms})")

    panels = [(after_on[a.snapshot], f"c  cavity Q, period {a.snapshot}"),
              (after_on[a.snapshot + 1], f"d  cavity Q, period {a.snapshot + 1}"),
              (floquet_on, "e  Floquet steady state ρ_F")]
    for col, (state, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        q = cavity_q(model, state, grid)
        ax.contourf(grid, grid, q, levels=14, cmap="Blues")
        ax.set(xlabel="Re α", ylabel="Im α" if col == 0 else "", title=title, aspect="equal")
        ax.axhline(0, color=MUTED, lw=0.4)
        ax.axvline(0, color=MUTED, lw=0.4)
    fig.suptitle("The discrete time crystal of the switched open Dicke model "
                 "(exact, ε = 0.01, κ = 0.05, λ₀ = 1; Q shown after the coupled segment)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    print("sigma_x first values:", np.round(sigma_x[:6], 3))


if __name__ == "__main__":
    main()
