"""Figures for extending the GP diagnostic to the open Dicke / DTC setup.

  python plot_extension.py --static results/static_superradiant.json --weak results/static_weak_coupling.json \
      --control results/dtc_control_toff1.json results/dtc_control_toff2.json \
      --transient results/floquet_atoms_eps0p01.json results/floquet_atoms_eps0p09.json \
      --detuning results/floquet_eps_N2.json results/floquet_eps_N4.json --out results/extension
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#2b2b2b", "#8a8986"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.5, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "legend.frameon": False})


def save(fig, prefix):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_points(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["points"]


def lambda_c(eps, kappa):
    w, wc = 1 + eps, 1 - eps
    return 0.5 * np.sqrt(w / wc * (wc ** 2 + kappa ** 2 / 4))


def static_figure(path, out, weak_path=None):
    pts = load_points(path)
    sizes = sorted({p["atoms"] for p in pts})
    lc = lambda_c(pts[0]["epsilon"], pts[0]["kappa"])
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4), layout="constrained")
    axs = axs.ravel()
    for color, N in zip(SERIES, sizes):
        rows = sorted((p for p in pts if p["atoms"] == N), key=lambda p: p["coupling"])
        g = np.array([p["coupling"] for p in rows])
        kw = dict(color=color, lw=2, marker="o", ms=3, label=f"N = {N}")
        axs[0].plot(g, [p["nbar_per_atom"] for p in rows], **kw)
        axs[1].plot(g, [abs(p["relative_sector_phase"]) / np.pi for p in rows], **kw)
        rate = np.array([-p["odd_slowest"][0] if "odd_slowest" in p else np.nan for p in rows])
        axs[2].semilogy(g, rate, **kw)
        axs[3].plot(g, [np.hypot(*p["Z_full"]) for p in rows], **kw)
        axs[5].plot(g, [np.hypot(*p["Z_half"]) for p in rows], **kw)
    if weak_path:
        weak = load_points(weak_path)
        for color, N in zip(SERIES, sizes):
            rows = sorted((p for p in weak if p["atoms"] == N), key=lambda p: p["coupling"])
            if not rows:
                continue
            g = np.array([p["coupling"] for p in rows])
            axs[4].plot(g, [p["exact_gamma"] / np.pi for p in rows], color=color, lw=0, marker="o", ms=4,
                        label=f"exact, N = {N}")
            axs[4].plot(g, [p["perturbative_gamma"] / np.pi for p in rows], color=color, lw=1.5, ls="--",
                        label=f"weak-coupling formula, N = {N}")
        axs[4].legend(fontsize=6, loc="upper left")
    specs = [("a  Photon number per atom", "⟨n⟩/N"),
             ("b  Half loop: relative sector phase", "|Δ_B| / π"),
             ("c  Branch-mixing rate (odd Liouvillian gap)", "rate"),
             ("d  Full loop: visibility |Z_full|", "|Z_full|"),
             ("e  Full loop, normal phase: Γ_full", "Γ_full / π"),
             ("f  Half loop: visibility |Z_half|", "|Z_half|")]
    for ax, (t, yl) in zip(axs, specs):
        ax.axvline(lc, color=MUTED, lw=0.8, ls="--")
        ax.set(xlabel="Coupling λ", ylabel=yl, title=t)
    axs[1].axhline(1.0, color=MUTED, lw=0.6, ls=":")
    handles, names = axs[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=len(sizes), fontsize=7)
    fig.suptitle("Undriven open Dicke model: GP of the counter-rotating-phase loop (generator N_exc = n + J_z + N/2)\n"
                 f"κ = {pts[0]['kappa']:g}, ε = {pts[0]['epsilon']:g}; dashed line: mean-field λ_c = {lc:.3f}",
                 fontsize=9)
    save(fig, out)


def control_figure(paths, out):
    data = [sorted(load_points(p), key=lambda q: q["coupling"]) for p in paths]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.8), layout="constrained")
    labels = {1.0: "T_off = π (branches swapped each period)",
              2.0: "T_off = 2π (branches kept)"}
    for color, rows in zip(SERIES, data):
        t_off = rows[0]["t_off_over_pi"]
        g = np.array([p["coupling"] for p in rows])
        kw = dict(color=color, lw=2, marker="o", ms=4, label=labels.get(t_off, f"T_off = {t_off:g}π"))
        axs[0].plot(g, [p["mu1"][0] for p in rows], **kw)
        cplx_mu = np.array([abs(p["mu1"][1]) > 1e-8 for p in rows])
        axs[0].scatter(g[cplx_mu], np.array([p["mu1"][0] for p in rows])[cplx_mu], s=40,
                       facecolors="white", edgecolors=color, zorder=3)
        axs[1].plot(g, [abs(p["relative_sector_phase"]) / np.pi for p in rows], **kw)
        axs[2].plot(g, [abs(p["exc_on_relative_sector_phase"]) / np.pi for p in rows], **kw)
    axs[0].axhline(0, color=MUTED, lw=0.6)
    axs[0].set(xlabel="Coupling λ₀", ylabel="Re μ₁", ylim=(-1.05, 1.05),
               title="a  Odd Floquet eigenvalue μ₁ (open: complex)")
    axs[1].set(xlabel="Coupling λ₀", ylabel="|Δ_A| / π", ylim=(-0.03, 1.05),
               title="b  Micromotion loop (time translation)")
    axs[2].set(xlabel="Coupling λ₀", ylabel="|Δ_B| / π", ylim=(-0.03, 1.05),
               title="c  Counter-rotating-phase loop (Z₂ branches)")
    for ax in axs[1:]:
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    handles, names = axs[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2, fontsize=7)
    r0 = data[0][0]
    fig.suptitle(f"Which GP sees what: N = {r0['atoms']}, ε = {r0['epsilon']:g}, κ = {r0['kappa']:g}",
                 fontsize=9)
    save(fig, out)


def sector_coherence(p):
    return (np.hypot(*p["Z_even"]) + np.hypot(*p["Z_odd"])) / (p["weight_even"] + p["weight_odd"])


def transient_figure(paths, out, detuning_paths=()):
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 2.9), layout="constrained")
    for color, path in zip(SERIES, paths):
        rows = sorted(load_points(path), key=lambda q: q["atoms"])
        N = np.array([p["atoms"] for p in rows])
        eps = rows[0]["epsilon"]
        kw = dict(color=color, lw=2, marker="o", ms=4, label=f"N scan, ε = {eps:g}")
        axs[0].semilogy(N, [p["dtc_rate_per_period"] for p in rows], **kw)
        axs[1].plot(N, [abs(p["relative_sector_phase"]) / np.pi for p in rows], **kw)
        axs[2].plot(N, [sector_coherence(p) for p in rows], **kw)
    for color, path in zip(SERIES[2:], detuning_paths):
        rows = [p for p in load_points(path) if abs(p["mu1"][1]) < 1e-8 and p["mu1"][0] < 0]
        axs[3].semilogx([p["dtc_rate_per_period"] for p in rows], [sector_coherence(p) for p in rows],
                        color=color, lw=0, marker="o", ms=4,
                        label=f"detuning scan, N = {rows[0]['atoms']}")
    axs[0].set(xlabel="Atom number N", ylabel="−ln|μ₁| per period", title="a  DTC decay rate (exact)")
    axs[1].set(xlabel="Atom number N", ylabel="|Δ_A| / π", ylim=(-0.03, 1.05),
               title="b  Micromotion relative phase")
    axs[2].set(xlabel="Atom number N", ylabel="S", ylim=(0, 1.05),
               title="c  Sector coherence S")
    axs[3].set(xlabel="−ln|μ₁| per period", ylabel="S", ylim=(0, 1.05),
               title="d  S vs decay rate (real μ₁ < 0)")
    handles, names = [], []
    for ax in (axs[0], axs[3]):
        h, n = ax.get_legend_handles_labels()
        handles += h; names += n
    fig.legend(handles, names, loc="outside lower center", ncol=len(handles), fontsize=7)
    fig.suptitle("Finite-N exact Floquet results (κ = 0.05, λ₀ = 1); "
                 "S = (|Z_even| + |Z_odd|)/(w_even + w_odd) of the micromotion loop", fontsize=9)
    save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--static", type=Path)
    p.add_argument("--weak", type=Path)
    p.add_argument("--control", type=Path, nargs="+")
    p.add_argument("--transient", type=Path, nargs="+")
    p.add_argument("--detuning", type=Path, nargs="+", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.static:
        static_figure(a.static, str(a.out) + "_superradiant", a.weak)
    if a.control:
        control_figure(a.control, str(a.out) + "_dtc_control")
    if a.transient:
        transient_figure(a.transient, str(a.out) + "_finite_N", a.detuning)


if __name__ == "__main__":
    main()
