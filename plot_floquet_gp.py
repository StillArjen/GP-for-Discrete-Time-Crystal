"""Figures for the Floquet analysis (dicke_floquet_scan / dicke_late_time_check).

  python plot_floquet_gp.py --coupling results/floquet_coupling_N2.json \
      --detuning results/floquet_eps_N2.json results/floquet_eps_N4.json --atoms results/floquet_atoms_eps0p01.json \
      --late results/late_time_eps_m0p056.json results/late_time_eps0p01.json \
      --kerr results/kerr_parity_half_loop.json --out results/floquet_gp
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#2b2b2b", "#8a8986"
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


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["scanned"], sorted(data["points"], key=lambda p: p[data["scanned"]])


def series(points, key):
    if key == "abs_dphi":
        return np.array([abs(p["relative_sector_phase"]) / np.pi for p in points])
    if key == "pairing":
        out = []
        for p in points:
            ze, zo = complex(*p["Z_even"]), complex(*p["Z_odd"])
            out.append(-2 * np.real(ze * np.conj(zo)) / (abs(ze) ** 2 + abs(zo) ** 2))
        return np.array(out)
    if key == "ZT":
        return np.array([abs(complex(*p["Z_T"])) for p in points])
    return np.array([p[key] for p in points])


def scans_figure(paths, out):
    labels = {"coupling": "Coupling λ₀", "epsilon": "Detuning ε", "atoms": "Atom number N"}
    short = {"coupling": "coupling", "epsilon": "detuning", "atoms": "atom number"}
    fig, axs = plt.subplots(2, len(paths), figsize=(2.5 * len(paths) + 0.6, 4.6),
                            layout="constrained", squeeze=False)
    for col, path in enumerate(paths):
        name, pts = load(path)
        x = series(pts, name)
        top, bottom = axs[0, col], axs[1, col]
        top.axhline(1.0, color=MUTED, lw=0.6, ls="--")
        top.plot(x, series(pts, "abs_dphi"), color=BLUE, lw=2, marker="o", ms=4,
                 label="|Δ|/π  relative sector GP")
        top.plot(x, series(pts, "pairing"), color=ORANGE, lw=2, marker="s", ms=4,
                 label="C  normalized π pairing")
        top.plot(x, series(pts, "ZT"), color=AQUA, lw=2, marker="^", ms=4,
                 label="|Z_T|  loop visibility")
        fixed = "" if name == "atoms" else f" (N = {pts[0]['atoms']})"
        top.set(ylim=(-1.05, 1.05), xlabel=labels[name],
                title=f"{chr(97 + col)}  GP vs {short[name]}{fixed}")
        bottom.semilogy(x, series(pts, "dtc_rate_per_period"), color=INK, lw=2, marker="o", ms=4)
        mu = np.array([complex(*p["mu1"]) for p in pts])
        complex_mu = np.abs(mu.imag) > 1e-8
        if complex_mu.any():
            bottom.scatter(x[complex_mu], series(pts, "dtc_rate_per_period")[complex_mu],
                           s=36, facecolors="white", edgecolors=INK, zorder=3,
                           label="μ₁ complex (no period doubling)")
        bottom.set(xlabel=labels[name], ylabel="−ln|μ₁| per period",
                   title=f"{chr(97 + len(paths) + col)}  Odd Floquet decay rate{fixed}")
        if col == 0:
            top.set_ylabel("dimensionless")
    handles, names = axs[0, 0].get_legend_handles_labels()
    for ax in axs[1]:
        h, n = ax.get_legend_handles_labels()
        if h:
            handles.append(h[0]); names.append(n[0]); break
    fig.legend(handles, names, loc="outside lower center", ncol=len(handles), fontsize=7)
    fig.suptitle("Kinematic GP of the Floquet steady-state micromotion loop | κ = 0.05\n"
                 "Ideal period-doubled pair: |Δ| = π, C = 1, |Z_T| = 0", fontsize=9)
    save(fig, out)


def phasor_figure(path, picks, out):
    name, pts = load(path)
    chosen = [min(pts, key=lambda p: abs(p[name] - v)) for v in picks]
    fig, axs = plt.subplots(1, len(chosen), figsize=(1.9 * len(chosen), 2.2), layout="constrained")
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    for idx, (ax, p) in enumerate(zip(axs, chosen)):
        ax.plot(circle.real, circle.imag, color="#e6e5e1", lw=0.8)
        ax.plot(circle.real * 0.5, circle.imag * 0.5, color="#e6e5e1", lw=0.8, ls=":")
        for key, color, label in (("Z_even", BLUE, "Z_even"), ("Z_odd", ORANGE, "Z_odd"),
                                  ("Z_T", INK, "Z_T = sum")):
            z = complex(*p[key])
            ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0))
            ax.plot([], [], color=color, lw=1.8, label=label)
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal", xticks=[-1, 0, 1], yticks=[-1, 0, 1],
               title=f"{chr(97 + idx)}  λ₀ = {p[name]:g}\n|Δ|/π = {abs(p['relative_sector_phase']) / np.pi:.2f}")
        ax.grid(False)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Parity-sector geometric amplitudes of the micromotion loop (N = 2, ε = 0.01; "
                 "circles: radius 1 and 1/2)\n", fontsize=9)
    save(fig, out)


def late_figure(paths, out):
    rows = [row for path in paths for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    rows.sort(key=lambda r: r["epsilon"])
    fig, axs = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 2.8), layout="constrained", squeeze=False)
    names = (("sigma_x", "Magnetization", BLUE), ("toy_scaled_g_xy", "Transverse-generator GP", ORANGE),
             ("S", "Paired-generator contrast", AQUA))
    for ax, r in zip(axs[0], rows):
        m = np.array(r["m"])
        for key, label, color in names:
            ax.semilogy(m, np.abs(r[key]), color=color, lw=1.2, label=label)
        mu = abs(complex(*r["odd_eigenvalues"][0]))
        anchor = r["fit_start"]
        ref = np.abs(r["sigma_x"][anchor]) * mu ** (m - anchor)
        ax.semilogy(m, ref, color=INK, lw=1, ls="--", label=f"|μ₁|^m, μ₁ = {-mu:.5f}")
        fits = ", ".join(f"{v:.5f}" for v in r["fitted_rates"].values())
        ax.set(xlabel="Drive period m", ylabel="|stroboscopic signal|",
               title=f"ε = {r['epsilon']:g}: Floquet rate {r['floquet_rate']:.5f}\nfitted (m ≥ {anchor}): {fits}")
    axs[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle(f"Late-time decay of parity-odd signals | N = {rows[0]['atoms']}, κ = {rows[0]['kappa']:g}, "
                 f"Fock dimension {rows[0]['cutoff']}", fontsize=9)
    save(fig, out)


def kerr_figure(path, out):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = sorted(data["points"], key=lambda p: p["delta"])
    x = np.array([p["delta"] for p in pts])
    dphi = np.array([abs(p["relative_sector_phase"]) / np.pi for p in pts])
    pairing = np.array([p["pairing"] for p in pts])
    vis = np.array([np.hypot(*p["Z_half"]) for p in pts])
    ideal = np.array([p["ideal_cat_abs_dphi"] / np.pi for p in pts])
    bright = np.array([p["N_B"] > 2 and p["p_B_eff"] > 0.9 for p in pts])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    ax = axs[0]
    ax.axhline(1.0, color=MUTED, lw=0.6, ls="--")
    ax.plot(x, dphi, color=BLUE, lw=2, label="|Δ|/π  relative sector GP")
    ax.plot(x, pairing, color=ORANGE, lw=2, label="C  normalized π pairing")
    ax.plot(x, vis, color=AQUA, lw=2, label="|Z_half|  loop visibility")
    ax.plot(x[bright], ideal[bright], color=INK, lw=1, ls=":",
            label="ideal cats: 1 − 2N_B/sinh 2N_B")
    ax.set(xlabel="Detuning Δ/|U|", ylabel="dimensionless", ylim=(-1.05, 1.05),
           title="a  Parity-resolved half-loop GP (one pump period)")
    ax.legend(fontsize=6.5, loc="lower center")
    ax = axs[1]
    ax.plot(x, [p["nbar"] for p in pts], color=INK, lw=2)
    ax.set(xlabel="Detuning Δ/|U|", ylabel="⟨n⟩", title="b  Steady-state photon number")
    fig.suptitle(f"Two-photon-driven Kerr resonator, η/|U| = {pts[0]['eta']:g}, "
                 f"γ₁ = γ₂ = κ = 0.1|U|, Fock dimension {data['cutoff']}", fontsize=9)
    save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coupling", type=Path)
    p.add_argument("--detuning", type=Path, nargs="+", default=[])
    p.add_argument("--atoms", type=Path)
    p.add_argument("--late", type=Path, nargs="+")
    p.add_argument("--kerr", type=Path)
    p.add_argument("--phasor-picks", type=float, nargs="+", default=[0.6, 0.9, 1.0, 1.4])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    scans = [x for x in [a.coupling, *a.detuning, a.atoms] if x]
    if scans:
        scans_figure(scans, a.out)
    if a.coupling:
        phasor_figure(a.coupling, a.phasor_picks, str(a.out) + "_phasors")
    if a.late:
        late_figure(a.late, str(a.out) + "_late_time")
    if a.kerr:
        kerr_figure(a.kerr, str(a.out) + "_kerr")


if __name__ == "__main__":
    main()
