"""
figures.py
==========
Publication-quality figure generation for all 6 paper figures.

All figures match the IEEE WCL submission exactly:
  - IEEEtran column width (3.5 in)
  - Serif fonts, 8 pt base, 7 pt ticks, 6.5 pt legend
  - 300 DPI, tight layout, minimal padding

Usage:
    from src.figures import setup_rcparams, plot_fig1, ...
    setup_rcparams()
    plot_fig1(data, outdir="figures/")
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────

FIG_W    = 3.5    # IEEE single-column width (inches)
FIG_H    = 1.90   # standard figure height (inches)
FIG_H_DY = 2.00   # slightly taller for dual-axis (Fig. 5)


def setup_rcparams():
    """Apply IEEE-style matplotlib settings globally."""
    plt.rcParams.update({
        "font.family"        : "serif",
        "font.size"          : 8,
        "axes.labelsize"     : 8,
        "axes.titlesize"     : 8,
        "xtick.labelsize"    : 7,
        "ytick.labelsize"    : 7,
        "legend.fontsize"    : 6.5,
        "legend.handlelength": 1.5,
        "lines.linewidth"    : 1.3,
        "lines.markersize"   : 4,
        "figure.dpi"         : 300,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.02,
        "axes.grid"          : True,
        "grid.linestyle"     : ":",
        "grid.linewidth"     : 0.5,
        "grid.alpha"         : 0.6,
    })


def _dBm(w):
    """Watts → dBm."""
    return 10.0 * np.log10(np.asarray(w) * 1000.0)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1 – TX energy vs. ε
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig1(eps_vals, results, outdir="figures"):
    """
    Parameters
    ----------
    eps_vals : array of ε values
    results  : dict scheme → list[MCResult]
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    schemes = [
        ("proposed",   "C0", "o", "-",  "Proposed (FAA+PC)"),
        ("faa_maxpow", "C1", "s", "--", "FAA Max Power [5]"),
        ("fpa_pc",     "C2", "^", "-.", "FPA+Power Ctrl"),
        ("fpa_maxpow", "C3", "D", ":",  "FPA Max Power"),
    ]

    for name, color, marker, ls, label in schemes:
        means = [r.mean for r in results[name]]
        ses   = [r.se   for r in results[name]]
        if name == "proposed":
            # Error bars ±1 SE
            ax.errorbar(eps_vals, _dBm(means),
                        yerr=[_dBm(np.array(means)) - _dBm(np.array(means) - np.array(ses)),
                               _dBm(np.array(means) + np.array(ses)) - _dBm(means)],
                        label=label, marker=marker, color=color, linestyle=ls)
        else:
            ax.plot(eps_vals, _dBm(means),
                    label=label, marker=marker, color=color, linestyle=ls)

    ax.set_xlabel(r"MSE Threshold $\varepsilon$")
    ax.set_ylabel(r"Total TX Energy [dBm$\cdot$symbol]")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return _save(fig, outdir, "fig1.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2 – ESR vs. N
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig2(N_vals, esr_data, outdir="figures"):
    """
    Parameters
    ----------
    N_vals   : list of N values on x-axis
    esr_data : dict K → list of ESR (%) values
    """
    fig, ax  = plt.subplots(figsize=(FIG_W, FIG_H))
    markers  = ["o", "s", "D"]

    for i, (K, esr_list) in enumerate(sorted(esr_data.items())):
        ax.plot(N_vals, esr_list, marker=markers[i % 3], label=f"$K={K}$")

    ax.set_xlabel(r"Number of FA Ports $N$")
    ax.set_ylabel(r"Energy Saving Ratio ESR (%)")
    ax.legend()
    ax.set_xlim([min(N_vals) - 0.5, max(N_vals) + 0.5])
    ax.set_ylim([-2, 70])
    plt.tight_layout()
    return _save(fig, outdir, "fig2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 3 – BCD Convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3(histories, outdir="figures"):
    """
    Parameters
    ----------
    histories : list of lists (energy per iteration for each run)
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for run_idx, hist in enumerate(histories):
        iters = list(range(1, len(hist) + 1))
        ax.plot(iters, _dBm(hist), label=f"Run {run_idx + 1}")

    ax.set_xlabel("BCD Iteration Index")
    ax.set_ylabel(r"Total TX Energy [dBm$\cdot$symbol]")
    ax.legend(ncol=2)
    max_iters = max(len(h) for h in histories)
    ax.set_xlim([1, max(max_iters, 16)])
    plt.tight_layout()
    return _save(fig, outdir, "fig3.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4 – Scaling Law
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig4(K_vals, mc_means, alpha, beta, A_fit, outdir="figures"):
    """
    Parameters
    ----------
    K_vals   : list of K values (x-axis markers)
    mc_means : dict N → list of mean E* values
    alpha, beta, A_fit : fitted scaling law parameters
    """
    K_fine   = np.linspace(min(K_vals), max(K_vals), 200)
    markers  = ["o", "s", "D"]
    colors   = ["C0", "C1", "C2"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, (N, means) in enumerate(sorted(mc_means.items())):
        e_fit = A_fit * K_fine ** alpha * N ** beta
        ax.plot(K_fine, _dBm(e_fit),
                color=colors[i], alpha=0.40, linewidth=0.9)
        ax.plot(K_vals, _dBm(means),
                marker=markers[i], color=colors[i],
                linestyle="None", label=f"$N={N}$")

    ax.set_xlabel(r"Number of Devices $K$")
    ax.set_ylabel(r"Min. TX Energy $E^*$ [dBm$\cdot$sym]")
    ax.legend()
    plt.tight_layout()
    return _save(fig, outdir, "fig4.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 5 – Straggler Robustness
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig5(eps_vals, E_strag, E_unif, E_base, strag_share, outdir="figures"):
    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H_DY))

    ax1.plot(eps_vals, _dBm(E_strag), "C0-o",  label="Proposed (straggler)")
    ax1.plot(eps_vals, _dBm(E_unif),  "C0--s", label="Proposed (uniform)")
    ax1.plot(eps_vals, _dBm(E_base),  "C3:D",  label="FPA–MaxPow")
    ax1.set_xlabel(r"MSE Threshold $\varepsilon$")
    ax1.set_ylabel(r"TX Energy [dBm$\cdot$sym]")
    ax1.legend(loc="upper right", fontsize=6)

    ax2 = ax1.twinx()
    ax2.plot(eps_vals, strag_share, "C2-.", linewidth=1)
    ax2.set_ylabel(r"Straggler $p_k$ (%)", color="C2")
    ax2.tick_params(axis="y", colors="C2")
    ax2.set_ylim([0, 45])

    plt.tight_layout()
    return _save(fig, outdir, "fig5.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 6 – APV Candidate Count Ablation
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig6(C_vals, E_means, outdir="figures"):
    fig, ax = plt.subplots(figsize=(FIG_W, 1.80))

    ax.semilogx(C_vals, _dBm(E_means), "C0-o")
    ax.axvline(x=40, color="C3", linestyle="--", linewidth=1,
               label="$C=40$ plateau")
    ax.set_xlabel(r"APV Candidate Count $C$")
    ax.set_ylabel(r"Converged $E^*$ [dBm$\cdot$sym]")
    ax.set_xticks(C_vals)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend()

    plt.tight_layout()
    return _save(fig, outdir, "fig6.pdf")
