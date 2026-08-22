from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"

FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL HD / IEEE-WCL SETTINGS
# ============================================================
DPI = 600

# Compact IEEE/WCL-friendly physical size for single-panel figures
FIGSIZE = (3.35, 2.35)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,

    "axes.labelsize": 10,
    "axes.titlesize": 10,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "legend.fontsize": 8,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,

    "figure.dpi": DPI,
    "savefig.dpi": DPI,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,

    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# SAVE PNG + VECTOR PDF
# ============================================================
def save_figure(fig, name):

    png_path = FIG / f"{name}.png"
    pdf_path = FIG / f"{name}.pdf"

    # --------------------------------------------------------
    # High-resolution raster image
    # --------------------------------------------------------
    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    # --------------------------------------------------------
    # Vector PDF for paper submission
    # --------------------------------------------------------
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print("Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

    plt.close(fig)


# ============================================================
# FIGURE 1
#
# Mean transmit energy vs MSE threshold epsilon
#
# Dataset:
# epsilon_sweep_500.csv
#
# Main FAA-vs-FPA energy comparison.
# ============================================================
def plot_fig1():

    path = OUT / "epsilon_sweep_500.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    grouped = (
        df.groupby("epsilon")
        .agg(
            faa_energy_W=("faa_energy_W", "mean"),
            fpa_energy_W=("fpa_energy_W", "mean"),
        )
        .reset_index()
        .sort_values("epsilon")
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # FAA + PC
    # --------------------------------------------------------
    ax.plot(
        grouped["epsilon"],
        grouped["faa_energy_W"],
        marker="o",
        markersize=5,
        linewidth=1.3,
        label="FAA+PC",
    )

    # --------------------------------------------------------
    # FPA + PC
    # --------------------------------------------------------
    ax.plot(
        grouped["epsilon"],
        grouped["fpa_energy_W"],
        marker="s",
        markersize=5,
        linewidth=1.3,
        label="FPA+PC",
    )

    ax.set_xlabel(
        r"MSE Threshold $\epsilon$",
        fontsize=10,
    )

    ax.set_ylabel(
        "Mean Transmit Energy (W)",
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
    )

    save_figure(
        fig,
        "fig1_epsilon_energy",
    )


# ============================================================
# FIGURE 2
#
# System-size robustness
#
# K = {4, 8, 12}
# N = {4, 6, 8}
# epsilon = 0.08
#
# Dataset:
# robustness_experiment_eps008.csv
# ============================================================
def plot_fig2():

    path = OUT / "robustness_experiment_eps008.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["K"] = df["K"].astype(int)
    df["N"] = df["N"].astype(int)

    grouped = (
        df.groupby(["K", "N"])
        .agg(
            faa_energy_W=("faa_energy_W", "mean")
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # One curve per N
    # --------------------------------------------------------
    for N in sorted(grouped["N"].unique()):

        sub = (
            grouped[grouped["N"] == N]
            .sort_values("K")
        )

        ax.plot(
            sub["K"],
            sub["faa_energy_W"],
            marker="o",
            markersize=5,
            linewidth=1.3,
            label=fr"$N={N}$",
        )

    ax.set_xlabel(
        "Number of Devices $K$",
        fontsize=10,
    )

    ax.set_ylabel(
        "Mean FAA+PC Energy (W)",
        fontsize=10,
    )

    # Energy spans multiple orders of magnitude.
    ax.set_yscale("log")

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
    )

    save_figure(
        fig,
        "fig2_system_size",
    )


# ============================================================
# FIGURE 3
#
# Mean BCD iterations vs epsilon
#
# Dataset:
# epsilon_sweep_500.csv
# ============================================================
def plot_fig3():

    path = OUT / "epsilon_sweep_500.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    grouped = (
        df.groupby("epsilon")
        .agg(
            faa_iterations=("faa_iterations", "mean"),
            fpa_iterations=("fpa_iterations", "mean"),
        )
        .reset_index()
        .sort_values("epsilon")
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # FAA + PC
    # --------------------------------------------------------
    ax.plot(
        grouped["epsilon"],
        grouped["faa_iterations"],
        marker="o",
        markersize=5,
        linewidth=1.3,
        label="FAA+PC",
    )

    # --------------------------------------------------------
    # FPA + PC
    # --------------------------------------------------------
    ax.plot(
        grouped["epsilon"],
        grouped["fpa_iterations"],
        marker="s",
        markersize=5,
        linewidth=1.3,
        label="FPA+PC",
    )

    ax.set_xlabel(
        r"MSE Threshold $\epsilon$",
        fontsize=10,
    )

    ax.set_ylabel(
        "Mean BCD Iterations",
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
    )

    save_figure(
        fig,
        "fig3_bcd_iterations",
    )


# ============================================================
# FIGURE 4
#
# Four-scheme comparison
#
# Dataset:
# four_scheme_epsilon_sweep_500.csv
#
# Panel (a):
# Mean transmit energy vs epsilon
#
# Panel (b):
# Mean MSE vs epsilon
#
# Schemes:
#   FAA+PC
#   FPA+PC
#   FAA+MaxP
#   FPA+MaxP
#
# Panel (b) reports the measured MSE of the MaxP baselines.
# The PC schemes enforce the prescribed MSE threshold through
# the power-control constraint.
# ============================================================
def plot_fig4():

    path = OUT / "four_scheme_epsilon_sweep_500.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    grouped = (
        df.groupby("epsilon")
        .agg(
            faa_pc_energy_W=(
                "faa_pc_energy_W",
                "mean"
            ),

            fpa_pc_energy_W=(
                "fpa_pc_energy_W",
                "mean"
            ),

            faa_maxpower_energy_W=(
                "faa_maxpower_energy_W",
                "mean"
            ),

            fpa_maxpower_energy_W=(
                "fpa_maxpower_energy_W",
                "mean"
            ),

            faa_maxpower_mse=(
                "faa_maxpower_mse",
                "mean"
            ),

            fpa_maxpower_mse=(
                "fpa_maxpower_mse",
                "mean"
            ),
        )
        .reset_index()
        .sort_values("epsilon")
    )

    # ========================================================
    # TWO-PANEL FIGURE
    #
    # Larger physical size than the previous version so that
    # labels and legends remain readable.
    # ========================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
    )

    ax1, ax2 = axes

    # ========================================================
    # PANEL (a): ENERGY
    # ========================================================

    ax1.plot(
        grouped["epsilon"],
        grouped["faa_pc_energy_W"],
        marker="o",
        markersize=5.5,
        linewidth=1.5,
        label="FAA+PC",
    )

    ax1.plot(
        grouped["epsilon"],
        grouped["fpa_pc_energy_W"],
        marker="s",
        markersize=5.5,
        linewidth=1.5,
        label="FPA+PC",
    )

    ax1.plot(
        grouped["epsilon"],
        grouped["faa_maxpower_energy_W"],
        marker="^",
        markersize=5.5,
        linewidth=1.5,
        linestyle="--",
        label="FAA+MaxP",
    )

    ax1.plot(
        grouped["epsilon"],
        grouped["fpa_maxpower_energy_W"],
        marker="v",
        markersize=5.5,
        linewidth=1.5,
        linestyle="--",
        label="FPA+MaxP",
    )

    ax1.set_xlabel(
        r"MSE Threshold $\epsilon$",
        fontsize=10,
    )

    ax1.set_ylabel(
        "Mean Transmit Energy (W)",
        fontsize=10,
    )

    ax1.tick_params(
        axis="both",
        labelsize=9,
    )

    # Energy spans several orders of magnitude.
    ax1.set_yscale("log")

    ax1.grid(
        True,
        alpha=0.25,
    )

    ax1.text(
        0.02,
        0.96,
        "(a)",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="normal",
        ha="left",
        va="top",
    )

    ax1.legend(
        fontsize=8,
        frameon=True,
        loc="lower left",
    )

    # ========================================================
    # PANEL (b): MSE
    #
    # MaxP baselines have measured MSE values.
    #
    # The PC schemes enforce the prescribed MSE threshold.
    # Therefore the target epsilon is plotted as the reference
    # rather than inventing an independently measured PC-MSE
    # quantity that does not exist in the CSV.
    # ========================================================

    ax2.plot(
        grouped["epsilon"],
        grouped["faa_maxpower_mse"],
        marker="o",
        markersize=5.5,
        linewidth=1.5,
        label="FAA+MaxP",
    )

    ax2.plot(
        grouped["epsilon"],
        grouped["fpa_maxpower_mse"],
        marker="s",
        markersize=5.5,
        linewidth=1.5,
        label="FPA+MaxP",
    )

    ax2.plot(
        grouped["epsilon"],
        grouped["epsilon"],
        linestyle="--",
        linewidth=1.5,
        label=r"Target $\epsilon$",
    )

    ax2.set_xlabel(
        r"MSE Threshold $\epsilon$",
        fontsize=10,
    )

    ax2.set_ylabel(
        "Mean MSE",
        fontsize=10,
    )

    ax2.tick_params(
        axis="both",
        labelsize=9,
    )

    ax2.grid(
        True,
        alpha=0.25,
    )

    ax2.text(
        0.02,
        0.96,
        "(b)",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="normal",
        ha="left",
        va="top",
    )

    ax2.legend(
        fontsize=8,
        frameon=True,
        loc="upper left",
    )

    # ========================================================
    # FINAL PANEL SPACING
    # ========================================================
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.97,
        wspace=0.28,
    )

    save_figure(
        fig,
        "fig4_four_scheme_comparison",
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("WCL HD FIGURE GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # FIGURE 1
    # --------------------------------------------------------
    print(
        "\n[1/4] Generating epsilon-energy figure..."
    )

    plot_fig1()

    # --------------------------------------------------------
    # FIGURE 2
    # --------------------------------------------------------
    print(
        "\n[2/4] Generating system-size figure..."
    )

    plot_fig2()

    # --------------------------------------------------------
    # FIGURE 3
    # --------------------------------------------------------
    print(
        "\n[3/4] Generating BCD iteration figure..."
    )

    plot_fig3()

    # --------------------------------------------------------
    # FIGURE 4
    # --------------------------------------------------------
    print(
        "\n[4/4] Generating four-scheme comparison..."
    )

    plot_fig4()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Output directory: {FIG}")
    print("PNG: 600 DPI")
    print("PDF: vector")