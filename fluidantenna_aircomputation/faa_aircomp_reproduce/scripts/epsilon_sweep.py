"""
epsilon_sweep.py
================

Energy-vs-MSE-target experiment for FAA-AirComp.

Experiment:
    K = 8
    N = 6
    epsilon = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    500 paired physical-channel realizations per epsilon.

For every realization:
    1. Generate ONE physical channel.
    2. Reuse exactly the same physical channel for FAA and FPA.
    3. Reuse the same g and phi within the FAA APV search.
    4. FAA uses APV + power control.
    5. FPA uses fixed positions + power control.
    6. Record energy, iterations and convergence.

Important:
    The same 500 channel seeds are reused at every epsilon.
    Therefore the epsilon sweep is paired across epsilon values.
"""

from __future__ import annotations

# =============================================================================
# PROJECT-ROOT IMPORT FIX
# =============================================================================
# This script lives in:
#     <project_root>/scripts/epsilon_sweep.py
#
# When executed as:
#     python scripts/epsilon_sweep.py
#
# Python otherwise searches from <project_root>/scripts, which can make
# "from src..." fail. Add the project root explicitly and deterministically.
# =============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if not PROJECT_ROOT.exists():
    raise RuntimeError(
        f"Project root does not exist: {PROJECT_ROOT}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# STANDARD LIBRARY / THIRD-PARTY IMPORTS
# =============================================================================

import csv
import math
import os
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.stats import t, wilcoxon, ttest_rel

from src.config import SystemConfig
from src.channel import make_channel
from src.bcd import run_bcd


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

K = 8
N = 6

EPSILONS = [
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.10,
    0.12,
]

NUM_REALIZATIONS = 500

BASE_SEED = 20260820

# Always write outputs relative to the project root, not the current shell
# directory. This prevents accidental output/<file> creation elsewhere.
OUTPUT_DIR = PROJECT_ROOT / "output"

CSV_PATH = OUTPUT_DIR / "epsilon_sweep_500.csv"
SUMMARY_PATH = OUTPUT_DIR / "epsilon_sweep_summary.txt"


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

MONO_TOL = 1e-9
WIN_TOL = 1e-12
LOG_FLOOR_W = 1e-15


# =============================================================================
# HELPERS
# =============================================================================

def is_monotone_nonincreasing(history):
    """Check E_1 >= E_2 >= ... with numerical tolerance."""
    if len(history) <= 1:
        return True

    return all(
        history[i] >= history[i + 1] - MONO_TOL
        for i in range(len(history) - 1)
    )


def confidence_interval_95(values):
    """95% CI for the sample mean using Student's t distribution."""
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(n)

    critical = float(t.ppf(0.975, df=n - 1))
    margin = critical * se

    return mean - margin, mean + margin


def safe_wilcoxon(x, y):
    """Paired two-sided Wilcoxon signed-rank test."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    differences = y - x

    if np.allclose(
        differences,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        return 0.0, 1.0

    result = wilcoxon(
        x,
        y,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def run_silent(func, *args, **kwargs):
    """
    Run the existing BCD implementation without flooding the terminal.

    This does not alter the BCD algorithm. It only redirects diagnostic output.
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            return func(*args, **kwargs)


def energy_to_dbm(energy_w):
    """Convert power in watts to dBm with a numerical floor."""
    return 10.0 * np.log10(
        max(float(energy_w), LOG_FLOOR_W) * 1000.0
    )


def validate_history(history, name):
    """Fail loudly if the BCD routine returns an invalid history."""
    if history is None:
        raise RuntimeError(f"{name} returned history=None.")

    history = list(history)

    if len(history) == 0:
        raise RuntimeError(f"{name} returned an empty convergence history.")

    history = np.asarray(history, dtype=float)

    if not np.all(np.isfinite(history)):
        raise RuntimeError(
            f"{name} convergence history contains non-finite values."
        )

    return history.tolist()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = SystemConfig()

    print("=" * 70)
    print("FAA-AirComp EPSILON SWEEP")
    print("=" * 70)

    print()
    print("PROJECT")
    print("-" * 70)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Script             : {Path(__file__).resolve()}")

    print()
    print("CONFIG")
    print("-" * 70)
    print(f"K                  : {K}")
    print(f"N                  : {N}")
    print(f"Realizations       : {NUM_REALIZATIONS}")
    print(f"Epsilon values     : {EPSILONS}")
    print(f"Base seed          : {BASE_SEED}")
    print(f"Pmax               : {cfg.Pmax} W")
    print(f"sigma2             : {cfg.sigma2}")
    print(f"rho                : {cfg.rho}")
    print(f"bcd_tol            : {cfg.bcd_tol}")
    print(f"bcd_max_iter       : {cfg.bcd_max_iter}")
    print(f"C_apv              : {cfg.C_apv}")

    print()
    print("OUTPUT")
    print("-" * 70)
    print(f"CSV                : {CSV_PATH}")
    print(f"Summary            : {SUMMARY_PATH}")

    rows = []

    total_start = time.perf_counter()

    successful = 0
    failed = 0

    # =========================================================================
    # EPSILON LOOP
    # =========================================================================

    for eps_index, eps in enumerate(EPSILONS, start=1):

        print()
        print("=" * 70)
        print(
            f"EPSILON {eps_index}/{len(EPSILONS)}"
            f"  ->  epsilon = {eps:.2f}"
        )
        print("=" * 70)

        eps_start = time.perf_counter()

        for realization in range(NUM_REALIZATIONS):

            # Same channel seed for the same realization across every epsilon.
            channel_seed = BASE_SEED + realization

            channel_rng = np.random.default_rng(channel_seed)

            # Separate deterministic RNG for FAA APV candidate generation.
            faa_rng = np.random.default_rng(
                BASE_SEED + 1_000_000 + realization
            )

            # FPA has no APV. Keep a deterministic independent RNG in case
            # the underlying BCD implementation uses its RNG internally.
            fpa_rng = np.random.default_rng(
                BASE_SEED + 2_000_000 + realization
            )

            try:
                # =============================================================
                # ONE PHYSICAL CHANNEL
                # =============================================================

                (
                    H,
                    dk,
                    bk,
                    pos,
                    g,
                    phi,
                ) = make_channel(
                    K=K,
                    N=N,
                    cfg=cfg,
                    rng=channel_rng,
                )

                # =============================================================
                # PHYSICAL CHANNEL VALIDATION
                # =============================================================

                H = np.asarray(H)
                bk = np.asarray(bk)
                g = np.asarray(g)
                phi = np.asarray(phi)

                if H.shape != (N, K):
                    raise RuntimeError(
                        f"Expected H shape {(N, K)}, got {H.shape}."
                    )

                if bk.shape != (K,):
                    raise RuntimeError(
                        f"Expected bk shape {(K,)}, got {bk.shape}."
                    )

                if g.shape[0] != K:
                    raise RuntimeError(
                        f"Expected g length {K}, got {g.shape}."
                    )

                if phi.shape[0] != K:
                    raise RuntimeError(
                        f"Expected phi length {K}, got {phi.shape}."
                    )

                if not (
                    np.all(np.isfinite(H))
                    and np.all(np.isfinite(bk))
                    and np.all(np.isfinite(g))
                    and np.all(np.isfinite(phi))
                ):
                    raise RuntimeError(
                        "Physical channel contains non-finite values."
                    )

                # =============================================================
                # FAA + POWER CONTROL
                # =============================================================

                (
                    E_faa,
                    history_faa,
                ) = run_silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=faa_rng,
                    eps=eps,
                    do_apv=True,
                    C_apv=cfg.C_apv,
                    return_history=True,
                    g=g.copy(),
                    phi=phi.copy(),
                )

                # =============================================================
                # FPA + POWER CONTROL
                # =============================================================

                (
                    E_fpa,
                    history_fpa,
                ) = run_silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=fpa_rng,
                    eps=eps,
                    do_apv=False,
                    C_apv=None,
                    return_history=True,
                    g=None,
                    phi=None,
                )

                E_faa = float(E_faa)
                E_fpa = float(E_fpa)

                if not (
                    np.isfinite(E_faa)
                    and np.isfinite(E_fpa)
                    and E_faa >= 0.0
                    and E_fpa >= 0.0
                ):
                    raise RuntimeError(
                        f"Invalid energy returned: "
                        f"FAA={E_faa}, FPA={E_fpa}"
                    )

                history_faa = validate_history(
                    history_faa,
                    "FAA",
                )

                history_fpa = validate_history(
                    history_fpa,
                    "FPA",
                )

                # =============================================================
                # STATISTICS
                # =============================================================

                saving_pct = (
                    100.0
                    * (E_fpa - E_faa)
                    / max(E_fpa, LOG_FLOOR_W)
                )

                faa_monotone = is_monotone_nonincreasing(
                    history_faa
                )

                fpa_monotone = is_monotone_nonincreasing(
                    history_fpa
                )

                # The existing BCD convention is that reaching max_iter means
                # the run did not satisfy the stopping criterion before the
                # iteration cap.
                faa_converged = len(history_faa) < cfg.bcd_max_iter
                fpa_converged = len(history_fpa) < cfg.bcd_max_iter

                faa_wins = E_faa < E_fpa - WIN_TOL
                fpa_wins = E_fpa < E_faa - WIN_TOL
                tie = not (faa_wins or fpa_wins)

                rows.append(
                    {
                        "epsilon": eps,
                        "realization": realization + 1,
                        "seed": channel_seed,

                        "faa_energy_W": E_faa,
                        "fpa_energy_W": E_fpa,

                        "faa_energy_dBm": energy_to_dbm(E_faa),
                        "fpa_energy_dBm": energy_to_dbm(E_fpa),

                        "faa_energy_saving_pct": saving_pct,

                        "faa_iterations": len(history_faa),
                        "fpa_iterations": len(history_fpa),

                        "faa_converged": int(faa_converged),
                        "fpa_converged": int(fpa_converged),

                        "faa_monotone": int(faa_monotone),
                        "fpa_monotone": int(fpa_monotone),

                        "faa_wins": int(faa_wins),
                        "fpa_wins": int(fpa_wins),
                        "tie": int(tie),

                        "same_physical_channel": 1,
                    }
                )

                successful += 1

            except Exception as exc:
                failed += 1

                print()
                print(
                    "ERROR:"
                    f" epsilon={eps}"
                    f" realization={realization + 1}"
                    f" seed={channel_seed}"
                )
                print(
                    f"       {type(exc).__name__}: {exc}"
                )

                # Do not silently continue after a failed paired realization.
                # A partial dataset would compromise the paired experiment.
                raise

            if (
                (realization + 1) % 50 == 0
                or realization + 1 == NUM_REALIZATIONS
            ):
                elapsed = time.perf_counter() - eps_start
                completed = realization + 1
                avg_time = elapsed / completed
                remaining = NUM_REALIZATIONS - completed
                eta = remaining * avg_time

                print(
                    f"  {completed:4d}/{NUM_REALIZATIONS} "
                    f"completed | "
                    f"{elapsed / 60.0:.2f} min | "
                    f"ETA {eta / 60.0:.2f} min"
                )

        # ---------------------------------------------------------------------
        # Per-epsilon live summary
        # ---------------------------------------------------------------------

        eps_rows = [
            row for row in rows if row["epsilon"] == eps
        ]

        faa_energy = np.array(
            [row["faa_energy_W"] for row in eps_rows],
            dtype=float,
        )

        fpa_energy = np.array(
            [row["fpa_energy_W"] for row in eps_rows],
            dtype=float,
        )

        saving = np.array(
            [row["faa_energy_saving_pct"] for row in eps_rows],
            dtype=float,
        )

        faa_diff = fpa_energy - faa_energy

        ci_low, ci_high = confidence_interval_95(faa_diff)

        t_stat, t_p = ttest_rel(
            fpa_energy,
            faa_energy,
        )

        w_stat, w_p = safe_wilcoxon(
            faa_energy,
            fpa_energy,
        )

        faa_wins_count = int(
            np.sum(faa_energy < fpa_energy - WIN_TOL)
        )

        fpa_wins_count = int(
            np.sum(fpa_energy < faa_energy - WIN_TOL)
        )

        ties_count = (
            NUM_REALIZATIONS
            - faa_wins_count
            - fpa_wins_count
        )

        print()
        print(f"EPSILON = {eps:.2f}")
        print("-" * 70)
        print(f"FAA mean energy : {np.mean(faa_energy):.9e} W")
        print(f"FPA mean energy : {np.mean(fpa_energy):.9e} W")
        print(f"Mean saving     : {np.mean(saving):.4f}%")
        print(
            f"FAA wins        : "
            f"{faa_wins_count}/{NUM_REALIZATIONS}"
        )
        print(
            f"FPA wins        : "
            f"{fpa_wins_count}/{NUM_REALIZATIONS}"
        )
        print(
            f"Ties            : "
            f"{ties_count}/{NUM_REALIZATIONS}"
        )
        print(
            f"Paired 95% CI   : "
            f"[{ci_low:.9e}, {ci_high:.9e}] W"
        )
        print(f"Wilcoxon p      : {w_p:.6e}")
        print(f"Paired t p      : {t_p:.6e}")

    # =========================================================================
    # SAVE PER-REALIZATION CSV
    # =========================================================================

    if not rows:
        raise RuntimeError("No successful realizations were recorded.")

    expected_rows = len(EPSILONS) * NUM_REALIZATIONS

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Row-count validation failed: "
            f"expected {expected_rows}, got {len(rows)}."
        )

    fieldnames = list(rows[0].keys())

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # =========================================================================
    # BUILD FINAL SUMMARY
    # =========================================================================

    total_runtime = time.perf_counter() - total_start

    summary_lines = [
        "=" * 70,
        "FAA-AirComp EPSILON SWEEP",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 70,
        f"K = {K}",
        f"N = {N}",
        f"Realizations per epsilon = {NUM_REALIZATIONS}",
        f"Epsilon values = {EPSILONS}",
        f"Base seed = {BASE_SEED}",
        f"Project root = {PROJECT_ROOT}",
        "",
        "GLOBAL VALIDATION",
        "-" * 70,
        f"Successful realizations = {successful}",
        f"Failed realizations     = {failed}",
        f"Total expected          = {expected_rows}",
        f"Total runtime           = {total_runtime / 60.0:.2f} min",
    ]

    # =========================================================================
    # PER-EPSILON FINAL STATISTICS
    # =========================================================================

    for eps in EPSILONS:

        eps_rows = [
            row for row in rows if row["epsilon"] == eps
        ]

        if len(eps_rows) != NUM_REALIZATIONS:
            raise RuntimeError(
                f"Epsilon {eps:.2f} has {len(eps_rows)} rows; "
                f"expected {NUM_REALIZATIONS}."
            )

        faa_energy = np.array(
            [row["faa_energy_W"] for row in eps_rows],
            dtype=float,
        )

        fpa_energy = np.array(
            [row["fpa_energy_W"] for row in eps_rows],
            dtype=float,
        )

        saving = np.array(
            [row["faa_energy_saving_pct"] for row in eps_rows],
            dtype=float,
        )

        faa_iterations = np.array(
            [row["faa_iterations"] for row in eps_rows],
            dtype=float,
        )

        fpa_iterations = np.array(
            [row["fpa_iterations"] for row in eps_rows],
            dtype=float,
        )

        faa_diff = fpa_energy - faa_energy

        ci_low, ci_high = confidence_interval_95(faa_diff)

        t_stat, t_p = ttest_rel(
            fpa_energy,
            faa_energy,
        )

        w_stat, w_p = safe_wilcoxon(
            faa_energy,
            fpa_energy,
        )

        faa_wins = int(
            np.sum(faa_energy < fpa_energy - WIN_TOL)
        )

        fpa_wins = int(
            np.sum(fpa_energy < faa_energy - WIN_TOL)
        )

        ties = (
            NUM_REALIZATIONS
            - faa_wins
            - fpa_wins
        )

        faa_monotone = int(
            sum(row["faa_monotone"] for row in eps_rows)
        )

        fpa_monotone = int(
            sum(row["fpa_monotone"] for row in eps_rows)
        )

        faa_converged = int(
            sum(row["faa_converged"] for row in eps_rows)
        )

        fpa_converged = int(
            sum(row["fpa_converged"] for row in eps_rows)
        )

        same_channel = int(
            sum(row["same_physical_channel"] for row in eps_rows)
        )

        summary_lines.extend(
            [
                "",
                "=" * 70,
                f"EPSILON = {eps:.2f}",
                "=" * 70,
                "",
                "ENERGY — FAA",
                "-" * 70,
                f"Mean       : {np.mean(faa_energy):.9e} W",
                f"Median     : {np.median(faa_energy):.9e} W",
                f"Std        : {np.std(faa_energy, ddof=1):.9e} W",
                f"P05        : {np.percentile(faa_energy, 5):.9e} W",
                f"P25        : {np.percentile(faa_energy, 25):.9e} W",
                f"P75        : {np.percentile(faa_energy, 75):.9e} W",
                f"P95        : {np.percentile(faa_energy, 95):.9e} W",
                "",
                "ENERGY — FPA",
                "-" * 70,
                f"Mean       : {np.mean(fpa_energy):.9e} W",
                f"Median     : {np.median(fpa_energy):.9e} W",
                f"Std        : {np.std(fpa_energy, ddof=1):.9e} W",
                f"P05        : {np.percentile(fpa_energy, 5):.9e} W",
                f"P25        : {np.percentile(fpa_energy, 25):.9e} W",
                f"P75        : {np.percentile(fpa_energy, 75):.9e} W",
                f"P95        : {np.percentile(fpa_energy, 95):.9e} W",
                "",
                "FAA ENERGY SAVING VS FPA",
                "-" * 70,
                f"Mean       : {np.mean(saving):.4f}%",
                f"Median     : {np.median(saving):.4f}%",
                f"Std        : {np.std(saving, ddof=1):.4f}%",
                f"P05        : {np.percentile(saving, 5):.4f}%",
                f"P25        : {np.percentile(saving, 25):.4f}%",
                f"P75        : {np.percentile(saving, 75):.4f}%",
                f"P95        : {np.percentile(saving, 95):.4f}%",
                "",
                "WIN COUNTS",
                "-" * 70,
                f"FAA wins   : {faa_wins}/{NUM_REALIZATIONS} "
                f"({100.0 * faa_wins / NUM_REALIZATIONS:.2f}%)",
                f"FPA wins   : {fpa_wins}/{NUM_REALIZATIONS} "
                f"({100.0 * fpa_wins / NUM_REALIZATIONS:.2f}%)",
                f"Ties       : {ties}/{NUM_REALIZATIONS} "
                f"({100.0 * ties / NUM_REALIZATIONS:.2f}%)",
                "",
                "CONVERGENCE",
                "-" * 70,
                f"FAA mean iterations   : {np.mean(faa_iterations):.3f}",
                f"FAA median iterations : {np.median(faa_iterations):.3f}",
                f"FAA P05/P25/P75/P95   : "
                f"{np.percentile(faa_iterations, 5):.0f}/"
                f"{np.percentile(faa_iterations, 25):.0f}/"
                f"{np.percentile(faa_iterations, 75):.0f}/"
                f"{np.percentile(faa_iterations, 95):.0f}",
                f"FPA mean iterations   : {np.mean(fpa_iterations):.3f}",
                f"FPA median iterations : {np.median(fpa_iterations):.3f}",
                f"FPA P05/P25/P75/P95   : "
                f"{np.percentile(fpa_iterations, 5):.0f}/"
                f"{np.percentile(fpa_iterations, 25):.0f}/"
                f"{np.percentile(fpa_iterations, 75):.0f}/"
                f"{np.percentile(fpa_iterations, 95):.0f}",
                "",
                "CONVERGENCE RATE",
                "-" * 70,
                f"FAA converged         : "
                f"{faa_converged}/{NUM_REALIZATIONS}",
                f"FPA converged         : "
                f"{fpa_converged}/{NUM_REALIZATIONS}",
                "",
                "MONOTONICITY",
                "-" * 70,
                f"FAA monotone          : "
                f"{faa_monotone}/{NUM_REALIZATIONS}",
                f"FPA monotone          : "
                f"{fpa_monotone}/{NUM_REALIZATIONS}",
                "",
                "PHYSICAL CHANNEL VALIDATION",
                "-" * 70,
                f"Valid physical channels : "
                f"{same_channel}/{NUM_REALIZATIONS}",
                "",
                "PAIRED STATISTICS",
                "-" * 70,
                f"Mean paired difference (FPA - FAA): "
                f"{np.mean(faa_diff):.9e} W",
                f"95% CI for difference: "
                f"[{ci_low:.9e}, {ci_high:.9e}] W",
                f"Wilcoxon statistic: {w_stat:.6g}",
                f"Wilcoxon p-value: {w_p:.6e}",
                f"Paired t statistic: {t_stat:.6f}",
                f"Paired t p-value: {t_p:.6e}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "=" * 70,
            "EXPERIMENT COMPLETE",
            "=" * 70,
            f"Per-realization data saved to: {CSV_PATH}",
            f"Summary saved to: {SUMMARY_PATH}",
        ]
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(summary_lines))

    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()