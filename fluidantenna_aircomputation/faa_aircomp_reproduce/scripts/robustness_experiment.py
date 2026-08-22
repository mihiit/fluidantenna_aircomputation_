"""
robustness_experiment.py
========================

FAA-AirComp robustness experiment.

PURPOSE
-------
Test whether the FAA + power-control energy advantage persists
across independent physical-channel realizations.

IMPORTANT
---------
This is a SEPARATE experiment.

It does NOT modify:
    - src/bcd.py
    - scripts/scaling_experiment.py
    - epsilon used by the existing scaling experiment

The robustness experiment deliberately uses:

    epsilon = 0.08

and only NON-DEGENERATE K values:

    K = [4, 8, 12]

because for epsilon = 0.08:

    K=4   -> 1/K = 0.250000
    K=8   -> 1/K = 0.125000
    K=12  -> 1/K = 0.083333

while:

    K=16  -> 1/K = 0.062500 < 0.08

which permits the zero-power solution.

EXPERIMENT
----------
For each (K, N):

    K = [4, 8, 12]
    N = [4, 6, 8]

Run:

    NUM_REALIZATIONS = 50

For every realization:

    1. Generate exactly ONE physical channel.
    2. Use that SAME physical channel for FAA and FPA.
    3. Keep g and phi fixed for the FAA APV search.
    4. Run FAA + power control.
    5. Run FPA + power control.
    6. Record energy and convergence information.
    7. Compute paired FAA-vs-FPA statistics.

The purpose is robustness, NOT replacement of the scaling experiment.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel, wilcoxon


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if not PROJECT_ROOT.exists():
    raise RuntimeError(
        f"Project root does not exist: {PROJECT_ROOT}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.config import SystemConfig
from src.channel import make_channel
from src.bcd import run_bcd


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# IMPORTANT:
# Keep this experiment at epsilon = 0.08.
EPSILON = 0.08


# Only non-degenerate K values for epsilon = 0.08.
K_VALUES = [
    4,
    8,
    12,
]


N_VALUES = [
    4,
    6,
    8,
]


# 50 realizations per configuration.
#
# This gives:
#
#     3 K values x 3 N values x 50
#
# = 450 paired physical-channel trials.
#
# This is large enough to provide useful robustness evidence
# without immediately launching another 500-realization-per-cell
# experiment.
NUM_REALIZATIONS = 50


# Keep this independent from the scaling experiment seed.
BASE_SEED = 20260821


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "output"

CSV_PATH = (
    OUTPUT_DIR
    / "robustness_experiment_eps008.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "robustness_experiment_eps008_summary.txt"
)

PROGRESS_PATH = (
    OUTPUT_DIR
    / "robustness_experiment_eps008_progress.txt"
)

FAILURE_PATH = (
    OUTPUT_DIR
    / "robustness_experiment_eps008_failures.csv"
)


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

MONO_TOL = 1e-9

WIN_TOL = 1e-12

LOG_FLOOR_W = 1e-15


# =============================================================================
# CSV SCHEMA
# =============================================================================

FIELDS = [
    "K",
    "N",
    "epsilon",
    "realization",
    "channel_seed",

    "faa_energy_W",
    "fpa_energy_W",

    "faa_energy_dBm",
    "fpa_energy_dBm",

    "faa_energy_saving_pct",

    "faa_iterations",
    "fpa_iterations",

    "faa_converged",
    "fpa_converged",

    "faa_monotone",
    "fpa_monotone",

    "faa_wins",
    "fpa_wins",
    "tie",

    "same_physical_channel",
]


# =============================================================================
# HELPERS
# =============================================================================

def silent(fn, *args, **kwargs):
    """
    Run BCD while suppressing its diagnostic console output.

    The underlying algorithm is unchanged.
    """

    with open(
        os.devnull,
        "w",
    ) as devnull:

        with redirect_stdout(
            devnull
        ):

            return fn(
                *args,
                **kwargs
            )


# -----------------------------------------------------------------------------


def monotone(history):
    """
    Check:

        E_1 >= E_2 >= ... >= E_T

    within numerical tolerance.
    """

    if len(history) <= 1:
        return True

    return all(
        history[i]
        >= history[i + 1]
        - MONO_TOL

        for i in range(
            len(history) - 1
        )
    )


# -----------------------------------------------------------------------------


def dbm(power_w):
    """
    Convert W to dBm.

        P[dBm] = 10 log10(P[W] * 1000)
    """

    return float(
        10.0
        * np.log10(
            max(
                float(power_w),
                LOG_FLOOR_W,
            )
            * 1000.0
        )
    )


# -----------------------------------------------------------------------------


def ci95(values):
    """
    95% confidence interval for the sample mean.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    n = len(x)

    if n < 2:
        return (
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(x)
    )

    std = float(
        np.std(
            x,
            ddof=1,
        )
    )

    se = (
        std
        / math.sqrt(n)
    )

    critical = float(
        t.ppf(
            0.975,
            n - 1,
        )
    )

    margin = (
        critical
        * se
    )

    return (
        mean - margin,
        mean + margin,
    )


# -----------------------------------------------------------------------------


def wilcoxon_test(
    faa,
    fpa,
):
    """
    Paired two-sided Wilcoxon signed-rank test.

    Positive difference means:

        FPA energy > FAA energy

    which favors FAA.
    """

    x = np.asarray(
        faa,
        dtype=float,
    )

    y = np.asarray(
        fpa,
        dtype=float,
    )

    difference = (
        y - x
    )

    if np.allclose(
        difference,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):

        return (
            0.0,
            1.0,
        )

    result = wilcoxon(
        x,
        y,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


# -----------------------------------------------------------------------------


def validate_energy(
    value,
    label,
):
    """
    Strictly validate energy.

    Zero is NOT converted into a fake positive number.
    """

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):

        raise RuntimeError(
            f"{label} returned non-finite energy: "
            f"{value}"
        )

    if value <= 0.0:

        raise RuntimeError(
            f"{label} returned non-positive energy: "
            f"{value}"
        )

    return value


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_rows():
    """
    Load previously completed realizations.
    """

    rows = {}

    if not CSV_PATH.exists():
        return rows

    with CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        if not reader.fieldnames:
            return rows

        missing = [
            field
            for field in FIELDS
            if field not in reader.fieldnames
        ]

        if missing:

            raise RuntimeError(
                "Existing robustness CSV has "
                f"incompatible schema. Missing: {missing}"
            )

        for row in reader:

            try:

                row_epsilon = float(
                    row["epsilon"]
                )

                key = (
                    int(row["K"]),
                    int(row["N"]),
                    int(row["realization"]),
                )

            except (
                ValueError,
                KeyError,
            ):

                continue

            if not np.isclose(
                row_epsilon,
                EPSILON,
                atol=0.0,
                rtol=0.0,
            ):

                raise RuntimeError(
                    "Robustness checkpoint epsilon mismatch: "
                    f"found {row_epsilon}, "
                    f"expected {EPSILON}."
                )

            rows[key] = row

    return rows


# =============================================================================
# TYPE CONVERSION
# =============================================================================

def typed_rows(rows):
    """
    Convert CSV strings to numeric values.
    """

    integer_fields = {
        "K",
        "N",
        "realization",
        "channel_seed",

        "faa_iterations",
        "fpa_iterations",

        "faa_converged",
        "fpa_converged",

        "faa_monotone",
        "fpa_monotone",

        "faa_wins",
        "fpa_wins",
        "tie",

        "same_physical_channel",
    }

    float_fields = {
        "epsilon",

        "faa_energy_W",
        "fpa_energy_W",

        "faa_energy_dBm",
        "fpa_energy_dBm",

        "faa_energy_saving_pct",
    }

    output = []

    source = (
        rows.values()
        if isinstance(rows, dict)
        else rows
    )

    for original in source:

        row = dict(
            original
        )

        for field in integer_fields:
            row[field] = int(
                row[field]
            )

        for field in float_fields:
            row[field] = float(
                row[field]
            )

        output.append(
            row
        )

    return output


# =============================================================================
# CHECKPOINT WRITE
# =============================================================================

def append_row(row):
    """
    Append one successful realization immediately.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        CSV_PATH.exists()
        and CSV_PATH.stat().st_size > 0
    )

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


# =============================================================================
# FAILURE LOG
# =============================================================================

def append_failure(
    K,
    N,
    realization,
    channel_seed,
    exc,
):
    """
    Record failure without fabricating a result.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "K",
        "N",
        "epsilon",
        "realization",
        "channel_seed",
        "error_type",
        "error_message",
    ]

    exists = (
        FAILURE_PATH.exists()
        and FAILURE_PATH.stat().st_size > 0
    )

    with FAILURE_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                "K": K,
                "N": N,
                "epsilon": EPSILON,
                "realization": realization,
                "channel_seed": channel_seed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


# =============================================================================
# PROGRESS
# =============================================================================

def write_progress(
    completed,
    total,
    start,
):
    """
    Write progress checkpoint.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    elapsed_minutes = (
        time.perf_counter()
        - start
    ) / 60.0

    with PROGRESS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FAA-AirComp ROBUSTNESS EXPERIMENT\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Epsilon        : {EPSILON}\n"
        )

        f.write(
            f"K values       : {K_VALUES}\n"
        )

        f.write(
            f"N values       : {N_VALUES}\n"
        )

        f.write(
            f"Realizations   : {NUM_REALIZATIONS}\n"
        )

        f.write(
            f"Completed      : {completed}/{total}\n"
        )

        f.write(
            f"Progress       : "
            f"{100.0 * completed / total:.2f}%\n"
        )

        f.write(
            f"Runtime        : "
            f"{elapsed_minutes:.2f} min\n"
        )

        f.write(
            f"CSV            : {CSV_PATH}\n"
        )


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(rows):
    """
    Compute paired FAA-vs-FPA statistics.
    """

    faa = np.asarray(
        [
            row["faa_energy_W"]
            for row in rows
        ],
        dtype=float,
    )

    fpa = np.asarray(
        [
            row["fpa_energy_W"]
            for row in rows
        ],
        dtype=float,
    )

    saving = np.asarray(
        [
            row["faa_energy_saving_pct"]
            for row in rows
        ],
        dtype=float,
    )

    difference = (
        fpa
        - faa
    )

    ci_low, ci_high = ci95(
        difference
    )

    _, paired_t_p = ttest_rel(
        fpa,
        faa,
    )

    _, wilcoxon_p = wilcoxon_test(
        faa,
        fpa,
    )

    faa_wins = int(
        np.sum(
            faa
            < fpa - WIN_TOL
        )
    )

    fpa_wins = int(
        np.sum(
            fpa
            < faa - WIN_TOL
        )
    )

    ties = (
        len(rows)
        - faa_wins
        - fpa_wins
    )

    return {
        "faa_mean": float(
            np.mean(faa)
        ),

        "fpa_mean": float(
            np.mean(fpa)
        ),

        "faa_median": float(
            np.median(faa)
        ),

        "fpa_median": float(
            np.median(fpa)
        ),

        "saving_mean": float(
            np.mean(saving)
        ),

        "saving_median": float(
            np.median(saving)
        ),

        "faa_wins": faa_wins,

        "fpa_wins": fpa_wins,

        "ties": ties,

        "faa_win_rate": (
            100.0
            * faa_wins
            / len(rows)
        ),

        "mean_difference": float(
            np.mean(difference)
        ),

        "ci_low": ci_low,

        "ci_high": ci_high,

        "paired_t_p": float(
            paired_t_p
        ),

        "wilcoxon_p": float(
            wilcoxon_p
        ),

        "faa_mean_iterations": float(
            np.mean(
                [
                    row["faa_iterations"]
                    for row in rows
                ]
            )
        ),

        "fpa_mean_iterations": float(
            np.mean(
                [
                    row["fpa_iterations"]
                    for row in rows
                ]
            )
        ),

        "faa_converged": int(
            sum(
                row["faa_converged"]
                for row in rows
            )
        ),

        "fpa_converged": int(
            sum(
                row["fpa_converged"]
                for row in rows
            )
        ),

        "faa_monotone": int(
            sum(
                row["faa_monotone"]
                for row in rows
            )
        ),

        "fpa_monotone": int(
            sum(
                row["fpa_monotone"]
                for row in rows
            )
        ),
    }


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(
    rows,
    start,
    total,
):
    """
    Write final robustness report.
    """

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "FAA-AirComp ROBUSTNESS EXPERIMENT"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        "EXPERIMENT CONFIGURATION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Epsilon = {EPSILON}"
    )

    lines.append(
        f"K values = {K_VALUES}"
    )

    lines.append(
        f"N values = {N_VALUES}"
    )

    lines.append(
        f"Realizations per configuration = "
        f"{NUM_REALIZATIONS}"
    )

    lines.append(
        f"Total paired trials = {total}"
    )

    lines.append(
        f"Base seed = {BASE_SEED}"
    )

    lines.append("")

    lines.append(
        "GLOBAL VALIDATION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Successful realizations : "
        f"{len(rows)}"
    )

    lines.append(
        f"Expected realizations   : "
        f"{total}"
    )

    lines.append(
        f"Failed realizations     : "
        f"{total - len(rows)}"
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # Per configuration
    # -------------------------------------------------------------------------

    for K in K_VALUES:

        for N in N_VALUES:

            cell = [
                row
                for row in rows
                if row["K"] == K
                and row["N"] == N
            ]

            if len(cell) != NUM_REALIZATIONS:
                continue

            stats = compute_statistics(
                cell
            )

            lines.append(
                "=" * 78
            )

            lines.append(
                f"K = {K}, N = {N}"
            )

            lines.append(
                "=" * 78
            )

            lines.append(
                f"FAA mean energy       : "
                f"{stats['faa_mean']:.9e} W"
            )

            lines.append(
                f"FPA mean energy       : "
                f"{stats['fpa_mean']:.9e} W"
            )

            lines.append(
                f"FAA median energy     : "
                f"{stats['faa_median']:.9e} W"
            )

            lines.append(
                f"FPA median energy     : "
                f"{stats['fpa_median']:.9e} W"
            )

            lines.append(
                f"Mean FAA saving       : "
                f"{stats['saving_mean']:.4f}%"
            )

            lines.append(
                f"Median FAA saving     : "
                f"{stats['saving_median']:.4f}%"
            )

            lines.append(
                f"FAA wins              : "
                f"{stats['faa_wins']}/{NUM_REALIZATIONS}"
            )

            lines.append(
                f"FPA wins              : "
                f"{stats['fpa_wins']}/{NUM_REALIZATIONS}"
            )

            lines.append(
                f"Ties                  : "
                f"{stats['ties']}/{NUM_REALIZATIONS}"
            )

            lines.append(
                f"FAA win rate          : "
                f"{stats['faa_win_rate']:.2f}%"
            )

            lines.append(
                f"Mean paired difference: "
                f"{stats['mean_difference']:.9e} W"
            )

            lines.append(
                f"95% CI difference     : "
                f"[{stats['ci_low']:.9e}, "
                f"{stats['ci_high']:.9e}] W"
            )

            lines.append(
                f"Wilcoxon p            : "
                f"{stats['wilcoxon_p']:.6e}"
            )

            lines.append(
                f"Paired t-test p       : "
                f"{stats['paired_t_p']:.6e}"
            )

            lines.append(
                f"FAA mean iterations   : "
                f"{stats['faa_mean_iterations']:.3f}"
            )

            lines.append(
                f"FPA mean iterations   : "
                f"{stats['fpa_mean_iterations']:.3f}"
            )

            lines.append(
                f"FAA converged         : "
                f"{stats['faa_converged']}/"
                f"{NUM_REALIZATIONS}"
            )

            lines.append(
                f"FPA converged         : "
                f"{stats['fpa_converged']}/"
                f"{NUM_REALIZATIONS}"
            )

            lines.append(
                f"FAA monotone          : "
                f"{stats['faa_monotone']}/"
                f"{NUM_REALIZATIONS}"
            )

            lines.append(
                f"FPA monotone          : "
                f"{stats['fpa_monotone']}/"
                f"{NUM_REALIZATIONS}"
            )

            lines.append("")

    # -------------------------------------------------------------------------
    # Aggregate across all paired trials
    # -------------------------------------------------------------------------

    if len(rows) > 0:

        overall = compute_statistics(
            rows
        )

        lines.append(
            "=" * 78
        )

        lines.append(
            "AGGREGATE ROBUSTNESS RESULT"
        )

        lines.append(
            "=" * 78
        )

        lines.append(
            f"All successful paired trials : "
            f"{len(rows)}"
        )

        lines.append(
            f"Mean FAA energy              : "
            f"{overall['faa_mean']:.9e} W"
        )

        lines.append(
            f"Mean FPA energy              : "
            f"{overall['fpa_mean']:.9e} W"
        )

        lines.append(
            f"Mean FAA saving              : "
            f"{overall['saving_mean']:.4f}%"
        )

        lines.append(
            f"FAA wins                     : "
            f"{overall['faa_wins']}"
        )

        lines.append(
            f"FPA wins                     : "
            f"{overall['fpa_wins']}"
        )

        lines.append(
            f"Ties                         : "
            f"{overall['ties']}"
        )

        lines.append(
            f"FAA win rate                 : "
            f"{overall['faa_win_rate']:.2f}%"
        )

        lines.append(
            f"Wilcoxon p                   : "
            f"{overall['wilcoxon_p']:.6e}"
        )

        lines.append(
            f"Paired t-test p              : "
            f"{overall['paired_t_p']:.6e}"
        )

        lines.append("")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    lines.append(
        "=" * 78
    )

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        "This experiment evaluates robustness at epsilon = 0.08 "
        "using only non-degenerate K values."
    )

    lines.append(
        "FAA and FPA are evaluated on the same physical channel "
        "for every paired realization."
    )

    lines.append(
        "The experiment does not modify the BCD implementation."
    )

    lines.append(
        "The experiment does not modify the existing scaling experiment."
    )

    lines.append("")

    lines.append(
        "A consistent FAA advantage should be supported by:"
    )

    lines.append(
        "  1. Positive mean energy savings."
    )

    lines.append(
        "  2. High FAA paired-win rate."
    )

    lines.append(
        "  3. A paired confidence interval that excludes zero."
    )

    lines.append(
        "  4. A statistically significant paired test."
    )

    lines.append(
        "  5. No unexplained feasibility failures."
    )

    lines.append("")

    lines.append(
        "These results are robustness evidence; they are not "
        "a theoretical scaling law."
    )

    lines.append("")

    lines.append(
        "OUTPUT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"CSV      : {CSV_PATH}"
    )

    lines.append(
        f"Summary  : {SUMMARY_PATH}"
    )

    lines.append(
        f"Progress : {PROGRESS_PATH}"
    )

    lines.append(
        f"Failures : {FAILURE_PATH}"
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = SystemConfig()

    start = time.perf_counter()

    configs = [
        (K, N)
        for K in K_VALUES
        for N in N_VALUES
    ]

    total = (
        len(configs)
        * NUM_REALIZATIONS
    )

    done = load_rows()

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "FAA-AirComp ROBUSTNESS EXPERIMENT"
    )

    print(
        "=" * 78
    )

    print(
        f"Epsilon            : {EPSILON}"
    )

    print(
        f"K values           : {K_VALUES}"
    )

    print(
        f"N values           : {N_VALUES}"
    )

    print(
        f"Realizations/cell  : {NUM_REALIZATIONS}"
    )

    print(
        f"Total paired trials: {total}"
    )

    print(
        f"Checkpoint rows    : "
        f"{len(done)}/{total}"
    )

    print(
        f"CSV checkpoint     : "
        f"{CSV_PATH}"
    )

    print()

    write_progress(
        len(done),
        total,
        start,
    )

    # -------------------------------------------------------------------------
    # Nothing to run
    # -------------------------------------------------------------------------

    if len(done) == total:

        print(
            "[RESUME] All robustness realizations "
            "already completed."
        )

        write_summary(
            typed_rows(done),
            start,
            total,
        )

        return

    # -------------------------------------------------------------------------
    # Main experiment
    # -------------------------------------------------------------------------

    for config_index, (K, N) in enumerate(
        configs,
        start=1,
    ):

        completed_cell = sum(
            1
            for key in done
            if key[:2] == (K, N)
        )

        print()

        print(
            f"CONFIGURATION "
            f"{config_index}/{len(configs)} "
            f"-> K={K}, N={N} "
            f"({completed_cell}/"
            f"{NUM_REALIZATIONS} already complete)"
        )

        # ---------------------------------------------------------------------
        # Realizations
        # ---------------------------------------------------------------------

        for realization in range(
            1,
            NUM_REALIZATIONS + 1,
        ):

            key = (
                K,
                N,
                realization,
            )

            if key in done:
                continue

            # -----------------------------------------------------------------
            # Deterministic independent seeds.
            #
            # One channel seed.
            # Separate FAA RNG.
            # Separate FPA RNG.
            # -----------------------------------------------------------------

            channel_seed = (
                BASE_SEED
                + 100000 * K
                + 1000 * N
                + (
                    realization - 1
                )
            )

            faa_seed = (
                BASE_SEED
                + 10000000
                + 100000 * K
                + 1000 * N
                + (
                    realization - 1
                )
            )

            fpa_seed = (
                BASE_SEED
                + 20000000
                + 100000 * K
                + 1000 * N
                + (
                    realization - 1
                )
            )

            channel_rng = (
                np.random.default_rng(
                    channel_seed
                )
            )

            faa_rng = (
                np.random.default_rng(
                    faa_seed
                )
            )

            fpa_rng = (
                np.random.default_rng(
                    fpa_seed
                )
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

                H = np.asarray(
                    H,
                    dtype=complex,
                )

                bk = np.asarray(
                    bk,
                    dtype=float,
                )

                g = np.asarray(
                    g
                )

                phi = np.asarray(
                    phi
                )

                # =============================================================
                # VALIDATE CHANNEL
                # =============================================================

                if H.shape != (
                    N,
                    K,
                ):

                    raise RuntimeError(
                        f"Invalid H shape {H.shape}; "
                        f"expected {(N, K)}"
                    )

                if bk.shape != (
                    K,
                ):

                    raise RuntimeError(
                        f"Invalid bk shape {bk.shape}; "
                        f"expected {(K,)}"
                    )

                if (
                    g.shape[0] != K
                    or phi.shape[0] != K
                ):

                    raise RuntimeError(
                        "Invalid g/phi user dimension."
                    )

                # =============================================================
                # FAA
                # =============================================================

                (
                    E_faa,
                    history_faa,
                ) = silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=faa_rng,
                    eps=EPSILON,
                    do_apv=True,
                    C_apv=cfg.C_apv,
                    return_history=True,
                    g=g.copy(),
                    phi=phi.copy(),
                )

                # =============================================================
                # FPA
                #
                # IMPORTANT:
                #
                # SAME H is used.
                #
                # No APV.
                # =============================================================

                (
                    E_fpa,
                    history_fpa,
                ) = silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=fpa_rng,
                    eps=EPSILON,
                    do_apv=False,
                    C_apv=None,
                    return_history=True,
                    g=None,
                    phi=None,
                )

                # =============================================================
                # VALIDATE RESULTS
                # =============================================================

                E_faa = validate_energy(
                    E_faa,
                    "FAA",
                )

                E_fpa = validate_energy(
                    E_fpa,
                    "FPA",
                )

                history_faa = list(
                    history_faa
                )

                history_fpa = list(
                    history_fpa
                )

                if not history_faa:

                    raise RuntimeError(
                        "FAA returned empty convergence history."
                    )

                if not history_fpa:

                    raise RuntimeError(
                        "FPA returned empty convergence history."
                    )

                # =============================================================
                # PAIRED COMPARISON
                # =============================================================

                saving_pct = (
                    100.0
                    * (
                        E_fpa
                        - E_faa
                    )
                    / max(
                        E_fpa,
                        LOG_FLOOR_W,
                    )
                )

                faa_wins = (
                    E_faa
                    < E_fpa - WIN_TOL
                )

                fpa_wins = (
                    E_fpa
                    < E_faa - WIN_TOL
                )

                tie = not (
                    faa_wins
                    or fpa_wins
                )

                # =============================================================
                # RECORD
                # =============================================================

                row = {
                    "K": K,
                    "N": N,
                    "epsilon": EPSILON,
                    "realization": realization,
                    "channel_seed": channel_seed,

                    "faa_energy_W": E_faa,
                    "fpa_energy_W": E_fpa,

                    "faa_energy_dBm": dbm(
                        E_faa
                    ),

                    "fpa_energy_dBm": dbm(
                        E_fpa
                    ),

                    "faa_energy_saving_pct": (
                        saving_pct
                    ),

                    "faa_iterations": len(
                        history_faa
                    ),

                    "fpa_iterations": len(
                        history_fpa
                    ),

                    "faa_converged": int(
                        len(history_faa)
                        < cfg.bcd_max_iter
                    ),

                    "fpa_converged": int(
                        len(history_fpa)
                        < cfg.bcd_max_iter
                    ),

                    "faa_monotone": int(
                        monotone(
                            history_faa
                        )
                    ),

                    "fpa_monotone": int(
                        monotone(
                            history_fpa
                        )
                    ),

                    "faa_wins": int(
                        faa_wins
                    ),

                    "fpa_wins": int(
                        fpa_wins
                    ),

                    "tie": int(
                        tie
                    ),

                    "same_physical_channel": 1,
                }

                append_row(
                    row
                )

                done[key] = {
                    field: str(
                        row[field]
                    )
                    for field in FIELDS
                }

                write_progress(
                    len(done),
                    total,
                    start,
                )

                # -------------------------------------------------------------
                # Console progress
                # -------------------------------------------------------------

                if (
                    realization % 10 == 0
                    or realization
                    == NUM_REALIZATIONS
                ):

                    print(
                        f"  "
                        f"{realization:4d}/"
                        f"{NUM_REALIZATIONS} "
                        f"| global "
                        f"{len(done):4d}/"
                        f"{total}"
                    )

            # -----------------------------------------------------------------
            # FAILURE
            # -----------------------------------------------------------------

            except Exception as exc:

                append_failure(
                    K,
                    N,
                    realization,
                    channel_seed,
                    exc,
                )

                print()

                print(
                    "=" * 78
                )

                print(
                    "ROBUSTNESS EXPERIMENT STOPPED"
                )

                print(
                    "=" * 78
                )

                print(
                    f"K             : {K}"
                )

                print(
                    f"N             : {N}"
                )

                print(
                    f"Realization   : {realization}"
                )

                print(
                    f"Channel seed  : {channel_seed}"
                )

                print(
                    f"Error         : "
                    f"{type(exc).__name__}: {exc}"
                )

                print()

                print(
                    "All previous successful "
                    "realizations are checkpointed."
                )

                print(
                    "No failed observation is "
                    "converted into a result."
                )

                raise

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    rows = typed_rows(
        done
    )

    if len(rows) == total:

        write_summary(
            rows,
            start,
            total,
        )

    else:

        print(
            f"Incomplete: "
            f"{len(rows)}/{total}."
        )

        print(
            "Rerun the script to continue."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()