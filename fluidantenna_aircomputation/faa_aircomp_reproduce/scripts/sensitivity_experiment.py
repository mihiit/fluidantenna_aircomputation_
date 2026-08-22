"""
sensitivity_experiment.py
=========================

FAA-AirComp epsilon-sensitivity experiment.

PURPOSE
-------
Evaluate whether the FAA energy advantage over FPA persists when the
MSE constraint parameter epsilon is varied.

This experiment is intentionally SEPARATE from:

    - src/bcd.py
    - scripts/scaling_experiment.py
    - scripts/robustness_experiment.py

It does NOT modify the optimization algorithm.

DESIGN
------
epsilon values:

    [0.04, 0.06, 0.08]

K values:

    [4, 8, 12]

N values:

    [4, 6, 8]

realizations per configuration:

    30

Total:

    3 epsilon x 3 K x 3 N x 30
    = 810 paired FAA/FPA trials

IMPORTANT DEGENERACY CONDITION
------------------------------
Under the current MSE definition, zero power becomes feasible when:

    1/K <= epsilon

Therefore:

    K=4:
        1/K = 0.250000
        all tested epsilon values are safe.

    K=8:
        1/K = 0.125000
        all tested epsilon values are safe.

    K=12:
        1/K = 0.083333
        epsilon <= 0.08 remains non-degenerate.

Thus this experiment deliberately avoids K >= 16.

PAIRING
-------
For each (K, N, realization):

    1. Generate ONE physical channel.
    2. Keep that physical channel fixed for all epsilon values.
    3. Keep g and phi fixed for all epsilon values.
    4. Run FAA at each epsilon.
    5. Run FPA at each epsilon.
    6. Compare FAA and FPA on the same physical channel.

This isolates the effect of epsilon from channel realization.

The experiment is sensitivity/robustness evidence.

It is NOT a theoretical scaling-law experiment.
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
from scipy.stats import t
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


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

# -------------------------------------------------------------------------
# Epsilon sensitivity range.
#
# All three values are deliberately non-degenerate for K=[4,8,12].
# -------------------------------------------------------------------------

EPSILON_VALUES = [
    0.04,
    0.06,
    0.08,
]


# -------------------------------------------------------------------------
# K values.
#
# K=16 and K=20 are intentionally excluded because:
#
#   K=16 -> 1/K = 0.0625 < 0.08
#   K=20 -> 1/K = 0.05   < 0.08
#
# and therefore epsilon=0.08 permits zero power.
# -------------------------------------------------------------------------

K_VALUES = [
    4,
    8,
    12,
]


# -------------------------------------------------------------------------
# Antenna-port values.
# -------------------------------------------------------------------------

N_VALUES = [
    4,
    6,
    8,
]


# -------------------------------------------------------------------------
# Number of independent physical-channel realizations per cell.
#
# 3 epsilon x 3 K x 3 N x 30
# = 810 paired trials.
# -------------------------------------------------------------------------

NUM_REALIZATIONS = 30


# -------------------------------------------------------------------------
# Independent seed from the other experiments.
# -------------------------------------------------------------------------

BASE_SEED = 20260822


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "output"


CSV_PATH = (
    OUTPUT_DIR
    / "sensitivity_experiment_eps004_006_008.csv"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "sensitivity_experiment_eps004_006_008_summary.txt"
)


PROGRESS_PATH = (
    OUTPUT_DIR
    / "sensitivity_experiment_eps004_006_008_progress.txt"
)


FAILURE_PATH = (
    OUTPUT_DIR
    / "sensitivity_experiment_eps004_006_008_failures.csv"
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
    "epsilon",
    "K",
    "N",
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
    "same_g_phi",
]


# =============================================================================
# HELPERS
# =============================================================================

def silent(fn, *args, **kwargs):
    """
    Run BCD while suppressing its console output.

    The underlying BCD implementation is NOT changed.
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
    Verify non-increasing energy history:

        E_1 >= E_2 >= ... >= E_T

    within numerical tolerance.
    """

    if len(history) <= 1:
        return True

    return all(
        history[i]
        >= history[i + 1] - MONO_TOL
        for i in range(
            len(history) - 1
        )
    )


# -----------------------------------------------------------------------------


def dbm(power_w):
    """
    Convert watts to dBm.
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

    Positive difference:

        FPA - FAA > 0

    favors FAA.
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
    Strictly validate returned energy.

    Never replace zero with an artificial positive value.
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
# DEGENERACY VALIDATION
# =============================================================================

def validate_experiment_grid():
    """
    Verify that every requested (K, epsilon) combination is non-degenerate.

    Zero-power becomes feasible when:

        1/K <= epsilon
    """

    invalid = []

    for K in K_VALUES:

        for epsilon in EPSILON_VALUES:

            threshold = 1.0 / K

            if threshold <= epsilon:

                invalid.append(
                    (
                        K,
                        epsilon,
                        threshold,
                    )
                )

    if invalid:

        message = [
            "Sensitivity experiment contains degenerate "
            "(K, epsilon) combinations:"
        ]

        for K, epsilon, threshold in invalid:

            message.append(
                f"  K={K}, epsilon={epsilon}, "
                f"1/K={threshold}"
            )

        raise RuntimeError(
            "\n".join(message)
        )


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
                "Existing sensitivity CSV has "
                f"incompatible schema. Missing: {missing}"
            )

        for row in reader:

            try:

                epsilon = float(
                    row["epsilon"]
                )

                K = int(
                    row["K"]
                )

                N = int(
                    row["N"]
                )

                realization = int(
                    row["realization"]
                )

            except (
                ValueError,
                KeyError,
            ):

                continue

            if not any(
                np.isclose(
                    epsilon,
                    value,
                    atol=0.0,
                    rtol=0.0,
                )
                for value in EPSILON_VALUES
            ):

                raise RuntimeError(
                    "Existing sensitivity checkpoint contains "
                    f"unexpected epsilon={epsilon}."
                )

            key = (
                epsilon,
                K,
                N,
                realization,
            )

            rows[key] = row

    return rows


# =============================================================================
# TYPE CONVERSION
# =============================================================================

def typed_rows(rows):
    """
    Convert CSV strings into numeric values.
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
        "same_g_phi",
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
    Append one successful result immediately.
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
    epsilon,
    K,
    N,
    realization,
    channel_seed,
    exc,
):
    """
    Record failure without fabricating an observation.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "epsilon",
        "K",
        "N",
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
                "epsilon": epsilon,
                "K": K,
                "N": N,
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
    Write resumable progress information.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    elapsed_minutes = (
        time.perf_counter()
        - start
    ) / 60.0

    percentage = (
        100.0
        * completed
        / total
        if total > 0
        else 0.0
    )

    with PROGRESS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FAA-AirComp EPSILON SENSITIVITY EXPERIMENT\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Epsilon values : {EPSILON_VALUES}\n"
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
            f"Progress       : {percentage:.2f}%\n"
        )

        f.write(
            f"Runtime        : {elapsed_minutes:.2f} min\n"
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

    if not rows:
        raise RuntimeError(
            "Cannot compute statistics on zero rows."
        )

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
        "count": len(rows),

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
    Write final sensitivity report.
    """

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "FAA-AirComp EPSILON SENSITIVITY EXPERIMENT"
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
        f"Epsilon values = {EPSILON_VALUES}"
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

    # -------------------------------------------------------------------------
    # Mathematical validity
    # -------------------------------------------------------------------------

    lines.append(
        "NON-DEGENERACY VALIDATION"
    )

    lines.append(
        "-" * 78
    )

    for K in K_VALUES:

        threshold = 1.0 / K

        lines.append(
            f"K={K:2d} | 1/K={threshold:.9f} | "
            f"max epsilon={max(EPSILON_VALUES):.5f} | "
            f"non-degenerate=YES"
        )

    lines.append("")

    # -------------------------------------------------------------------------
    # Global validation
    # -------------------------------------------------------------------------

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
    # Per epsilon / K / N
    # -------------------------------------------------------------------------

    for epsilon in EPSILON_VALUES:

        lines.append(
            "=" * 78
        )

        lines.append(
            f"EPSILON = {epsilon:.2f}"
        )

        lines.append(
            "=" * 78
        )

        for K in K_VALUES:

            for N in N_VALUES:

                cell = [
                    row
                    for row in rows
                    if np.isclose(
                        row["epsilon"],
                        epsilon,
                        atol=0.0,
                        rtol=0.0,
                    )
                    and row["K"] == K
                    and row["N"] == N
                ]

                if len(cell) != NUM_REALIZATIONS:
                    continue

                stats = compute_statistics(
                    cell
                )

                lines.append("")

                lines.append(
                    f"K = {K}, N = {N}"
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
    # Epsilon-level aggregate
    # -------------------------------------------------------------------------

    lines.append(
        "=" * 78
    )

    lines.append(
        "EPSILON-LEVEL AGGREGATE RESULTS"
    )

    lines.append(
        "=" * 78
    )

    for epsilon in EPSILON_VALUES:

        epsilon_rows = [
            row
            for row in rows
            if np.isclose(
                row["epsilon"],
                epsilon,
                atol=0.0,
                rtol=0.0,
            )
        ]

        if not epsilon_rows:
            continue

        stats = compute_statistics(
            epsilon_rows
        )

        lines.append("")

        lines.append(
            f"Epsilon = {epsilon:.2f}"
        )

        lines.append(
            f"Trials                  : "
            f"{len(epsilon_rows)}"
        )

        lines.append(
            f"Mean FAA energy         : "
            f"{stats['faa_mean']:.9e} W"
        )

        lines.append(
            f"Mean FPA energy         : "
            f"{stats['fpa_mean']:.9e} W"
        )

        lines.append(
            f"Mean FAA saving         : "
            f"{stats['saving_mean']:.4f}%"
        )

        lines.append(
            f"FAA win rate            : "
            f"{stats['faa_win_rate']:.2f}%"
        )

        lines.append(
            f"Mean paired difference  : "
            f"{stats['mean_difference']:.9e} W"
        )

        lines.append(
            f"95% CI difference       : "
            f"[{stats['ci_low']:.9e}, "
            f"{stats['ci_high']:.9e}] W"
        )

        lines.append(
            f"Wilcoxon p              : "
            f"{stats['wilcoxon_p']:.6e}"
        )

        lines.append(
            f"Paired t-test p         : "
            f"{stats['paired_t_p']:.6e}"
        )

    # -------------------------------------------------------------------------
    # Global aggregate
    # -------------------------------------------------------------------------

    if rows:

        overall = compute_statistics(
            rows
        )

        lines.append("")

        lines.append(
            "=" * 78
        )

        lines.append(
            "GLOBAL AGGREGATE"
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

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    lines.append("")

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
        "This experiment tests sensitivity to epsilon."
    )

    lines.append(
        "The same physical channel is reused across "
        "epsilon values for each realization."
    )

    lines.append(
        "The same g and phi are reused across epsilon "
        "values for each realization."
    )

    lines.append(
        "FAA and FPA are paired on the same physical channel "
        "at every epsilon."
    )

    lines.append(
        "No zero-power-degenerate (K, epsilon) combination "
        "is included."
    )

    lines.append(
        "The BCD implementation is not modified."
    )

    lines.append(
        "The existing scaling experiment is not modified."
    )

    lines.append("")

    lines.append(
        "The strongest sensitivity evidence is:"
    )

    lines.append(
        "  1. FAA remains lower-energy than FPA across epsilon."
    )

    lines.append(
        "  2. FAA maintains a high paired win rate."
    )

    lines.append(
        "  3. Confidence intervals remain separated from zero."
    )

    lines.append(
        "  4. Statistical significance persists."
    )

    lines.append(
        "  5. No unexplained feasibility failures occur."
    )

    lines.append("")

    lines.append(
        "These results establish empirical sensitivity/robustness."
    )

    lines.append(
        "They do NOT establish a theoretical scaling law."
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

    # -------------------------------------------------------------------------
    # Validate experiment before doing ANY expensive computation.
    # -------------------------------------------------------------------------

    validate_experiment_grid()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = SystemConfig()

    start = time.perf_counter()

    configs = [
        (
            epsilon,
            K,
            N,
        )
        for epsilon in EPSILON_VALUES
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
        "FAA-AirComp EPSILON SENSITIVITY EXPERIMENT"
    )

    print(
        "=" * 78
    )

    print(
        f"Epsilon values    : {EPSILON_VALUES}"
    )

    print(
        f"K values          : {K_VALUES}"
    )

    print(
        f"N values          : {N_VALUES}"
    )

    print(
        f"Realizations/cell : {NUM_REALIZATIONS}"
    )

    print(
        f"Total paired trials: {total}"
    )

    print(
        f"Checkpoint rows   : "
        f"{len(done)}/{total}"
    )

    print(
        f"CSV checkpoint    : "
        f"{CSV_PATH}"
    )

    print()

    # -------------------------------------------------------------------------
    # Explicit mathematical validation output.
    # -------------------------------------------------------------------------

    print(
        "NON-DEGENERACY CHECK"
    )

    print(
        "-" * 78
    )

    for K in K_VALUES:

        threshold = 1.0 / K

        print(
            f"K={K:2d} | "
            f"1/K={threshold:.9f} | "
            f"tested epsilon <= {max(EPSILON_VALUES):.2f} | "
            f"SAFE"
        )

    print()

    write_progress(
        len(done),
        total,
        start,
    )

    # -------------------------------------------------------------------------
    # Resume complete experiment.
    # -------------------------------------------------------------------------

    if len(done) == total:

        print(
            "[RESUME] All sensitivity realizations "
            "already completed."
        )

        write_summary(
            typed_rows(done),
            start,
            total,
        )

        return

    # =========================================================================
    # MAIN EXPERIMENT
    # =========================================================================

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We loop by K/N/realization first, and epsilon inside.
    #
    # This guarantees that ONE physical channel is generated and reused
    # for all epsilon values for that realization.
    # -------------------------------------------------------------------------

    base_configs = [
        (K, N)
        for K in K_VALUES
        for N in N_VALUES
    ]

    for config_index, (K, N) in enumerate(
        base_configs,
        start=1,
    ):

        print()

        print(
            f"BASE CONFIGURATION "
            f"{config_index}/{len(base_configs)} "
            f"-> K={K}, N={N}"
        )

        for realization in range(
            1,
            NUM_REALIZATIONS + 1,
        ):

            # -----------------------------------------------------------------
            # One deterministic channel seed per physical realization.
            #
            # The same channel seed is used for every epsilon.
            # -----------------------------------------------------------------

            channel_seed = (
                BASE_SEED
                + 100000 * K
                + 1000 * N
                + (
                    realization - 1
                )
            )

            # -----------------------------------------------------------------
            # Generate ONE channel.
            # -----------------------------------------------------------------

            channel_rng = (
                np.random.default_rng(
                    channel_seed
                )
            )

            try:

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
                # CHANNEL VALIDATION
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
                # EPSILON SWEEP
                # =============================================================

                for epsilon_index, epsilon in enumerate(
                    EPSILON_VALUES
                ):

                    key = (
                        epsilon,
                        K,
                        N,
                        realization,
                    )

                    # ---------------------------------------------------------
                    # Checkpoint resume.
                    # ---------------------------------------------------------

                    if key in done:
                        continue

                    # ---------------------------------------------------------
                    # Independent RNG streams for FAA/FPA.
                    #
                    # They are deterministic and independent.
                    # ---------------------------------------------------------

                    faa_seed = (
                        BASE_SEED
                        + 10000000
                        + 1000000 * int(
                            round(
                                epsilon * 100
                            )
                        )
                        + 100000 * K
                        + 1000 * N
                        + (
                            realization - 1
                        )
                    )

                    fpa_seed = (
                        BASE_SEED
                        + 20000000
                        + 1000000 * int(
                            round(
                                epsilon * 100
                            )
                        )
                        + 100000 * K
                        + 1000 * N
                        + (
                            realization - 1
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

                    # =========================================================
                    # FAA
                    # =========================================================

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
                        eps=epsilon,
                        do_apv=True,
                        C_apv=cfg.C_apv,
                        return_history=True,
                        g=g.copy(),
                        phi=phi.copy(),
                    )

                    # =========================================================
                    # FPA
                    #
                    # SAME H.
                    #
                    # No APV.
                    # =========================================================

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
                        eps=epsilon,
                        do_apv=False,
                        C_apv=None,
                        return_history=True,
                        g=None,
                        phi=None,
                    )

                    # =========================================================
                    # RESULT VALIDATION
                    # =========================================================

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

                    # =========================================================
                    # PAIRED ENERGY COMPARISON
                    # =========================================================

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

                    # =========================================================
                    # RECORD
                    # =========================================================

                    row = {
                        "epsilon": epsilon,

                        "K": K,

                        "N": N,

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

                        "same_g_phi": 1,
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

                    # ---------------------------------------------------------
                    # Console progress
                    # ---------------------------------------------------------

                    if (
                        epsilon_index
                        == len(EPSILON_VALUES) - 1
                    ):

                        print(
                            f"  realization "
                            f"{realization:3d}/"
                            f"{NUM_REALIZATIONS} "
                            f"| epsilon sweep complete "
                            f"| global "
                            f"{len(done):4d}/"
                            f"{total}"
                        )

            # =================================================================
            # FAILURE
            # =================================================================

            except Exception as exc:

                # -------------------------------------------------------------
                # Determine which epsilon failed.
                # If failure occurred during channel generation, record all
                # remaining epsilon values as failed only for diagnostics.
                # -------------------------------------------------------------

                current_epsilon = EPSILON_VALUES[0]

                try:
                    # If the exception occurred after entering the epsilon
                    # loop, current_epsilon will have been updated.
                    current_epsilon = epsilon
                except NameError:
                    pass

                append_failure(
                    current_epsilon,
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
                    "SENSITIVITY EXPERIMENT STOPPED"
                )

                print(
                    "=" * 78
                )

                print(
                    f"Epsilon       : {current_epsilon}"
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

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

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