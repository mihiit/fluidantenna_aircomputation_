"""
FAA-AirComp Scaling Experiment
==============================

Purpose
-------
Evaluate empirical FAA energy scaling with respect to:

    K = number of devices
    N = number of FAA ports

Experiment
----------
epsilon = 0.05
K = [4, 8, 12, 16]
N = [4, 6, 8]
500 realizations per configuration

Total expected realizations:

    4 * 3 * 500 = 6000

IMPORTANT
---------
This script is CHECKPOINTED and RESUMABLE.

Every successful realization is immediately written to CSV.

If execution stops or crashes, rerunning this script will:

    - load the existing CSV
    - verify that it belongs to this exact experiment
    - identify completed (K,N,realization) tuples
    - skip completed realizations
    - continue from the first missing realization

IMPORTANT NUMERICAL POLICY
--------------------------
The experiment does NOT silently convert zero or negative
energies into successful observations.

A physically meaningful zero-power solution is treated as a
degenerate numerical/optimization outcome and is NOT included
as a successful positive-energy observation.

Small negative values are NOT automatically floored into valid
energy measurements.

The experiment therefore stops on non-positive energy.

COMPARISON
----------
FAA and FPA are evaluated on the SAME physical channel H.

The deterministic solver seeds are also fixed per realization,
making the experiment reproducible.

SCALING LAW
-----------
After all 6000 realizations are completed, the script fits:

    E_FAA = C * K^alpha * N^beta

using the mean FAA energy for each of the 12 (K,N)
configurations.

The fitted alpha and beta are empirical scaling exponents,
NOT theoretical complexity exponents.

EXISTING PROJECT FILES
----------------------
This script does NOT modify:

    src/channel.py
    src/bcd.py
    src/config.py
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

EPSILON = 0.05

K_VALUES = [4, 8, 12, 16]

N_VALUES = [4, 6, 8]

NUM_REALIZATIONS = 500

BASE_SEED = 20260820


# =============================================================================
# OUTPUT FILES
# =============================================================================
#
# IMPORTANT:
# These filenames are intentionally different from the old epsilon=0.08
# experiment so that the old checkpoint cannot accidentally be reused.
#

OUTPUT_DIR = PROJECT_ROOT / "output"

CSV_PATH = OUTPUT_DIR / "scaling_experiment_eps005_500.csv"

SUMMARY_PATH = OUTPUT_DIR / "scaling_experiment_eps005_summary.txt"

FAILURE_PATH = OUTPUT_DIR / "scaling_experiment_eps005_failures.csv"


# =============================================================================
# NUMERICAL SETTINGS
# =============================================================================

MONO_TOL = 1e-9

WIN_TOL = 1e-12

LOG_FLOOR_W = 1e-15

# IMPORTANT:
# No negative energy is silently converted into a valid positive value.
#
# This tolerance is only used to report whether a negative result is
# extremely close to floating-point zero. The result is STILL rejected.
ENERGY_ZERO_TOL = 1e-12


# =============================================================================
# CSV SCHEMA
# =============================================================================

FIELDNAMES = [
    "K",
    "N",
    "epsilon",
    "realization",
    "channel_seed",
    "faa_seed",
    "fpa_seed",

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


FAILURE_FIELDNAMES = [
    "K",
    "N",
    "epsilon",
    "realization",
    "channel_seed",
    "error_type",
    "error_message",
]


# =============================================================================
# HELPERS
# =============================================================================

def run_silent(func, *args, **kwargs):
    """
    Execute the existing BCD implementation while suppressing
    diagnostic stdout.
    """

    with open(os.devnull, "w") as devnull:

        with redirect_stdout(devnull):

            return func(
                *args,
                **kwargs
            )


# =============================================================================
# MONOTONICITY
# =============================================================================

def is_monotone_nonincreasing(history):
    """
    Check whether an optimization history is monotonically
    non-increasing within MONO_TOL.
    """

    history = list(history)

    if len(history) <= 1:
        return True

    return all(
        history[i]
        >= history[i + 1] - MONO_TOL
        for i in range(len(history) - 1)
    )


# =============================================================================
# ENERGY CONVERSION
# =============================================================================

def energy_to_dbm(energy_w):
    """
    Convert positive power in watts to dBm.

    LOG_FLOOR_W only protects the logarithm. It does NOT make an
    invalid optimization result valid because energy validation
    occurs before this function is called.
    """

    return (
        10.0
        * np.log10(
            max(
                float(energy_w),
                LOG_FLOOR_W,
            )
            * 1000.0
        )
    )


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================

def confidence_interval_95(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n < 2:

        return (
            float("nan"),
            float("nan"),
        )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(
            values,
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
            df=n - 1,
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


# =============================================================================
# WILCOXON
# =============================================================================

def safe_wilcoxon(x, y):
    """
    Safe paired Wilcoxon test.

    x and y must correspond to the same physical realization order.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    difference = y - x

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


# =============================================================================
# CONFIGURATION FILTER
# =============================================================================

def get_configuration_rows(rows, K, N):

    return [
        row
        for row in rows
        if (
            row["K"] == K
            and row["N"] == N
        )
    ]


# =============================================================================
# SCALING FIT
# =============================================================================

def fit_power_law(rows):
    """
    Fit:

        E_FAA = C * K^alpha * N^beta

    using configuration-level mean FAA energies.

    There are 12 configurations:

        K = 4,8,12,16
        N = 4,6,8

    Each configuration contributes exactly one point to the
    scaling regression, avoiding overweighting any particular
    configuration simply because of realization-level noise.
    """

    config_points = []

    for K in K_VALUES:

        for N in N_VALUES:

            config_rows = get_configuration_rows(
                rows,
                K,
                N,
            )

            if len(config_rows) != NUM_REALIZATIONS:

                raise RuntimeError(
                    "Scaling fit requires exactly "
                    f"{NUM_REALIZATIONS} realizations for "
                    f"K={K}, N={N}. "
                    f"Found {len(config_rows)}."
                )

            energies = np.asarray(
                [
                    row["faa_energy_W"]
                    for row in config_rows
                ],
                dtype=float,
            )

            if np.any(
                ~np.isfinite(energies)
            ):

                raise RuntimeError(
                    f"Scaling fit encountered non-finite "
                    f"FAA energy for K={K}, N={N}."
                )

            if np.any(
                energies <= 0
            ):

                raise RuntimeError(
                    f"Scaling fit encountered non-positive "
                    f"FAA energy for K={K}, N={N}."
                )

            mean_energy = float(
                np.mean(energies)
            )

            config_points.append(
                (
                    K,
                    N,
                    mean_energy,
                )
            )

    K = np.asarray(
        [
            point[0]
            for point in config_points
        ],
        dtype=float,
    )

    N = np.asarray(
        [
            point[1]
            for point in config_points
        ],
        dtype=float,
    )

    E = np.asarray(
        [
            point[2]
            for point in config_points
        ],
        dtype=float,
    )

    if np.any(E <= 0):

        raise RuntimeError(
            "Scaling fit encountered non-positive "
            "configuration mean FAA energy."
        )

    X = np.column_stack(
        [
            np.ones(len(E)),
            np.log(K),
            np.log(N),
        ]
    )

    y = np.log(E)

    coefficients, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    log_C = float(
        coefficients[0]
    )

    alpha = float(
        coefficients[1]
    )

    beta = float(
        coefficients[2]
    )

    y_hat = X @ coefficients

    ss_res = float(
        np.sum(
            (y - y_hat) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    if ss_tot <= 0:

        r_squared = float("nan")

    else:

        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    C = float(
        np.exp(log_C)
    )

    return (
        C,
        alpha,
        beta,
        r_squared,
        config_points,
    )


# =============================================================================
# LOAD EXISTING CHECKPOINT
# =============================================================================

def load_existing_rows():

    rows = []

    completed = set()

    if not CSV_PATH.exists():

        return rows, completed

    print()
    print("CHECKPOINT")
    print("-" * 78)

    print(
        f"Existing CSV found: {CSV_PATH}"
    )

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:

            raise RuntimeError(
                "Checkpoint CSV has no header."
            )

        missing_fields = [
            field
            for field in FIELDNAMES
            if field not in reader.fieldnames
        ]

        if missing_fields:

            raise RuntimeError(
                "Checkpoint CSV is incompatible with "
                "the current experiment.\n"
                f"Missing fields: {missing_fields}"
            )

        for raw in reader:

            row = {
                "K": int(raw["K"]),

                "N": int(raw["N"]),

                "epsilon": float(
                    raw["epsilon"]
                ),

                "realization": int(
                    raw["realization"]
                ),

                "channel_seed": int(
                    raw["channel_seed"]
                ),

                "faa_seed": int(
                    raw["faa_seed"]
                ),

                "fpa_seed": int(
                    raw["fpa_seed"]
                ),

                "faa_energy_W": float(
                    raw["faa_energy_W"]
                ),

                "fpa_energy_W": float(
                    raw["fpa_energy_W"]
                ),

                "faa_energy_dBm": float(
                    raw["faa_energy_dBm"]
                ),

                "fpa_energy_dBm": float(
                    raw["fpa_energy_dBm"]
                ),

                "faa_energy_saving_pct": float(
                    raw["faa_energy_saving_pct"]
                ),

                "faa_iterations": int(
                    raw["faa_iterations"]
                ),

                "fpa_iterations": int(
                    raw["fpa_iterations"]
                ),

                "faa_converged": int(
                    raw["faa_converged"]
                ),

                "fpa_converged": int(
                    raw["fpa_converged"]
                ),

                "faa_monotone": int(
                    raw["faa_monotone"]
                ),

                "fpa_monotone": int(
                    raw["fpa_monotone"]
                ),

                "faa_wins": int(
                    raw["faa_wins"]
                ),

                "fpa_wins": int(
                    raw["fpa_wins"]
                ),

                "tie": int(
                    raw["tie"]
                ),

                "same_physical_channel": int(
                    raw["same_physical_channel"]
                ),
            }

            # -------------------------------------------------------------
            # CHECKPOINT CONFIGURATION CONSISTENCY
            # -------------------------------------------------------------

            if not np.isclose(
                row["epsilon"],
                EPSILON,
                atol=0.0,
                rtol=0.0,
            ):

                raise RuntimeError(
                    "Existing checkpoint belongs to a different "
                    "epsilon.\n"
                    f"Checkpoint epsilon = {row['epsilon']}\n"
                    f"Current epsilon    = {EPSILON}\n"
                    f"CSV                = {CSV_PATH}"
                )

            if row["K"] not in K_VALUES:

                raise RuntimeError(
                    f"Checkpoint contains unsupported K="
                    f"{row['K']}."
                )

            if row["N"] not in N_VALUES:

                raise RuntimeError(
                    f"Checkpoint contains unsupported N="
                    f"{row['N']}."
                )

            if not (
                1
                <= row["realization"]
                <= NUM_REALIZATIONS
            ):

                raise RuntimeError(
                    "Checkpoint contains invalid realization "
                    f"number: {row['realization']}"
                )

            key = (
                row["K"],
                row["N"],
                row["realization"],
            )

            if key in completed:

                raise RuntimeError(
                    "Duplicate checkpoint row detected for "
                    f"K={row['K']}, "
                    f"N={row['N']}, "
                    f"realization={row['realization']}."
                )

            # -------------------------------------------------------------
            # CHECK STORED ENERGY
            # -------------------------------------------------------------

            if (
                not np.isfinite(
                    row["faa_energy_W"]
                )
                or row["faa_energy_W"] <= 0
            ):

                raise RuntimeError(
                    "Checkpoint contains invalid FAA energy "
                    f"for {key}: "
                    f"{row['faa_energy_W']}"
                )

            if (
                not np.isfinite(
                    row["fpa_energy_W"]
                )
                or row["fpa_energy_W"] <= 0
            ):

                raise RuntimeError(
                    "Checkpoint contains invalid FPA energy "
                    f"for {key}: "
                    f"{row['fpa_energy_W']}"
                )

            rows.append(row)

            completed.add(key)

    print(
        f"Recovered rows       : {len(rows)}"
    )

    print(
        f"Completed realizations: "
        f"{len(completed)}"
    )

    return rows, completed


# =============================================================================
# CHECKPOINT SAVE
# =============================================================================

def append_row(row):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(row)

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = FAILURE_PATH.exists()

    with FAILURE_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FAILURE_FIELDNAMES,
        )

        if not file_exists:

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
# CONFIGURATION SUMMARY
# =============================================================================

def calculate_configuration_statistics(
    config_rows,
):

    faa_energy = np.asarray(
        [
            row["faa_energy_W"]
            for row in config_rows
        ],
        dtype=float,
    )

    fpa_energy = np.asarray(
        [
            row["fpa_energy_W"]
            for row in config_rows
        ],
        dtype=float,
    )

    saving = np.asarray(
        [
            row["faa_energy_saving_pct"]
            for row in config_rows
        ],
        dtype=float,
    )

    faa_iterations = np.asarray(
        [
            row["faa_iterations"]
            for row in config_rows
        ],
        dtype=float,
    )

    fpa_iterations = np.asarray(
        [
            row["fpa_iterations"]
            for row in config_rows
        ],
        dtype=float,
    )

    difference = (
        fpa_energy
        - faa_energy
    )

    ci_low, ci_high = (
        confidence_interval_95(
            difference
        )
    )

    _, t_p = ttest_rel(
        fpa_energy,
        faa_energy,
    )

    _, w_p = safe_wilcoxon(
        faa_energy,
        fpa_energy,
    )

    faa_wins = int(
        np.sum(
            faa_energy
            < fpa_energy - WIN_TOL
        )
    )

    fpa_wins = int(
        np.sum(
            fpa_energy
            < faa_energy - WIN_TOL
        )
    )

    ties = (
        len(config_rows)
        - faa_wins
        - fpa_wins
    )

    return {
        "faa_energy": faa_energy,
        "fpa_energy": fpa_energy,
        "saving": saving,
        "faa_iterations": faa_iterations,
        "fpa_iterations": fpa_iterations,
        "difference": difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_p": t_p,
        "w_p": w_p,
        "faa_wins": faa_wins,
        "fpa_wins": fpa_wins,
        "ties": ties,
    }


def print_configuration_summary(
    rows,
    K,
    N,
):

    config_rows = get_configuration_rows(
        rows,
        K,
        N,
    )

    if len(config_rows) == 0:

        return

    stats = calculate_configuration_statistics(
        config_rows
    )

    faa_energy = stats["faa_energy"]

    fpa_energy = stats["fpa_energy"]

    saving = stats["saving"]

    faa_iterations = stats["faa_iterations"]

    fpa_iterations = stats["fpa_iterations"]

    print()
    print(
        f"CONFIGURATION SUMMARY "
        f"K={K}, N={N}"
    )

    print("-" * 78)

    print(
        f"Rows completed       : "
        f"{len(config_rows)}/"
        f"{NUM_REALIZATIONS}"
    )

    print(
        f"FAA + PC mean energy : "
        f"{np.mean(faa_energy):.9e} W"
    )

    print(
        f"FPA + PC mean energy : "
        f"{np.mean(fpa_energy):.9e} W"
    )

    print(
        f"Mean saving          : "
        f"{np.mean(saving):.4f}%"
    )

    print(
        f"Median saving        : "
        f"{np.median(saving):.4f}%"
    )

    print(
        f"FAA wins             : "
        f"{stats['faa_wins']}/"
        f"{len(config_rows)}"
    )

    print(
        f"FPA wins             : "
        f"{stats['fpa_wins']}/"
        f"{len(config_rows)}"
    )

    print(
        f"Ties                 : "
        f"{stats['ties']}/"
        f"{len(config_rows)}"
    )

    print(
        f"FAA mean iterations  : "
        f"{np.mean(faa_iterations):.3f}"
    )

    print(
        f"FPA mean iterations  : "
        f"{np.mean(fpa_iterations):.3f}"
    )

    print(
        f"FAA monotone         : "
        f"{np.mean([r['faa_monotone'] for r in config_rows]) * 100:.2f}%"
    )

    print(
        f"FPA monotone         : "
        f"{np.mean([r['fpa_monotone'] for r in config_rows]) * 100:.2f}%"
    )

    print(
        f"FAA converged        : "
        f"{np.mean([r['faa_converged'] for r in config_rows]) * 100:.2f}%"
    )

    print(
        f"FPA converged        : "
        f"{np.mean([r['fpa_converged'] for r in config_rows]) * 100:.2f}%"
    )

    print(
        f"95% CI difference    : "
        f"[{stats['ci_low']:.9e}, "
        f"{stats['ci_high']:.9e}] W"
    )

    print(
        f"Wilcoxon p           : "
        f"{stats['w_p']:.6e}"
    )

    print(
        f"Paired t p           : "
        f"{stats['t_p']:.6e}"
    )


# =============================================================================
# SUMMARY FILE
# =============================================================================

def write_summary(rows, cfg):

    expected = (
        len(K_VALUES)
        * len(N_VALUES)
        * NUM_REALIZATIONS
    )

    completed = len(rows)

    summary = []

    summary.append("=" * 78)

    summary.append(
        "FAA-AirComp SCALING EXPERIMENT"
    )

    summary.append("=" * 78)

    summary.append("")

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    summary.append(
        "CONFIGURATION"
    )

    summary.append("-" * 78)

    summary.append(
        f"Epsilon = {EPSILON}"
    )

    summary.append(
        f"K values = {K_VALUES}"
    )

    summary.append(
        f"N values = {N_VALUES}"
    )

    summary.append(
        f"Realizations per configuration = "
        f"{NUM_REALIZATIONS}"
    )

    summary.append(
        f"Expected rows = {expected}"
    )

    summary.append(
        f"Completed rows = {completed}"
    )

    summary.append(
        f"Base seed = {BASE_SEED}"
    )

    summary.append("")

    # -------------------------------------------------------------------------
    # System
    # -------------------------------------------------------------------------

    summary.append(
        "SYSTEM"
    )

    summary.append("-" * 78)

    summary.append(
        f"Pmax = {cfg.Pmax} W"
    )

    summary.append(
        f"sigma2 = {cfg.sigma2}"
    )

    summary.append(
        f"rho = {cfg.rho}"
    )

    summary.append(
        f"BCD tolerance = {cfg.bcd_tol}"
    )

    summary.append(
        f"BCD max iterations = "
        f"{cfg.bcd_max_iter}"
    )

    summary.append(
        f"C_apv = {cfg.C_apv}"
    )

    summary.append("")

    # -------------------------------------------------------------------------
    # Per configuration
    # -------------------------------------------------------------------------

    for K in K_VALUES:

        for N in N_VALUES:

            config_rows = get_configuration_rows(
                rows,
                K,
                N,
            )

            summary.append("=" * 78)

            summary.append(
                f"K = {K}, N = {N}"
            )

            summary.append("=" * 78)

            summary.append(
                f"Rows = "
                f"{len(config_rows)}/"
                f"{NUM_REALIZATIONS}"
            )

            if len(config_rows) == 0:

                summary.append(
                    "No completed observations."
                )

                continue

            stats = calculate_configuration_statistics(
                config_rows
            )

            faa = stats["faa_energy"]

            fpa = stats["fpa_energy"]

            saving = stats["saving"]

            summary.append(
                f"FAA mean energy = "
                f"{np.mean(faa):.9e} W"
            )

            summary.append(
                f"FPA mean energy = "
                f"{np.mean(fpa):.9e} W"
            )

            summary.append(
                f"FAA median energy = "
                f"{np.median(faa):.9e} W"
            )

            summary.append(
                f"FPA median energy = "
                f"{np.median(fpa):.9e} W"
            )

            summary.append(
                f"Mean saving = "
                f"{np.mean(saving):.4f}%"
            )

            summary.append(
                f"Median saving = "
                f"{np.median(saving):.4f}%"
            )

            summary.append(
                f"FAA wins = "
                f"{stats['faa_wins']}/"
                f"{len(config_rows)}"
            )

            summary.append(
                f"FPA wins = "
                f"{stats['fpa_wins']}/"
                f"{len(config_rows)}"
            )

            summary.append(
                f"Ties = "
                f"{stats['ties']}/"
                f"{len(config_rows)}"
            )

            summary.append(
                f"FAA mean iterations = "
                f"{np.mean([r['faa_iterations'] for r in config_rows]):.3f}"
            )

            summary.append(
                f"FPA mean iterations = "
                f"{np.mean([r['fpa_iterations'] for r in config_rows]):.3f}"
            )

            summary.append(
                f"FAA monotone = "
                f"{np.mean([r['faa_monotone'] for r in config_rows]) * 100:.2f}%"
            )

            summary.append(
                f"FPA monotone = "
                f"{np.mean([r['fpa_monotone'] for r in config_rows]) * 100:.2f}%"
            )

            summary.append(
                f"FAA converged = "
                f"{np.mean([r['faa_converged'] for r in config_rows]) * 100:.2f}%"
            )

            summary.append(
                f"FPA converged = "
                f"{np.mean([r['fpa_converged'] for r in config_rows]) * 100:.2f}%"
            )

            summary.append(
                f"95% CI difference = "
                f"[{stats['ci_low']:.9e}, "
                f"{stats['ci_high']:.9e}] W"
            )

            summary.append(
                f"Wilcoxon p = "
                f"{stats['w_p']:.6e}"
            )

            summary.append(
                f"Paired t p = "
                f"{stats['t_p']:.6e}"
            )

    # -------------------------------------------------------------------------
    # Scaling fit
    # -------------------------------------------------------------------------

    summary.append("")

    summary.append("=" * 78)

    summary.append(
        "EMPIRICAL FAA ENERGY SCALING"
    )

    summary.append("=" * 78)

    if completed == expected:

        (
            C,
            alpha,
            beta,
            r_squared,
            config_points,
        ) = fit_power_law(rows)

        summary.append(
            "Model:"
        )

        summary.append(
            "E_FAA = C * K^alpha * N^beta"
        )

        summary.append("")

        summary.append(
            "Configuration mean energies:"
        )

        for K, N, E in config_points:

            summary.append(
                f"K={K:2d}, N={N:2d} -> "
                f"{E:.9e} W"
            )

        summary.append("")

        summary.append(
            f"C     = {C:.9e}"
        )

        summary.append(
            f"alpha = {alpha:.6f}"
        )

        summary.append(
            f"beta  = {beta:.6f}"
        )

        summary.append(
            f"R^2   = {r_squared:.6f}"
        )

        summary.append("")

        summary.append(
            "Interpretation:"
        )

        summary.append(
            "alpha is the empirical scaling exponent "
            "with respect to K."
        )

        summary.append(
            "beta is the empirical scaling exponent "
            "with respect to N."
        )

        summary.append(
            "These are empirical scaling parameters, "
            "not theoretical computational-complexity exponents."
        )

    else:

        summary.append(
            "Scaling fit NOT performed."
        )

        summary.append(
            "Reason: experiment is incomplete."
        )

        summary.append(
            f"Completed {completed}/{expected} rows."
        )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    summary.append("")

    summary.append("=" * 78)

    summary.append(
        "FINAL VALIDATION"
    )

    summary.append("=" * 78)

    physical_channel_valid = sum(
        row["same_physical_channel"]
        for row in rows
    )

    summary.append(
        f"Physical-channel-valid rows : "
        f"{physical_channel_valid}/{expected}"
    )

    summary.append(
        f"Successful rows              : "
        f"{completed}/{expected}"
    )

    summary.append(
        f"Missing rows                 : "
        f"{expected - completed}"
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    summary.append("")

    summary.append(
        "OUTPUT"
    )

    summary.append("-" * 78)

    summary.append(
        f"CSV     : {CSV_PATH}"
    )

    summary.append(
        f"Failures: {FAILURE_PATH}"
    )

    summary.append(
        f"Summary : {SUMMARY_PATH}"
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(summary)
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

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------

    rows, completed = (
        load_existing_rows()
    )

    expected_total = (
        len(K_VALUES)
        * len(N_VALUES)
        * NUM_REALIZATIONS
    )

    print()
    print("=" * 78)

    print(
        "FAA-AirComp SCALING EXPERIMENT"
    )

    print("=" * 78)

    print()

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
        f"Realizations/cell  : "
        f"{NUM_REALIZATIONS}"
    )

    print(
        f"Expected rows      : "
        f"{expected_total}"
    )

    print(
        f"Already completed  : "
        f"{len(completed)}"
    )

    print(
        f"Remaining          : "
        f"{expected_total - len(completed)}"
    )

    print()

    print(
        f"CSV checkpoint     : "
        f"{CSV_PATH}"
    )

    print(
        f"Failure log        : "
        f"{FAILURE_PATH}"
    )

    print()

    # -------------------------------------------------------------------------
    # Configuration loop
    # -------------------------------------------------------------------------

    configurations = [
        (K, N)
        for K in K_VALUES
        for N in N_VALUES
    ]

    for config_index, (K, N) in enumerate(
        configurations,
        start=1,
    ):

        config_start = (
            time.perf_counter()
        )

        config_completed_before = sum(
            1
            for row in rows
            if (
                row["K"] == K
                and row["N"] == N
            )
        )

        print()
        print("=" * 78)

        print(
            f"CONFIGURATION "
            f"{config_index}/"
            f"{len(configurations)}"
            f" -> K={K}, N={N}"
        )

        print("=" * 78)

        print(
            f"Existing completed rows: "
            f"{config_completed_before}/"
            f"{NUM_REALIZATIONS}"
        )

        for realization in range(
            1,
            NUM_REALIZATIONS + 1,
        ):

            key = (
                K,
                N,
                realization,
            )

            # -------------------------------------------------------------
            # RESUME LOGIC
            # -------------------------------------------------------------

            if key in completed:

                continue

            # -------------------------------------------------------------
            # Deterministic seeds
            # -------------------------------------------------------------

            channel_seed = (
                BASE_SEED
                + 100_000 * K
                + 1_000 * N
                + (realization - 1)
            )

            faa_seed = (
                BASE_SEED
                + 10_000_000
                + 100_000 * K
                + 1_000 * N
                + (realization - 1)
            )

            fpa_seed = (
                BASE_SEED
                + 20_000_000
                + 100_000 * K
                + 1_000 * N
                + (realization - 1)
            )

            channel_rng = np.random.default_rng(
                channel_seed
            )

            faa_rng = np.random.default_rng(
                faa_seed
            )

            fpa_rng = np.random.default_rng(
                fpa_seed
            )

            try:

                # =========================================================
                # ONE PHYSICAL CHANNEL
                # =========================================================

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

                # =========================================================
                # CHANNEL VALIDATION
                # =========================================================

                if H.shape != (N, K):

                    raise RuntimeError(
                        f"Invalid H shape: "
                        f"expected {(N, K)}, "
                        f"got {H.shape}"
                    )

                if bk.shape != (K,):

                    raise RuntimeError(
                        f"Invalid bk shape: "
                        f"expected {(K,)}, "
                        f"got {bk.shape}"
                    )

                if g.shape[0] != K:

                    raise RuntimeError(
                        f"Invalid g dimension: "
                        f"expected {K}, "
                        f"got {g.shape[0]}"
                    )

                if phi.shape[0] != K:

                    raise RuntimeError(
                        f"Invalid phi dimension: "
                        f"expected {K}, "
                        f"got {phi.shape[0]}"
                    )

                # =========================================================
                # FAA + PC
                # =========================================================

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
                    eps=EPSILON,
                    do_apv=True,
                    C_apv=cfg.C_apv,
                    return_history=True,
                    g=g.copy(),
                    phi=phi.copy(),
                )

                # =========================================================
                # FPA + PC
                # =========================================================

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
                    eps=EPSILON,
                    do_apv=False,
                    C_apv=None,
                    return_history=True,
                    g=None,
                    phi=None,
                )

                E_faa = float(
                    E_faa
                )

                E_fpa = float(
                    E_fpa
                )

                history_faa = list(
                    history_faa
                )

                history_fpa = list(
                    history_fpa
                )

                # =========================================================
                # ENERGY VALIDATION
                # =========================================================

                if not np.isfinite(
                    E_faa
                ):

                    raise RuntimeError(
                        f"FAA returned non-finite "
                        f"energy: {E_faa}"
                    )

                if not np.isfinite(
                    E_fpa
                ):

                    raise RuntimeError(
                        f"FPA returned non-finite "
                        f"energy: {E_fpa}"
                    )

                # ---------------------------------------------------------
                # DO NOT HIDE ZERO/NEGATIVE SOLUTIONS
                # ---------------------------------------------------------
                #
                # A zero-power solution is not converted to a small
                # positive number.
                #
                # A negative result, even if extremely small, is not
                # silently converted to zero and accepted.
                #

                if E_faa <= 0.0:

                    if abs(E_faa) <= ENERGY_ZERO_TOL:

                        raise RuntimeError(
                            "FAA returned zero/near-zero "
                            "energy. This realization is treated "
                            "as a degenerate optimization outcome "
                            "and is NOT accepted as a successful "
                            f"positive-energy observation: {E_faa}"
                        )

                    raise RuntimeError(
                        "FAA returned negative energy: "
                        f"{E_faa}"
                    )

                if E_fpa <= 0.0:

                    if abs(E_fpa) <= ENERGY_ZERO_TOL:

                        raise RuntimeError(
                            "FPA returned zero/near-zero "
                            "energy. This realization is treated "
                            "as a degenerate optimization outcome "
                            "and is NOT accepted as a successful "
                            f"positive-energy observation: {E_fpa}"
                        )

                    raise RuntimeError(
                        "FPA returned negative energy: "
                        f"{E_fpa}"
                    )

                # =========================================================
                # HISTORY VALIDATION
                # =========================================================

                if len(history_faa) == 0:

                    raise RuntimeError(
                        "FAA returned empty "
                        "convergence history."
                    )

                if len(history_fpa) == 0:

                    raise RuntimeError(
                        "FPA returned empty "
                        "convergence history."
                    )

                # =========================================================
                # ENERGY COMPARISON
                # =========================================================

                saving_pct = (
                    100.0
                    * (
                        E_fpa
                        - E_faa
                    )
                    / E_fpa
                )

                if not np.isfinite(
                    saving_pct
                ):

                    raise RuntimeError(
                        "Non-finite energy saving percentage."
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
                # CONVERGENCE
                # =========================================================

                faa_monotone = (
                    is_monotone_nonincreasing(
                        history_faa
                    )
                )

                fpa_monotone = (
                    is_monotone_nonincreasing(
                        history_fpa
                    )
                )

                #
                # Existing BCD convention:
                # reaching bcd_max_iter means the solver exhausted
                # its iteration budget.
                #
                # Therefore:
                #
                # iterations < max_iter -> converged
                # iterations >= max_iter -> not converged
                #

                faa_converged = (
                    len(history_faa)
                    < cfg.bcd_max_iter
                )

                fpa_converged = (
                    len(history_fpa)
                    < cfg.bcd_max_iter
                )

                # =========================================================
                # ROW
                # =========================================================

                row = {
                    "K": K,

                    "N": N,

                    "epsilon": EPSILON,

                    "realization": realization,

                    "channel_seed": channel_seed,

                    "faa_seed": faa_seed,

                    "fpa_seed": fpa_seed,

                    "faa_energy_W": E_faa,

                    "fpa_energy_W": E_fpa,

                    "faa_energy_dBm":
                        energy_to_dbm(
                            E_faa
                        ),

                    "fpa_energy_dBm":
                        energy_to_dbm(
                            E_fpa
                        ),

                    "faa_energy_saving_pct":
                        saving_pct,

                    "faa_iterations":
                        len(history_faa),

                    "fpa_iterations":
                        len(history_fpa),

                    "faa_converged":
                        int(
                            faa_converged
                        ),

                    "fpa_converged":
                        int(
                            fpa_converged
                        ),

                    "faa_monotone":
                        int(
                            faa_monotone
                        ),

                    "fpa_monotone":
                        int(
                            fpa_monotone
                        ),

                    "faa_wins":
                        int(
                            faa_wins
                        ),

                    "fpa_wins":
                        int(
                            fpa_wins
                        ),

                    "tie":
                        int(
                            tie
                        ),

                    "same_physical_channel":
                        1,
                }

                # =========================================================
                # CRITICAL:
                # SAVE IMMEDIATELY
                # =========================================================

                append_row(
                    row
                )

                rows.append(
                    row
                )

                completed.add(
                    key
                )

            except Exception as exc:

                append_failure(
                    K=K,
                    N=N,
                    realization=realization,
                    channel_seed=channel_seed,
                    exc=exc,
                )

                print()
                print("=" * 78)
                print(
                    "EXPERIMENT STOPPED"
                )
                print("=" * 78)

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
                    "realizations have been checkpointed."
                )

                print(
                    "The failed realization was NOT written "
                    "to the successful CSV."
                )

                print(
                    "Fix the numerical issue and "
                    "rerun the script."
                )

                raise

            # -------------------------------------------------------------
            # Progress
            # -------------------------------------------------------------

            completed_in_config = sum(
                1
                for row in rows
                if (
                    row["K"] == K
                    and row["N"] == N
                )
            )

            if (
                completed_in_config % 50 == 0
                or completed_in_config
                == NUM_REALIZATIONS
            ):

                elapsed = (
                    time.perf_counter()
                    - config_start
                )

                newly_completed = (
                    completed_in_config
                    - config_completed_before
                )

                avg_time = (
                    elapsed
                    / max(
                        newly_completed,
                        1,
                    )
                )

                remaining = (
                    NUM_REALIZATIONS
                    - completed_in_config
                )

                eta = (
                    remaining
                    * avg_time
                )

                print(
                    f"  {completed_in_config:4d}/"
                    f"{NUM_REALIZATIONS} "
                    f"completed | "
                    f"{elapsed / 60:.2f} min | "
                    f"ETA "
                    f"{eta / 60:.2f} min"
                )

        # ---------------------------------------------------------------------
        # Configuration completed
        # ---------------------------------------------------------------------

        print_configuration_summary(
            rows,
            K,
            N,
        )

        write_summary(
            rows,
            cfg,
        )

        print()
        print(
            "CHECKPOINT SAVED."
        )

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print()
    print("=" * 78)

    print(
        "EXPERIMENT COMPLETE"
    )

    print("=" * 78)

    print(
        f"Completed rows : "
        f"{len(rows)}/{expected_total}"
    )

    if len(rows) != expected_total:

        print(
            "WARNING: experiment incomplete."
        )

        print(
            "Scaling fit has NOT been performed."
        )

        write_summary(
            rows,
            cfg,
        )

        return

    # =========================================================================
    # FINAL SCALING FIT
    # =========================================================================

    (
        C,
        alpha,
        beta,
        r_squared,
        config_points,
    ) = fit_power_law(
        rows
    )

    print()
    print("=" * 78)

    print(
        "FINAL EMPIRICAL SCALING LAW"
    )

    print("=" * 78)

    print(
        "E_FAA = C * K^alpha * N^beta"
    )

    print()

    print(
        "Configuration mean energies:"
    )

    for K, N, E in config_points:

        print(
            f"  K={K:2d}, "
            f"N={N:2d} -> "
            f"{E:.9e} W"
        )

    print()

    print(
        f"C     = {C:.9e}"
    )

    print(
        f"alpha = {alpha:.6f}"
    )

    print(
        f"beta  = {beta:.6f}"
    )

    print(
        f"R^2   = {r_squared:.6f}"
    )

    print()

    print(
        "The fitted exponents are empirical "
        "scaling parameters."
    )

    print(
        "They are NOT theoretical complexity exponents."
    )

    # =========================================================================
    # FINAL SUMMARY WRITE
    # =========================================================================

    write_summary(
        rows,
        cfg,
    )

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    print()
    print(
        f"Total runtime this invocation: "
        f"{total_runtime / 60:.2f} min"
    )

    print()

    print(
        f"CSV     : {CSV_PATH}"
    )

    print(
        f"Failures: {FAILURE_PATH}"
    )

    print(
        f"Summary : {SUMMARY_PATH}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()