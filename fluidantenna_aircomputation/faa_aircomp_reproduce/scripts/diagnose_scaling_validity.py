"""
scripts/diagnose_scaling_validity.py
====================================

Mathematical validity diagnostic for the FAA-AirComp scaling grid.

Purpose
-------
The current optimization problem minimizes

    E = sum_k p_k

subject to

    MSE <= epsilon
    0 <= p_k <= Pmax.

If all transmit powers are zero, then

    MSE_zero_power = 1 / K.

Therefore, whenever

    1 / K <= epsilon,

the optimization problem admits the trivial zero-energy solution:

    p = 0
    E = 0.

This script DOES NOT modify BCD.
This script DOES NOT modify scaling_experiment.py.
This script DOES NOT run channel simulations.

It only identifies which (K, epsilon) combinations are
mathematically degenerate under the current formulation.

Paper scaling grid
------------------
K       = {4, 6, 8, 10, 12, 16, 20}
N       = {4, 6, 8}

epsilon = {0.04, 0.06, 0.08, 0.10, 0.12}

Important
---------
A point is classified as ZERO-POWER FEASIBLE when

    1/K <= epsilon.

This is a mathematical property of the current optimization
formulation, not a numerical failure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# =============================================================================
# EXPERIMENT GRID
# =============================================================================

K_VALUES = [4, 6, 8, 10, 12, 16, 20]

N_VALUES = [4, 6, 8]

EPSILON_VALUES = [
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
]

# Small numerical tolerance only for classification.
# This does NOT alter epsilon used by any experiment.
TOL = 1e-12


# =============================================================================
# MATHEMATICAL CHECK
# =============================================================================

def zero_power_mse(K: int) -> float:
    """
    MSE obtained when all transmit powers are zero.

    With p_k = 0 for every k,

        MSE = ||-(1/K) 1^H||^2
            = K * (1/K)^2
            = 1/K.
    """

    if K <= 0:
        raise ValueError(
            f"K must be positive, got {K}"
        )

    return 1.0 / float(K)


def is_zero_power_feasible(
    K: int,
    epsilon: float,
) -> bool:
    """
    Return True when the zero-power solution satisfies

        MSE_zero_power <= epsilon.
    """

    mse0 = zero_power_mse(K)

    return bool(
        mse0 <= epsilon + TOL
    )


def critical_epsilon(K: int) -> float:
    """
    Critical epsilon at which zero-power feasibility begins.

        epsilon_critical = 1/K
    """

    return zero_power_mse(K)


def critical_K(epsilon: float) -> int:
    """
    Smallest integer K for which

        1/K <= epsilon.

    Equivalently,

        K >= 1/epsilon.
    """

    return int(
        np.ceil(
            1.0 / epsilon
        )
    )


# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================

def main():

    print("=" * 78)
    print("FAA-AIRCOMP SCALING VALIDITY DIAGNOSTIC")
    print("=" * 78)

    print()
    print("This diagnostic does NOT run BCD.")
    print("This diagnostic does NOT modify epsilon.")
    print("This diagnostic does NOT modify scaling_experiment.py.")
    print()

    print("K values       :", K_VALUES)
    print("N values       :", N_VALUES)
    print("epsilon values :", EPSILON_VALUES)
    print()

    # =========================================================================
    # BASIC K THRESHOLDS
    # =========================================================================

    print("=" * 78)
    print("CRITICAL EPSILON BY K")
    print("=" * 78)

    print()
    print(
        f"{'K':>6}"
        f"{'1/K':>16}"
        f"{'Zero-power becomes feasible when epsilon >= 1/K':>52}"
    )

    print("-" * 78)

    for K in K_VALUES:

        eps_critical = critical_epsilon(K)

        print(
            f"{K:>6d}"
            f"{eps_critical:>16.9f}"
            f"{'YES':>52}"
        )

    print()

    # =========================================================================
    # EPSILON THRESHOLDS
    # =========================================================================

    print("=" * 78)
    print("CRITICAL K BY EPSILON")
    print("=" * 78)

    print()
    print(
        f"{'epsilon':>12}"
        f"{'1/epsilon':>16}"
        f"{'smallest K with zero-power feasibility':>44}"
    )

    print("-" * 78)

    for epsilon in EPSILON_VALUES:

        threshold = 1.0 / epsilon
        Kcrit = critical_K(epsilon)

        print(
            f"{epsilon:>12.5f}"
            f"{threshold:>16.6f}"
            f"{Kcrit:>44d}"
        )

    print()

    # =========================================================================
    # FULL K x EPSILON GRID
    # =========================================================================

    print("=" * 78)
    print("FULL K x EPSILON VALIDITY GRID")
    print("=" * 78)

    print()
    print(
        f"{'K':>6}"
        f"{'epsilon':>12}"
        f"{'1/K':>16}"
        f"{'zero-power feasible':>24}"
        f"{'classification':>20}"
    )

    print("-" * 78)

    results = []

    for K in K_VALUES:

        mse_zero = zero_power_mse(K)

        for epsilon in EPSILON_VALUES:

            degenerate = is_zero_power_feasible(
                K,
                epsilon,
            )

            if degenerate:
                classification = "DEGENERATE"
            else:
                classification = "NON-DEGENERATE"

            row = {
                "K": K,
                "epsilon": epsilon,
                "zero_power_mse": mse_zero,
                "zero_power_feasible": degenerate,
                "classification": classification,
            }

            results.append(row)

            print(
                f"{K:>6d}"
                f"{epsilon:>12.5f}"
                f"{mse_zero:>16.9f}"
                f"{str(degenerate):>24}"
                f"{classification:>20}"
            )

    print()

    # =========================================================================
    # PAPER GRID SUMMARY
    # =========================================================================

    print("=" * 78)
    print("DEGENERATE POINTS IN THE PAPER SCALING GRID")
    print("=" * 78)

    degenerate_results = [
        row
        for row in results
        if row["zero_power_feasible"]
    ]

    nondegenerate_results = [
        row
        for row in results
        if not row["zero_power_feasible"]
    ]

    print()

    print(
        f"Total K x epsilon combinations : "
        f"{len(results)}"
    )

    print(
        f"Non-degenerate combinations    : "
        f"{len(nondegenerate_results)}"
    )

    print(
        f"Degenerate combinations        : "
        f"{len(degenerate_results)}"
    )

    print()

    if degenerate_results:

        print("DEGENERATE:")
        print()

        for row in degenerate_results:

            print(
                f"    K={row['K']:>2d}, "
                f"epsilon={row['epsilon']:.2f}, "
                f"1/K={row['zero_power_mse']:.6f}"
            )

    else:

        print("No degenerate points found.")

    print()

    # =========================================================================
    # GROUP BY K
    # =========================================================================

    print("=" * 78)
    print("SUMMARY BY K")
    print("=" * 78)

    print()

    for K in K_VALUES:

        eps_critical = critical_epsilon(K)

        degenerate_eps = [
            epsilon
            for epsilon in EPSILON_VALUES
            if is_zero_power_feasible(
                K,
                epsilon,
            )
        ]

        valid_eps = [
            epsilon
            for epsilon in EPSILON_VALUES
            if not is_zero_power_feasible(
                K,
                epsilon,
            )
        ]

        print(
            f"K={K:>2d}"
            f" | critical epsilon = {eps_critical:.6f}"
        )

        print(
            f"    non-degenerate epsilon: "
            f"{valid_eps}"
        )

        print(
            f"    degenerate epsilon:     "
            f"{degenerate_eps}"
        )

        print()

    # =========================================================================
    # IMPORTANT CURRENT POINT
    # =========================================================================

    print("=" * 78)
    print("CURRENT EXPERIMENT: epsilon = 0.08")
    print("=" * 78)

    current_epsilon = 0.08

    print()

    print(
        f"For epsilon = {current_epsilon:.2f}:"
    )

    print()

    for K in K_VALUES:

        mse_zero = zero_power_mse(K)

        degenerate = is_zero_power_feasible(
            K,
            current_epsilon,
        )

        if degenerate:
            status = "DEGENERATE / E=0 ALLOWED"
        else:
            status = "NON-DEGENERATE"

        print(
            f"    K={K:>2d} : "
            f"1/K={mse_zero:.9f} "
            f"-> {status}"
        )

    print()

    # =========================================================================
    # N DOES NOT CHANGE THIS PARTICULAR TEST
    # =========================================================================

    print("=" * 78)
    print("IMPORTANT: N DOES NOT CHANGE THE ZERO-POWER THRESHOLD")
    print("=" * 78)

    print()

    print(
        "The zero-power MSE is 1/K under the current MSE definition."
    )

    print(
        "Therefore the degeneracy condition depends on K and epsilon,"
    )

    print(
        "not on the number of antenna ports N."
    )

    print()

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    output_dir = Path("results")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "scaling_validity_diagnostic.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "epsilon",
                "zero_power_mse",
                "zero_power_feasible",
                "classification",
            ],
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )

    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print()

    print(
        f"CSV saved to:"
    )

    print(
        f"    {output_file}"
    )

    print()

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print()

    print(
        "If zero_power_feasible=True, the current optimization"
    )

    print(
        "mathematically permits p=0 and therefore E=0."
    )

    print()

    print(
        "That is NOT a floating-point error."
    )

    print(
        "That is NOT something we should hide with an epsilon floor"
    )

    print(
        "or by replacing zero with an arbitrary tiny positive power."
    )

    print()

    print(
        "Those points must be handled explicitly before using them"
    )

    print(
        "to support a positive-energy scaling law."
    )

    print()

    print(
        "DO NOT modify bcd.py or scaling_experiment.py based only"
    )

    print(
        "on this diagnostic."
    )

    print()

    print("=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()