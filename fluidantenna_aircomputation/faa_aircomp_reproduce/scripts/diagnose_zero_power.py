"""
scripts/diagnose_zero_power.py
==============================

Diagnostic ONLY.

Purpose:
    Verify the zero-power feasibility threshold of the current
    MSE formulation for epsilon = 0.08.

This file does NOT modify:
    - src/bcd.py
    - scaling_experiment.py
    - any completed experiment

For p = 0:

    MSE = || -1/K * 1 ||^2
        = 1/K

Therefore zero power is feasible whenever:

    1/K <= epsilon
"""

from __future__ import annotations


EPSILON = 0.08

K_VALUES = [4, 8, 12, 16]


def main():

    print("=" * 78)
    print("ZERO-POWER / EPSILON DIAGNOSTIC")
    print("=" * 78)

    print(f"epsilon = {EPSILON}")
    print(f"K       = {K_VALUES}")
    print()

    print(
        f"{'K':>6}"
        f"{'1/K':>16}"
        f"{'epsilon':>16}"
        f"{'zero-power feasible':>24}"
    )

    print("-" * 78)

    for K in K_VALUES:

        zero_power_mse = 1.0 / K

        feasible = (
            zero_power_mse
            <= EPSILON
        )

        print(
            f"{K:>6}"
            f"{zero_power_mse:>16.9f}"
            f"{EPSILON:>16.9f}"
            f"{str(feasible):>24}"
        )

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    threshold = 1.0 / EPSILON

    print(
        f"For epsilon = {EPSILON}, "
        f"zero-power feasibility begins when:"
    )

    print()

    print(
        "    1 / K <= epsilon"
    )

    print()

    print(
        "which is equivalent to:"
    )

    print()

    print(
        "    K >= 1 / epsilon"
    )

    print()

    print(
        f"1 / epsilon = {threshold:.6f}"
    )

    print()

    print(
        "Therefore:"
    )

    for K in K_VALUES:

        zero_power_mse = 1.0 / K

        if zero_power_mse <= EPSILON:

            status = "DEGENERATE"

        else:

            status = "NON-DEGENERATE"

        print(
            f"    K={K:2d}: "
            f"1/K={zero_power_mse:.9f} "
            f"-> {status}"
        )

    print()
    print("=" * 78)
    print("IMPORTANT")
    print("=" * 78)

    print(
        "This diagnostic does not change the BCD algorithm."
    )

    print(
        "This diagnostic does not change the scaling experiment."
    )

    print(
        "This diagnostic does not modify epsilon."
    )

    print(
        "It only verifies the mathematical threshold."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()