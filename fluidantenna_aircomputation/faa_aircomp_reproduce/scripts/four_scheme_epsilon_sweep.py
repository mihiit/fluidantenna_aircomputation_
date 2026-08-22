"""
four_scheme_epsilon_sweep.py
=============================

Four-scheme validation experiment for the FAA-AirComp paper.

Schemes
-------
1. FAA + Power Control       : Proposed method
2. FAA + Max Power           : APV optimization, p_k = Pmax
3. FPA + Power Control       : Fixed uniform array + power control
4. FPA + Max Power           : Fixed uniform array, p_k = Pmax

Experiment
----------
K = 8
N = 6

epsilon =
    [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

500 paired physical-channel realizations per epsilon.

CRITICAL FAIRNESS RULE
----------------------
For every realization:

    ONE physical channel is generated.

    The same:
        - distances
        - path-loss coefficients
        - multipath coefficients g
        - path angles phi

    are reused by ALL FOUR schemes.

Only the optimization method differs.

The fixed-power baselines always use:
    p_k = Pmax

Therefore their total transmit energy is exactly:

    E = K * Pmax

independent of epsilon and APV.

For the max-power schemes, APV optimization is performed only
to establish feasibility / final MSE. It does NOT change the
energy because p_k is fixed.

Outputs
-------
output/four_scheme_epsilon_sweep_500.csv
output/four_scheme_epsilon_sweep_summary.txt

This script does NOT modify src/bcd.py.
"""

from __future__ import annotations

# ============================================================================
# PROJECT ROOT
# ============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if not PROJECT_ROOT.exists():
    raise RuntimeError(
        f"Project root does not exist: {PROJECT_ROOT}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# IMPORTS
# ============================================================================

import csv
import math
import os
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.stats import t, ttest_rel, wilcoxon

from src.config import SystemConfig

from src.channel import (
    make_channel,
    build_channel_matrix,
    random_port_positions,
)

from src.bcd import (
    run_bcd,
    step_S1_mmse,
    step_S3_precoders,
    compute_mse,
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

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

OUTPUT_DIR = PROJECT_ROOT / "output"

CSV_PATH = (
    OUTPUT_DIR
    / "four_scheme_epsilon_sweep_500.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "four_scheme_epsilon_sweep_summary.txt"
)


# ============================================================================
# NUMERICAL TOLERANCES
# ============================================================================

MONO_TOL = 1e-9
MSE_TOL = 1e-9
WIN_TOL = 1e-12
LOG_FLOOR_W = 1e-15


# ============================================================================
# HELPERS
# ============================================================================

def run_silent(func, *args, **kwargs):
    """
    Run existing implementation without terminal flooding.

    The algorithm itself is unchanged.
    Only stdout is redirected.
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            return func(*args, **kwargs)


def energy_to_dbm(energy_w):
    """
    Convert watts to dBm.
    """
    return 10.0 * np.log10(
        max(float(energy_w), LOG_FLOOR_W) * 1000.0
    )


def is_monotone_nonincreasing(history):
    """
    Check:

        E_1 >= E_2 >= ... >= E_n

    with numerical tolerance.
    """
    if len(history) <= 1:
        return True

    return all(
        history[i]
        >= history[i + 1] - MONO_TOL
        for i in range(len(history) - 1)
    )


def confidence_interval_95(values):
    """
    95% Student-t CI for the sample mean.
    """
    values = np.asarray(values, dtype=float)

    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    mean = float(np.mean(values))

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = std / math.sqrt(n)

    critical = float(
        t.ppf(
            0.975,
            df=n - 1,
        )
    )

    margin = critical * se

    return (
        mean - margin,
        mean + margin,
    )


def safe_wilcoxon(x, y):
    """
    Paired two-sided Wilcoxon signed-rank test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = y - x

    if np.allclose(
        diff,
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

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def validate_physical_channel(
    H,
    bk,
    g,
    phi,
):
    """
    Strict validation of one physical realization.
    """
    H = np.asarray(H)
    bk = np.asarray(bk)
    g = np.asarray(g)
    phi = np.asarray(phi)

    if H.shape != (N, K):
        raise RuntimeError(
            f"Invalid H shape: "
            f"expected {(N, K)}, got {H.shape}"
        )

    if bk.shape != (K,):
        raise RuntimeError(
            f"Invalid bk shape: "
            f"expected {(K,)}, got {bk.shape}"
        )

    if g.shape[0] != K:
        raise RuntimeError(
            f"Invalid g shape: {g.shape}"
        )

    if phi.shape[0] != K:
        raise RuntimeError(
            f"Invalid phi shape: {phi.shape}"
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


# ============================================================================
# MAX-POWER BASELINE
# ============================================================================

def run_max_power(
    H_init,
    bk,
    K,
    cfg,
    rng,
    eps,
    do_apv,
    C_apv,
    g,
    phi,
):
    """
    Fixed-Pmax baseline.

    Power is NEVER optimized.

        p_k = Pmax  for every k.

    For FPA:
        do_apv=False

    For FAA:
        do_apv=True and candidate APVs are generated using the
        SAME g and phi physical realization.

    APV selection is based on the resulting MSE under fixed
    maximum power.

    Energy is therefore always:

        K * Pmax

    for both max-power schemes.

    Returns
    -------
    energy
    mse
    history
    """

    H = np.asarray(
        H_init,
        dtype=complex,
    ).copy()

    p = np.ones(K) * cfg.Pmax

    tau = np.ones(
        K,
        dtype=complex,
    )

    # ------------------------------------------------------------------------
    # Initial MMSE combiner + phase alignment
    # ------------------------------------------------------------------------

    m = step_S1_mmse(
        tau,
        p,
        H,
        cfg.sigma2,
    )

    tau = step_S3_precoders(
        m,
        H,
    )

    mse = compute_mse(
        m,
        tau,
        p,
        H,
        cfg.sigma2,
    )

    history = [float(mse)]

    # ------------------------------------------------------------------------
    # Fixed-array baseline
    # ------------------------------------------------------------------------

    if not do_apv:

        for _ in range(cfg.bcd_max_iter - 1):

            m = step_S1_mmse(
                tau,
                p,
                H,
                cfg.sigma2,
            )

            tau = step_S3_precoders(
                m,
                H,
            )

            mse_new = compute_mse(
                m,
                tau,
                p,
                H,
                cfg.sigma2,
            )

            history.append(
                float(mse_new)
            )

            relative_change = (
                abs(mse_new - mse)
                / max(abs(mse), 1e-14)
            )

            mse = mse_new

            if relative_change < cfg.bcd_tol:
                break

        final_mse = compute_mse(
            m,
            tau,
            p,
            H,
            cfg.sigma2,
        )

        energy = float(np.sum(p))

        return (
            energy,
            float(final_mse),
            history,
        )

    # ------------------------------------------------------------------------
    # FAA max-power baseline
    # ------------------------------------------------------------------------

    if g is None or phi is None:
        raise ValueError(
            "FAA max-power baseline requires g and phi."
        )

    if C_apv is None:
        C_apv = cfg.C_apv

    for _ in range(cfg.bcd_max_iter):

        # ------------------------------------------------------------
        # S1
        # ------------------------------------------------------------

        m = step_S1_mmse(
            tau,
            p,
            H,
            cfg.sigma2,
        )

        # ------------------------------------------------------------
        # S3
        # ------------------------------------------------------------

        tau = step_S3_precoders(
            m,
            H,
        )

        current_mse = compute_mse(
            m,
            tau,
            p,
            H,
            cfg.sigma2,
        )

        best_H = H
        best_mse = current_mse
        best_tau = tau.copy()

        # ------------------------------------------------------------
        # APV candidate search
        #
        # IMPORTANT:
        # g and phi remain fixed.
        # Only APV positions change.
        # ------------------------------------------------------------

        for _candidate in range(C_apv):

            pos = random_port_positions(
                N,
                cfg,
                rng,
            )

            H_candidate = build_channel_matrix(
                pos_wl=pos,
                bk=bk,
                K=K,
                cfg=cfg,
                g=g,
                phi=phi,
            )

            m_candidate = step_S1_mmse(
                tau,
                p,
                H_candidate,
                cfg.sigma2,
            )

            tau_candidate = step_S3_precoders(
                m_candidate,
                H_candidate,
            )

            mse_candidate = compute_mse(
                m_candidate,
                tau_candidate,
                p,
                H_candidate,
                cfg.sigma2,
            )

            if (
                mse_candidate
                < best_mse - MSE_TOL
            ):
                best_mse = float(
                    mse_candidate
                )

                best_H = H_candidate

                best_tau = (
                    tau_candidate
                )

        # ------------------------------------------------------------
        # Accept APV only if MSE decreases
        # ------------------------------------------------------------

        if best_mse < current_mse - MSE_TOL:

            H = best_H

            tau = best_tau

            mse_new = best_mse

        else:

            mse_new = current_mse

        history.append(
            float(mse_new)
        )

        relative_change = (
            abs(mse_new - mse)
            / max(abs(mse), 1e-14)
        )

        mse = mse_new

        if relative_change < cfg.bcd_tol:
            break

    # ------------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------------

    final_mse = compute_mse(
        m,
        tau,
        p,
        H,
        cfg.sigma2,
    )

    energy = float(
        np.sum(p)
    )

    return (
        energy,
        float(final_mse),
        history,
    )


# ============================================================================
# STATISTICS
# ============================================================================

def paired_statistics(
    proposed,
    baseline,
):
    """
    Paired statistics:

        difference = baseline - proposed

    Positive difference means the proposed scheme
    uses less energy.
    """

    proposed = np.asarray(
        proposed,
        dtype=float,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    diff = (
        baseline
        - proposed
    )

    ci_low, ci_high = (
        confidence_interval_95(
            diff
        )
    )

    t_stat, t_p = ttest_rel(
        baseline,
        proposed,
    )

    w_stat, w_p = safe_wilcoxon(
        proposed,
        baseline,
    )

    saving = (
        100.0
        * diff
        / np.maximum(
            baseline,
            LOG_FLOOR_W,
        )
    )

    return {
        "mean_difference_W":
            float(np.mean(diff)),

        "median_difference_W":
            float(np.median(diff)),

        "ci_low_W":
            float(ci_low),

        "ci_high_W":
            float(ci_high),

        "mean_saving_pct":
            float(np.mean(saving)),

        "median_saving_pct":
            float(np.median(saving)),

        "std_saving_pct":
            float(
                np.std(
                    saving,
                    ddof=1,
                )
            ),

        "p05_saving_pct":
            float(
                np.percentile(
                    saving,
                    5,
                )
            ),

        "p25_saving_pct":
            float(
                np.percentile(
                    saving,
                    25,
                )
            ),

        "p75_saving_pct":
            float(
                np.percentile(
                    saving,
                    75,
                )
            ),

        "p95_saving_pct":
            float(
                np.percentile(
                    saving,
                    95,
                )
            ),

        "wilcoxon_stat":
            float(w_stat),

        "wilcoxon_p":
            float(w_p),

        "paired_t_stat":
            float(t_stat),

        "paired_t_p":
            float(t_p),

        "saving_per_realization":
            saving,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = SystemConfig()

    total_start = (
        time.perf_counter()
    )

    rows = []

    successful = 0
    failed = 0

    # ------------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------------

    print("=" * 78)
    print(
        "FAA-AirComp FOUR-SCHEME EPSILON SWEEP"
    )
    print("=" * 78)

    print()
    print("PROJECT")
    print("-" * 78)
    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Script       : "
        f"{Path(__file__).resolve()}"
    )

    print()
    print("CONFIGURATION")
    print("-" * 78)
    print(f"K            : {K}")
    print(f"N            : {N}")
    print(
        f"Realizations : "
        f"{NUM_REALIZATIONS}"
    )
    print(
        f"Epsilons     : "
        f"{EPSILONS}"
    )
    print(
        f"Base seed    : "
        f"{BASE_SEED}"
    )
    print(
        f"Pmax         : "
        f"{cfg.Pmax} W"
    )
    print(
        f"Max-power total energy : "
        f"{K * cfg.Pmax:.9e} W"
    )
    print(
        f"Max-power total energy : "
        f"{energy_to_dbm(K * cfg.Pmax):.6f} dBm"
    )
    print(
        f"C_apv        : "
        f"{cfg.C_apv}"
    )
    print(
        f"BCD tolerance: "
        f"{cfg.bcd_tol}"
    )
    print(
        f"BCD max iter : "
        f"{cfg.bcd_max_iter}"
    )

    print()
    print("OUTPUT")
    print("-" * 78)
    print(
        f"CSV     : {CSV_PATH}"
    )
    print(
        f"Summary : {SUMMARY_PATH}"
    )

    # =========================================================================
    # EPSILON LOOP
    # =========================================================================

    for eps_index, eps in enumerate(
        EPSILONS,
        start=1,
    ):

        print()
        print("=" * 78)
        print(
            f"EPSILON "
            f"{eps_index}/{len(EPSILONS)}"
            f"  ->  epsilon = {eps:.2f}"
        )
        print("=" * 78)

        eps_start = (
            time.perf_counter()
        )

        for realization in range(
            NUM_REALIZATIONS
        ):

            # =================================================================
            # ONE PHYSICAL CHANNEL
            # =================================================================

            channel_seed = (
                BASE_SEED
                + realization
            )

            channel_rng = (
                np.random.default_rng(
                    channel_seed
                )
            )

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

            validate_physical_channel(
                H,
                bk,
                g,
                phi,
            )

            # =================================================================
            # INDEPENDENT OPTIMIZATION RNGs
            #
            # These DO NOT generate the physical channel.
            # They only control stochastic APV candidate generation.
            # =================================================================

            faa_pc_rng = (
                np.random.default_rng(
                    BASE_SEED
                    + 1_000_000
                    + realization
                )
            )

            faa_max_rng = (
                np.random.default_rng(
                    BASE_SEED
                    + 2_000_000
                    + realization
                )
            )

            fpa_pc_rng = (
                np.random.default_rng(
                    BASE_SEED
                    + 3_000_000
                    + realization
                )
            )

            fpa_max_rng = (
                np.random.default_rng(
                    BASE_SEED
                    + 4_000_000
                    + realization
                )
            )

            try:

                # =============================================================
                # 1. FAA + POWER CONTROL
                # =============================================================

                (
                    E_faa_pc,
                    hist_faa_pc,
                ) = run_silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=faa_pc_rng,
                    eps=eps,
                    do_apv=True,
                    C_apv=cfg.C_apv,
                    return_history=True,
                    g=g.copy(),
                    phi=phi.copy(),
                )

                # =============================================================
                # 2. FAA + MAX POWER
                # =============================================================

                (
                    E_faa_mp,
                    MSE_faa_mp,
                    hist_faa_mp,
                ) = run_silent(
                    run_max_power,
                    H.copy(),
                    bk.copy(),
                    K,
                    cfg,
                    faa_max_rng,
                    eps,
                    True,
                    cfg.C_apv,
                    g.copy(),
                    phi.copy(),
                )

                # =============================================================
                # 3. FPA + POWER CONTROL
                # =============================================================

                (
                    E_fpa_pc,
                    hist_fpa_pc,
                ) = run_silent(
                    run_bcd,
                    H.copy(),
                    bk.copy(),
                    K=K,
                    cfg=cfg,
                    rng=fpa_pc_rng,
                    eps=eps,
                    do_apv=False,
                    C_apv=None,
                    return_history=True,
                    g=None,
                    phi=None,
                )

                # =============================================================
                # 4. FPA + MAX POWER
                # =============================================================

                (
                    E_fpa_mp,
                    MSE_fpa_mp,
                    hist_fpa_mp,
                ) = run_silent(
                    run_max_power,
                    H.copy(),
                    bk.copy(),
                    K,
                    cfg,
                    fpa_max_rng,
                    eps,
                    False,
                    None,
                    None,
                    None,
                )

                # =============================================================
                # FINAL MSE FOR POWER-CONTROL METHODS
                #
                # run_bcd does not expose m/tau.
                # Therefore we only record its energy/history here.
                # The max-power routine explicitly exposes final MSE.
                # =============================================================

                E_faa_pc = float(
                    E_faa_pc
                )

                E_fpa_pc = float(
                    E_fpa_pc
                )

                E_faa_mp = float(
                    E_faa_mp
                )

                E_fpa_mp = float(
                    E_fpa_mp
                )

                # =============================================================
                # VALIDATION
                # =============================================================

                energies = [
                    E_faa_pc,
                    E_faa_mp,
                    E_fpa_pc,
                    E_fpa_mp,
                ]

                if not all(
                    np.isfinite(E)
                    and E >= 0.0
                    for E in energies
                ):
                    raise RuntimeError(
                        "Non-finite or negative energy."
                    )

                expected_max_power_energy = (
                    K * cfg.Pmax
                )

                if not np.isclose(
                    E_faa_mp,
                    expected_max_power_energy,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "FAA-MaxPower energy "
                        "does not equal K*Pmax."
                    )

                if not np.isclose(
                    E_fpa_mp,
                    expected_max_power_energy,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "FPA-MaxPower energy "
                        "does not equal K*Pmax."
                    )

                # =============================================================
                # WIN COUNTS
                # =============================================================

                faa_pc_vs_fpa_pc_win = (
                    E_faa_pc
                    < E_fpa_pc - WIN_TOL
                )

                faa_pc_vs_faa_mp_win = (
                    E_faa_pc
                    < E_faa_mp - WIN_TOL
                )

                faa_pc_vs_fpa_mp_win = (
                    E_faa_pc
                    < E_fpa_mp - WIN_TOL
                )

                # =============================================================
                # STORE
                # =============================================================

                rows.append(
                    {
                        "epsilon": eps,
                        "realization":
                            realization + 1,
                        "seed":
                            channel_seed,

                        "faa_pc_energy_W":
                            E_faa_pc,

                        "faa_maxpower_energy_W":
                            E_faa_mp,

                        "fpa_pc_energy_W":
                            E_fpa_pc,

                        "fpa_maxpower_energy_W":
                            E_fpa_mp,

                        "faa_pc_energy_dBm":
                            energy_to_dbm(
                                E_faa_pc
                            ),

                        "faa_maxpower_energy_dBm":
                            energy_to_dbm(
                                E_faa_mp
                            ),

                        "fpa_pc_energy_dBm":
                            energy_to_dbm(
                                E_fpa_pc
                            ),

                        "fpa_maxpower_energy_dBm":
                            energy_to_dbm(
                                E_fpa_mp
                            ),

                        "faa_pc_vs_fpa_pc_saving_pct":
                            100.0
                            * (
                                E_fpa_pc
                                - E_faa_pc
                            )
                            / max(
                                E_fpa_pc,
                                LOG_FLOOR_W,
                            ),

                        "faa_pc_vs_faa_maxpower_saving_pct":
                            100.0
                            * (
                                E_faa_mp
                                - E_faa_pc
                            )
                            / max(
                                E_faa_mp,
                                LOG_FLOOR_W,
                            ),

                        "faa_pc_vs_fpa_maxpower_saving_pct":
                            100.0
                            * (
                                E_fpa_mp
                                - E_faa_pc
                            )
                            / max(
                                E_fpa_mp,
                                LOG_FLOOR_W,
                            ),

                        "faa_maxpower_mse":
                            float(MSE_faa_mp),

                        "fpa_maxpower_mse":
                            float(MSE_fpa_mp),

                        "faa_pc_iterations":
                            len(hist_faa_pc),

                        "fpa_pc_iterations":
                            len(hist_fpa_pc),

                        "faa_maxpower_iterations":
                            len(hist_faa_mp),

                        "fpa_maxpower_iterations":
                            len(hist_fpa_mp),

                        "faa_pc_monotone":
                            int(
                                is_monotone_nonincreasing(
                                    hist_faa_pc
                                )
                            ),

                        "fpa_pc_monotone":
                            int(
                                is_monotone_nonincreasing(
                                    hist_fpa_pc
                                )
                            ),

                        "faa_maxpower_monotone_mse":
                            int(
                                is_monotone_nonincreasing(
                                    hist_faa_mp
                                )
                            ),

                        "fpa_maxpower_monotone_mse":
                            int(
                                is_monotone_nonincreasing(
                                    hist_fpa_mp
                                )
                            ),

                        "faa_pc_vs_fpa_pc_win":
                            int(
                                faa_pc_vs_fpa_pc_win
                            ),

                        "faa_pc_vs_faa_maxpower_win":
                            int(
                                faa_pc_vs_faa_mp_win
                            ),

                        "faa_pc_vs_fpa_maxpower_win":
                            int(
                                faa_pc_vs_fpa_mp_win
                            ),

                        "same_physical_channel":
                            1,
                    }
                )

                successful += 1

            except Exception as exc:

                failed += 1

                print()
                print(
                    "ERROR"
                )
                print(
                    f"epsilon      = {eps}"
                )
                print(
                    f"realization  = "
                    f"{realization + 1}"
                )
                print(
                    f"seed         = "
                    f"{channel_seed}"
                )
                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                raise

            # =================================================================
            # PROGRESS
            # =================================================================

            completed = (
                realization + 1
            )

            if (
                completed % 50 == 0
                or completed
                == NUM_REALIZATIONS
            ):

                elapsed = (
                    time.perf_counter()
                    - eps_start
                )

                avg_time = (
                    elapsed
                    / completed
                )

                remaining = (
                    NUM_REALIZATIONS
                    - completed
                )

                eta = (
                    remaining
                    * avg_time
                )

                print(
                    f"  {completed:4d}/"
                    f"{NUM_REALIZATIONS} "
                    f"completed | "
                    f"{elapsed / 60.0:.2f} min | "
                    f"ETA "
                    f"{eta / 60.0:.2f} min"
                )

        # =====================================================================
        # LIVE EPSILON SUMMARY
        # =====================================================================

        eps_rows = [
            r
            for r in rows
            if r["epsilon"] == eps
        ]

        proposed = np.array(
            [
                r[
                    "faa_pc_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        fpa_pc = np.array(
            [
                r[
                    "fpa_pc_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        faa_mp = np.array(
            [
                r[
                    "faa_maxpower_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        fpa_mp = np.array(
            [
                r[
                    "fpa_maxpower_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        stats_pc = paired_statistics(
            proposed,
            fpa_pc,
        )

        stats_mp = paired_statistics(
            proposed,
            fpa_mp,
        )

        stats_faa_power_control = (
            paired_statistics(
                proposed,
                faa_mp,
            )
        )

        print()
        print(
            f"EPSILON = {eps:.2f}"
        )
        print("-" * 78)

        print(
            f"FAA + PC mean        : "
            f"{np.mean(proposed):.9e} W"
        )

        print(
            f"FAA + MaxPower mean  : "
            f"{np.mean(faa_mp):.9e} W"
        )

        print(
            f"FPA + PC mean        : "
            f"{np.mean(fpa_pc):.9e} W"
        )

        print(
            f"FPA + MaxPower mean  : "
            f"{np.mean(fpa_mp):.9e} W"
        )

        print()

        print(
            "FAA+PC vs FPA+PC"
        )

        print(
            f"Mean saving : "
            f"{stats_pc['mean_saving_pct']:.4f}%"
        )

        print(
            f"Wilcoxon p  : "
            f"{stats_pc['wilcoxon_p']:.6e}"
        )

        print(
            "FAA+PC vs FAA-MaxPower"
        )

        print(
            f"Mean saving : "
            f"{stats_faa_power_control['mean_saving_pct']:.4f}%"
        )

        print(
            "FAA+PC vs FPA-MaxPower"
        )

        print(
            f"Mean saving : "
            f"{stats_mp['mean_saving_pct']:.4f}%"
        )

        print(
            f"FAA+PC wins over FPA+PC : "
            f"{sum(r['faa_pc_vs_fpa_pc_win'] for r in eps_rows)}"
            f"/{NUM_REALIZATIONS}"
        )

        print(
            f"FAA-MaxPower MSE mean : "
            f"{np.mean([r['faa_maxpower_mse'] for r in eps_rows]):.9e}"
        )

        print(
            f"FPA-MaxPower MSE mean : "
            f"{np.mean([r['fpa_maxpower_mse'] for r in eps_rows]):.9e}"
        )

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    expected_rows = (
        len(EPSILONS)
        * NUM_REALIZATIONS
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(rows)}."
        )

    if successful != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} "
            f"successful realizations, "
            f"got {successful}."
        )

    if failed != 0:
        raise RuntimeError(
            f"Experiment has {failed} failures."
        )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    fieldnames = list(
        rows[0].keys()
    )

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
    # SUMMARY
    # =========================================================================

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    summary = []

    summary.append(
        "=" * 78
    )

    summary.append(
        "FAA-AirComp FOUR-SCHEME EPSILON SWEEP"
    )

    summary.append(
        "=" * 78
    )

    summary.append("")

    summary.append(
        "GLOBAL VALIDATION"
    )

    summary.append(
        "-" * 78
    )

    summary.append(
        f"Successful realizations : "
        f"{successful}"
    )

    summary.append(
        f"Failed realizations     : "
        f"{failed}"
    )

    summary.append(
        f"Expected rows           : "
        f"{expected_rows}"
    )

    summary.append(
        f"Total runtime           : "
        f"{total_runtime / 60.0:.2f} min"
    )

    summary.append("")

    summary.append(
        "MAX-POWER VALIDATION"
    )

    summary.append(
        "-" * 78
    )

    summary.append(
        f"Pmax per device        : "
        f"{cfg.Pmax:.9e} W"
    )

    summary.append(
        f"K * Pmax               : "
        f"{K * cfg.Pmax:.9e} W"
    )

    summary.append(
        f"K * Pmax               : "
        f"{energy_to_dbm(K * cfg.Pmax):.6f} dBm"
    )

    summary.append("")

    # =========================================================================
    # PER-EPSILON FINAL SUMMARY
    # =========================================================================

    for eps in EPSILONS:

        eps_rows = [
            r
            for r in rows
            if r["epsilon"] == eps
        ]

        faa_pc = np.array(
            [
                r[
                    "faa_pc_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        faa_mp = np.array(
            [
                r[
                    "faa_maxpower_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        fpa_pc = np.array(
            [
                r[
                    "fpa_pc_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        fpa_mp = np.array(
            [
                r[
                    "fpa_maxpower_energy_W"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        faa_pc_stats = paired_statistics(
            faa_pc,
            fpa_pc,
        )

        faa_mp_stats = paired_statistics(
            faa_pc,
            faa_mp,
        )

        fpa_mp_stats = paired_statistics(
            faa_pc,
            fpa_mp,
        )

        faa_pc_wins = int(
            np.sum(
                faa_pc
                < fpa_pc - WIN_TOL
            )
        )

        fpa_pc_wins = int(
            np.sum(
                fpa_pc
                < faa_pc - WIN_TOL
            )
        )

        ties = (
            NUM_REALIZATIONS
            - faa_pc_wins
            - fpa_pc_wins
        )

        faa_max_mse = np.array(
            [
                r[
                    "faa_maxpower_mse"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        fpa_max_mse = np.array(
            [
                r[
                    "fpa_maxpower_mse"
                ]
                for r in eps_rows
            ],
            dtype=float,
        )

        summary.extend(
            [
                "",
                "=" * 78,
                f"EPSILON = {eps:.2f}",
                "=" * 78,
                "",
                "MEAN ENERGY",
                "-" * 78,
                (
                    f"FAA + PC       : "
                    f"{np.mean(faa_pc):.9e} W"
                ),
                (
                    f"FAA + MaxPower : "
                    f"{np.mean(faa_mp):.9e} W"
                ),
                (
                    f"FPA + PC       : "
                    f"{np.mean(fpa_pc):.9e} W"
                ),
                (
                    f"FPA + MaxPower : "
                    f"{np.mean(fpa_mp):.9e} W"
                ),
                "",
                "FAA+PC vs FPA+PC",
                "-" * 78,
                (
                    f"Mean saving    : "
                    f"{faa_pc_stats['mean_saving_pct']:.6f}%"
                ),
                (
                    f"Median saving  : "
                    f"{faa_pc_stats['median_saving_pct']:.6f}%"
                ),
                (
                    f"95% CI diff    : "
                    f"[{faa_pc_stats['ci_low_W']:.9e}, "
                    f"{faa_pc_stats['ci_high_W']:.9e}] W"
                ),
                (
                    f"Wilcoxon p     : "
                    f"{faa_pc_stats['wilcoxon_p']:.6e}"
                ),
                (
                    f"Paired t p     : "
                    f"{faa_pc_stats['paired_t_p']:.6e}"
                ),
                "",
                "WIN COUNTS — FAA+PC vs FPA+PC",
                "-" * 78,
                (
                    f"FAA+PC wins    : "
                    f"{faa_pc_wins}/{NUM_REALIZATIONS} "
                    f"({100.0 * faa_pc_wins / NUM_REALIZATIONS:.2f}%)"
                ),
                (
                    f"FPA+PC wins    : "
                    f"{fpa_pc_wins}/{NUM_REALIZATIONS} "
                    f"({100.0 * fpa_pc_wins / NUM_REALIZATIONS:.2f}%)"
                ),
                (
                    f"Ties           : "
                    f"{ties}/{NUM_REALIZATIONS} "
                    f"({100.0 * ties / NUM_REALIZATIONS:.2f}%)"
                ),
                "",
                "FAA+PC VS FAA-MAXPOWER",
                "-" * 78,
                (
                    f"Mean saving    : "
                    f"{faa_mp_stats['mean_saving_pct']:.6f}%"
                ),
                "",
                "FAA+PC VS FPA-MAXPOWER",
                "-" * 78,
                (
                    f"Mean saving    : "
                    f"{fpa_mp_stats['mean_saving_pct']:.6f}%"
                ),
                "",
                "MAX-POWER MSE",
                "-" * 78,
                (
                    f"FAA-MaxPower mean MSE : "
                    f"{np.mean(faa_max_mse):.9e}"
                ),
                (
                    f"FAA-MaxPower median MSE : "
                    f"{np.median(faa_max_mse):.9e}"
                ),
                (
                    f"FPA-MaxPower mean MSE : "
                    f"{np.mean(fpa_max_mse):.9e}"
                ),
                (
                    f"FPA-MaxPower median MSE : "
                    f"{np.median(fpa_max_mse):.9e}"
                ),
                "",
                "CONVERGENCE",
                "-" * 78,
                (
                    f"FAA+PC mean iterations : "
                    f"{np.mean([r['faa_pc_iterations'] for r in eps_rows]):.3f}"
                ),
                (
                    f"FPA+PC mean iterations : "
                    f"{np.mean([r['fpa_pc_iterations'] for r in eps_rows]):.3f}"
                ),
            ]
        )

    summary.extend(
        [
            "",
            "=" * 78,
            "EXPERIMENT COMPLETE",
            "=" * 78,
            (
                f"CSV saved     : "
                f"{CSV_PATH}"
            ),
            (
                f"Summary saved : "
                f"{SUMMARY_PATH}"
            ),
        ]
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(summary)
        )

    print()
    print(
        "\n".join(summary)
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()