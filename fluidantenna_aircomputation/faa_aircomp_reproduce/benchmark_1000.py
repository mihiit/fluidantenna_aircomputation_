import os
import csv
import time
import contextlib

import numpy as np

from scipy.stats import wilcoxon, ttest_rel

from src.config import SystemConfig
from src.channel import make_channel
from src.bcd import run_bcd


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

NUM_REALIZATIONS = 1000

# Seeds 0...999.
# Realization 0 therefore corresponds to the same seed used
# by debug_case.py.
BASE_SEED = 0

# Exact Fig. 3 configuration
K = 8
N = 6
EPS = 0.06

# Explicitly lock APV candidate count.
C_APV = 40

# Output files
OUTPUT_DIR = "output"
CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "benchmark_1000.csv"
)
SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "benchmark_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def is_monotone_nonincreasing(history, tol=1e-9):
    """
    Energy should never increase across accepted BCD states.
    """
    if len(history) < 2:
        return True

    return all(
        history[i] >= history[i + 1] - tol
        for i in range(len(history) - 1)
    )


def percentile_stats(values):
    """
    Return the requested percentiles.
    """
    values = np.asarray(values, dtype=float)

    return {
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def confidence_interval_mean(values, confidence=0.95):
    """
    Normal-theory 95% CI for the sample mean.
    """
    values = np.asarray(values, dtype=float)

    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)

    # 1.96 is sufficient for n=1000.
    z = 1.96

    return (
        float(mean - z * se),
        float(mean + z * se),
    )


def run_one_realization(seed, cfg):
    """
    Run FAA and FPA on EXACTLY the same physical channel.

    Important:
        - Same seed -> same distances
        - Same seed -> same path loss
        - Same seed -> same g
        - Same seed -> same phi
        - Same initial uniform positions
        - FAA and FPA then use separate RNG objects

    FAA uses APV candidate randomness after channel creation.
    FPA has no APV randomness.
    """

    # --------------------------------------------------------
    # FAA
    #
    # This deliberately mirrors debug_case.py:
    #
    # rng = default_rng(seed)
    # make_channel(...)
    # run_bcd(..., do_apv=True)
    # --------------------------------------------------------

    rng_faa = np.random.default_rng(seed)

    (
        H_faa,
        dk_faa,
        bk_faa,
        pos_faa,
        g_faa,
        phi_faa,
    ) = make_channel(
        K=K,
        N=N,
        cfg=cfg,
        rng=rng_faa,
    )

    E_faa, history_faa = run_bcd(
        H_faa,
        bk_faa,
        K=K,
        cfg=cfg,
        rng=rng_faa,
        eps=EPS,
        do_apv=True,
        C_apv=C_APV,
        return_history=True,
        g=g_faa,
        phi=phi_faa,
    )

    # --------------------------------------------------------
    # FPA
    #
    # Reset to EXACT SAME seed.
    #
    # Therefore make_channel() produces the identical
    # physical realization.
    # --------------------------------------------------------

    rng_fpa = np.random.default_rng(seed)

    (
        H_fpa,
        dk_fpa,
        bk_fpa,
        pos_fpa,
        g_fpa,
        phi_fpa,
    ) = make_channel(
        K=K,
        N=N,
        cfg=cfg,
        rng=rng_fpa,
    )

    # --------------------------------------------------------
    # PHYSICAL CHANNEL IDENTITY CHECK
    # --------------------------------------------------------

    same_channel = (
        np.allclose(dk_faa, dk_fpa)
        and np.allclose(bk_faa, bk_fpa)
        and np.allclose(pos_faa, pos_fpa)
        and np.allclose(g_faa, g_fpa)
        and np.allclose(phi_faa, phi_fpa)
        and np.allclose(H_faa, H_fpa)
    )

    if not same_channel:
        raise RuntimeError(
            f"Physical channel mismatch for seed {seed}"
        )

    # --------------------------------------------------------
    # Run FPA
    # --------------------------------------------------------

    E_fpa, history_fpa = run_bcd(
        H_fpa,
        bk_fpa,
        K=K,
        cfg=cfg,
        rng=rng_fpa,
        eps=EPS,
        do_apv=False,
        C_apv=C_APV,
        return_history=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    E_faa = float(E_faa)
    E_fpa = float(E_fpa)

    if E_fpa <= 0:
        raise RuntimeError(
            f"Invalid FPA energy for seed {seed}: {E_fpa}"
        )

    energy_saving_pct = (
        (E_fpa - E_faa)
        / E_fpa
        * 100.0
    )

    energy_ratio = E_faa / E_fpa

    faa_wins = E_faa < E_fpa - 1e-12
    fpa_wins = E_fpa < E_faa - 1e-12
    tie = not faa_wins and not fpa_wins

    return {
        "seed": seed,

        "faa_energy_W": E_faa,
        "fpa_energy_W": E_fpa,

        "faa_energy_dBm":
            10.0 * np.log10(E_faa * 1000.0),

        "fpa_energy_dBm":
            10.0 * np.log10(E_fpa * 1000.0),

        "faa_saving_pct":
            energy_saving_pct,

        "faa_fpa_ratio":
            energy_ratio,

        "faa_iterations":
            len(history_faa),

        "fpa_iterations":
            len(history_fpa),

        "faa_monotone":
            is_monotone_nonincreasing(history_faa),

        "fpa_monotone":
            is_monotone_nonincreasing(history_fpa),

        "faa_feasible":
            True,

        "fpa_feasible":
            True,

        "same_physical_channel":
            same_channel,

        "faa_wins":
            faa_wins,

        "fpa_wins":
            fpa_wins,

        "tie":
            tie,
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = SystemConfig()

    print("=" * 70)
    print("FAA vs FPA — 1000 INDEPENDENT CHANNEL REALIZATIONS")
    print("=" * 70)

    print(f"K                = {K}")
    print(f"N                = {N}")
    print(f"epsilon          = {EPS}")
    print(f"Pmax (W)         = {cfg.Pmax}")
    print(f"sigma2           = {cfg.sigma2}")
    print(f"rho              = {cfg.rho}")
    print(f"bcd_tol          = {cfg.bcd_tol}")
    print(f"bcd_max_iter     = {cfg.bcd_max_iter}")
    print(f"C_apv            = {C_APV}")
    print(f"realizations     = {NUM_REALIZATIONS}")
    print(f"seed range       = {BASE_SEED} ... "
          f"{BASE_SEED + NUM_REALIZATIONS - 1}")

    print()
    print("IMPORTANT:")
    print("FAA and FPA use the SAME physical channel per seed.")
    print("FAA APV candidate randomness is isolated from channel generation.")
    print()

    # --------------------------------------------------------
    # CSV setup
    # --------------------------------------------------------

    fieldnames = [
        "seed",
        "faa_energy_W",
        "fpa_energy_W",
        "faa_energy_dBm",
        "fpa_energy_dBm",
        "faa_saving_pct",
        "faa_fpa_ratio",
        "faa_iterations",
        "fpa_iterations",
        "faa_monotone",
        "fpa_monotone",
        "faa_feasible",
        "fpa_feasible",
        "same_physical_channel",
        "faa_wins",
        "fpa_wins",
        "tie",
    ]

    # --------------------------------------------------------
    # Start benchmark
    # --------------------------------------------------------

    start_time = time.perf_counter()

    results = []
    failures = []

    # Write header immediately so partial progress survives.
    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(NUM_REALIZATIONS):

            seed = BASE_SEED + i

            try:

                # Suppress the huge BCD diagnostic output.
                #
                # Otherwise 1000 realizations would flood
                # the VS Code terminal.
                with open(
                    os.devnull,
                    "w",
                ) as devnull:

                    with contextlib.redirect_stdout(devnull):

                        result = run_one_realization(
                            seed,
                            cfg,
                        )

                results.append(result)

                writer.writerow(result)
                csvfile.flush()

                # ------------------------------------------------
                # Progress reporting
                # ------------------------------------------------

                if (
                    (i + 1) % 10 == 0
                    or i == 0
                    or i + 1 == NUM_REALIZATIONS
                ):

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    completed = i + 1

                    avg_time = elapsed / completed

                    remaining = (
                        NUM_REALIZATIONS
                        - completed
                    )

                    eta_seconds = (
                        avg_time
                        * remaining
                    )

                    print(
                        f"[{completed:4d}/{NUM_REALIZATIONS}] "
                        f"seed={seed} "
                        f"FAA={result['faa_energy_W']:.6e} W "
                        f"FPA={result['fpa_energy_W']:.6e} W "
                        f"saving={result['faa_saving_pct']:.3f}% "
                        f"FAA_win={result['faa_wins']} "
                        f"ETA={eta_seconds / 60:.1f} min"
                    )

            except Exception as exc:

                failures.append(
                    {
                        "seed": seed,
                        "error": repr(exc),
                    }
                )

                print(
                    f"[FAILED] seed={seed}: {exc}"
                )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if not results:
        raise RuntimeError(
            "No successful realizations."
        )

    faa_energy = np.array(
        [r["faa_energy_W"] for r in results]
    )

    fpa_energy = np.array(
        [r["fpa_energy_W"] for r in results]
    )

    saving = np.array(
        [r["faa_saving_pct"] for r in results]
    )

    ratio = np.array(
        [r["faa_fpa_ratio"] for r in results]
    )

    faa_iterations = np.array(
        [r["faa_iterations"] for r in results],
        dtype=float,
    )

    fpa_iterations = np.array(
        [r["fpa_iterations"] for r in results],
        dtype=float,
    )

    paired_difference = (
        fpa_energy - faa_energy
    )

    # --------------------------------------------------------
    # Wins
    # --------------------------------------------------------

    faa_wins = sum(
        r["faa_wins"]
        for r in results
    )

    fpa_wins = sum(
        r["fpa_wins"]
        for r in results
    )

    ties = sum(
        r["tie"]
        for r in results
    )

    same_channel_count = sum(
        r["same_physical_channel"]
        for r in results
    )

    faa_monotone_count = sum(
        r["faa_monotone"]
        for r in results
    )

    fpa_monotone_count = sum(
        r["fpa_monotone"]
        for r in results
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    faa_pct = percentile_stats(
        faa_energy
    )

    fpa_pct = percentile_stats(
        fpa_energy
    )

    saving_pct = percentile_stats(
        saving
    )

    ratio_pct = percentile_stats(
        ratio
    )

    faa_iter_pct = percentile_stats(
        faa_iterations
    )

    fpa_iter_pct = percentile_stats(
        fpa_iterations
    )

    # --------------------------------------------------------
    # Paired statistical tests
    # --------------------------------------------------------

    # Remove exact zero differences for Wilcoxon.
    nonzero_diff = paired_difference[
        np.abs(paired_difference) > 1e-15
    ]

    if len(nonzero_diff) >= 2:

        wilcoxon_result = wilcoxon(
            nonzero_diff,
            alternative="greater",
            method="auto",
        )

        wilcoxon_stat = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    else:

        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    # Paired t-test as a secondary parametric check.
    ttest_result = ttest_rel(
        fpa_energy,
        faa_energy,
    )

    ttest_stat = float(
        ttest_result.statistic
    )

    ttest_p = float(
        ttest_result.pvalue
    )

    # --------------------------------------------------------
    # Mean paired energy difference
    # --------------------------------------------------------

    mean_difference = float(
        np.mean(paired_difference)
    )

    ci_low, ci_high = confidence_interval_mean(
        paired_difference
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"Successful realizations : "
        f"{len(results)}"
    )

    print(
        f"Failed realizations     : "
        f"{len(failures)}"
    )

    print(
        f"Total runtime           : "
        f"{total_time / 60:.2f} min"
    )

    print()
    print("PHYSICAL CHANNEL VALIDATION")
    print("-" * 70)

    print(
        f"Same physical channel   : "
        f"{same_channel_count}/{len(results)}"
    )

    print()
    print("ENERGY — FAA")
    print("-" * 70)

    print(
        f"Mean                    : "
        f"{np.mean(faa_energy):.9e} W"
    )

    print(
        f"Median                  : "
        f"{np.median(faa_energy):.9e} W"
    )

    print(
        f"Std                     : "
        f"{np.std(faa_energy, ddof=1):.9e} W"
    )

    print(
        f"P05                     : "
        f"{faa_pct['p05']:.9e} W"
    )

    print(
        f"P25                     : "
        f"{faa_pct['p25']:.9e} W"
    )

    print(
        f"P75                     : "
        f"{faa_pct['p75']:.9e} W"
    )

    print(
        f"P95                     : "
        f"{faa_pct['p95']:.9e} W"
    )

    print()
    print("ENERGY — FPA")
    print("-" * 70)

    print(
        f"Mean                    : "
        f"{np.mean(fpa_energy):.9e} W"
    )

    print(
        f"Median                  : "
        f"{np.median(fpa_energy):.9e} W"
    )

    print(
        f"Std                     : "
        f"{np.std(fpa_energy, ddof=1):.9e} W"
    )

    print(
        f"P05                     : "
        f"{fpa_pct['p05']:.9e} W"
    )

    print(
        f"P25                     : "
        f"{fpa_pct['p25']:.9e} W"
    )

    print(
        f"P75                     : "
        f"{fpa_pct['p75']:.9e} W"
    )

    print(
        f"P95                     : "
        f"{fpa_pct['p95']:.9e} W"
    )

    print()
    print("FAA ENERGY SAVING VS FPA")
    print("-" * 70)

    print(
        f"Mean                    : "
        f"{np.mean(saving):.4f}%"
    )

    print(
        f"Median                  : "
        f"{np.median(saving):.4f}%"
    )

    print(
        f"Std                     : "
        f"{np.std(saving, ddof=1):.4f}%"
    )

    print(
        f"P05                     : "
        f"{saving_pct['p05']:.4f}%"
    )

    print(
        f"P25                     : "
        f"{saving_pct['p25']:.4f}%"
    )

    print(
        f"P75                     : "
        f"{saving_pct['p75']:.4f}%"
    )

    print(
        f"P95                     : "
        f"{saving_pct['p95']:.4f}%"
    )

    print()
    print("WIN COUNTS")
    print("-" * 70)

    print(
        f"FAA wins                : "
        f"{faa_wins}/{len(results)} "
        f"({100 * faa_wins / len(results):.2f}%)"
    )

    print(
        f"FPA wins                : "
        f"{fpa_wins}/{len(results)} "
        f"({100 * fpa_wins / len(results):.2f}%)"
    )

    print(
        f"Ties                   : "
        f"{ties}/{len(results)} "
        f"({100 * ties / len(results):.2f}%)"
    )

    print()
    print("CONVERGENCE")
    print("-" * 70)

    print(
        f"FAA mean iterations     : "
        f"{np.mean(faa_iterations):.3f}"
    )

    print(
        f"FAA median iterations   : "
        f"{np.median(faa_iterations):.3f}"
    )

    print(
        f"FAA P05/P25/P75/P95     : "
        f"{faa_iter_pct['p05']:.0f}/"
        f"{faa_iter_pct['p25']:.0f}/"
        f"{faa_iter_pct['p75']:.0f}/"
        f"{faa_iter_pct['p95']:.0f}"
    )

    print(
        f"FPA mean iterations     : "
        f"{np.mean(fpa_iterations):.3f}"
    )

    print(
        f"FPA median iterations   : "
        f"{np.median(fpa_iterations):.3f}"
    )

    print(
        f"FPA P05/P25/P75/P95     : "
        f"{fpa_iter_pct['p05']:.0f}/"
        f"{fpa_iter_pct['p25']:.0f}/"
        f"{fpa_iter_pct['p75']:.0f}/"
        f"{fpa_iter_pct['p95']:.0f}"
    )

    print()
    print("MONOTONICITY")
    print("-" * 70)

    print(
        f"FAA monotone            : "
        f"{faa_monotone_count}/{len(results)}"
    )

    print(
        f"FPA monotone            : "
        f"{fpa_monotone_count}/{len(results)}"
    )

    print()
    print("PAIRED STATISTICS")
    print("-" * 70)

    print(
        f"Mean paired difference "
        f"(FPA - FAA): "
        f"{mean_difference:.9e} W"
    )

    print(
        f"95% CI for difference   : "
        f"[{ci_low:.9e}, {ci_high:.9e}] W"
    )

    print(
        f"Wilcoxon statistic      : "
        f"{wilcoxon_stat}"
    )

    print(
        f"Wilcoxon p-value        : "
        f"{wilcoxon_p:.6e}"
    )

    print(
        f"Paired t statistic      : "
        f"{ttest_stat:.6f}"
    )

    print(
        f"Paired t p-value        : "
        f"{ttest_p:.6e}"
    )

    print()
    print("=" * 70)

    # ========================================================
    # WRITE SUMMARY FILE
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FAA vs FPA — 1000 REALIZATION BENCHMARK\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Successful realizations: "
            f"{len(results)}\n"
        )

        f.write(
            f"Failed realizations: "
            f"{len(failures)}\n"
        )

        f.write(
            f"Total runtime: "
            f"{total_time:.6f} s\n\n"
        )

        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"K = {K}\n")
        f.write(f"N = {N}\n")
        f.write(f"epsilon = {EPS}\n")
        f.write(f"Pmax = {cfg.Pmax}\n")
        f.write(f"sigma2 = {cfg.sigma2}\n")
        f.write(f"rho = {cfg.rho}\n")
        f.write(f"bcd_tol = {cfg.bcd_tol}\n")
        f.write(f"bcd_max_iter = {cfg.bcd_max_iter}\n")
        f.write(f"C_apv = {C_APV}\n\n")

        f.write("ENERGY\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"FAA mean = "
            f"{np.mean(faa_energy):.12e} W\n"
        )

        f.write(
            f"FAA median = "
            f"{np.median(faa_energy):.12e} W\n"
        )

        f.write(
            f"FAA std = "
            f"{np.std(faa_energy, ddof=1):.12e} W\n"
        )

        f.write(
            f"FAA P05 = "
            f"{faa_pct['p05']:.12e} W\n"
        )

        f.write(
            f"FAA P25 = "
            f"{faa_pct['p25']:.12e} W\n"
        )

        f.write(
            f"FAA P75 = "
            f"{faa_pct['p75']:.12e} W\n"
        )

        f.write(
            f"FAA P95 = "
            f"{faa_pct['p95']:.12e} W\n\n"
        )

        f.write(
            f"FPA mean = "
            f"{np.mean(fpa_energy):.12e} W\n"
        )

        f.write(
            f"FPA median = "
            f"{np.median(fpa_energy):.12e} W\n"
        )

        f.write(
            f"FPA std = "
            f"{np.std(fpa_energy, ddof=1):.12e} W\n"
        )

        f.write(
            f"FPA P05 = "
            f"{fpa_pct['p05']:.12e} W\n"
        )

        f.write(
            f"FPA P25 = "
            f"{fpa_pct['p25']:.12e} W\n"
        )

        f.write(
            f"FPA P75 = "
            f"{fpa_pct['p75']:.12e} W\n"
        )

        f.write(
            f"FPA P95 = "
            f"{fpa_pct['p95']:.12e} W\n\n"
        )

        f.write("FAA SAVING\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Mean = "
            f"{np.mean(saving):.6f}%\n"
        )

        f.write(
            f"Median = "
            f"{np.median(saving):.6f}%\n"
        )

        f.write(
            f"Std = "
            f"{np.std(saving, ddof=1):.6f}%\n"
        )

        f.write(
            f"P05 = "
            f"{saving_pct['p05']:.6f}%\n"
        )

        f.write(
            f"P25 = "
            f"{saving_pct['p25']:.6f}%\n"
        )

        f.write(
            f"P75 = "
            f"{saving_pct['p75']:.6f}%\n"
        )

        f.write(
            f"P95 = "
            f"{saving_pct['p95']:.6f}%\n\n"
        )

        f.write("WIN COUNTS\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"FAA wins = {faa_wins}\n"
        )

        f.write(
            f"FPA wins = {fpa_wins}\n"
        )

        f.write(
            f"Ties = {ties}\n\n"
        )

        f.write("CONVERGENCE\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"FAA mean iterations = "
            f"{np.mean(faa_iterations):.4f}\n"
        )

        f.write(
            f"FAA median iterations = "
            f"{np.median(faa_iterations):.4f}\n"
        )

        f.write(
            f"FAA P05/P25/P75/P95 = "
            f"{faa_iter_pct['p05']:.0f}/"
            f"{faa_iter_pct['p25']:.0f}/"
            f"{faa_iter_pct['p75']:.0f}/"
            f"{faa_iter_pct['p95']:.0f}\n"
        )

        f.write(
            f"FPA mean iterations = "
            f"{np.mean(fpa_iterations):.4f}\n"
        )

        f.write(
            f"FPA median iterations = "
            f"{np.median(fpa_iterations):.4f}\n"
        )

        f.write(
            f"FPA P05/P25/P75/P95 = "
            f"{fpa_iter_pct['p05']:.0f}/"
            f"{fpa_iter_pct['p25']:.0f}/"
            f"{fpa_iter_pct['p75']:.0f}/"
            f"{fpa_iter_pct['p95']:.0f}\n\n"
        )

        f.write("STATISTICAL TESTS\n")
        f.write("-" * 70 + "\n")

        f.write(
            "Wilcoxon signed-rank test:\n"
        )

        f.write(
            f"  statistic = "
            f"{wilcoxon_stat}\n"
        )

        f.write(
            f"  p-value = "
            f"{wilcoxon_p:.12e}\n\n"
        )

        f.write(
            "Paired t-test:\n"
        )

        f.write(
            f"  statistic = "
            f"{ttest_stat:.12f}\n"
        )

        f.write(
            f"  p-value = "
            f"{ttest_p:.12e}\n\n"
        )

        f.write(
            f"Mean paired difference (FPA - FAA) = "
            f"{mean_difference:.12e} W\n"
        )

        f.write(
            f"95% CI = "
            f"[{ci_low:.12e}, "
            f"{ci_high:.12e}] W\n"
        )

        if failures:

            f.write("\nFAILURES\n")
            f.write("-" * 70 + "\n")

            for failure in failures:

                f.write(
                    f"seed={failure['seed']} "
                    f"{failure['error']}\n"
                )

    print()
    print(
        f"Per-realization data saved to: "
        f"{CSV_FILE}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()