#!/usr/bin/env python3
"""
scripts/reproduce_all_figures.py
=================================
End-to-end script to reproduce all 6 figures from the paper.

Usage
-----
    # Full paper-quality run (~30–60 min on a modern laptop)
    python scripts/reproduce_all_figures.py

    # Quick test run (~3–5 min, MC=20)
    python scripts/reproduce_all_figures.py --quick

    # Single figure
    python scripts/reproduce_all_figures.py --fig 1
    python scripts/reproduce_all_figures.py --fig 4

    # Custom output directory
    python scripts/reproduce_all_figures.py --outdir my_figures/

Output
------
Figures saved to ./figures/ (or --outdir) as fig1.pdf … fig6.pdf.
A summary table is printed at the end with key paper numbers.

Notes
-----
All random seeds are fixed; results are deterministic and identical
to the paper values at MC=500.
"""

import argparse
import sys
import os
import time
import numpy as np

# Allow running from project root or scripts/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config      import SystemConfig
from src.figures     import setup_rcparams, plot_fig1, plot_fig2, plot_fig3
from src.figures     import plot_fig4, plot_fig5, plot_fig6
from src.experiments import (
    monte_carlo, exp_fig1, exp_fig2, exp_fig3,
    exp_fig4, exp_fig5, exp_fig6,
    _run_proposed, _run_fpa_maxpow,
)


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce FAA-AirComp paper figures")
    p.add_argument("--quick",  action="store_true",
                   help="Quick test run (MC=20 instead of 500)")
    p.add_argument("--fig",    type=int, default=None,
                   choices=[1, 2, 3, 4, 5, 6],
                   help="Generate a single figure only")
    p.add_argument("--outdir", type=str, default="figures",
                   help="Output directory for figures (default: figures/)")
    p.add_argument("--mc",     type=int, default=None,
                   help="Override MC count (overrides --quick)")
    return p.parse_args()


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    args   = parse_args()
    cfg    = SystemConfig()
    outdir = args.outdir
    setup_rcparams()

    # Monte Carlo sizes
    if args.mc is not None:
        MC_MAIN = MC_FIG2 = MC_FIG5 = MC_FIG6 = args.mc
    elif args.quick:
        MC_MAIN = MC_FIG2 = MC_FIG5 = MC_FIG6 = 20
    else:
        MC_MAIN = 500
        MC_FIG2 = 300
        MC_FIG5 = 300
        MC_FIG6 = 200

    mode = "QUICK (MC=20)" if args.quick and args.mc is None else \
           f"MC={MC_MAIN}"
    print(f"\nFAA-AirComp Reproducibility Script")
    print(f"Mode: {mode}")
    print(f"Output: {outdir}/")
    print(cfg.summary())

    t_total = time.time()
    results_summary = {}

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 1:
        header("Figure 1: TX Energy vs. MSE Threshold ε  (N=6, K=8)")
        t0 = time.time()

        eps_vals = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
        K, N     = 8, 6
        results  = exp_fig1(eps_vals, K, N, cfg, MC_MAIN, base_seed=1000)
        plot_fig1(eps_vals, results, outdir=outdir)

        # Key result: ESR at ε=0.06
        idx_006 = list(eps_vals).index(0.06)
        E_prop  = results["proposed"][idx_006].mean
        E_base  = results["fpa_maxpow"][idx_006].mean
        esr_006 = 100 * (E_base - E_prop) / E_base
        results_summary["ESR at ε=0.06"] = f"{esr_006:.1f}%  (paper: ~47%)"
        print(f"\n  Key result: ESR at ε=0.06 = {esr_006:.1f}%  (paper: ~47%)")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 2:
        header("Figure 2: ESR vs. N  (ε=0.06)")
        t0 = time.time()

        N_vals  = [2, 4, 6, 8, 10, 12]
        K_list  = [4, 8, 16]
        eps     = 0.06
        esr_data = exp_fig2(N_vals, K_list, eps, cfg, MC_FIG2, base_seed=2000)
        plot_fig2(N_vals, esr_data, outdir=outdir)

        # Key result: ESR at N=6, K=8
        idx_n6 = N_vals.index(6)
        esr_k8_n6 = esr_data[8][idx_n6]
        results_summary["ESR N=6,K=8"] = f"{esr_k8_n6:.1f}%  (paper: ~35–47%)"
        print(f"\n  Key result: ESR at N=6, K=8 = {esr_k8_n6:.1f}%")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 3:
        header("Figure 3: BCD Convergence  (N=6, K=8, ε=0.06)")
        t0 = time.time()

        histories = exp_fig3(K=8, N=6, eps=0.06, cfg=cfg, n_runs=5, base_seed=3000)
        plot_fig3(histories, outdir=outdir)

        avg_iters = np.mean([len(h) for h in histories])
        all_mono  = all(
            all(h[i] >= h[i+1] - 1e-9 for i in range(len(h)-1))
            for h in histories
        )
        results_summary["Avg BCD iters"] = f"{avg_iters:.1f}  (paper: 12–16)"
        results_summary["Monotone"] = str(all_mono)
        print(f"\n  Avg iterations: {avg_iters:.1f}  (paper: 12–16)")
        print(f"  All runs monotone: {all_mono}")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 4:
        header("Figure 4: Scaling Law E* ∝ K^α · N^β")
        t0 = time.time()

        K_vals   = [4, 6, 8, 10, 12, 16, 20]
        N_list   = [4, 6, 8]
        eps_list = [0.04, 0.06, 0.08, 0.10, 0.12]

        mc_means, alpha, beta, A_fit, R2, K_vals_out, eps_plot = \
            exp_fig4(K_vals, N_list, eps_list, cfg, MC_MAIN, base_seed=4000)
        plot_fig4(K_vals, mc_means, alpha, beta, A_fit, outdir=outdir)

        results_summary["α (K exponent)"] = f"{alpha:.2f}  (paper: 1.34)"
        results_summary["β (N exponent)"] = f"{beta:.2f}  (paper: 0.28)"
        results_summary["R²"]             = f"{R2:.3f}  (paper: 0.95)"
        print(f"\n  E* ≈ {A_fit:.4f} · K^{alpha:.2f} · N^{beta:.2f}  R²={R2:.3f}")
        print(f"  Paper: α=1.34, β=0.28, R²=0.95")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Figure 5 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 5:
        header("Figure 5: Straggler Robustness  (N=6, K=8)")
        t0 = time.time()

        eps_vals = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
        K, N     = 8, 6

        E_strag, E_unif, E_base, strag_share = \
            exp_fig5(eps_vals, K, N, cfg, MC_FIG5, base_seed=5000)
        plot_fig5(eps_vals, E_strag, E_unif, E_base, strag_share, outdir=outdir)

        # Key results: gap at ε=0.08
        idx_008 = list(eps_vals).index(0.08)
        gap_dB  = 10*np.log10(E_base[idx_008]*1000) - 10*np.log10(E_strag[idx_008]*1000)
        share_007 = strag_share[list(eps_vals).index(0.07)]
        results_summary["Straggler gap at ε=0.08"]   = f"{gap_dB:.1f} dB  (paper: ~3.1 dBm)"
        results_summary["Straggler share at ε=0.07"]  = f"{share_007:.1f}%  (paper: <2%)"
        print(f"\n  Energy gap at ε=0.08: {gap_dB:.2f} dB  (paper: ~3.1 dBm)")
        print(f"  Straggler power share at ε=0.07: {share_007:.1f}%  (paper: <2%)")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Figure 6 ──────────────────────────────────────────────────────────────
    if args.fig is None or args.fig == 6:
        header("Figure 6: APV Candidate Count Ablation  (N=6, K=8, ε=0.06)")
        t0 = time.time()

        C_vals   = [5, 10, 20, 40, 80, 160]
        K, N, eps = 8, 6, 0.06
        E_means  = exp_fig6(C_vals, K, N, eps, cfg, MC_FIG6, base_seed=6000)
        plot_fig6(C_vals, E_means, outdir=outdir)

        # Key result: plateau gain C=40→160
        idx_40  = C_vals.index(40)
        idx_160 = C_vals.index(160)
        plateau_dB = abs(10*np.log10(E_means[idx_40]*1000)
                         - 10*np.log10(E_means[idx_160]*1000))
        results_summary["Plateau gain C=40→160"] = (
            f"{plateau_dB:.3f} dB  (paper: <0.1 dBm)")
        print(f"\n  Plateau gain C=40→160: {plateau_dB:.4f} dB  (paper: <0.1 dBm)")
        print(f"  Time: {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    if args.fig is None:
        header(f"Summary (mode: {mode})")
        for k, v in results_summary.items():
            print(f"  {k:<35} {v}")
        print(f"\n  Total time: {time.time()-t_total:.1f}s")
        print(f"  All figures saved to {outdir}/\n")


if __name__ == "__main__":
    main()
