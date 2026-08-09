"""
experiments.py
==============
Monte Carlo wrappers for the four comparison schemes and all
experiments in the paper (Figs 1–6).

Scheme definitions
------------------
proposed     : FAA + BCD power control           (Section III)
faa_maxpow   : FAA + fixed max power             ([5] baseline)
fpa_pc       : Fixed uniform array + power ctrl  (no APV)
fpa_maxpow   : Fixed uniform array + max power   (conventional AirComp)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from .config import SystemConfig
from .channel import (make_channel, make_channel_straggler,
                      uniform_port_positions, build_channel_matrix)
from .bcd import run_bcd, compute_mse


# ─────────────────────────────────────────────────────────────────────────────
# Per-realisation scheme runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_proposed(K, N, cfg, rng, eps=0.06):
    """
    FAA + Power Control (proposed scheme).

    For infeasible channel realisations (MSE at Pmax > ε), the problem
    P0 has no solution — the scheme must use Pmax and the energy saving
    is zero. This matches the paper's convention for computing mean ESR.
    """
    H, _, bk, _ = make_channel(K, N, cfg, rng)
    # Check feasibility at Pmax
    from .bcd import step_S1_mmse, step_S3_precoders
    p0   = np.ones(K) * cfg.Pmax
    tau0 = np.ones(K, dtype=complex)
    m0   = step_S1_mmse(tau0, p0, H, cfg.sigma2)
    tau0 = step_S3_precoders(m0, H)
    mse_max = compute_mse(m0, tau0, p0, H, cfg.sigma2)
    if mse_max > eps:
        return float(K * cfg.Pmax)   # infeasible: must use max power
    E, _ = run_bcd(H, bk, K, cfg, rng, eps=eps, do_apv=True)
    return E


def _run_fpa_pc(K, N, cfg, rng, eps=0.06):
    """
    Fixed Planar Array + Power Control.
    Same feasibility handling as proposed scheme.
    """
    H, _, bk, _ = make_channel(K, N, cfg, rng)
    from .bcd import step_S1_mmse, step_S3_precoders
    p0   = np.ones(K) * cfg.Pmax
    tau0 = np.ones(K, dtype=complex)
    m0   = step_S1_mmse(tau0, p0, H, cfg.sigma2)
    tau0 = step_S3_precoders(m0, H)
    mse_max = compute_mse(m0, tau0, p0, H, cfg.sigma2)
    if mse_max > eps:
        return float(K * cfg.Pmax)
    E, _ = run_bcd(H, bk, K, cfg, rng, eps=eps, do_apv=False)
    return E


def _run_fpa_maxpow(K, N, cfg, rng, eps=0.06):
    """Fixed Planar Array + Maximum Power (conventional AirComp)."""
    return float(K * cfg.Pmax)


def _run_faa_maxpow(K, N, cfg, rng, eps=0.06):
    """
    FAA + Maximum Power ([5] Zhang et al. baseline).

    Zhang et al. optimise antenna position (APV) to minimise MSE, but
    never reduce transmit power below Pmax. Since total transmit energy
    is K*Pmax regardless of antenna position (repositioning changes MSE,
    not power), this is numerically identical to fpa_maxpow in terms of
    ENERGY -- the two curves coincide in Fig. 1, which is precisely the
    point the paper makes: repositioning alone cannot reduce energy
    without active power control (see Proposition 2(ii) and Q1 in
    Section V-A).
    """
    return float(K * cfg.Pmax)


def _run_straggler_proposed(K, N, cfg, rng, eps=0.06):
    """Proposed scheme with heterogeneous (straggler) channel."""
    H, _, bk, _ = make_channel_straggler(K, N, cfg, rng)
    from .bcd import step_S1_mmse, step_S3_precoders
    p0   = np.ones(K) * cfg.Pmax
    tau0 = np.ones(K, dtype=complex)
    m0   = step_S1_mmse(tau0, p0, H, cfg.sigma2)
    tau0 = step_S3_precoders(m0, H)
    if compute_mse(m0, tau0, p0, H, cfg.sigma2) > eps:
        return float(K * cfg.Pmax)
    E, _ = run_bcd(H, bk, K, cfg, rng, eps=eps, do_apv=True)
    return E


def _straggler_power_share(K, N, cfg, rng, eps=0.06):
    """
    Return the straggler device's power fraction p[0] / Σp_k
    at BCD convergence (for the right-axis of Fig. 5).
    """
    from .bcd import (step_S1_mmse, step_S2_power, step_S3_precoders,
                      step_S4_apv)
    H, _, bk, _ = make_channel_straggler(K, N, cfg, rng)
    p   = np.ones(K) * cfg.Pmax
    tau = np.ones(K, dtype=complex)
    m   = np.ones(N, dtype=complex) / np.sqrt(N)
    E_prev = float(np.sum(p))
    for _ in range(cfg.bcd_max_iter):
        m     = step_S1_mmse(tau, p, H, cfg.sigma2)
        p_new = step_S2_power(m, H, cfg, eps=eps)
        p     = cfg.rho * p_new + (1.0 - cfg.rho) * p
        tau   = step_S3_precoders(m, H)
        H     = step_S4_apv(m, tau, p, H, bk, K, cfg, rng)
        E = float(np.sum(p))
        if abs(E - E_prev) / max(E_prev, 1e-14) < cfg.bcd_tol:
            break
        E_prev = E
    total = float(np.sum(p))
    return 100.0 * float(p[0]) / max(total, 1e-15)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Container for Monte Carlo results."""
    mean: float
    std: float
    se: float          # standard error = std / sqrt(MC)
    samples: np.ndarray

    @property
    def mean_dBm(self):
        return 10.0 * np.log10(self.mean * 1000.0)

    @property
    def se_dBm(self):
        """Approximate SE in dBm via linearisation."""
        return 10.0 * self.se / (self.mean * np.log(10.0))


def monte_carlo(runner_fn, K, N, cfg, MC, base_seed, eps=0.06) -> MCResult:
    """
    Run runner_fn for MC independent channel realisations.

    Each realisation gets a deterministic seed = base_seed + i, ensuring
    reproducibility without requiring a single global random state.
    """
    samples = np.empty(MC)
    for i in range(MC):
        rng = np.random.default_rng(base_seed + i)
        samples[i] = runner_fn(K, N, cfg, rng, eps=eps)
    return MCResult(
        mean    = float(np.mean(samples)),
        std     = float(np.std(samples, ddof=1)),
        se      = float(np.std(samples, ddof=1) / np.sqrt(MC)),
        samples = samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# High-level experiment functions (one per figure)
# ─────────────────────────────────────────────────────────────────────────────

def exp_fig1(eps_vals, K, N, cfg, MC, base_seed=1000):
    """
    Fig. 1: TX energy vs. MSE threshold ε for all four schemes.
    Returns dict of scheme → list of MCResult (one per ε).
    """
    results = {s: [] for s in ["proposed", "faa_maxpow", "fpa_pc", "fpa_maxpow"]}
    runners = {
        "proposed"  : _run_proposed,
        "faa_maxpow": _run_faa_maxpow,
        "fpa_pc"    : _run_fpa_pc,
        "fpa_maxpow": _run_fpa_maxpow,
    }
    for i, eps in enumerate(eps_vals):
        cfg_e = SystemConfig(**{**cfg.__dict__})  # clone (eps not in cfg)
        for name, fn in runners.items():
            r = monte_carlo(fn, K, N, cfg, MC, base_seed + i * 100, eps=eps)
            results[name].append(r)
        print(f"  ε={eps:.2f}  E*={results['proposed'][-1].mean_dBm:.2f} dBm  "
              f"ESR={100*(results['fpa_maxpow'][-1].mean - results['proposed'][-1].mean)/results['fpa_maxpow'][-1].mean:.1f}%")
    return results


def exp_fig2(N_vals, K_list, eps, cfg, MC, base_seed=2000):
    """Fig. 2: ESR vs. N for K ∈ K_list at fixed ε."""
    esr_data = {}
    for ki, K in enumerate(K_list):
        esr_list = []
        for ni, N in enumerate(N_vals):
            r_prop = monte_carlo(_run_proposed,   K, N, cfg, MC, base_seed + ki*100 + ni*10, eps=eps)
            E_base = K * cfg.Pmax
            esr    = max(100.0 * (E_base - r_prop.mean) / E_base, 0.0)
            esr_list.append(esr)
            print(f"  K={K}, N={N}  ESR={esr:.1f}%")
        esr_data[K] = esr_list
    return esr_data


def exp_fig3(K, N, eps, cfg, n_runs=5, base_seed=3000):
    """Fig. 3: BCD convergence histories for n_runs independent realisations."""
    histories = []
    for run in range(n_runs):
        rng     = np.random.default_rng(base_seed + run)
        H, _, bk, _ = make_channel(K, N, cfg, rng)
        _, hist = run_bcd(H, bk, K, cfg, rng, eps=eps, do_apv=True,
                          return_history=True)
        histories.append(hist)
        print(f"  Run {run+1}: {hist[0]*1000:.1f}→{hist[-1]*1000:.1f} mW  "
              f"({len(hist)} iters)  monotone={all(a>=b-1e-10 for a,b in zip(hist,hist[1:]))}")
    return histories


def exp_fig4(K_vals, N_list, eps_list, cfg, MC, base_seed=4000):
    """
    Fig. 4: Scaling law E* ∝ K^α N^β.
    Returns mc_means (dict N→list), and fitted (α, β, A(ε), R²).
    """
    all_logK, all_logN, all_logE = [], [], []
    mc_means = {N: [] for N in N_list}
    eps_plot = eps_list[len(eps_list) // 2]    # middle ε for the plot

    for ei, eps in enumerate(eps_list):
        for ni, N in enumerate(N_list):
            for ki, K in enumerate(K_vals):
                seed = base_seed + ei*1000 + ni*100 + ki*10
                r    = monte_carlo(_run_proposed, K, N, cfg, MC, seed, eps=eps)
                all_logK.append(np.log(K))
                all_logN.append(np.log(N))
                all_logE.append(np.log(r.mean))
                if eps == eps_plot:
                    mc_means[N].append(r.mean)
                    print(f"  K={K:2d}, N={N}, ε={eps:.2f}  "
                          f"E*={r.mean_dBm:.2f} dBm")

    # Log-linear OLS: log E = log A + α log K + β log N
    X      = np.column_stack([np.ones(len(all_logK)), all_logK, all_logN])
    coeffs, *_ = np.linalg.lstsq(X, all_logE, rcond=None)
    logA, alpha, beta = coeffs
    A_fit = np.exp(logA)

    pred   = X @ coeffs
    logE_a = np.array(all_logE)
    R2     = 1.0 - np.sum((logE_a - pred)**2) / np.sum((logE_a - logE_a.mean())**2)
    print(f"\n  Scaling law: E* ≈ {A_fit:.4f}·K^{alpha:.2f}·N^{beta:.2f}  R²={R2:.3f}")
    return mc_means, alpha, beta, A_fit, R2, K_vals, eps_plot


def exp_fig5(eps_vals, K, N, cfg, MC, share_mc=50, base_seed=5000):
    """
    Fig. 5: Straggler robustness.
    Returns E_strag, E_unif, E_base lists and straggler power share %.
    """
    E_strag, E_unif, E_base, share_list = [], [], [], []
    for i, eps in enumerate(eps_vals):
        seed = base_seed + i * 10
        r_strag = monte_carlo(_run_straggler_proposed, K, N, cfg, MC, seed, eps=eps)
        r_unif  = monte_carlo(_run_proposed,           K, N, cfg, MC, seed + 1, eps=eps)
        shares  = []
        for j in range(share_mc):
            rng = np.random.default_rng(seed + 500 + j)
            shares.append(_straggler_power_share(K, N, cfg, rng))
        E_strag.append(r_strag.mean)
        E_unif.append(r_unif.mean)
        E_base.append(K * cfg.Pmax)
        share_list.append(float(np.mean(shares)))
        print(f"  ε={eps:.2f}  E_strag={r_strag.mean_dBm:.2f} dBm  "
              f"straggler share={share_list[-1]:.1f}%")
    return E_strag, E_unif, E_base, share_list


def exp_fig6(C_vals, K, N, eps, cfg, MC, base_seed=6000):
    """
    Fig. 6: APV candidate count ablation.
    Returns list of mean converged E* for each C.
    """
    from .channel import make_channel
    E_means = []
    for C in C_vals:
        energies = []
        for i in range(MC):
            rng      = np.random.default_rng(base_seed + i)
            H, _, bk, _ = make_channel(K, N, cfg, rng)
            E, _     = run_bcd(H, bk, K, cfg, rng, eps=eps, do_apv=True, C_apv=C)
            energies.append(E)
        E_means.append(float(np.mean(energies)))
        print(f"  C={C:3d}  E*={10*np.log10(E_means[-1]*1000):.3f} dBm")
    return E_means
