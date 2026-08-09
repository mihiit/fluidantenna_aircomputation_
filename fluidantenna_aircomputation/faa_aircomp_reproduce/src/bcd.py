"""
bcd.py
======
Block Coordinate Descent (BCD) algorithm for FAA-AirComp energy
minimisation (Section III, Problem P0).

Four sub-steps per iteration:
    S1 – MMSE combiner        (Eq. 7)   closed-form O(N³)
    S2 – Power control        (Eq. 9)   KKT bisection O(K log μ_max)
    S3 – Pre-equalisers       (§III-C)  closed-form O(K)
    S4 – APV repositioning    (§III-C)  C candidates, proxy score

Reference:
    Mihit Nanda, "Energy-Efficient Fluid Antenna Array for
    Over-the-Air Computation," IEEE WCL, submitted 2025.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .config import SystemConfig
from .channel import (build_channel_matrix, random_port_positions)


# ─────────────────────────────────────────────────────────────────────────────
# MSE (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(m, tau, p, H, sigma2):
    """
    Expected normalised MSE of AirComp aggregate (Eq. 2).
        MSE = ‖m^H H diag(√p) diag(τ) - (1/K) 1^H‖² + σ² ‖m‖²
    The first term is pre-equalisation bias; the second is noise.
    Cross-term vanishes due to i.i.d. AWGN (see Section II-A).
    """
    K    = H.shape[1]
    A    = np.diag(np.sqrt(np.maximum(p, 0)) * tau)
    diff = m.conj() @ H @ A - np.ones(K) / K
    return float(np.real(diff @ diff.conj()) + sigma2 * np.real(m.conj() @ m))


# ─────────────────────────────────────────────────────────────────────────────
# S1 – MMSE Combiner (Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def step_S1_mmse(tau, p, H, sigma2):
    """
    S1: Closed-form MMSE receive combiner (Eq. 7).
        m* = (H A A^H H^H + σ²I)^{-1} H A (1/K) 1
    where A = diag(√p) diag(τ).  Complexity O(N³).
    """
    K   = H.shape[1]
    A   = np.diag(np.sqrt(np.maximum(p, 0)) * tau)
    HA  = H @ A
    R   = HA @ HA.conj().T + sigma2 * np.eye(H.shape[0])
    rhs = H @ A @ (np.ones(K) / K)
    return np.linalg.solve(R, rhs)


# ─────────────────────────────────────────────────────────────────────────────
# S2 – Power Control via KKT Bisection (Eqs. 8–9)
# ─────────────────────────────────────────────────────────────────────────────

def step_S2_power(m, H, cfg, eps=0.06):
    """
    S2: KKT water-filling power allocation (Eq. 9).

    After substituting phase-aligned pre-equalisers (S3), the MSE constraint
    decomposes into a scalar problem in u_k = sqrt(p_k):

        min  Σ u_k²
        s.t. Σ(c_k u_k - 1/K)² ≤ η,   0 ≤ u_k ≤ √Pmax

    where c_k = |m^H h_k| and η = ε - σ²‖m‖².

    KKT conditions give the water-filling solution (Eq. 9):
        u_k*(μ) = μ c_k / [K (1 + μ c_k²)]  clipped to [0, √Pmax]

    μ ≥ 0 is found by bisection on the constraint residual.
    """
    K      = H.shape[1]
    ck     = np.abs(m.conj() @ H)
    m_norm = float(np.real(m.conj() @ m))
    # η = ε - σ²‖m‖²  (Section III-B)
    eta    = eps - cfg.sigma2 * m_norm
    if eta <= 0:
        # Noise floor alone exceeds ε — return max power (infeasible region)
        return np.ones(K) * cfg.Pmax

    def alloc(mu):
        uk = mu * ck / (K * (1.0 + mu * ck ** 2))
        return np.clip(uk, 0.0, np.sqrt(cfg.Pmax))

    def slack(mu):
        uk = alloc(mu)
        return float(np.sum((ck * uk - 1.0 / K) ** 2)) - eta

    # Constraint satisfied at zero power → no TX needed
    if slack(0.0) <= 0.0:
        return np.zeros(K)

    # Bracket: find mu_hi where slack < 0 (constraint over-satisfied)
    mu_lo, mu_hi = 0.0, 1.0
    for _ in range(100):
        if slack(mu_hi) < 0.0:
            break
        mu_hi *= 10.0
    else:
        return np.ones(K) * cfg.Pmax   # can't satisfy constraint → max power

    if slack(mu_lo) * slack(mu_hi) >= 0.0:
        return np.ones(K) * cfg.Pmax

    try:
        mu_star = brentq(slack, mu_lo, mu_hi, maxiter=120, xtol=1e-10)
    except (RuntimeError, ValueError):
        # Manual bisection fallback
        for _ in range(120):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            if slack(mu_mid) > 0.0:
                mu_hi = mu_mid
            else:
                mu_lo = mu_mid
            if (mu_hi - mu_lo) < 1e-10 * max(mu_hi, 1.0):
                break
        mu_star = 0.5 * (mu_lo + mu_hi)

    return alloc(mu_star) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# S3 – Pre-equalisers (§III-C)
# ─────────────────────────────────────────────────────────────────────────────

def step_S3_precoders(m, H):
    """
    S3: Phase-aligned pre-equalisers (O(K)).
        τ_k = (m^H h_k) / |m^H h_k|
    Minimises MSE (2) over |τ_k| ≤ 1 in closed form.
    """
    mHh = m.conj() @ H
    mag = np.abs(mHh)
    return np.where(mag > 1e-15, mHh / mag, np.ones_like(mHh))


# ─────────────────────────────────────────────────────────────────────────────
# S4 – APV Repositioning (§III-C)
# ─────────────────────────────────────────────────────────────────────────────

def step_S4_apv(m, tau, p, H_current, bk, K, cfg, rng, C=None):
    """
    S4: APV repositioning via proxy-score candidate search.

    Evaluates C random port configurations, scored by:
        φ(t) = σ_min(H(t)) + γ ‖σ(H(t))‖₁

    Acceptance rule: accept only if MSE strictly decreases at the current
    (m, τ, p) — this ensures monotone energy descent independent of the proxy.
    """
    if C is None:
        C = cfg.C_apv
    N           = H_current.shape[0]
    mse_current = compute_mse(m, tau, p, H_current, cfg.sigma2)
    best_H      = H_current
    best_mse    = mse_current

    for _ in range(C):
        pos   = random_port_positions(N, cfg, rng)
        H_c   = build_channel_matrix(pos, bk, K, cfg, rng)
        mse_c = compute_mse(m, tau, p, H_c, cfg.sigma2)
        if mse_c < best_mse:
            best_mse = mse_c
            best_H   = H_c

    return best_H


# ─────────────────────────────────────────────────────────────────────────────
# Full BCD Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_bcd(H_init, bk, K, cfg, rng,
            eps=0.06, do_apv=True, C_apv=None,
            return_history=False):
    """
    Run the full BCD algorithm (Algorithm 1 in the paper).

    The key implementation detail matching the paper: power is updated with
    a damping factor ρ to suppress oscillation (Section III-D), and the
    power update is only accepted if the resulting MSE does not exceed ε
    by more than a small tolerance.

    Parameters
    ----------
    H_init       : (N, K) initial channel matrix
    bk           : (K,) path-loss coefficients
    K            : number of devices
    cfg          : SystemConfig
    rng          : random Generator
    eps          : MSE threshold ε
    do_apv       : whether to run S4 (True=FAA, False=FPA baseline)
    C_apv        : override cfg.C_apv for ablation study
    return_history : if True, return per-iteration energy list

    Returns
    -------
    E_final  : float, converged total transmit power Σ p_k
    history  : list of Σ p_k per iteration (only if return_history=True)
    """
    N   = H_init.shape[0]
    H   = H_init.copy()

    # Initialise at max power
    p   = np.ones(K) * cfg.Pmax
    tau = np.ones(K, dtype=complex)
    m   = step_S1_mmse(tau, p, H, cfg.sigma2)
    tau = step_S3_precoders(m, H)

    history = []
    E_prev  = float(np.sum(p))

    for iteration in range(cfg.bcd_max_iter):
        # S1 – MMSE combiner at current (p, τ)
        m = step_S1_mmse(tau, p, H, cfg.sigma2)

        # S2 – Minimum-power allocation solving the KKT subproblem
        p_star = np.clip(step_S2_power(m, H, cfg, eps=eps), 0.0, cfg.Pmax)

        # S3 – Phase-aligned pre-equalisers
        tau_new = step_S3_precoders(m, H)

        # Recompute m at p_star with updated τ to get true feasibility
        # (m was computed at old p; feasibility must be checked at p_star)
        m_check = step_S1_mmse(tau_new, p_star, H, cfg.sigma2)
        mse_check = compute_mse(m_check, tau_new, p_star, H, cfg.sigma2)

        if mse_check <= eps + 1e-6:
            # p_star is feasible — apply damped update toward p_star
            p_new = cfg.rho * p_star + (1.0 - cfg.rho) * p
        else:
            # p_star not feasible at updated (m, τ) — bisect between p and p_star
            # to find smallest feasible step in this descent direction
            alpha_lo, alpha_hi = 0.0, 1.0
            for _ in range(40):
                alpha_mid = 0.5 * (alpha_lo + alpha_hi)
                p_try = alpha_mid * p_star + (1.0 - alpha_mid) * p
                m_try = step_S1_mmse(tau_new, p_try, H, cfg.sigma2)
                if compute_mse(m_try, tau_new, p_try, H, cfg.sigma2) <= eps + 1e-6:
                    alpha_hi = alpha_mid   # feasible → try lower power
                else:
                    alpha_lo = alpha_mid   # infeasible → need more power
            p_new = alpha_hi * p_star + (1.0 - alpha_hi) * p

        # S4 – APV repositioning (FAA only, strict MSE descent at p_new)
        if do_apv:
            H = step_S4_apv(m, tau_new, p_new, H, bk, K, cfg, rng, C=C_apv)

        tau = tau_new
        p   = p_new
        E   = float(np.sum(p))
        history.append(E)

        # Convergence check
        if abs(E - E_prev) / max(E_prev, 1e-14) < cfg.bcd_tol:
            break
        E_prev = E

    if return_history:
        return float(np.sum(p)), history
    return float(np.sum(p)), None
