"""
src/bcd.py
==========

Block Coordinate Descent (BCD) algorithm for FAA-AirComp energy
minimisation.

Physical-channel rule
---------------------
For ONE physical channel realization:

    - bk is fixed.
    - g is fixed.
    - phi is fixed.
    - Only antenna-port positions change during APV.
    - Every APV candidate is generated using the SAME
      multipath realization (g, phi).

Optimization variables
----------------------
    S1 : MMSE receive combiner m
    S2 : transmit power p
    S3 : phase pre-equalisers tau
    S4 : antenna-port positions (APV)

Core invariants
---------------
For every ACCEPTED state:

    1. MSE <= epsilon
    2. E = sum(p)
    3. E_new <= E_previous

No infeasible or energy-increasing candidate is accepted.

The implementation is intentionally strict:
numerical failures are raised rather than silently converted
into successful observations.

This file is compatible with the scaling experiment:

    K = [4, 8, 12, 16]
    N = [4, 6, 8]
    epsilon = 0.08
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .config import SystemConfig
from .channel import (
    build_channel_matrix,
    random_port_positions,
)


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

FEAS_TOL = 1e-9

ENERGY_TOL = 1e-12

ZERO_TOL = 1e-15

RELATIVE_ZERO = 1e-14


# =============================================================================
# MSE
# =============================================================================

def compute_mse(
    m,
    tau,
    p,
    H,
    sigma2,
):
    """
    Compute normalized AirComp MSE.

    MSE =
        || m^H H diag(sqrt(p)) diag(tau)
           - (1/K) 1^H ||^2
        + sigma2 ||m||^2
    """

    H = np.asarray(
        H,
        dtype=complex,
    )

    m = np.asarray(
        m,
        dtype=complex,
    )

    tau = np.asarray(
        tau,
        dtype=complex,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    K = H.shape[1]

    if m.shape != (H.shape[0],):
        raise ValueError(
            f"Invalid m shape: "
            f"expected {(H.shape[0],)}, "
            f"got {m.shape}"
        )

    if tau.shape != (K,):
        raise ValueError(
            f"Invalid tau shape: "
            f"expected {(K,)}, "
            f"got {tau.shape}"
        )

    if p.shape != (K,):
        raise ValueError(
            f"Invalid p shape: "
            f"expected {(K,)}, "
            f"got {p.shape}"
        )

    amplitudes = (
        np.sqrt(
            np.maximum(
                p,
                0.0,
            )
        )
        * tau
    )

    effective = m.conj() @ H

    diff = (
        effective * amplitudes
        - np.ones(
            K,
            dtype=complex,
        ) / K
    )

    mse_signal = float(
        np.real(
            diff @ diff.conj()
        )
    )

    mse_noise = float(
        sigma2
        * np.real(
            m.conj() @ m
        )
    )

    mse = (
        mse_signal
        + mse_noise
    )

    if not np.isfinite(mse):
        raise FloatingPointError(
            f"Computed non-finite MSE: {mse}"
        )

    return float(mse)


# =============================================================================
# FEASIBILITY
# =============================================================================

def is_feasible(
    mse,
    eps,
    tol=FEAS_TOL,
):
    """
    Return True iff:

        MSE <= eps + tol
    """

    if not np.isfinite(mse):
        return False

    return bool(
        mse <= eps + tol
    )


# =============================================================================
# ENERGY
# =============================================================================

def compute_energy(p):
    """
    Compute total transmit energy/power:

        E = sum_k p_k
    """

    p = np.asarray(
        p,
        dtype=float,
    )

    if not np.all(
        np.isfinite(p)
    ):
        raise FloatingPointError(
            "Power vector contains "
            "non-finite values."
        )

    if np.any(p < -ZERO_TOL):
        raise FloatingPointError(
            "Power vector contains "
            f"negative values: min={np.min(p)}"
        )

    return float(
        np.sum(
            np.maximum(
                p,
                0.0,
            )
        )
    )


# =============================================================================
# S1 – MMSE COMBINER
# =============================================================================

def step_S1_mmse(
    tau,
    p,
    H,
    sigma2,
):
    """
    MMSE receive combiner for fixed p and tau.
    """

    H = np.asarray(
        H,
        dtype=complex,
    )

    tau = np.asarray(
        tau,
        dtype=complex,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    K = H.shape[1]

    amplitudes = (
        np.sqrt(
            np.maximum(
                p,
                0.0,
            )
        )
        * tau
    )

    HA = (
        H
        * amplitudes[np.newaxis, :]
    )

    N = H.shape[0]

    R = (
        HA @ HA.conj().T
        + sigma2
        * np.eye(
            N,
            dtype=complex,
        )
    )

    rhs = (
        HA
        @ (
            np.ones(
                K,
                dtype=complex,
            )
            / K
        )
    )

    try:

        m = np.linalg.solve(
            R,
            rhs,
        )

    except np.linalg.LinAlgError as exc:

        raise FloatingPointError(
            "S1 MMSE linear solve failed."
        ) from exc

    if not np.all(
        np.isfinite(m)
    ):
        raise FloatingPointError(
            "S1 produced non-finite combiner."
        )

    return m


# =============================================================================
# S2 – POWER CONTROL
# =============================================================================

def step_S2_power(
    m,
    H,
    cfg: SystemConfig,
    eps=0.08,
    return_diagnostics=False,
):
    """
    Minimum transmit-power solution for fixed m.

        c_k = |m^H h_k|
        u_k = sqrt(p_k)

    minimize:

        sum_k u_k^2

    subject to:

        sum_k (c_k u_k - 1/K)^2
        + sigma2 ||m||^2
        <= eps

        0 <= u_k <= sqrt(Pmax)

    The power solution is obtained from the KKT multiplier.
    """

    H = np.asarray(
        H,
        dtype=complex,
    )

    m = np.asarray(
        m,
        dtype=complex,
    )

    K = H.shape[1]

    mHh = (
        m.conj()
        @ H
    )

    ck = np.abs(
        mHh
    )

    if not np.all(
        np.isfinite(ck)
    ):
        raise FloatingPointError(
            "S2 channel coefficients are non-finite."
        )

    m_norm_sq = float(
        np.real(
            m.conj() @ m
        )
    )

    noise_term = float(
        cfg.sigma2
        * m_norm_sq
    )

    eta = float(
        eps
        - noise_term
    )

    sqrt_Pmax = float(
        np.sqrt(
            cfg.Pmax
        )
    )

    # -------------------------------------------------------------------------
    # Noise alone exceeds the target.
    # -------------------------------------------------------------------------

    if eta <= 0.0:

        p = (
            np.ones(K)
            * cfg.Pmax
        )

        constraint_value = float(
            np.sum(
                (
                    ck * sqrt_Pmax
                    - 1.0 / K
                ) ** 2
            )
        )

        actual_mse = (
            constraint_value
            + noise_term
        )

        diagnostics = {
            "eta": eta,
            "noise_term": noise_term,
            "constraint_value": constraint_value,
            "mse": actual_mse,
            "feasible": False,
            "mu": None,
            "residual": None,
        }

        if return_diagnostics:
            return p, diagnostics

        return p

    # -------------------------------------------------------------------------
    # Power allocation for a given KKT multiplier.
    # -------------------------------------------------------------------------

    def alloc(mu):

        denominator = (
            K
            * (
                1.0
                + mu * ck ** 2
            )
        )

        uk = np.zeros(
            K,
            dtype=float,
        )

        nonzero = (
            denominator
            > ZERO_TOL
        )

        uk[nonzero] = (
            mu
            * ck[nonzero]
            / denominator[nonzero]
        )

        uk = np.clip(
            uk,
            0.0,
            sqrt_Pmax,
        )

        return uk

    # -------------------------------------------------------------------------
    # Constraint residual.
    # -------------------------------------------------------------------------

    def residual(mu):

        uk = alloc(mu)

        value = float(
            np.sum(
                (
                    ck * uk
                    - 1.0 / K
                ) ** 2
            )
        )

        return float(
            value - eta
        )

    residual_zero = residual(
        0.0
    )

    # -------------------------------------------------------------------------
    # Zero-power solution already satisfies the constraint.
    # -------------------------------------------------------------------------

    if residual_zero <= 0.0:

        p = np.zeros(
            K,
            dtype=float,
        )

        constraint_value = float(
            np.sum(
                (
                    -1.0 / K
                ) ** 2
            )
        )

        actual_mse = (
            constraint_value
            + noise_term
        )

        diagnostics = {
            "eta": eta,
            "noise_term": noise_term,
            "constraint_value": constraint_value,
            "mse": actual_mse,
            "feasible": (
                actual_mse
                <= eps + FEAS_TOL
            ),
            "mu": 0.0,
            "residual": residual_zero,
        }

        if return_diagnostics:
            return p, diagnostics

        return p

    # -------------------------------------------------------------------------
    # Find multiplier bracket.
    # -------------------------------------------------------------------------

    mu_lo = 0.0
    mu_hi = 1.0

    bracket_found = False

    for _ in range(160):

        value = residual(
            mu_hi
        )

        if value <= 0.0:

            bracket_found = True
            break

        mu_hi *= 10.0

        if not np.isfinite(
            mu_hi
        ):
            break

    # -------------------------------------------------------------------------
    # Even maximum power cannot satisfy the constraint.
    # -------------------------------------------------------------------------

    if not bracket_found:

        uk = (
            np.ones(K)
            * sqrt_Pmax
        )

        p = uk ** 2

        constraint_value = float(
            np.sum(
                (
                    ck * uk
                    - 1.0 / K
                ) ** 2
            )
        )

        actual_mse = (
            constraint_value
            + noise_term
        )

        diagnostics = {
            "eta": eta,
            "noise_term": noise_term,
            "constraint_value": constraint_value,
            "mse": actual_mse,
            "feasible": (
                actual_mse
                <= eps + FEAS_TOL
            ),
            "mu": None,
            "residual": residual(
                mu_hi
            ),
        }

        if return_diagnostics:
            return p, diagnostics

        return p

    # -------------------------------------------------------------------------
    # Solve KKT multiplier.
    # -------------------------------------------------------------------------

    try:

        mu_star = brentq(
            residual,
            mu_lo,
            mu_hi,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=300,
        )

    except (
        RuntimeError,
        ValueError,
    ):

        # Deterministic bisection fallback.
        for _ in range(300):

            mu_mid = (
                0.5
                * (
                    mu_lo
                    + mu_hi
                )
            )

            if residual(
                mu_mid
            ) > 0.0:

                mu_lo = mu_mid

            else:

                mu_hi = mu_mid

            if (
                abs(
                    mu_hi
                    - mu_lo
                )
                <= 1e-13
                * max(
                    1.0,
                    abs(mu_hi),
                )
            ):
                break

        mu_star = (
            0.5
            * (
                mu_lo
                + mu_hi
            )
        )

    # -------------------------------------------------------------------------
    # Final allocation.
    # -------------------------------------------------------------------------

    uk = alloc(
        mu_star
    )

    p = np.clip(
        uk ** 2,
        0.0,
        cfg.Pmax,
    )

    constraint_value = float(
        np.sum(
            (
                ck * uk
                - 1.0 / K
            ) ** 2
        )
    )

    actual_mse = (
        constraint_value
        + noise_term
    )

    feasible = (
        actual_mse
        <= eps + FEAS_TOL
    )

    diagnostics = {
        "eta": eta,
        "noise_term": noise_term,
        "constraint_value": constraint_value,
        "mse": actual_mse,
        "feasible": feasible,
        "mu": mu_star,
        "residual": residual(
            mu_star
        ),
    }

    if return_diagnostics:
        return p, diagnostics

    return p


# =============================================================================
# S3 – PRE-EQUALISERS
# =============================================================================

def step_S3_precoders(
    m,
    H,
):
    """
    Phase cancellation:

        tau_k =
            conj(m^H h_k)
            / |m^H h_k|

    For zero effective channels, tau_k = 1.
    """

    m = np.asarray(
        m,
        dtype=complex,
    )

    H = np.asarray(
        H,
        dtype=complex,
    )

    mHh = (
        m.conj()
        @ H
    )

    mag = np.abs(
        mHh
    )

    tau = np.ones(
        H.shape[1],
        dtype=complex,
    )

    mask = (
        mag > ZERO_TOL
    )

    tau[mask] = (
        np.conj(
            mHh[mask]
        )
        / mag[mask]
    )

    if not np.all(
        np.isfinite(tau)
    ):
        raise FloatingPointError(
            "S3 produced non-finite phase precoders."
        )

    return tau


# =============================================================================
# FIXED-CHANNEL LOCAL OPTIMISATION
# =============================================================================

def optimise_fixed_channel(
    H,
    cfg: SystemConfig,
    eps=0.08,
    initial_p=None,
    initial_tau=None,
    initial_m=None,
    max_inner_iter=3,
):
    """
    Locally optimize m, tau and p for a fixed channel H.

    Returns the BEST FEASIBLE state found.

    The function never returns an energy-increasing state if a
    feasible initial state exists.

    This is used by APV candidate evaluation.
    """

    H = np.asarray(
        H,
        dtype=complex,
    )

    K = H.shape[1]

    # -------------------------------------------------------------------------
    # Initial power.
    # -------------------------------------------------------------------------

    if initial_p is None:

        p = (
            np.ones(K)
            * cfg.Pmax
        )

    else:

        p = np.clip(
            np.asarray(
                initial_p,
                dtype=float,
            ),
            0.0,
            cfg.Pmax,
        ).copy()

    # -------------------------------------------------------------------------
    # Initial combiner.
    # -------------------------------------------------------------------------

    if initial_m is None:

        if initial_tau is None:

            tau = np.ones(
                K,
                dtype=complex,
            )

        else:

            tau = np.asarray(
                initial_tau,
                dtype=complex,
            ).copy()

        m = step_S1_mmse(
            tau,
            p,
            H,
            cfg.sigma2,
        )

    else:

        m = np.asarray(
            initial_m,
            dtype=complex,
        ).copy()

        if initial_tau is None:

            tau = step_S3_precoders(
                m,
                H,
            )

        else:

            tau = np.asarray(
                initial_tau,
                dtype=complex,
            ).copy()

    # Always normalize phase using current m.
    tau = step_S3_precoders(
        m,
        H,
    )

    # -------------------------------------------------------------------------
    # Current state.
    # -------------------------------------------------------------------------

    mse_current = compute_mse(
        m,
        tau,
        p,
        H,
        cfg.sigma2,
    )

    energy_current = compute_energy(
        p
    )

    # -------------------------------------------------------------------------
    # Best feasible state.
    # -------------------------------------------------------------------------

    best_p = None
    best_tau = None
    best_m = None

    best_mse = np.inf
    best_energy = np.inf

    if is_feasible(
        mse_current,
        eps,
    ):

        best_p = p.copy()
        best_tau = tau.copy()
        best_m = m.copy()

        best_mse = float(
            mse_current
        )

        best_energy = float(
            energy_current
        )

    # -------------------------------------------------------------------------
    # Local BCD iterations.
    # -------------------------------------------------------------------------

    last_mse = mse_current

    for _ in range(
        max_inner_iter
    ):

        # =====================================================================
        # S1
        # =====================================================================

        m_candidate = step_S1_mmse(
            tau,
            p,
            H,
            cfg.sigma2,
        )

        # =====================================================================
        # S3
        # =====================================================================

        tau_candidate = step_S3_precoders(
            m_candidate,
            H,
        )

        # =====================================================================
        # S2
        # =====================================================================

        (
            p_candidate,
            s2_diag,
        ) = step_S2_power(
            m_candidate,
            H,
            cfg,
            eps=eps,
            return_diagnostics=True,
        )

        p_candidate = np.clip(
            p_candidate,
            0.0,
            cfg.Pmax,
        )

        # =====================================================================
        # Recompute receive processing using updated power.
        # =====================================================================

        m_candidate = step_S1_mmse(
            tau_candidate,
            p_candidate,
            H,
            cfg.sigma2,
        )

        tau_candidate = step_S3_precoders(
            m_candidate,
            H,
        )

        m_candidate = step_S1_mmse(
            tau_candidate,
            p_candidate,
            H,
            cfg.sigma2,
        )

        # =====================================================================
        # Evaluate candidate.
        # =====================================================================

        mse_candidate = compute_mse(
            m_candidate,
            tau_candidate,
            p_candidate,
            H,
            cfg.sigma2,
        )

        energy_candidate = compute_energy(
            p_candidate
        )

        last_mse = mse_candidate

        # =====================================================================
        # Only feasible states can become the best state.
        # =====================================================================

        if (
            is_feasible(
                mse_candidate,
                eps,
            )
            and
            energy_candidate
            < best_energy
            - ENERGY_TOL
        ):

            best_p = p_candidate.copy()
            best_tau = tau_candidate.copy()
            best_m = m_candidate.copy()

            best_mse = float(
                mse_candidate
            )

            best_energy = float(
                energy_candidate
            )

        # Continue local iteration from the block solution.
        p = p_candidate
        tau = tau_candidate
        m = m_candidate

    # -------------------------------------------------------------------------
    # No feasible candidate found.
    # -------------------------------------------------------------------------

    if best_p is None:

        return (
            p.copy(),
            tau.copy(),
            m.copy(),
            float(last_mse),
            float(
                compute_energy(p)
            ),
        )

    # -------------------------------------------------------------------------
    # Return best feasible state.
    # -------------------------------------------------------------------------

    return (
        best_p,
        best_tau,
        best_m,
        best_mse,
        best_energy,
    )


# =============================================================================
# S4 – APV REPOSITIONING
# =============================================================================

def step_S4_apv(
    m,
    tau,
    p,
    H_current,
    bk,
    K,
    cfg: SystemConfig,
    rng,
    g,
    phi,
    C=None,
    eps=0.08,
    candidate_inner_iter=3,
):
    """
    APV candidate search.

    PHYSICAL-CHANNEL INVARIANT
    --------------------------

    The supplied:

        bk
        g
        phi

    remain fixed for ALL candidates.

    Only antenna-port positions are randomized.

    Therefore every candidate corresponds to the SAME underlying
    physical multipath realization with a different antenna
    configuration.

    The current accepted state is always the baseline.

    A candidate is accepted only when:

        MSE_candidate <= epsilon

    AND:

        E_candidate < E_best - ENERGY_TOL
    """

    if C is None:
        C = cfg.C_apv

    if C is None:
        raise ValueError(
            "APV candidate count C is not specified."
        )

    C = int(C)

    if C <= 0:
        raise ValueError(
            f"APV candidate count must be positive, got {C}."
        )

    if g is None or phi is None:
        raise ValueError(
            "step_S4_apv requires fixed g and phi."
        )

    # -------------------------------------------------------------------------
    # Baseline = exact currently accepted BCD state.
    # -------------------------------------------------------------------------

    baseline_H = np.asarray(
        H_current,
        dtype=complex,
    ).copy()

    baseline_p = np.asarray(
        p,
        dtype=float,
    ).copy()

    baseline_tau = np.asarray(
        tau,
        dtype=complex,
    ).copy()

    baseline_m = np.asarray(
        m,
        dtype=complex,
    ).copy()

    baseline_mse = compute_mse(
        baseline_m,
        baseline_tau,
        baseline_p,
        baseline_H,
        cfg.sigma2,
    )

    baseline_energy = compute_energy(
        baseline_p
    )

    if not is_feasible(
        baseline_mse,
        eps,
    ):

        raise RuntimeError(
            "APV baseline is infeasible. "
            "APV requires a feasible accepted state."
        )

    # -------------------------------------------------------------------------
    # Best state begins as the current state.
    # -------------------------------------------------------------------------

    best_H = baseline_H.copy()
    best_p = baseline_p.copy()
    best_tau = baseline_tau.copy()
    best_m = baseline_m.copy()

    best_mse = float(
        baseline_mse
    )

    best_energy = float(
        baseline_energy
    )

    improved_candidate = -1

    candidate_energies = []
    candidate_mses = []

    feasible_candidates = 0

    # -------------------------------------------------------------------------
    # APV candidates.
    # -------------------------------------------------------------------------

    for candidate_idx in range(
        C
    ):

        # =====================================================================
        # ONLY antenna positions change.
        # =====================================================================

        pos = random_port_positions(
            baseline_H.shape[0],
            cfg,
            rng,
        )

        # =====================================================================
        # SAME bk, SAME g, SAME phi.
        # =====================================================================

        H_candidate = build_channel_matrix(
            pos_wl=pos,
            bk=bk,
            K=K,
            cfg=cfg,
            g=g,
            phi=phi,
        )

        H_candidate = np.asarray(
            H_candidate,
            dtype=complex,
        )

        if H_candidate.shape != baseline_H.shape:
            raise RuntimeError(
                "APV candidate H shape changed unexpectedly: "
                f"baseline={baseline_H.shape}, "
                f"candidate={H_candidate.shape}"
            )

        # =====================================================================
        # Optimize candidate.
        # =====================================================================

        (
            p_candidate,
            tau_candidate,
            m_candidate,
            mse_candidate,
            energy_candidate,
        ) = optimise_fixed_channel(
            H_candidate,
            cfg,
            eps=eps,
            initial_p=baseline_p,
            initial_tau=baseline_tau,
            initial_m=None,
            max_inner_iter=candidate_inner_iter,
        )

        candidate_energies.append(
            float(energy_candidate)
        )

        candidate_mses.append(
            float(mse_candidate)
        )

        # =====================================================================
        # Reject infeasible candidates.
        # =====================================================================

        if not is_feasible(
            mse_candidate,
            eps,
        ):
            continue

        feasible_candidates += 1

        # =====================================================================
        # Accept only strict energy improvements.
        # =====================================================================

        if (
            energy_candidate
            < best_energy
            - ENERGY_TOL
        ):

            best_H = H_candidate.copy()

            best_p = np.asarray(
                p_candidate,
                dtype=float,
            ).copy()

            best_tau = np.asarray(
                tau_candidate,
                dtype=complex,
            ).copy()

            best_m = np.asarray(
                m_candidate,
                dtype=complex,
            ).copy()

            best_mse = float(
                mse_candidate
            )

            best_energy = float(
                energy_candidate
            )

            improved_candidate = (
                candidate_idx + 1
            )

    # -------------------------------------------------------------------------
    # Diagnostics.
    # -------------------------------------------------------------------------

    improved = (
        best_energy
        < baseline_energy
        - ENERGY_TOL
    )

    if candidate_energies:

        min_candidate_energy = float(
            np.min(
                candidate_energies
            )
        )

        mean_candidate_energy = float(
            np.mean(
                candidate_energies
            )
        )

    else:

        min_candidate_energy = (
            baseline_energy
        )

        mean_candidate_energy = (
            baseline_energy
        )

    gain = float(
        baseline_energy
        - best_energy
    )

    gain_pct = (
        100.0
        * gain
        / max(
            baseline_energy,
            RELATIVE_ZERO,
        )
    )

    print(
        f"[APV] "
        f"C={C} "
        f"current_E={baseline_energy:.9f} "
        f"current_MSE={baseline_mse:.9f} "
        f"best_E={best_energy:.9f} "
        f"best_MSE={best_mse:.9f} "
        f"gain={gain:.9e} "
        f"gain_pct={gain_pct:.6f}% "
        f"candidate_min_E={min_candidate_energy:.9f} "
        f"candidate_mean_E={mean_candidate_energy:.9f} "
        f"feasible_candidates={feasible_candidates}/{C} "
        f"improved={improved} "
        f"candidate={improved_candidate}"
    )

    return (
        best_H,
        best_p,
        best_tau,
        best_m,
        best_mse,
        best_energy,
    )


# =============================================================================
# FULL BCD
# =============================================================================

def run_bcd(
    H_init,
    bk,
    K,
    cfg: SystemConfig,
    rng,
    eps=0.08,
    do_apv=True,
    C_apv=None,
    return_history=False,
    g=None,
    phi=None,
):
    """
    Full FAA-AirComp BCD.

    Accepted-state invariants:

        MSE <= eps

        E = sum(p)

        E_new <= E_previous

    For FAA mode:

        do_apv=True
        g is fixed
        phi is fixed

    For FPA mode:

        do_apv=False
        H remains fixed

    The function raises on numerical inconsistency rather than
    returning a silently corrupted result.
    """

    # =========================================================================
    # VALIDATE INPUTS
    # =========================================================================

    H = np.asarray(
        H_init,
        dtype=complex,
    ).copy()

    bk = np.asarray(
        bk,
        dtype=float,
    ).copy()

    if H.ndim != 2:
        raise ValueError(
            f"H must be 2-D, got ndim={H.ndim}"
        )

    if H.shape[1] != K:
        raise ValueError(
            f"H has {H.shape[1]} users but K={K}."
        )

    if bk.shape != (K,):
        raise ValueError(
            f"bk has shape {bk.shape} "
            f"but expected {(K,)}."
        )

    if not np.all(
        np.isfinite(H.real)
    ) or not np.all(
        np.isfinite(H.imag)
    ):
        raise ValueError(
            "H contains non-finite values."
        )

    if not np.all(
        np.isfinite(bk)
    ):
        raise ValueError(
            "bk contains non-finite values."
        )

    if do_apv and (
        g is None
        or phi is None
    ):
        raise ValueError(
            "run_bcd(..., do_apv=True) requires fixed "
            "multipath arrays g and phi."
        )

    # =========================================================================
    # INITIAL STATE
    # =========================================================================

    p = (
        np.ones(K)
        * cfg.Pmax
    )

    tau = np.ones(
        K,
        dtype=complex,
    )

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

    initial_mse = compute_mse(
        m,
        tau,
        p,
        H,
        cfg.sigma2,
    )

    initial_energy = compute_energy(
        p
    )

    print(
        f"[INITIAL] "
        f"E={initial_energy:.9f} "
        f"MSE={initial_mse:.9f} "
        f"target={eps:.9f}"
    )

    # =========================================================================
    # HISTORY
    # =========================================================================

    history = []

    # IMPORTANT:
    #
    # The initial state is not placed in history because it may be
    # infeasible. History contains ONLY accepted feasible states.
    #
    # This makes the monotonicity test meaningful:
    #
    #     history[i] >= history[i+1]
    #

    E_current = float(
        initial_energy
    )

    # =========================================================================
    # OUTER BCD ITERATIONS
    # =========================================================================

    for iteration in range(
        cfg.bcd_max_iter
    ):

        print(
            f"\n----- BCD ITERATION "
            f"{iteration + 1:02d} -----"
        )

        E_before_iteration = float(
            E_current
        )

        # =====================================================================
        # S1
        # =====================================================================

        m_s1 = step_S1_mmse(
            tau,
            p,
            H,
            cfg.sigma2,
        )

        mse_s1 = compute_mse(
            m_s1,
            tau,
            p,
            H,
            cfg.sigma2,
        )

        print(
            f"[S1] "
            f"MSE={mse_s1:.9f}"
        )

        # =====================================================================
        # S3
        # =====================================================================

        tau_s3 = step_S3_precoders(
            m_s1,
            H,
        )

        mse_s3 = compute_mse(
            m_s1,
            tau_s3,
            p,
            H,
            cfg.sigma2,
        )

        print(
            f"[S3] "
            f"MSE={mse_s3:.9f}"
        )

        # =====================================================================
        # S2
        # =====================================================================

        (
            p_star,
            s2_diag,
        ) = step_S2_power(
            m_s1,
            H,
            cfg,
            eps=eps,
            return_diagnostics=True,
        )

        p_star = np.clip(
            p_star,
            0.0,
            cfg.Pmax,
        )

        E_star = compute_energy(
            p_star
        )

        print(
            f"[S2] "
            f"E_old={np.sum(p):.9f} "
            f"E_star={E_star:.9f} "
            f"min_p={np.min(p_star):.9e} "
            f"max_p={np.max(p_star):.9e}"
        )

        print(
            f"[S2-MATH] "
            f"constraint="
            f"{s2_diag['constraint_value']:.12f} "
            f"noise="
            f"{s2_diag['noise_term']:.12f} "
            f"MSE="
            f"{s2_diag['mse']:.12f} "
            f"target="
            f"{eps:.12f} "
            f"residual="
            f"{s2_diag.get('residual', np.nan):+.3e}"
        )

        # =====================================================================
        # Recompute receive processing using updated power.
        # =====================================================================

        m_star = step_S1_mmse(
            tau_s3,
            p_star,
            H,
            cfg.sigma2,
        )

        tau_star = step_S3_precoders(
            m_star,
            H,
        )

        m_star = step_S1_mmse(
            tau_star,
            p_star,
            H,
            cfg.sigma2,
        )

        mse_star = compute_mse(
            m_star,
            tau_star,
            p_star,
            H,
            cfg.sigma2,
        )

        feasible_star = is_feasible(
            mse_star,
            eps,
        )

        print(
            f"[CHECK-S2] "
            f"E_star={E_star:.9f} "
            f"MSE={mse_star:.9f} "
            f"target={eps:.9f} "
            f"error={mse_star - eps:+.3e} "
            f"feasible={feasible_star}"
        )

        # =====================================================================
        # ACCEPT / REJECT S1-S2-S3
        # =====================================================================

        accepted_s123 = False

        if (
            feasible_star
            and
            E_star
            <= E_current
            + ENERGY_TOL
        ):

            p = p_star.copy()
            tau = tau_star.copy()
            m = m_star.copy()

            E_current = float(
                E_star
            )

            accepted_s123 = True

            print(
                f"[BCD-ACCEPT] "
                f"E={E_current:.9f} "
                f"MSE={mse_star:.9f}"
            )

        else:

            print(
                f"[BCD-REJECT] "
                f"E_candidate={E_star:.9f} "
                f"MSE_candidate={mse_star:.9f} "
                f"candidate_feasible={feasible_star} "
                f"current_E={E_current:.9f}"
            )

        # =====================================================================
        # S4 APV
        # =====================================================================

        if do_apv:

            (
                H_candidate,
                p_candidate,
                tau_candidate,
                m_candidate,
                mse_candidate,
                E_candidate,
            ) = step_S4_apv(
                m=m,
                tau=tau,
                p=p,
                H_current=H,
                bk=bk,
                K=K,
                cfg=cfg,
                rng=rng,
                g=g,
                phi=phi,
                C=C_apv,
                eps=eps,
                candidate_inner_iter=3,
            )

            # -------------------------------------------------------------
            # APV accepts ONLY a strict energy improvement.
            # -------------------------------------------------------------

            if (
                is_feasible(
                    mse_candidate,
                    eps,
                )
                and
                E_candidate
                < E_current
                - ENERGY_TOL
            ):

                H = np.asarray(
                    H_candidate,
                    dtype=complex,
                ).copy()

                p = np.asarray(
                    p_candidate,
                    dtype=float,
                ).copy()

                tau = np.asarray(
                    tau_candidate,
                    dtype=complex,
                ).copy()

                m = np.asarray(
                    m_candidate,
                    dtype=complex,
                ).copy()

                E_current = float(
                    E_candidate
                )

                print(
                    f"[APV-ACCEPT] "
                    f"iter={iteration + 1:02d} "
                    f"E={E_current:.9f} "
                    f"MSE={mse_candidate:.9f}"
                )

            else:

                print(
                    f"[APV-REJECT] "
                    f"iter={iteration + 1:02d} "
                    f"E_candidate={E_candidate:.9f} "
                    f"MSE_candidate={mse_candidate:.9f} "
                    f"current_E={E_current:.9f}"
                )

        # =====================================================================
        # FINAL ACCEPTED STATE VERIFICATION
        # =====================================================================

        mse_final = compute_mse(
            m,
            tau,
            p,
            H,
            cfg.sigma2,
        )

        E_final_state = compute_energy(
            p
        )

        # ---------------------------------------------------------------------
        # Energy bookkeeping.
        # ---------------------------------------------------------------------

        if not np.isclose(
            E_final_state,
            E_current,
            rtol=1e-10,
            atol=ENERGY_TOL,
        ):

            raise RuntimeError(
                "Internal BCD inconsistency: "
                "stored energy does not match sum(p). "
                f"stored={E_current:.15e}, "
                f"actual={E_final_state:.15e}"
            )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # If no feasible state has ever been reached, stop with a numerical
        # failure. The scaling experiment must NOT record this realization
        # as a successful observation.
        # ---------------------------------------------------------------------

        if not is_feasible(
            mse_final,
            eps,
        ):

            raise RuntimeError(
                "BCD could not produce a feasible accepted state. "
                f"MSE={mse_final:.15e}, "
                f"epsilon={eps:.15e}, "
                f"energy={E_final_state:.15e}"
            )

        print(
            f"[STATE] "
            f"iter={iteration + 1:02d} "
            f"E_new={E_final_state:.9f} "
            f"MSE_new={mse_final:.9f} "
            f"target={eps:.9f} "
            f"feasible=True"
        )

        # =====================================================================
        # GLOBAL MONOTONICITY
        # =====================================================================

        if (
            E_final_state
            > E_before_iteration
            + ENERGY_TOL
        ):

            raise RuntimeError(
                "BCD monotonicity violation: "
                f"E_previous={E_before_iteration:.15e}, "
                f"E_current={E_final_state:.15e}"
            )

        # =====================================================================
        # STORE ACCEPTED ENERGY
        # =====================================================================

        history.append(
            float(
                E_final_state
            )
        )

        # =====================================================================
        # CONVERGENCE
        # =====================================================================

        relative_change = (
            abs(
                E_final_state
                - E_before_iteration
            )
            / max(
                abs(
                    E_before_iteration
                ),
                RELATIVE_ZERO,
            )
        )

        if (
            relative_change
            < cfg.bcd_tol
        ):

            print(
                f"[CONVERGED] "
                f"iter={iteration + 1:02d} "
                f"relative_energy_change="
                f"{relative_change:.3e}"
            )

            break

        E_current = float(
            E_final_state
        )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    E_final = compute_energy(
        p
    )

    final_mse = compute_mse(
        m,
        tau,
        p,
        H,
        cfg.sigma2,
    )

    final_feasible = is_feasible(
        final_mse,
        eps,
    )

    print(
        f"[FINAL] "
        f"E={E_final:.12f} "
        f"MSE={final_mse:.12f} "
        f"target={eps:.12f} "
        f"feasible={final_feasible}"
    )

    # -------------------------------------------------------------------------
    # Final hard validation.
    # -------------------------------------------------------------------------

    if not final_feasible:

        raise RuntimeError(
            "Final BCD state is infeasible. "
            f"MSE={final_mse:.15e}, "
            f"epsilon={eps:.15e}"
        )

    if not np.isfinite(
        E_final
    ) or E_final <= 0.0:

        raise RuntimeError(
            "Final BCD energy is invalid: "
            f"E={E_final}"
        )

    # -------------------------------------------------------------------------
    # History validation.
    # -------------------------------------------------------------------------

    if len(history) == 0:

        raise RuntimeError(
            "BCD finished without producing "
            "a feasible accepted history."
        )

    for i in range(
        len(history) - 1
    ):

        if (
            history[i + 1]
            > history[i]
            + ENERGY_TOL
        ):

            raise RuntimeError(
                "Final BCD history is not monotone "
                "non-increasing."
            )

    # =========================================================================
    # RETURN
    # =========================================================================

    if return_history:

        return (
            E_final,
            history,
        )

    return (
        E_final,
        None,
    )