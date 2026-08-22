"""
channel.py
==========

Channel model for FAA-AirComp system (Section II-A, Eq. (1)).

The multipath channel realization (g_kl, phi_kl) is sampled ONCE.
Changing the APV only changes the spatial phase term.

This is critical for valid APV optimization.
"""

import numpy as np
from .config import SystemConfig


def sample_path_loss(
    K: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
    d_override: np.ndarray = None,
) -> tuple:
    """
    Sample device distances and compute normalized path-loss coefficients.

    Paper:
        beta_k = d_k^{-alpha}
        d_k in [5, 20] m
        mean(beta) = 1
    """
    if d_override is not None:
        dk = np.asarray(d_override, dtype=float)
        if dk.shape != (K,):
            raise ValueError(
                f"d_override must have shape ({K},), got {dk.shape}"
            )
    else:
        dk = rng.uniform(cfg.d_min, cfg.d_max, K)

    if np.any(dk <= 0):
        raise ValueError("All device distances must be positive.")

    bk = dk ** (-cfg.path_loss_exp)

    # Paper normalization: mean(beta) = 1
    bk /= bk.mean()

    return dk, bk


def sample_multipath(
    K: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> tuple:
    """
    Sample the fixed multipath realization.

    For each device k and path l:
        g_kl ~ CN(0, 1)
        phi_kl ~ Uniform[-pi/2, pi/2]

    These values remain FIXED while APV positions are optimized.
    """
    # Complex Gaussian CN(0,1)
    g = (
        rng.standard_normal((K, cfg.Lp))
        + 1j * rng.standard_normal((K, cfg.Lp))
    ) / np.sqrt(2.0)

    # AoA / path angle
    phi = rng.uniform(
        -np.pi / 2,
        np.pi / 2,
        size=(K, cfg.Lp),
    )

    return g, phi


def build_channel_matrix(
    pos_wl: np.ndarray,
    bk: np.ndarray,
    K: int,
    cfg: SystemConfig,
    g: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Build the N x K channel matrix for a GIVEN APV.

    Implements Eq. (1):
        h_k(x) =
            sqrt(beta_k)
            sum_l g_kl exp(j 2 pi x sin(phi_kl))

    IMPORTANT:
        g and phi are NOT regenerated here.
        The same physical channel realization must be used
        for every candidate APV.
    """
    pos_wl = np.asarray(pos_wl, dtype=float)
    bk = np.asarray(bk, dtype=float)

    N = len(pos_wl)

    if bk.shape != (K,):
        raise ValueError(
            f"bk must have shape ({K},), got {bk.shape}"
        )

    if g.shape != (K, cfg.Lp):
        raise ValueError(
            f"g must have shape ({K}, {cfg.Lp}), got {g.shape}"
        )

    if phi.shape != (K, cfg.Lp):
        raise ValueError(
            f"phi must have shape ({K}, {cfg.Lp}), got {phi.shape}"
        )

    H = np.zeros((N, K), dtype=complex)

    for k in range(K):
        h_k = np.zeros(N, dtype=complex)

        for l in range(cfg.Lp):
            spatial_phase = np.exp(
                1j
                * 2.0
                * np.pi
                * pos_wl
                * np.sin(phi[k, l])
            )
            h_k += g[k, l] * spatial_phase

        H[:, k] = np.sqrt(bk[k]) * h_k

    return H


def uniform_port_positions(
    N: int,
    cfg: SystemConfig,
) -> np.ndarray:
    """
    Uniformly spaced initial positions.

    Positions are represented in wavelengths.
    The aperture is [0, L/lambda] = [0, 5].
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    if N > 1:
        max_possible_ports = int(
            np.floor(cfg.L_wl / cfg.min_sep_wl) + 1
        )
        if N > max_possible_ports:
            raise ValueError(
                f"N={N} cannot fit inside aperture "
                f"L={cfg.L_wl} lambda with minimum spacing "
                f"{cfg.min_sep_wl} lambda."
            )

    return np.linspace(0.0, cfg.L_wl, N)


def random_port_positions(
    N: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
    max_tries: int = 2000,
) -> np.ndarray:
    """
    Generate a random feasible APV.

    Positions:
        x_n in [0, L]

    Constraint:
        |x_n - x_n'| >= lambda/2

    Since positions are represented in wavelengths:
        minimum separation = 0.5
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    min_sep = cfg.min_sep_wl
    L_wl = cfg.L_wl

    # Basic feasibility check
    required_length = (N - 1) * min_sep
    if required_length > L_wl + 1e-12:
        raise ValueError(
            f"Infeasible APV: N={N}, minimum separation={min_sep}, "
            f"aperture={L_wl} lambda."
        )

    for _ in range(max_tries):
        pts = np.sort(
            rng.uniform(0.0, L_wl, N)
        )

        if N == 1 or np.min(np.diff(pts)) >= min_sep:
            return pts

    # Deterministic feasible fallback
    return uniform_port_positions(N, cfg)


def make_channel(
    K: int,
    N: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
    d_override: np.ndarray = None,
) -> tuple:
    """
    Generate ONE complete physical channel realization.

    Returns:
        H
        dk
        bk
        pos
        g
        phi

    g and phi must be reused whenever evaluating a different APV.
    """
    dk, bk = sample_path_loss(
        K,
        cfg,
        rng,
        d_override,
    )

    # Initial FPA/uniform APV
    pos = uniform_port_positions(N, cfg)

    # ONE fixed channel realization
    g, phi = sample_multipath(
        K,
        cfg,
        rng,
    )

    H = build_channel_matrix(
        pos,
        bk,
        K,
        cfg,
        g,
        phi,
    )

    return H, dk, bk, pos, g, phi


def make_channel_straggler(
    K: int,
    N: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> tuple:
    """
    Heterogeneous channel for straggler experiment.

    Device 0:
        d = 19 m

    Devices 1...K-1:
        d ~ Uniform[5, 8] m
    """
    if K < 2:
        raise ValueError("Straggler experiment requires K >= 2.")

    dk = np.empty(K)
    dk[0] = cfg.straggler_dist
    dk[1:] = rng.uniform(
        cfg.straggler_near_min,
        cfg.straggler_near_max,
        K - 1,
    )

    _, bk = sample_path_loss(
        K,
        cfg,
        rng,
        d_override=dk,
    )

    pos = uniform_port_positions(N, cfg)

    # ONE fixed physical channel realization
    g, phi = sample_multipath(
        K,
        cfg,
        rng,
    )

    H = build_channel_matrix(
        pos,
        bk,
        K,
        cfg,
        g,
        phi,
    )

    return H, dk, bk, pos, g, phi