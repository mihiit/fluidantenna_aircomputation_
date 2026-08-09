"""
channel.py
==========
Channel model for FAA-AirComp system (Section II-A, Eq. 1).

Multi-path channel from K single-antenna IoT devices to an N-port
fluid antenna array (FAA) at the access point.

Reference:
    Mihit Nanda, "Energy-Efficient Fluid Antenna Array for
    Over-the-Air Computation: Joint Port Selection and Power Control,"
    IEEE Wireless Communications Letters, submitted 2025.
"""

import numpy as np
from .config import SystemConfig


def build_channel_matrix(pos_wl: np.ndarray,
                         bk: np.ndarray,
                         K: int,
                         cfg: SystemConfig,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Build the N×K complex channel matrix H(t) for given port positions.

    Implements Eq. (1):
        h_k(x) = sqrt(β_k) Σ_{l=1}^{Lp} g_kl exp(j2π x sin φ_kl)

    Parameters
    ----------
    pos_wl : (N,) array of port positions in wavelengths ∈ [0, L/λ]
    bk     : (K,) large-scale path-loss coefficients (normalised)
    K      : number of devices
    cfg    : SystemConfig instance
    rng    : numpy random Generator

    Returns
    -------
    H : (N, K) complex channel matrix
    """
    N = len(pos_wl)
    H = np.zeros((N, K), dtype=complex)
    for k in range(K):
        for _ in range(cfg.Lp):
            g   = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            phi = rng.uniform(-np.pi / 2, np.pi / 2)
            H[:, k] += np.sqrt(bk[k]) * g * np.exp(
                1j * 2 * np.pi * pos_wl * np.sin(phi)
            )
    return H


def sample_path_loss(K: int,
                     cfg: SystemConfig,
                     rng: np.random.Generator,
                     d_override: np.ndarray = None) -> tuple:
    """
    Sample device distances and compute normalised path-loss coefficients.

    Parameters
    ----------
    K           : number of devices
    cfg         : SystemConfig
    rng         : numpy random Generator
    d_override  : if not None, use these distances instead of sampling

    Returns
    -------
    dk : (K,) distances in metres
    bk : (K,) normalised path-loss  β_k = d_k^{-α} / mean(d^{-α})
    """
    if d_override is not None:
        dk = np.asarray(d_override, dtype=float)
    else:
        dk = rng.uniform(cfg.d_min, cfg.d_max, K)
    bk = dk ** (-cfg.path_loss_exp)
    bk /= bk.mean()
    return dk, bk


def uniform_port_positions(N: int, cfg: SystemConfig) -> np.ndarray:
    """
    Uniformly spaced initial port positions in wavelengths ∈ [0, L/λ].
    Used as starting point and for fixed-array (FPA) baseline.
    """
    return np.linspace(0.0, cfg.L_wl, N)


def random_port_positions(N: int,
                          cfg: SystemConfig,
                          rng: np.random.Generator,
                          max_tries: int = 2000) -> np.ndarray:
    """
    Sample N port positions uniformly in [0, L/λ] with minimum
    inter-port spacing ≥ λ/2 (= 0.5 in wavelength units).

    Uses rejection sampling; falls back to uniform spacing after
    max_tries failed attempts (practically never triggered for N ≤ 12).
    """
    half_sep = cfg.min_sep_wl
    L_wl     = cfg.L_wl
    for _ in range(max_tries):
        pts = np.sort(rng.uniform(0.0, L_wl, N))
        if N == 1 or np.min(np.diff(pts)) >= half_sep:
            return pts
    # Fallback
    return uniform_port_positions(N, cfg)


def make_channel(K: int,
                 N: int,
                 cfg: SystemConfig,
                 rng: np.random.Generator,
                 d_override: np.ndarray = None) -> tuple:
    """
    Full channel realisation: sample distances, build H with uniform APV.

    Returns
    -------
    H   : (N, K) channel matrix
    dk  : (K,) device distances
    bk  : (K,) path-loss coefficients
    pos : (N,) initial port positions in wavelengths
    """
    dk, bk = sample_path_loss(K, cfg, rng, d_override)
    pos    = uniform_port_positions(N, cfg)
    H      = build_channel_matrix(pos, bk, K, cfg, rng)
    return H, dk, bk, pos


def make_channel_straggler(K: int,
                           N: int,
                           cfg: SystemConfig,
                           rng: np.random.Generator) -> tuple:
    """
    Heterogeneous channel for straggler experiment (Section V-A):
      - Device 0: fixed at d = 19 m
      - Devices 1…K-1: drawn from Uniform[5, 8] m
    """
    dk       = np.empty(K)
    dk[0]    = cfg.straggler_dist
    dk[1:]   = rng.uniform(cfg.straggler_near_min,
                            cfg.straggler_near_max, K - 1)
    _, bk    = sample_path_loss(K, cfg, rng, d_override=dk)
    pos      = uniform_port_positions(N, cfg)
    H        = build_channel_matrix(pos, bk, K, cfg, rng)
    return H, dk, bk, pos
