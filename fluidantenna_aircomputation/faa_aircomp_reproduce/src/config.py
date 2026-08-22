"""
config.py
=========

Centralized system parameters for FAA-AirComp simulation.

Parameters are based on the system model / simulation settings
reported in the WCL paper.
"""

from dataclasses import dataclass
import math


@dataclass
class SystemConfig:

    # ==============================================================
    # RF / Carrier
    # ==============================================================

    fc: float = 5e9
    c_light: float = 3e8

    # ==============================================================
    # FAA aperture
    # ==============================================================

    # Paper:
    # L = 5 lambda = 30 cm at fc = 5 GHz
    L_lambda: float = 5.0

    # ==============================================================
    # Channel
    # ==============================================================

    # Paper:
    # Lp = 3
    Lp: int = 3

    # beta_k = d_k^{-3}
    path_loss_exp: float = 3.0

    # d_k in [5, 20] m
    d_min: float = 5.0
    d_max: float = 20.0

    # ==============================================================
    # Power / noise
    # ==============================================================

    # Pmax = 200 mW = 23 dBm
    Pmax: float = 200e-3

    # Normalized noise variance
    sigma2: float = 1e-3

    # ==============================================================
    # BCD
    # ==============================================================

    bcd_tol: float = 1e-4
    bcd_max_iter: int = 25

    # Power-update damping
    rho: float = 0.15

    # Number of APV candidates
    C_apv: int = 40

    # Proxy parameter
    gamma: float = 0.3

    # ==============================================================
    # Straggler experiment
    # ==============================================================

    straggler_dist: float = 19.0
    straggler_near_min: float = 5.0
    straggler_near_max: float = 8.0

    # ==============================================================
    # Derived quantities
    # ==============================================================

    @property
    def lam(self) -> float:
        """Carrier wavelength in meters."""
        return self.c_light / self.fc

    @property
    def L_m(self) -> float:
        """FAA aperture in meters."""
        return self.L_lambda * self.lam

    @property
    def L_wl(self) -> float:
        """FAA aperture in wavelengths."""
        return self.L_lambda

    @property
    def min_sep_wl(self) -> float:
        """Minimum port separation in wavelengths."""
        return 0.5

    def validate(self) -> None:
        """Validate physically/model-consistent parameters."""

        if self.fc <= 0:
            raise ValueError("fc must be positive.")

        if self.c_light <= 0:
            raise ValueError("c_light must be positive.")

        if self.L_lambda <= 0:
            raise ValueError("L_lambda must be positive.")

        if self.Lp < 1:
            raise ValueError("Lp must be >= 1.")

        if self.path_loss_exp <= 0:
            raise ValueError("path_loss_exp must be positive.")

        if self.d_min <= 0:
            raise ValueError("d_min must be positive.")

        if self.d_max <= self.d_min:
            raise ValueError("d_max must be greater than d_min.")

        if self.Pmax <= 0:
            raise ValueError("Pmax must be positive.")

        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")

        if self.bcd_tol <= 0:
            raise ValueError("bcd_tol must be positive.")

        if self.bcd_max_iter < 1:
            raise ValueError("bcd_max_iter must be >= 1.")

        if not 0 <= self.rho <= 1:
            raise ValueError("rho must lie in [0,1].")

        if self.C_apv < 1:
            raise ValueError("C_apv must be >= 1.")

    def summary(self) -> str:

        return (
            f"SystemConfig: "
            f"fc={self.fc / 1e9:.1f} GHz, "
            f"lambda={self.lam * 100:.1f} cm, "
            f"L={self.L_m * 100:.1f} cm "
            f"({self.L_lambda:.1f} lambda), "
            f"Lp={self.Lp}, "
            f"alpha={self.path_loss_exp:.1f}, "
            f"Pmax={10 * math.log10(self.Pmax * 1000):.2f} dBm, "
            f"sigma2={self.sigma2:.1e}"
        )


DEFAULT_CFG = SystemConfig()
DEFAULT_CFG.validate()