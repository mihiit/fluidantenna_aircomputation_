"""
config.py
=========
Centralised system parameters matching Table II of the paper.

All simulation modules import from this single source of truth,
ensuring reproducibility across figures.
"""

from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """
    System parameters for FAA-AirComp simulation.

    All values match Table II of the paper exactly.
    """

    # ── RF / Carrier ─────────────────────────────────────────────────────────
    fc: float = 5e9           # carrier frequency (Hz)
    c_light: float = 3e8     # speed of light (m/s)

    # ── FAA Aperture ─────────────────────────────────────────────────────────
    L_lambda: float = 5.0    # aperture in wavelengths (= 5λ = 30 cm)

    # ── Channel Model ────────────────────────────────────────────────────────
    Lp: int   = 3            # number of multipath components
    path_loss_exp: float = 3.0   # path-loss exponent α
    d_min: float = 5.0       # min device distance (m)
    d_max: float = 20.0      # max device distance (m)

    # ── Power / Noise ────────────────────────────────────────────────────────
    Pmax: float  = 200e-3    # max TX power = 23 dBm = 200 mW
    sigma2: float = 1e-3     # normalised noise variance (30 dB SNR)

    # ── BCD Algorithm ────────────────────────────────────────────────────────
    bcd_tol: float  = 1e-4   # convergence tolerance |ΔE|/E
    bcd_max_iter: int = 25   # maximum BCD iterations
    rho: float = 0.15        # power-update damping factor
    C_apv: int = 40          # APV candidate port vectors per iteration
    gamma: float = 0.3       # proxy weight: φ(t) = σ_min + γ‖σ‖₁

    # ── Straggler Experiment (Section V-A) ──────────────────────────────────
    straggler_dist: float     = 19.0   # straggler device distance (m)
    straggler_near_min: float = 5.0    # near-device range min (m)
    straggler_near_max: float = 8.0    # near-device range max (m)

    # ── Derived quantities (computed automatically) ──────────────────────────
    @property
    def lam(self) -> float:
        """Wavelength λ = c / fc (m)."""
        return self.c_light / self.fc

    @property
    def L_m(self) -> float:
        """Aperture in metres = L_lambda × λ."""
        return self.L_lambda * self.lam

    @property
    def L_wl(self) -> float:
        """Aperture in wavelengths (= L_lambda, kept for clarity)."""
        return self.L_lambda

    @property
    def min_sep_wl(self) -> float:
        """Minimum inter-port spacing in wavelengths = λ/2 = 0.5."""
        return 0.5

    def summary(self) -> str:
        return (
            f"SystemConfig: fc={self.fc/1e9:.0f} GHz, "
            f"λ={self.lam*100:.1f} cm, "
            f"L={self.L_m*100:.1f} cm ({self.L_lambda:.0f}λ), "
            f"Lp={self.Lp}, α={self.path_loss_exp}, "
            f"Pmax={10*__import__('math').log10(self.Pmax*1000):.0f} dBm, "
            f"σ²={self.sigma2:.0e}"
        )


# Default configuration used throughout the paper
DEFAULT_CFG = SystemConfig()
