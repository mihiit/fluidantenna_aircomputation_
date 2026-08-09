"""
FAA-AirComp Reproducibility Package
=====================================
Source modules for:
  "Energy-Efficient Fluid Antenna Array for Over-the-Air Computation:
   Joint Port Selection and Power Control"
  Mihit Nanda, IEEE WCL, 2025.

Submodules
----------
config      : SystemConfig dataclass (Table II parameters)
channel     : Channel model (Eq. 1), port placement
bcd         : BCD algorithm (S1–S4) and MSE computation
experiments : Monte Carlo wrappers for all paper experiments
figures     : Publication-quality figure generation
"""

from .config      import SystemConfig, DEFAULT_CFG
from .channel     import make_channel, make_channel_straggler
from .bcd         import run_bcd, compute_mse
from .experiments import monte_carlo, MCResult
from .figures     import setup_rcparams

__all__ = [
    "SystemConfig", "DEFAULT_CFG",
    "make_channel", "make_channel_straggler",
    "run_bcd", "compute_mse",
    "monte_carlo", "MCResult",
    "setup_rcparams",
]
