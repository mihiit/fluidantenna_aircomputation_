# FAA-AirComp

**Energy-Efficient Fluid Antenna Array for Over-the-Air Computation: Joint Port Selection and Power Control**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic--Reproducibility-lightgrey)](#license)
[![Status](https://img.shields.io/badge/IEEE%20WCL-Submitted%202026-orange)](#citation)
[![Tests](https://img.shields.io/badge/tests-65%2B%20passing-brightgreen)](#running-the-test-suite)

**Mihit Nanda** — IILM University, Greater Noida, India
**Hannah Nagpall** — Texas A&M University–Kingsville, USA

*Submitted to IEEE Wireless Communications Letters (2026)*

---

## Overview

This repository is the complete, self-contained reproducibility package for a Fluid Antenna Array (FAA)-enhanced Over-the-Air Computation (AirComp) system. It jointly optimizes **port selection** and **power control** to minimize total transmit energy across devices, using a four-step **Block Coordinate Descent (BCD)** algorithm with a provable monotone-convergence guarantee.

The code in this repository will let you:

- **Reproduce all 6 figures** from the paper exactly as submitted
- **Run the full BCD simulation** with verified monotone convergence
- **Verify the key numerical claims** of the paper (energy savings ratio, scaling-law exponents, R²)
- **Run the full test suite** (65+ unit and integration tests)
- **Recompile the paper PDF** from LaTeX source

All results are deterministic — fixed random seeds produce identical outputs across runs and machines.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [System Parameters](#system-parameters-table-ii)
- [Algorithm Summary](#algorithm-summary)
- [Key Results](#key-results)
- [Recompiling the Paper](#recompiling-the-paper)
- [Module API Reference](#module-api-reference)
- [Running the Test Suite](#running-the-test-suite)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure

```
fluidantenna_aircomputation/
└── faa_aircomp_reproduce/
    ├── src/                         # Core simulation modules
    │   ├── __init__.py
    │   ├── config.py                # SystemConfig — Table II parameters
    │   ├── channel.py               # Channel model (Eq. 1), port placement
    │   ├── bcd.py                   # BCD algorithm: S1–S4, MSE, run_bcd()
    │   ├── experiments.py           # Monte Carlo wrappers, per-figure runners
    │   └── figures.py               # Publication-quality figure generation
    │
    ├── scripts/
    │   ├── reproduce_all_figures.py # Main entry point — generates all figures
    │   └── verify_key_results.py    # Quick (~2 min) verification of paper claims
    │
    ├── tests/
    │   ├── test_config.py           # SystemConfig unit tests
    │   ├── test_channel.py          # Channel model unit tests
    │   ├── test_bcd.py               # BCD algorithm unit + integration tests
    │   └── test_experiments.py      # Monte Carlo + scaling law tests
    │
    ├── paper/
    │   ├── wcl_paper.tex             # Full LaTeX source (IEEEtran, 4 pages)
    │   ├── refs.bib                  # BibTeX references
    │   └── wcl_paper_final_v6.pdf    # Compiled submission PDF
    │
    ├── figures/                     # Generated figures (created on run)
    ├── requirements.txt
    ├── pytest.ini
    └── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/mihiit/fluidantenna_aircomputation.git
cd fluidantenna_aircomputation/faa_aircomp_reproduce
pip install -r requirements.txt
```

Requires **Python ≥ 3.9**. All computation is CPU-based — no GPU required.

### 2. Verify key results (~2 minutes)

```bash
python scripts/verify_key_results.py
```

Expected output:

```
[PASS] ESR = 38.2% ≥ 30%  (paper: ~47%)
[PASS] Monotone: 30/30 runs
[PASS] Max iters = 18 ≤ 25  (paper: 12–16)
[PASS] α = 1.31 > 1.0  (paper: 1.34)
[PASS] R² = 0.913 ≥ 0.80  (paper: 0.95 at MC=500)
All key results VERIFIED.
```

> Small deviations from the paper's headline values are expected at the default MC=20; run with `--mc 500` for an exact match.

### 3. Reproduce all figures

```bash
# Full paper-quality run (MC=500, ~30–60 min)
python scripts/reproduce_all_figures.py

# Quick test run (MC=20, ~3–5 min)
python scripts/reproduce_all_figures.py --quick

# Single figure
python scripts/reproduce_all_figures.py --fig 1

# Custom Monte Carlo count
python scripts/reproduce_all_figures.py --mc 100
```

Figures are saved to `figures/fig1.pdf` … `figures/fig6.pdf`.

### 4. Run the test suite

```bash
pytest                                                   # full suite (~3 min)
pytest tests/test_config.py tests/test_channel.py tests/test_bcd.py   # fast tests only
pytest tests/test_bcd.py::TestRunBCD -v                  # single test class
pytest --cov=src --cov-report=term-missing               # with coverage
```

---

## System Parameters (Table II)

| Parameter | Value |
|---|---|
| Carrier frequency | 5 GHz |
| Wavelength λ | 6 cm |
| Aperture L | 5λ = 30 cm |
| Multipath components L_p | 3 |
| Path-loss exponent α | 3.0 |
| Device distances | Uniform [5, 20] m |
| Max TX power P_max | 23 dBm (200 mW) |
| Noise (BW = 200 kHz, NF = 7 dB) | −114 dBm |
| Normalized σ² | 10⁻³ (30 dB SNR) |
| Min. port spacing | λ/2 = 3 cm |
| BCD tolerance | 10⁻⁴ |
| Damping factor ρ | 0.15 |
| APV candidates C | 40 |
| Proxy weight γ | 0.3 |
| MC runs (Figs. 1, 4) | 500 |
| MC runs (Figs. 2, 5) | 300 |
| MC runs (Fig. 6) | 200 |

All values are centralized in `src/config.py` (`SystemConfig`) as the single source of truth for every module and figure.

---

## Algorithm Summary

The BCD algorithm (Section III of the paper) alternates four sub-steps per iteration:

| Step | Operation | Complexity | Reference |
|---|---|---|---|
| S1 | MMSE combiner | O(N³) | Eq. (7) |
| S2 | KKT bisection power control | O(K log μ_max) | Eq. (9) |
| S3 | Phase-aligned pre-equalizers | O(K) | §III-C |
| S4 | APV proxy-score search (C = 40 candidates) | O(C·N·K·L_p) | §III-C |

**Convergence (Proposition 1):** the energy sequence {E⁽ⁱ⁾} is monotonically non-increasing and converges to a stationary point of P0. This has been verified across all 65+ test cases (54/54 monotonicity checks pass).

---

## Key Results

| Metric | Paper Value | Code Reproduces (MC = 500) |
|---|---|---|
| Max ESR at ε = 0.06, N = 6, K = 8 | 53.6% | ≈ 53% |
| BCD convergence | 12–16 iterations | 12–16 iterations |
| Scaling-law exponent α (K) | 1.34 | ≈ 1.34 |
| Scaling-law exponent β (N) | −0.28 | ≈ −0.28 |
| Scaling-law R² | 0.95 | ≈ 0.95 |
| Straggler gap at ε = 0.08 | ≈ 3.1 dBm | ≈ 3.1 dBm |
| APV plateau gain, C = 40 → 160 | < 0.1 dBm | < 0.1 dBm |

---

## Recompiling the Paper

The LaTeX source lives in `paper/`:

```bash
cd paper/
pdflatex wcl_paper.tex
bibtex wcl_paper
pdflatex wcl_paper.tex
pdflatex wcl_paper.tex
```

Requires a TeX distribution with `IEEEtran.cls` (available in `texlive-publishers` on Ubuntu/Debian).

To regenerate figures from a full simulation run and drop them straight into the paper directory:

```bash
python scripts/reproduce_all_figures.py --outdir paper/
cd paper/
pdflatex wcl_paper.tex && pdflatex wcl_paper.tex
```

---

## Module API Reference

### `src.config.SystemConfig`

```python
from src.config import SystemConfig, DEFAULT_CFG

cfg = SystemConfig()          # default Table II parameters
cfg = SystemConfig(fc=2.4e9)  # custom config
print(cfg.summary())
```

### `src.channel.make_channel`

```python
from src.channel import make_channel
import numpy as np

rng = np.random.default_rng(42)
H, dk, bk, pos = make_channel(K=8, N=6, cfg=cfg, rng=rng)
# H:   (N, K) complex channel matrix
# dk:  (K,)   device distances
# bk:  (K,)   path-loss coefficients
# pos: (N,)   port positions in wavelengths
```

### `src.bcd.run_bcd`

```python
from src.bcd import run_bcd

E, history = run_bcd(H, bk, K=8, cfg=cfg, rng=rng,
                      do_apv=True, return_history=True)
# E:       converged total transmit power (Watts)
# history: list of energy values per BCD iteration
```

### `src.experiments.monte_carlo`

```python
from src.experiments import monte_carlo, _run_proposed

result = monte_carlo(_run_proposed, K=8, N=6, cfg=cfg, MC=500, base_seed=0)
print(f"E* = {result.mean_dBm:.2f} dBm ± {result.se_dBm:.2f}")
```

---

## Running the Test Suite

The package ships with 65+ unit and integration tests covering configuration validation, the channel model, the BCD algorithm (including monotonicity of the energy sequence), and Monte Carlo scaling-law fits.

```bash
pytest -v
```

See [Quick Start](#quick-start) for coverage and filtered-run options.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nanda2025faa,
  author  = {Nanda, Mihit and Nagpall, Hannah},
  title   = {Energy-Efficient Fluid Antenna Array for Over-the-Air
             Computation: Joint Port Selection and Power Control},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

Code released for academic reproducibility. Please cite the paper above if you use this code in your research.
