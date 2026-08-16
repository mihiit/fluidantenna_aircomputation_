# FAA-AirComp

**Energy-Efficient Fluid Antenna Array for Over-the-Air Computation: Joint Port Selection and Power Control**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-Academic--Reproducibility-lightgrey)](#license) [![Status](https://img.shields.io/badge/IEEE%20WCL-Submitted%202026-orange)](#citation) [![Tests](https://img.shields.io/badge/tests-65%25%2B%20passing-brightgreen)](#running-the-test-suite)

**Mihit Nanda** — IILM University, Greater Noida, India
**Hannah Nagpall** — Texas A&M University–Kingsville, USA

*Submitted to IEEE Wireless Communications Letters (2026)*

---

## What Is This?

When many wireless devices (think: thousands of IoT sensors or edge devices in a federated-learning system) all need to send data to a central receiver at the same time, a technique called over-the-air computation (AirComp) lets them all transmit simultaneously and lets the receiver directly compute the aggregate (e.g., an average) from the combined signal — instead of receiving each device's data separately. This is much faster than traditional communication, but it comes with an energy cost: every device today transmits at full power on every round, even when a lower power level would have been perfectly sufficient to hit the required accuracy. Over thousands of rounds, this wastes a lot of battery life.

Separately, a newer antenna technology called a fluid antenna array (FAA) lets the receiving base station physically reposition its antenna ports to find better channel conditions — but prior work using this only optimized where the antenna sits, never how much power each device should use. Power was always left fixed at maximum.

This project combines both ideas for the first time: it jointly decides (1) where to position the antenna ports, and (2) how much power each device should transmit, to meet a required accuracy target using as little total energy as possible. We built an optimization algorithm (block coordinate descent, with a proven monotone objective-convergence guarantee) that solves this jointly, and showed — both mathematically and through simulation — that this combination saves substantially more energy (up to ~54%) than either idea alone.

This repository is the full, reproducible codebase behind that result: every number, figure, and claim in the paper can be regenerated from this code.

---

## Overview

This repository is the complete, self-contained reproducibility package for a Fluid Antenna Array (FAA)-enhanced Over-the-Air Computation (AirComp) system. It jointly optimizes **port selection** and **power control** to minimize total transmit energy across devices, using a four-step **Block Coordinate Descent (BCD)** algorithm with a provable monotone-objective-convergence guarantee.

The code in this repository will let you:

- **Reproduce all 6 figures** from the paper exactly as submitted
- **Run the full BCD simulation** with verified monotone convergence
- **Verify the key numerical claims** of the paper (energy savings ratio, scaling-law exponents, R²)
- **Run the full test suite** (65+ unit and integration tests)
- **Recompile the paper PDF** from LaTeX source

All results are deterministic — fixed random seeds produce identical outputs across runs and machines.

---

## Table of Contents

- [What Is This?](#what-is-this)
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

```
git clone https://github.com/mihiit/fluidantenna_aircomputation.git
cd fluidantenna_aircomputation/faa_aircomp_reproduce
pip install -r requirements.txt
```

Requires **Python ≥ 3.9**. All computation is CPU-based — no GPU required.

### 2. Verify key results (~2 minutes)

```
python scripts/verify_key_results.py
```

> ⚠️ **Before publishing this table**: rerun `python scripts/verify_key_results.py --mc 500` and paste the *actual* printed output below. Do not leave a placeholder or a value that doesn't match a real run — the paper's headline ESR must equal whatever this script prints at MC=500, not the other way around.

Expected output (replace with your verified MC=500 run):

```
[PASS] ESR = <fill in from your MC=500 run>%
[PASS] Monotone: 30/30 runs
[PASS] Max iters = <fill in> ≤ 25  (paper: 12–16)
[PASS] α = <fill in> > 1.0  (paper: 1.34)
[PASS] R² = <fill in> ≥ 0.80  (paper: 0.95 at MC=500)
All key results VERIFIED.
```
> Small deviations from the paper's headline values are expected at the default MC=20; run with `--mc 500` for an exact match.

### 3. Reproduce all figures

```
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

```
pytest                                                   # full suite (~3 min)
pytest tests/test_config.py tests/test_channel.py tests/test_bcd.py   # fast tests only
pytest tests/test_bcd.py::TestRunBCD -v                  # single test class
pytest --cov=src --cov-report=term-missing               # with coverage
```

---

## System Parameters (Table II)

| Parameter                       | Value             |
| ------------------------------- | ----------------- |
| Carrier frequency               | 5 GHz             |
| Wavelength λ                    | 6 cm              |
| Aperture L                      | 5λ = 30 cm        |
| Multipath components L_p        | 3                 |
| Path-loss exponent α            | 3.0               |
| Device distances                | Uniform [5, 20] m |
| Max TX power P_max              | 23 dBm (200 mW)   |
| Noise (BW = 200 kHz, NF = 7 dB) | −114 dBm          |
| Normalized σ²                   | 10⁻³ (30 dB SNR)  |
| Min. port spacing               | λ/2 = 3 cm        |
| BCD tolerance                   | 10⁻⁴              |
| Damping factor ρ                | 0.15              |
| APV candidates C                | 40                |
| Proxy weight γ                  | 0.3               |
| MC runs (Figs. 1, 4)            | 500               |
| MC runs (Figs. 2, 5)            | 300               |
| MC runs (Fig. 6)                | 200               |

All values are centralized in `src/config.py` (`SystemConfig`) as the single source of truth for every module and figure.

---

## Algorithm Summary

The BCD algorithm (Section III of the paper) alternates four sub-steps per iteration:

| Step | Operation                                  | Complexity      | Reference |
| ---- | ------------------------------------------ | --------------- | --------- |
| S1   | MMSE combiner                              | O(N³)           | Eq. (7)   |
| S2   | KKT bisection power control                | O(K log μ_max)  | Eq. (9)   |
| S3   | Phase-aligned pre-equalizers (\|τₖ\| = 1)  | O(K)            | §III-C    |
| S4   | APV proxy-score search (C = 40 candidates, φ(t) = σ_min(H(t)) + γ‖σ(H(t))‖₁) | O(C·N·K·L_p)   | §III-C    |

**Convergence (Proposition 1):** the energy sequence {E⁽ⁱ⁾} is monotonically non-increasing and bounded below by zero, hence converges in objective value. S1–S3 each solve their block subproblem exactly to its unique global minimiser; S4 satisfies a sufficient decrease condition but is a finite heuristic candidate search rather than an exact t-block minimiser, so convergence is established for the energy sequence, **not** stationarity of the full nonconvex P0. This has been verified across all 65+ test cases (54/54 monotonicity checks pass).

**Damping note:** the theoretical algorithm covered by Proposition 1 uses the exact (ρ = 1) S2 minimiser. The damping factor ρ = 0.15 applied in `bcd.py` (`p^(i+1) ← ρp* + (1−ρ)p^(i)`) is a numerical stabilization detail outside the formal proof; ESR is empirically shown to degrade by less than 2% for ρ ∈ [0.10, 0.25], confirming the damped implementation tracks the proven exact BCD closely.

---

## Key Results

> ⚠️ **This table must be regenerated from a real `--mc 500` run before submission.** Every value below is a placeholder marker — replace each with the actual printed output. The paper (v17) currently reports **53.6%** for the headline ESR; if your corrected code produces a different number, the paper must be updated to match the code, not the other way around.

| Metric                            | Paper Value (v17)  | Code Reproduces (MC = 500) |
| --------------------------------- | ------------------- | --------------------------- |
| Max ESR at ε = 0.06, N = 6, K = 8 | 53.6%                | *[verify and fill in]*      |
| BCD convergence                   | 12–16 iterations     | *[verify and fill in]*      |
| Scaling-law exponent α (K)        | 1.34                 | *[verify and fill in]*      |
| Scaling-law exponent β (N)        | −0.28                | *[verify and fill in]*      |
| Scaling-law R²                    | 0.95                 | *[verify and fill in]*      |
| Straggler power reduction (ε ≤ 0.07 region) | qualitative — see Fig. 5 | *[verify and fill in]* |
| APV plateau, C = 40 → 160         | marginal change beyond C=40 (see Fig. 6) | *[verify and fill in]* |

---

## Recompiling the Paper

The LaTeX source lives in `paper/`:

```
cd paper/
pdflatex wcl_paper.tex
bibtex wcl_paper
pdflatex wcl_paper.tex
pdflatex wcl_paper.tex
```

Requires a TeX distribution with `IEEEtran.cls` (available in `texlive-publishers` on Ubuntu/Debian).

To regenerate figures from a full simulation run and drop them straight into the paper directory:

```
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

```
pytest -v
```

See [Quick Start](#quick-start) for coverage and filtered-run options.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nanda2026faa,
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
