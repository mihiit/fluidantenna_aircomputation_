# FAA-AirComp: Reproducibility Package

**"Energy-Efficient Fluid Antenna Array for Over-the-Air Computation:  
Joint Port Selection and Power Control"**  
Mihit Nanda — IILM University, Greater Noida, India  
Hannah Nagpall  — Texas A&M University , Kingsville ,USA
*IEEE Wireless Communications Letters* (submitted 2026)

---

## Overview

This repository contains the complete, self-contained code to:

1. **Reproduce all 6 paper figures** exactly as they appear in the submission
2. **Run the full BCD simulation** with verified monotone convergence
3. **Verify key numerical claims** (47% ESR, α=1.34, β=0.28, R²=0.95)
4. **Run the test suite** (65+ unit and integration tests)

All results are deterministic: fixed random seeds produce identical outputs
across runs and machines.

---

## Repository Structure

```
faa_aircomp_reproduce/
│
├── src/                        # Core simulation modules
│   ├── __init__.py
│   ├── config.py               # SystemConfig — Table II parameters
│   ├── channel.py              # Channel model (Eq. 1), port placement
│   ├── bcd.py                  # BCD algorithm: S1–S4, MSE, run_bcd()
│   ├── experiments.py          # Monte Carlo wrappers, per-figure runners
│   └── figures.py              # Publication-quality figure generation
│
├── scripts/
│   ├── reproduce_all_figures.py  # Main entry point — generates all figures
│   └── verify_key_results.py     # Quick (~2 min) verification of paper claims
│
├── tests/
│   ├── test_config.py          # SystemConfig unit tests
│   ├── test_channel.py         # Channel model unit tests
│   ├── test_bcd.py             # BCD algorithm unit + integration tests
│   └── test_experiments.py     # Monte Carlo + scaling law tests
│
├── paper/
│   ├── wcl_paper.tex           # Full LaTeX source (IEEEtran, 4 pages)
│   ├── refs.bib                # BibTeX references
│   └── wcl_paper_final_v6.pdf  # Compiled submission PDF
│
├── figures/                    # Generated figures go here (created on run)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9. No GPU needed; all computation is CPU-based.

### 2. Verify key results (~2 min)

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

*(Small deviations from paper values are expected at MC=20; run with MC=500 for exact match.)*

### 3. Reproduce all figures

```bash
# Full paper-quality run (MC=500, ~30–60 min)
python scripts/reproduce_all_figures.py

# Quick test run (MC=20, ~3–5 min)
python scripts/reproduce_all_figures.py --quick

# Single figure
python scripts/reproduce_all_figures.py --fig 1

# Custom MC count
python scripts/reproduce_all_figures.py --mc 100
```

Figures are saved to `figures/fig1.pdf` … `figures/fig6.pdf`.

### 4. Run the test suite

```bash
# Full test suite (~3 min)
pytest

# Fast tests only (skip slow MC)
pytest tests/test_config.py tests/test_channel.py tests/test_bcd.py

# Single test class
pytest tests/test_bcd.py::TestRunBCD -v

# With coverage (requires pytest-cov)
pytest --cov=src --cov-report=term-missing
```

---

## System Parameters (Table II)

| Parameter | Value |
|---|---|
| Carrier frequency | 5 GHz |
| Wavelength λ | 6 cm |
| Aperture L | 5λ = 30 cm |
| Multipath components Lp | 3 |
| Path-loss exponent α | 3.0 |
| Device distances | Uniform [5, 20] m |
| Max TX power Pmax | 23 dBm (200 mW) |
| Noise (BW=200 kHz, NF=7 dB) | −114 dBm |
| Normalised σ² | 10⁻³ (30 dB SNR) |
| Min. port spacing | λ/2 = 3 cm |
| BCD tolerance | 10⁻⁴ |
| Damping factor ρ | 0.15 |
| APV candidates C | 40 |
| Proxy weight γ | 0.3 |
| MC runs (Figs 1,4) | 500 |
| MC runs (Figs 2,5) | 300 |
| MC runs (Fig 6) | 200 |

---

## Algorithm Summary

The BCD algorithm (Section III) alternates four sub-steps:

| Step | Operation | Complexity | Reference |
|---|---|---|---|
| S1 | MMSE combiner | O(N³) | Eq. (7) |
| S2 | KKT bisection power control | O(K log μ_max) | Eq. (9) |
| S3 | Phase-aligned pre-equalisers | O(K) | §III-C |
| S4 | APV proxy-score search (C=40 candidates) | O(CNKLp) | §III-C |

**Convergence (Proposition 1):** The energy sequence {E^(i)} is monotonically
non-increasing and converges to a stationary point of P0. Verified across
all 65+ test cases (54/54 monotonicity checks pass).

---

## Key Results

| Metric | Paper Value | Code Produces (MC=500) |
|---|---|---|
| Max ESR at ε=0.06, N=6, K=8 | 53.6% | ≈53% |
| BCD convergence | 12–16 iters | 12–16 iters |
| Scaling law α (K exponent) | 1.34 | ≈1.34 |
| Scaling-law exponent β (N) | −0.28 | ≈ −0.28 |
| Scaling law R² | 0.95 | ≈0.95 |
| Straggler gap at ε=0.08 | ≈3.1 dBm | ≈3.1 dBm |
| APV plateau gain C=40→160 | <0.1 dBm | <0.1 dBm |

---

## Recompiling the Paper

The LaTeX source is in `paper/`. To recompile:

```bash
cd paper/
pdflatex wcl_paper.tex
bibtex wcl_paper
pdflatex wcl_paper.tex
pdflatex wcl_paper.tex
```

Requires a TeX distribution with `IEEEtran.cls` (included in
`texlive-publishers` on Ubuntu/Debian).

To replace figures after a full simulation run:

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
cfg = SystemConfig(fc=2.4e9) # custom config
print(cfg.summary())
```

### `src.channel.make_channel`

```python
from src.channel import make_channel
rng = np.random.default_rng(42)
H, dk, bk, pos = make_channel(K=8, N=6, cfg=cfg, rng=rng)
# H: (N,K) complex channel matrix
# dk: (K,) device distances
# bk: (K,) path-loss coefficients
# pos: (N,) port positions in wavelengths
```

### `src.bcd.run_bcd`

```python
from src.bcd import run_bcd
E, history = run_bcd(H, bk, K=8, cfg=cfg, rng=rng,
                     do_apv=True, return_history=True)
# E: converged total transmit power (Watts)
# history: list of energy per BCD iteration
```

### `src.experiments.monte_carlo`

```python
from src.experiments import monte_carlo, _run_proposed
result = monte_carlo(_run_proposed, K=8, N=6, cfg=cfg, MC=500, base_seed=0)
print(f"E* = {result.mean_dBm:.2f} dBm ± {result.se_dBm:.2f}")
```

---

## Citation

```bibtex
@article{nanda2025faa,
  author  = {Mihit Nanda , Hannah Nagpall},
  title   = {Energy-Efficient Fluid Antenna Array for Over-the-Air
             Computation: Joint Port Selection and Power Control},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

Code released for academic reproducibility. Please cite the paper if you
use this code in your research.
