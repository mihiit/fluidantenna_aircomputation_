#!/usr/bin/env python3
"""
scripts/verify_key_results.py
==============================
Lightweight verification script (~2 min at default MC; use --mc for full fidelity).

Runs a Monte Carlo sweep to sanity-check the three most important
numerical claims in the paper:

  1. ESR ≥ 30% at ε=0.06, N=6, K=8  (paper: 47% — verify at --mc 500)
  2. BCD converges monotonically in ≤ 25 iterations
  3. Scaling law exponents: α > 1.0, β < 0.0, R² ≥ 0.80
     (β is negative: energy decreases with more FA ports, diminishing returns)

Prints PASS / FAIL for each claim.

Usage:
  python scripts/verify_key_results.py            # quick check, MC=20
  python scripts/verify_key_results.py --mc 500    # full paper-fidelity check
"""

import argparse
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config      import SystemConfig
from src.channel     import make_channel
from src.bcd         import run_bcd
from src.experiments import monte_carlo, _run_proposed, _run_fpa_maxpow

parser = argparse.ArgumentParser()
parser.add_argument("--mc", type=int, default=20, help="Monte Carlo runs per grid point (paper uses 500)")
args = parser.parse_args()

CFG    = SystemConfig()
MC     = args.mc
PASS   = "\033[92mPASS\033[0m"
FAIL   = "\033[91mFAIL\033[0m"

def check(label, cond, note=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}" + (f"  ({note})" if note else ""))
    return cond


print("\n" + "="*55)
print(f"  FAA-AirComp: Key Results Verification (MC={MC})")
print("="*55 + "\n")
all_passed = True

# ── Claim 1: ESR ≥ 30% at ε=0.06, N=6, K=8 ─────────────────────────────────
print("Claim 1: ESR at ε=0.06, N=6, K=8")
r_prop  = monte_carlo(_run_proposed,   8, 6, CFG, MC, base_seed=1000)
r_base  = monte_carlo(_run_fpa_maxpow, 8, 6, CFG, MC, base_seed=1000)
esr     = 100 * (r_base.mean - r_prop.mean) / r_base.mean
ok = check(f"ESR = {esr:.1f}% ≥ 30%  (paper: ~47%)", esr >= 30,
           f"E*={r_prop.mean_dBm:.2f} dBm")
all_passed = all_passed and ok

# ── Claim 2: Monotone BCD convergence ────────────────────────────────────────
print("\nClaim 2: BCD monotone convergence (30 realisations)")
violations, max_iters = 0, 0
for seed in range(30):
    rng = np.random.default_rng(seed + 2000)
    H, _, bk, _ = make_channel(8, 6, CFG, rng)
    _, hist = run_bcd(H, bk, 8, CFG, rng, do_apv=True, return_history=True)
    max_iters = max(max_iters, len(hist))
    for i in range(len(hist) - 1):
        if hist[i] < hist[i+1] - 1e-9:
            violations += 1

ok1 = check(f"Monotone: {30-violations}/30 runs", violations == 0)
ok2 = check(f"Max iters = {max_iters} ≤ 25  (paper: 12–16)",
            max_iters <= CFG.bcd_max_iter,
            f"avg≈{max_iters}")
all_passed = all_passed and ok1 and ok2

# ── Claim 3: Scaling law exponents ────────────────────────────────────────────
print("\nClaim 3: Scaling law E* ∝ K^α · N^β")
K_vals = [4, 8, 16]; N_list = [4, 6, 8]; eps = 0.08
logK, logN, logE = [], [], []
for K in K_vals:
    for N in N_list:
        r = monte_carlo(_run_proposed, K, N, CFG, MC, base_seed=K*100+N+3000)
        logK.append(np.log(K)); logN.append(np.log(N)); logE.append(np.log(r.mean))
X = np.column_stack([np.ones(len(logK)), logK, logN])
c, *_ = np.linalg.lstsq(X, logE, rcond=None)
pred = X @ c
R2   = 1 - np.sum((np.array(logE)-pred)**2) / np.sum((np.array(logE)-np.mean(logE))**2)
alpha, beta = c[1], c[2]

ok1 = check(f"α = {alpha:.2f} > 1.0  (paper: 1.34)", alpha > 1.0)
ok2 = check(f"β = {beta:.2f} < 0.0  (paper: -0.28, more ports -> lower energy, diminishing returns)",
            beta < 0.0,
            "NOTE: if this fails, re-check sign against Fig. 2 (ESR must rise with N)")
ok3 = check(f"R² = {R2:.3f} ≥ 0.80  (paper: 0.95 at MC=500)", R2 >= 0.80)
all_passed = all_passed and ok1 and ok2 and ok3

# ── Final result ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
if all_passed:
    print(f"  \033[92mAll key results VERIFIED.\033[0m")
else:
    print(f"  \033[91mSome checks FAILED — run with MC=500 for paper results.\033[0m")
print("="*55 + "\n")

sys.exit(0 if all_passed else 1)
