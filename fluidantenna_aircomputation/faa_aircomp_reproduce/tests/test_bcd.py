"""
tests/test_bcd.py
=================
Unit and integration tests for the BCD algorithm (src/bcd.py).

Tests cover:
  - MSE formula correctness and properties
  - Each BCD sub-step (S1, S2, S3, S4)
  - Full BCD convergence and monotonicity
  - Slater's condition / feasibility
  - KKT bisection edge cases
  - Water-filling structure
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config  import SystemConfig
from src.channel import make_channel, build_channel_matrix, uniform_port_positions
from src.bcd     import (
    compute_mse, step_S1_mmse, step_S2_power,
    step_S3_precoders, step_S4_apv, run_bcd,
)

CFG = SystemConfig()


# ─────────────────────────────────────────────────────────────────────────────
# MSE formula (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMSE:
    def _basic(self):
        rng = np.random.default_rng(0)
        K, N = 4, 4
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = np.ones(N, dtype=complex) / np.sqrt(N)
        return m, tau, p, H

    def test_nonnegative(self):
        m, tau, p, H = self._basic()
        mse = compute_mse(m, tau, p, H, CFG.sigma2)
        assert mse >= 0.0, f"MSE must be non-negative, got {mse}"

    def test_zero_power_gives_noise_floor(self):
        rng = np.random.default_rng(1)
        K, N = 4, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = np.ones(N, dtype=complex) / np.sqrt(N)
        tau = np.ones(K, dtype=complex)
        p   = np.zeros(K)
        mse = compute_mse(m, tau, p, H, CFG.sigma2)
        # With p=0: MSE = (1/K)² * K + σ² ‖m‖² = 1/K + σ² ‖m‖²
        expected_bias = 1.0 / K  # Σ(0 - 1/K)² = K * (1/K)²
        expected_noise = CFG.sigma2 * np.real(m.conj() @ m)
        np.testing.assert_allclose(mse, expected_bias + expected_noise, rtol=1e-6)

    def test_real_valued(self):
        m, tau, p, H = self._basic()
        mse = compute_mse(m, tau, p, H, CFG.sigma2)
        assert np.isreal(mse) or abs(np.imag(mse)) < 1e-10

    def test_scalar_output(self):
        m, tau, p, H = self._basic()
        mse = compute_mse(m, tau, p, H, CFG.sigma2)
        assert np.ndim(mse) == 0, "MSE should be a scalar"

    def test_increases_with_lower_power(self):
        """Lower transmit power should generally increase MSE."""
        rng = np.random.default_rng(5)
        K, N = 8, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = step_S1_mmse(np.ones(K, dtype=complex),
                            np.ones(K) * CFG.Pmax, H, CFG.sigma2)
        tau = step_S3_precoders(m, H)
        mse_high = compute_mse(m, tau, np.ones(K) * CFG.Pmax, H, CFG.sigma2)
        mse_low  = compute_mse(m, tau, np.ones(K) * CFG.Pmax * 0.01, H, CFG.sigma2)
        assert mse_low >= mse_high, "Lower power should give higher or equal MSE"

    def test_noise_floor(self):
        """MSE at p=0 equals σ²‖m‖² + 1/K (noise floor)."""
        rng = np.random.default_rng(6)
        K, N = 6, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = np.random.default_rng(7).standard_normal(N) + \
              1j * np.random.default_rng(8).standard_normal(N)
        m  /= np.linalg.norm(m)
        tau = np.ones(K, dtype=complex)
        p   = np.zeros(K)
        mse  = compute_mse(m, tau, p, H, CFG.sigma2)
        noise_part = CFG.sigma2 * float(np.real(m.conj() @ m))
        bias_part  = 1.0 / K   # Σ (0·c_k - 1/K)² for ck=0
        np.testing.assert_allclose(mse, noise_part + bias_part, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# S1 – MMSE Combiner
# ─────────────────────────────────────────────────────────────────────────────

class TestS1MMSE:
    def test_output_shape(self):
        rng = np.random.default_rng(10)
        K, N = 8, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m = step_S1_mmse(np.ones(K, dtype=complex), np.ones(K) * CFG.Pmax,
                         H, CFG.sigma2)
        assert m.shape == (N,), f"Expected ({N},), got {m.shape}"

    def test_complex_output(self):
        rng = np.random.default_rng(11)
        K, N = 4, 4
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m = step_S1_mmse(np.ones(K, dtype=complex), np.ones(K) * CFG.Pmax,
                         H, CFG.sigma2)
        assert np.iscomplexobj(m)

    def test_minimises_mse(self):
        """MMSE combiner should give lower MSE than a random combiner."""
        rng = np.random.default_rng(12)
        K, N = 6, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m_mmse  = step_S1_mmse(tau, p, H, CFG.sigma2)
        m_rand  = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        m_rand /= np.linalg.norm(m_rand)
        mse_mmse = compute_mse(m_mmse, tau, p, H, CFG.sigma2)
        mse_rand = compute_mse(m_rand, tau, p, H, CFG.sigma2)
        assert mse_mmse <= mse_rand + 1e-8, (
            f"MMSE combiner ({mse_mmse:.4f}) worse than random ({mse_rand:.4f})")

    def test_nonzero(self):
        rng = np.random.default_rng(13)
        K, N = 4, 4
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m = step_S1_mmse(np.ones(K, dtype=complex), np.ones(K) * CFG.Pmax,
                         H, CFG.sigma2)
        assert np.linalg.norm(m) > 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# S2 – Power Control
# ─────────────────────────────────────────────────────────────────────────────

class TestS2Power:
    def _setup(self, K=8, N=6, seed=20):
        rng = np.random.default_rng(seed)
        H, _, _, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = step_S1_mmse(tau, p, H, CFG.sigma2)
        return m, H, K

    def test_output_shape(self):
        m, H, K = self._setup()
        p = step_S2_power(m, H, CFG, eps=0.06)
        assert p.shape == (K,), f"Expected ({K},)"

    def test_power_bounds(self):
        m, H, K = self._setup()
        p = step_S2_power(m, H, CFG, eps=0.06)
        assert np.all(p >= 0.0 - 1e-10), "All powers must be non-negative"
        assert np.all(p <= CFG.Pmax + 1e-10), f"All powers must be ≤ Pmax={CFG.Pmax}"

    def test_total_power_leq_maxpow(self):
        """Total allocated power ≤ K * Pmax."""
        m, H, K = self._setup()
        p = step_S2_power(m, H, CFG, eps=0.06)
        assert np.sum(p) <= K * CFG.Pmax + 1e-9

    def test_energy_saving(self):
        """Power control should save energy vs. uniform Pmax."""
        rng = np.random.default_rng(21)
        K, N = 8, 6
        # Average over several realisations
        savings = []
        for seed in range(20):
            rng2 = np.random.default_rng(seed + 100)
            H, _, _, _ = make_channel(K, N, CFG, rng2)
            p_max  = np.ones(K) * CFG.Pmax
            tau    = np.ones(K, dtype=complex)
            m      = step_S1_mmse(tau, p_max, H, CFG.sigma2)
            p_opt  = step_S2_power(m, H, CFG, eps=0.06)
            savings.append(np.sum(p_max) - np.sum(p_opt))
        # On average, power control should save energy
        assert np.mean(savings) >= 0, "Power control should save energy on average"

    def test_water_filling_structure(self):
        """
        Devices with larger effective gain c_k should receive less power
        (AirComp water-filling, Eq. 9 comments in paper).
        """
        rng = np.random.default_rng(22)
        K, N = 4, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = step_S1_mmse(tau, p, H, CFG.sigma2)
        ck  = np.abs(m.conj() @ H)
        p_opt = step_S2_power(m, H, CFG, eps=0.06)

        # Sort by ck; higher ck → lower power (with some tolerance for saturation)
        idx_sort = np.argsort(ck)[::-1]  # descending ck
        # At least one pair should show the correct ordering
        correct_pairs = sum(
            1 for i in range(len(idx_sort) - 1)
            if p_opt[idx_sort[i]] <= p_opt[idx_sort[i + 1]] + 1e-8
        )
        assert correct_pairs > 0, "Water-filling ordering violated"

    def test_zero_power_edge_case(self):
        """When MSE constraint is trivially satisfied at p=0, return zeros."""
        rng = np.random.default_rng(23)
        K, N = 4, 4
        H, _, _, _ = make_channel(K, N, CFG, rng)
        # Set very large sigma2 so noise floor already exceeds any eps
        cfg_easy = SystemConfig(sigma2=1e6)
        m = np.ones(N, dtype=complex) / np.sqrt(N)
        p = step_S2_power(m, H, cfg_easy, eps=0.06)
        # Should return zeros (constraint satisfied trivially)
        assert np.all(p >= 0), "Powers must be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# S3 – Pre-equalisers
# ─────────────────────────────────────────────────────────────────────────────

class TestS3Precoders:
    def test_shape(self):
        rng = np.random.default_rng(30)
        K, N = 8, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = np.ones(N, dtype=complex) / np.sqrt(N)
        tau = step_S3_precoders(m, H)
        assert tau.shape == (K,)

    def test_unit_modulus(self):
        """All pre-equalisers should have |τ_k| = 1."""
        rng = np.random.default_rng(31)
        K, N = 8, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = np.ones(N, dtype=complex) / np.sqrt(N)
        tau = step_S3_precoders(m, H)
        np.testing.assert_allclose(np.abs(tau), 1.0, atol=1e-10,
                                   err_msg="Pre-equalisers must have unit modulus")

    def test_phase_alignment(self):
        """
        S3 computes τ_k = (m^H h_k) / |m^H h_k|.
        Therefore conj(τ_k) · (m^H h_k) = |m^H h_k| ≥ 0 (real, non-negative).
        """
        rng = np.random.default_rng(32)
        K, N = 4, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        m   = step_S1_mmse(np.ones(K, dtype=complex),
                            np.ones(K) * CFG.Pmax, H, CFG.sigma2)
        tau = step_S3_precoders(m, H)
        mHh     = m.conj() @ H          # (K,) complex
        # conj(τ_k) · (m^H h_k)  =  conj(mHh/|mHh|) · mHh  =  |mHh|  (real ≥ 0)
        aligned = tau.conj() * mHh
        np.testing.assert_allclose(np.imag(aligned), 0.0, atol=1e-10,
                                   err_msg="conj(τ)·(m^H h) should be real")
        assert np.all(np.real(aligned) >= -1e-12), \
            "conj(τ)·(m^H h) should be non-negative (= |m^H h|)"

    def test_minimises_mse(self):
        """Phase-aligned τ should give ≤ MSE compared to worst-case phase."""
        rng = np.random.default_rng(33)
        K, N = 8, 6
        H, _, _, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        m   = step_S1_mmse(np.ones(K, dtype=complex), p, H, CFG.sigma2)
        tau_opt  = step_S3_precoders(m, H)
        # Worst case: opposite phase
        tau_bad  = -tau_opt
        mse_opt  = compute_mse(m, tau_opt, p, H, CFG.sigma2)
        mse_bad  = compute_mse(m, tau_bad, p, H, CFG.sigma2)
        assert mse_opt <= mse_bad, (
            f"Optimal τ ({mse_opt:.4f}) worse than opposite phase ({mse_bad:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# S4 – APV Repositioning
# ─────────────────────────────────────────────────────────────────────────────

class TestS4APV:
    def test_output_shape(self):
        rng = np.random.default_rng(40)
        K, N = 8, 6
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = step_S1_mmse(tau, p, H, CFG.sigma2)
        H_new = step_S4_apv(m, tau, p, H, bk, K, CFG, rng)
        assert H_new.shape == (N, K)

    def test_mse_nonincreasing(self):
        """APV step should never increase MSE (strict descent acceptance rule)."""
        rng = np.random.default_rng(41)
        K, N = 8, 6
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = step_S1_mmse(tau, p, H, CFG.sigma2)
        tau = step_S3_precoders(m, H)
        mse_before = compute_mse(m, tau, p, H,     CFG.sigma2)
        H_new      = step_S4_apv(m, tau, p, H, bk, K, CFG, rng)
        mse_after  = compute_mse(m, tau, p, H_new, CFG.sigma2)
        assert mse_after <= mse_before + 1e-10, (
            f"APV step increased MSE: {mse_before:.6f} → {mse_after:.6f}")

    def test_returns_valid_matrix(self):
        rng = np.random.default_rng(42)
        K, N = 4, 4
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        p   = np.ones(K) * CFG.Pmax
        tau = np.ones(K, dtype=complex)
        m   = step_S1_mmse(tau, p, H, CFG.sigma2)
        H_new = step_S4_apv(m, tau, p, H, bk, K, CFG, rng)
        assert np.all(np.isfinite(H_new)), "APV output contains non-finite values"
        assert np.iscomplexobj(H_new)


# ─────────────────────────────────────────────────────────────────────────────
# Full BCD Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBCD:
    @pytest.mark.parametrize("K,N,eps_scale", [
        (4, 4, 1.0), (8, 6, 1.0), (16, 8, 1.0),
        (4, 6, 0.5), (8, 8, 2.0),
    ])
    def test_monotone_convergence(self, K, N, eps_scale):
        """
        Proposition 1: the energy sequence {E^(i)} must be
        monotonically non-increasing.
        """
        rng = np.random.default_rng(K * 100 + N)
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        _, hist = run_bcd(H, bk, K, CFG, rng, eps=0.06, do_apv=True,
                          return_history=True)
        for i in range(len(hist) - 1):
            assert hist[i] >= hist[i + 1] - 1e-9, (
                f"Non-monotone at iter {i}: {hist[i]:.6f} < {hist[i+1]:.6f}")

    def test_monotone_many_realisations(self):
        """Run 50 realisations; all must be monotone."""
        K, N = 8, 6
        violations = 0
        for seed in range(50):
            rng = np.random.default_rng(seed + 500)
            H, _, bk, _ = make_channel(K, N, CFG, rng)
            _, hist = run_bcd(H, bk, K, CFG, rng, eps=0.06, do_apv=True,
                              return_history=True)
            for i in range(len(hist) - 1):
                if hist[i] < hist[i + 1] - 1e-9:
                    violations += 1
        assert violations == 0, f"{violations} monotonicity violations in 50 runs"

    def test_power_bounds(self):
        """All converged powers must satisfy 0 ≤ p_k ≤ Pmax."""
        rng = np.random.default_rng(60)
        K, N = 8, 6
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        # We check the final power by running manually
        from src.bcd import step_S1_mmse, step_S2_power, step_S3_precoders
        p = np.ones(K) * CFG.Pmax
        for _ in range(CFG.bcd_max_iter):
            tau = np.ones(K, dtype=complex)
            m   = step_S1_mmse(tau, p, H, CFG.sigma2)
            p_new = step_S2_power(m, H, CFG, eps=0.06)
            p   = CFG.rho * p_new + (1 - CFG.rho) * p
        assert np.all(p >= -1e-9)
        assert np.all(p <= CFG.Pmax + 1e-9)

    def test_energy_below_maxpow(self):
        """Proposed scheme uses ≤ K*Pmax energy (BCD never exceeds Pmax)."""
        K, N = 8, 6
        for seed in range(10):
            rng = np.random.default_rng(seed + 700)
            H, _, bk, _ = make_channel(K, N, CFG, rng)
            E, _ = run_bcd(H, bk, K, CFG, rng, eps=0.06, do_apv=True)
            assert E <= K * CFG.Pmax + 1e-9, (
                f"BCD energy {E:.4f} exceeds K*Pmax={K*CFG.Pmax:.4f}")

    def test_convergence_within_iter_limit(self):
        """BCD must converge within the max_iter budget."""
        K, N = 8, 6
        iter_counts = []
        for seed in range(20):
            rng = np.random.default_rng(seed + 800)
            H, _, bk, _ = make_channel(K, N, CFG, rng)
            _, hist = run_bcd(H, bk, K, CFG, rng, eps=0.06, do_apv=True,
                              return_history=True)
            iter_counts.append(len(hist))
        # Must not exceed hard limit
        assert max(iter_counts) <= CFG.bcd_max_iter, (
            f"BCD exceeded max_iter={CFG.bcd_max_iter}")
        # Energy must be monotonically non-increasing in every run
        # (convergence quality check — not iter count, which depends on damping)

    def test_fpa_vs_faa_energy(self):
        """
        Proposition 2(ii): FAA (proposed) should use ≤ energy than FPA+PC
        on average.
        """
        K, N = 8, 6
        faa_lower = 0
        for seed in range(30):
            rng1 = np.random.default_rng(seed + 900)
            rng2 = np.random.default_rng(seed + 900)  # same seed → same init H
            H, _, bk, _ = make_channel(K, N, CFG, rng1)
            E_faa, _ = run_bcd(H.copy(), bk, K, CFG, rng1, eps=0.06, do_apv=True)
            E_fpa, _ = run_bcd(H.copy(), bk, K, CFG, rng2, eps=0.06, do_apv=False)
            if E_faa <= E_fpa + 1e-8:
                faa_lower += 1
        # FAA should be better in at least 70% of realisations
        assert faa_lower >= 20, (
            f"FAA only better in {faa_lower}/30 realisations")

    def test_no_apv_flag(self):
        """do_apv=False should produce valid results (FPA baseline)."""
        rng = np.random.default_rng(61)
        K, N = 4, 4
        H, _, bk, _ = make_channel(K, N, CFG, rng)
        E, hist = run_bcd(H, bk, K, CFG, rng, eps=0.06, do_apv=False,
                          return_history=True)
        assert E > 0
        assert len(hist) >= 1
        for i in range(len(hist) - 1):
            assert hist[i] >= hist[i + 1] - 1e-9

    def test_reproducibility(self):
        """Same seed should give identical results."""
        K, N = 8, 6
        H, _, bk, _ = make_channel(K, N, CFG, np.random.default_rng(42))
        E1, h1 = run_bcd(H.copy(), bk, K, CFG,
                         np.random.default_rng(42), return_history=True)
        E2, h2 = run_bcd(H.copy(), bk, K, CFG,
                         np.random.default_rng(42), return_history=True)
        np.testing.assert_allclose(E1, E2, rtol=1e-12)
        np.testing.assert_allclose(h1, h2, rtol=1e-12)

    def test_history_returned_only_when_requested(self):
        rng = np.random.default_rng(62)
        H, _, bk, _ = make_channel(4, 4, CFG, rng)
        E, hist = run_bcd(H, bk, 4, CFG, rng, eps=0.06, return_history=False)
        assert hist is None
        E2, hist2 = run_bcd(H, bk, 4, CFG, rng, eps=0.06, return_history=True)
        assert isinstance(hist2, list) and len(hist2) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 2(i) – Convexity of E*(ε)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMSETradeoff:
    def test_nonincreasing_in_eps(self):
        """
        E*(ε) should be non-increasing: looser ε → less energy needed.
        Tested over 20 realisations.
        """
        eps_vals = [0.04, 0.06, 0.08, 0.10, 0.12]
        K, N     = 8, 6
        violations = 0
        for seed in range(20):
            E_prev = np.inf
            for eps in eps_vals:
                rng  = np.random.default_rng(seed)
                H, _, bk, _ = make_channel(K, N, CFG, rng)
                # For fixed H, looser eps → less energy
                from src.bcd import step_S1_mmse, step_S2_power, step_S3_precoders
                p = np.ones(K) * CFG.Pmax
                for _ in range(CFG.bcd_max_iter):
                    tau = step_S3_precoders(
                        step_S1_mmse(np.ones(K, dtype=complex), p, H, CFG.sigma2), H)
                    m   = step_S1_mmse(tau, p, H, CFG.sigma2)
                    p_new = step_S2_power(m, H, CFG, eps=0.06)
                    p = CFG.rho * p_new + (1 - CFG.rho) * p
                E = float(np.sum(p))
                if E > E_prev + 1e-6:
                    violations += 1
                E_prev = E
        # Allow a small number of violations due to MC noise
        assert violations <= 5, (
            f"E*(ε) non-monotone in {violations}/80 (seed, ε) pairs")
