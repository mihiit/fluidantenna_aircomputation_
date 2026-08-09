"""
tests/test_channel.py
=====================
Unit tests for the channel model (src/channel.py).

Tests cover:
  - Channel matrix dimensions and dtype
  - Path-loss normalisation
  - Port placement constraints (spacing, bounds)
  - Straggler setup
  - Reproducibility (same seed → same output)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config  import SystemConfig
from src.channel import (
    make_channel, make_channel_straggler,
    build_channel_matrix, sample_path_loss,
    uniform_port_positions, random_port_positions,
)

CFG = SystemConfig()


class TestBuildChannel:
    def test_shape(self):
        rng = np.random.default_rng(0)
        N, K = 6, 8
        dk, bk = sample_path_loss(K, CFG, rng)
        pos = uniform_port_positions(N, CFG)
        H   = build_channel_matrix(pos, bk, K, CFG, rng)
        assert H.shape == (N, K), f"Expected ({N},{K}), got {H.shape}"

    def test_dtype_complex(self):
        rng = np.random.default_rng(1)
        dk, bk = sample_path_loss(4, CFG, rng)
        pos = uniform_port_positions(4, CFG)
        H   = build_channel_matrix(pos, bk, 4, CFG, rng)
        assert np.iscomplexobj(H), "Channel matrix should be complex"

    def test_nonzero(self):
        rng = np.random.default_rng(2)
        dk, bk = sample_path_loss(8, CFG, rng)
        pos = uniform_port_positions(6, CFG)
        H   = build_channel_matrix(pos, bk, 8, CFG, rng)
        assert np.all(np.abs(H) > 0), "Channel entries should be non-zero"


class TestPathLoss:
    def test_normalisation(self):
        rng = np.random.default_rng(10)
        _, bk = sample_path_loss(100, CFG, rng)
        assert abs(bk.mean() - 1.0) < 1e-10, "mean(β) should equal 1"

    def test_distance_range(self):
        rng = np.random.default_rng(11)
        dk, _ = sample_path_loss(1000, CFG, rng)
        assert dk.min() >= CFG.d_min - 1e-9
        assert dk.max() <= CFG.d_max + 1e-9

    def test_override_distances(self):
        rng = np.random.default_rng(12)
        d_in = np.array([5.0, 10.0, 20.0])
        dk, _ = sample_path_loss(3, CFG, rng, d_override=d_in)
        np.testing.assert_array_almost_equal(dk, d_in)

    def test_positive_pathloss(self):
        rng = np.random.default_rng(13)
        _, bk = sample_path_loss(50, CFG, rng)
        assert np.all(bk > 0)


class TestPortPositions:
    def test_uniform_length(self):
        for N in [1, 4, 6, 8, 12]:
            pos = uniform_port_positions(N, CFG)
            assert len(pos) == N

    def test_uniform_bounds(self):
        pos = uniform_port_positions(8, CFG)
        assert pos[0] >= 0.0
        assert pos[-1] <= CFG.L_wl + 1e-10

    def test_uniform_spacing(self):
        N   = 6
        pos = uniform_port_positions(N, CFG)
        gaps = np.diff(pos)
        expected_gap = CFG.L_wl / (N - 1)
        np.testing.assert_allclose(gaps, expected_gap, rtol=1e-6)

    def test_random_bounds(self):
        rng = np.random.default_rng(20)
        for _ in range(50):
            pos = random_port_positions(6, CFG, rng)
            assert np.all(pos >= 0.0 - 1e-10)
            assert np.all(pos <= CFG.L_wl + 1e-10)

    def test_random_min_spacing(self):
        rng = np.random.default_rng(21)
        violations = 0
        for _ in range(200):
            pos = random_port_positions(6, CFG, rng)
            if len(pos) > 1 and np.min(np.diff(pos)) < CFG.min_sep_wl - 1e-9:
                violations += 1
        assert violations == 0, f"{violations}/200 spacing violations"

    def test_random_sorted(self):
        rng = np.random.default_rng(22)
        for _ in range(100):
            pos = random_port_positions(8, CFG, rng)
            assert np.all(np.diff(pos) >= 0), "Positions should be sorted"

    def test_single_port(self):
        rng = np.random.default_rng(23)
        pos = random_port_positions(1, CFG, rng)
        assert len(pos) == 1
        assert 0.0 <= pos[0] <= CFG.L_wl


class TestMakeChannel:
    def test_output_shapes(self):
        rng = np.random.default_rng(30)
        K, N = 8, 6
        H, dk, bk, pos = make_channel(K, N, CFG, rng)
        assert H.shape  == (N, K)
        assert dk.shape == (K,)
        assert bk.shape == (K,)
        assert pos.shape == (N,)

    def test_reproducibility(self):
        K, N = 4, 4
        H1, dk1, bk1, _ = make_channel(K, N, CFG, np.random.default_rng(99))
        H2, dk2, bk2, _ = make_channel(K, N, CFG, np.random.default_rng(99))
        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(dk1, dk2)

    def test_different_seeds(self):
        K, N = 4, 4
        H1, *_ = make_channel(K, N, CFG, np.random.default_rng(0))
        H2, *_ = make_channel(K, N, CFG, np.random.default_rng(1))
        assert not np.allclose(H1, H2), "Different seeds should give different channels"


class TestStragglerChannel:
    def test_straggler_distance(self):
        rng = np.random.default_rng(40)
        _, dk, _, _ = make_channel_straggler(8, 6, CFG, rng)
        assert dk[0] == CFG.straggler_dist, (
            f"Device 0 should be at {CFG.straggler_dist} m, got {dk[0]}")

    def test_near_devices_range(self):
        rng = np.random.default_rng(41)
        _, dk, _, _ = make_channel_straggler(8, 6, CFG, rng)
        near = dk[1:]
        assert np.all(near >= CFG.straggler_near_min - 1e-9)
        assert np.all(near <= CFG.straggler_near_max + 1e-9)

    def test_shape(self):
        rng = np.random.default_rng(42)
        K, N = 8, 6
        H, dk, bk, pos = make_channel_straggler(K, N, CFG, rng)
        assert H.shape == (N, K)
