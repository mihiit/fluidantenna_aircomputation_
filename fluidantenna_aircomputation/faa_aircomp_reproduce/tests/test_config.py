"""
tests/test_config.py
====================
Tests for SystemConfig (src/config.py).
Ensures all derived properties match Table II of the paper.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SystemConfig, DEFAULT_CFG


class TestSystemConfig:
    def test_wavelength(self):
        cfg = SystemConfig()
        expected = 3e8 / 5e9
        np.testing.assert_allclose(cfg.lam, expected, rtol=1e-10,
                                   err_msg="λ = c/fc = 0.06 m")

    def test_aperture_metres(self):
        cfg = SystemConfig()
        np.testing.assert_allclose(cfg.L_m, 5 * cfg.lam, rtol=1e-10,
                                   err_msg="L = 5λ = 0.30 m")

    def test_aperture_wavelengths(self):
        cfg = SystemConfig()
        assert cfg.L_wl == 5.0, "Aperture should be 5λ"

    def test_min_spacing(self):
        cfg = SystemConfig()
        assert cfg.min_sep_wl == 0.5, "Min spacing = λ/2 = 0.5λ"

    def test_pmax_dBm(self):
        cfg = SystemConfig()
        pmax_dBm = 10 * np.log10(cfg.Pmax * 1000)
        np.testing.assert_allclose(pmax_dBm, 23.0, atol=0.02,
                                   err_msg="Pmax ≈ 23 dBm (200 mW = 23.01 dBm)")

    def test_sigma2(self):
        cfg = SystemConfig()
        snr_dB = -10 * np.log10(cfg.sigma2)
        np.testing.assert_allclose(snr_dB, 30.0, atol=0.01,
                                   err_msg="SNR = 30 dB → σ² = 1e-3")

    def test_default_cfg_singleton(self):
        from src.config import DEFAULT_CFG
        assert isinstance(DEFAULT_CFG, SystemConfig)

    def test_summary_string(self):
        cfg = SystemConfig()
        s = cfg.summary()
        assert "5 GHz" in s
        assert "30.0 cm" in s

    def test_custom_config(self):
        cfg = SystemConfig(fc=2.4e9, Pmax=100e-3, Lp=5)
        assert cfg.fc == 2.4e9
        assert cfg.Pmax == 100e-3
        assert cfg.Lp == 5
        np.testing.assert_allclose(cfg.lam, 3e8 / 2.4e9, rtol=1e-10)
