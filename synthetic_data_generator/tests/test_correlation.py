"""Tests for synthgen.correlation — factor-model, random, and manual covariance builders."""
from __future__ import annotations

import numpy as np
import pytest

from synthgen.correlation import (
    build_covariance,
    build_factor_model_covariance,
    is_psd,
    nearest_psd,
)


class TestCorrelation:
    def test_factor_model_psd(self):
        names = ["EQUITY_1", "EQUITY_2", "BOND_1", "GOLD", "CRYPTO_1"]
        vols = np.array([0.01, 0.011, 0.003, 0.01, 0.044])
        result = build_factor_model_covariance(names, vols, n_factors=4)
        assert is_psd(result.sigma_daily), "Sigma must be PSD"

    def test_factor_model_diagonal_matches_vols(self):
        names = ["EQUITY_1", "EQUITY_2", "BOND_1", "GOLD", "CRYPTO_1"]
        vols = np.array([0.01, 0.011, 0.003, 0.01, 0.044])
        result = build_factor_model_covariance(names, vols, n_factors=4)
        actual_vols = np.sqrt(np.diag(result.sigma_daily))
        np.testing.assert_allclose(actual_vols, vols, rtol=1e-6)

    def test_equity_equity_correlation_range(self):
        names = ["EQUITY_1", "EQUITY_2", "BOND_1", "GOLD", "CRYPTO_1"]
        vols = np.array([0.01, 0.011, 0.003, 0.01, 0.044])
        result = build_factor_model_covariance(names, vols, n_factors=4)
        eq_eq = result.corr[0, 1]
        assert 0.40 < eq_eq < 0.90, f"Equity-equity corr {eq_eq:.3f} out of range"

    def test_equity_bond_correlation_range(self):
        names = ["EQUITY_1", "EQUITY_2", "BOND_1", "GOLD", "CRYPTO_1"]
        vols = np.array([0.01, 0.011, 0.003, 0.01, 0.044])
        result = build_factor_model_covariance(names, vols, n_factors=4)
        eq_bond = result.corr[0, 2]
        assert -0.30 < eq_bond < 0.20, f"Equity-bond corr {eq_bond:.3f} out of range"

    def test_equity_crypto_correlation_range(self):
        names = ["EQUITY_1", "EQUITY_2", "BOND_1", "EQUITY_3", "CRYPTO_1"]
        vols = np.array([0.01, 0.011, 0.003, 0.01, 0.044])
        result = build_factor_model_covariance(names, vols, n_factors=4)
        eq_crypto = result.corr[0, 4]
        assert 0.20 < eq_crypto < 0.60, f"Equity-crypto corr {eq_crypto:.3f} out of range"

    def test_correlation_diagonal_is_one(self):
        names = ["EQUITY_1", "BOND_1", "CRYPTO_1"]
        vols = np.array([0.01, 0.003, 0.044])
        result = build_factor_model_covariance(names, vols)
        np.testing.assert_allclose(np.diag(result.corr), 1.0, atol=1e-8)

    def test_nearest_psd_makes_psd(self):
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PSD
        fixed = nearest_psd(bad)
        assert is_psd(fixed)

    def test_random_method_psd(self):
        rng = np.random.default_rng(0)
        names = ["A", "B", "C", "D"]
        vols = np.array([0.01, 0.005, 0.02, 0.03])
        result = build_covariance(names, vols, method="random", rng=rng)
        assert is_psd(result.sigma_daily)

    def test_manual_method_invalid_raises(self):
        bad_corr = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalue < 0
        vols = np.array([0.01, 0.01])
        # Should not raise — nearest_psd is applied internally with a warning
        result = build_covariance(
            ["A", "B"], vols, method="manual", manual_corr=bad_corr
        )
        assert is_psd(result.sigma_daily)
