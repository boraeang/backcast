"""Tests for backcast.validation.metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from backcast.validation import metrics as M


class TestPointMetrics:
    def test_rmse_zero_for_identical(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(M.rmse(x, x), [0.0, 0.0])

    def test_rmse_known_value(self):
        a = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, -1.0, 1.0])
        # errors = [1, -1, 1], squared = [1,1,1], mean = 1, sqrt = 1
        assert M.rmse(a, p) == pytest.approx(1.0)

    def test_mae_known_value(self):
        a = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, -2.0, 3.0])
        assert M.mae(a, p) == pytest.approx(2.0)

    def test_correlation_error_zero(self):
        c = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert M.correlation_error(c, c) == 0.0

    def test_vol_ratio(self):
        assert M.vol_ratio(2.0, 1.0) == 0.5


class TestKS:
    def test_identical_samples_ks_high_pvalue(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(1000)
        p = rng.standard_normal(1000)
        _, pvals = M.ks_test_per_asset(a, p)
        assert pvals[0] > 0.05

    def test_different_distributions_ks_low_pvalue(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(2000)
        p = rng.standard_normal(2000) * 3.0
        _, pvals = M.ks_test_per_asset(a, p)
        assert pvals[0] < 1e-3


class TestCoverage:
    def test_full_coverage(self):
        a = np.array([[0.5, 0.3], [0.2, 0.7]])
        lo = np.full_like(a, -1.0)
        hi = np.full_like(a,  1.0)
        assert M.coverage_rate(a, lo, hi) == 1.0

    def test_zero_coverage(self):
        a = np.array([[0.5], [0.3]])
        lo = np.array([[1.0], [1.0]])
        hi = np.array([[2.0], [2.0]])
        assert M.coverage_rate(a, lo, hi) == 0.0

    def test_coverage_per_asset(self):
        a = np.array([[0.5, 1.5], [0.5, 1.5]])
        lo = np.array([[0.0, 0.0], [0.0, 0.0]])
        hi = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(M.coverage_rate_per_asset(a, lo, hi), [1.0, 0.0])


class TestPIT:
    def test_uniform_pit_for_well_calibrated(self):
        rng = np.random.default_rng(42)
        mu = np.zeros(10_000)
        sigma = np.ones(10_000)
        draws = rng.standard_normal(10_000)
        pit, counts, edges = M.pit_histogram(draws, mu, sigma, bins=10)
        # Expected count per bin ≈ 1000.  Check χ²-like uniformity.
        expected = len(pit) / 10
        chi2 = ((counts - expected) ** 2 / expected).sum()
        # For 9 dof χ²_{0.99} ≈ 21.67.  Comfortably well below.
        assert chi2 < 25.0

    def test_pit_bounds(self):
        draws = np.array([[0.0, 0.5], [1.0, -1.0]])
        pit, _, _ = M.pit_histogram(draws, 0.0, 1.0)
        assert (pit >= 0.0).all() and (pit <= 1.0).all()


class TestTailDependence:
    def test_independence_has_low_tail_dep(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(10_000)
        y = rng.standard_normal(10_000)
        td = M.tail_dependence_coeff(x, y, quantile=0.05, tail="lower")
        # Independent Gaussians: theoretical λ_L = 0 (asymptotic).
        # Empirically with q=5% we expect ≈ 0.05.
        assert 0.0 <= td < 0.15

    def test_perfect_dependence_high_tail_dep(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(5000)
        y = x.copy()
        td = M.tail_dependence_coeff(x, y, quantile=0.05, tail="lower")
        assert td > 0.9

    def test_bad_tail_arg_raises(self):
        with pytest.raises(ValueError, match="tail must be"):
            M.tail_dependence_coeff([0.0], [0.0], tail="middle")
