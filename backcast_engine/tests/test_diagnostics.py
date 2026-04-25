"""Tests for backcast.validation.diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backcast.validation import diagnostics as D


class TestNormality:
    def test_gaussian_passes(self):
        rng = np.random.default_rng(0)
        r = pd.DataFrame(rng.standard_normal((2000, 2)), columns=["a", "b"])
        out = D.residual_normality(r)
        # JB p-values should not reject at 0.01 for clean Gaussian
        assert (out["jb_pvalue"] > 0.01).all()

    def test_non_gaussian_rejected(self):
        rng = np.random.default_rng(1)
        r = pd.DataFrame(rng.standard_t(3, size=(3000, 1)), columns=["a"])
        out = D.residual_normality(r)
        assert out["jb_pvalue"].iloc[0] < 1e-5


class TestAutocorrelation:
    def test_iid_noise_not_rejected(self):
        rng = np.random.default_rng(2)
        r = pd.DataFrame(rng.standard_normal((2000, 1)), columns=["a"])
        out = D.residual_autocorrelation(r, lag=10)
        assert out["lb_pvalue"].iloc[0] > 0.05

    def test_ar1_series_rejected(self):
        rng = np.random.default_rng(3)
        n = 2000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()
        out = D.residual_autocorrelation(pd.DataFrame({"ar1": x}))
        assert out["lb_pvalue"].iloc[0] < 1e-5


class TestSpectrum:
    def test_identical_matrices(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((5, 5))
        S = A @ A.T
        out = D.eigenvalue_comparison(S, S)
        assert out["max_abs_diff"] < 1e-10
        np.testing.assert_allclose(out["eig_a"], out["eig_b"])


class TestRollingCorrelation:
    def test_shape_and_leading_nan(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.standard_normal((300, 2)), columns=["a", "b"])
        s = D.rolling_correlation(df, ("a", "b"), window=100)
        assert s.iloc[:99].isna().all()
        assert s.iloc[99:].notna().all()


class TestQQPlotData:
    def test_sorted_output_matches_input(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(500)
        theo, samp = D.qq_plot_data(x)
        assert len(theo) == len(samp)
        assert np.all(np.diff(samp) >= 0)  # sorted ascending


class TestSummary:
    def test_summary_columns(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(rng.standard_normal((500, 3)), columns=list("abc"))
        out = D.summarise_residual_diagnostics(df)
        for col in ("mean", "std", "skew", "ex_kurtosis", "jb_pvalue", "lb_pvalue"):
            assert col in out.columns
        assert len(out) == 3
