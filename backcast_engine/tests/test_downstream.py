"""Tests for downstream modules (covariance, uncertainty, backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset
from backcast.downstream.backtest import (
    STRATEGY_REGISTRY,
    equal_weight,
    inverse_volatility,
    min_variance,
    risk_parity,
    run_backtest,
)
from backcast.downstream.covariance import (
    combined_covariance,
    denoise_covariance,
    from_em_result,
    sample_covariance,
    shrink_covariance,
)
from backcast.downstream.uncertainty import (
    box_uncertainty,
    ellipsoidal_uncertainty,
    portfolio_risk_distribution,
)
from backcast.imputation.multiple_impute import multiple_impute
from backcast.models.em_stambaugh import em_stambaugh


# ---------------------------------------------------------------------------
# Fixture: small synthetic imputations
# ---------------------------------------------------------------------------

def _mi_and_ds(T=1500, N_long=3, N_short=2, start=800, seed=0, M=20):
    rng = np.random.default_rng(seed)
    N = N_long + N_short
    A = rng.standard_normal((N, N))
    sigma = (A @ A.T) * 1e-4 + np.eye(N) * 5e-5
    mu = np.zeros(N)
    R = rng.multivariate_normal(mu, sigma, size=T)
    cols = [f"L{i}" for i in range(N_long)] + [f"S{i}" for i in range(N_short)]
    idx = pd.date_range("1990-01-02", periods=T, freq="B")
    df = pd.DataFrame(R, index=idx, columns=cols)
    df.iloc[:start, N_long:] = np.nan
    df.index.name = "date"
    ds = build_backcast_dataset(df)
    em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
    mi = multiple_impute(ds, em, n_imputations=M, seed=0)
    return ds, em, mi, sigma


# ---------------------------------------------------------------------------
# covariance.py
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_from_em_result(self):
        _, em, _, _ = _mi_and_ds(seed=1)
        res = from_em_result(em)
        assert res.covariance.shape == em.sigma.shape
        assert (np.linalg.eigvalsh(res.covariance) > -1e-10).all()
        assert res.method == "em"

    def test_sample_covariance_matches_numpy(self):
        _, _, mi, _ = _mi_and_ds(seed=2)
        df = mi.imputations[0]
        res = sample_covariance(df)
        expected = np.cov(df.to_numpy(), rowvar=False, bias=False)
        np.testing.assert_allclose(res.covariance, expected, atol=1e-10)

    def test_combined_covariance_rubin(self):
        _, _, mi, _ = _mi_and_ds(seed=3)
        res = combined_covariance(mi.imputations)
        assert res.method == "rubin_combined"
        assert res.n_imputations == 20
        assert res.within_variance is not None
        assert res.between_variance is not None
        # Total = within + (1 + 1/M)*between
        M = res.n_imputations
        expected_total = res.within_variance + (1 + 1/M) * res.between_variance
        np.testing.assert_allclose(res.total_variance, expected_total)

    def test_shrinkage_reduces_condition(self):
        _, _, mi, _ = _mi_and_ds(seed=4)
        df = mi.imputations[0]
        s = shrink_covariance(df)
        p = sample_covariance(df)
        # Shrinkage should not make the condition number worse
        assert s.condition_number <= p.condition_number * 1.001

    def test_denoise_preserves_trace(self):
        _, _, mi, _ = _mi_and_ds(seed=5)
        df = mi.imputations[0]
        d = denoise_covariance(df)
        p = sample_covariance(df)
        # Trace should be approximately preserved
        assert abs(np.trace(d.covariance) - np.trace(p.covariance)) / np.trace(p.covariance) < 0.05
        # Denoised should be PSD
        assert (np.linalg.eigvalsh(d.covariance) > -1e-10).all()

    def test_shrinkage_rejects_nan(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 3)) * 0.01, columns=list("ABC"))
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="fully-observed"):
            shrink_covariance(df)


# ---------------------------------------------------------------------------
# uncertainty.py
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_ellipsoid_contains_center(self):
        _, _, mi, _ = _mi_and_ds(seed=6)
        ell = ellipsoidal_uncertainty(mi.imputations, confidence=0.95)
        # Distance from center to itself = 0 ≤ κ²
        assert ell.kappa > 0
        assert ell.scaling.shape == (5, 5)
        # Scaling PSD after regularisation
        assert (np.linalg.eigvalsh(ell.scaling) > 0).all()

    def test_ellipsoid_confidence_bounds(self):
        _, _, mi, _ = _mi_and_ds(seed=7)
        e95 = ellipsoidal_uncertainty(mi.imputations, confidence=0.95)
        e99 = ellipsoidal_uncertainty(mi.imputations, confidence=0.99)
        assert e99.kappa > e95.kappa

    def test_box_uncertainty_shapes(self):
        _, _, mi, _ = _mi_and_ds(seed=8)
        N = mi.imputations[0].shape[1]
        box = box_uncertainty(mi.imputations, confidence=0.9)
        assert box.mu_lower.shape == (N,)
        assert box.mu_upper.shape == (N,)
        assert box.cov_lower.shape == (N, N)
        assert (box.mu_lower <= box.mu_upper).all()

    def test_portfolio_risk_distribution(self):
        _, _, mi, _ = _mi_and_ds(seed=9)
        N = mi.imputations[0].shape[1]
        w = np.full(N, 1.0 / N)
        dist = portfolio_risk_distribution(w, mi.imputations)
        assert dist.portfolio_risks.shape == (mi.n_imputations,)
        assert dist.percentile_5 <= dist.median_risk <= dist.percentile_95
        assert dist.portfolio_risks.min() > 0

    def test_risk_distribution_shape_check(self):
        _, _, mi, _ = _mi_and_ds(seed=10)
        with pytest.raises(ValueError, match="weights length"):
            portfolio_risk_distribution(np.array([0.5]), mi.imputations)


# ---------------------------------------------------------------------------
# backtest.py
# ---------------------------------------------------------------------------

class TestStrategies:
    def _window(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((200, 4)) * 0.01, columns=list("ABCD"))
        return df

    def test_equal_weight(self):
        w = equal_weight(self._window(), lookback=63)
        np.testing.assert_allclose(w, [0.25] * 4)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_inverse_volatility_sums_to_one(self):
        w = inverse_volatility(self._window(), lookback=63)
        assert abs(w.sum() - 1.0) < 1e-12
        assert (w > 0).all()

    def test_min_variance_sums_to_one(self):
        w = min_variance(self._window(), lookback=63)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_risk_parity_sums_to_one(self):
        w = risk_parity(self._window(), lookback=63)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= 0).all()


class TestBacktestHarness:
    def test_equal_weight_runs_end_to_end(self):
        _, _, mi, _ = _mi_and_ds(T=800, start=400, seed=11, M=5)
        res = run_backtest(mi.imputations, strategy="equal_weight",
                           lookback=30, rebalance_freq=20)
        assert res.n_imputations == 5
        assert len(res.per_imputation) == 5
        assert res.cumulative_median.iloc[0] > 0
        assert res.sharpe_distribution.shape == (5,)

    def test_percentile_bands_ordered(self):
        _, _, mi, _ = _mi_and_ds(T=600, start=300, seed=12, M=10)
        res = run_backtest(mi.imputations, strategy="inverse_volatility",
                           lookback=30, rebalance_freq=20)
        # At every t, p05 <= median <= p95
        assert (res.cumulative_p05 <= res.cumulative_median + 1e-10).all()
        assert (res.cumulative_median <= res.cumulative_p95 + 1e-10).all()

    def test_unknown_strategy_raises(self):
        _, _, mi, _ = _mi_and_ds(seed=13, M=3)
        with pytest.raises(ValueError, match="unknown strategy"):
            run_backtest(mi.imputations, strategy="blargh")

    def test_custom_strategy_callable(self):
        _, _, mi, _ = _mi_and_ds(T=500, start=250, seed=14, M=3)
        def my_strat(window, lookback):
            N = window.shape[1]
            w = np.zeros(N)
            w[0] = 1.0
            return w
        res = run_backtest(mi.imputations, strategy=my_strat,
                           strategy_name="first_asset",
                           lookback=30, rebalance_freq=10)
        assert res.strategy_name == "first_asset"

    def test_all_registry_strategies_run(self):
        _, _, mi, _ = _mi_and_ds(T=500, start=250, seed=15, M=3)
        for name in STRATEGY_REGISTRY:
            res = run_backtest(mi.imputations, strategy=name,
                                lookback=30, rebalance_freq=10)
            assert res.n_imputations == 3
