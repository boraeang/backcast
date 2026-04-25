"""Tests for Tier 1: i.i.d. multivariate Gaussian returns.

Validation criteria from the spec
-----------------------------------
1. Sample mean within 2 * sigma / sqrt(T) of true mean for each asset.
2. Sample covariance Frobenius relative error < 5% for T=5000.
3. Marginal distributions pass KS test for normality (p > 0.01).
4. No autocorrelation: Ljung-Box test does not reject at 5% for each asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import kstest, norm

from synthgen.config import SyntheticConfig, build_asset_profiles, get_short_start_indices
from synthgen.tier1_stationary import generate_tier1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ljung_box_p(series: np.ndarray, lags: int = 10) -> float:
    """Return the minimum p-value from Ljung-Box test across lags 1..lags."""
    from scipy.stats import chi2

    n = len(series)
    acf = np.correlate(series - series.mean(), series - series.mean(), mode="full")
    acf = acf[n - 1 :]
    acf = acf / acf[0]

    q_stat = 0.0
    min_p = 1.0
    for lag in range(1, lags + 1):
        q_stat += (n * (n + 2) * acf[lag] ** 2) / (n - lag)
        p_val = 1.0 - chi2.cdf(q_stat, df=lag)
        min_p = min(min_p, p_val)
    return min_p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier1_output():
    cfg = SyntheticConfig(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        correlation_method="factor_model",
        n_factors=4,
    )
    masked, complete, gt = generate_tier1(cfg)
    return masked, complete, gt, cfg


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_profiles(self):
        cfg = SyntheticConfig()
        profiles = build_asset_profiles(cfg)
        assert len(profiles) == cfg.n_long_assets + cfg.n_short_assets

    def test_staggered_starts(self):
        cfg = SyntheticConfig(
            n_short_assets=3,
            t_total=5000,
            short_start_day=[2000, 3000, 4000],
        )
        profiles = build_asset_profiles(cfg)
        short_names = list(profiles.keys())[cfg.n_long_assets:]
        starts = get_short_start_indices(cfg, short_names)
        assert list(starts.values()) == [2000, 3000, 4000]

    def test_invalid_start_day_raises(self):
        cfg = SyntheticConfig(n_short_assets=2, t_total=1000, short_start_day=[500, 1500])
        profiles = build_asset_profiles(cfg)
        short_names = list(profiles.keys())[cfg.n_long_assets:]
        with pytest.raises(ValueError, match="must be in"):
            get_short_start_indices(cfg, short_names)

    def test_mismatched_profiles_raises(self):
        cfg = SyntheticConfig(n_long_assets=2, n_short_assets=2,
                               asset_profiles={"A": (8, 16)})  # length 1 != 4
        with pytest.raises(ValueError, match="n_long_assets"):
            build_asset_profiles(cfg)


# ---------------------------------------------------------------------------
# calendar.py tests
# ---------------------------------------------------------------------------

class TestCalendar:
    def test_length(self):
        from synthgen.calendar import generate_business_days
        dates = generate_business_days("1990-01-02", 100)
        assert len(dates) == 100

    def test_no_weekends(self):
        from synthgen.calendar import generate_business_days
        dates = generate_business_days("1990-01-02", 500)
        assert all(d.day_of_week < 5 for d in dates)

    def test_weekend_start_advances(self):
        from synthgen.calendar import generate_business_days
        # 1990-01-06 is a Saturday → should start on Monday 1990-01-08
        dates = generate_business_days("1990-01-06", 5)
        assert dates[0].day_of_week < 5  # Must start on a weekday


# ---------------------------------------------------------------------------
# io.py tests
# ---------------------------------------------------------------------------

class TestIO:
    def test_round_trip(self, tmp_path):
        from synthgen.io import load_ground_truth, load_returns, save_dataset

        cfg = SyntheticConfig(t_total=50, n_long_assets=2, n_short_assets=1,
                               short_start_day=30, seed=0)
        masked, complete, gt = generate_tier1(cfg)
        paths = save_dataset(tmp_path, masked, gt, complete)

        assert paths["returns"].exists()
        assert paths["returns_complete"].exists()
        assert paths["ground_truth"].exists()

        df_loaded = load_returns(paths["returns"])
        assert df_loaded.shape == masked.shape

        gt_loaded = load_ground_truth(paths["ground_truth"])
        assert gt_loaded["tier"] == 1
        assert "mu" in gt_loaded

    def test_ground_truth_no_numpy_types(self, tmp_path):
        import json
        from synthgen.io import save_dataset

        cfg = SyntheticConfig(t_total=50, n_long_assets=2, n_short_assets=1,
                               short_start_day=30, seed=1)
        masked, complete, gt = generate_tier1(cfg)
        paths = save_dataset(tmp_path, masked, gt, complete)

        # Should not raise JSONDecodeError or produce non-JSON types
        with open(paths["ground_truth"]) as f:
            data = json.load(f)
        assert isinstance(data["mu"][0], float)


# ---------------------------------------------------------------------------
# Tier 1 statistical tests
# ---------------------------------------------------------------------------

class TestTier1Statistics:
    def test_sample_mean_within_2sigma_over_sqrtT(self, tier1_output):
        """Sample mean should be within 2 * sigma / sqrt(T) of true mean."""
        masked, complete, gt, cfg = tier1_output
        true_mu = np.array(gt["mu_daily"])
        true_sigma_diag = np.sqrt(np.diag(np.array(gt["sigma_daily"])))
        T = cfg.t_total

        sample_mu = complete.mean().values
        tolerance = 2.0 * true_sigma_diag / np.sqrt(T)

        # Allow up to 1 asset to fail (multiple testing)
        failures = np.abs(sample_mu - true_mu) > tolerance
        assert failures.sum() <= 1, (
            f"{failures.sum()} assets have sample mean outside 2σ/√T tolerance: "
            f"{np.array(gt['asset_names'])[failures]}"
        )

    def test_sample_covariance_frobenius_error(self, tier1_output):
        """Relative Frobenius error of sample covariance < 5% for T=5000."""
        masked, complete, gt, cfg = tier1_output
        true_sigma = np.array(gt["sigma_daily"])
        sample_sigma = complete.cov().values

        rel_error = np.linalg.norm(sample_sigma - true_sigma, "fro") / np.linalg.norm(
            true_sigma, "fro"
        )
        assert rel_error < 0.05, f"Frobenius relative error = {rel_error:.4f} ≥ 5%"

    def test_marginal_normality_ks(self, tier1_output):
        """Marginal distributions should pass KS test for normality (p > 0.01)."""
        masked, complete, gt, cfg = tier1_output
        true_mu = np.array(gt["mu_daily"])
        true_vols = np.sqrt(np.diag(np.array(gt["sigma_daily"])))

        n_fail = 0
        for i, col in enumerate(complete.columns):
            standardised = (complete[col].values - true_mu[i]) / true_vols[i]
            _, p = kstest(standardised, "norm")
            if p < 0.01:
                n_fail += 1

        # Allow up to 1 failure (multiple testing with 8 assets)
        assert n_fail <= 1, f"{n_fail} assets fail KS normality test at p=0.01"

    def test_no_autocorrelation_ljung_box(self, tier1_output):
        """Ljung-Box should not reject at 5% significance (i.i.d. DGP)."""
        masked, complete, gt, cfg = tier1_output
        n_fail = 0
        for col in complete.columns:
            p = _ljung_box_p(complete[col].values, lags=10)
            if p < 0.05:
                n_fail += 1

        # Allow up to 1 failure
        assert n_fail <= 1, f"{n_fail} assets reject the no-autocorrelation hypothesis"

    def test_returns_shape(self, tier1_output):
        masked, complete, gt, cfg = tier1_output
        assert complete.shape == (cfg.t_total, cfg.n_long_assets + cfg.n_short_assets)
        assert masked.shape == complete.shape

    def test_masked_nan_pattern(self, tier1_output):
        """Short assets have NaN before their start date and valid values after."""
        masked, complete, gt, cfg = tier1_output
        for name, idx in gt["short_asset_start_indices"].items():
            assert masked[name].iloc[:idx].isna().all(), \
                f"{name}: expected NaN before row {idx}"
            assert masked[name].iloc[idx:].notna().all(), \
                f"{name}: expected non-NaN from row {idx}"

    def test_long_assets_fully_observed(self, tier1_output):
        masked, complete, gt, cfg = tier1_output
        for name in gt["long_assets"]:
            assert masked[name].notna().all(), f"{name} should have no NaNs"

    def test_ground_truth_covariance_psd(self, tier1_output):
        masked, complete, gt, cfg = tier1_output
        sigma = np.array(gt["sigma_daily"])
        from synthgen.correlation import is_psd
        assert is_psd(sigma), "Ground truth covariance must be PSD"

    def test_ground_truth_correlation_diagonal(self, tier1_output):
        masked, complete, gt, cfg = tier1_output
        corr = np.array(gt["correlation"])
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-8)

    def test_reproducibility(self):
        """Same seed → identical output."""
        cfg = SyntheticConfig(t_total=200, short_start_day=150, seed=99)
        _, comp1, _ = generate_tier1(cfg)
        _, comp2, _ = generate_tier1(cfg)
        pd.testing.assert_frame_equal(comp1, comp2)

    def test_different_seeds_differ(self):
        cfg1 = SyntheticConfig(t_total=200, short_start_day=150, seed=1)
        cfg2 = SyntheticConfig(t_total=200, short_start_day=150, seed=2)
        _, comp1, _ = generate_tier1(cfg1)
        _, comp2, _ = generate_tier1(cfg2)
        assert not comp1.equals(comp2)
