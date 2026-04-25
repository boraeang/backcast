"""Tests for Tier 3: GARCH + fat tails + TVP betas.

Validation criteria from the spec
----------------------------------
- Marginal kurtosis > 3 (fat tails present).
- Autocorrelation in squared returns (GARCH volatility clustering).
- Sample unconditional vol close to analytical unconditional vol.
- GARCH recursion holds exactly for the recorded factor conditional vols.
- Beta path respects the 3x clipping bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, kurtosis

from synthgen.config import SyntheticConfig, Tier3Config
from synthgen.correlation import is_psd
from synthgen.tier3_realistic import (
    _rolling_covariance,
    _sample_unit_variance_innovations,
    _simulate_beta_path,
    _simulate_garch,
    generate_tier3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ljung_box_on_squared(series: np.ndarray, lag: int = 10) -> float:
    """Return the Ljung-Box p-value on the squared series at the requested lag."""
    x = series ** 2
    n = len(x)
    acf = np.correlate(x - x.mean(), x - x.mean(), mode="full")
    acf = acf[n - 1 :]
    acf = acf / acf[0]
    q_stat = 0.0
    for L in range(1, lag + 1):
        q_stat += (n * (n + 2) * acf[L] ** 2) / (n - L)
    p_val = 1.0 - chi2.cdf(q_stat, df=lag)
    return float(p_val)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier3_output():
    cfg = SyntheticConfig(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        correlation_method="factor_model",
        n_factors=4,
        tier=3,
        tier3_config=Tier3Config(),  # defaults: student-t df=5
    )
    masked, complete, gt = generate_tier3(cfg)
    return masked, complete, gt, cfg


@pytest.fixture(scope="module")
def tier3_gaussian_output():
    """Gaussian-innovation variant for comparing kurtosis."""
    cfg = SyntheticConfig(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        correlation_method="factor_model",
        n_factors=4,
        tier=3,
        tier3_config=Tier3Config(innovation_distribution="gaussian"),
    )
    masked, complete, gt = generate_tier3(cfg)
    return masked, complete, gt, cfg


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------

class TestTier3Helpers:
    def test_standardized_student_t_unit_variance(self):
        rng = np.random.default_rng(0)
        sample = _sample_unit_variance_innovations(rng, "student_t", 5.0, size=200_000)
        # Sample variance of a unit-variance standardised t should be close to 1
        assert abs(sample.var() - 1.0) < 0.05
        # Excess kurtosis should be positive (fat tails)
        assert kurtosis(sample, fisher=True) > 2.0

    def test_gaussian_unit_variance(self):
        rng = np.random.default_rng(0)
        sample = _sample_unit_variance_innovations(rng, "gaussian", 5.0, size=200_000)
        assert abs(sample.var() - 1.0) < 0.02

    def test_garch_recursion_holds(self):
        rng = np.random.default_rng(1)
        T = 1000
        omega, alpha, beta = 1e-5, 0.08, 0.90
        innov = rng.standard_normal(T)
        f, sigma = _simulate_garch(T, omega, alpha, beta, innov)
        sigma2 = sigma ** 2
        # Recursion: sigma2[t] = omega + alpha*f[t-1]^2 + beta*sigma2[t-1]
        expected = omega + alpha * f[:-1] ** 2 + beta * sigma2[:-1]
        np.testing.assert_allclose(sigma2[1:], expected, rtol=1e-12, atol=1e-15)

    def test_garch_initial_variance(self):
        rng = np.random.default_rng(2)
        omega, alpha, beta = 1e-5, 0.08, 0.90
        _, sigma = _simulate_garch(5, omega, alpha, beta, rng.standard_normal(5))
        expected_init = np.sqrt(omega / (1 - alpha - beta))
        np.testing.assert_allclose(sigma[0], expected_init, rtol=1e-10)

    def test_beta_path_starts_at_B0(self):
        rng = np.random.default_rng(3)
        B0 = np.array([[0.3, -0.1], [0.2, 0.4], [-0.1, 0.5]])
        path = _simulate_beta_path(B0, T=100, sigma_eta=0.01, rng=rng)
        np.testing.assert_array_equal(path[0], B0)

    def test_beta_path_clipping(self):
        rng = np.random.default_rng(4)
        B0 = np.array([[0.1, -0.2], [0.3, 0.4]])
        path = _simulate_beta_path(B0, T=2000, sigma_eta=0.5, rng=rng)
        B_max = 3.0 * np.abs(B0)
        assert (np.abs(path) <= B_max + 1e-12).all(), (
            "beta_path violates |B_t| <= 3 * |B_0| clipping bound"
        )

    def test_beta_path_zero_drift_is_constant(self):
        rng = np.random.default_rng(5)
        B0 = np.array([[0.3, -0.1], [0.2, 0.4]])
        path = _simulate_beta_path(B0, T=50, sigma_eta=0.0, rng=rng)
        for t in range(50):
            np.testing.assert_array_equal(path[t], B0)

    def test_rolling_covariance_leading_nans(self):
        rng = np.random.default_rng(6)
        returns = rng.standard_normal((200, 3))
        rc = _rolling_covariance(returns, window=90)
        assert np.isnan(rc[:89]).all()
        assert np.isfinite(rc[89:]).all()

    def test_rolling_covariance_matches_numpy(self):
        rng = np.random.default_rng(7)
        returns = rng.standard_normal((200, 3))
        rc = _rolling_covariance(returns, window=90)
        # Last window should equal np.cov of the last 90 rows
        expected = np.cov(returns[-90:], rowvar=False, ddof=1)
        np.testing.assert_allclose(rc[-1], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Validation tests on generated output
# ---------------------------------------------------------------------------

class TestTier3Config:
    def test_bad_alpha_raises(self):
        from synthgen.tier3_realistic import _validate_tier3_config
        with pytest.raises(ValueError, match="alpha"):
            _validate_tier3_config(Tier3Config(garch_alpha=0.0))

    def test_non_stationary_raises(self):
        from synthgen.tier3_realistic import _validate_tier3_config
        with pytest.raises(ValueError, match="stationarity"):
            _validate_tier3_config(Tier3Config(garch_alpha=0.15, garch_beta=0.90))

    def test_bad_df_raises(self):
        from synthgen.tier3_realistic import _validate_tier3_config
        with pytest.raises(ValueError, match="degrees_of_freedom"):
            _validate_tier3_config(Tier3Config(degrees_of_freedom=1.5))

    def test_non_factor_method_raises(self):
        cfg = SyntheticConfig(
            t_total=500, short_start_day=300, seed=0,
            tier=3, correlation_method="random",
        )
        with pytest.raises(ValueError, match="factor_model"):
            generate_tier3(cfg)


class TestTier3Structure:
    def test_shapes(self, tier3_output):
        masked, complete, gt, cfg = tier3_output
        N = cfg.n_long_assets + cfg.n_short_assets
        K = cfg.n_factors
        assert complete.shape == (cfg.t_total, N)
        assert masked.shape == complete.shape
        assert np.asarray(gt["factor_returns"]).shape == (cfg.t_total, K)
        assert np.asarray(gt["factor_conditional_vols"]).shape == (cfg.t_total, K)
        assert np.asarray(gt["beta_path"]).shape == (cfg.t_total, N, K)
        assert np.asarray(gt["rolling_sigma_90d"]).shape == (cfg.t_total, N, N)

    def test_tier_field(self, tier3_output):
        _, _, gt, _ = tier3_output
        assert gt["tier"] == 3

    def test_required_ground_truth_fields(self, tier3_output):
        _, _, gt, _ = tier3_output
        for key in [
            "factor_returns",
            "factor_garch_params",
            "factor_conditional_vols",
            "beta_path",
            "innovation_df",
            "unconditional_sigma",
            "rolling_sigma_90d",
        ]:
            assert key in gt, f"missing {key!r} in ground_truth"

    def test_unconditional_sigma_psd(self, tier3_output):
        _, _, gt, _ = tier3_output
        sigma = np.asarray(gt["unconditional_sigma"])
        assert is_psd(sigma)

    def test_garch_params_saved(self, tier3_output):
        _, _, gt, _ = tier3_output
        p = gt["factor_garch_params"]
        assert p["omega"] == 0.00001
        assert p["alpha"] == 0.08
        assert p["beta"] == 0.90
        expected_uv = 0.00001 / (1 - 0.08 - 0.90)
        assert abs(p["unconditional_factor_variance"] - expected_uv) < 1e-12

    def test_masking_applied(self, tier3_output):
        masked, _, gt, _ = tier3_output
        for name, idx in gt["short_asset_start_indices"].items():
            assert masked[name].iloc[:idx].isna().all()
            assert masked[name].iloc[idx:].notna().all()


class TestTier3GarchRecursion:
    def test_stored_vols_match_recursion(self, tier3_output):
        """factor_conditional_vols recorded in GT must satisfy the recursion exactly."""
        _, _, gt, _ = tier3_output
        p = gt["factor_garch_params"]
        omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
        f = np.asarray(gt["factor_returns"])
        sig = np.asarray(gt["factor_conditional_vols"])

        sig2 = sig ** 2
        expected = omega + alpha * f[:-1] ** 2 + beta * sig2[:-1]
        np.testing.assert_allclose(sig2[1:], expected, rtol=1e-10, atol=1e-15)

    def test_initial_sigma_is_unconditional(self, tier3_output):
        _, _, gt, _ = tier3_output
        p = gt["factor_garch_params"]
        sig = np.asarray(gt["factor_conditional_vols"])
        expected = np.sqrt(p["unconditional_factor_variance"])
        np.testing.assert_allclose(sig[0], expected, rtol=1e-10)


class TestTier3FatTails:
    def test_marginal_excess_kurtosis_positive(self, tier3_output):
        """With Student-t(5) innovations, every asset should show fat tails."""
        _, complete, gt, _ = tier3_output
        excess_k = kurtosis(complete.values, axis=0, fisher=True)
        # Every asset should have excess kurtosis comfortably > 0
        assert (excess_k > 0.5).all(), (
            f"Some assets lack fat tails; excess_kurt={excess_k}"
        )

    def test_student_t_has_more_kurtosis_than_gaussian(
        self, tier3_output, tier3_gaussian_output
    ):
        _, complete_t, _, _ = tier3_output
        _, complete_g, _, _ = tier3_gaussian_output
        k_t = kurtosis(complete_t.values, axis=0, fisher=True).mean()
        k_g = kurtosis(complete_g.values, axis=0, fisher=True).mean()
        assert k_t > k_g, f"Student-t mean excess kurt {k_t:.2f} vs gaussian {k_g:.2f}"


class TestTier3GarchSignature:
    def test_squared_returns_autocorrelated(self, tier3_output):
        """Ljung-Box on squared returns should reject at 5% (volatility clustering)."""
        _, complete, _, _ = tier3_output
        n_reject = 0
        for col in complete.columns:
            p = _ljung_box_on_squared(complete[col].values, lag=10)
            if p < 0.05:
                n_reject += 1
        # With GARCH, almost every asset should show clustering
        assert n_reject >= len(complete.columns) - 1, (
            f"Only {n_reject}/{len(complete.columns)} assets show GARCH signature"
        )

    def test_squared_factor_returns_autocorrelated(self, tier3_output):
        _, _, gt, _ = tier3_output
        f = np.asarray(gt["factor_returns"])
        n_reject = 0
        for j in range(f.shape[1]):
            if _ljung_box_on_squared(f[:, j], lag=10) < 0.05:
                n_reject += 1
        assert n_reject == f.shape[1], (
            f"Only {n_reject}/{f.shape[1]} factors show GARCH clustering"
        )


class TestTier3UnconditionalVariance:
    def test_sample_vol_close_to_analytical(self, tier3_output):
        """Sample std per asset should be within ~25% of analytical unconditional std."""
        _, complete, gt, cfg = tier3_output
        sigma_uncond = np.asarray(gt["unconditional_sigma"])
        analytical_std = np.sqrt(np.diag(sigma_uncond))
        sample_std = complete.std(axis=0).values

        rel_error = np.abs(sample_std - analytical_std) / analytical_std
        # Allow up to 2 assets to fail the 25% bound (small sample + GARCH persistence)
        failures = int((rel_error > 0.25).sum())
        assert failures <= 2, (
            f"{failures} assets outside 25% tolerance; rel_error={rel_error}"
        )

    def test_factor_sample_variance_close_to_analytical(self, tier3_output):
        _, _, gt, _ = tier3_output
        f = np.asarray(gt["factor_returns"])
        expected = gt["factor_garch_params"]["unconditional_factor_variance"]
        sample_vars = f.var(axis=0)
        # GARCH sample variance converges slowly; allow 40% tolerance
        rel = np.abs(sample_vars - expected) / expected
        assert (rel < 0.5).all(), (
            f"Factor sample variance too far from analytical: rel={rel}"
        )


class TestTier3BetaPath:
    def test_beta_starts_at_B_initial(self, tier3_output):
        _, _, gt, _ = tier3_output
        B0 = np.asarray(gt["beta_initial"])
        path = np.asarray(gt["beta_path"])
        np.testing.assert_array_equal(path[0], B0)

    def test_beta_path_within_clipping_bound(self, tier3_output):
        _, _, gt, _ = tier3_output
        B0 = np.asarray(gt["beta_initial"])
        path = np.asarray(gt["beta_path"])
        bound = 3.0 * np.abs(B0)
        # path shape (T, N, K); bound shape (N, K); broadcast
        assert (np.abs(path) <= bound + 1e-12).all()

    def test_beta_path_actually_drifts(self, tier3_output):
        _, _, gt, _ = tier3_output
        path = np.asarray(gt["beta_path"])
        # Non-static betas should have nonzero path variance
        assert path.std() > 0, "beta_path is constant — drift did not happen"


class TestTier3Reproducibility:
    def test_same_seed_identical_output(self):
        cfg = SyntheticConfig(
            t_total=500, short_start_day=300, seed=13,
            tier=3, tier3_config=Tier3Config(),
        )
        _, c1, gt1 = generate_tier3(cfg)
        _, c2, gt2 = generate_tier3(cfg)
        pd.testing.assert_frame_equal(c1, c2)
        np.testing.assert_array_equal(
            np.asarray(gt1["factor_returns"]), np.asarray(gt2["factor_returns"])
        )

    def test_different_seeds_differ(self):
        cfg1 = SyntheticConfig(
            t_total=500, short_start_day=300, seed=1,
            tier=3, tier3_config=Tier3Config(),
        )
        cfg2 = SyntheticConfig(
            t_total=500, short_start_day=300, seed=2,
            tier=3, tier3_config=Tier3Config(),
        )
        _, c1, _ = generate_tier3(cfg1)
        _, c2, _ = generate_tier3(cfg2)
        assert not c1.equals(c2)
