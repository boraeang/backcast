"""Tests for backcast.imputation.single_impute."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset
from backcast.imputation.single_impute import single_impute
from backcast.models.em_stambaugh import em_stambaugh


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns.csv"
TIER1_COMPLETE = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns_complete.csv"


def _mv_normal(n, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, sigma, size=n)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSingleImpute:
    def _make_dataset(self, t_total=3000, seed=0, start=1500):
        mu = np.array([0.0002, -0.0001, 0.0003])
        A = np.array([[1.0, 0.4, 0.2],
                      [0.4, 0.9, 0.3],
                      [0.2, 0.3, 1.1]])
        sigma = (A @ A.T) * 1e-4
        R = _mv_normal(t_total, mu, sigma, seed=seed)
        dates = pd.date_range("1990-01-02", periods=t_total, freq="B")
        df = pd.DataFrame(R, index=dates, columns=["L1", "L2", "S1"])
        complete = df.copy()
        df.iloc[:start, 2] = np.nan
        df.index.name = "date"
        return df, complete, mu, sigma

    def test_produces_no_nan(self):
        df, _, _, _ = self._make_dataset()
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        filled = single_impute(ds, em)
        assert not filled.isna().any().any()
        assert filled.shape == df.shape

    def test_long_assets_unchanged(self):
        df, _, _, _ = self._make_dataset()
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        filled = single_impute(ds, em)
        for col in ds.long_assets:
            pd.testing.assert_series_equal(filled[col], df[col])

    def test_overlap_rows_unchanged(self):
        df, _, _, _ = self._make_dataset()
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        filled = single_impute(ds, em)
        for col in ds.short_assets:
            start = ds.short_start_indices[col]
            pd.testing.assert_series_equal(
                filled[col].iloc[start:], df[col].iloc[start:]
            )

    def test_imputed_matches_conditional_mean(self):
        """For row t in backcast, filled value == α + β · r_long_t."""
        df, _, _, _ = self._make_dataset()
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        filled = single_impute(ds, em)
        cp = em.conditional_params
        # For this simple case obs_cols == long, mis_cols == short
        obs_cols = cp.observed_cols
        mis_cols = cp.missing_cols
        # First backcast row
        t = 0
        r_long = df.iloc[t, obs_cols].to_numpy()
        expected = cp.alpha + cp.beta @ r_long
        got = filled.iloc[t, mis_cols].to_numpy()
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_staggered_missingness_imputes_all(self):
        mu = np.zeros(4)
        A = np.eye(4) + 0.2
        sigma = (A @ A.T) * 1e-4
        R = _mv_normal(2500, mu, sigma, seed=42)
        dates = pd.date_range("1990-01-02", periods=2500, freq="B")
        df = pd.DataFrame(R, index=dates, columns=["L1", "L2", "S1", "S2"])
        df.iloc[:800, 2] = np.nan
        df.iloc[:1400, 3] = np.nan
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=300, tolerance=1e-8, track_loglikelihood=False)
        filled = single_impute(ds, em)
        assert not filled.isna().any().any()

    def test_mismatched_columns_raises(self):
        df, _, _, _ = self._make_dataset()
        ds = build_backcast_dataset(df)
        em = em_stambaugh(df, max_iter=5, track_loglikelihood=False)
        em.asset_order = ["wrong", "names", "here"]
        with pytest.raises(ValueError, match="asset_order"):
            single_impute(ds, em)


# ---------------------------------------------------------------------------
# Calibration test against the Tier 1 synthetic data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER1_CSV.exists() or not TIER1_COMPLETE.exists(),
                     reason="Tier 1 fixtures not generated")
class TestTier1Imputation:
    @pytest.fixture(scope="class")
    def filled_vs_true(self):
        masked = pd.read_csv(TIER1_CSV, index_col="date", parse_dates=True).astype(np.float64)
        complete = pd.read_csv(TIER1_COMPLETE, index_col="date", parse_dates=True).astype(np.float64)
        ds = build_backcast_dataset(masked)
        em = em_stambaugh(masked, max_iter=500, tolerance=1e-8,
                          track_loglikelihood=False)
        filled = single_impute(ds, em)
        return masked, complete, filled, em, ds

    def test_complete_output(self, filled_vs_true):
        _, _, filled, _, _ = filled_vs_true
        assert not filled.isna().any().any()

    def test_observed_entries_preserved(self, filled_vs_true):
        masked, _, filled, _, _ = filled_vs_true
        # Wherever the masked value is non-NaN, filled should match bit-exactly
        obs_mask = ~masked.isna()
        diff = (filled.values - masked.values)
        assert np.all(np.abs(diff[obs_mask.values]) < 1e-15)

    def test_backcast_period_reasonable_error(self, filled_vs_true):
        """Point estimate can't recover the idio noise, but on average the
        residual variance should be close to the conditional variance —
        the conditional mean is the Bayes-optimal predictor."""
        masked, complete, filled, em, ds = filled_vs_true
        # Use CSV column order (what cond_cov's row order matches) rather
        # than ds.short_assets (alphabetical).
        short_col_idx = em.conditional_params.missing_cols
        short_names = [em.asset_order[i] for i in short_col_idx]

        overlap_start = ds.overlap_start
        backcast = filled.loc[:overlap_start - pd.Timedelta(days=1), short_names]
        true_backcast = complete.loc[backcast.index, short_names]
        resid = (backcast - true_backcast).values

        cond_var_diag = np.diag(em.conditional_params.cond_cov)
        sample_var = resid.var(axis=0, ddof=0)
        ratio = sample_var / cond_var_diag
        assert (ratio > 0.5).all() and (ratio < 1.5).all(), (
            f"Residual/conditional variance ratio out of band "
            f"(cols={short_names}): {ratio}"
        )
