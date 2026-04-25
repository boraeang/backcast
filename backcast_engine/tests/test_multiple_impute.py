"""Tests for backcast.imputation.multiple_impute.

Includes a Tier 1 coverage-calibration test showing that 95 % prediction
intervals from M imputations cover ~95 % of the true complete-data values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset, load_backcast_dataset
from backcast.imputation.multiple_impute import (
    MultipleImputationResult,
    RubinResult,
    apply_rubin,
    combine_estimates,
    multiple_impute,
    multiple_impute_regime,
    prediction_intervals,
)
from backcast.imputation.single_impute import single_impute
from backcast.models.em_stambaugh import em_stambaugh


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns.csv"
TIER1_COMPLETE = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns_complete.csv"


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _synthetic_dataset(t_total=3000, n_long=3, n_short=2, start=1500, seed=0):
    rng = np.random.default_rng(seed)
    N = n_long + n_short
    A = rng.standard_normal((N, N))
    sigma = (A @ A.T) * 1e-4 + np.eye(N) * 5e-5
    mu = np.zeros(N)
    R = rng.multivariate_normal(mu, sigma, size=t_total)
    idx = pd.date_range("1990-01-02", periods=t_total, freq="B")
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    df = pd.DataFrame(R, index=idx, columns=cols)
    complete = df.copy()
    df.iloc[:start, n_long:] = np.nan
    df.index.name = "date"
    return build_backcast_dataset(df), complete


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestMultipleImputeBasics:
    def test_returns_m_imputations(self):
        ds, _ = _synthetic_dataset(seed=1)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=20, seed=0)
        assert isinstance(mi, MultipleImputationResult)
        assert len(mi.imputations) == 20
        assert mi.method == "unconditional_em"

    def test_no_nan_in_output(self):
        ds, _ = _synthetic_dataset(seed=2)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=5, seed=0)
        for df in mi.imputations:
            assert not df.isna().any().any()

    def test_observed_entries_preserved(self):
        ds, _ = _synthetic_dataset(seed=3)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=5, seed=0)
        orig = ds.returns_full.to_numpy()
        mask_obs = ~np.isnan(orig)
        for df in mi.imputations:
            vals = df.to_numpy()
            assert np.allclose(vals[mask_obs], orig[mask_obs])

    def test_imputations_differ(self):
        ds, _ = _synthetic_dataset(seed=4)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=3, seed=0)
        a = mi.imputations[0].to_numpy()
        b = mi.imputations[1].to_numpy()
        # They differ on imputed (originally-NaN) cells
        mask_imp = np.isnan(ds.returns_full.to_numpy())
        assert not np.allclose(a[mask_imp], b[mask_imp])

    def test_mean_of_imputations_matches_single_impute(self):
        """Average of M draws should approach the conditional mean (= single impute)."""
        ds, _ = _synthetic_dataset(t_total=4000, start=2000, seed=5)
        em = em_stambaugh(ds.returns_full, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=200, seed=0)
        avg = sum(df.to_numpy() for df in mi.imputations) / mi.n_imputations
        single = single_impute(ds, em).to_numpy()
        mask = np.isnan(ds.returns_full.to_numpy())
        # Concentrates around single imputation as M grows
        diff = np.abs(avg[mask] - single[mask])
        # With M=200 draws of cond_std ~0.01 and sqrt(M)=14 → noise ~0.01/14 ≈ 7e-4
        assert diff.mean() < 5e-3

    def test_sample_variance_matches_cond_cov(self):
        ds, _ = _synthetic_dataset(t_total=2500, start=1500, seed=6)
        em = em_stambaugh(ds.returns_full, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=500, seed=0)
        # Variance across M draws at a single imputed cell should match cond var
        stack = np.stack([df.to_numpy() for df in mi.imputations], axis=0)  # (M, T, N)
        cond_cov = em.conditional_params.cond_cov
        mis_cols = em.conditional_params.missing_cols
        mask = np.isnan(ds.returns_full.to_numpy())
        # Focus on backcast rows where ALL short assets are missing
        # → pattern conditional cov = cond_cov
        sample_var = stack.var(axis=0)                 # (T, N)
        # Per short asset, average sample variance over the first 100 backcast rows
        for idx_in_cp, col_idx in enumerate(mis_cols):
            diag = cond_cov[idx_in_cp, idx_in_cp]
            sv_col = sample_var[:100, col_idx]
            mean_sv = sv_col[~np.isnan(sv_col)].mean() if np.any(~np.isnan(sv_col)) else 0.0
            # Not every row in the first 100 is fully missing — but the ratio
            # should be close to 1 overall.  Allow generous tolerance.
            if mean_sv > 0:
                assert 0.7 < (mean_sv / diag) < 1.3


# ---------------------------------------------------------------------------
# Rubin's rules unit tests
# ---------------------------------------------------------------------------

class TestRubinRules:
    def test_identical_estimates_no_between(self):
        est = [np.array([1.0, 2.0, 3.0])] * 5
        var = [np.array([0.1, 0.2, 0.3])] * 5
        res = combine_estimates(est, var)
        np.testing.assert_allclose(res.between_variance, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.within_variance, [0.1, 0.2, 0.3])
        # total = within + (1+1/M) * 0 = within
        np.testing.assert_allclose(res.total_variance, [0.1, 0.2, 0.3])

    def test_scalar_variance_added_correctly(self):
        est = [np.array([1.0]), np.array([3.0])]
        var = [np.array([0.5]), np.array([0.5])]
        res = combine_estimates(est, var)
        # Between = ((1-2)^2 + (3-2)^2) / 1 = 2
        np.testing.assert_allclose(res.between_variance, 2.0)
        np.testing.assert_allclose(res.within_variance, 0.5)
        # Total = 0.5 + (1 + 1/2)*2 = 0.5 + 3 = 3.5
        np.testing.assert_allclose(res.total_variance, 3.5)

    def test_without_variances(self):
        est = [np.array([1.0]), np.array([3.0])]
        res = combine_estimates(est)
        assert np.isnan(res.within_variance).all()
        # Total = (1 + 1/M) * B
        np.testing.assert_allclose(res.total_variance, 1.5 * 2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot combine"):
            combine_estimates([])

    def test_apply_rubin_callback(self):
        ds, _ = _synthetic_dataset(seed=7)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=10, seed=0)
        res = apply_rubin(mi.imputations, lambda df: df.mean().values)
        assert isinstance(res, RubinResult)
        # point estimate matches mean of per-imputation means
        direct = np.mean([df.mean().values for df in mi.imputations], axis=0)
        np.testing.assert_allclose(res.point_estimate, direct)


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

class TestPredictionIntervals:
    def test_shape_and_ordering(self):
        ds, _ = _synthetic_dataset(seed=8)
        em = em_stambaugh(ds.returns_full, max_iter=100, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=30, seed=0)
        med, lo, hi = prediction_intervals(mi, confidence=0.95)
        assert med.shape == ds.returns_full.shape
        assert ((lo.values <= med.values) & (med.values <= hi.values)).all()

    def test_coverage_on_gaussian_synthetic(self):
        """Expected ~95% coverage of true complete-data values."""
        ds, complete = _synthetic_dataset(t_total=4000, start=2000, seed=9)
        em = em_stambaugh(ds.returns_full, max_iter=200, tolerance=1e-8, track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=200, seed=0)
        _, lo, hi = prediction_intervals(mi, confidence=0.95)
        # Evaluate ONLY on imputed cells (mask on backcast rows, short assets)
        mask = np.isnan(ds.returns_full.to_numpy())
        covered = ((complete.to_numpy() >= lo.to_numpy()) &
                   (complete.to_numpy() <= hi.to_numpy()) & mask)
        n_cells = mask.sum()
        cov_rate = covered.sum() / n_cells
        assert 0.92 < cov_rate < 0.98, f"coverage {cov_rate:.3f} off from 0.95"


# ---------------------------------------------------------------------------
# Regime-conditional MI (uses HMM module)
# ---------------------------------------------------------------------------

class TestRegimeConditionalMI:
    def test_runs_with_regime_params(self):
        from backcast.models.regime_hmm import compute_regime_params
        ds, _ = _synthetic_dataset(seed=10)
        # Trivial single-regime params
        overlap = ds.overlap_matrix
        labels = np.zeros(len(ds.returns_full), dtype=np.int64)
        overlap_labels = labels[-ds.overlap_length:]
        regime_params = compute_regime_params(overlap, overlap_labels)
        mi = multiple_impute_regime(ds, labels, regime_params,
                                     n_imputations=5, seed=0)
        assert mi.method == "regime_conditional"
        assert len(mi.imputations) == 5
        for df in mi.imputations:
            assert not df.isna().any().any()


# ---------------------------------------------------------------------------
# Tier 1 coverage calibration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (TIER1_CSV.exists() and TIER1_COMPLETE.exists()),
                     reason="Tier 1 fixtures not generated")
class TestTier1Coverage:
    @pytest.fixture(scope="class")
    def fitted(self):
        masked = pd.read_csv(TIER1_CSV, index_col="date", parse_dates=True).astype(np.float64)
        complete = pd.read_csv(TIER1_COMPLETE, index_col="date", parse_dates=True).astype(np.float64)
        ds = build_backcast_dataset(masked)
        em = em_stambaugh(masked, max_iter=500, tolerance=1e-8,
                          track_loglikelihood=False)
        mi = multiple_impute(ds, em, n_imputations=100, seed=0)
        return masked, complete, ds, em, mi

    def test_coverage_close_to_95(self, fitted):
        masked, complete, ds, em, mi = fitted
        _, lo, hi = prediction_intervals(mi, confidence=0.95)
        mask = np.isnan(masked.to_numpy())
        covered = ((complete.to_numpy() >= lo.to_numpy()) &
                   (complete.to_numpy() <= hi.to_numpy()) & mask)
        cov = covered.sum() / mask.sum()
        assert 0.93 < cov < 0.97, f"Tier 1 coverage {cov:.3f} off from 0.95"

    def test_per_asset_coverage_near_nominal(self, fitted):
        masked, complete, ds, em, mi = fitted
        _, lo, hi = prediction_intervals(mi, confidence=0.95)
        for col in ds.short_assets:
            mask_col = masked[col].isna().to_numpy()
            if not mask_col.any():
                continue
            actual = complete[col].to_numpy()[mask_col]
            l = lo[col].to_numpy()[mask_col]
            h = hi[col].to_numpy()[mask_col]
            cov = ((actual >= l) & (actual <= h)).mean()
            assert 0.91 < cov < 0.99, f"{col} coverage {cov:.3f}"
