"""Tests for backcast.validation.holdout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset, load_backcast_dataset
from backcast.validation.holdout import HoldoutReport, run_holdout_validation


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns.csv"


def _synthetic_dataset(t_total=3000, n_long=3, n_short=2, start=1500, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.zeros(n_long + n_short)
    A = rng.standard_normal((n_long + n_short, n_long + n_short))
    sigma = (A @ A.T) * 1e-4 + np.eye(n_long + n_short) * 1e-4
    R = rng.multivariate_normal(mu, sigma, size=t_total)
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    idx = pd.date_range("1990-01-02", periods=t_total, freq="B")
    df = pd.DataFrame(R, index=idx, columns=cols)
    df.iloc[:start, n_long:] = np.nan
    df.index.name = "date"
    return build_backcast_dataset(df)


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------

class TestHoldoutBasics:
    def test_returns_report_with_expected_windows(self):
        ds = _synthetic_dataset(t_total=3000, start=1500, seed=0)
        # overlap has 1500 rows; 3 windows of 400 = 1200 fits
        report = run_holdout_validation(
            ds, holdout_days=400, n_windows=3, coverage_level=0.95,
        )
        assert isinstance(report, HoldoutReport)
        assert len(report.windows) == 3
        assert report.config["holdout_days"] == 400
        for w in report.windows:
            assert w.n_rows == 400
            assert w.em_converged

    def test_windows_are_non_overlapping(self):
        ds = _synthetic_dataset(seed=1)
        report = run_holdout_validation(ds, holdout_days=300, n_windows=3)
        ends = [w.end_date for w in report.windows]
        starts_next = [w.start_date for w in report.windows[1:]]
        for prev_end, next_start in zip(ends[:-1], starts_next):
            assert next_start > prev_end

    def test_predicted_shape_matches_actual(self):
        ds = _synthetic_dataset(seed=2)
        report = run_holdout_validation(ds, holdout_days=300, n_windows=2)
        for w in report.windows:
            assert w.predicted.shape == w.actual.shape
            assert list(w.predicted.columns) == list(w.actual.columns)

    def test_prediction_interval_covers_roughly_95_percent(self):
        """Well-specified model + Gaussian data → coverage ≈ nominal."""
        ds = _synthetic_dataset(t_total=5000, start=2500, seed=3)
        report = run_holdout_validation(ds, holdout_days=500, n_windows=3,
                                         coverage_level=0.95)
        assert 0.90 < report.overall_coverage < 0.99

    def test_residual_diagnostics_present(self):
        ds = _synthetic_dataset(seed=4)
        report = run_holdout_validation(ds, holdout_days=300, n_windows=2)
        assert "jb_pvalue" in report.residual_diagnostics.columns
        assert "lb_pvalue" in report.residual_diagnostics.columns
        assert len(report.residual_diagnostics) == ds.n_short

    def test_rejects_insufficient_overlap(self):
        ds = _synthetic_dataset(t_total=2500, start=2000, seed=5)
        # overlap has 500 rows; asking for 3 × 200 = 600 is too much
        with pytest.raises(ValueError, match="Overlap has only"):
            run_holdout_validation(ds, holdout_days=200, n_windows=3)

    def test_rejects_no_short_assets(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame(rng.standard_normal((1000, 3)) * 0.01,
                           columns=list("ABC"),
                           index=pd.date_range("1990-01-02", periods=1000, freq="B"))
        df.index.name = "date"
        ds = build_backcast_dataset(df)
        with pytest.raises(ValueError, match="no short-history assets"):
            run_holdout_validation(ds)


# ---------------------------------------------------------------------------
# Tier 1 end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER1_CSV.exists(), reason="Tier 1 fixtures not generated")
class TestHoldoutTier1:
    @pytest.fixture(scope="class")
    def report(self):
        ds = load_backcast_dataset(TIER1_CSV)
        return run_holdout_validation(
            ds, holdout_days=504, n_windows=3, coverage_level=0.95,
        )

    def test_three_windows(self, report):
        assert len(report.windows) == 3

    def test_em_converged_every_window(self, report):
        for w in report.windows:
            assert w.em_converged

    def test_overall_coverage_close_to_nominal(self, report):
        assert 0.88 < report.overall_coverage < 0.99

    def test_per_asset_coverage_reasonable(self, report):
        # No single asset should fall far below nominal
        assert (report.per_asset_mean["coverage"] > 0.85).all()

    def test_vol_ratio_close_to_one(self, report):
        ratios = report.per_asset_mean["vol_ratio"]
        # Predicted vol uses cond_mean of Gaussian — naturally smaller than
        # actual by a factor of sqrt(1 - R²).  Ratios should still be > 0.3.
        assert (ratios > 0.3).all() and (ratios < 1.5).all()

    def test_residual_diagnostics_shape(self, report):
        assert len(report.residual_diagnostics) == 3   # CRYPTO_1, CRYPTO_2, ALT_1
