"""Tests for backcast.models.kalman_tvp."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset
from backcast.models.kalman_tvp import (
    KalmanAssetResult,
    fit_kalman_all,
    fit_kalman_tvp,
    kalman_impute,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER3_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier3" / "returns.csv"
TIER3_GT = REPO_ROOT / "synthetic_data_generator" / "output" / "tier3" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_factor_model(
    T: int,
    beta_path: np.ndarray,   # (T, K+1): intercept, K slopes
    residual_std: float,
    seed: int = 0,
):
    """Generate factor returns and a short-asset series with the given β path."""
    rng = np.random.default_rng(seed)
    K = beta_path.shape[1] - 1
    factors = rng.standard_normal((T, K)) * 0.01
    design = np.column_stack([np.ones(T), factors])
    y = np.sum(design * beta_path, axis=1) + rng.standard_normal(T) * residual_std
    return factors, y


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_constant_beta_recovered(self):
        """With constant true β, smoothed states should be nearly constant."""
        T = 2000
        K = 3
        true_beta = np.array([0.0005, 0.3, 0.5, -0.1])
        beta_path = np.tile(true_beta, (T, 1))
        factors, y = _simulate_factor_model(T, beta_path, residual_std=0.005, seed=1)

        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        df_long = pd.DataFrame(factors, index=idx, columns=list("ABC"))
        s_short = pd.Series(y, index=idx, name="S")

        res = fit_kalman_tvp(df_long, s_short, state_noise_scale=0.001)
        assert isinstance(res, KalmanAssetResult)
        assert res.smoothed_state.shape == (T, K + 1)
        # Mean of smoothed should be within a few % of true
        mean_beta = res.smoothed_state.mean(axis=0)
        np.testing.assert_allclose(mean_beta, true_beta, atol=0.05)

    def test_filter_vs_smoother(self):
        """Smoother should have smaller total posterior variance than the filter."""
        T = 1500
        K = 2
        true_beta = np.array([0.0, 0.5, 0.2])
        beta_path = np.tile(true_beta, (T, 1))
        factors, y = _simulate_factor_model(T, beta_path, residual_std=0.005, seed=2)
        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        df_long = pd.DataFrame(factors, index=idx, columns=list("AB"))
        s_short = pd.Series(y, index=idx, name="S")

        res_smooth = fit_kalman_tvp(df_long, s_short, use_smoother=True)
        res_filt = fit_kalman_tvp(df_long, s_short, use_smoother=False)
        # Average trace of state cov should be smaller under smoother
        tr_smooth = np.trace(res_smooth.smoothed_state_cov, axis1=1, axis2=2).mean()
        tr_filt = np.trace(res_filt.smoothed_state_cov, axis1=1, axis2=2).mean()
        assert tr_smooth <= tr_filt

    def test_backcast_methods(self):
        T = 500
        beta_path = np.tile(np.array([0.0, 0.5, 0.3]), (T, 1))
        factors, y = _simulate_factor_model(T, beta_path, residual_std=0.005, seed=3)
        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        df_long = pd.DataFrame(factors, index=idx, columns=list("AB"))
        s_short = pd.Series(y, index=idx, name="S")

        r1 = fit_kalman_tvp(df_long, s_short, backcast_beta_method="earliest_smoothed")
        r2 = fit_kalman_tvp(df_long, s_short, backcast_beta_method="mean_first_k",
                             backcast_beta_k=30)
        assert r1.backcast_state.shape == (3,)
        assert r2.backcast_state.shape == (3,)
        # mean_first_k uses more data so should be closer to true β=[0, 0.5, 0.3]
        dist1 = np.linalg.norm(r1.backcast_state - np.array([0.0, 0.5, 0.3]))
        dist2 = np.linalg.norm(r2.backcast_state - np.array([0.0, 0.5, 0.3]))
        # With K=30 this isn't guaranteed but typically holds
        assert dist2 <= dist1 * 2.0  # at least not wildly worse

    def test_rejects_nan_input(self):
        T = 100
        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        df_long = pd.DataFrame(np.zeros((T, 2)), index=idx, columns=list("AB"))
        s_short = pd.Series(np.zeros(T), index=idx, name="S")
        s_short.iloc[5] = np.nan
        with pytest.raises(ValueError, match="fully-observed"):
            fit_kalman_tvp(df_long, s_short)


class TestKalmanMulti:
    def test_fit_all_assets(self):
        rng = np.random.default_rng(0)
        T = 1000
        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        df = pd.DataFrame(
            rng.standard_normal((T, 5)) * 0.01,
            index=idx, columns=["L1", "L2", "L3", "S1", "S2"],
        )
        multi = fit_kalman_all(
            df, long_assets=["L1", "L2", "L3"], short_assets=["S1", "S2"],
        )
        assert set(multi.per_asset.keys()) == {"S1", "S2"}
        # Backcast matrix: row per short asset, col per state
        assert multi.backcast_matrix.shape == (2, 4)
        assert list(multi.backcast_matrix.columns) == ["intercept", "L1", "L2", "L3"]


class TestKalmanImpute:
    def test_impute_fills_missing_only(self):
        rng = np.random.default_rng(1)
        T = 1500
        idx = pd.date_range("1990-01-02", periods=T, freq="B")
        long_names = ["L1", "L2"]
        short_names = ["S1"]
        df = pd.DataFrame(rng.standard_normal((T, 3)) * 0.01,
                           index=idx, columns=long_names + short_names)
        df.iloc[:500, 2] = np.nan
        ds = build_backcast_dataset(df)
        overlap = ds.overlap_matrix
        multi = fit_kalman_all(overlap, long_names, short_names)
        filled = kalman_impute(df, multi)
        assert not filled.isna().any().any()
        # Observed rows unchanged
        observed = df.iloc[500:]
        pd.testing.assert_frame_equal(filled.iloc[500:], observed)


# ---------------------------------------------------------------------------
# Tier 3 tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER3_CSV.exists(), reason="Tier 3 fixtures not generated")
class TestTier3Kalman:
    @pytest.fixture(scope="class")
    def fitted(self):
        masked = pd.read_csv(TIER3_CSV, index_col="date", parse_dates=True).astype(np.float64)
        with open(TIER3_GT) as fh:
            gt = json.load(fh)
        ds = build_backcast_dataset(masked)
        multi = fit_kalman_all(
            ds.overlap_matrix, ds.long_assets, ds.short_assets,
            state_noise_scale=0.01, use_smoother=True,
        )
        return masked, gt, ds, multi

    def test_per_asset_results_present(self, fitted):
        _, _, ds, multi = fitted
        for name in ds.short_assets:
            assert name in multi.per_asset
            res = multi.per_asset[name]
            assert res.smoothed_state.shape[0] == ds.overlap_length

    def test_backcast_matrix_shape(self, fitted):
        _, _, ds, multi = fitted
        n_state = 1 + len(ds.long_assets)   # intercept + K
        assert multi.backcast_matrix.shape == (len(ds.short_assets), n_state)

    def test_impute_completes_matrix(self, fitted):
        masked, _, ds, multi = fitted
        filled = kalman_impute(masked, multi)
        assert not filled.isna().any().any()

    def test_backcast_state_is_finite(self, fitted):
        _, _, ds, multi = fitted
        for name in ds.short_assets:
            res = multi.per_asset[name]
            assert np.isfinite(res.backcast_state).all()
            assert np.isfinite(res.residual_variance)
            assert res.residual_variance > 0
