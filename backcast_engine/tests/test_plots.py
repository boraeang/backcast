"""Smoke tests for backcast.visualization.plots — every plot should return a
matplotlib Figure without raising."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from backcast.data.loader import build_backcast_dataset
from backcast.downstream.backtest import run_backtest
from backcast.downstream.covariance import combined_covariance
from backcast.downstream.uncertainty import ellipsoidal_uncertainty
from backcast.imputation.multiple_impute import multiple_impute
from backcast.models.em_stambaugh import em_stambaugh
from backcast.models.kalman_tvp import fit_kalman_all
from backcast.models.regime_hmm import fit_regime_hmm
from backcast.validation.holdout import run_holdout_validation
from backcast.visualization import plots as P


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_bundle():
    rng = np.random.default_rng(0)
    T, n_long, n_short = 1800, 3, 2
    N = n_long + n_short
    A = rng.standard_normal((N, N))
    sigma = (A @ A.T) * 1e-4 + np.eye(N) * 5e-5
    R = rng.multivariate_normal(np.zeros(N), sigma, size=T)
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    idx = pd.date_range("1990-01-02", periods=T, freq="B")
    df = pd.DataFrame(R, index=idx, columns=cols)
    df.iloc[:900, n_long:] = np.nan
    df.index.name = "date"
    ds = build_backcast_dataset(df)
    em = em_stambaugh(df, max_iter=200, tolerance=1e-8, track_loglikelihood=True)
    kalman = fit_kalman_all(ds.overlap_matrix, ds.long_assets, ds.short_assets)
    hmm = fit_regime_hmm(ds.returns_full[ds.long_assets], n_regimes=2, seed=0)
    mi = multiple_impute(ds, em, n_imputations=15, seed=0)
    holdout = run_holdout_validation(ds, holdout_days=200, n_windows=3)
    combined = combined_covariance(mi.imputations)
    ellipse = ellipsoidal_uncertainty(mi.imputations, confidence=0.95)
    bt = run_backtest(mi.imputations, strategy="equal_weight",
                      lookback=30, rebalance_freq=20)
    return {
        "ds": ds, "df": df, "em": em, "kalman": kalman, "hmm": hmm,
        "mi": mi, "holdout": holdout, "combined": combined, "ellipse": ellipse,
        "bt": bt,
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plot_missingness(self, tiny_bundle):
        fig = P.plot_missingness(tiny_bundle["df"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_em_convergence(self, tiny_bundle):
        fig = P.plot_em_convergence(tiny_bundle["em"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_kalman_betas(self, tiny_bundle):
        fig = P.plot_kalman_betas(tiny_bundle["kalman"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_regime_timeline(self, tiny_bundle):
        hmm = tiny_bundle["hmm"]
        fig = P.plot_regime_timeline(
            hmm.regime_labels, tiny_bundle["df"].index,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_backcast_fan(self, tiny_bundle):
        fig = P.plot_backcast_fan(
            tiny_bundle["mi"], asset=tiny_bundle["ds"].short_assets[0],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_correlation_comparison(self, tiny_bundle):
        corr_a = np.corrcoef(tiny_bundle["ds"].overlap_matrix.to_numpy(), rowvar=False)
        corr_b = tiny_bundle["combined"].correlation
        fig = P.plot_correlation_comparison(corr_a, corr_b,
                                             labels=list(tiny_bundle["df"].columns))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_holdout_scatter(self, tiny_bundle):
        fig = P.plot_holdout_scatter(tiny_bundle["holdout"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_pit_histogram(self):
        rng = np.random.default_rng(0)
        pit = rng.uniform(0, 1, 1000)
        fig = P.plot_pit_histogram(pit)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_eigenvalue_spectrum(self, tiny_bundle):
        fig = P.plot_eigenvalue_spectrum(
            tiny_bundle["combined"].covariance, T=len(tiny_bundle["df"]),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_uncertainty_ellipses(self, tiny_bundle):
        fig = P.plot_uncertainty_ellipses(tiny_bundle["ellipse"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_backtest_fan(self, tiny_bundle):
        fig = P.plot_backtest_fan(tiny_bundle["bt"])
        assert isinstance(fig, Figure)
        plt.close(fig)
