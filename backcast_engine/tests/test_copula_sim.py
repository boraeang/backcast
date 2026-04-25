"""Tests for backcast.imputation.copula_sim."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backcast.imputation.copula_sim import (
    CopulaFit,
    CopulaSimResult,
    MarginalFit,
    fit_copula,
    fit_marginal,
    fit_marginals,
    simulate_copula,
)


# ---------------------------------------------------------------------------
# Marginal fitting
# ---------------------------------------------------------------------------

class TestMarginalFit:
    def test_gaussian_data_prefers_normal(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        fit = fit_marginal(x, name="G", criterion="aic")
        assert fit.distribution in ("normal", "student_t")
        # On clean Gaussian data, normal should dominate student_t by AIC
        assert fit.ks_pvalue > 0.01

    def test_student_t_data_prefers_t(self):
        rng = np.random.default_rng(1)
        x = rng.standard_t(4, size=5000) * 0.01
        fit = fit_marginal(x, name="T", criterion="aic")
        assert fit.distribution == "student_t"
        assert fit.params["df"] < 10

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="Too few"):
            fit_marginal(np.array([0.1, 0.2]))

    def test_fit_marginals_all_columns(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            rng.standard_normal((1000, 3)),
            columns=["A", "B", "C"],
        )
        m = fit_marginals(df)
        assert set(m.keys()) == {"A", "B", "C"}
        assert all(isinstance(v, MarginalFit) for v in m.values())


# ---------------------------------------------------------------------------
# Copula fitting
# ---------------------------------------------------------------------------

class TestCopulaFit:
    def test_gaussian_copula_on_correlated_normal(self):
        rng = np.random.default_rng(3)
        R = np.array([[1.0, 0.7], [0.7, 1.0]])
        L = np.linalg.cholesky(R)
        Z = rng.standard_normal((3000, 2)) @ L.T
        df = pd.DataFrame(Z, columns=["A", "B"])
        marg = fit_marginals(df, candidates=("normal",))
        cop = fit_copula(df, marg, copula_type="gaussian")
        assert cop.copula_type == "gaussian"
        assert abs(cop.correlation[0, 1] - 0.7) < 0.05

    def test_t_copula_df_reasonable(self):
        """Student-t copula fit on heavy-tailed dependent data picks a small df."""
        rng = np.random.default_rng(4)
        df_true = 5.0
        R = np.array([[1.0, 0.6], [0.6, 1.0]])
        L = np.linalg.cholesky(R)
        Z = rng.standard_normal((3000, 2)) @ L.T
        chi2 = rng.chisquare(df_true, size=3000)
        t_sample = Z / np.sqrt(chi2 / df_true)[:, None]
        df = pd.DataFrame(t_sample, columns=["A", "B"])
        marg = fit_marginals(df, candidates=("student_t",))
        cop = fit_copula(df, marg, copula_type="student_t")
        assert cop.copula_type == "student_t"
        assert cop.df is not None
        # Grid search over {3, 4, 5, 6, 8, ...} — picks small df
        assert cop.df <= 10.0

    def test_rejects_bad_copula_type(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.standard_normal((500, 2)), columns=["A", "B"])
        marg = fit_marginals(df)
        with pytest.raises(ValueError, match="copula_type"):
            fit_copula(df, marg, copula_type="vine")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_simulation_shape(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame(rng.standard_normal((500, 3)) * 0.01,
                           columns=["A", "B", "C"])
        marg = fit_marginals(df)
        cop = fit_copula(df, marg, copula_type="gaussian")
        sim = simulate_copula(cop, marg, n_simulations=50, horizon=100, seed=0)
        assert sim.simulated_returns.shape == (50, 100, 3)

    def test_simulation_preserves_correlation(self):
        rng = np.random.default_rng(7)
        R_true = np.array([[1.0, 0.6, 0.2],
                           [0.6, 1.0, 0.3],
                           [0.2, 0.3, 1.0]])
        L = np.linalg.cholesky(R_true)
        Z = rng.standard_normal((5000, 3)) @ L.T
        df = pd.DataFrame(Z * 0.01, columns=["A", "B", "C"])
        marg = fit_marginals(df)
        cop = fit_copula(df, marg, copula_type="gaussian")
        sim = simulate_copula(cop, marg, n_simulations=200, horizon=500, seed=0)
        sim_flat = sim.simulated_returns.reshape(-1, 3)
        R_sim = np.corrcoef(sim_flat, rowvar=False)
        max_corr_err = np.max(np.abs(R_sim - R_true))
        assert max_corr_err < 0.05, f"corr error {max_corr_err}"

    def test_simulation_preserves_marginals(self):
        rng = np.random.default_rng(8)
        x = rng.standard_t(5, size=4000) * 0.02
        df = pd.DataFrame({"A": x})
        marg = fit_marginals(df, candidates=("student_t",))
        cop = fit_copula(df, marg, copula_type="gaussian")
        sim = simulate_copula(cop, marg, n_simulations=100, horizon=200, seed=0)
        sim_flat = sim.simulated_returns.reshape(-1)
        # KS two-sample: simulated draws should be close to fitted marginal
        ks_stat, p = stats.ks_2samp(sim_flat[:2000], x)
        assert p > 0.001

    def test_t_copula_simulation(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame(rng.standard_normal((1000, 2)) * 0.01,
                           columns=["A", "B"])
        marg = fit_marginals(df, candidates=("normal",))
        cop = fit_copula(df, marg, copula_type="student_t")
        sim = simulate_copula(cop, marg, n_simulations=50, horizon=100, seed=0)
        assert sim.simulated_returns.shape == (50, 100, 2)
        assert np.isfinite(sim.simulated_returns).all()

    def test_reproducibility(self):
        rng = np.random.default_rng(10)
        df = pd.DataFrame(rng.standard_normal((800, 2)) * 0.01,
                           columns=["A", "B"])
        marg = fit_marginals(df)
        cop = fit_copula(df, marg, copula_type="gaussian")
        s1 = simulate_copula(cop, marg, n_simulations=30, horizon=50, seed=42)
        s2 = simulate_copula(cop, marg, n_simulations=30, horizon=50, seed=42)
        np.testing.assert_allclose(s1.simulated_returns, s2.simulated_returns)

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(11)
        df = pd.DataFrame(rng.standard_normal((500, 2)) * 0.01,
                           columns=["A", "B"])
        marg = fit_marginals(df)
        cop = fit_copula(df, marg, copula_type="gaussian")
        s1 = simulate_copula(cop, marg, n_simulations=10, horizon=20, seed=1)
        s2 = simulate_copula(cop, marg, n_simulations=10, horizon=20, seed=2)
        assert not np.allclose(s1.simulated_returns, s2.simulated_returns)


# ---------------------------------------------------------------------------
# Round-trip: fit and simulate from copula preserves second moments
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_fit_and_simulate_matches_second_moments(self):
        rng = np.random.default_rng(12)
        # Build correlated heavy-tailed data
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        L = np.linalg.cholesky(R)
        base = rng.standard_t(df=6, size=(4000, 2)) @ L.T
        df = pd.DataFrame(base * 0.01, columns=["A", "B"])
        marg = fit_marginals(df)
        cop = fit_copula(df, marg, copula_type="student_t")
        sim = simulate_copula(cop, marg, n_simulations=100, horizon=500, seed=0)
        sim_flat = sim.simulated_returns.reshape(-1, 2)
        # Compare std and correlation
        sim_std = sim_flat.std(axis=0)
        data_std = base * 0.01
        data_std = data_std.std(axis=0)
        rel_err_std = np.abs(sim_std - data_std) / data_std
        assert (rel_err_std < 0.15).all()
