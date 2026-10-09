"""Tests for backcast.models.regime_hmm.

Includes a Tier 2 end-to-end test showing that regime-conditional imputation
outperforms unconditional EM on regime-switching data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset, load_backcast_dataset
from backcast.imputation.single_impute import impute_missing_values, single_impute
from backcast.models.em_stambaugh import em_stambaugh
import backcast.models.regime_hmm as regime_hmm_mod
from backcast.exceptions import BackcastConvergenceError
from backcast.models.regime_hmm import (
    _fit_hmm_candidate,
    compute_regime_params,
    fit_and_select_hmm,
    fit_regime_hmm,
    regime_conditional_impute,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER2_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "returns.csv"
TIER2_COMPLETE = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "returns_complete.csv"
TIER2_GT = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Synthetic regime-switching data generator (for unit tests)
# ---------------------------------------------------------------------------

def _simulate_2regime(T=3000, N=3, seed=0):
    rng = np.random.default_rng(seed)
    mu_calm = np.array([0.0003, 0.0001, 0.0002])[:N]
    mu_crisis = -mu_calm
    sigma_calm = np.eye(N) * (0.01 ** 2)
    sigma_crisis = np.eye(N) * (0.03 ** 2)
    P = np.array([[0.98, 0.02], [0.08, 0.92]])

    X = np.empty((T, N))
    labels = np.empty(T, dtype=np.int64)
    s = 0
    for t in range(T):
        labels[t] = s
        if s == 0:
            X[t] = rng.multivariate_normal(mu_calm, sigma_calm)
        else:
            X[t] = rng.multivariate_normal(mu_crisis, sigma_crisis)
        u = rng.random()
        s = 0 if u < P[s, 0] else 1
    return X, labels, P, (mu_calm, mu_crisis), (sigma_calm, sigma_crisis)


# ---------------------------------------------------------------------------
# HMM unit tests
# ---------------------------------------------------------------------------

class TestHMMFit:
    def test_recover_two_regime_data(self):
        X, true_labels, _, _, _ = _simulate_2regime(T=3000, seed=0)
        df = pd.DataFrame(X, columns=list("ABC"))
        res = fit_regime_hmm(df, n_regimes=2, seed=0)
        assert res.converged
        assert res.n_regimes == 2
        # Viterbi should match the true regime labels on the vast majority of rows
        # (allowing for random label-switching — already canonicalised by vol).
        acc = (res.regime_labels == true_labels).mean()
        assert acc > 0.90, f"regime label accuracy {acc:.3f} <= 0.90"

    def test_transition_matrix_close_to_true(self):
        X, _, P_true, _, _ = _simulate_2regime(T=4000, seed=1)
        df = pd.DataFrame(X, columns=list("ABC"))
        res = fit_regime_hmm(df, n_regimes=2, seed=0)
        # Compare transition diagonals
        assert abs(res.transition_matrix[0, 0] - P_true[0, 0]) < 0.05
        assert abs(res.transition_matrix[1, 1] - P_true[1, 1]) < 0.05

    def test_canonical_ordering(self):
        """Regime 0 should be the lower-volatility regime after fit."""
        X, _, _, _, (sigma_calm, sigma_crisis) = _simulate_2regime(T=3000, seed=2)
        df = pd.DataFrame(X, columns=list("ABC"))
        res = fit_regime_hmm(df, n_regimes=2, seed=0)
        tr0 = np.trace(res.covariances[0])
        tr1 = np.trace(res.covariances[1])
        assert tr0 < tr1

    def test_loglikelihood_non_decreasing(self):
        X, _, _, _, _ = _simulate_2regime(T=2000, seed=3)
        df = pd.DataFrame(X, columns=list("ABC"))
        res = fit_regime_hmm(df, n_regimes=2, tolerance=1e-6, seed=0)
        diffs = np.diff(res.ll_trace)
        # Baum-Welch is monotone up to floating-point noise
        assert (diffs > -1e-6).all()


class TestModelSelection:
    def test_selects_two_regimes_on_two_regime_data(self):
        X, _, _, _, _ = _simulate_2regime(T=3000, seed=5)
        df = pd.DataFrame(X, columns=list("ABC"))
        sel = fit_and_select_hmm(df, n_regimes_candidates=(2, 3), criterion="bic", seed=0)
        assert sel.best_n_regimes == 2

    def test_aic_criterion(self):
        X, _, _, _, _ = _simulate_2regime(T=2000, seed=7)
        df = pd.DataFrame(X, columns=list("ABC"))
        sel = fit_and_select_hmm(df, n_regimes_candidates=(2, 3), criterion="aic", seed=0)
        assert sel.criterion == "aic"
        assert sel.best_n_regimes in (2, 3)


def _assert_finite_model(res) -> None:
    for arr in (res.means, res.covariances, res.transition_matrix, res.initial_probs):
        assert np.isfinite(arr).all()
    assert np.isfinite(res.log_likelihood)


class TestDegenerateRejection:
    """Part 1 of the regime-NaN fix: degenerate K is rejected, never returned."""

    def test_rejects_overparameterised_k(self):
        """Data supporting only 2 regimes, candidates [2..5] → K≤2, finite."""
        X, _, _, _, _ = _simulate_2regime(T=1500, seed=11)
        df = pd.DataFrame(X, columns=list("ABC"))
        sel = fit_and_select_hmm(
            df, n_regimes_candidates=(2, 3, 4, 5), criterion="bic",
            n_short=10, seed=0,
        )
        assert sel.best_n_regimes <= 2
        _assert_finite_model(sel.best)
        assert sel.best_n_regimes in sel.surviving_candidates
        for k, score in sel.scores.items():
            if k not in sel.surviving_candidates:
                assert score == float("inf")
        assert (sel.state_occupancy >= 11).all()

    def test_nan_fit_is_rejected(self, monkeypatch):
        X, _, _, _, _ = _simulate_2regime(T=800, seed=12)
        real_fit = regime_hmm_mod.fit_regime_hmm

        def nan_fit(*args, **kwargs):
            res = real_fit(*args, **kwargs)
            res.means = res.means.copy()
            res.means[0, 0] = np.nan
            return res

        monkeypatch.setattr(regime_hmm_mod, "fit_regime_hmm", nan_fit)
        model, bic = _fit_hmm_candidate(X, 2, 1, 1e-3, 200, np.random.default_rng(0))
        assert model is None and bic == float("inf")

    def test_inf_covariance_is_rejected(self, monkeypatch):
        X, _, _, _, _ = _simulate_2regime(T=800, seed=12)
        real_fit = regime_hmm_mod.fit_regime_hmm

        def inf_fit(*args, **kwargs):
            res = real_fit(*args, **kwargs)
            res.covariances = res.covariances.copy()
            res.covariances[1, 0, 0] = np.inf
            return res

        monkeypatch.setattr(regime_hmm_mod, "fit_regime_hmm", inf_fit)
        model, bic = _fit_hmm_candidate(X, 2, 1, 1e-3, 200, np.random.default_rng(0))
        assert model is None and bic == float("inf")

    def test_raising_fit_is_rejected(self, monkeypatch):
        def boom(*args, **kwargs):
            raise np.linalg.LinAlgError("Matrix is not positive definite")

        monkeypatch.setattr(regime_hmm_mod, "fit_regime_hmm", boom)
        X = np.random.default_rng(0).standard_normal((200, 2))
        model, bic = _fit_hmm_candidate(X, 2, 1, 1e-3, 50, np.random.default_rng(0))
        assert model is None and bic == float("inf")

    def test_non_converged_fit_is_rejected(self):
        X, _, _, _, _ = _simulate_2regime(T=1000, seed=13)
        model, bic = _fit_hmm_candidate(
            X, 2, 1, 1e-3, 2, np.random.default_rng(0), tolerance=1e-12,
        )
        assert model is None and bic == float("inf")

    def test_non_converged_kept_when_toggle_off(self, caplog):
        X, _, _, _, _ = _simulate_2regime(T=1000, seed=13)
        with caplog.at_level("WARNING", logger="backcast.models.regime_hmm"):
            model, bic = _fit_hmm_candidate(
                X, 2, 1, 1e-3, 2, np.random.default_rng(0), tolerance=1e-12,
                reject_nonconverged=False,
            )
        assert model is not None and not model.converged
        assert np.isfinite(bic)
        _assert_finite_model(model)
        assert any("kept for selection" in r.message for r in caplog.records)

    def test_selection_with_toggle_off_does_not_fall_back(self):
        X, _, _, _, _ = _simulate_2regime(T=1000, seed=13)
        strict = fit_and_select_hmm(X, (2, 3), max_iter=2, tolerance=1e-12,
                                    n_short=1, seed=0)
        relaxed = fit_and_select_hmm(X, (2, 3), max_iter=2, tolerance=1e-12,
                                     n_short=1, reject_nonconverged=False, seed=0)
        assert strict.fell_back_to_single_regime
        assert not relaxed.fell_back_to_single_regime
        assert relaxed.surviving_candidates

    def test_fallback_to_single_regime(self, caplog):
        X, _, _, _, _ = _simulate_2regime(T=600, seed=14)
        df = pd.DataFrame(X, columns=list("ABC"))
        # n_short larger than T → every K≥2 state is under-filled
        with caplog.at_level("WARNING", logger="backcast.models.regime_hmm"):
            sel = fit_and_select_hmm(
                df, n_regimes_candidates=(2, 3), n_short=len(df), seed=0,
            )
        assert sel.fell_back_to_single_regime
        assert sel.best_n_regimes == 1
        assert sel.surviving_candidates == []
        assert (sel.best.regime_labels == 0).all()
        _assert_finite_model(sel.best)
        np.testing.assert_allclose(sel.best.means[0], X.mean(axis=0), atol=1e-10)
        assert any("falling back to K=1" in r.message for r in caplog.records)

    def test_fallback_survives_tiny_max_iter(self):
        """Regression: K=1 fallback must not itself be rejected as non-converged."""
        X, _, _, _, _ = _simulate_2regime(T=600, seed=14)
        sel = fit_and_select_hmm(X, (2,), max_iter=1, n_short=1, seed=0)
        assert sel.fell_back_to_single_regime and sel.best_n_regimes == 1
        np.testing.assert_allclose(sel.best.means[0], X.mean(axis=0), atol=1e-10)

    def test_fallback_disabled_raises(self):
        X, _, _, _, _ = _simulate_2regime(T=600, seed=14)
        with pytest.raises(BackcastConvergenceError):
            fit_and_select_hmm(
                X, n_regimes_candidates=(2,), n_short=len(X),
                fallback_to_single_regime=False, seed=0,
            )

    def test_overlap_occupancy_reported(self):
        X, _, _, _, _ = _simulate_2regime(T=2000, seed=15)
        mask = np.zeros(len(X), dtype=bool)
        mask[-700:] = True
        sel = fit_and_select_hmm(
            X, n_regimes_candidates=(2,), n_short=2, overlap_mask=mask, seed=0,
        )
        assert sel.overlap_occupancy is not None
        assert sel.overlap_occupancy.sum() == 700
        assert sel.state_occupancy.sum() == len(X)
        assert len(sel.overlap_occupancy) == sel.best_n_regimes

    def test_min_covar_floor_keeps_regimes_separable(self):
        """The floor is relative to per-asset variance, so it must not wash
        out the calm/crisis split on daily-return-scale data."""
        X, true_labels, _, _, _ = _simulate_2regime(T=3000, seed=0)
        res = fit_regime_hmm(X, n_regimes=2, min_covar=1e-3, seed=0)
        assert (res.regime_labels == true_labels).mean() > 0.90
        ratio = np.trace(res.covariances[1]) / np.trace(res.covariances[0])
        assert ratio > 4.0   # true ratio is 9


class TestRegimeParams:
    def test_regime_params_on_known_data(self):
        X, labels, _, (mu_calm, mu_crisis), _ = _simulate_2regime(T=3000, seed=8)
        df = pd.DataFrame(X, columns=list("ABC"))
        params = compute_regime_params(df, labels)
        # Calm regime mean
        np.testing.assert_allclose(params[0]["mu"], mu_calm, atol=1e-3)

    def test_tiny_regime_is_pooled_not_dropped(self, caplog):
        X, labels, _, _, _ = _simulate_2regime(T=1000, seed=9)
        labels = labels.copy()
        labels[:] = 0
        labels[:3] = 2   # only 3 obs for regime 2 (<= N=3 columns)
        df = pd.DataFrame(X, columns=list("ABC"))
        with caplog.at_level("WARNING"):
            params = compute_regime_params(df, labels, min_obs_per_regime=30)
        assert params[2]["source"] == "pooled"
        np.testing.assert_allclose(params[2]["mu"], X.mean(axis=0))
        assert any("DEPRECATION" in r.message for r in caplog.records)

    def test_nan_input_rejected(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        labels = np.zeros(200, dtype=np.int64)
        df = pd.DataFrame(X, columns=["A", "B"])
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="fully observed"):
            compute_regime_params(df, labels)


class TestRegimeConditionalImpute:
    def test_shape_and_nan_filling(self):
        rng = np.random.default_rng(0)
        T = 1500
        n_long, n_short = 3, 2
        data = rng.standard_normal((T, n_long + n_short)) * 0.01
        df = pd.DataFrame(data, columns=[f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)])
        df.iloc[:500, n_long:] = np.nan
        labels = np.zeros(T, dtype=np.int64)
        # Fit regime params on rows where all observed
        overlap = df.iloc[500:]
        params = compute_regime_params(overlap, labels[500:])
        filled = regime_conditional_impute(df, labels, params)
        assert not filled.isna().any().any()


# ---------------------------------------------------------------------------
# Tier 2 end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (TIER2_CSV.exists() and TIER2_COMPLETE.exists() and TIER2_GT.exists()),
    reason="Tier 2 fixtures not generated",
)
class TestTier2End2End:
    @pytest.fixture(scope="class")
    def tier2(self):
        masked = pd.read_csv(TIER2_CSV, index_col="date", parse_dates=True).astype(np.float64)
        complete = pd.read_csv(TIER2_COMPLETE, index_col="date", parse_dates=True).astype(np.float64)
        with open(TIER2_GT) as fh:
            gt = json.load(fh)
        ds = build_backcast_dataset(masked)
        return masked, complete, gt, ds

    def test_hmm_finds_two_regimes(self, tier2):
        _, _, gt, ds = tier2
        long_returns = ds.returns_full[ds.long_assets]
        sel = fit_and_select_hmm(
            long_returns, n_regimes_candidates=(2, 3), criterion="bic", seed=0,
        )
        # Tier 2 default has n_regimes=2 — BIC should prefer K=2
        assert sel.best_n_regimes == 2
        hmm = sel.best
        assert hmm.converged

    def test_regime_conditional_beats_unconditional_em(self, tier2):
        """Key requirement: regime-conditional imputation should yield
        lower RMSE than unconditional EM on regime-switching data."""
        masked, complete, gt, ds = tier2

        # Unconditional EM imputation
        em = em_stambaugh(masked, max_iter=500, tolerance=1e-8,
                          track_loglikelihood=False)
        em_filled = single_impute(ds, em)

        # Regime-conditional imputation
        long_returns = ds.returns_full[ds.long_assets]
        hmm = fit_regime_hmm(long_returns, n_regimes=2, seed=0)
        overlap_labels = hmm.regime_labels[-ds.overlap_length:]
        overlap = ds.overlap_matrix
        regime_params = compute_regime_params(overlap, overlap_labels)
        regime_filled = regime_conditional_impute(
            ds.returns_full, hmm.regime_labels, regime_params,
        )

        # Compute RMSE over the BACKCAST period on short assets
        short_assets = ds.short_assets
        backcast_idx = slice(None, ds.overlap_length * -1 + len(ds.returns_full))
        backcast_dates = ds.returns_full.index[:-ds.overlap_length]

        resid_em = em_filled.loc[backcast_dates, short_assets] - complete.loc[backcast_dates, short_assets]
        resid_rc = regime_filled.loc[backcast_dates, short_assets] - complete.loc[backcast_dates, short_assets]

        rmse_em = float(np.sqrt((resid_em.values ** 2).mean()))
        rmse_rc = float(np.sqrt((resid_rc.values ** 2).mean()))
        # Key requirement: regime-conditional strictly beats unconditional EM.
        # Improvement magnitude is naturally modest for Tier 2 defaults
        # (uniform vol_multiplier leaves β = Σ₂₁·Σ₁₁⁻¹ invariant; only the small
        #  mean shift moves point estimates).  The larger win for
        # regime-conditional is variance/coverage calibration — see below.
        assert rmse_rc < rmse_em, (
            f"regime-conditional RMSE {rmse_rc:.6f} not below unconditional "
            f"{rmse_em:.6f}"
        )

        # Variance calibration: regime-conditional residuals have lower
        # squared-error variance on CRISIS days because it knows the 2.5x
        # vol multiplier.  Compare within-regime MSE.
        overlap_labels = hmm.regime_labels[-ds.overlap_length:]
        backcast_labels = hmm.regime_labels[: -ds.overlap_length]
        crisis_mask = backcast_labels == 1  # regime 1 = higher-vol after canonical sort
        if crisis_mask.sum() > 50:
            mse_em_crisis = (resid_em.values[crisis_mask] ** 2).mean()
            mse_rc_crisis = (resid_rc.values[crisis_mask] ** 2).mean()
            # On crisis days, regime-conditional should be at least as good
            assert mse_rc_crisis <= mse_em_crisis * 1.01
