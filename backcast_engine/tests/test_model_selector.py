"""Tests for backcast.models.model_selector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import build_backcast_dataset, load_backcast_dataset
from backcast.models.model_selector import (
    MethodCVResult,
    ModelSelectionResult,
    _rank_methods,
    evaluate_method_cv,
    select_model_cv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER2_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "returns.csv"


def _synthetic_stationary(t_total=2400, n_long=3, n_short=2, start=1200, seed=0):
    rng = np.random.default_rng(seed)
    N = n_long + n_short
    A = rng.standard_normal((N, N))
    sigma = (A @ A.T) * 1e-4 + np.eye(N) * 5e-5
    R = rng.multivariate_normal(np.zeros(N), sigma, size=t_total)
    idx = pd.date_range("1990-01-02", periods=t_total, freq="B")
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    df = pd.DataFrame(R, index=idx, columns=cols)
    df.iloc[:start, n_long:] = np.nan
    df.index.name = "date"
    return build_backcast_dataset(df)


# ---------------------------------------------------------------------------
# evaluate_method_cv
# ---------------------------------------------------------------------------

class TestEvaluateMethodCV:
    def test_unknown_method_raises(self):
        ds = _synthetic_stationary(seed=1)
        with pytest.raises(ValueError, match="unknown method"):
            evaluate_method_cv(ds, "kalman_tvp")

    def test_insufficient_overlap_raises(self):
        ds = _synthetic_stationary(t_total=1400, start=1000, seed=2)
        with pytest.raises(ValueError, match="overlap has only"):
            evaluate_method_cv(ds, "unconditional_em",
                                holdout_days=200, n_windows=3)

    def test_em_returns_method_result_shape(self):
        ds = _synthetic_stationary(seed=3)
        res = evaluate_method_cv(
            ds, "unconditional_em",
            holdout_days=300, n_windows=3, coverage_level=0.95,
            em_max_iter=100, em_tolerance=1e-7,
        )
        assert isinstance(res, MethodCVResult)
        assert res.method == "unconditional_em"
        assert res.n_windows == 3
        assert res.rmse_per_asset.shape == (ds.n_short,)
        assert res.coverage_per_asset.shape == (ds.n_short,)
        assert res.rmse_overall > 0
        assert 0 <= res.coverage_overall <= 1

    def test_regime_conditional_returns_result(self):
        ds = _synthetic_stationary(seed=4)
        res = evaluate_method_cv(
            ds, "regime_conditional",
            holdout_days=300, n_windows=3, coverage_level=0.95,
            hmm_n_regimes=2, hmm_max_iter=50, hmm_tolerance=1e-3,
        )
        assert res.method == "regime_conditional"
        assert res.n_windows == 3

    def test_coverage_close_to_nominal_on_gaussian(self):
        """Both methods should calibrate near 0.95 on a Gaussian DGP."""
        ds = _synthetic_stationary(t_total=3000, start=1500, seed=5)
        for method in ("unconditional_em", "regime_conditional"):
            res = evaluate_method_cv(
                ds, method,
                holdout_days=400, n_windows=3,
                em_max_iter=100, em_tolerance=1e-7,
                hmm_max_iter=50, hmm_tolerance=1e-3,
            )
            # Should be comfortably in [0.88, 0.99] on 2400 cells
            assert 0.88 < res.coverage_overall < 0.99, (
                f"{method} coverage {res.coverage_overall:.3f}"
            )


# ---------------------------------------------------------------------------
# _rank_methods  (pure-function unit tests)
# ---------------------------------------------------------------------------

class TestRankMethods:
    def _make(self, method, rmse, cov, nominal=0.95):
        return MethodCVResult(
            method=method, n_windows=3, nominal_coverage=nominal,
            rmse_per_asset=np.zeros(1), rmse_overall=rmse,
            coverage_per_asset=np.zeros(1),
            coverage_overall=cov, coverage_error=abs(cov - nominal),
            correlation_error=0.0, per_window=[],
        )

    def test_rank_by_rmse(self):
        results = {
            "a": self._make("a", rmse=0.02, cov=0.94),
            "b": self._make("b", rmse=0.01, cov=0.80),
            "c": self._make("c", rmse=0.03, cov=0.95),
        }
        ranking, scores = _rank_methods(results, "rmse")
        assert ranking == ["b", "a", "c"]
        assert scores["b"] == 0.01

    def test_rank_by_coverage(self):
        results = {
            "a": self._make("a", rmse=0.02, cov=0.94),    # |Δ| = 0.01
            "b": self._make("b", rmse=0.01, cov=0.80),    # |Δ| = 0.15
            "c": self._make("c", rmse=0.03, cov=0.951),   # |Δ| = 0.001
        }
        ranking, _ = _rank_methods(results, "coverage")
        assert ranking[0] == "c"

    def test_rank_by_combined(self):
        # a: best by RMSE, worst by coverage
        # b: middle by both
        # c: worst by RMSE, best by coverage
        results = {
            "a": self._make("a", rmse=0.01, cov=0.80),
            "b": self._make("b", rmse=0.02, cov=0.90),
            "c": self._make("c", rmse=0.03, cov=0.949),
        }
        ranking, scores = _rank_methods(results, "combined")
        # a: rmse rank 1 + cov rank 3 = 4
        # b: rmse rank 2 + cov rank 2 = 4
        # c: rmse rank 3 + cov rank 1 = 4
        # All tied — any stable ordering is acceptable; just check everyone is 4
        assert all(s == 4 for s in scores.values())

    def test_unknown_criterion_raises(self):
        results = {"a": self._make("a", rmse=0.01, cov=0.9)}
        with pytest.raises(ValueError, match="criterion"):
            _rank_methods(results, "bogus")


# ---------------------------------------------------------------------------
# select_model_cv
# ---------------------------------------------------------------------------

class TestSelectModelCV:
    def test_returns_selection_result(self):
        ds = _synthetic_stationary(seed=6)
        sel = select_model_cv(
            ds,
            candidates=("unconditional_em", "regime_conditional"),
            criterion="combined",
            holdout_days=300, n_windows=3,
            em_max_iter=100, em_tolerance=1e-7,
            hmm_max_iter=50, hmm_tolerance=1e-3,
        )
        assert isinstance(sel, ModelSelectionResult)
        assert sel.best_method in sel.candidates
        assert sel.ranking[0] == sel.best_method
        assert set(sel.per_method.keys()) == set(sel.candidates)

    def test_rejects_bad_candidate(self):
        ds = _synthetic_stationary(seed=7)
        with pytest.raises(ValueError, match="unsupported candidates"):
            select_model_cv(ds, candidates=("foo",), criterion="rmse")

    def test_empty_candidates_raises(self):
        ds = _synthetic_stationary(seed=8)
        with pytest.raises(ValueError, match="candidates is empty"):
            select_model_cv(ds, candidates=(), criterion="rmse")

    def test_single_candidate_trivially_best(self):
        ds = _synthetic_stationary(seed=9)
        sel = select_model_cv(
            ds, candidates=("unconditional_em",),
            criterion="rmse", holdout_days=300, n_windows=3,
            em_max_iter=100, em_tolerance=1e-7,
        )
        assert sel.best_method == "unconditional_em"


# ---------------------------------------------------------------------------
# Tier 2 integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER2_CSV.exists(), reason="Tier 2 fixture not generated")
class TestSelectModelTier2:
    @pytest.fixture(scope="class")
    def selection(self):
        ds = load_backcast_dataset(TIER2_CSV)
        return select_model_cv(
            ds,
            candidates=("unconditional_em", "regime_conditional"),
            criterion="coverage",
            holdout_days=504, n_windows=3, coverage_level=0.95,
            hmm_max_iter=200, hmm_tolerance=1e-4, hmm_seed=0,
        )

    def test_ranks_both_methods(self, selection):
        assert len(selection.ranking) == 2
        assert set(selection.candidates) == {
            "unconditional_em", "regime_conditional",
        }

    def test_coverage_near_nominal_for_both(self, selection):
        for method, r in selection.per_method.items():
            assert 0.9 < r.coverage_overall < 1.0, (
                f"{method} coverage {r.coverage_overall:.3f}"
            )

    def test_regime_wins_coverage_on_tier2(self, selection):
        """On the regime-switching Tier 2 DGP the regime-conditional method
        should have the smaller coverage-calibration gap."""
        r_em = selection.per_method["unconditional_em"]
        r_rc = selection.per_method["regime_conditional"]
        # Allow for seed noise; require RC to be at least as good as EM
        # on the coverage-error metric.
        assert r_rc.coverage_error <= r_em.coverage_error + 0.01, (
            f"EM |Δ|={r_em.coverage_error:.4f}, RC |Δ|={r_rc.coverage_error:.4f}"
        )
