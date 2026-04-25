"""End-to-end pipeline smoke tests (on synthetic data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.pipeline import BackcastPipeline, FullResults


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER2_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "returns.csv"


# ---------------------------------------------------------------------------
# Synthetic fixture CSV
# ---------------------------------------------------------------------------

def _synthetic_csv(tmp_path, T=1800, n_long=3, n_short=2, start=900, seed=0):
    rng = np.random.default_rng(seed)
    N = n_long + n_short
    A = rng.standard_normal((N, N))
    sigma = (A @ A.T) * 1e-4 + np.eye(N) * 5e-5
    R = rng.multivariate_normal(np.zeros(N), sigma, size=T)
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    idx = pd.date_range("1990-01-02", periods=T, freq="B")
    df = pd.DataFrame(R, index=idx, columns=cols)
    df.iloc[:start, n_long:] = np.nan
    df.index.name = "date"
    p = tmp_path / "returns.csv"
    df.to_csv(p)
    return p


_TINY_CONFIG = {
    "random_seed": 0,
    "data": {"min_overlap_days": 100},
    "em": {"max_iterations": 100, "tolerance": 1e-7, "track_loglikelihood": True},
    "kalman": {"state_noise_scale": 0.01, "use_smoother": True},
    "hmm": {"n_regimes_candidates": [2], "max_iterations": 100, "tolerance": 1e-3},
    "imputation": {"n_imputations": 10, "method": "unconditional_em"},
    "validation": {"holdout_days": 200, "n_windows": 3, "coverage_level": 0.95},
    "downstream": {
        "covariance_shrinkage": True,
        "denoise_eigenvalues": True,
        "uncertainty_confidence": 0.95,
        "backtest_strategies": ["equal_weight", "inverse_volatility"],
        "backtest_lookback": 30,
        "backtest_rebalance_freq": 15,
    },
    "output": {"plot_format": "png", "plot_dpi": 80, "save_imputations": False},
}


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_end_to_end(self, tmp_path):
        csv = _synthetic_csv(tmp_path, seed=1)
        pipe = BackcastPipeline(config_dict=_TINY_CONFIG, log_level="WARNING")
        res = pipe.run(csv)
        assert isinstance(res, FullResults)
        assert res.em_result.converged
        assert res.hmm is not None
        assert res.kalman is not None
        assert res.imputation.n_imputations == 10
        assert len(res.holdout.windows) == 3
        assert res.downstream.covariance_combined.covariance.shape == (5, 5)
        assert res.downstream.ellipsoidal_mu.kappa > 0
        assert "equal_weight" in res.downstream.backtests
        assert "inverse_volatility" in res.downstream.backtests

    def test_export_creates_expected_files(self, tmp_path):
        csv = _synthetic_csv(tmp_path, seed=2)
        pipe = BackcastPipeline(config_dict=_TINY_CONFIG, log_level="WARNING")
        res = pipe.run(csv)
        out_dir = tmp_path / "artefacts"
        paths = pipe.export(res, out_dir)
        # Summary JSON
        assert (out_dir / "summary.json").exists()
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        assert summary["dataset"]["n_long"] == 3
        assert summary["dataset"]["n_short"] == 2
        # A handful of expected plots
        for name in ("01_missingness", "02_em_convergence", "05_backcast_fan",
                     "07_holdout_scatter", "10_uncertainty_ellipse"):
            key = name
            assert key in paths
            assert paths[key].exists()

    def test_pipeline_from_yaml(self, tmp_path):
        """Pipeline should load the packaged default YAML config if no dict is supplied."""
        pipe = BackcastPipeline(log_level="WARNING")
        assert pipe.config.get("random_seed") is not None

    def test_run_with_auto_method_triggers_model_selection(self, tmp_path):
        csv = _synthetic_csv(tmp_path, seed=33)
        cfg = dict(_TINY_CONFIG)
        cfg["imputation"] = {"n_imputations": 5, "method": "auto"}
        cfg["model_selection"] = {
            "enabled": False,  # "auto" alone is enough to trigger it
            "candidates": ["unconditional_em", "regime_conditional"],
            "criterion": "combined",
            "hmm_n_regimes": 2,
        }
        pipe = BackcastPipeline(config_dict=cfg, log_level="WARNING")
        res = pipe.run(csv)
        assert res.model_selection is not None
        assert res.model_selection.best_method in (
            "unconditional_em", "regime_conditional",
        )
        # The imputation method should be patched to the selected method
        assert res.imputation.method == res.model_selection.best_method


# ---------------------------------------------------------------------------
# Tier 2 end-to-end (skips when fixture is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER2_CSV.exists(), reason="Tier 2 fixture not generated")
class TestPipelineTier2:
    def test_runs_on_tier2(self, tmp_path):
        cfg = dict(_TINY_CONFIG)
        # Use regime-conditional imputation for Tier 2 (regime-switching DGP)
        cfg["imputation"] = {"n_imputations": 20, "method": "regime_conditional"}
        cfg["validation"] = {"holdout_days": 504, "n_windows": 3, "coverage_level": 0.95}
        cfg["hmm"] = {"n_regimes_candidates": [2, 3], "max_iterations": 200, "tolerance": 1e-3}
        cfg["downstream"] = dict(cfg["downstream"])
        cfg["downstream"]["backtest_strategies"] = ["equal_weight"]
        pipe = BackcastPipeline(config_dict=cfg, log_level="WARNING")
        res = pipe.run(TIER2_CSV)
        assert res.hmm is not None
        assert res.hmm.n_regimes == 2
        assert res.imputation.method == "regime_conditional"
        out_dir = tmp_path / "tier2_out"
        paths = pipe.export(res, out_dir)
        assert (out_dir / "summary.json").exists()
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        # Regime labels serialised correctly
        assert summary["hmm"]["n_regimes"] == 2
        # HMM model-selection scores captured
        assert summary["hmm_selection"]["best_n_regimes"] == 2
