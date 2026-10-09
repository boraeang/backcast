"""End-to-end pipeline smoke tests (on synthetic data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.pipeline import BackcastPipeline, FullResults, normalize_config


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

    def test_regime_conditional_exports_source_breakdown(self, tmp_path):
        csv = _synthetic_csv(tmp_path, seed=3)
        cfg = {**_TINY_CONFIG,
               "imputation": {"n_imputations": 5, "method": "regime_conditional"}}
        pipe = BackcastPipeline(config_dict=cfg, log_level="WARNING")
        res = pipe.run(csv)
        assert res.imputation.regime_sources is not None
        assert all(df.isna().sum().sum() == 0 for df in res.imputation.imputations)
        out_dir = tmp_path / "artefacts"
        paths = pipe.export(res, out_dir)
        assert paths["12_imputation_source"].exists()
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        bd = summary["imputation"]["source_breakdown"]
        assert set(bd) == set(res.dataset.short_assets)
        for asset, counts in bd.items():
            assert set(counts) == {"full", "shrunk", "pooled"}
            assert sum(counts.values()) == int(res.dataset.returns_full[asset].isna().sum())

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


# ---------------------------------------------------------------------------
# Config normalisation / backward compatibility
# ---------------------------------------------------------------------------

class TestConfigCompat:
    def test_default_yaml_has_regime_keys(self):
        pipe = BackcastPipeline(log_level="WARNING")
        icfg = pipe.config["imputation"]
        assert "min_obs_per_regime" not in icfg
        assert icfg["regime_reliable_threshold"] is None
        assert icfg["regime_shrinkage"] == "auto"
        assert icfg["regime_fallback_to_pooled"] is True
        assert icfg["psd_epsilon"] == pytest.approx(1e-10)
        hcfg = pipe.config["hmm"]
        assert hcfg["min_covar"] == pytest.approx(1e-3)
        assert hcfg["reject_underfilled_states"] is True
        assert hcfg["fallback_to_single_regime"] is True

    def test_legacy_min_obs_mapped_with_deprecation(self, caplog):
        raw = {"imputation": {"method": "regime_conditional", "min_obs_per_regime": 15}}
        with caplog.at_level("WARNING", logger="backcast.pipeline"):
            pipe = BackcastPipeline(config_dict=raw, log_level="WARNING")
        icfg = pipe.config["imputation"]
        assert "min_obs_per_regime" not in icfg
        assert icfg["regime_reliable_threshold"] == 15
        assert any("DEPRECATION" in r.message and "min_obs_per_regime" in r.message
                   for r in caplog.records)
        # caller's dict is untouched
        assert raw["imputation"]["min_obs_per_regime"] == 15

    def test_explicit_threshold_wins_over_legacy(self, caplog):
        raw = {"imputation": {"min_obs_per_regime": 15, "regime_reliable_threshold": 80}}
        with caplog.at_level("WARNING", logger="backcast.pipeline"):
            cfg = normalize_config(raw)
        assert cfg["imputation"]["regime_reliable_threshold"] == 80
        assert "min_obs_per_regime" not in cfg["imputation"]
        assert any("ignored" in r.message for r in caplog.records)

    def test_legacy_yaml_file(self, tmp_path, caplog):
        p = tmp_path / "old.yaml"
        p.write_text("random_seed: 1\nimputation:\n  method: regime_conditional\n"
                     "  min_obs_per_regime: 30\n")
        with caplog.at_level("WARNING", logger="backcast.pipeline"):
            pipe = BackcastPipeline(p, log_level="WARNING")
        assert pipe.config["imputation"]["regime_reliable_threshold"] == 30
        assert any("DEPRECATION" in r.message for r in caplog.records)

    @pytest.mark.parametrize("icfg", [
        {"regime_shrinkage": 1.5},
        {"regime_shrinkage": "ledoit"},
        {"regime_reliable_threshold": 0},
        {"regime_reliable_threshold": 12.5},
        {"psd_epsilon": 0.0},
    ])
    def test_invalid_imputation_settings_rejected(self, icfg):
        with pytest.raises(ValueError):
            normalize_config({"imputation": icfg})

    def test_invalid_min_covar_rejected(self):
        with pytest.raises(ValueError, match="min_covar"):
            normalize_config({"hmm": {"min_covar": -1e-3}})

    @pytest.mark.parametrize("shrink", ["auto", "heuristic", None, 0.0, 0.3, 1])
    def test_valid_shrinkage_accepted(self, shrink):
        cfg = normalize_config({"imputation": {"regime_shrinkage": shrink}})
        assert cfg["imputation"]["regime_shrinkage"] == shrink
