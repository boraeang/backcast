"""Tests for scripts/run_backcast.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Import the CLI module by path (it lives under scripts/, not src/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "backcast_engine" / "scripts" / "run_backcast.py"

spec = importlib.util.spec_from_file_location("run_backcast", CLI_PATH)
_cli = importlib.util.module_from_spec(spec)
sys.modules["run_backcast"] = _cli
spec.loader.exec_module(_cli)


TIER2_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier2" / "returns.csv"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

class TestCLIParser:
    def test_requires_input_and_output(self):
        parser = _cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_basic_args(self):
        parser = _cli.build_parser()
        args = parser.parse_args(["--input", "r.csv", "--output", "./out"])
        assert args.input == "r.csv"
        assert args.output == "./out"
        assert args.method is None
        assert args.save_imputations is False

    def test_method_restricted_choices(self):
        parser = _cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--input", "r.csv", "--output", "o", "--method", "kalman"]
            )

    def test_quiet_flag(self):
        parser = _cli.build_parser()
        args = parser.parse_args(["-i", "r.csv", "-o", "o", "--quiet"])
        assert args.quiet is True


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_seed_and_method_applied(self, tmp_path):
        # Build a tiny in-memory pipeline with a dict config
        from backcast.pipeline import BackcastPipeline
        pipe = BackcastPipeline(
            config_dict={
                "random_seed": 0,
                "imputation": {"n_imputations": 10, "method": "unconditional_em"},
            },
            log_level="WARNING",
        )
        parser = _cli.build_parser()
        args = parser.parse_args([
            "-i", "x.csv", "-o", str(tmp_path),
            "--seed", "99", "--n-imputations", "42",
            "--method", "regime_conditional", "--save-imputations",
        ])
        _cli.apply_overrides(pipe, args)
        assert pipe.config["random_seed"] == 99
        assert pipe.seed == 99
        assert pipe.config["imputation"]["n_imputations"] == 42
        assert pipe.config["imputation"]["method"] == "regime_conditional"
        assert pipe.config["output"]["save_imputations"] is True


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _tiny_csv(tmp_path, T=1200, n_long=3, n_short=2, start=600, seed=0):
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


_TINY_CONFIG_YAML = """\
random_seed: 0
data: {min_overlap_days: 100}
em: {max_iterations: 50, tolerance: 1.0e-6, track_loglikelihood: true}
kalman: {state_noise_scale: 0.01, use_smoother: true}
hmm: {n_regimes_candidates: [2], max_iterations: 50, tolerance: 1.0e-3}
imputation: {n_imputations: 5, method: unconditional_em}
validation: {holdout_days: 150, n_windows: 3, coverage_level: 0.95}
downstream:
  covariance_shrinkage: true
  denoise_eigenvalues: false
  uncertainty_confidence: 0.95
  backtest_strategies: [equal_weight]
  backtest_lookback: 30
  backtest_rebalance_freq: 15
output: {plot_format: png, plot_dpi: 80, save_imputations: false}
"""


class TestCLIEndToEnd:
    def test_missing_input_returns_code_2(self, tmp_path):
        rc = _cli.main([
            "-i", str(tmp_path / "does_not_exist.csv"),
            "-o", str(tmp_path / "out"),
            "--quiet",
        ])
        assert rc == 2

    def test_runs_with_custom_config(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_TINY_CONFIG_YAML)
        csv_path = _tiny_csv(tmp_path)
        out_dir = tmp_path / "out"
        rc = _cli.main([
            "-i", str(csv_path), "-o", str(out_dir),
            "-c", str(cfg_path), "--quiet",
        ])
        assert rc == 0
        assert (out_dir / "summary.json").exists()
        # Summary JSON is loadable and has the expected top-level keys
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        assert summary["dataset"]["n_long"] == 3
        assert summary["dataset"]["n_short"] == 2
        assert summary["imputation"]["n_imputations"] == 5
        captured = capsys.readouterr()
        assert "BACKCAST PIPELINE RESULT" in captured.out

    def test_overrides_take_effect_in_run(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_TINY_CONFIG_YAML)
        csv_path = _tiny_csv(tmp_path, seed=1)
        out_dir = tmp_path / "out"
        rc = _cli.main([
            "-i", str(csv_path), "-o", str(out_dir), "-c", str(cfg_path),
            "--n-imputations", "8", "--seed", "99", "--quiet",
        ])
        assert rc == 0
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        assert summary["imputation"]["n_imputations"] == 8
        assert summary["imputation"]["seed"] == 99

    @pytest.mark.skipif(not TIER2_CSV.exists(), reason="Tier 2 fixture not generated")
    def test_runs_on_tier2(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_TINY_CONFIG_YAML)
        out_dir = tmp_path / "out"
        rc = _cli.main([
            "-i", str(TIER2_CSV), "-o", str(out_dir),
            "-c", str(cfg_path), "--method", "regime_conditional",
            "--n-imputations", "10", "--quiet",
        ])
        assert rc == 0
        assert (out_dir / "summary.json").exists()
        with open(out_dir / "summary.json") as fh:
            summary = json.load(fh)
        assert summary["imputation"]["method"] == "regime_conditional"
        # Tier 2 has 5 long + 3 short
        assert summary["dataset"]["n_long"] == 5
        assert summary["dataset"]["n_short"] == 3
