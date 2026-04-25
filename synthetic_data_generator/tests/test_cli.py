"""Tests for the command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthgen.cli import build_config, build_parser, main, run
from synthgen.config import SyntheticConfig


# ---------------------------------------------------------------------------
# Parser / config
# ---------------------------------------------------------------------------

class TestParser:
    def test_tier_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_tier_out_of_range(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tier", "5"])

    def test_basic_tier1_args(self):
        parser = build_parser()
        args = parser.parse_args(["--tier", "1", "--output", "/tmp/x", "--seed", "7"])
        assert args.tier == 1
        assert args.output == "/tmp/x"
        assert args.seed == 7


class TestBuildConfig:
    def _ns(self, argv):
        return build_parser().parse_args(argv)

    def test_tier1_defaults(self):
        cfg = build_config(self._ns(["--tier", "1"]))
        assert isinstance(cfg, SyntheticConfig)
        assert cfg.tier == 1
        assert cfg.tier2_config is None
        assert cfg.tier3_config is None
        assert cfg.tier4_config is None

    def test_tier2_n_regimes(self):
        cfg = build_config(self._ns(["--tier", "2", "--n-regimes", "3"]))
        assert cfg.tier2_config.n_regimes == 3
        assert cfg.tier2_config.adversarial is False

    def test_tier2_adversarial(self):
        cfg = build_config(self._ns(["--tier", "2", "--adversarial"]))
        assert cfg.tier2_config.adversarial is True

    def test_tier3_df(self):
        cfg = build_config(self._ns(["--tier", "3", "--df", "4"]))
        assert cfg.tier3_config.degrees_of_freedom == 4.0

    def test_tier3_gaussian(self):
        cfg = build_config(self._ns(["--tier", "3", "--innovation", "gaussian"]))
        assert cfg.tier3_config.innovation_distribution == "gaussian"

    def test_tier4_scenario(self):
        cfg = build_config(self._ns(["--tier", "4", "--scenario", "near_singular"]))
        assert cfg.tier4_config.scenario == "near_singular"

    def test_custom_dimensions(self):
        cfg = build_config(self._ns([
            "--tier", "1",
            "--n-long", "8", "--n-short", "5",
            "--t-total", "7500", "--seed", "123",
        ]))
        assert cfg.n_long_assets == 8
        assert cfg.n_short_assets == 5
        assert cfg.t_total == 7500
        assert cfg.seed == 123

    def test_no_complete(self):
        cfg = build_config(self._ns(["--tier", "1", "--no-complete"]))
        assert cfg.save_complete_returns is False


# ---------------------------------------------------------------------------
# End-to-end runs (via main(argv=...))
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _run(self, tmp_path, *extra_args):
        out_dir = tmp_path / "out"
        argv = ["--tier", "1", "--output", str(out_dir),
                "--t-total", "500", "--short-start-day", "300",
                "--log-level", "WARNING"]
        argv.extend(extra_args)
        return main(argv), out_dir

    def test_tier1_writes_expected_files(self, tmp_path):
        code, out_dir = self._run(tmp_path)
        assert code == 0
        assert (out_dir / "returns.csv").exists()
        assert (out_dir / "returns_complete.csv").exists()
        assert (out_dir / "ground_truth.json").exists()

    def test_tier1_no_complete(self, tmp_path):
        out_dir = tmp_path / "out"
        main([
            "--tier", "1", "--output", str(out_dir),
            "--t-total", "500", "--short-start-day", "300",
            "--no-complete", "--log-level", "WARNING",
        ])
        assert (out_dir / "returns.csv").exists()
        assert not (out_dir / "returns_complete.csv").exists()

    def test_tier1_ground_truth_content(self, tmp_path):
        _, out_dir = self._run(tmp_path)
        with open(out_dir / "ground_truth.json") as fh:
            gt = json.load(fh)
        assert gt["tier"] == 1
        assert "mu" in gt and "sigma" in gt

    def test_tier2_ground_truth_regimes(self, tmp_path):
        out_dir = tmp_path / "t2"
        main([
            "--tier", "2", "--output", str(out_dir),
            "--n-regimes", "3",
            "--t-total", "1000", "--short-start-day", "600",
            "--log-level", "WARNING",
        ])
        with open(out_dir / "ground_truth.json") as fh:
            gt = json.load(fh)
        assert gt["tier"] == 2
        assert gt["n_regimes"] == 3

    def test_tier2_adversarial(self, tmp_path):
        out_dir = tmp_path / "t2a"
        main([
            "--tier", "2", "--output", str(out_dir), "--adversarial",
            "--t-total", "1500", "--short-start-day", "900",
            "--log-level", "WARNING",
        ])
        with open(out_dir / "ground_truth.json") as fh:
            gt = json.load(fh)
        assert gt["adversarial"] is True

    def test_tier3_df(self, tmp_path):
        out_dir = tmp_path / "t3"
        main([
            "--tier", "3", "--output", str(out_dir), "--df", "4",
            "--t-total", "1000", "--short-start-day", "600",
            "--log-level", "WARNING",
        ])
        with open(out_dir / "ground_truth.json") as fh:
            gt = json.load(fh)
        assert gt["tier"] == 3
        assert gt["innovation_df"] == 4.0

    def test_tier4_single_scenario(self, tmp_path):
        out_dir = tmp_path / "t4"
        main([
            "--tier", "4", "--output", str(out_dir),
            "--scenario", "near_singular",
            "--t-total", "2000", "--short-start-day", "1500",
            "--log-level", "WARNING",
        ])
        with open(out_dir / "ground_truth.json") as fh:
            gt = json.load(fh)
        assert gt["tier"] == 4
        assert gt["scenario"] == "near_singular"
        assert gt["condition_number"] > 1000

    def test_tier4_all_creates_subdirs(self, tmp_path):
        out_dir = tmp_path / "t4all"
        main([
            "--tier", "4", "--output", str(out_dir), "--scenario", "all",
            "--t-total", "3500", "--short-start-day", "2500",
            "--log-level", "WARNING",
        ])
        for scen in ("short_overlap", "high_dimension",
                     "near_singular", "staggered_heavy"):
            assert (out_dir / scen / "returns.csv").exists(), f"{scen} missing returns.csv"
            assert (out_dir / scen / "ground_truth.json").exists()
            with open(out_dir / scen / "ground_truth.json") as fh:
                gt = json.load(fh)
            assert gt["scenario"] == scen


# ---------------------------------------------------------------------------
# run() dispatcher  (bypassing argparse)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_returns_paths(self, tmp_path):
        cfg = SyntheticConfig(
            tier=1, t_total=400, short_start_day=300,
            output_dir=str(tmp_path / "direct"),
        )
        dirs = run(cfg)
        assert len(dirs) == 1
        assert (dirs[0] / "returns.csv").exists()
