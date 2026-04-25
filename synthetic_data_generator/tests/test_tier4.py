"""Tests for Tier 4: stress scenarios.

Each scenario must exhibit the property it was designed to test:

- ``short_overlap``    — exactly 250 days of overlap.
- ``high_dimension``   — 25 assets (5 long, 20 short).
- ``near_singular``    — condition_number > 1000, injected correlations preserved.
- ``staggered_heavy``  — 10 short assets with starts [2000, 2100, ..., 2900].
- ``all``              — generate_tier4_all yields all four scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthgen.config import SyntheticConfig, Tier4Config
from synthgen.correlation import is_psd
from synthgen.tier4_stress import (
    SCENARIO_NAMES,
    generate_tier4,
    generate_tier4_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(scenario: str, **overrides) -> SyntheticConfig:
    base = dict(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        tier=4,
        tier4_config=Tier4Config(scenario=scenario),
    )
    base.update(overrides)
    return SyntheticConfig(**base)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestTier4Dispatch:
    def test_unknown_scenario_raises(self):
        cfg = _cfg("short_overlap")
        cfg.tier4_config = Tier4Config(scenario="does_not_exist")
        with pytest.raises(ValueError, match="Unknown Tier 4 scenario"):
            generate_tier4(cfg)

    def test_all_scenario_raises_on_single_call(self):
        cfg = _cfg("all")
        with pytest.raises(ValueError, match="all"):
            generate_tier4(cfg)


# ---------------------------------------------------------------------------
# Scenario: short_overlap
# ---------------------------------------------------------------------------

class TestShortOverlap:
    @pytest.fixture(scope="class")
    def out(self):
        return generate_tier4(_cfg("short_overlap"))

    def test_tier_and_scenario(self, out):
        _, _, gt = out
        assert gt["tier"] == 4
        assert gt["scenario"] == "short_overlap"

    def test_expected_challenges(self, out):
        _, _, gt = out
        assert "overlap" in gt["expected_challenges"].lower()

    def test_overlap_length(self, out):
        _, _, gt = out
        # Every short asset starts at t_total - 250
        expected_start = gt["n_observations"] - 250
        for start in gt["short_asset_start_indices"].values():
            assert start == expected_start

    def test_condition_number_present(self, out):
        _, _, gt = out
        assert "condition_number" in gt
        assert np.isfinite(gt["condition_number"])

    def test_masking_pattern(self, out):
        masked, _, gt = out
        for name, idx in gt["short_asset_start_indices"].items():
            assert masked[name].iloc[:idx].isna().all()
            assert masked[name].iloc[idx:].notna().all()

    def test_too_small_t_total_raises(self):
        cfg = _cfg("short_overlap", t_total=200)
        with pytest.raises(ValueError, match="t_total"):
            generate_tier4(cfg)


# ---------------------------------------------------------------------------
# Scenario: high_dimension
# ---------------------------------------------------------------------------

class TestHighDimension:
    @pytest.fixture(scope="class")
    def out(self):
        return generate_tier4(_cfg("high_dimension"))

    def test_shape(self, out):
        masked, complete, gt = out
        assert complete.shape[1] == 25  # 5 long + 20 short
        assert len(gt["long_assets"]) == 5
        assert len(gt["short_assets"]) == 20

    def test_sigma_psd(self, out):
        _, _, gt = out
        sigma = np.asarray(gt["sigma_daily"])
        assert sigma.shape == (25, 25)
        assert is_psd(sigma)

    def test_all_short_assets_masked(self, out):
        masked, _, gt = out
        for name in gt["short_assets"]:
            assert masked[name].isna().any()


# ---------------------------------------------------------------------------
# Scenario: near_singular
# ---------------------------------------------------------------------------

class TestNearSingular:
    @pytest.fixture(scope="class")
    def out(self):
        return generate_tier4(_cfg("near_singular"))

    def test_condition_number_above_1000(self, out):
        _, _, gt = out
        cond = gt["condition_number"]
        assert cond > 1000.0, f"condition number {cond:.1f} <= 1000"

    def test_sigma_psd(self, out):
        _, _, gt = out
        sigma = np.asarray(gt["sigma_daily"])
        assert is_psd(sigma)

    def test_injected_correlations_preserved(self, out):
        _, _, gt = out
        injected = gt["injected_correlations"]
        realised = gt["realised_correlations"]
        # After PSD projection, realised correlations should be close to
        # injected values (tolerance 0.05).
        for key, target in injected.items():
            assert abs(realised[key] - target) < 0.05, (
                f"Pair {key}: injected {target:.3f}, realised {realised[key]:.3f}"
            )

    def test_sample_correlation_close_to_true(self, out):
        _, complete, gt = out
        corr = np.asarray(gt["correlation"])
        sample_corr = complete.corr().values
        # With T=5000 i.i.d. draws, sample correlation is close to true
        max_err = float(np.max(np.abs(sample_corr - corr)))
        assert max_err < 0.1, f"max corr err {max_err:.3f} > 0.1"


# ---------------------------------------------------------------------------
# Scenario: staggered_heavy
# ---------------------------------------------------------------------------

class TestStaggeredHeavy:
    @pytest.fixture(scope="class")
    def out(self):
        return generate_tier4(_cfg("staggered_heavy"))

    def test_ten_short_assets(self, out):
        _, _, gt = out
        assert len(gt["short_assets"]) == 10

    def test_staggered_starts(self, out):
        _, _, gt = out
        starts = list(gt["short_asset_start_indices"].values())
        expected = [2000 + 100 * i for i in range(10)]
        assert starts == expected

    def test_monotone_per_asset(self, out):
        masked, _, gt = out
        for name, idx in gt["short_asset_start_indices"].items():
            assert masked[name].iloc[:idx].isna().all()
            assert masked[name].iloc[idx:].notna().all()

    def test_too_small_t_total_raises(self):
        cfg = _cfg("staggered_heavy", t_total=2500)
        with pytest.raises(ValueError, match="t_total"):
            generate_tier4(cfg)


# ---------------------------------------------------------------------------
# Scenario: all
# ---------------------------------------------------------------------------

class TestAll:
    @pytest.fixture(scope="class")
    def all_out(self):
        return generate_tier4_all(_cfg("all"))

    def test_all_four_scenarios_present(self, all_out):
        assert set(all_out.keys()) == set(SCENARIO_NAMES)

    def test_each_has_valid_structure(self, all_out):
        for name, (masked, complete, gt) in all_out.items():
            assert gt["tier"] == 4
            assert gt["scenario"] == name
            assert "condition_number" in gt
            assert "expected_challenges" in gt
            assert isinstance(masked, pd.DataFrame)
            assert isinstance(complete, pd.DataFrame)
            assert masked.shape == complete.shape

    def test_distinct_outputs(self, all_out):
        """Each scenario should produce distinct output (different seeds)."""
        shapes = {name: complete.shape for name, (_, complete, _) in all_out.items()}
        # high_dimension has 25 cols, others have 8, staggered has 15
        assert shapes["high_dimension"][1] == 25
        assert shapes["staggered_heavy"][1] == 15  # 5 long + 10 short
        assert shapes["short_overlap"][1] == 8
        assert shapes["near_singular"][1] == 8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestTier4Reproducibility:
    @pytest.mark.parametrize("scenario", list(SCENARIO_NAMES))
    def test_same_seed_identical(self, scenario):
        cfg = _cfg(scenario, t_total=3500)
        _, c1, _ = generate_tier4(cfg)
        _, c2, _ = generate_tier4(cfg)
        pd.testing.assert_frame_equal(c1, c2)
