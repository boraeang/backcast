"""Tests for Tier 2: regime-switching multivariate Gaussian returns.

Validation criteria from the spec
-----------------------------------
- Empirical transition frequencies within a few standard errors of the true P.
- Regime-conditional sample mean and covariance are close to true regime params.
- In the adversarial variant, the crisis regime never appears in the overlap
  period.
- Regime durations follow a geometric distribution with mean 1/(1-P_kk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthgen.config import SyntheticConfig, Tier2Config
from synthgen.correlation import is_psd
from synthgen.tier2_regime import (
    _compute_regime_durations,
    _default_transition_matrix,
    _generate_regime_sequence,
    _resolve_tier2_config,
    _stationary_distribution,
    generate_tier2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier2_output():
    cfg = SyntheticConfig(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        correlation_method="factor_model",
        n_factors=4,
        tier=2,
        tier2_config=Tier2Config(n_regimes=2),
    )
    masked, complete, gt = generate_tier2(cfg)
    return masked, complete, gt, cfg


@pytest.fixture(scope="module")
def tier2_adversarial_output():
    cfg = SyntheticConfig(
        n_long_assets=5,
        n_short_assets=3,
        t_total=5000,
        short_start_day=3000,
        seed=42,
        tier=2,
        tier2_config=Tier2Config(n_regimes=2, adversarial=True),
    )
    masked, complete, gt = generate_tier2(cfg)
    return masked, complete, gt, cfg


# ---------------------------------------------------------------------------
# Helper / internal function tests
# ---------------------------------------------------------------------------

class TestTier2Helpers:
    def test_default_transition_matrix_two_regimes(self):
        P = _default_transition_matrix(2)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        assert P[0, 0] == 0.98
        assert P[1, 1] == 0.95

    def test_default_transition_matrix_three_regimes_stochastic(self):
        P = _default_transition_matrix(3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        assert (P.diagonal() > 0.9).all()

    def test_resolve_config_fills_defaults(self):
        cfg = _resolve_tier2_config(Tier2Config(n_regimes=2))
        assert cfg.transition_matrix is not None
        assert len(cfg.regime_vol_multipliers) == 2
        assert cfg.regime_vol_multipliers == [1.0, 2.5]
        assert cfg.regime_mean_adjustments == [1.0, -0.5]

    def test_resolve_config_invalid_P_raises(self):
        bad = Tier2Config(n_regimes=2, transition_matrix=[[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="rows must sum to 1"):
            _resolve_tier2_config(bad)

    def test_resolve_config_wrong_multiplier_length_raises(self):
        bad = Tier2Config(n_regimes=2, regime_vol_multipliers=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n_regimes"):
            _resolve_tier2_config(bad)

    def test_regime_sequence_length_and_range(self):
        rng = np.random.default_rng(0)
        P = np.array([[0.9, 0.1], [0.2, 0.8]])
        seq = _generate_regime_sequence(P, 1000, rng)
        assert seq.shape == (1000,)
        assert set(np.unique(seq).tolist()).issubset({0, 1})

    def test_compute_regime_durations_structure(self):
        seq = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
        d = _compute_regime_durations(seq, 2)
        assert "regime_0" in d and "regime_1" in d
        # Regime 0 runs: [3, 1]; Regime 1 runs: [2, 3]
        assert d["regime_0"]["count"] == 2
        assert d["regime_1"]["count"] == 2
        assert d["regime_0"]["total_days"] == 4
        assert d["regime_1"]["total_days"] == 5

    def test_stationary_distribution_two_regimes(self):
        P = np.array([[0.98, 0.02], [0.05, 0.95]])
        pi = _stationary_distribution(P)
        # Analytic stationary for 2-state chain:
        # pi_0 = P[1,0] / (P[0,1] + P[1,0]) = 0.05/0.07 ≈ 0.714
        np.testing.assert_allclose(pi[0], 0.05 / 0.07, atol=1e-6)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)

    def test_stationary_distribution_uniform_for_symmetric_P(self):
        P = np.array([[0.7, 0.3], [0.3, 0.7]])
        pi = _stationary_distribution(P)
        np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Shape / structure tests
# ---------------------------------------------------------------------------

class TestTier2Structure:
    def test_returns_shape(self, tier2_output):
        masked, complete, gt, cfg = tier2_output
        N = cfg.n_long_assets + cfg.n_short_assets
        assert complete.shape == (cfg.t_total, N)
        assert masked.shape == complete.shape

    def test_tier_field(self, tier2_output):
        _, _, gt, _ = tier2_output
        assert gt["tier"] == 2

    def test_ground_truth_required_fields(self, tier2_output):
        _, _, gt, _ = tier2_output
        for key in [
            "regime_labels", "transition_matrix", "regime_params",
            "regime_durations", "adversarial",
            "regime_counts_overlap", "regime_counts_backcast",
            "n_regimes", "stationary_distribution",
        ]:
            assert key in gt, f"missing {key} in ground_truth"

    def test_regime_labels_length(self, tier2_output):
        _, _, gt, cfg = tier2_output
        assert len(gt["regime_labels"]) == cfg.t_total

    def test_transition_matrix_row_stochastic(self, tier2_output):
        _, _, gt, _ = tier2_output
        P = np.asarray(gt["transition_matrix"])
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)

    def test_regime_params_structure(self, tier2_output):
        _, _, gt, _ = tier2_output
        rp = gt["regime_params"]
        N = len(gt["asset_names"])
        for key, params in rp.items():
            assert "mu" in params and "sigma" in params and "correlation" in params
            assert len(params["mu"]) == N
            sigma = np.asarray(params["sigma"])
            assert sigma.shape == (N, N)

    def test_masked_nan_pattern(self, tier2_output):
        masked, _, gt, _ = tier2_output
        for name, idx in gt["short_asset_start_indices"].items():
            assert masked[name].iloc[:idx].isna().all()
            assert masked[name].iloc[idx:].notna().all()

    def test_long_assets_fully_observed(self, tier2_output):
        masked, _, gt, _ = tier2_output
        for name in gt["long_assets"]:
            assert masked[name].notna().all()


# ---------------------------------------------------------------------------
# PSD tests for regime covariances
# ---------------------------------------------------------------------------

class TestTier2PSD:
    def test_each_regime_sigma_is_psd(self, tier2_output):
        _, _, gt, _ = tier2_output
        for name, params in gt["regime_params"].items():
            sigma = np.asarray(params["sigma"])
            assert is_psd(sigma), f"{name} sigma is not PSD"

    def test_each_regime_correlation_diagonal(self, tier2_output):
        _, _, gt, _ = tier2_output
        for name, params in gt["regime_params"].items():
            corr = np.asarray(params["correlation"])
            np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-8)

    def test_crisis_vol_exceeds_calm_vol(self, tier2_output):
        """Crisis regime (k=1) has elevated vols by construction."""
        _, _, gt, _ = tier2_output
        sigma_0 = np.asarray(gt["regime_params"]["regime_0"]["sigma"])
        sigma_1 = np.asarray(gt["regime_params"]["regime_1"]["sigma"])
        vol_0 = np.sqrt(np.diag(sigma_0))
        vol_1 = np.sqrt(np.diag(sigma_1))
        assert (vol_1 > vol_0).all(), "Every asset vol should be higher in regime 1"


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestTier2Statistics:
    def test_empirical_transition_matrix_close_to_true(self, tier2_output):
        """Empirical P_hat[i,j] within 3 SE of true P[i,j]."""
        _, _, gt, cfg = tier2_output
        seq = np.asarray(gt["regime_labels"])
        P_true = np.asarray(gt["transition_matrix"])
        K = P_true.shape[0]

        # Build transition counts
        counts = np.zeros((K, K), dtype=np.float64)
        for t in range(len(seq) - 1):
            counts[seq[t], seq[t + 1]] += 1

        row_totals = counts.sum(axis=1)
        assert (row_totals > 30).all(), "Not enough transitions for statistical test"
        P_hat = counts / row_totals[:, np.newaxis]

        # Standard error of a Bernoulli-like estimate: sqrt(p*(1-p)/n)
        max_z = 0.0
        for i in range(K):
            for j in range(K):
                p = P_true[i, j]
                se = np.sqrt(p * (1 - p) / row_totals[i])
                if se == 0:
                    continue
                z = abs(P_hat[i, j] - p) / se
                max_z = max(max_z, z)

        assert max_z < 4.0, (
            f"Max z-score of |P_hat - P| / SE = {max_z:.2f} (expected < 4). "
            f"P_hat=\n{P_hat}\nP=\n{P_true}"
        )

    def test_regime_conditional_mean_close_to_true(self, tier2_output):
        """For each regime, sample mean close to true regime mean."""
        _, complete, gt, cfg = tier2_output
        seq = np.asarray(gt["regime_labels"])

        for k_name, params in gt["regime_params"].items():
            k = int(k_name.split("_")[1])
            mask = seq == k
            n_k = int(mask.sum())
            if n_k < 200:
                continue  # Too few obs for reliable check
            mu_true = np.asarray(params["mu"])
            sigma_true = np.asarray(params["sigma"])
            vols_true = np.sqrt(np.diag(sigma_true))

            sample_mu = complete.values[mask].mean(axis=0)
            tol = 3.0 * vols_true / np.sqrt(n_k)

            failures = int((np.abs(sample_mu - mu_true) > tol).sum())
            # Allow 1 failure across N assets (multiple testing)
            assert failures <= 1, (
                f"{k_name}: {failures} assets outside 3σ/√n tolerance"
            )

    def test_regime_conditional_covariance_frobenius(self, tier2_output):
        """Regime-conditional Frobenius covariance error within reasonable bound."""
        _, complete, gt, _ = tier2_output
        seq = np.asarray(gt["regime_labels"])

        for k_name, params in gt["regime_params"].items():
            k = int(k_name.split("_")[1])
            mask = seq == k
            n_k = int(mask.sum())
            if n_k < 300:
                continue
            sigma_true = np.asarray(params["sigma"])
            sample_sigma = np.cov(complete.values[mask], rowvar=False)

            rel_err = (
                np.linalg.norm(sample_sigma - sigma_true, "fro")
                / np.linalg.norm(sigma_true, "fro")
            )
            # Looser bound than Tier 1 (smaller sample per regime)
            assert rel_err < 0.15, (
                f"{k_name}: Frobenius relative error {rel_err:.3f} >= 0.15 "
                f"(n={n_k})"
            )

    def test_regime_durations_geometric(self, tier2_output):
        """Empirical mean duration ≈ 1/(1-P_kk) within 30% tolerance."""
        _, _, gt, _ = tier2_output
        P = np.asarray(gt["transition_matrix"])
        durations = gt["regime_durations"]

        for k_name, stats in durations.items():
            k = int(k_name.split("_")[1])
            if stats["count"] < 10:
                continue  # Not enough runs for reliable check
            expected = 1.0 / (1.0 - P[k, k])
            observed = stats["mean"]
            # Use relative tolerance (SE of sample mean of geometric is large)
            assert 0.5 * expected < observed < 2.0 * expected, (
                f"{k_name}: mean duration {observed:.1f} vs expected {expected:.1f}"
            )


# ---------------------------------------------------------------------------
# Adversarial variant
# ---------------------------------------------------------------------------

class TestTier2Adversarial:
    def test_overlap_is_all_regime_zero(self, tier2_adversarial_output):
        _, _, gt, cfg = tier2_adversarial_output
        seq = np.asarray(gt["regime_labels"])
        min_start = min(gt["short_asset_start_indices"].values())
        assert (seq[min_start:] == 0).all(), (
            "Adversarial: overlap period must be all regime 0"
        )

    def test_backcast_has_multiple_regimes(self, tier2_adversarial_output):
        """The backcast period should retain both regimes."""
        _, _, gt, _ = tier2_adversarial_output
        seq = np.asarray(gt["regime_labels"])
        min_start = min(gt["short_asset_start_indices"].values())
        unique_backcast = np.unique(seq[:min_start])
        assert len(unique_backcast) >= 2, (
            "Adversarial: backcast should still contain the crisis regime"
        )

    def test_regime_counts_overlap_only_zero(self, tier2_adversarial_output):
        _, _, gt, _ = tier2_adversarial_output
        counts = gt["regime_counts_overlap"]
        assert counts["regime_0"] > 0
        for name, c in counts.items():
            if name != "regime_0":
                assert c == 0, f"Adversarial: {name} count in overlap should be 0"

    def test_adversarial_flag_propagated(self, tier2_adversarial_output):
        _, _, gt, _ = tier2_adversarial_output
        assert gt["adversarial"] is True


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestTier2Reproducibility:
    def test_same_seed_identical_output(self):
        cfg = SyntheticConfig(
            t_total=500, short_start_day=300, seed=7,
            tier=2, tier2_config=Tier2Config(n_regimes=2),
        )
        _, c1, gt1 = generate_tier2(cfg)
        _, c2, gt2 = generate_tier2(cfg)
        pd.testing.assert_frame_equal(c1, c2)
        assert gt1["regime_labels"] == gt2["regime_labels"]

    def test_different_seeds_differ(self):
        cfg1 = SyntheticConfig(
            t_total=500, short_start_day=300, seed=1,
            tier=2, tier2_config=Tier2Config(n_regimes=2),
        )
        cfg2 = SyntheticConfig(
            t_total=500, short_start_day=300, seed=2,
            tier=2, tier2_config=Tier2Config(n_regimes=2),
        )
        _, c1, _ = generate_tier2(cfg1)
        _, c2, _ = generate_tier2(cfg2)
        assert not c1.equals(c2)

    def test_three_regime_config(self):
        """Tier 2 runs with n_regimes=3 and produces valid output."""
        cfg = SyntheticConfig(
            t_total=2000, short_start_day=1500, seed=11,
            tier=2, tier2_config=Tier2Config(n_regimes=3),
        )
        _, complete, gt = generate_tier2(cfg)
        assert gt["n_regimes"] == 3
        assert len(gt["regime_params"]) == 3
        # At least two regimes should actually appear (chain should mix)
        unique_regimes = set(gt["regime_labels"])
        assert len(unique_regimes) >= 2
