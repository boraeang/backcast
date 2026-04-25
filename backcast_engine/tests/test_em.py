"""Tests for backcast.models.em_stambaugh.

Includes a ground-truth-recovery test against the Tier 1 synthetic dataset in
``synthetic_data_generator/output/tier1/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backcast.models.em_stambaugh import em_stambaugh, _nearest_psd


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_CSV = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns.csv"
TIER1_COMPLETE = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "returns_complete.csv"
TIER1_GT = REPO_ROOT / "synthetic_data_generator" / "output" / "tier1" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mv_normal(n, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, sigma, size=n)


def _frobenius_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro") / np.linalg.norm(b, "fro"))


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestEMBehaviour:
    def test_fully_observed_converges_to_sample(self):
        """With no missing data, EM should match the sample mean & covariance."""
        mu_true = np.array([0.0005, 0.0002, -0.0003])
        sigma_true = np.array([[1e-4, 3e-5, -1e-5],
                               [3e-5, 9e-5,  2e-5],
                               [-1e-5, 2e-5, 1.2e-4]])
        R = _mv_normal(5000, mu_true, sigma_true, seed=0)
        df = pd.DataFrame(R, columns=["A", "B", "C"])

        result = em_stambaugh(df, max_iter=50, tolerance=1e-10)
        np.testing.assert_allclose(result.mu, df.mean().values, atol=1e-12)
        np.testing.assert_allclose(
            result.sigma, np.cov(R, rowvar=False, bias=True), atol=1e-10
        )
        # No missing data → convergence within 2 iters (init uses n-1
        # normalisation, M-step uses T, so one settling step is expected).
        assert result.n_iter <= 2
        assert result.converged

    def test_mask_and_recover_covariance(self):
        """Mask a fully observed dataset; EM should recover the true Σ well."""
        mu_true = np.array([0.0004, -0.0002, 0.0003, 0.0001])
        # PSD cov
        A = np.array([[1.0, 0.5, 0.2, -0.1],
                      [0.5, 1.5, 0.3,  0.1],
                      [0.2, 0.3, 0.8,  0.2],
                      [-0.1, 0.1, 0.2, 1.2]])
        sigma_true = (A @ A.T) * 1e-4
        R = _mv_normal(4000, mu_true, sigma_true, seed=1)
        df = pd.DataFrame(R, columns=["A", "B", "C", "D"])

        # Mask last two columns for the first 1500 rows
        df.iloc[:1500, 2:] = np.nan

        result = em_stambaugh(df, max_iter=200, tolerance=1e-10)
        assert result.converged
        err = _frobenius_rel_err(result.sigma, sigma_true)
        assert err < 0.10, f"Frobenius rel err {err:.3f} >= 0.10"

    def test_sigma_is_psd(self):
        mu = np.zeros(3)
        sigma = np.eye(3) * 1e-4
        R = _mv_normal(2000, mu, sigma, seed=2)
        df = pd.DataFrame(R, columns=list("ABC"))
        df.iloc[:500, 2] = np.nan
        result = em_stambaugh(df, max_iter=100)
        eigvals = np.linalg.eigvalsh(result.sigma)
        assert (eigvals > -1e-10).all(), f"Sigma not PSD: min eig = {eigvals.min()}"

    def test_nearest_psd_helper(self):
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues -1, 3
        fixed = _nearest_psd(bad)
        eigvals = np.linalg.eigvalsh(fixed)
        assert (eigvals > -1e-12).all()

    def test_loglikelihood_monotone_increasing(self):
        """EM should increase the observed-data log-likelihood each iter."""
        mu = np.array([0.0002, 0.0001, -0.0001])
        sigma = np.array([[1e-4, 2e-5, 1e-5],
                          [2e-5, 1.5e-4, 3e-5],
                          [1e-5, 3e-5, 8e-5]])
        R = _mv_normal(3000, mu, sigma, seed=3)
        df = pd.DataFrame(R, columns=list("ABC"))
        df.iloc[:1000, 2] = np.nan
        result = em_stambaugh(df, max_iter=100, tolerance=1e-12,
                              track_loglikelihood=True)
        ll = np.asarray(result.log_likelihood_trace)
        # Allow minor numerical noise; require (nearly) monotonic increase
        diffs = np.diff(ll)
        assert (diffs > -1e-4).all(), f"Non-monotone LL: min diff = {diffs.min()}"

    def test_staggered_missingness(self):
        """Three groups of short assets with different start dates."""
        mu = np.zeros(5)
        A = np.array([[1.0, 0.3, 0.2, 0.1, -0.1],
                      [0.3, 0.9, 0.2, 0.1,  0.0],
                      [0.2, 0.2, 1.1, 0.2,  0.1],
                      [0.1, 0.1, 0.2, 0.8,  0.1],
                      [-0.1, 0.0, 0.1, 0.1, 1.3]])
        sigma = (A @ A.T) * 1e-4
        R = _mv_normal(4000, mu, sigma, seed=4)
        df = pd.DataFrame(R, columns=list("ABCDE"))
        # A, B long.  C starts at 1000, D at 1500, E at 2000.
        df.iloc[:1000, 2] = np.nan
        df.iloc[:1500, 3] = np.nan
        df.iloc[:2000, 4] = np.nan
        result = em_stambaugh(df, max_iter=200, tolerance=1e-10)
        assert result.converged
        # Long-long block should be very close
        err_ll = _frobenius_rel_err(result.sigma[:2, :2], sigma[:2, :2])
        assert err_ll < 0.05
        # Full Σ tolerance
        err_full = _frobenius_rel_err(result.sigma, sigma)
        assert err_full < 0.15


# ---------------------------------------------------------------------------
# Ground-truth recovery against the Tier 1 synthetic data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TIER1_CSV.exists(), reason="Tier 1 fixtures not generated")
class TestTier1Recovery:
    @pytest.fixture(scope="class")
    def em(self):
        df = pd.read_csv(TIER1_CSV, index_col="date", parse_dates=True).astype(np.float64)
        result = em_stambaugh(df, max_iter=500, tolerance=1e-8,
                              track_loglikelihood=False)
        with open(TIER1_GT) as fh:
            gt = json.load(fh)
        return df, result, gt

    def test_em_converged(self, em):
        _, result, _ = em
        assert result.converged

    def test_mu_close_to_truth(self, em):
        df, result, gt = em
        mu_true = np.asarray(gt["mu_daily"])
        true_vols = np.sqrt(np.diag(np.asarray(gt["sigma_daily"])))
        # Long assets: n=5000.  Short assets: n_obs=2000 (overlap period),
        # plus 3000 imputed — use the full T=5000 for the tolerance.
        T = len(df)
        tol = 3.0 * true_vols / np.sqrt(T)
        err = np.abs(result.mu - mu_true)
        assert (err < tol).sum() >= len(mu_true) - 1, (
            f"mu err vs tol:\n{err}\n{tol}"
        )

    def test_sigma_frobenius_within_tolerance(self, em):
        _, result, gt = em
        sigma_true = np.asarray(gt["sigma_daily"])
        rel_err = _frobenius_rel_err(result.sigma, sigma_true)
        # Long-long block has T=5000 samples, short blocks T=2000.
        # Expect Frobenius rel err comfortably < 10%.
        assert rel_err < 0.10, f"Frobenius rel err {rel_err:.4f} >= 0.10"

    def test_long_long_block_very_close(self, em):
        _, result, gt = em
        sigma_true = np.asarray(gt["sigma_daily"])
        n_long = len(gt["long_assets"])
        err = _frobenius_rel_err(
            result.sigma[:n_long, :n_long],
            sigma_true[:n_long, :n_long],
        )
        # Long block should be very tight with T=5000
        assert err < 0.05, f"Long-long Frobenius err {err:.4f} >= 0.05"

    def test_sigma_is_psd(self, em):
        _, result, _ = em
        eigvals = np.linalg.eigvalsh(result.sigma)
        assert (eigvals > -1e-10).all()

    def test_conditional_params_shapes(self, em):
        _, result, gt = em
        n_long = len(gt["long_assets"])
        n_short = len(gt["short_assets"])
        cp = result.conditional_params
        assert cp.beta.shape == (n_short, n_long)
        assert cp.alpha.shape == (n_short,)
        assert cp.cond_cov.shape == (n_short, n_short)
