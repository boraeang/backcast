"""Tier 2 regression tests for the regime-NaN fix (docs/fix_regime_nan_prompt.md, Part 5).

Spec test -> location:

1. No-NaN guarantee on Tier 2              -> TestNoNaNTier2 (here)
2. Adversarial regime -> pooled, not NaN   -> TestAdversarialPooled (here)
3. Thin regime -> shrunk, PSD              -> TestThinRegimeShrunk (here);
                                              synthetic cases in test_regime_params.TestCascade
4. HMM degenerate-K rejection              -> test_regime_hmm.TestDegenerateRejection
                                              ::test_rejects_overparameterised_k
5. PSD enforcement passes cho_factor       -> test_regime_params.TestNearestPSD
                                              ::test_singular_matrix_passes_cholesky
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from backcast.data.loader import build_backcast_dataset
from backcast.imputation.multiple_impute import multiple_impute_regime
from backcast.imputation.regime_params import build_regime_params
from backcast.imputation.single_impute import regime_single_impute
from backcast.models.regime_hmm import fit_and_select_hmm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "synthetic_data_generator" / "output"


def _load(tier: str):
    d = OUT / tier
    if not (d / "returns.csv").exists() or not (d / "ground_truth.json").exists():
        pytest.skip(f"{tier} fixtures not generated")
    masked = pd.read_csv(d / "returns.csv", index_col="date", parse_dates=True).astype(np.float64)
    with open(d / "ground_truth.json") as fh:
        gt = json.load(fh)
    ds = build_backcast_dataset(masked)
    true_labels = np.asarray(gt["regime_labels"], dtype=np.int64)
    assert len(true_labels) == len(ds.returns_full)
    return ds, gt, true_labels


def _hmm_labels(ds) -> np.ndarray:
    """Production path: hardened HMM selection on the long-history assets."""
    mask = np.zeros(len(ds.returns_full), dtype=bool)
    mask[-ds.overlap_length:] = True
    sel = fit_and_select_hmm(
        ds.returns_full[ds.long_assets], n_regimes_candidates=(2, 3),
        n_short=ds.n_short, overlap_mask=mask, seed=42,
    )
    for arr in (sel.best.means, sel.best.covariances, sel.best.transition_matrix):
        assert np.isfinite(arr).all()
    return sel.best.regime_labels


def _params(ds, labels: np.ndarray) -> dict:
    return build_regime_params(
        ds.overlap_matrix, labels[-ds.overlap_length:], ds.short_assets,
        regimes=np.unique(labels),
    )


@pytest.fixture(scope="module")
def tier2():
    ds, gt, true_labels = _load("tier2")
    return ds, gt, true_labels, _hmm_labels(ds)


@pytest.fixture(scope="module")
def tier2_adv():
    ds, gt, true_labels = _load("tier2_adversarial")
    assert gt["adversarial"] is True
    return ds, gt, true_labels, _hmm_labels(ds)


# ---------------------------------------------------------------------------
# 1. No-NaN guarantee
# ---------------------------------------------------------------------------

class TestNoNaNTier2:
    @pytest.mark.parametrize("which", ["tier2", "tier2_adv"])
    def test_multiple_imputation_has_no_nan(self, which, request):
        ds, _, _, labels = request.getfixturevalue(which)
        mi = multiple_impute_regime(ds, labels, _params(ds, labels),
                                    n_imputations=10, seed=0)
        imputed_histories = mi.imputations
        assert all(df.isna().sum().sum() == 0 for df in imputed_histories)
        assert all(np.isfinite(df.to_numpy()).all() for df in imputed_histories)

    @pytest.mark.parametrize("which", ["tier2", "tier2_adv"])
    def test_single_imputation_has_no_nan(self, which, request):
        ds, _, _, labels = request.getfixturevalue(which)
        filled = regime_single_impute(ds.returns_full, labels, _params(ds, labels))
        assert filled.isna().sum().sum() == 0

    def test_observed_cells_untouched(self, tier2):
        ds, _, _, labels = tier2
        mi = multiple_impute_regime(ds, labels, _params(ds, labels), n_imputations=3, seed=0)
        observed = ds.returns_full.notna().to_numpy()
        for df in mi.imputations:
            np.testing.assert_array_equal(df.to_numpy()[observed],
                                          ds.returns_full.to_numpy()[observed])


# ---------------------------------------------------------------------------
# 2. Adversarial regime -> pooled, not NaN
# ---------------------------------------------------------------------------

class TestAdversarialPooled:
    def test_ground_truth_regime_is_pooled(self, tier2_adv, caplog):
        """Using the true labels: every backcast date of the backcast-only
        regime is imputed and tagged ``pooled``; a WARNING is logged."""
        ds, gt, true_labels, _ = tier2_adv
        assert gt["regime_counts_overlap"]["regime_1"] == 0
        with caplog.at_level("WARNING", logger="backcast.imputation.regime_params"):
            params = _params(ds, true_labels)
        assert params[1].source == "pooled" and params[1].n_obs == 0
        assert params[0].source == "full"
        assert any(r.levelname == "WARNING" and "Regime 1" in r.message
                   and "pooled" in r.message for r in caplog.records)

        mi = multiple_impute_regime(ds, true_labels, params, n_imputations=5, seed=0)
        assert all(df.isna().sum().sum() == 0 for df in mi.imputations)

        fill = mi.regime_sources.fill_source
        backcast_labels = pd.Series(true_labels, index=ds.returns_full.index).loc[fill.index]
        assert (fill[backcast_labels == 1] == "pooled").all()
        assert (fill[backcast_labels == 0] == "full").all()
        n_adv = gt["regime_counts_backcast"]["regime_1"]
        for asset in ds.short_assets:
            assert mi.regime_sources.breakdown.loc[asset, "pooled"] == n_adv

    def test_hmm_path_tags_adversarial_dates_pooled(self, tier2_adv, caplog):
        """Production path (HMM-decoded labels): the decoded crisis regime is
        absent from the overlap, so it is pooled; nearly all true crisis
        backcast dates end up pooled."""
        ds, _, true_labels, hmm_labels = tier2_adv
        with caplog.at_level("WARNING", logger="backcast.imputation.regime_params"):
            params = _params(ds, hmm_labels)
        pooled = [k for k, p in params.items() if p.source == "pooled"]
        assert pooled, "expected the backcast-only regime to be pooled"
        assert any("pooled" in r.message for r in caplog.records if r.levelname == "WARNING")

        filled, src = regime_single_impute(ds.returns_full, hmm_labels, params,
                                           return_sources=True)
        assert filled.isna().sum().sum() == 0
        true_bc = pd.Series(true_labels, index=ds.returns_full.index).loc[src.fill_source.index]
        hit = (src.fill_source[true_bc == 1] == "pooled").mean()
        assert hit > 0.9, f"only {hit:.1%} of true adversarial dates tagged pooled"


# ---------------------------------------------------------------------------
# 3. Thin regime -> shrunk, PSD
# ---------------------------------------------------------------------------

class TestThinRegimeShrunk:
    N_KEEP = 20   # N_short (3) < 20 < reliable_threshold (60)

    @pytest.fixture(scope="class")
    def thin(self, tier2):
        """Tier 2 with all but N_KEEP regime-1 rows removed from the overlap."""
        ds, _, true_labels, _ = tier2
        ov_labels = true_labels[-ds.overlap_length:]
        r1_rows = np.where(ov_labels == 1)[0]
        keep = np.ones(ds.overlap_length, dtype=bool)
        keep[r1_rows[self.N_KEEP:]] = False
        overlap_thin = ds.overlap_matrix.iloc[keep]
        params = build_regime_params(overlap_thin, ov_labels[keep], ds.short_assets,
                                     regimes=np.unique(true_labels))
        return ds, true_labels, params

    def test_source_is_shrunk(self, thin):
        ds, _, params = thin
        assert ds.n_short < self.N_KEEP < 60
        assert params[1].source == "shrunk"
        assert params[1].n_obs == self.N_KEEP
        assert 0.0 < params[1].shrinkage <= 1.0
        assert params[0].source == "full"

    def test_covariances_are_psd(self, thin):
        _, _, params = thin
        p = params[1]
        assert (np.linalg.eigvalsh(p.sigma) > 0).all()
        assert (np.linalg.eigvalsh(p.sigma_cond) > 0).all()
        cho_factor(p.sigma)
        cho_factor(p.sigma_cond)
        for arr in (p.mu, p.beta, p.alpha):
            assert np.isfinite(arr).all()

    def test_backcast_dates_tagged_shrunk(self, thin):
        ds, true_labels, params = thin
        mi = multiple_impute_regime(ds, true_labels, params, n_imputations=3, seed=0)
        assert all(df.isna().sum().sum() == 0 for df in mi.imputations)
        fill = mi.regime_sources.fill_source
        bc_labels = pd.Series(true_labels, index=ds.returns_full.index).loc[fill.index]
        assert (fill[bc_labels == 1] == "shrunk").all()
        assert (fill[bc_labels == 0] == "full").all()
