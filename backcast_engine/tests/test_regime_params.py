"""Tests for backcast.imputation.regime_params (full / shrunk / pooled cascade)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from backcast.data.loader import build_backcast_dataset
from backcast.exceptions import BackcastDataError
from backcast.imputation.multiple_impute import multiple_impute_regime
from backcast.imputation.regime_params import (
    build_regime_params,
    default_reliable_threshold,
    nearest_psd,
)
from backcast.imputation.single_impute import regime_single_impute

LONG = ["L0", "L1", "L2"]
SHORT = ["S0", "S1"]


def _overlap(T: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    N = len(LONG) + len(SHORT)
    A = rng.standard_normal((N, N)) * 0.004
    cov = A @ A.T + np.eye(N) * 1e-5
    X = rng.multivariate_normal(np.zeros(N), cov, size=T)
    return pd.DataFrame(X, columns=LONG + SHORT)


def _labels(T: int, counts: dict[int, int]) -> np.ndarray:
    labels = np.zeros(T, dtype=np.int64)
    pos = 0
    for k, n in counts.items():
        labels[pos:pos + n] = k
        pos += n
    return labels


class TestNearestPSD:
    def test_singular_matrix_passes_cholesky(self):
        v = np.array([1.0, 2.0, -1.0])
        singular = np.outer(v, v)        # rank 1
        with pytest.raises(np.linalg.LinAlgError):
            cho_factor(singular)
        fixed = nearest_psd(singular, 1e-10)
        cho_factor(fixed)                # must not raise
        assert np.linalg.eigvalsh(fixed).min() >= 1e-10 * 0.999

    def test_psd_input_unchanged(self):
        m = np.array([[2.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(nearest_psd(m), m, atol=1e-14)


class TestCascade:
    def test_default_threshold(self):
        assert default_reliable_threshold(2) == 60
        assert default_reliable_threshold(30) == 90

    def test_tiers(self):
        df = _overlap()
        # regime 0: 560 (full), 1: 30 (shrunk), 2: 2 (pooled, == N_short), 3: absent
        labels = _labels(len(df), {0: 568, 1: 30, 2: 2})
        params = build_regime_params(df, labels, SHORT, regimes=[0, 1, 2, 3])
        assert params[0].source == "full" and params[0].shrinkage == 0.0
        assert params[1].source == "shrunk" and 0.0 < params[1].shrinkage <= 1.0
        assert params[2].source == "pooled" and params[2].n_obs == 2
        assert params[3].source == "pooled" and params[3].n_obs == 0
        np.testing.assert_allclose(params[3].mu, df.mean().to_numpy())

    def test_all_outputs_finite_and_psd(self):
        df = _overlap()
        labels = _labels(len(df), {0: 590, 1: 7, 2: 3})
        params = build_regime_params(df, labels, SHORT, regimes=[0, 1, 2, 3])
        for p in params.values():
            for arr in (p.mu, p.sigma, p.sigma_cond, p.beta, p.alpha):
                assert np.isfinite(arr).all()
            assert np.linalg.eigvalsh(p.sigma).min() > 0
            assert np.linalg.eigvalsh(p.sigma_cond).min() > 0
            cho_factor(p.sigma)
            assert p.beta.shape == (len(SHORT), len(LONG))
            assert p.sigma_cond.shape == (len(SHORT), len(SHORT))

    def test_shrunk_below_n_long_still_psd(self):
        """n_k=3 > N_short=2 but < N=5: raw Σ_k is singular; shrinkage fixes it."""
        df = _overlap()
        labels = _labels(len(df), {0: 597, 1: 3})
        params = build_regime_params(df, labels, SHORT, shrinkage=0.0)
        assert params[1].source == "shrunk"
        assert np.linalg.eigvalsh(params[1].sigma_cond).min() > 0
        cho_factor(params[1].sigma)

    def test_fixed_and_heuristic_shrinkage(self):
        df = _overlap()
        labels = _labels(len(df), {0: 570, 1: 30})
        fixed = build_regime_params(df, labels, SHORT, shrinkage=0.25)
        assert fixed[1].shrinkage == pytest.approx(0.25)
        heur = build_regime_params(df, labels, SHORT, shrinkage="heuristic")
        assert heur[1].shrinkage == pytest.approx(60 / (60 + 30))

    def test_shrinkage_interpolates_toward_pooled(self):
        df = _overlap()
        labels = _labels(len(df), {0: 570, 1: 30})
        R1 = df.to_numpy()[labels == 1]
        raw = np.cov(R1, rowvar=False)
        pooled = np.cov(df.to_numpy(), rowvar=False)
        p = build_regime_params(df, labels, SHORT, shrinkage=0.4)[1]
        np.testing.assert_allclose(p.sigma, 0.4 * pooled + 0.6 * raw, rtol=1e-8, atol=1e-14)

    def test_shrink_mean_toggle(self):
        df = _overlap()
        labels = _labels(len(df), {0: 570, 1: 30})
        raw_mu = df.to_numpy()[labels == 1].mean(axis=0)
        pooled_mu = df.to_numpy().mean(axis=0)
        on = build_regime_params(df, labels, SHORT, shrinkage=0.4)[1]
        off = build_regime_params(df, labels, SHORT, shrinkage=0.4, shrink_mean=False)[1]
        np.testing.assert_allclose(on.mu, 0.4 * pooled_mu + 0.6 * raw_mu, rtol=1e-12)
        np.testing.assert_allclose(off.mu, raw_mu, rtol=1e-12)
        np.testing.assert_allclose(on.sigma, off.sigma)       # covariance unaffected
        # full and pooled tiers are unaffected by the toggle
        labels2 = _labels(len(df), {0: 600})
        a = build_regime_params(df, labels2, SHORT, regimes=[0, 1])
        b = build_regime_params(df, labels2, SHORT, regimes=[0, 1], shrink_mean=False)
        for k in (0, 1):
            np.testing.assert_allclose(a[k].mu, b[k].mu)

    def test_reliable_threshold_override(self):
        df = _overlap()
        labels = _labels(len(df), {0: 570, 1: 30})
        params = build_regime_params(df, labels, SHORT, reliable_threshold=25)
        assert params[1].source == "full"

    def test_pooled_warning_logged(self, caplog):
        df = _overlap()
        labels = _labels(len(df), {0: 600})
        with caplog.at_level("WARNING", logger="backcast.imputation.regime_params"):
            build_regime_params(df, labels, SHORT, regimes=[0, 1])
        assert any("Regime 1" in r.message and "pooled" in r.message for r in caplog.records)

    def test_fallback_disabled_raises(self):
        df = _overlap()
        labels = _labels(len(df), {0: 600})
        with pytest.raises(BackcastDataError):
            build_regime_params(df, labels, SHORT, regimes=[0, 1], fallback_to_pooled=False)

    @pytest.mark.parametrize("bad", [1.5, -0.1, "ledoit"])
    def test_invalid_shrinkage(self, bad):
        df = _overlap()
        with pytest.raises(ValueError):
            build_regime_params(df, np.zeros(len(df), dtype=int), SHORT, shrinkage=bad)

    def test_nan_overlap_rejected(self):
        df = _overlap()
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="fully observed"):
            build_regime_params(df, np.zeros(len(df), dtype=int), SHORT)


class TestImputersUseCascade:
    @pytest.fixture()
    def setup(self):
        df_ov = _overlap(T=600, seed=3)
        rng = np.random.default_rng(4)
        back = pd.DataFrame(rng.standard_normal((300, 5)) * 0.01, columns=LONG + SHORT)
        back[SHORT] = np.nan
        full = pd.concat([back, df_ov], ignore_index=True)
        full.index = pd.bdate_range("2000-01-03", periods=len(full))
        ds = build_backcast_dataset(full)
        # regime 2 appears only in the backcast period (adversarial)
        labels = np.zeros(len(full), dtype=np.int64)
        labels[:100] = 2
        labels[300:320] = 1             # thin regime in the overlap
        return ds, labels

    def test_single_and_multiple_fill_everything(self, setup):
        ds, labels = setup
        ov_labels = labels[-ds.overlap_length:]
        params = build_regime_params(ds.overlap_matrix, ov_labels, ds.short_assets,
                                     regimes=np.unique(labels))
        assert params[2].source == "pooled" and params[1].source == "shrunk"
        single = regime_single_impute(ds.returns_full, labels, params)
        assert not single.isna().any().any()
        mi = multiple_impute_regime(ds, labels, params, n_imputations=4, seed=0)
        assert all(df.isna().sum().sum() == 0 for df in mi.imputations)

    def test_uncovered_regime_raises_instead_of_nan(self, setup):
        ds, labels = setup
        ov_labels = labels[-ds.overlap_length:]
        params = build_regime_params(ds.overlap_matrix, ov_labels, ds.short_assets)
        assert 2 not in params
        with pytest.raises(BackcastDataError, match=r"\[2\]"):
            regime_single_impute(ds.returns_full, labels, params)
        with pytest.raises(BackcastDataError, match=r"\[2\]"):
            multiple_impute_regime(ds, labels, params, n_imputations=2, seed=0)


class TestSourceTagging:
    @pytest.fixture()
    def staggered(self):
        """S0 starts at row 300, S1 at row 400; regime 2 only in rows 0-99."""
        df_ov = _overlap(T=600, seed=5)
        rng = np.random.default_rng(6)
        back = pd.DataFrame(rng.standard_normal((400, 5)) * 0.01, columns=LONG + SHORT)
        back.loc[:299, "S0"] = np.nan
        back["S1"] = np.nan
        full = pd.concat([back, df_ov], ignore_index=True)
        full.index = pd.bdate_range("2000-01-03", periods=len(full))
        ds = build_backcast_dataset(full)
        labels = np.zeros(len(full), dtype=np.int64)
        labels[:100] = 2                 # pooled (absent from overlap)
        labels[350:380] = 1              # shrunk: 30 rows, all in S1-only backcast
        labels[500:530] = 1              # 30 overlap rows for regime 1
        params = build_regime_params(ds.overlap_matrix, labels[-ds.overlap_length:],
                                     ds.short_assets, regimes=np.unique(labels))
        return ds, labels, params

    def test_breakdown_counts_per_asset(self, staggered):
        ds, labels, params = staggered
        assert params[1].source == "shrunk" and params[2].source == "pooled"
        mi = multiple_impute_regime(ds, labels, params, n_imputations=2, seed=0)
        bd = mi.regime_sources.breakdown
        assert list(bd.columns) == ["full", "shrunk", "pooled"]
        # S0: rows 0-299 → 100 pooled, 200 full
        assert bd.loc["S0"].to_dict() == {"full": 200, "shrunk": 0, "pooled": 100}
        # S1: rows 0-399 → 100 pooled, 30 shrunk, 270 full
        assert bd.loc["S1"].to_dict() == {"full": 270, "shrunk": 30, "pooled": 100}
        # Every imputed cell is tagged, every observed cell is None
        cells = mi.regime_sources.cell_source
        nan_mask = ds.returns_full[cells.columns].isna()
        assert cells[nan_mask.to_numpy()].notna().equals(nan_mask[nan_mask.to_numpy()])
        assert cells.where(~nan_mask).isna().all().all()
        assert (mi.regime_sources.fill_source.iloc[:100] == "pooled").all()
        assert len(mi.regime_sources.fill_source) == 400
        np.testing.assert_allclose(mi.regime_sources.fractions().sum(axis=1), 1.0)

    def test_single_and_multiple_agree(self, staggered):
        ds, labels, params = staggered
        _, single_src = regime_single_impute(ds.returns_full, labels, params,
                                             return_sources=True)
        mi = multiple_impute_regime(ds, labels, params, n_imputations=2, seed=0)
        pd.testing.assert_frame_equal(single_src.breakdown, mi.regime_sources.breakdown)
        pd.testing.assert_frame_equal(single_src.cell_source, mi.regime_sources.cell_source)

    def test_breakdown_logged_at_info(self, staggered, caplog):
        ds, labels, params = staggered
        with caplog.at_level("INFO", logger="backcast.imputation.regime_params"):
            regime_single_impute(ds.returns_full, labels, params)
        msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
        assert any("parameter source" in m and "pooled" in m and "S1" in m for m in msgs)

    def test_source_timeline_plot(self, staggered):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from backcast.visualization.plots import plot_source_timeline

        ds, labels, params = staggered
        mi = multiple_impute_regime(ds, labels, params, n_imputations=2, seed=0)
        fig = plot_source_timeline(mi.regime_sources)
        assert isinstance(fig, Figure)
        title = fig.axes[0].get_title()
        assert "pooled" in title and "%" in title
        legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "pooled (200)" in legend and "shrunk (30)" in legend
        plt.close(fig)
