"""Tests for backcast.data.loader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backcast.data.loader import (
    BackcastDataset,
    build_backcast_dataset,
    detect_start_indices,
    load_backcast_dataset,
    load_returns_csv,
)
from backcast.exceptions import BackcastDataError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns(
    n_rows: int = 100,
    n_long: int = 2,
    n_short: int = 2,
    short_starts: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1990-01-02", periods=n_rows, freq="B")
    cols = [f"L{i}" for i in range(n_long)] + [f"S{i}" for i in range(n_short)]
    data = rng.standard_normal((n_rows, n_long + n_short)) * 0.01
    df = pd.DataFrame(data, index=dates, columns=cols)
    if short_starts is None:
        short_starts = [n_rows // 2] * n_short
    for i, start in enumerate(short_starts):
        df.iloc[:start, n_long + i] = np.nan
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# detect_start_indices
# ---------------------------------------------------------------------------

class TestDetectStartIndices:
    def test_full_history_all_zero(self):
        df = _make_returns(n_short=0)
        starts = detect_start_indices(df)
        assert all(v == 0 for v in starts.values())

    def test_short_assets_start_at_half(self):
        df = _make_returns(n_rows=100, short_starts=[50, 50])
        starts = detect_start_indices(df)
        assert starts["S0"] == 50 and starts["S1"] == 50

    def test_staggered_starts(self):
        df = _make_returns(n_rows=200, short_starts=[60, 120])
        starts = detect_start_indices(df)
        assert starts["S0"] == 60
        assert starts["S1"] == 120

    def test_mid_series_gap_rejected(self):
        df = _make_returns(n_rows=50)
        df.iloc[20, 0] = np.nan     # a hole in a long asset
        with pytest.raises(BackcastDataError, match="Non-monotone"):
            detect_start_indices(df)

    def test_all_nan_column_rejected(self):
        df = _make_returns(n_rows=50)
        df.iloc[:, 0] = np.nan
        with pytest.raises(BackcastDataError, match="entirely NaN"):
            detect_start_indices(df)


# ---------------------------------------------------------------------------
# build_backcast_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_basic_partition(self):
        df = _make_returns(n_rows=100, short_starts=[60, 80])
        ds = build_backcast_dataset(df)
        assert ds.long_assets == ["L0", "L1"]
        assert ds.short_assets == ["S0", "S1"]   # ordered by start index
        # overlap begins at last-starting short asset (index 80)
        assert ds.overlap_start == df.index[80]
        assert ds.backcast_end == df.index[79]
        assert ds.overlap_length == 20
        assert ds.backcast_length == 80

    def test_only_long_assets(self):
        df = _make_returns(n_rows=50, n_short=0)
        ds = build_backcast_dataset(df)
        assert ds.short_assets == []
        assert ds.backcast_start is None
        assert ds.backcast_length == 0

    def test_min_overlap_enforced(self):
        df = _make_returns(n_rows=100, short_starts=[99, 99])
        with pytest.raises(BackcastDataError, match="Overlap has only"):
            build_backcast_dataset(df, min_overlap_days=10)

    def test_long_history_matrix_is_long_only(self):
        df = _make_returns(n_rows=100, short_starts=[50, 50])
        ds = build_backcast_dataset(df)
        assert list(ds.long_history_matrix.columns) == ["L0", "L1"]
        # Only rows before the overlap start
        assert len(ds.long_history_matrix) == 50


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_round_trip(self, tmp_path):
        df = _make_returns(n_rows=80, short_starts=[30, 50])
        p = tmp_path / "returns.csv"
        df.to_csv(p)
        loaded = load_returns_csv(p)
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    def test_duplicate_dates_rejected(self, tmp_path):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03"])
        df = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        df.index.name = "date"
        p = tmp_path / "dup.csv"
        df.to_csv(p)
        with pytest.raises(BackcastDataError, match="Duplicate dates"):
            load_returns_csv(p)

    def test_load_backcast_dataset_on_tier1_like(self, tmp_path):
        df = _make_returns(n_rows=200, short_starts=[100, 150])
        p = tmp_path / "r.csv"
        df.to_csv(p)
        ds = load_backcast_dataset(p)
        assert ds.overlap_start == df.index[150]
        assert ds.n_long == 2 and ds.n_short == 2
