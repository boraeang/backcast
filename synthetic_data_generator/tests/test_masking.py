"""Tests for synthgen.masking — apply monotone NaN patterns to a returns frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthgen.masking import apply_masking


class TestMasking:
    def _make_df(self, n_rows: int = 100, n_cols: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((n_rows, n_cols))
        cols = [f"A{i}" for i in range(n_cols)]
        idx = pd.date_range("1990-01-02", periods=n_rows, freq="B")
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_masked_shape_unchanged(self):
        df = self._make_df()
        masked, _ = apply_masking(df, {"A3": 50, "A4": 70})
        assert masked.shape == df.shape

    def test_nan_prefix_applied(self):
        df = self._make_df(100)
        masked, meta = apply_masking(df, {"A3": 40})
        assert masked["A3"].iloc[:40].isna().all()
        assert masked["A3"].iloc[40:].notna().all()

    def test_long_assets_unchanged(self):
        df = self._make_df(100)
        masked, _ = apply_masking(df, {"A4": 60})
        pd.testing.assert_series_equal(masked["A0"], df["A0"])
        pd.testing.assert_series_equal(masked["A1"], df["A1"])

    def test_monotone_flag_true(self):
        df = self._make_df(100)
        _, meta = apply_masking(df, {"A2": 30, "A3": 60})
        assert meta.is_monotone

    def test_missing_column_raises(self):
        df = self._make_df(100)
        with pytest.raises(KeyError):
            apply_masking(df, {"NONEXISTENT": 10})

    def test_out_of_bounds_raises(self):
        df = self._make_df(100)
        with pytest.raises(ValueError):
            apply_masking(df, {"A1": 200})

    def test_missing_fraction_correct(self):
        df = self._make_df(100, 5)
        _, meta = apply_masking(df, {"A0": 25, "A1": 50})
        expected = (25 + 50) / (100 * 5)
        assert abs(meta.missing_fraction - expected) < 1e-10
