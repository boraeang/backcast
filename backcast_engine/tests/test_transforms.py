"""Tests for backcast.data.transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backcast.data.transforms import (
    log_to_simple_returns,
    prices_to_returns,
    returns_to_prices,
    simple_to_log_returns,
)


def _small_returns():
    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    return pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01, 0.02],
                         "B": [0.005, 0.01, -0.005, 0.02, -0.01]}, index=idx)


class TestConversionRoundtrip:
    def test_simple_log_roundtrip(self):
        r = _small_returns()
        r2 = log_to_simple_returns(simple_to_log_returns(r))
        pd.testing.assert_frame_equal(r, r2)

    def test_price_return_roundtrip(self):
        r = _small_returns()
        prices = returns_to_prices(r, initial=100.0)
        r_recovered = prices_to_returns(prices).iloc[1:]
        pd.testing.assert_frame_equal(r_recovered, r.iloc[1:], check_names=False)

    def test_prices_start_from_initial(self):
        r = _small_returns()
        prices = returns_to_prices(r, initial=50.0)
        expected_first = 50.0 * (1.0 + r.iloc[0])
        np.testing.assert_allclose(prices.iloc[0].values, expected_first.values)

    def test_nan_preserved_in_returns(self):
        r = _small_returns()
        r.iloc[0, 1] = np.nan
        out = simple_to_log_returns(r)
        assert np.isnan(out.iloc[0, 1])

    def test_prices_handle_leading_nans(self):
        r = _small_returns()
        r.iloc[:2, 1] = np.nan
        prices = returns_to_prices(r, initial=100.0)
        # Column B starts from row 2
        assert prices["B"].iloc[:2].isna().all()
        assert prices["B"].iloc[2:].notna().all()
