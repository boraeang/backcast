"""Return / price / log-return conversions."""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def simple_to_log_returns(simple_returns: ArrayLike) -> ArrayLike:
    """Convert daily simple returns to log returns.

    log_r = log(1 + r)

    Parameters
    ----------
    simple_returns : pd.DataFrame, pd.Series, or np.ndarray
        Simple returns.  NaN is preserved.

    Returns
    -------
    Same type as input.
    """
    if isinstance(simple_returns, (pd.DataFrame, pd.Series)):
        return np.log1p(simple_returns)
    return np.log1p(np.asarray(simple_returns, dtype=np.float64))


def log_to_simple_returns(log_returns: ArrayLike) -> ArrayLike:
    """Convert daily log returns to simple returns.

    r = exp(log_r) - 1

    Parameters
    ----------
    log_returns : pd.DataFrame, pd.Series, or np.ndarray
        Log returns.  NaN is preserved.

    Returns
    -------
    Same type as input.
    """
    if isinstance(log_returns, (pd.DataFrame, pd.Series)):
        return np.expm1(log_returns)
    return np.expm1(np.asarray(log_returns, dtype=np.float64))


def returns_to_prices(
    simple_returns: pd.DataFrame,
    initial: float = 1.0,
) -> pd.DataFrame:
    """Cumulate simple returns into a synthetic price series.

    NaN rows in *simple_returns* produce NaN prices; the cumulative product
    resumes from the most recent valid price when observations begin.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Rows are dates, columns are asset names.
    initial : float, default 1.0
        Starting price applied to every asset.

    Returns
    -------
    pd.DataFrame
        Price series with the same index and columns.  The first non-NaN
        observation in each column is ``initial * (1 + r)``.
    """
    result = pd.DataFrame(
        np.nan, index=simple_returns.index, columns=simple_returns.columns,
        dtype=np.float64,
    )
    for col in simple_returns.columns:
        series = simple_returns[col]
        valid = series.notna()
        if not valid.any():
            continue
        first_idx = valid.idxmax()
        growth = (1.0 + series.loc[first_idx:]).cumprod()
        result.loc[first_idx:, col] = initial * growth
    return result


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from a price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Rows are dates, columns are asset names.  NaN is preserved.

    Returns
    -------
    pd.DataFrame
        Simple returns (first row per column is NaN).
    """
    return prices.pct_change()
