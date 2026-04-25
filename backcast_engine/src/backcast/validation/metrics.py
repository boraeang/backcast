"""Validation metrics for backcast output.

Every metric works with either NumPy arrays or pandas objects.  Array inputs
are broadcast along ``axis=0`` (time), so a ``(T, N)`` input produces a
length-``N`` output when ``axis=0``.

All NaN-safe reductions use ``np.nanmean`` / ``np.nanvar`` so holdout
residuals with some missing cells still yield sensible aggregates.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_array(x: ArrayLike) -> np.ndarray:
    """Return a float64 ndarray view without copying when possible."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return np.asarray(x.to_numpy(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def rmse(actual: ArrayLike, predicted: ArrayLike, axis: int = 0) -> np.ndarray:
    """Root mean squared error along *axis*.

    Parameters
    ----------
    actual, predicted : array-like
        Must broadcast to the same shape.
    axis : int
        Axis along which to average (default 0 — returns one RMSE per asset).

    Returns
    -------
    np.ndarray
    """
    a = _as_array(actual)
    p = _as_array(predicted)
    return np.sqrt(np.nanmean((a - p) ** 2, axis=axis))


def mae(actual: ArrayLike, predicted: ArrayLike, axis: int = 0) -> np.ndarray:
    """Mean absolute error along *axis*."""
    a = _as_array(actual)
    p = _as_array(predicted)
    return np.nanmean(np.abs(a - p), axis=axis)


def correlation_error(
    actual_corr: np.ndarray, predicted_corr: np.ndarray
) -> float:
    """Frobenius norm ``‖C_pred - C_actual‖_F``.

    Parameters
    ----------
    actual_corr, predicted_corr : np.ndarray
        Symmetric correlation matrices of identical shape.
    """
    return float(np.linalg.norm(predicted_corr - actual_corr, "fro"))


def vol_ratio(
    actual_vol: ArrayLike, predicted_vol: ArrayLike
) -> np.ndarray:
    """Ratio of predicted to actual volatility.  A value of 1.0 is perfect."""
    return _as_array(predicted_vol) / _as_array(actual_vol)


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------

def ks_test_per_asset(
    actual: ArrayLike, predicted: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample Kolmogorov–Smirnov test, one per column.

    Parameters
    ----------
    actual, predicted : array-like, shape (T, N) or (T,)
        Samples (may have different T per side).

    Returns
    -------
    statistic : np.ndarray, shape (N,)
    p_value : np.ndarray, shape (N,)
    """
    a = _as_array(actual)
    p = _as_array(predicted)
    if a.ndim == 1:
        a = a[:, None]
    if p.ndim == 1:
        p = p[:, None]
    n_cols = a.shape[1]
    stat = np.empty(n_cols)
    pval = np.empty(n_cols)
    for j in range(n_cols):
        aj = a[:, j]
        pj = p[:, j]
        aj = aj[~np.isnan(aj)]
        pj = pj[~np.isnan(pj)]
        res = ks_2samp(aj, pj)
        stat[j] = res.statistic
        pval[j] = res.pvalue
    return stat, pval


def pit_values(
    actual: ArrayLike,
    predicted_mean: ArrayLike,
    predicted_std: ArrayLike,
) -> np.ndarray:
    """Probability-integral-transform values for a normal predictive distribution.

    If the predictive distribution is correctly specified, the returned values
    are ``~Uniform(0, 1)``.

    Parameters
    ----------
    actual : array-like, shape (T, N)
    predicted_mean : array-like, shape (T, N) or (N,)
    predicted_std : array-like, shape (T, N) or (N,)

    Returns
    -------
    np.ndarray, shape (T, N)
    """
    a = _as_array(actual)
    m = _as_array(predicted_mean)
    s = _as_array(predicted_std)
    # Broadcasting: per-asset std is allowed to be 1-D
    return norm.cdf(a, loc=m, scale=s)


def pit_histogram(
    actual: ArrayLike,
    predicted_mean: ArrayLike,
    predicted_std: ArrayLike,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PIT values and histogram counts.

    Returns
    -------
    pit : np.ndarray
        Flattened PIT values.
    counts : np.ndarray, shape (bins,)
    edges : np.ndarray, shape (bins+1,)
    """
    pit = pit_values(actual, predicted_mean, predicted_std)
    flat = pit.ravel()
    flat = flat[~np.isnan(flat)]
    counts, edges = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    return flat, counts, edges


def coverage_rate(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """Fraction of *actual* values falling in ``[lower, upper]``."""
    a = _as_array(actual)
    lo = _as_array(lower)
    hi = _as_array(upper)
    mask = ~np.isnan(a)
    covered = (a >= lo) & (a <= hi) & mask
    return float(covered.sum() / max(mask.sum(), 1))


def coverage_rate_per_asset(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> np.ndarray:
    """Per-column coverage rate, ``shape (N,)``."""
    a = _as_array(actual)
    lo = _as_array(lower)
    hi = _as_array(upper)
    mask = ~np.isnan(a)
    covered = (a >= lo) & (a <= hi) & mask
    denom = mask.sum(axis=0)
    denom = np.where(denom == 0, 1, denom)
    return covered.sum(axis=0) / denom


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

def tail_dependence_coeff(
    returns_1: ArrayLike,
    returns_2: ArrayLike,
    quantile: float = 0.05,
    tail: str = "lower",
) -> float:
    """Empirical tail-dependence coefficient.

    For the lower tail::

        λ_L(q) ≈ #(X_1 ≤ F_1^{-1}(q) and X_2 ≤ F_2^{-1}(q)) / #(X_1 ≤ F_1^{-1}(q))

    Upper-tail version uses the ``(1 - q)``-quantile instead.

    Parameters
    ----------
    returns_1, returns_2 : array-like, shape (T,)
    quantile : float
        Tail quantile in (0, 1).  Small values (0.01–0.05) probe the tails.
    tail : {'lower', 'upper'}
        Which tail to probe.

    Returns
    -------
    float
        Empirical tail-dependence coefficient in [0, 1].
    """
    x1 = _as_array(returns_1).ravel()
    x2 = _as_array(returns_2).ravel()
    mask = ~(np.isnan(x1) | np.isnan(x2))
    x1, x2 = x1[mask], x2[mask]
    if len(x1) == 0:
        return float("nan")

    if tail == "lower":
        q1 = np.quantile(x1, quantile)
        q2 = np.quantile(x2, quantile)
        both = np.sum((x1 <= q1) & (x2 <= q2))
        marg = np.sum(x1 <= q1)
    elif tail == "upper":
        q1 = np.quantile(x1, 1.0 - quantile)
        q2 = np.quantile(x2, 1.0 - quantile)
        both = np.sum((x1 >= q1) & (x2 >= q2))
        marg = np.sum(x1 >= q1)
    else:
        raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")

    if marg == 0:
        return float("nan")
    return float(both / marg)
