"""Residual and covariance diagnostics for backcast output.

These helpers are stateless and return plain dicts / arrays so they compose
cleanly with the holdout module and any reporting layer built on top.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, jarque_bera, norm


def _as_df(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, columns=[f"col{i}" for i in range(arr.shape[1])])


# ---------------------------------------------------------------------------
# Residual normality
# ---------------------------------------------------------------------------

def residual_normality(residuals) -> pd.DataFrame:
    """Jarque–Bera test per column.

    Parameters
    ----------
    residuals : array-like or DataFrame, shape (T, N)

    Returns
    -------
    pd.DataFrame
        Indexed by column name, with ``jb_stat`` and ``jb_pvalue`` columns.
    """
    df = _as_df(residuals)
    out = {}
    for col in df.columns:
        series = df[col].dropna().to_numpy()
        if len(series) < 8:
            out[col] = {"jb_stat": float("nan"), "jb_pvalue": float("nan")}
            continue
        stat, p = jarque_bera(series)
        out[col] = {"jb_stat": float(stat), "jb_pvalue": float(p)}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------
# Residual autocorrelation (Ljung–Box)
# ---------------------------------------------------------------------------

def _ljung_box_pvalue(series: np.ndarray, lag: int) -> float:
    """Ljung–Box Q-statistic p-value at the given *lag*."""
    n = len(series)
    if n <= lag + 1:
        return float("nan")
    centered = series - series.mean()
    acf = np.correlate(centered, centered, mode="full")
    acf = acf[n - 1 :]
    acf = acf / acf[0]
    q_stat = 0.0
    for k in range(1, lag + 1):
        q_stat += (n * (n + 2) * acf[k] ** 2) / (n - k)
    return float(1.0 - chi2.cdf(q_stat, df=lag))


def residual_autocorrelation(residuals, lag: int = 10) -> pd.DataFrame:
    """Ljung–Box test per column.

    Parameters
    ----------
    residuals : array-like or DataFrame, shape (T, N)
    lag : int
        Number of lags included in the Q-statistic.

    Returns
    -------
    pd.DataFrame
        Indexed by column name; contains ``lb_pvalue``.
    """
    df = _as_df(residuals)
    out = {}
    for col in df.columns:
        series = df[col].dropna().to_numpy()
        out[col] = {"lb_pvalue": _ljung_box_pvalue(series, lag)}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------
# Covariance / eigenvalue comparison
# ---------------------------------------------------------------------------

def eigenvalue_comparison(
    sigma_a: np.ndarray, sigma_b: np.ndarray
) -> dict:
    """Compare eigenvalue spectra of two covariance matrices.

    Parameters
    ----------
    sigma_a, sigma_b : np.ndarray, shape (N, N)
        Symmetric covariance matrices.

    Returns
    -------
    dict
        Keys: ``eig_a``, ``eig_b`` (descending), ``spectrum_diff``
        (``eig_b - eig_a``), ``cond_a``, ``cond_b``,
        ``max_abs_diff`` (scalar).
    """
    eig_a = np.sort(np.linalg.eigvalsh(sigma_a))[::-1]
    eig_b = np.sort(np.linalg.eigvalsh(sigma_b))[::-1]
    diff = eig_b - eig_a
    return {
        "eig_a": eig_a,
        "eig_b": eig_b,
        "spectrum_diff": diff,
        "cond_a": float(eig_a[0] / max(eig_a[-1], 1e-30)),
        "cond_b": float(eig_b[0] / max(eig_b[-1], 1e-30)),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def rolling_correlation(
    returns: pd.DataFrame,
    pair: tuple[str, str],
    window: int = 252,
) -> pd.Series:
    """Rolling Pearson correlation between two columns.

    Parameters
    ----------
    returns : pd.DataFrame
    pair : (str, str)
        Column names.
    window : int
        Window length in rows.

    Returns
    -------
    pd.Series
        Rolling correlation with the same index as *returns*; leading
        ``window - 1`` entries are NaN.
    """
    a, b = pair
    return returns[a].rolling(window).corr(returns[b])


# ---------------------------------------------------------------------------
# QQ-plot data (returned, not plotted)
# ---------------------------------------------------------------------------

def qq_plot_data(
    series, dist: str = "norm"
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical vs sample quantiles for a QQ plot.

    Parameters
    ----------
    series : array-like
    dist : {'norm'}
        Only standard normal is supported here.

    Returns
    -------
    theoretical_q : np.ndarray, shape (n,)
    sample_q : np.ndarray, shape (n,)
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    x.sort()
    n = len(x)
    if n == 0:
        return np.empty(0), np.empty(0)
    probs = (np.arange(1, n + 1) - 0.5) / n
    if dist == "norm":
        theo = norm.ppf(probs)
    else:
        raise ValueError(f"unknown dist {dist!r}")
    # Standardise sample to the same scale
    return theo, x


# ---------------------------------------------------------------------------
# High-level helper used by holdout
# ---------------------------------------------------------------------------

def summarise_residual_diagnostics(
    residuals, ljung_box_lag: int = 10
) -> pd.DataFrame:
    """Per-column diagnostics combining normality, autocorrelation, moments.

    Parameters
    ----------
    residuals : array-like or DataFrame
    ljung_box_lag : int

    Returns
    -------
    pd.DataFrame
        Index: column names.  Columns: ``mean``, ``std``, ``skew``,
        ``ex_kurtosis``, ``jb_pvalue``, ``lb_pvalue``.
    """
    df = _as_df(residuals)
    from scipy.stats import kurtosis, skew

    rows: dict[str, dict[str, float]] = {}
    for col in df.columns:
        s = df[col].dropna().to_numpy()
        if len(s) < 3:
            rows[col] = {k: float("nan") for k in (
                "mean", "std", "skew", "ex_kurtosis", "jb_pvalue", "lb_pvalue"
            )}
            continue
        rows[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "skew": float(skew(s)),
            "ex_kurtosis": float(kurtosis(s, fisher=True)),
            "jb_pvalue": float(jarque_bera(s)[1]) if len(s) >= 8 else float("nan"),
            "lb_pvalue": _ljung_box_pvalue(s, ljung_box_lag),
        }
    return pd.DataFrame(rows).T
