"""Walk-forward holdout validation on the overlap period.

Procedure
---------
Within the overlap window (where every asset is observed), mask the short
assets over a sequence of consecutive ``holdout_days``-long windows.  For each
window:

1. Build the masked returns matrix — long assets always observed, short
   assets missing for the window's rows **and** for their original leading
   backcast block.
2. Refit the EM model on the masked matrix.
3. Fill the missing cells with their conditional means.
4. Collect actual vs predicted values, prediction intervals, and residuals.

Metrics are aggregated per asset and per window.  A 95 % prediction interval
comes from ``μ_cond ± z · sqrt(diag(Σ_{M|O}))`` where Σ_{M|O} is taken from
the refitted EM's conditional params.

This module does not (yet) parallelise across windows — n_windows is small
(default 3) so the sequential cost is fine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backcast.data.loader import BackcastDataset
from backcast.imputation.single_impute import impute_missing_values
from backcast.models.em_stambaugh import em_stambaugh
from backcast.validation import metrics as M
from backcast.validation.diagnostics import summarise_residual_diagnostics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class HoldoutWindow:
    """Results for one walk-forward window.

    Attributes
    ----------
    window_idx : int
        0-based window index.
    start_date, end_date : pd.Timestamp
        Inclusive date range of the held-out rows.
    n_rows : int
    actual : pd.DataFrame
        True short-asset returns over the window.
    predicted : pd.DataFrame
        Conditional means from the refitted EM.
    lower, upper : pd.DataFrame
        Per-row prediction bounds (same shape as *predicted*).
    cond_std : np.ndarray, shape (n_short,)
        ``sqrt(diag(Σ_{M|O}))`` from the refitted EM.
    em_n_iter : int
        Iterations used by the EM fit.
    em_converged : bool
    per_asset : pd.DataFrame
        One row per short asset with columns:
        ``rmse``, ``mae``, ``vol_actual``, ``vol_predicted``, ``vol_ratio``,
        ``coverage``, ``ks_stat``, ``ks_pvalue``.
    correlation_error : float
        Frobenius norm of the short-asset correlation-matrix discrepancy
        between actual and predicted windows.
    coverage : float
        Overall fraction of (row, asset) cells inside ``[lower, upper]``.
    """

    window_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_rows: int
    actual: pd.DataFrame
    predicted: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    cond_std: np.ndarray
    em_n_iter: int
    em_converged: bool
    per_asset: pd.DataFrame
    correlation_error: float
    coverage: float


@dataclass
class HoldoutReport:
    """Aggregated walk-forward holdout validation results.

    Attributes
    ----------
    short_assets : list[str]
    windows : list[HoldoutWindow]
    per_asset_mean : pd.DataFrame
        Per-asset metrics averaged across windows.
    overall_coverage : float
    overall_correlation_error : float
    residual_diagnostics : pd.DataFrame
        JB, Ljung-Box, skew, kurtosis on stacked residuals per asset.
    config : dict
        Echoed input arguments for reproducibility.
    """

    short_assets: list[str]
    windows: list[HoldoutWindow]
    per_asset_mean: pd.DataFrame
    overall_coverage: float
    overall_correlation_error: float
    residual_diagnostics: pd.DataFrame
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _compute_window_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Per-asset and aggregate metrics for one window."""
    asset_names = list(actual.columns)
    a = actual.to_numpy()
    p = predicted.to_numpy()
    lo = lower.to_numpy()
    hi = upper.to_numpy()

    rmse_v = M.rmse(a, p, axis=0)
    mae_v = M.mae(a, p, axis=0)
    vol_a = np.nanstd(a, axis=0, ddof=1)
    vol_p = np.nanstd(p, axis=0, ddof=1)
    vol_r = vol_p / vol_a
    ks_stat, ks_p = M.ks_test_per_asset(a, p)
    cov_per = M.coverage_rate_per_asset(a, lo, hi)

    per_asset = pd.DataFrame(
        {
            "rmse": rmse_v,
            "mae": mae_v,
            "vol_actual": vol_a,
            "vol_predicted": vol_p,
            "vol_ratio": vol_r,
            "coverage": cov_per,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
        },
        index=asset_names,
    )

    # Correlation matrices (only meaningful if n_short > 1)
    if actual.shape[1] > 1:
        corr_a = pd.DataFrame(a, columns=asset_names).corr().to_numpy()
        corr_p = pd.DataFrame(p, columns=asset_names).corr().to_numpy()
        corr_err = M.correlation_error(corr_a, corr_p)
    else:
        corr_err = 0.0

    overall_cov = M.coverage_rate(a, lo, hi)
    return per_asset, float(corr_err), float(overall_cov)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_holdout_validation(
    dataset: BackcastDataset,
    *,
    holdout_days: int = 504,
    n_windows: int = 3,
    coverage_level: float = 0.95,
    em_max_iter: int = 500,
    em_tolerance: float = 1e-8,
) -> HoldoutReport:
    """Walk-forward holdout validation of the EM backcast.

    Parameters
    ----------
    dataset : BackcastDataset
        Must have at least ``n_windows * holdout_days`` rows of overlap.
    holdout_days : int
        Length of each holdout window.
    n_windows : int
        Number of consecutive windows to evaluate.
    coverage_level : float
        Nominal coverage of the prediction intervals (e.g. 0.95).
    em_max_iter, em_tolerance : int, float
        EM refit hyperparameters.

    Returns
    -------
    HoldoutReport

    Raises
    ------
    ValueError
        If the overlap period has fewer than ``n_windows * holdout_days`` rows
        or there are no short-history assets.
    """
    if not dataset.short_assets:
        raise ValueError("dataset has no short-history assets — nothing to validate")

    overlap = dataset.overlap_matrix
    T_overlap = len(overlap)
    required = n_windows * holdout_days
    if T_overlap < required:
        raise ValueError(
            f"Overlap has only {T_overlap} rows, need ≥ "
            f"{required} = {n_windows} × {holdout_days}"
        )

    asset_order = list(dataset.returns_full.columns)
    short_assets = list(dataset.short_assets)
    short_col_positions = np.asarray(
        [asset_order.index(name) for name in short_assets], dtype=np.int64,
    )
    z = float(norm.ppf(0.5 + coverage_level / 2.0))

    windows: list[HoldoutWindow] = []
    all_residuals: list[pd.DataFrame] = []

    for w in range(n_windows):
        lo_i = w * holdout_days
        hi_i = lo_i + holdout_days
        window_dates = overlap.index[lo_i:hi_i]
        start_date, end_date = window_dates[0], window_dates[-1]

        # Mask short assets over the window
        masked = dataset.returns_full.copy()
        masked.loc[window_dates, short_assets] = np.nan

        logger.info(
            "Holdout window %d / %d: %s → %s (%d rows)",
            w + 1, n_windows, start_date.date(), end_date.date(), len(window_dates),
        )

        em = em_stambaugh(
            masked,
            max_iter=em_max_iter,
            tolerance=em_tolerance,
            track_loglikelihood=False,
        )

        filled = impute_missing_values(masked, em.mu, em.sigma)

        predicted = filled.loc[window_dates, short_assets]
        actual = dataset.returns_full.loc[window_dates, short_assets]

        # Conditional std from the refitted EM's short/long partition.
        # Missing columns of em.conditional_params are in CSV order — reorder
        # to match *short_assets* ordering for downstream consumption.
        cond_cov = em.conditional_params.cond_cov
        missing_csv = list(em.conditional_params.missing_cols)
        perm = np.asarray(
            [missing_csv.index(pos) for pos in short_col_positions], dtype=np.int64,
        )
        cond_std_asset = np.sqrt(np.diag(cond_cov))[perm]

        lower_vals = predicted.values - z * cond_std_asset
        upper_vals = predicted.values + z * cond_std_asset
        lower = pd.DataFrame(lower_vals, index=predicted.index, columns=short_assets)
        upper = pd.DataFrame(upper_vals, index=predicted.index, columns=short_assets)

        per_asset, corr_err, overall_cov = _compute_window_metrics(
            actual, predicted, lower, upper,
        )

        windows.append(
            HoldoutWindow(
                window_idx=w,
                start_date=start_date,
                end_date=end_date,
                n_rows=len(window_dates),
                actual=actual,
                predicted=predicted,
                lower=lower,
                upper=upper,
                cond_std=cond_std_asset,
                em_n_iter=em.n_iter,
                em_converged=em.converged,
                per_asset=per_asset,
                correlation_error=corr_err,
                coverage=overall_cov,
            )
        )
        all_residuals.append(actual - predicted)

    # Aggregate
    per_asset_mean = (
        pd.concat([w.per_asset for w in windows], keys=range(len(windows)))
        .groupby(level=1)
        .mean()
        .loc[short_assets]
    )

    stacked_residuals = pd.concat(all_residuals, axis=0)
    residual_diag = summarise_residual_diagnostics(stacked_residuals)

    # Aggregate coverage & correlation-error weighted by rows
    total_cells = sum(w.n_rows for w in windows) * len(short_assets)
    covered_cells = sum(w.coverage * w.n_rows * len(short_assets) for w in windows)
    overall_coverage = covered_cells / total_cells

    overall_corr_err = float(np.mean([w.correlation_error for w in windows]))

    return HoldoutReport(
        short_assets=short_assets,
        windows=windows,
        per_asset_mean=per_asset_mean,
        overall_coverage=float(overall_coverage),
        overall_correlation_error=overall_corr_err,
        residual_diagnostics=residual_diag,
        config={
            "holdout_days": holdout_days,
            "n_windows": n_windows,
            "coverage_level": coverage_level,
            "em_max_iter": em_max_iter,
            "em_tolerance": em_tolerance,
        },
    )
