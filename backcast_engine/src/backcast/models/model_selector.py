"""Cross-validated model selection for imputation methods.

Given a :class:`BackcastDataset`, evaluate each candidate imputation method
with walk-forward holdout on the overlap period and rank by one of three
criteria: RMSE, coverage-calibration error, or a combined rank score.

Supported methods
-----------------
``unconditional_em``
    Refit Stambaugh EM per window; impute held-out short assets via the
    unconditional conditional mean.

``regime_conditional``
    Fit a Gaussian HMM once on the always-observed long-history assets, then
    for each window estimate per-regime ``(μ^{(k)}, Σ^{(k)})`` on the overlap
    rows **excluding** that window and impute held-out rows with the
    regime-specific conditional distribution.  Prediction intervals use a
    per-row conditional std that varies with the active regime.

The module intentionally does NOT evaluate the Kalman TVP model: per the
spec it is a robustness check rather than a competing imputer.  It can be
wired in by passing a custom callable (see :func:`evaluate_method_cv`).

References
----------
Bishop, C.M. (2006).  *Pattern Recognition and Machine Learning*, §1.3
(cross-validation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from backcast.data.loader import BackcastDataset
from backcast.models.em_stambaugh import em_stambaugh
from backcast.models.regime_hmm import compute_regime_params, fit_regime_hmm

logger = logging.getLogger(__name__)

_SUPPORTED = ("unconditional_em", "regime_conditional")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MethodCVResult:
    """CV performance of a single imputation method.

    Attributes
    ----------
    method : str
    n_windows : int
    nominal_coverage : float
    rmse_per_asset : np.ndarray, shape (n_short,)
        RMSE averaged across windows, per short asset.
    rmse_overall : float
        RMSE pooled across windows and assets.
    coverage_per_asset : np.ndarray, shape (n_short,)
        Mean coverage across windows, per short asset.
    coverage_overall : float
        Pooled coverage.
    coverage_error : float
        ``|coverage_overall - nominal|`` — the calibration gap.
    correlation_error : float
        Mean Frobenius norm of the short-asset correlation-matrix difference
        (actual vs predicted) across windows.
    per_window : list[dict]
        Per-window diagnostics (dates, rmse, coverage, ...).
    """

    method: str
    n_windows: int
    nominal_coverage: float
    rmse_per_asset: np.ndarray
    rmse_overall: float
    coverage_per_asset: np.ndarray
    coverage_overall: float
    coverage_error: float
    correlation_error: float
    per_window: list


@dataclass
class ModelSelectionResult:
    """Output of :func:`select_model_cv`.

    Attributes
    ----------
    criterion : str
        ``'rmse'``, ``'coverage'``, or ``'combined'``.
    candidates : list[str]
    per_method : dict[str, MethodCVResult]
    best_method : str
    ranking : list[str]
        Sorted best → worst according to *criterion*.
    scores : dict[str, float]
        Raw score used for ranking (lower is better).
    nominal_coverage : float
    """

    criterion: str
    candidates: list[str]
    per_method: dict
    best_method: str
    ranking: list[str]
    scores: dict
    nominal_coverage: float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _conditional_block(
    mu: np.ndarray,
    sigma: np.ndarray,
    obs_idx: np.ndarray,
    mis_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(alpha, beta, cond_std)`` for the (obs, mis) partition."""
    S_OO = sigma[np.ix_(obs_idx, obs_idx)]
    S_OM = sigma[np.ix_(obs_idx, mis_idx)]
    S_MM = sigma[np.ix_(mis_idx, mis_idx)]
    L, low = cho_factor(S_OO, lower=True)
    beta = cho_solve((L, low), S_OM).T               # (|M|, |O|)
    alpha = mu[mis_idx] - beta @ mu[obs_idx]
    cond_cov = S_MM - beta @ S_OM
    cond_std = np.sqrt(np.clip(np.diag(cond_cov), 0.0, None))
    return alpha, beta, cond_std


def _aggregate(
    method: str,
    per_window: list,
    nominal: float,
) -> MethodCVResult:
    """Collapse a list of per-window diagnostics into a :class:`MethodCVResult`."""
    if not per_window:
        raise ValueError("per_window is empty")
    rmse_mat = np.stack([w["rmse_per_asset"] for w in per_window])  # (W, n_short)
    cov_mat = np.stack([w["coverage_per_asset"] for w in per_window])  # (W, n_short)

    total_sq_err = sum(w["sum_sq_err"] for w in per_window)
    total_cells = sum(w["n_cells"] for w in per_window)
    total_covered = sum(w["n_covered"] for w in per_window)

    rmse_overall = float(np.sqrt(total_sq_err / total_cells))
    coverage_overall = float(total_covered / total_cells)
    coverage_error = float(abs(coverage_overall - nominal))
    correlation_error = float(np.mean([w["correlation_error"] for w in per_window]))

    return MethodCVResult(
        method=method,
        n_windows=len(per_window),
        nominal_coverage=nominal,
        rmse_per_asset=rmse_mat.mean(axis=0),
        rmse_overall=rmse_overall,
        coverage_per_asset=cov_mat.mean(axis=0),
        coverage_overall=coverage_overall,
        coverage_error=coverage_error,
        correlation_error=correlation_error,
        per_window=per_window,
    )


# ---------------------------------------------------------------------------
# Per-method CV loops
# ---------------------------------------------------------------------------

def _cv_unconditional_em(
    dataset: BackcastDataset,
    *,
    holdout_days: int,
    n_windows: int,
    coverage_level: float,
    em_max_iter: int,
    em_tolerance: float,
) -> MethodCVResult:
    z = float(norm.ppf(0.5 + coverage_level / 2.0))
    overlap = dataset.overlap_matrix
    asset_order = list(dataset.returns_full.columns)
    long_idx = np.asarray(
        [asset_order.index(a) for a in dataset.long_assets], dtype=np.int64,
    )
    short_idx = np.asarray(
        [asset_order.index(a) for a in dataset.short_assets], dtype=np.int64,
    )
    short_names = list(dataset.short_assets)

    per_window: list[dict] = []
    for w in range(n_windows):
        lo, hi = w * holdout_days, w * holdout_days + holdout_days
        window_dates = overlap.index[lo:hi]
        masked = dataset.returns_full.copy()
        masked.loc[window_dates, short_names] = np.nan

        em = em_stambaugh(
            masked, max_iter=em_max_iter, tolerance=em_tolerance,
            track_loglikelihood=False,
        )
        alpha, beta, cond_std = _conditional_block(em.mu, em.sigma, long_idx, short_idx)

        obs_data = overlap.iloc[lo:hi][dataset.long_assets].to_numpy()
        predicted = alpha + obs_data @ beta.T
        actual = overlap.iloc[lo:hi][short_names].to_numpy()
        # Per-row cond std is the same for every row (no regime structure)
        per_row_std = np.tile(cond_std, (predicted.shape[0], 1))
        lower = predicted - z * per_row_std
        upper = predicted + z * per_row_std

        sq_err = (predicted - actual) ** 2
        covered = ((actual >= lower) & (actual <= upper))
        corr_err = _corr_err(actual, predicted)

        per_window.append({
            "window": w,
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "n_rows": len(window_dates),
            "rmse_per_asset": np.sqrt(sq_err.mean(axis=0)),
            "coverage_per_asset": covered.mean(axis=0),
            "sum_sq_err": float(sq_err.sum()),
            "n_cells": int(covered.size),
            "n_covered": int(covered.sum()),
            "correlation_error": corr_err,
            "em_n_iter": em.n_iter,
        })
    return _aggregate("unconditional_em", per_window, coverage_level)


def _cv_regime_conditional(
    dataset: BackcastDataset,
    *,
    holdout_days: int,
    n_windows: int,
    coverage_level: float,
    hmm_n_regimes: int,
    hmm_max_iter: int,
    hmm_tolerance: float,
    hmm_seed: int,
) -> MethodCVResult:
    z = float(norm.ppf(0.5 + coverage_level / 2.0))
    overlap = dataset.overlap_matrix
    T_overlap = len(overlap)
    asset_order = list(dataset.returns_full.columns)
    long_idx = np.asarray(
        [asset_order.index(a) for a in dataset.long_assets], dtype=np.int64,
    )
    short_idx = np.asarray(
        [asset_order.index(a) for a in dataset.short_assets], dtype=np.int64,
    )
    short_names = list(dataset.short_assets)

    # HMM is fit on the long-history panel — invariant to window choice
    # because long assets are never masked.
    hmm = fit_regime_hmm(
        dataset.returns_full[dataset.long_assets],
        n_regimes=hmm_n_regimes,
        max_iter=hmm_max_iter,
        tolerance=hmm_tolerance,
        seed=hmm_seed,
    )
    overlap_labels = hmm.regime_labels[-T_overlap:]

    per_window: list[dict] = []
    for w in range(n_windows):
        lo, hi = w * holdout_days, w * holdout_days + holdout_days
        window_dates = overlap.index[lo:hi]

        # Training rows = overlap rows outside the window
        train_mask = np.ones(T_overlap, dtype=bool)
        train_mask[lo:hi] = False
        train_returns = overlap.iloc[train_mask]
        train_labels = overlap_labels[train_mask]
        regime_params = compute_regime_params(train_returns, train_labels)

        window_labels = overlap_labels[lo:hi]
        obs_data_all = overlap.iloc[lo:hi][dataset.long_assets].to_numpy()
        actual = overlap.iloc[lo:hi][short_names].to_numpy()

        predicted = np.zeros_like(actual)
        per_row_std = np.zeros_like(actual)

        any_unfitted = False
        for k, params in regime_params.items():
            mask = (window_labels == k)
            if not mask.any():
                continue
            alpha_k, beta_k, cond_std_k = _conditional_block(
                params["mu"], params["sigma"], long_idx, short_idx,
            )
            predicted[mask] = alpha_k + obs_data_all[mask] @ beta_k.T
            per_row_std[mask] = cond_std_k

        # Rows whose regime has no params (e.g., only a handful of points)
        # fall back to the unconditional pooled estimate
        unfitted = ~np.isin(window_labels, list(regime_params.keys()))
        if unfitted.any():
            any_unfitted = True
            pooled_mu = train_returns.mean().to_numpy()
            pooled_sigma = train_returns.cov().to_numpy()
            alpha_p, beta_p, cond_std_p = _conditional_block(
                pooled_mu, pooled_sigma, long_idx, short_idx,
            )
            predicted[unfitted] = alpha_p + obs_data_all[unfitted] @ beta_p.T
            per_row_std[unfitted] = cond_std_p
            logger.debug(
                "regime-conditional CV: window %d has %d rows with unfitted regime",
                w, int(unfitted.sum()),
            )

        lower = predicted - z * per_row_std
        upper = predicted + z * per_row_std
        sq_err = (predicted - actual) ** 2
        covered = ((actual >= lower) & (actual <= upper))
        corr_err = _corr_err(actual, predicted)

        per_window.append({
            "window": w,
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "n_rows": len(window_dates),
            "rmse_per_asset": np.sqrt(sq_err.mean(axis=0)),
            "coverage_per_asset": covered.mean(axis=0),
            "sum_sq_err": float(sq_err.sum()),
            "n_cells": int(covered.size),
            "n_covered": int(covered.sum()),
            "correlation_error": corr_err,
            "fallback_rows": int(unfitted.sum()) if any_unfitted else 0,
        })
    return _aggregate("regime_conditional", per_window, coverage_level)


def _corr_err(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Frobenius norm of correlation-matrix difference; 0 for a single asset."""
    if actual.shape[1] < 2:
        return 0.0
    a = pd.DataFrame(actual).corr().to_numpy()
    p = pd.DataFrame(predicted).corr().to_numpy()
    return float(np.linalg.norm(a - p, "fro"))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def evaluate_method_cv(
    dataset: BackcastDataset,
    method: str,
    *,
    holdout_days: int = 504,
    n_windows: int = 3,
    coverage_level: float = 0.95,
    em_max_iter: int = 500,
    em_tolerance: float = 1e-8,
    hmm_n_regimes: int = 2,
    hmm_max_iter: int = 200,
    hmm_tolerance: float = 1e-4,
    hmm_seed: int = 0,
) -> MethodCVResult:
    """Walk-forward holdout evaluation of a single imputation method.

    Parameters
    ----------
    dataset : BackcastDataset
    method : {'unconditional_em', 'regime_conditional'}
    holdout_days : int
    n_windows : int
    coverage_level : float
        Nominal coverage of the prediction intervals (e.g. 0.95).
    em_* : float/int
        EM hyperparameters used when refitting per window.
    hmm_* : float/int
        HMM hyperparameters (only used for ``regime_conditional``).

    Returns
    -------
    MethodCVResult

    Raises
    ------
    ValueError
        For unknown *method* or insufficient overlap.
    """
    if method not in _SUPPORTED:
        raise ValueError(
            f"unknown method {method!r}; supported = {_SUPPORTED}"
        )
    if not dataset.short_assets:
        raise ValueError("dataset has no short-history assets")
    required = n_windows * holdout_days
    if len(dataset.overlap_matrix) < required:
        raise ValueError(
            f"overlap has only {len(dataset.overlap_matrix)} rows; "
            f"need at least {required} = {n_windows} × {holdout_days}"
        )

    if method == "unconditional_em":
        return _cv_unconditional_em(
            dataset,
            holdout_days=holdout_days, n_windows=n_windows,
            coverage_level=coverage_level,
            em_max_iter=em_max_iter, em_tolerance=em_tolerance,
        )
    return _cv_regime_conditional(
        dataset,
        holdout_days=holdout_days, n_windows=n_windows,
        coverage_level=coverage_level,
        hmm_n_regimes=hmm_n_regimes,
        hmm_max_iter=hmm_max_iter, hmm_tolerance=hmm_tolerance,
        hmm_seed=hmm_seed,
    )


def _rank_methods(
    per_method: dict,
    criterion: str,
) -> tuple[list[str], dict]:
    """Return ``(ranking, scores)``.  Lower score is always better."""
    names = list(per_method.keys())

    if criterion == "rmse":
        scores = {n: per_method[n].rmse_overall for n in names}
    elif criterion == "coverage":
        scores = {n: per_method[n].coverage_error for n in names}
    elif criterion == "combined":
        # Rank by each sub-criterion (1 = best) and sum.
        rmse_vals = np.array([per_method[n].rmse_overall for n in names])
        cov_vals = np.array([per_method[n].coverage_error for n in names])
        rmse_rank = rmse_vals.argsort().argsort() + 1   # average-rank; ties are rare here
        cov_rank = cov_vals.argsort().argsort() + 1
        combined = rmse_rank + cov_rank
        scores = {n: float(s) for n, s in zip(names, combined)}
    else:
        raise ValueError(
            f"criterion must be 'rmse', 'coverage', or 'combined'; got {criterion!r}"
        )
    ranking = sorted(names, key=lambda n: scores[n])
    return ranking, scores


def select_model_cv(
    dataset: BackcastDataset,
    *,
    candidates: tuple[str, ...] = ("unconditional_em", "regime_conditional"),
    criterion: str = "combined",
    holdout_days: int = 504,
    n_windows: int = 3,
    coverage_level: float = 0.95,
    em_max_iter: int = 500,
    em_tolerance: float = 1e-8,
    hmm_n_regimes: int = 2,
    hmm_max_iter: int = 200,
    hmm_tolerance: float = 1e-4,
    hmm_seed: int = 0,
) -> ModelSelectionResult:
    """Cross-validated selection across imputation methods.

    Evaluates every candidate method using walk-forward holdout, then ranks
    them by *criterion* and returns the winner.

    Parameters
    ----------
    dataset : BackcastDataset
    candidates : tuple of str
        Subset of ``('unconditional_em', 'regime_conditional')``.
    criterion : {'rmse', 'coverage', 'combined'}
        ``'rmse'``      — minimum pooled RMSE wins.
        ``'coverage'``  — smallest ``|overall_coverage − nominal|`` wins.
        ``'combined'``  — sum of RMSE-rank and coverage-rank wins.
    holdout_days, n_windows, coverage_level : validation window controls.
    em_*, hmm_* : per-method hyperparameters.

    Returns
    -------
    ModelSelectionResult

    Raises
    ------
    ValueError
        If any candidate is not supported, on an unknown *criterion*, or on
        insufficient overlap.
    """
    bad = [c for c in candidates if c not in _SUPPORTED]
    if bad:
        raise ValueError(
            f"unsupported candidates {bad}; supported = {_SUPPORTED}"
        )
    if len(candidates) == 0:
        raise ValueError("candidates is empty")

    per_method: dict[str, MethodCVResult] = {}
    for m in candidates:
        logger.info("Evaluating method %s via walk-forward CV ...", m)
        per_method[m] = evaluate_method_cv(
            dataset, method=m,
            holdout_days=holdout_days, n_windows=n_windows,
            coverage_level=coverage_level,
            em_max_iter=em_max_iter, em_tolerance=em_tolerance,
            hmm_n_regimes=hmm_n_regimes, hmm_max_iter=hmm_max_iter,
            hmm_tolerance=hmm_tolerance, hmm_seed=hmm_seed,
        )
        r = per_method[m]
        logger.info(
            "  %s: rmse=%.5f  coverage=%.4f  (|Δ|=%.4f)",
            m, r.rmse_overall, r.coverage_overall, r.coverage_error,
        )

    ranking, scores = _rank_methods(per_method, criterion)
    return ModelSelectionResult(
        criterion=criterion,
        candidates=list(candidates),
        per_method=per_method,
        best_method=ranking[0],
        ranking=ranking,
        scores=scores,
        nominal_coverage=coverage_level,
    )
