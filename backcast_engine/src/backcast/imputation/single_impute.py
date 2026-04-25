"""Point-estimate backcasting via conditional means.

Given a fitted :class:`~backcast.models.em_stambaugh.EMResult` (which exposes
the full ``μ``, ``Σ``) and a :class:`~backcast.data.loader.BackcastDataset`,
fill in every missing entry with its conditional expectation given the
observed columns in that row.  For staggered missingness, each distinct
observed/missing pattern receives its own conditional mean formula.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from backcast.data.loader import BackcastDataset
from backcast.models.em_stambaugh import EMResult

logger = logging.getLogger(__name__)


def _fill_rows_conditional(
    R: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Replace each missing entry of ``R`` with its conditional expectation.

    Rows are grouped by missingness pattern so each distinct ``(O, M)`` split
    is solved once.

    Parameters
    ----------
    R : np.ndarray, shape (T, N)
        Returns matrix with NaN for missing entries.  Modified in place.
    mu : np.ndarray, shape (N,)
    sigma : np.ndarray, shape (N, N)

    Returns
    -------
    np.ndarray
        ``R`` itself (modified in-place), no NaNs.
    """
    T, N = R.shape
    nan_mask = np.isnan(R)
    if not nan_mask.any():
        return R

    # Group by pattern (tuple of bool) — same encoding as the EM helper
    groups: dict[bytes, list[int]] = {}
    row_bytes = np.ascontiguousarray(nan_mask).view(np.uint8).reshape(T, -1)
    for t in range(T):
        if not nan_mask[t].any():
            continue   # nothing to fill for fully-observed rows
        groups.setdefault(row_bytes[t].tobytes(), []).append(t)

    for key, rows in groups.items():
        rows_arr = np.asarray(rows, dtype=np.int64)
        pattern = nan_mask[rows_arr[0]]
        obs_cols = np.where(~pattern)[0]
        mis_cols = np.where(pattern)[0]

        if len(obs_cols) == 0:
            # Fully missing row — fall back to marginal mean
            R[np.ix_(rows_arr, mis_cols)] = mu[mis_cols]
            continue

        S_OO = sigma[np.ix_(obs_cols, obs_cols)]
        S_OM = sigma[np.ix_(obs_cols, mis_cols)]
        L, low = cho_factor(S_OO, lower=True)
        beta = cho_solve((L, low), S_OM).T      # (|M|, |O|)
        alpha = mu[mis_cols] - beta @ mu[obs_cols]
        obs_data = R[np.ix_(rows_arr, obs_cols)]
        R[np.ix_(rows_arr, mis_cols)] = alpha + obs_data @ beta.T

    return R


def impute_missing_values(
    returns: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> pd.DataFrame:
    """Fill every NaN in *returns* with the conditional mean given ``(μ, Σ)``.

    Unlike :func:`single_impute`, this helper operates directly on a DataFrame
    and tolerates **arbitrary** NaN patterns (including the non-monotone
    holes introduced by holdout-validation windows).  Rows with no missing
    entries are left untouched.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with missing entries encoded as NaN.
    mu : np.ndarray, shape (N,)
        Mean vector matching the column order of *returns*.
    sigma : np.ndarray, shape (N, N)
        Covariance matrix matching the column order of *returns*.

    Returns
    -------
    pd.DataFrame
        Same index, columns, and shape — no NaNs.
    """
    R = returns.to_numpy(dtype=np.float64, copy=True)
    _fill_rows_conditional(R, mu, sigma)
    return pd.DataFrame(R, index=returns.index, columns=returns.columns)


def single_impute(
    dataset: BackcastDataset,
    em_result: EMResult,
) -> pd.DataFrame:
    """Produce a complete returns matrix using conditional means.

    Parameters
    ----------
    dataset : BackcastDataset
        Input returns (may have staggered missingness).
    em_result : EMResult
        Fitted EM parameters.  ``em_result.asset_order`` must match the
        column order of ``dataset.returns_full``; it is validated.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and column order as ``dataset.returns_full``, with
        every NaN replaced by :math:`E[R_{M,t} | R_{O,t}, \\mu, \\Sigma]`.

    Raises
    ------
    ValueError
        If the asset order of ``em_result`` does not match the dataset columns.
    """
    if list(dataset.returns_full.columns) != em_result.asset_order:
        raise ValueError(
            "EMResult.asset_order does not match dataset columns; "
            "re-run em_stambaugh on the dataset's returns matrix."
        )

    filled = dataset.returns_full.to_numpy(dtype=np.float64, copy=True)
    _fill_rows_conditional(filled, em_result.mu, em_result.sigma)

    if np.isnan(filled).any():
        raise RuntimeError("single_impute left NaN values — this is a bug")

    return pd.DataFrame(
        filled,
        index=dataset.returns_full.index,
        columns=dataset.returns_full.columns,
    )
