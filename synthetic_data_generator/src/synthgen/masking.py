"""Apply monotone missingness patterns to complete return DataFrames."""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MaskingMetadata(NamedTuple):
    """Summary of the missingness pattern applied to a DataFrame.

    Attributes
    ----------
    short_asset_start_indices : dict[str, int]
        Mapping from short-asset name to first valid row index (0-based).
    n_missing_per_asset : dict[str, int]
        Number of NaN values introduced per short asset.
    total_missing : int
        Total number of NaN cells introduced.
    missing_fraction : float
        Fraction of cells that are NaN in the masked DataFrame.
    is_monotone : bool
        True when the pattern is verified to be monotone (no mid-series gaps).
    """

    short_asset_start_indices: dict[str, int]
    n_missing_per_asset: dict[str, int]
    total_missing: int
    missing_fraction: float
    is_monotone: bool


def apply_masking(
    returns_complete: pd.DataFrame,
    short_asset_start_indices: dict[str, int],
) -> tuple[pd.DataFrame, MaskingMetadata]:
    """Apply monotone missingness to a complete returns DataFrame.

    For each short asset, all rows before its start index are set to NaN.
    Long assets and rows at/after the start index are unchanged.

    Parameters
    ----------
    returns_complete : pd.DataFrame
        Complete return matrix with no missing values.
        Rows are dates; columns are asset names.
    short_asset_start_indices : dict[str, int]
        Mapping from short-asset column name to the first valid row index
        (0-based).  Assets not in this dict are treated as long assets and
        left unchanged.

    Returns
    -------
    tuple[pd.DataFrame, MaskingMetadata]
        (masked DataFrame, metadata describing the missingness pattern)

    Raises
    ------
    KeyError
        If any key in *short_asset_start_indices* is not a column of
        *returns_complete*.
    ValueError
        If any start index is out of the valid range (0, n_rows).

    Notes
    -----
    Validate monotone missingness: for each column the NaN block must be a
    leading prefix — no NaN values may appear after the first non-NaN value.
    """
    n_rows, n_cols = returns_complete.shape
    missing_cols = [c for c in short_asset_start_indices if c not in returns_complete.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    for name, idx in short_asset_start_indices.items():
        if not (0 < idx < n_rows):
            raise ValueError(
                f"start_index for '{name}' is {idx}, must be in (0, {n_rows})"
            )

    masked = returns_complete.copy()
    n_missing_per_asset: dict[str, int] = {}

    for name, start_idx in short_asset_start_indices.items():
        masked.iloc[:start_idx, masked.columns.get_loc(name)] = np.nan
        n_missing_per_asset[name] = start_idx

    # Verify monotone missingness: no NaN after first non-NaN per column
    is_monotone = _check_monotone(masked)
    if not is_monotone:
        logger.warning("Missingness pattern is NOT monotone — unexpected mid-series gaps detected.")

    total_missing = sum(n_missing_per_asset.values())
    missing_fraction = total_missing / (n_rows * n_cols)

    meta = MaskingMetadata(
        short_asset_start_indices=dict(short_asset_start_indices),
        n_missing_per_asset=n_missing_per_asset,
        total_missing=total_missing,
        missing_fraction=missing_fraction,
        is_monotone=is_monotone,
    )
    logger.info(
        "Masking applied: %d NaN values (%.1f%% of cells), monotone=%s",
        total_missing,
        100.0 * missing_fraction,
        is_monotone,
    )
    return masked, meta


def _check_monotone(df: pd.DataFrame) -> bool:
    """Return True if every column has at most a leading block of NaNs."""
    for col in df.columns:
        series = df[col].values
        nan_mask = np.isnan(series.astype(float))
        if not nan_mask.any():
            continue
        # Find first non-NaN
        first_valid = int(np.argmin(nan_mask))
        # Any NaN after first_valid → not monotone
        if nan_mask[first_valid:].any():
            return False
    return True
