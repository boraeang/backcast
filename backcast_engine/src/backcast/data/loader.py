"""CSV loader and dataset container for the backcast engine.

Responsibilities
----------------
- Load daily simple returns from a CSV with a date column.
- Validate the data (warn on suspicious magnitudes, reject duplicate dates,
  reject non-monotone missingness).
- Detect the monotone missingness pattern, partition assets into long / short
  history, and expose overlap + backcast slices as a :class:`BackcastDataset`.

Supports staggered short-history starts: short assets are ordered by start
index, and ``overlap_start`` is the date on which the last-starting short asset
first appears.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backcast.exceptions import BackcastDataError

logger = logging.getLogger(__name__)

_DEFAULT_DATE_COLUMN = "date"
_RETURN_WARN_THRESHOLD = 0.5  # any |r| > 0.5 triggers a suspicious-data warning


@dataclass
class BackcastDataset:
    """Container for a returns matrix partitioned into long / short history.

    Attributes
    ----------
    returns_full : pd.DataFrame
        Full returns matrix (rows = dates, columns = assets).  NaN marks
        missing rows for short-history assets.
    long_assets : list[str]
        Assets with full history (first valid row index == 0).
    short_assets : list[str]
        Short-history assets, ordered by start index (earliest first).
    short_start_indices : dict[str, int]
        Mapping from short-asset name to first valid row index (0-based).
    overlap_start : pd.Timestamp or None
        First date on which every asset is observed.  None when there are no
        short assets (no imputation needed).
    overlap_end : pd.Timestamp or None
        Last date on which every asset is observed (= last date in the file).
    backcast_start : pd.Timestamp or None
        First date to impute (= first date in the file).  None when nothing
        needs to be imputed.
    backcast_end : pd.Timestamp or None
        Last date to impute (= one row before ``overlap_start``).
    overlap_matrix : pd.DataFrame
        Returns of ALL assets over the overlap window.
    long_history_matrix : pd.DataFrame
        Returns of the long-history assets over the backcast window.
    """

    returns_full: pd.DataFrame
    long_assets: list[str]
    short_assets: list[str]
    short_start_indices: dict[str, int]
    overlap_start: Optional[pd.Timestamp]
    overlap_end: Optional[pd.Timestamp]
    backcast_start: Optional[pd.Timestamp]
    backcast_end: Optional[pd.Timestamp]
    overlap_matrix: pd.DataFrame
    long_history_matrix: pd.DataFrame

    # Derived, cached for downstream convenience
    @property
    def n_long(self) -> int:
        return len(self.long_assets)

    @property
    def n_short(self) -> int:
        return len(self.short_assets)

    @property
    def n_total(self) -> int:
        return self.n_long + self.n_short

    @property
    def asset_names(self) -> list[str]:
        """Assets in storage order (long first, then short by start index)."""
        return list(self.long_assets) + list(self.short_assets)

    @property
    def overlap_length(self) -> int:
        return len(self.overlap_matrix)

    @property
    def backcast_length(self) -> int:
        return len(self.long_history_matrix)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_returns_csv(
    path: str | Path,
    date_column: str = _DEFAULT_DATE_COLUMN,
    return_warn_threshold: float = _RETURN_WARN_THRESHOLD,
) -> pd.DataFrame:
    """Load a CSV of daily simple returns.

    Parameters
    ----------
    path : str or Path
        Location of the CSV file.
    date_column : str
        Name of the date column (defaults to ``"date"``).
    return_warn_threshold : float
        Any single observation with ``|r| > threshold`` triggers a warning —
        values that large usually mean the file holds prices or unusual units.

    Returns
    -------
    pd.DataFrame
        Returns matrix with a sorted DatetimeIndex.

    Raises
    ------
    BackcastDataError
        On duplicate dates.
    """
    df = pd.read_csv(path, index_col=date_column, parse_dates=True)
    df = df.astype(np.float64)

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        raise BackcastDataError(
            f"Duplicate dates in {path}: {sorted({str(d) for d in dupes})[:5]} …"
        )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    df.index.name = date_column

    max_abs = float(df.abs().max().max())
    if np.isfinite(max_abs) and max_abs > return_warn_threshold:
        logger.warning(
            "Max |value| = %.3f > %.2f in %s — file may contain prices "
            "or non-return quantities.",
            max_abs, return_warn_threshold, path,
        )
    logger.info("Loaded %s: %d rows × %d cols", path, df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Monotone-missingness detection
# ---------------------------------------------------------------------------

def detect_start_indices(returns: pd.DataFrame) -> dict[str, int]:
    """Return the first-valid row index for each column; 0 for full-history.

    Raises
    ------
    BackcastDataError
        If any column has a NaN after its first non-NaN observation
        (mid-series gap), which violates the monotone missingness assumption.
    """
    starts: dict[str, int] = {}
    for col in returns.columns:
        series = returns[col].to_numpy(dtype=np.float64)
        nan_mask = np.isnan(series)
        if not nan_mask.any():
            starts[col] = 0
            continue
        if nan_mask.all():
            raise BackcastDataError(f"Column {col!r} is entirely NaN")
        # First non-NaN index
        first_valid = int(np.argmin(nan_mask))
        # Every value from first_valid onward must be non-NaN
        if nan_mask[first_valid:].any():
            bad_rows = np.where(nan_mask[first_valid:])[0][:5] + first_valid
            raise BackcastDataError(
                f"Non-monotone missingness in column {col!r}: "
                f"NaN found at rows {list(bad_rows)} after first valid row {first_valid}"
            )
        starts[col] = first_valid
    return starts


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_backcast_dataset(
    returns: pd.DataFrame,
    *,
    min_overlap_days: int = 0,
) -> BackcastDataset:
    """Assemble a :class:`BackcastDataset` from a validated returns matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with DatetimeIndex.  Must have monotone missingness.
    min_overlap_days : int
        Minimum overlap length required; raises if the overlap is shorter.

    Returns
    -------
    BackcastDataset

    Raises
    ------
    BackcastDataError
        On non-monotone missingness (via :func:`detect_start_indices`) or when
        ``overlap_length < min_overlap_days``.
    """
    starts = detect_start_indices(returns)

    long_assets = [c for c, i in starts.items() if i == 0]
    short_assets = sorted(
        (c for c, i in starts.items() if i > 0),
        key=lambda c: (starts[c], c),
    )
    short_start_indices = {c: starts[c] for c in short_assets}

    if not short_assets:
        # Degenerate: nothing to impute
        logger.info("No short-history assets detected — nothing to backcast.")
        return BackcastDataset(
            returns_full=returns,
            long_assets=long_assets,
            short_assets=[],
            short_start_indices={},
            overlap_start=returns.index[0],
            overlap_end=returns.index[-1],
            backcast_start=None,
            backcast_end=None,
            overlap_matrix=returns.copy(),
            long_history_matrix=returns.iloc[0:0][long_assets],
        )

    # Overlap begins at the LAST-starting short asset
    overlap_first_idx = max(short_start_indices.values())
    overlap_start = returns.index[overlap_first_idx]
    overlap_end = returns.index[-1]
    backcast_start = returns.index[0]
    backcast_end = returns.index[overlap_first_idx - 1]

    overlap_matrix = returns.iloc[overlap_first_idx:]
    long_history_matrix = returns.iloc[:overlap_first_idx][long_assets]

    if len(overlap_matrix) < min_overlap_days:
        raise BackcastDataError(
            f"Overlap has only {len(overlap_matrix)} rows, "
            f"less than min_overlap_days={min_overlap_days}"
        )

    logger.info(
        "Dataset: %d long assets, %d short assets, %d overlap rows, "
        "%d backcast rows (overlap_start=%s)",
        len(long_assets), len(short_assets),
        len(overlap_matrix), len(long_history_matrix), overlap_start.date(),
    )
    return BackcastDataset(
        returns_full=returns,
        long_assets=long_assets,
        short_assets=short_assets,
        short_start_indices=short_start_indices,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        backcast_start=backcast_start,
        backcast_end=backcast_end,
        overlap_matrix=overlap_matrix,
        long_history_matrix=long_history_matrix,
    )


def load_backcast_dataset(
    path: str | Path,
    *,
    date_column: str = _DEFAULT_DATE_COLUMN,
    min_overlap_days: int = 0,
    return_warn_threshold: float = _RETURN_WARN_THRESHOLD,
) -> BackcastDataset:
    """Convenience: :func:`load_returns_csv` + :func:`build_backcast_dataset`.

    Parameters
    ----------
    path : str or Path
    date_column : str
    min_overlap_days : int
    return_warn_threshold : float

    Returns
    -------
    BackcastDataset
    """
    returns = load_returns_csv(
        path, date_column=date_column, return_warn_threshold=return_warn_threshold,
    )
    return build_backcast_dataset(returns, min_overlap_days=min_overlap_days)
