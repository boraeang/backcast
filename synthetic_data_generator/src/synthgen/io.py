"""CSV and JSON export for synthetic datasets."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _deep_convert(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python types for JSON safety."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_convert(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_dataset(
    output_dir: str | Path,
    returns_masked: pd.DataFrame,
    ground_truth: dict[str, Any],
    returns_complete: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write a synthetic dataset to disk.

    Creates *output_dir* (and any parents) if it does not exist.

    Parameters
    ----------
    output_dir : str or Path
        Destination directory.
    returns_masked : pd.DataFrame
        Masked returns matrix (with NaN for short-history assets).
        Index must be a DatetimeIndex; columns are asset names.
    ground_truth : dict
        DGP parameters and metadata.  Numpy arrays are converted to lists
        automatically.
    returns_complete : pd.DataFrame or None
        Unmasked returns matrix.  Written as 'returns_complete.csv' when
        not None.

    Returns
    -------
    dict[str, Path]
        Mapping from logical name ('returns', 'ground_truth',
        optionally 'returns_complete') to the written file path.

    Notes
    -----
    Dates are formatted as YYYY-MM-DD.  All float values use full float64
    precision.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # returns.csv
    returns_path = out / "returns.csv"
    _write_returns(returns_masked, returns_path)
    paths["returns"] = returns_path

    # returns_complete.csv
    if returns_complete is not None:
        complete_path = out / "returns_complete.csv"
        _write_returns(returns_complete, complete_path)
        paths["returns_complete"] = complete_path

    # ground_truth.json
    gt_path = out / "ground_truth.json"
    _write_ground_truth(ground_truth, gt_path)
    paths["ground_truth"] = gt_path

    return paths


def _write_returns(df: pd.DataFrame, path: Path) -> None:
    """Write a returns DataFrame to CSV with date column formatted YYYY-MM-DD."""
    df_out = df.copy()
    df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out.index.name = "date"
    df_out.to_csv(path, float_format="%.10f")
    size_kb = path.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB, %d rows × %d cols)", path, size_kb, *df.shape)


def _write_ground_truth(ground_truth: dict[str, Any], path: Path) -> None:
    """Serialise ground_truth dict to JSON, converting numpy types."""
    converted = _deep_convert(ground_truth)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(converted, fh, indent=2, cls=_NumpyEncoder)
    size_kb = path.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", path, size_kb)


def load_returns(path: str | Path) -> pd.DataFrame:
    """Load a returns CSV previously written by :func:`save_dataset`.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Returns matrix with DatetimeIndex and float64 dtype.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.astype(np.float64)


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    """Load a ground_truth JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed ground-truth parameters.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
