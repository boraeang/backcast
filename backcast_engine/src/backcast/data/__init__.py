"""Data loading, validation, and transforms."""
from backcast.data.loader import (
    BackcastDataset,
    build_backcast_dataset,
    detect_start_indices,
    load_backcast_dataset,
    load_returns_csv,
)
from backcast.data.transforms import (
    log_to_simple_returns,
    prices_to_returns,
    returns_to_prices,
    simple_to_log_returns,
)

__all__ = [
    "BackcastDataset",
    "build_backcast_dataset",
    "detect_start_indices",
    "load_backcast_dataset",
    "load_returns_csv",
    "simple_to_log_returns",
    "log_to_simple_returns",
    "returns_to_prices",
    "prices_to_returns",
]
