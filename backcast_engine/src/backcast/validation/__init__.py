"""Validation — metrics, holdout, and diagnostics."""
from backcast.validation.holdout import (
    HoldoutReport,
    HoldoutWindow,
    run_holdout_validation,
)

__all__ = ["HoldoutReport", "HoldoutWindow", "run_holdout_validation"]
