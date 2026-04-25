"""Custom exception classes for the backcast engine."""
from __future__ import annotations


class BackcastError(Exception):
    """Base class for all backcast-engine errors."""


class BackcastDataError(BackcastError):
    """Raised for invalid input data (non-monotone gaps, duplicate dates, etc.)."""


class BackcastConvergenceError(BackcastError):
    """Raised when an iterative algorithm fails to converge."""
