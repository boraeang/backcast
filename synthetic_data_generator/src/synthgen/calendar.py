"""Business-day calendar generation."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_business_days(start_date: str, t_total: int) -> pd.DatetimeIndex:
    """Generate a business-day date index of exactly *t_total* days.

    Uses ``pandas.bdate_range`` (Mon–Fri, no holidays).  If *start_date*
    falls on a weekend, ``bdate_range`` automatically advances to the next
    Monday so the returned index always begins on a weekday.

    Parameters
    ----------
    start_date : str
        First trading day in 'YYYY-MM-DD' format.
    t_total : int
        Number of business days to generate.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex of exactly *t_total* consecutive business days.

    Examples
    --------
    >>> idx = generate_business_days("1990-01-02", 5)
    >>> len(idx)
    5
    >>> idx[0].day_of_week < 5  # Mon–Fri
    True
    """
    dates = pd.bdate_range(start=start_date, periods=t_total, freq="B")
    logger.info(
        "Generated %d business days: %s → %s",
        len(dates),
        dates[0].date(),
        dates[-1].date(),
    )
    return dates
