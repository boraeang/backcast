"""Backtest harness that runs on each imputed history.

Built-in strategies (``strategy_fn`` takes a returns DataFrame and a
``lookback`` and returns per-date weights):

- ``equal_weight``      — 1/N across assets every day
- ``inverse_volatility`` — 1/σ_i weights, renormalised
- ``min_variance``      — analytic minimum-variance portfolio (long/short)
- ``risk_parity``       — iterative equal-risk-contribution weights

The harness rebalances every ``rebalance_freq`` days, then evaluates the
strategy's cumulative return on every imputed history.  Aggregate statistics
(median path, 5/95 pct bands, per-imputation Sharpe and max-drawdown
distributions) come for free via the :class:`BacktestResult`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Strategy = Callable[[pd.DataFrame, int], np.ndarray]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImputedBacktest:
    """Result of running one strategy on one imputed history."""

    weights_path: pd.DataFrame     # (T, N)
    returns_path: pd.Series         # (T,)
    cumulative: pd.Series           # (T,)
    sharpe_annual: float
    max_drawdown: float             # negative number
    total_return: float


@dataclass
class BacktestResult:
    """Cross-imputation aggregation.

    Attributes
    ----------
    strategy_name : str
    per_imputation : list[ImputedBacktest]
    cumulative_median : pd.Series
    cumulative_p05, cumulative_p95 : pd.Series
    sharpe_distribution : np.ndarray
    max_drawdown_distribution : np.ndarray
    total_return_distribution : np.ndarray
    n_imputations : int
    config : dict
    """

    strategy_name: str
    per_imputation: list
    cumulative_median: pd.Series
    cumulative_p05: pd.Series
    cumulative_p95: pd.Series
    sharpe_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    total_return_distribution: np.ndarray
    n_imputations: int
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

def equal_weight(returns_window: pd.DataFrame, lookback: int) -> np.ndarray:
    """Equal-weight 1/N regardless of the lookback window."""
    N = returns_window.shape[1]
    return np.full(N, 1.0 / N)


def inverse_volatility(returns_window: pd.DataFrame, lookback: int) -> np.ndarray:
    """Weights proportional to ``1/σ_i`` over the lookback window."""
    window = returns_window.iloc[-lookback:]
    vols = window.std(axis=0, ddof=1).to_numpy()
    vols = np.where(vols < 1e-12, 1e-12, vols)
    w = 1.0 / vols
    return w / w.sum()


def min_variance(returns_window: pd.DataFrame, lookback: int) -> np.ndarray:
    """Analytic minimum-variance portfolio: ``w ∝ Σ⁻¹ 1`` (long/short allowed)."""
    window = returns_window.iloc[-lookback:]
    cov = window.cov().to_numpy()
    N = cov.shape[0]
    cov = cov + 1e-12 * np.eye(N)  # regularise
    try:
        inv_ones = np.linalg.solve(cov, np.ones(N))
    except np.linalg.LinAlgError:
        return np.full(N, 1.0 / N)
    return inv_ones / inv_ones.sum()


def risk_parity(
    returns_window: pd.DataFrame, lookback: int, max_iter: int = 200, tol: float = 1e-8,
) -> np.ndarray:
    """Equal-risk-contribution weights via the classic fixed-point iteration.

    Solves ``w_i · (Σw)_i = const`` for i = 1..N with Σ w = 1, ``w ≥ 0``.
    """
    window = returns_window.iloc[-lookback:]
    cov = window.cov().to_numpy()
    N = cov.shape[0]
    cov = cov + 1e-12 * np.eye(N)
    # Initialise at inverse-vol
    vols = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    w = 1.0 / vols
    w = w / w.sum()
    for _ in range(max_iter):
        # Risk contributions: w_i * (Σw)_i
        sigma_w = cov @ w
        rc = w * sigma_w
        # Target equal contribution
        w_new = w * (rc.mean() / np.where(rc < 1e-30, 1e-30, rc))
        w_new = np.clip(w_new, 0.0, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


STRATEGY_REGISTRY: dict[str, Strategy] = {
    "equal_weight": equal_weight,
    "inverse_volatility": inverse_volatility,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _max_drawdown(cumulative: pd.Series) -> float:
    """Maximum drawdown of a cumulative-return series starting at 1.0."""
    equity = cumulative.to_numpy()
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _run_on_one_history(
    returns: pd.DataFrame,
    strategy_fn: Strategy,
    *,
    lookback: int,
    rebalance_freq: int,
) -> ImputedBacktest:
    """Evaluate *strategy_fn* on one complete returns history."""
    T, N = returns.shape
    weights = np.zeros((T, N), dtype=np.float64)
    current_w = np.full(N, 1.0 / N)
    for t in range(T):
        if t < lookback:
            weights[t] = 1.0 / N            # warm-up
            continue
        if (t - lookback) % rebalance_freq == 0:
            window = returns.iloc[: t + 1]  # includes up to t inclusive
            try:
                current_w = strategy_fn(window, lookback)
            except Exception as exc:
                logger.warning("strategy_fn failed at row %d: %s", t, exc)
        weights[t] = current_w

    weights_df = pd.DataFrame(weights, index=returns.index, columns=returns.columns)
    port_ret = (weights_df * returns).sum(axis=1)
    port_ret.name = "portfolio_return"
    cum = (1.0 + port_ret).cumprod()
    cum.name = "cumulative"

    vol_annual = port_ret.std(ddof=1) * np.sqrt(252)
    mean_annual = port_ret.mean() * 252
    sharpe = mean_annual / vol_annual if vol_annual > 1e-30 else 0.0

    return ImputedBacktest(
        weights_path=weights_df,
        returns_path=port_ret,
        cumulative=cum,
        sharpe_annual=float(sharpe),
        max_drawdown=_max_drawdown(cum),
        total_return=float(cum.iloc[-1] - 1.0),
    )


def run_backtest(
    imputations: list,
    strategy: "str | Strategy",
    *,
    strategy_name: Optional[str] = None,
    lookback: int = 63,
    rebalance_freq: int = 21,
) -> BacktestResult:
    """Run *strategy* on every imputed history and aggregate.

    Parameters
    ----------
    imputations : list of pd.DataFrame
        Full histories (typically ``MultipleImputationResult.imputations``).
    strategy : str or callable
        Either a key in ``STRATEGY_REGISTRY`` or a custom callable
        ``(window_df, lookback) -> weights_vector``.
    strategy_name : str, optional
        Reported in the result.  Defaults to the registry key or
        ``strategy.__name__``.
    lookback : int
        Rows of history supplied to the strategy.
    rebalance_freq : int
        Rebalance every this many trading days.

    Returns
    -------
    BacktestResult
    """
    if isinstance(strategy, str):
        strategy_name = strategy_name or strategy
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"unknown strategy {strategy!r}; "
                f"known = {sorted(STRATEGY_REGISTRY)}"
            )
        strategy_fn = STRATEGY_REGISTRY[strategy]
    else:
        strategy_fn = strategy
        strategy_name = strategy_name or getattr(strategy, "__name__", "custom")

    per_imputation: list[ImputedBacktest] = []
    for i, df in enumerate(imputations):
        res = _run_on_one_history(
            df, strategy_fn, lookback=lookback, rebalance_freq=rebalance_freq,
        )
        per_imputation.append(res)

    cum_stack = pd.concat([r.cumulative for r in per_imputation], axis=1)
    cum_median = cum_stack.median(axis=1)
    cum_p05 = cum_stack.quantile(0.05, axis=1)
    cum_p95 = cum_stack.quantile(0.95, axis=1)

    sharpes = np.array([r.sharpe_annual for r in per_imputation])
    drawdowns = np.array([r.max_drawdown for r in per_imputation])
    total_returns = np.array([r.total_return for r in per_imputation])

    return BacktestResult(
        strategy_name=strategy_name,
        per_imputation=per_imputation,
        cumulative_median=cum_median,
        cumulative_p05=cum_p05,
        cumulative_p95=cum_p95,
        sharpe_distribution=sharpes,
        max_drawdown_distribution=drawdowns,
        total_return_distribution=total_returns,
        n_imputations=len(imputations),
        config={
            "strategy_name": strategy_name,
            "lookback": lookback,
            "rebalance_freq": rebalance_freq,
        },
    )
