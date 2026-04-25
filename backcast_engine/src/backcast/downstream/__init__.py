"""Downstream analytics — covariance, uncertainty, backtest."""
from backcast.downstream.backtest import (
    BacktestResult,
    ImputedBacktest,
    STRATEGY_REGISTRY,
    equal_weight,
    inverse_volatility,
    min_variance,
    risk_parity,
    run_backtest,
)
from backcast.downstream.covariance import (
    CovarianceResult,
    combined_covariance,
    denoise_covariance,
    from_em_result,
    sample_covariance,
    shrink_covariance,
)
from backcast.downstream.uncertainty import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PortfolioRiskDistribution,
    box_uncertainty,
    ellipsoidal_uncertainty,
    portfolio_risk_distribution,
)

__all__ = [
    # covariance
    "CovarianceResult", "combined_covariance", "denoise_covariance",
    "from_em_result", "sample_covariance", "shrink_covariance",
    # uncertainty
    "BoxUncertaintySet", "EllipsoidalUncertaintySet",
    "PortfolioRiskDistribution", "box_uncertainty",
    "ellipsoidal_uncertainty", "portfolio_risk_distribution",
    # backtest
    "BacktestResult", "ImputedBacktest", "STRATEGY_REGISTRY",
    "equal_weight", "inverse_volatility", "min_variance", "risk_parity",
    "run_backtest",
]
