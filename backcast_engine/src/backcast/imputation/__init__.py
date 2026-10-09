"""Imputation algorithms (single and multiple) for backcasting."""
from backcast.imputation.copula_sim import (
    CopulaFit,
    CopulaSimResult,
    MarginalFit,
    fit_copula,
    fit_marginal,
    fit_marginals,
    simulate_copula,
)
from backcast.imputation.multiple_impute import (
    MultipleImputationResult,
    RubinResult,
    apply_rubin,
    combine_estimates,
    multiple_impute,
    multiple_impute_regime,
    prediction_intervals,
)
from backcast.imputation.regime_params import (
    RegimeParams,
    RegimeSourceSummary,
    build_regime_params,
    nearest_psd,
    summarize_regime_sources,
)
from backcast.imputation.single_impute import (
    impute_missing_values,
    regime_single_impute,
    single_impute,
)

__all__ = [
    "impute_missing_values",
    "regime_single_impute",
    "single_impute",
    # regime parameter cascade
    "RegimeParams",
    "RegimeSourceSummary",
    "build_regime_params",
    "nearest_psd",
    "summarize_regime_sources",
    # multiple imputation
    "MultipleImputationResult",
    "RubinResult",
    "apply_rubin",
    "combine_estimates",
    "multiple_impute",
    "multiple_impute_regime",
    "prediction_intervals",
    # copula simulation
    "CopulaFit",
    "CopulaSimResult",
    "MarginalFit",
    "fit_copula",
    "fit_marginal",
    "fit_marginals",
    "simulate_copula",
]
