"""Statistical models for backcasting."""
from backcast.models.em_stambaugh import (
    ConditionalParams,
    EMResult,
    em_stambaugh,
)
from backcast.models.kalman_tvp import (
    KalmanAssetResult,
    KalmanMultiAssetResult,
    fit_kalman_all,
    fit_kalman_tvp,
    kalman_impute,
)
from backcast.models.model_selector import (
    MethodCVResult,
    ModelSelectionResult,
    evaluate_method_cv,
    select_model_cv,
)
from backcast.models.regime_hmm import (
    HMMResult,
    HMMSelectionResult,
    compute_regime_params,
    fit_and_select_hmm,
    fit_regime_hmm,
    regime_conditional_impute,
)

__all__ = [
    "ConditionalParams",
    "EMResult",
    "em_stambaugh",
    # Kalman
    "KalmanAssetResult",
    "KalmanMultiAssetResult",
    "fit_kalman_all",
    "fit_kalman_tvp",
    "kalman_impute",
    # HMM
    "HMMResult",
    "HMMSelectionResult",
    "compute_regime_params",
    "fit_and_select_hmm",
    "fit_regime_hmm",
    "regime_conditional_impute",
    # Model selection
    "MethodCVResult",
    "ModelSelectionResult",
    "evaluate_method_cv",
    "select_model_cv",
]
