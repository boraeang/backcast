"""Multiple imputation for the backcast engine.

Given a fitted EM model (or regime-conditional parameters), draw ``M`` plausible
complete histories by sampling from the row-wise conditional distribution
``N(α + β·r_{O,t}, Σ_{M|O})``.  Each imputed history is a standalone DataFrame
and can feed into downstream analyses independently.

Rubin's rules (:func:`combine_estimates`, :func:`apply_rubin`) compose the
results of a per-imputation statistic into a single inference with proper
total variance accounting:

    θ̄      = mean_m θ̂_m                             (point estimate)
    W̄      = mean_m V̂(θ̂_m)                          (within-imputation variance)
    B      = (1/(M-1)) Σ_m (θ̂_m − θ̄)²                (between-imputation variance)
    T      = W̄ + (1 + 1/M) · B                       (Rubin's total variance)
    ν      = (M − 1) · (1 + W̄ / ((1 + 1/M) · B))²    (Barnard-Rubin df)

References
----------
Rubin, D.B. (1987).  *Multiple Imputation for Nonresponse in Surveys.*  Wiley.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import numpy.random as npr
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from backcast.data.loader import BackcastDataset
from backcast.models.em_stambaugh import EMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MultipleImputationResult:
    """Output of :func:`multiple_impute` and :func:`multiple_impute_regime`.

    Attributes
    ----------
    imputations : list[pd.DataFrame]
        ``M`` imputed DataFrames, each with the same shape/index/columns as
        the input dataset and no NaN values.
    n_imputations : int
    seed : int
    method : str
        ``'unconditional_em'`` or ``'regime_conditional'``.
    asset_order : list[str]
    conditional_cov : dict | None
        Per-pattern ``Σ_{M|O}`` actually used for drawing — useful for
        diagnostics.
    """

    imputations: list
    n_imputations: int
    seed: int
    method: str
    asset_order: list[str]
    conditional_cov: Optional[dict] = None


@dataclass
class RubinResult:
    """Combined inference after applying Rubin's rules.

    Attributes
    ----------
    point_estimate : np.ndarray
    within_variance : np.ndarray
        May be NaN if per-imputation variances were not supplied.
    between_variance : np.ndarray
    total_variance : np.ndarray
    std_error : np.ndarray
    relative_increase : np.ndarray
        ``(1 + 1/M) · B / W̄`` — interpretable as the fraction of total
        variance attributable to missing information.
    degrees_of_freedom : np.ndarray
        Barnard-Rubin df.  NaN where within_variance is NaN.
    n_imputations : int
    """

    point_estimate: np.ndarray
    within_variance: np.ndarray
    between_variance: np.ndarray
    total_variance: np.ndarray
    std_error: np.ndarray
    relative_increase: np.ndarray
    degrees_of_freedom: np.ndarray
    n_imputations: int


# ---------------------------------------------------------------------------
# Internal: pattern partitioning + Cholesky cache
# ---------------------------------------------------------------------------

def _pattern_groups(R: np.ndarray) -> dict:
    """Group row indices by NaN pattern → {tuple(bool): np.ndarray[int]}."""
    T, _ = R.shape
    nan_mask = np.isnan(R)
    row_bytes = np.ascontiguousarray(nan_mask).view(np.uint8).reshape(T, -1)
    groups: dict[bytes, list[int]] = {}
    for t in range(T):
        if not nan_mask[t].any():
            continue
        groups.setdefault(row_bytes[t].tobytes(), []).append(t)
    out: dict[tuple, np.ndarray] = {}
    for key, rows in groups.items():
        pattern = tuple(nan_mask[rows[0]].tolist())
        out[pattern] = np.asarray(rows, dtype=np.int64)
    return out


def _precompute_conditional(
    mu: np.ndarray,
    sigma: np.ndarray,
    patterns: dict,
) -> dict:
    """Per-pattern β, α, Cholesky(Σ_{M|O}) for fast sampling.

    Returns
    -------
    dict
        Keys are patterns; values are dicts with keys
        ``observed_cols``, ``missing_cols``, ``beta``, ``alpha``, ``cond_cov``,
        ``chol_cond_cov``, ``rows``.
    """
    N = len(mu)
    out: dict = {}
    for pattern, rows in patterns.items():
        missing_mask = np.asarray(pattern, dtype=bool)
        obs_cols = np.where(~missing_mask)[0]
        mis_cols = np.where(missing_mask)[0]
        if len(mis_cols) == 0:
            continue
        if len(obs_cols) == 0:
            # Marginal distribution for fully-missing rows
            cond_cov = sigma
            beta = np.zeros((len(mis_cols), 0))
            alpha = mu.copy()
        else:
            S_OO = sigma[np.ix_(obs_cols, obs_cols)]
            S_OM = sigma[np.ix_(obs_cols, mis_cols)]
            S_MM = sigma[np.ix_(mis_cols, mis_cols)]
            L, low = cho_factor(S_OO, lower=True)
            beta = cho_solve((L, low), S_OM).T                 # (|M|, |O|)
            cond_cov = S_MM - beta @ S_OM
            cond_cov = 0.5 * (cond_cov + cond_cov.T)
            alpha = mu[mis_cols] - beta @ mu[obs_cols]
        try:
            chol = np.linalg.cholesky(cond_cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(cond_cov + 1e-12 * np.eye(len(mis_cols)))
        out[pattern] = {
            "observed_cols": obs_cols,
            "missing_cols": mis_cols,
            "beta": beta,
            "alpha": alpha,
            "cond_cov": cond_cov,
            "chol_cond_cov": chol,
            "rows": rows,
        }
    return out


def _draw_one_imputation(
    R: np.ndarray,
    cond_data: dict,
    rng: npr.Generator,
) -> np.ndarray:
    """Return one imputed matrix using the cached conditional params."""
    R_m = R.copy()
    for pattern_info in cond_data.values():
        obs_cols = pattern_info["observed_cols"]
        mis_cols = pattern_info["missing_cols"]
        beta = pattern_info["beta"]
        alpha = pattern_info["alpha"]
        chol = pattern_info["chol_cond_cov"]
        rows = pattern_info["rows"]

        if len(obs_cols) == 0:
            cond_mean = np.tile(alpha, (len(rows), 1))
        else:
            obs_data = R_m[np.ix_(rows, obs_cols)]
            cond_mean = alpha + obs_data @ beta.T
        # noise = z @ L^T with L lower-triangular, z ~ N(0, I)
        z = rng.standard_normal((len(rows), len(mis_cols)))
        R_m[np.ix_(rows, mis_cols)] = cond_mean + z @ chol.T
    return R_m


# ---------------------------------------------------------------------------
# Public: unconditional multiple imputation
# ---------------------------------------------------------------------------

def multiple_impute(
    dataset: BackcastDataset,
    em_result: EMResult,
    *,
    n_imputations: int = 50,
    seed: int = 0,
) -> MultipleImputationResult:
    """Draw ``n_imputations`` complete histories from the EM conditional model.

    Each imputation draws every missing entry from
    ``N(α + β·r_{O,t}, Σ_{M|O})`` where α, β, Σ_{M|O} are derived from the
    fitted EM parameters for that row's observed/missing partition.  Observed
    entries are left untouched across imputations.

    Parameters
    ----------
    dataset : BackcastDataset
    em_result : EMResult
        Must share column ordering with ``dataset.returns_full``.
    n_imputations : int
    seed : int

    Returns
    -------
    MultipleImputationResult

    Raises
    ------
    ValueError
        If ``em_result.asset_order`` does not match the dataset columns.
    """
    if list(dataset.returns_full.columns) != em_result.asset_order:
        raise ValueError("EMResult.asset_order does not match dataset columns")

    R = dataset.returns_full.to_numpy(dtype=np.float64)
    patterns = _pattern_groups(R)
    cond_data = _precompute_conditional(em_result.mu, em_result.sigma, patterns)
    rng = npr.default_rng(seed)

    index = dataset.returns_full.index
    columns = list(dataset.returns_full.columns)

    imputations: list[pd.DataFrame] = []
    for _m in range(n_imputations):
        R_m = _draw_one_imputation(R, cond_data, rng)
        imputations.append(pd.DataFrame(R_m, index=index, columns=columns))

    logger.info(
        "multiple_impute: %d imputations drawn (method=unconditional_em, patterns=%d)",
        n_imputations, len(cond_data),
    )
    return MultipleImputationResult(
        imputations=imputations,
        n_imputations=n_imputations,
        seed=seed,
        method="unconditional_em",
        asset_order=columns,
        conditional_cov={k: v["cond_cov"] for k, v in cond_data.items()},
    )


# ---------------------------------------------------------------------------
# Public: regime-conditional multiple imputation
# ---------------------------------------------------------------------------

def multiple_impute_regime(
    dataset: BackcastDataset,
    regime_labels: np.ndarray,
    regime_params: dict,
    *,
    n_imputations: int = 50,
    seed: int = 0,
) -> MultipleImputationResult:
    """Regime-conditional multiple imputation.

    For each row ``t`` with regime label ``s_t = k``, missing entries are drawn
    from the regime-specific conditional distribution derived from
    ``regime_params[k] = (μ^{(k)}, Σ^{(k)})``.

    Parameters
    ----------
    dataset : BackcastDataset
    regime_labels : np.ndarray, shape (T,)
        Regime label for every row of ``dataset.returns_full``.
    regime_params : dict[int, {'mu': ..., 'sigma': ...}]
        Per-regime parameters (from
        :func:`backcast.models.regime_hmm.compute_regime_params`).
    n_imputations : int
    seed : int

    Returns
    -------
    MultipleImputationResult

    Raises
    ------
    ValueError
        If *regime_labels* length does not match ``dataset.returns_full``.
    """
    if len(regime_labels) != len(dataset.returns_full):
        raise ValueError(
            f"regime_labels length {len(regime_labels)} != "
            f"dataset rows {len(dataset.returns_full)}"
        )

    R = dataset.returns_full.to_numpy(dtype=np.float64)
    index = dataset.returns_full.index
    columns = list(dataset.returns_full.columns)
    rng = npr.default_rng(seed)

    # Build per-regime, per-pattern conditional params
    regime_cond: dict[int, dict] = {}
    for k, params in regime_params.items():
        sub_rows = np.where(regime_labels == k)[0]
        if len(sub_rows) == 0:
            continue
        # NaN pattern restricted to those rows
        R_sub = R[sub_rows]
        patterns_k = _pattern_groups(R_sub)
        # Remap local row indices to global
        for patt, local_rows in patterns_k.items():
            patterns_k[patt] = sub_rows[local_rows]
        regime_cond[k] = _precompute_conditional(params["mu"], params["sigma"], patterns_k)

    imputations: list[pd.DataFrame] = []
    for _m in range(n_imputations):
        R_m = R.copy()
        for k, cond_data in regime_cond.items():
            for pattern_info in cond_data.values():
                obs_cols = pattern_info["observed_cols"]
                mis_cols = pattern_info["missing_cols"]
                beta = pattern_info["beta"]
                alpha = pattern_info["alpha"]
                chol = pattern_info["chol_cond_cov"]
                rows = pattern_info["rows"]
                if len(obs_cols) == 0:
                    cond_mean = np.tile(alpha, (len(rows), 1))
                else:
                    obs_data = R_m[np.ix_(rows, obs_cols)]
                    cond_mean = alpha + obs_data @ beta.T
                z = rng.standard_normal((len(rows), len(mis_cols)))
                R_m[np.ix_(rows, mis_cols)] = cond_mean + z @ chol.T
        imputations.append(pd.DataFrame(R_m, index=index, columns=columns))

    logger.info(
        "multiple_impute_regime: %d imputations drawn, regimes=%s",
        n_imputations, sorted(regime_params.keys()),
    )
    return MultipleImputationResult(
        imputations=imputations,
        n_imputations=n_imputations,
        seed=seed,
        method="regime_conditional",
        asset_order=columns,
    )


# ---------------------------------------------------------------------------
# Rubin's rules
# ---------------------------------------------------------------------------

def combine_estimates(
    estimates: list,
    variances: Optional[list] = None,
) -> RubinResult:
    """Combine per-imputation estimates via Rubin's rules.

    Parameters
    ----------
    estimates : list
        ``M`` point estimates, each a scalar or np.ndarray of the same shape.
    variances : list, optional
        ``M`` variances of the same shape.  When None the within-imputation
        variance is treated as zero (total = between-only, rarely useful but
        occasionally the right answer when the statistic is a sample mean of
        already-summarised data).

    Returns
    -------
    RubinResult

    Raises
    ------
    ValueError
        On empty input or mismatched list lengths.
    """
    M = len(estimates)
    if M == 0:
        raise ValueError("Cannot combine 0 imputations")
    if variances is not None and len(variances) != M:
        raise ValueError(
            f"variances length {len(variances)} != estimates length {M}"
        )

    est_arr = np.asarray(estimates, dtype=np.float64)
    mean_est = est_arr.mean(axis=0)
    if M > 1:
        between = ((est_arr - mean_est) ** 2).sum(axis=0) / (M - 1)
    else:
        between = np.zeros_like(mean_est)

    if variances is None:
        within = np.full_like(mean_est, np.nan)
        total = (1.0 + 1.0 / M) * between
        rel_incr = np.full_like(mean_est, np.nan)
        df = np.full_like(mean_est, np.nan)
    else:
        var_arr = np.asarray(variances, dtype=np.float64)
        within = var_arr.mean(axis=0)
        total = within + (1.0 + 1.0 / M) * between
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_incr = (1.0 + 1.0 / M) * between / within
            df = (M - 1) * (1.0 + 1.0 / rel_incr) ** 2
        # When B == 0 (no between-imputation variance) df diverges — cap at ∞
        df = np.where(between == 0, np.inf, df)

    se = np.sqrt(np.maximum(total, 0.0))

    return RubinResult(
        point_estimate=mean_est,
        within_variance=within,
        between_variance=between,
        total_variance=total,
        std_error=se,
        relative_increase=rel_incr,
        degrees_of_freedom=df,
        n_imputations=M,
    )


def apply_rubin(
    imputations: list,
    statistic_fn: Callable,
    variance_fn: Optional[Callable] = None,
) -> RubinResult:
    """Apply Rubin's rules to a statistic computed on each imputation.

    Parameters
    ----------
    imputations : list of pd.DataFrame
    statistic_fn : callable
        Takes a DataFrame, returns a scalar or np.ndarray.
    variance_fn : callable, optional
        Takes a DataFrame, returns the variance of the statistic with matching
        shape.  Required for Barnard-Rubin degrees of freedom.

    Returns
    -------
    RubinResult

    Examples
    --------
    >>> apply_rubin(mi.imputations, lambda df: df.mean().values)
    """
    estimates = [np.asarray(statistic_fn(df), dtype=np.float64) for df in imputations]
    variances = None
    if variance_fn is not None:
        variances = [np.asarray(variance_fn(df), dtype=np.float64) for df in imputations]
    return combine_estimates(estimates, variances)


# ---------------------------------------------------------------------------
# Prediction-interval helper
# ---------------------------------------------------------------------------

def prediction_intervals(
    mi_result: MultipleImputationResult,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median / lower / upper prediction surfaces from the M imputations.

    Parameters
    ----------
    mi_result : MultipleImputationResult
    confidence : float
        Nominal coverage (e.g. 0.95 → 2.5 / 97.5 percentiles).

    Returns
    -------
    median, lower, upper : pd.DataFrame
        Same shape as each imputation.
    """
    if not mi_result.imputations:
        raise ValueError("mi_result has no imputations")
    stack = np.stack([df.values for df in mi_result.imputations], axis=0)  # (M, T, N)
    alpha = 1.0 - confidence
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    median = np.median(stack, axis=0)
    lower = np.percentile(stack, lo_q, axis=0)
    upper = np.percentile(stack, hi_q, axis=0)
    idx = mi_result.imputations[0].index
    cols = mi_result.imputations[0].columns
    return (
        pd.DataFrame(median, index=idx, columns=cols),
        pd.DataFrame(lower, index=idx, columns=cols),
        pd.DataFrame(upper, index=idx, columns=cols),
    )
