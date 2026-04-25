"""Kalman filter + RTS smoother for time-varying factor loadings (TVP betas).

One Kalman filter is fit per short-history asset.  The observation is that
asset's return at each day of the overlap period; the latent state is the
regression coefficients ``(α_t, B_t)`` of that asset on the long-history
factors — treated as a random walk::

    y_t       = Z_t · β_t + ε_t,       ε_t ~ N(0, H)           (observation)
    β_t       = β_{t-1} + η_t,         η_t ~ N(0, Q)            (state)
    Z_t       = [1, r_{long,t,1}, ..., r_{long,t,K}]                  (design row)

where ``H`` is a scalar observation variance (per short asset) and ``Q`` is
the diagonal state-noise covariance.  Defaults follow the config:

- ``H`` is the sample residual variance from an OLS fit on the overlap.
- ``Q = state_noise_scale · H · I`` (small RW drift, tuneable).
- Initial state is the OLS estimate; initial state covariance is
  ``initial_state_cov_scale · I`` (effectively diffuse).

After the filter + RTS smoother, the backcast loading is chosen per the spec
— the earliest smoothed state or the mean of the first ``K`` smoothed states
— and explicitly NOT extrapolated backward.

References
----------
Harvey, A.C. (1989).  *Forecasting, Structural Time Series Models and the
Kalman Filter.*  Cambridge University Press.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KalmanAssetResult:
    """Kalman output for a single short asset.

    Attributes
    ----------
    asset_name : str
    state_names : list[str]
        ``['intercept', 'beta_<long1>', ..., 'beta_<longK>']``
    filtered_state : np.ndarray, shape (T, K+1)
    filtered_state_cov : np.ndarray, shape (T, K+1, K+1)
    smoothed_state : np.ndarray, shape (T, K+1)
    smoothed_state_cov : np.ndarray, shape (T, K+1, K+1)
    innovations : np.ndarray, shape (T,)
    innovation_variances : np.ndarray, shape (T,)
    residual_variance : float
        Estimated ``H``.
    state_noise_cov : np.ndarray, shape (K+1, K+1)
        Estimated ``Q``.
    log_likelihood : float
    backcast_state : np.ndarray, shape (K+1,)
        The loading vector to use in the backcast period.
    date_index : pd.DatetimeIndex
    """

    asset_name: str
    state_names: list[str]
    filtered_state: np.ndarray
    filtered_state_cov: np.ndarray
    smoothed_state: np.ndarray
    smoothed_state_cov: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    residual_variance: float
    state_noise_cov: np.ndarray
    log_likelihood: float
    backcast_state: np.ndarray
    date_index: pd.DatetimeIndex


@dataclass
class KalmanMultiAssetResult:
    """Combined Kalman results across every short asset.

    Attributes
    ----------
    long_assets : list[str]
    short_assets : list[str]
    per_asset : dict[str, KalmanAssetResult]
    backcast_matrix : pd.DataFrame
        Rows = short asset, columns = ``['intercept'] + long_assets``.
    smoothed_betas : dict[str, pd.DataFrame]
        Per-short-asset smoothed state paths, shape ``(T_overlap, K+1)``.
    """

    long_assets: list[str]
    short_assets: list[str]
    per_asset: dict
    backcast_matrix: pd.DataFrame
    smoothed_betas: dict


# ---------------------------------------------------------------------------
# Filter + smoother core (single asset, scalar observation)
# ---------------------------------------------------------------------------

def _kalman_filter(
    y: np.ndarray,
    Z: np.ndarray,
    H: float,
    Q: np.ndarray,
    beta_0: np.ndarray,
    P_0: np.ndarray,
) -> dict:
    """Run the Kalman filter for a scalar observation, random-walk state.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
    Z : np.ndarray, shape (T, S)
        Design matrix (each row ``Z_t`` has a leading 1 for the intercept).
    H : float
        Observation noise variance.
    Q : np.ndarray, shape (S, S)
        State-noise covariance.
    beta_0 : np.ndarray, shape (S,)
        Prior mean.
    P_0 : np.ndarray, shape (S, S)
        Prior covariance (use a large scalar * I for a diffuse prior).

    Returns
    -------
    dict with keys:
        ``beta_filt``, ``P_filt`` — posterior at each step
        ``beta_pred``, ``P_pred`` — prior at each step (for smoother)
        ``v``, ``F`` — innovations and their variances
        ``ll`` — log-likelihood
    """
    T, S = Z.shape
    beta_filt = np.empty((T, S), dtype=np.float64)
    P_filt = np.empty((T, S, S), dtype=np.float64)
    beta_pred = np.empty((T, S), dtype=np.float64)
    P_pred = np.empty((T, S, S), dtype=np.float64)
    v = np.empty(T, dtype=np.float64)
    F = np.empty(T, dtype=np.float64)

    beta_curr = beta_0.astype(np.float64).copy()
    P_curr = P_0.astype(np.float64).copy()

    ll = 0.0
    log_2pi = np.log(2.0 * np.pi)
    for t in range(T):
        # Predict
        beta_t_t1 = beta_curr                     # identity transition
        P_t_t1 = P_curr + Q
        beta_pred[t] = beta_t_t1
        P_pred[t] = P_t_t1

        # Update
        Zt = Z[t]
        y_hat = float(Zt @ beta_t_t1)
        innov = float(y[t] - y_hat)
        F_t = float(Zt @ P_t_t1 @ Zt + H)
        if F_t <= 0:
            F_t = 1e-20
        K_gain = (P_t_t1 @ Zt) / F_t
        beta_curr = beta_t_t1 + K_gain * innov
        # Joseph / symmetric form: (I - K Z) P (I - K Z)^T + K R K^T is more stable,
        # but the simpler form below suffices after we symmetrise.
        P_curr = P_t_t1 - np.outer(K_gain, Zt @ P_t_t1)
        P_curr = 0.5 * (P_curr + P_curr.T)

        beta_filt[t] = beta_curr
        P_filt[t] = P_curr
        v[t] = innov
        F[t] = F_t
        ll += -0.5 * (log_2pi + np.log(F_t) + innov * innov / F_t)

    return {
        "beta_filt": beta_filt,
        "P_filt": P_filt,
        "beta_pred": beta_pred,
        "P_pred": P_pred,
        "v": v,
        "F": F,
        "ll": float(ll),
    }


def _rts_smoother(
    beta_filt: np.ndarray, P_filt: np.ndarray,
    beta_pred: np.ndarray, P_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother over the filter output."""
    T, S = beta_filt.shape
    beta_sm = beta_filt.copy()
    P_sm = P_filt.copy()
    eye = np.eye(S, dtype=np.float64)
    for t in range(T - 2, -1, -1):
        # C_t = P_filt[t] @ inv(P_pred[t+1])
        try:
            cfac = cho_factor(P_pred[t + 1] + 1e-14 * eye, lower=True)
            # inv(P_pred) @ P_filt[t].T  →  solve P_pred X = P_filt[t].T,  then C = X.T
            X = cho_solve(cfac, P_filt[t].T)
            C = X.T
        except np.linalg.LinAlgError:
            C = P_filt[t] @ np.linalg.pinv(P_pred[t + 1])
        beta_sm[t] = beta_filt[t] + C @ (beta_sm[t + 1] - beta_pred[t + 1])
        P_sm[t] = P_filt[t] + C @ (P_sm[t + 1] - P_pred[t + 1]) @ C.T
        P_sm[t] = 0.5 * (P_sm[t] + P_sm[t].T)
    return beta_sm, P_sm


# ---------------------------------------------------------------------------
# Single-asset Kalman TVP fit
# ---------------------------------------------------------------------------

def _ols_initial(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS coefficient and residual variance (used for H and β_0)."""
    # Solve (Z^T Z) β = Z^T y
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    resid_var = float(np.var(residuals, ddof=len(coef)))
    if resid_var <= 0:
        resid_var = 1e-12
    return coef, resid_var


def fit_kalman_tvp(
    r_long: pd.DataFrame,
    r_short: pd.Series,
    *,
    state_noise_scale: float = 0.01,
    initial_state_cov_scale: float = 1.0,
    use_smoother: bool = True,
    backcast_beta_method: str = "earliest_smoothed",
    backcast_beta_k: int = 63,
) -> KalmanAssetResult:
    """Kalman filter + RTS smoother for one short asset.

    Parameters
    ----------
    r_long : pd.DataFrame, shape (T, K)
        Long-history (fully observed) returns over the overlap.
    r_short : pd.Series, shape (T,)
        Short asset returns over the overlap (fully observed).
    state_noise_scale : float
        ``Q = state_noise_scale * H * I``.
    initial_state_cov_scale : float
        ``P_0 = initial_state_cov_scale * I`` — diffuse prior when large.
    use_smoother : bool
        If True, return RTS-smoothed states; else filtered states.
    backcast_beta_method : {'earliest_smoothed', 'mean_first_k'}
    backcast_beta_k : int
        When ``method='mean_first_k'`` — number of leading smoothed states
        to average.

    Returns
    -------
    KalmanAssetResult

    Raises
    ------
    ValueError
        On NaN inputs, mismatched lengths, or invalid *backcast_beta_method*.
    """
    if not r_long.index.equals(r_short.index):
        raise ValueError("r_long and r_short must share an index")
    if r_long.isna().any().any() or r_short.isna().any():
        raise ValueError("Kalman TVP requires fully-observed overlap data")
    if backcast_beta_method not in ("earliest_smoothed", "mean_first_k"):
        raise ValueError(
            f"unknown backcast_beta_method {backcast_beta_method!r}"
        )

    long_names = list(r_long.columns)
    state_names = ["intercept"] + [f"beta_{n}" for n in long_names]
    K = r_long.shape[1]
    S = K + 1

    X = r_long.to_numpy(dtype=np.float64)
    y = r_short.to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(y)), X])   # (T, S)

    beta_0, H = _ols_initial(design, y)
    Q = state_noise_scale * H * np.eye(S)
    P_0 = initial_state_cov_scale * np.eye(S)

    filt = _kalman_filter(y, design, H, Q, beta_0, P_0)

    if use_smoother:
        beta_sm, P_sm = _rts_smoother(
            filt["beta_filt"], filt["P_filt"],
            filt["beta_pred"], filt["P_pred"],
        )
    else:
        beta_sm, P_sm = filt["beta_filt"], filt["P_filt"]

    if backcast_beta_method == "earliest_smoothed":
        backcast_state = beta_sm[0].copy()
    else:  # mean_first_k
        k = min(backcast_beta_k, len(beta_sm))
        backcast_state = beta_sm[:k].mean(axis=0)

    logger.debug(
        "Kalman for %s: H=%.3e, Q_diag=%.3e, log-L=%.1f, backcast=%s",
        r_short.name, H, Q[0, 0], filt["ll"], backcast_beta_method,
    )
    return KalmanAssetResult(
        asset_name=str(r_short.name),
        state_names=state_names,
        filtered_state=filt["beta_filt"],
        filtered_state_cov=filt["P_filt"],
        smoothed_state=beta_sm,
        smoothed_state_cov=P_sm,
        innovations=filt["v"],
        innovation_variances=filt["F"],
        residual_variance=H,
        state_noise_cov=Q,
        log_likelihood=filt["ll"],
        backcast_state=backcast_state,
        date_index=r_long.index,
    )


# ---------------------------------------------------------------------------
# Multi-asset wrapper
# ---------------------------------------------------------------------------

def fit_kalman_all(
    overlap_matrix: pd.DataFrame,
    long_assets: list[str],
    short_assets: list[str],
    **kwargs,
) -> KalmanMultiAssetResult:
    """Fit one Kalman TVP per short asset, all sharing the same long factors.

    Parameters
    ----------
    overlap_matrix : pd.DataFrame
        Must be fully observed (overlap period).
    long_assets, short_assets : list[str]
    **kwargs
        Forwarded to :func:`fit_kalman_tvp`.

    Returns
    -------
    KalmanMultiAssetResult
    """
    r_long = overlap_matrix[long_assets]
    per_asset: dict[str, KalmanAssetResult] = {}
    smoothed_betas: dict[str, pd.DataFrame] = {}
    backcast_rows: list[np.ndarray] = []

    for name in short_assets:
        res = fit_kalman_tvp(r_long, overlap_matrix[name], **kwargs)
        per_asset[name] = res
        smoothed_betas[name] = pd.DataFrame(
            res.smoothed_state, index=res.date_index, columns=res.state_names,
        )
        backcast_rows.append(res.backcast_state)

    backcast_matrix = pd.DataFrame(
        np.array(backcast_rows),
        index=short_assets,
        columns=["intercept"] + long_assets,
    )
    return KalmanMultiAssetResult(
        long_assets=long_assets,
        short_assets=short_assets,
        per_asset=per_asset,
        backcast_matrix=backcast_matrix,
        smoothed_betas=smoothed_betas,
    )


# ---------------------------------------------------------------------------
# Kalman-based single imputation (convenience)
# ---------------------------------------------------------------------------

def kalman_impute(
    returns_full: pd.DataFrame,
    multi: KalmanMultiAssetResult,
) -> pd.DataFrame:
    """Impute the backcast period using fixed backcast betas from the Kalman fit.

    For each row t where a short asset is missing, that asset's imputation is
    ``α̂ + B̂ · r_long_t`` where (α̂, B̂) is the *backcast_state* from the
    Kalman fit — NOT extrapolated.  Observed entries are left untouched.

    Parameters
    ----------
    returns_full : pd.DataFrame
    multi : KalmanMultiAssetResult

    Returns
    -------
    pd.DataFrame
        Same shape as *returns_full*.  Only previously-NaN cells are replaced.
    """
    filled = returns_full.copy()
    long_names = multi.long_assets
    for name in multi.short_assets:
        res = multi.per_asset[name]
        alpha = res.backcast_state[0]
        betas = res.backcast_state[1:]
        mask = filled[name].isna()
        if not mask.any():
            continue
        r_long_rows = filled.loc[mask, long_names].to_numpy(dtype=np.float64)
        # Any NaN in long rows would break this; Kalman assumes long is observed.
        filled.loc[mask, name] = alpha + r_long_rows @ betas
    return filled
