"""Copula-based scenario simulation.

Fit per-asset marginals (Normal / Student-t / empirical) + a joint copula
(Gaussian or Student-t) on a fully observed (or imputed) returns history,
then simulate ``S`` scenarios of length ``horizon`` that respect both the
marginal distributions and the dependence structure.

Design choices
--------------
- Marginal candidates are restricted to three families — the ones named in
  the spec (normal, Student-t, empirical CDF).  Model selection uses AIC
  by default (KS-statistic is reported too for diagnostics).
- Gaussian and Student-t copulas are implemented directly — no dependence on
  ``pyvinecopulib`` or any other non-standard package.
- Student-t copula degrees-of-freedom ν are estimated by a bounded grid
  search over the profile log-likelihood.
- Simulation draws are vectorised: ``(S, horizon, N)`` scenarios in a single
  call to avoid Python-level loops over S.

References
----------
Joe, H. (2014).  *Dependence Modeling with Copulas.*  Chapman & Hall.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.random as npr
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from scipy.special import gammaln

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarginalFit:
    """Fitted marginal distribution for a single asset.

    Attributes
    ----------
    asset_name : str
    distribution : str
        'normal', 'student_t', or 'empirical'.
    params : dict
        - 'normal'     : {'loc', 'scale'}
        - 'student_t'  : {'df', 'loc', 'scale'}
        - 'empirical'  : {'sorted_values': np.ndarray}
    aic : float
    ks_statistic : float
    ks_pvalue : float
    """

    asset_name: str
    distribution: str
    params: dict
    aic: float
    ks_statistic: float
    ks_pvalue: float


@dataclass
class CopulaFit:
    """Fitted dependence copula.

    Attributes
    ----------
    copula_type : str
        'gaussian' or 'student_t'.
    correlation : np.ndarray, shape (N, N)
        Correlation matrix of the latent normal/t variates.
    df : float or None
        Only set for 'student_t'.
    asset_order : list[str]
    n_samples : int
        Sample size used for the fit.
    """

    copula_type: str
    correlation: np.ndarray
    df: Optional[float]
    asset_order: list[str]
    n_samples: int


@dataclass
class CopulaSimResult:
    """Output of :func:`simulate_copula`.

    Attributes
    ----------
    simulated_returns : np.ndarray, shape (n_simulations, horizon, N)
    marginals : dict[str, MarginalFit]
    copula : CopulaFit
    asset_names : list[str]
    seed : int
    """

    simulated_returns: np.ndarray
    marginals: dict
    copula: CopulaFit
    asset_names: list[str]
    seed: int


# ---------------------------------------------------------------------------
# Marginal fitting
# ---------------------------------------------------------------------------

def _fit_normal(x: np.ndarray) -> tuple[dict, float]:
    loc, scale = float(x.mean()), float(x.std(ddof=1))
    scale = max(scale, 1e-12)
    ll = float(stats.norm.logpdf(x, loc=loc, scale=scale).sum())
    aic = 2 * 2 - 2 * ll                                     # k=2 params
    return {"loc": loc, "scale": scale}, aic


def _fit_student_t(x: np.ndarray) -> tuple[dict, float]:
    df, loc, scale = stats.t.fit(x)
    scale = max(float(scale), 1e-12)
    df = max(float(df), 2.1)                                 # finite variance
    ll = float(stats.t.logpdf(x, df=df, loc=loc, scale=scale).sum())
    aic = 2 * 3 - 2 * ll                                     # k=3 params
    return {"df": df, "loc": float(loc), "scale": scale}, aic


def _fit_empirical(x: np.ndarray) -> tuple[dict, float]:
    # AIC for the empirical CDF is ill-defined (n free "parameters").
    # Penalise with k = min(n, 20) so an empirical fit is only chosen when the
    # parametric options are clearly worse.
    sorted_vals = np.sort(x)
    # Rough log-likelihood via kernel density at the observed points
    # Use a scaled-normal kernel with Silverman bandwidth.
    n = len(x)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma_kde = 0.9 * min(float(x.std(ddof=1)), iqr / 1.34) * n ** (-0.2)
    sigma_kde = max(sigma_kde, 1e-10)
    # Approximate log-pdf at each sample via leave-one-out KDE
    diffs = x[:, None] - x[None, :]
    k = stats.norm.pdf(diffs, loc=0.0, scale=sigma_kde)
    np.fill_diagonal(k, 0.0)
    dens = k.sum(axis=1) / max(n - 1, 1)
    dens = np.maximum(dens, 1e-300)
    ll = float(np.log(dens).sum())
    aic = 2 * min(n, 20) - 2 * ll
    return {"sorted_values": sorted_vals}, aic


def _ks_pvalue(x: np.ndarray, dist: str, params: dict) -> tuple[float, float]:
    """One-sample KS goodness-of-fit against the fitted distribution."""
    if dist == "normal":
        res = stats.kstest(x, "norm", args=(params["loc"], params["scale"]))
    elif dist == "student_t":
        res = stats.kstest(
            x, "t", args=(params["df"], params["loc"], params["scale"]),
        )
    else:  # empirical — self-test, trivially perfect
        return 0.0, 1.0
    return float(res.statistic), float(res.pvalue)


def fit_marginal(
    series: "np.ndarray | pd.Series",
    candidates: tuple[str, ...] = ("normal", "student_t", "empirical"),
    criterion: str = "aic",
    name: str = "unknown",
) -> MarginalFit:
    """Select the best-fitting marginal from the candidate families.

    Parameters
    ----------
    series : array-like
    candidates : tuple of str
        Subset of ``{'normal', 'student_t', 'empirical'}``.
    criterion : {'aic', 'ks'}
        'aic' — minimum AIC wins; 'ks' — lowest KS statistic wins.
    name : str
        Asset name, copied into the returned :class:`MarginalFit`.

    Returns
    -------
    MarginalFit
    """
    if criterion not in ("aic", "ks"):
        raise ValueError(f"criterion must be 'aic' or 'ks', got {criterion!r}")
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        raise ValueError(f"Too few observations for marginal fit: {len(x)}")

    fitted: dict[str, tuple[dict, float, float, float]] = {}  # name → (params, aic, ks, p)
    for cand in candidates:
        if cand == "normal":
            params, aic = _fit_normal(x)
        elif cand == "student_t":
            params, aic = _fit_student_t(x)
        elif cand == "empirical":
            params, aic = _fit_empirical(x)
        else:
            raise ValueError(f"Unknown marginal family {cand!r}")
        ks_stat, ks_p = _ks_pvalue(x, cand, params)
        fitted[cand] = (params, aic, ks_stat, ks_p)

    if criterion == "aic":
        best = min(fitted, key=lambda k: fitted[k][1])
    else:  # ks
        best = min(fitted, key=lambda k: fitted[k][2])
    params, aic, ks_stat, ks_p = fitted[best]
    return MarginalFit(
        asset_name=name, distribution=best, params=params,
        aic=aic, ks_statistic=ks_stat, ks_pvalue=ks_p,
    )


def fit_marginals(
    returns: pd.DataFrame,
    candidates: tuple[str, ...] = ("normal", "student_t", "empirical"),
    criterion: str = "aic",
) -> dict:
    """Fit the best marginal per column (using :func:`fit_marginal`)."""
    if returns.isna().any().any():
        raise ValueError("fit_marginals requires a fully-observed DataFrame")
    out: dict[str, MarginalFit] = {}
    for col in returns.columns:
        out[col] = fit_marginal(returns[col], candidates=candidates,
                                 criterion=criterion, name=col)
    return out


# ---------------------------------------------------------------------------
# Marginal CDFs / quantiles (vectorised)
# ---------------------------------------------------------------------------

def _marginal_cdf(x: np.ndarray, mf: MarginalFit) -> np.ndarray:
    """CDF value ``F_i(x)``, clipped to (eps, 1-eps)."""
    eps = 1e-10
    if mf.distribution == "normal":
        u = stats.norm.cdf(x, loc=mf.params["loc"], scale=mf.params["scale"])
    elif mf.distribution == "student_t":
        u = stats.t.cdf(x, df=mf.params["df"], loc=mf.params["loc"],
                        scale=mf.params["scale"])
    else:  # empirical
        sorted_vals = mf.params["sorted_values"]
        ranks = np.searchsorted(sorted_vals, x, side="right")
        u = (ranks - 0.5) / len(sorted_vals)
    return np.clip(u, eps, 1.0 - eps)


def _marginal_quantile(u: np.ndarray, mf: MarginalFit) -> np.ndarray:
    """Inverse CDF ``F_i^{-1}(u)``."""
    if mf.distribution == "normal":
        return stats.norm.ppf(u, loc=mf.params["loc"], scale=mf.params["scale"])
    if mf.distribution == "student_t":
        return stats.t.ppf(u, df=mf.params["df"], loc=mf.params["loc"],
                           scale=mf.params["scale"])
    sorted_vals = mf.params["sorted_values"]
    n = len(sorted_vals)
    # linear interpolation on sorted ECDF
    idx_cont = u * n - 0.5
    idx_lo = np.clip(np.floor(idx_cont).astype(int), 0, n - 1)
    idx_hi = np.clip(idx_lo + 1, 0, n - 1)
    frac = idx_cont - idx_lo
    return sorted_vals[idx_lo] * (1 - frac) + sorted_vals[idx_hi] * frac


# ---------------------------------------------------------------------------
# Copula fitting
# ---------------------------------------------------------------------------

def _pit_transform(returns: pd.DataFrame, marginals: dict) -> np.ndarray:
    """Return U in (0, 1)^{T×N} from PIT of each column."""
    T, N = returns.shape
    U = np.empty((T, N), dtype=np.float64)
    for j, col in enumerate(returns.columns):
        U[:, j] = _marginal_cdf(returns[col].to_numpy(dtype=np.float64),
                                 marginals[col])
    return U


def _fit_gaussian_copula(U: np.ndarray, asset_order: list[str]) -> CopulaFit:
    """Gaussian copula ≡ correlation of ``Φ^{-1}(U)``."""
    Z = stats.norm.ppf(U)
    if Z.shape[1] == 1:
        R = np.array([[1.0]])
    else:
        R = np.corrcoef(Z, rowvar=False)
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
    return CopulaFit(
        copula_type="gaussian",
        correlation=R,
        df=None,
        asset_order=list(asset_order),
        n_samples=U.shape[0],
    )


def _t_copula_loglik(U: np.ndarray, R: np.ndarray, df: float) -> float:
    """Student-t copula log-likelihood at a given ν."""
    Z = stats.t.ppf(U, df=df)
    try:
        L = cholesky(R, lower=True)
    except np.linalg.LinAlgError:
        L = cholesky(R + 1e-8 * np.eye(R.shape[0]), lower=True)
    # mahalanobis via triangular solve
    y = np.linalg.solve(L, Z.T).T
    mahal = np.einsum("ij,ij->i", y, y)
    N = R.shape[0]
    log_det_R = 2 * np.sum(np.log(np.diag(L)))
    # Multivariate t log-density
    const = (
        gammaln(0.5 * (df + N))
        - gammaln(0.5 * df)
        - 0.5 * N * np.log(df * np.pi)
        - 0.5 * log_det_R
    )
    mv_log = const - 0.5 * (df + N) * np.log1p(mahal / df)
    # Subtract marginal t-log-pdfs
    marg_log = stats.t.logpdf(Z, df=df).sum(axis=1)
    return float((mv_log - marg_log).sum())


def _fit_t_copula(U: np.ndarray, asset_order: list[str]) -> CopulaFit:
    """Student-t copula: grid-search ν over a realistic range."""
    Z_std = stats.norm.ppf(U)
    if Z_std.shape[1] == 1:
        R0 = np.array([[1.0]])
    else:
        R0 = np.corrcoef(Z_std, rowvar=False)
        R0 = 0.5 * (R0 + R0.T)
        np.fill_diagonal(R0, 1.0)

    df_grid = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])
    best_df = df_grid[0]
    best_ll = -np.inf
    for df in df_grid:
        ll = _t_copula_loglik(U, R0, df)
        if ll > best_ll:
            best_ll, best_df = ll, float(df)
    return CopulaFit(
        copula_type="student_t",
        correlation=R0,
        df=best_df,
        asset_order=list(asset_order),
        n_samples=U.shape[0],
    )


def fit_copula(
    returns: pd.DataFrame,
    marginals: dict,
    copula_type: str = "gaussian",
) -> CopulaFit:
    """Fit a Gaussian or Student-t copula on PIT-transformed returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Fully-observed returns matrix.
    marginals : dict[str, MarginalFit]
    copula_type : {'gaussian', 'student_t'}

    Returns
    -------
    CopulaFit
    """
    if copula_type not in ("gaussian", "student_t"):
        raise ValueError(
            f"copula_type must be 'gaussian' or 'student_t', got {copula_type!r}"
        )
    if returns.isna().any().any():
        raise ValueError("fit_copula requires a fully-observed DataFrame")
    for col in returns.columns:
        if col not in marginals:
            raise ValueError(f"marginals is missing column {col!r}")

    U = _pit_transform(returns, marginals)
    order = list(returns.columns)
    if copula_type == "gaussian":
        return _fit_gaussian_copula(U, order)
    return _fit_t_copula(U, order)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_copula(
    copula: CopulaFit,
    marginals: dict,
    n_simulations: int = 10_000,
    horizon: int = 252,
    seed: int = 0,
) -> CopulaSimResult:
    """Draw ``n_simulations`` paths of length ``horizon`` from the copula model.

    Parameters
    ----------
    copula : CopulaFit
    marginals : dict[str, MarginalFit]
        Must include every asset in ``copula.asset_order``.
    n_simulations : int
    horizon : int
        Number of days per scenario.
    seed : int

    Returns
    -------
    CopulaSimResult

    Notes
    -----
    Total samples drawn = ``n_simulations × horizon`` from the latent joint
    distribution, then PIT-inverted through the per-asset marginal quantiles.
    """
    rng = npr.default_rng(seed)
    names = copula.asset_order
    N = len(names)
    S, T = int(n_simulations), int(horizon)

    # Latent joint draws (flattened over scenarios × days)
    total = S * T
    try:
        L = cholesky(copula.correlation, lower=True)
    except np.linalg.LinAlgError:
        L = cholesky(copula.correlation + 1e-10 * np.eye(N), lower=True)
    Z = rng.standard_normal(size=(total, N)) @ L.T          # N(0, R)

    if copula.copula_type == "gaussian":
        U = stats.norm.cdf(Z)
    else:  # student_t
        chi2 = rng.chisquare(copula.df, size=total)
        t_sample = Z / np.sqrt(chi2 / copula.df)[:, None]    # multivariate t
        U = stats.t.cdf(t_sample, df=copula.df)

    U = np.clip(U, 1e-10, 1.0 - 1e-10)

    # Transform to returns per marginal
    X = np.empty_like(U)
    for j, name in enumerate(names):
        X[:, j] = _marginal_quantile(U[:, j], marginals[name])

    simulated = X.reshape(S, T, N)

    return CopulaSimResult(
        simulated_returns=simulated,
        marginals={k: marginals[k] for k in names},
        copula=copula,
        asset_names=list(names),
        seed=seed,
    )
