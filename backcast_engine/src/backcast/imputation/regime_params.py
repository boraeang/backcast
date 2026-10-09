"""Per-regime imputation parameters with a graceful fallback cascade.

Both the single (:func:`backcast.imputation.single_impute.regime_single_impute`)
and multiple (:func:`backcast.imputation.multiple_impute.multiple_impute_regime`)
regime-conditional imputers consume the output of :func:`build_regime_params`,
so their treatment of thin regimes cannot drift apart.

For each regime ``k`` with ``n_k`` overlap observations, ``N_short``
short-history assets and ``τ = reliable_threshold`` (default
``max(3·N_short, 60)``):

=====================  ==========  =============================================
condition              source      parameters
=====================  ==========  =============================================
``n_k >= τ``           ``full``    regime's own ``(μ_k, Σ_k)``
``N_short < n_k < τ``  ``shrunk``  ``λ·(μ, Σ)_pooled + (1−λ)·(μ_k, Σ_k)``
``n_k <= N_short``     ``pooled``  unconditional ``(μ, Σ)`` from all overlap rows
=====================  ==========  =============================================

with ``λ = τ / (τ + n_k)`` for a fixed rule, or the Ledoit-Wolf intensity
toward the pooled target when ``shrinkage="auto"``.  Shrinkage is applied to
the **joint** covariance (not only the conditional block) so that both
``β = Σ₂₁Σ₁₁⁻¹`` and ``Σ₂|₁`` are well defined even when ``n_k <= N_long``.
Every covariance passes through :func:`nearest_psd` before any Cholesky.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from backcast.exceptions import BackcastDataError

logger = logging.getLogger(__name__)

SOURCES: tuple[str, ...] = ("full", "shrunk", "pooled")


@dataclass
class RegimeParams:
    """Imputation parameters for one regime.

    Attributes
    ----------
    mu : np.ndarray, shape (N,)
        Joint mean over all assets (column order of the overlap matrix).
    sigma : np.ndarray, shape (N, N)
        Joint covariance, guaranteed PSD (eigenvalues ``>= psd_epsilon``).
        Imputers condition on it per missingness pattern, which also covers
        staggered short-asset start dates.
    sigma_cond : np.ndarray, shape (N_short, N_short)
        ``Σ₂|₁ = Σ₂₂ − Σ₂₁Σ₁₁⁻¹Σ₁₂`` of short assets given long assets, PSD.
    beta : np.ndarray, shape (N_short, N_long)
        ``Σ₂₁Σ₁₁⁻¹``.
    alpha : np.ndarray, shape (N_short,)
        ``μ₂ − β μ₁``.
    source : str
        ``"full"``, ``"shrunk"`` or ``"pooled"``.
    n_obs : int
        Overlap observations labelled with this regime.
    shrinkage : float
        Weight ``λ`` on the pooled parameters (0 for ``full``, 1 for ``pooled``).
    """

    mu: np.ndarray
    sigma: np.ndarray
    sigma_cond: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    source: str
    n_obs: int
    shrinkage: float


def nearest_psd(sigma: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Clip eigenvalues to ``>= epsilon`` to guarantee a PSD matrix.

    Parameters
    ----------
    sigma : np.ndarray, shape (N, N)
        Symmetric (or nearly symmetric) matrix.
    epsilon : float
        Eigenvalue floor.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric matrix whose eigenvalues are all ``>= epsilon``.
    """
    sym = 0.5 * (sigma + sigma.T)
    vals, vecs = np.linalg.eigh(sym)
    out = (vecs * np.clip(vals, epsilon, None)) @ vecs.T
    return 0.5 * (out + out.T)


def default_reliable_threshold(n_short: int) -> int:
    """``max(3 · n_short, 60)`` — overlap rows needed for a ``full`` regime."""
    return max(3 * n_short, 60)


def _conditional_split(
    mu: np.ndarray,
    sigma: np.ndarray,
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    psd_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(beta, alpha, sigma_cond)`` of short given long assets."""
    S11 = sigma[np.ix_(long_idx, long_idx)]
    S12 = sigma[np.ix_(long_idx, short_idx)]
    S22 = sigma[np.ix_(short_idx, short_idx)]
    if len(long_idx) == 0:
        return (np.zeros((len(short_idx), 0)), mu[short_idx].copy(),
                nearest_psd(S22, psd_epsilon))
    c = cho_factor(S11, lower=True)
    beta = cho_solve(c, S12).T
    alpha = mu[short_idx] - beta @ mu[long_idx]
    sigma_cond = nearest_psd(S22 - beta @ S12, psd_epsilon)
    return beta, alpha, sigma_cond


def _shrinkage_intensity(
    block: np.ndarray,
    sigma_raw: np.ndarray,
    sigma_pooled: np.ndarray,
    n_k: int,
    reliable_threshold: int,
    shrinkage: Union[str, float],
) -> float:
    """``λ`` for the shrunk tier: Ledoit-Wolf toward pooled, or fixed."""
    if shrinkage == "auto":
        from backcast.downstream.covariance import _ledoit_wolf_alpha

        lam, _ = _ledoit_wolf_alpha(block, sigma_raw, target=sigma_pooled)
        return lam
    if shrinkage is None or shrinkage == "heuristic":
        return reliable_threshold / (reliable_threshold + n_k)
    return float(shrinkage)


def build_regime_params(
    overlap: pd.DataFrame,
    overlap_labels: np.ndarray,
    short_assets: Sequence[str],
    *,
    regimes: Optional[Iterable[int]] = None,
    reliable_threshold: Optional[int] = None,
    shrinkage: Union[str, float, None] = "auto",
    fallback_to_pooled: bool = True,
    psd_epsilon: float = 1e-10,
) -> dict[int, RegimeParams]:
    """Build usable parameters for every regime via the full/shrunk/pooled cascade.

    Parameters
    ----------
    overlap : pd.DataFrame, shape (T_overlap, N)
        Fully observed returns of ALL assets over the overlap period.
    overlap_labels : np.ndarray, shape (T_overlap,)
        Regime label of each overlap row.
    short_assets : sequence of str
        Short-history columns of *overlap*; the rest are treated as long.
    regimes : iterable of int, optional
        Every regime that appears anywhere in the data (typically
        ``np.unique(hmm.regime_labels)`` over the full history).  Regimes
        absent from the overlap get ``pooled`` parameters.  Defaults to the
        regimes present in *overlap_labels*.
    reliable_threshold : int, optional
        Overlap rows needed for ``full``; ``None`` →
        :func:`default_reliable_threshold`.
    shrinkage : {"auto", "heuristic"} or float or None
        ``"auto"``: Ledoit-Wolf intensity toward the pooled covariance.
        ``"heuristic"`` / ``None``: ``λ = τ / (τ + n_k)``.
        float in ``[0, 1]``: fixed ``λ``.
    fallback_to_pooled : bool
        If False, a regime with ``n_k <= N_short`` raises instead of being
        pooled.
    psd_epsilon : float
        Eigenvalue floor applied by :func:`nearest_psd` at every level.

    Returns
    -------
    dict[int, RegimeParams]
        One entry per regime in *regimes* — never missing, never NaN.

    Raises
    ------
    ValueError
        On misaligned inputs, NaN in *overlap*, unknown short assets, or an
        invalid *shrinkage*.
    BackcastDataError
        If a regime needs pooling and *fallback_to_pooled* is False.
    """
    overlap_labels = np.asarray(overlap_labels)
    if len(overlap_labels) != len(overlap):
        raise ValueError(
            f"overlap_labels length {len(overlap_labels)} != overlap rows {len(overlap)}"
        )
    if overlap.isna().any().any():
        raise ValueError("overlap must be fully observed to estimate regime params")
    columns = list(overlap.columns)
    missing = [a for a in short_assets if a not in columns]
    if missing:
        raise ValueError(f"short_assets not in overlap columns: {missing}")
    if isinstance(shrinkage, (int, float)) and not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage}")
    if isinstance(shrinkage, str) and shrinkage not in ("auto", "heuristic"):
        raise ValueError(f"shrinkage must be 'auto', 'heuristic' or a float, got {shrinkage!r}")

    short_set = set(short_assets)
    short_idx = np.array([i for i, c in enumerate(columns) if c in short_set], dtype=np.int64)
    long_idx = np.array([i for i, c in enumerate(columns) if c not in short_set], dtype=np.int64)
    n_short = len(short_idx)
    tau = int(reliable_threshold) if reliable_threshold is not None else default_reliable_threshold(n_short)

    R = overlap.to_numpy(dtype=np.float64)
    mu_pooled = R.mean(axis=0)
    sigma_pooled = nearest_psd(np.atleast_2d(np.cov(R, rowvar=False, bias=False)), psd_epsilon)

    regime_ids = sorted({int(k) for k in (np.unique(overlap_labels) if regimes is None else regimes)})
    out: dict[int, RegimeParams] = {}
    for k in regime_ids:
        mask = overlap_labels == k
        n_k = int(mask.sum())
        if n_k >= tau:
            source, lam = "full", 0.0
        elif n_k > n_short and n_k >= 2:
            source = "shrunk"
        else:
            source, lam = "pooled", 1.0

        if source == "pooled":
            if not fallback_to_pooled:
                raise BackcastDataError(
                    f"Regime {k} has {n_k} overlap observations (<= N_short={n_short}) "
                    "and regime_fallback_to_pooled is disabled."
                )
            logger.warning(
                "Regime %d has only %d overlap observations (<= N_short=%d) — "
                "using pooled (unconditional) parameters.", k, n_k, n_short,
            )
            mu, sigma = mu_pooled.copy(), sigma_pooled.copy()
        else:
            block = R[mask]
            mu_raw = block.mean(axis=0)
            sigma_raw = np.atleast_2d(np.cov(block, rowvar=False, bias=False))
            if source == "full":
                mu, sigma = mu_raw, sigma_raw
            else:
                lam = _shrinkage_intensity(block, sigma_raw, sigma_pooled, n_k, tau, shrinkage)
                mu = lam * mu_pooled + (1.0 - lam) * mu_raw
                sigma = lam * sigma_pooled + (1.0 - lam) * sigma_raw
                logger.info(
                    "Regime %d: %d overlap observations (< %d) — shrunk toward "
                    "pooled with lambda=%.3f.", k, n_k, tau, lam,
                )
            sigma = nearest_psd(sigma, psd_epsilon)

        beta, alpha, sigma_cond = _conditional_split(mu, sigma, long_idx, short_idx, psd_epsilon)
        out[k] = RegimeParams(
            mu=mu, sigma=sigma, sigma_cond=sigma_cond, beta=beta, alpha=alpha,
            source=source, n_obs=n_k, shrinkage=float(lam),
        )
    logger.info(
        "Regime params: %s",
        ", ".join(f"k={k}:{p.source}(n={p.n_obs})" for k, p in out.items()),
    )
    return out


def as_regime_params(params: Union[RegimeParams, dict]) -> RegimeParams:
    """Accept a :class:`RegimeParams` or a legacy ``{'mu', 'sigma'}`` dict.

    Parameters
    ----------
    params : RegimeParams or dict
        Legacy dicts (from older ``compute_regime_params`` callers) are
        wrapped with ``source="full"`` and an empty conditional split.

    Returns
    -------
    RegimeParams
    """
    if isinstance(params, RegimeParams):
        return params
    mu = np.asarray(params["mu"], dtype=np.float64)
    return RegimeParams(
        mu=mu,
        sigma=np.asarray(params["sigma"], dtype=np.float64),
        sigma_cond=np.zeros((0, 0)),
        beta=np.zeros((0, 0)),
        alpha=np.zeros(0),
        source=str(params.get("source", "full")),
        n_obs=int(params.get("n_obs", 0)),
        shrinkage=float(params.get("shrinkage", 0.0)),
    )


def check_regime_coverage(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    regime_params: dict,
) -> None:
    """Raise if any row with missing entries has a regime without params.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    regime_labels : np.ndarray, shape (T,)
    regime_params : dict[int, RegimeParams]

    Raises
    ------
    BackcastDataError
        Listing the uncovered regimes; use :func:`build_regime_params` with
        ``regimes=np.unique(regime_labels)`` to cover every regime.
    """
    needs_fill = np.isnan(returns).any(axis=1)
    uncovered = sorted(set(np.unique(regime_labels[needs_fill]).tolist()) - set(regime_params))
    if uncovered:
        raise BackcastDataError(
            f"Regimes {uncovered} label rows that need imputation but have no "
            "parameters; build them with build_regime_params(..., "
            "regimes=np.unique(regime_labels))."
        )


@dataclass
class RegimeSourceSummary:
    """Which parameter tier filled each imputed cell.

    Attributes
    ----------
    cell_source : pd.DataFrame, shape (T, N_imputed)
        One column per asset with at least one imputed cell; values are
        ``"full"``/``"shrunk"``/``"pooled"`` where the cell was imputed and
        ``None`` where it was observed.
    fill_source : pd.Series
        Source per imputed date (rows with at least one missing cell).
    breakdown : pd.DataFrame, shape (N_imputed, 3)
        Count of imputed dates per asset (rows) and source (columns, always
        ``full``, ``shrunk``, ``pooled`` in that order).
    """

    cell_source: pd.DataFrame
    fill_source: pd.Series
    breakdown: pd.DataFrame

    def fractions(self) -> pd.DataFrame:
        """Per-asset share of imputed dates by source (rows sum to 1)."""
        totals = self.breakdown.sum(axis=1).replace(0, 1)
        return self.breakdown.div(totals, axis=0)


def summarize_regime_sources(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    regime_params: dict,
) -> RegimeSourceSummary:
    """Tag every imputed cell with the source of the params that fill it.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Returns matrix *before* imputation (NaN = to be imputed).
    regime_labels : np.ndarray, shape (T,)
        Regime label of every row.
    regime_params : dict[int, RegimeParams]
        Must cover every regime labelling a row with missing entries.

    Returns
    -------
    RegimeSourceSummary
        Also logged at INFO as a per-asset, per-source table.
    """
    regime_labels = np.asarray(regime_labels)
    source_of = {k: as_regime_params(p).source for k, p in regime_params.items()}
    row_source = np.array([source_of.get(int(k)) for k in regime_labels], dtype=object)
    nan_mask = returns.isna()
    imputed_cols = [c for c in returns.columns if nan_mask[c].any()]

    cells = np.where(nan_mask[imputed_cols].to_numpy(), row_source[:, None], None)
    cell_source = pd.DataFrame(cells, index=returns.index, columns=imputed_cols, dtype=object)
    any_missing = nan_mask.any(axis=1).to_numpy()
    fill_source = pd.Series(row_source[any_missing], index=returns.index[any_missing],
                            name="source", dtype=object)
    breakdown = pd.DataFrame(
        {s: (cell_source == s).sum(axis=0) for s in SOURCES},
        index=pd.Index(imputed_cols, name="asset"),
    ).astype(np.int64)
    logger.info("Imputed dates by parameter source (per asset):\n%s", breakdown.to_string())
    return RegimeSourceSummary(cell_source=cell_source, fill_source=fill_source, breakdown=breakdown)
