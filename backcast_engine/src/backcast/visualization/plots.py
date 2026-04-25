"""Diagnostic and result plots for the backcast engine.

Every function returns a ``matplotlib.figure.Figure`` — the caller is
responsible for saving or displaying it.  Agg backend is set at import time so
these work in non-interactive environments.

Implements the 11 plots named in the spec:

1. :func:`plot_missingness`
2. :func:`plot_em_convergence`
3. :func:`plot_kalman_betas`
4. :func:`plot_regime_timeline`
5. :func:`plot_backcast_fan`
6. :func:`plot_correlation_comparison`
7. :func:`plot_holdout_scatter`
8. :func:`plot_pit_histogram`
9. :func:`plot_eigenvalue_spectrum`
10. :func:`plot_uncertainty_ellipses`
11. :func:`plot_backtest_fan`
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

import matplotlib

# If pyplot has already been imported (e.g. because the user ran
# `%matplotlib inline` in a notebook), respect that choice.  Otherwise fall
# back to the Agg backend so these functions work in headless scripts/tests.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Missingness
# ---------------------------------------------------------------------------

def plot_missingness(returns: pd.DataFrame, *, figsize=(10, 4)) -> Figure:
    """Heatmap of NaN mask: ``True`` (NaN) in dark, observed in light."""
    fig, ax = plt.subplots(figsize=figsize)
    missing = returns.isna().to_numpy().astype(int).T
    im = ax.imshow(missing, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(returns.shape[1]))
    ax.set_yticklabels(list(returns.columns))
    ax.set_xlabel("Row index (time)")
    ax.set_title(f"Missingness pattern  ({returns.isna().values.mean()*100:.1f}% NaN)")
    plt.colorbar(im, ax=ax, label="NaN (1) / observed (0)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. EM convergence
# ---------------------------------------------------------------------------

def plot_em_convergence(em_result, *, figsize=(8, 4)) -> Figure:
    """Log-likelihood trace vs iteration."""
    fig, ax = plt.subplots(figsize=figsize)
    ll = np.asarray(em_result.log_likelihood_trace)
    if len(ll) == 0:
        ax.text(0.5, 0.5, "log-likelihood trace is empty\n"
                "(track_loglikelihood=False)",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(ll, "o-", ms=4)
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("Observed-data log-likelihood")
        ax.set_title(
            f"EM convergence  ({em_result.n_iter} iters, Δℓ = {ll[-1] - ll[0]:+.2f})"
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Kalman β evolution
# ---------------------------------------------------------------------------

def plot_kalman_betas(kalman_multi, *, figsize=(12, 8)) -> Figure:
    """Grid of smoothed β time series per short asset."""
    short_assets = list(kalman_multi.short_assets)
    n = len(short_assets)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no short assets", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, name in enumerate(short_assets):
        ax = axes[i // ncols][i % ncols]
        path = kalman_multi.smoothed_betas[name]
        cov_path = kalman_multi.per_asset[name].smoothed_state_cov
        state_names = list(path.columns)
        for j, col in enumerate(state_names):
            series = path[col]
            std = np.sqrt(cov_path[:, j, j])
            ax.plot(series.index, series.values, label=col, lw=1)
            ax.fill_between(series.index,
                            series.values - 1.96 * std,
                            series.values + 1.96 * std,
                            alpha=0.15)
        ax.set_title(name)
        ax.set_ylabel("β")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    # Hide unused subplots
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Kalman smoothed factor loadings (95 % CI shaded)", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Regime timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    regime_labels: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    figsize=(12, 2),
) -> Figure:
    """Colour-coded bar showing regime at each date."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = np.asarray(regime_labels)
    K = int(labels.max()) + 1
    cmap = plt.get_cmap("RdYlGn_r", K)
    ax.imshow(
        labels[None, :], aspect="auto", cmap=cmap,
        extent=[0, len(labels), 0, 1], interpolation="nearest",
    )
    # Label x-axis with dates
    n_ticks = min(8, len(dates))
    positions = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(positions)
    ax.set_xticklabels([dates[p].strftime("%Y-%m") for p in positions], rotation=30)
    ax.set_yticks([])
    ax.set_title(f"Regime timeline  (K = {K})")
    # Colour-bar with regime codes
    cbar = plt.colorbar(ax.images[0], ax=ax, ticks=list(range(K)))
    cbar.set_label("Regime")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Backcast fan chart
# ---------------------------------------------------------------------------

def plot_backcast_fan(
    mi_result,
    *,
    actual: Optional[pd.DataFrame] = None,
    asset: Optional[str] = None,
    confidence: float = 0.95,
    figsize=(12, 4),
) -> Figure:
    """Median / lower / upper backcast with optional actual overlay."""
    from backcast.imputation.multiple_impute import prediction_intervals

    median, lower, upper = prediction_intervals(mi_result, confidence=confidence)
    if asset is None:
        # pick the first imputed column that has NaN in the original
        candidates = [c for c in median.columns
                      if mi_result.imputations[0][c].notna().all()]
        # all imputations have the same mask — prefer any short asset
        asset = median.columns[-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(median.index, median[asset], "-", color="C0",
            label="Median imputation", lw=1.2)
    ax.fill_between(median.index, lower[asset], upper[asset],
                    color="C0", alpha=0.2,
                    label=f"{int(confidence*100)}% interval")
    if actual is not None and asset in actual.columns:
        ax.plot(actual.index, actual[asset], "k-", lw=0.6, alpha=0.5,
                label="Actual (complete)")
    ax.set_title(f"Backcast fan chart — {asset}")
    ax.set_ylabel("Daily return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Correlation matrix comparison
# ---------------------------------------------------------------------------

def plot_correlation_comparison(
    corr_a: np.ndarray,
    corr_b: np.ndarray,
    *,
    labels: Optional[list[str]] = None,
    titles: tuple[str, str] = ("Overlap", "Backcast"),
    figsize=(10, 4),
) -> Figure:
    """Side-by-side heatmaps with a shared colour scale."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [4, 4, 4]})
    vmin = min(corr_a.min(), corr_b.min(), -1.0)
    vmax = max(corr_a.max(), corr_b.max(), 1.0)
    for ax, mat, title in zip(axes[:2], (corr_a, corr_b), titles):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
    diff = corr_b - corr_a
    mag = np.max(np.abs(diff)) if diff.size else 1.0
    im3 = axes[2].imshow(diff, cmap="PiYG", vmin=-mag, vmax=mag, aspect="auto")
    axes[2].set_title("Δ (B − A)")
    if labels is not None:
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels(labels, rotation=45, fontsize=8)
        axes[2].set_yticks(range(len(labels)))
        axes[2].set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=axes[:2], shrink=0.8)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Holdout scatter
# ---------------------------------------------------------------------------

def plot_holdout_scatter(holdout_report, *, figsize=(10, 8)) -> Figure:
    """Scatter of predicted vs actual per short asset across all windows."""
    windows = holdout_report.windows
    assets = holdout_report.short_assets
    n = len(assets)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, name in enumerate(assets):
        ax = axes[i // ncols][i % ncols]
        actuals = pd.concat([w.actual[name] for w in windows]).to_numpy()
        preds = pd.concat([w.predicted[name] for w in windows]).to_numpy()
        ax.scatter(actuals, preds, s=4, alpha=0.3)
        lim = float(max(np.abs(actuals).max(), np.abs(preds).max()))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Holdout: predicted vs actual (pooled across windows)", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. PIT histogram
# ---------------------------------------------------------------------------

def plot_pit_histogram(pit_values: np.ndarray, *, bins: int = 20, figsize=(6, 4)) -> Figure:
    """Histogram of PIT values with the uniform reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    vals = np.asarray(pit_values).ravel()
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, range=(0, 1), edgecolor="k",
            alpha=0.7, density=True)
    ax.axhline(1.0, color="r", lw=1.2, ls="--", label="Uniform(0,1)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram  (n = {len(vals)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Eigenvalue spectrum vs Marchenko-Pastur
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    covariance: np.ndarray,
    T: int,
    *,
    figsize=(8, 4),
) -> Figure:
    """Eigenvalue histogram with MP upper bound.

    Works on the correlation matrix derived from *covariance* so bars are
    comparable across asset classes.
    """
    vols = np.sqrt(np.clip(np.diag(covariance), 1e-30, None))
    corr = covariance / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    N = len(eig)
    q = N / max(T, 1)
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    lam_minus = (1.0 - np.sqrt(q)) ** 2

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(N), eig, color="C0", edgecolor="k", alpha=0.75)
    ax.axhline(lam_plus, color="r", lw=1.2, ls="--",
               label=f"MP λ₊ = (1+√q)² = {lam_plus:.2f}")
    ax.axhline(lam_minus, color="r", lw=1.0, ls=":",
               label=f"MP λ₋ = (1-√q)² = {lam_minus:.2f}")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue (correlation)")
    ax.set_title(f"Eigenvalue spectrum  (N={N}, T={T}, q={q:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Uncertainty ellipses (μ)
# ---------------------------------------------------------------------------

def plot_uncertainty_ellipses(
    ellipse,
    *,
    pair: Optional[tuple[str, str]] = None,
    figsize=(6, 6),
) -> Figure:
    """2-D projection of the ellipsoidal μ uncertainty set.

    The scatter points are per-imputation means (if attached to the result);
    here we just draw the centre and the ellipse.
    """
    names = list(ellipse.asset_names)
    if pair is None:
        pair = (names[0], names[1]) if len(names) >= 2 else (names[0], names[0])
    i = names.index(pair[0])
    j = names.index(pair[1])

    mu = ellipse.mu_center[[i, j]]
    S = ellipse.scaling[np.ix_([i, j], [i, j])]
    kappa = ellipse.kappa

    # Parameterise ellipse: {mu + κ · S^{1/2} · u : ||u|| = 1}
    eigvals, eigvecs = np.linalg.eigh(S)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    pts = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ circle * kappa
    pts = pts + mu[:, None]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pts[0], pts[1], "C0-", lw=1.3)
    ax.plot(mu[0], mu[1], "ro", ms=6, label="μ̄ (centre)")
    ax.set_xlabel(f"μ({pair[0]})")
    ax.set_ylabel(f"μ({pair[1]})")
    ax.set_title(
        f"Ellipsoidal μ uncertainty set  "
        f"(κ = {kappa:.2f}, confidence = {ellipse.confidence:.0%})"
    )
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11. Backtest fan chart
# ---------------------------------------------------------------------------

def plot_backtest_fan(backtest_result, *, figsize=(12, 5)) -> Figure:
    """Median cumulative path with 5/95 % bands across imputations."""
    fig, ax = plt.subplots(figsize=figsize)
    idx = backtest_result.cumulative_median.index
    ax.plot(idx, backtest_result.cumulative_median.values, "C0-",
            lw=1.4, label="Median")
    ax.fill_between(idx,
                    backtest_result.cumulative_p05.values,
                    backtest_result.cumulative_p95.values,
                    color="C0", alpha=0.2, label="5–95 % band")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative equity (starts at 1)")
    sharpes = backtest_result.sharpe_distribution
    ax.set_title(
        f"Backtest fan: {backtest_result.strategy_name}  "
        f"(M={backtest_result.n_imputations}, Sharpe median={np.median(sharpes):+.2f})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
