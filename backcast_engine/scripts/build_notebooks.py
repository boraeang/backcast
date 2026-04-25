"""Build (and optionally execute) the four demo notebooks.

Usage
-----
    python scripts/build_notebooks.py          # build + execute
    python scripts/build_notebooks.py --no-run # just build (empty outputs)

The generated notebooks live in ``backcast_engine/notebooks/`` and reference
the Tier 2 synthetic data from ``synthetic_data_generator/output/tier2/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJ_ROOT / "notebooks"


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


# Preamble: add the src directory to sys.path so the notebook finds the package
# without requiring an editable install.  Also configures matplotlib inline.
PREAMBLE = [
    "# --- preamble ----------------------------------------------------------",
    "%matplotlib inline",
    "import sys",
    "from pathlib import Path",
    "sys.path.insert(0, str(Path.cwd().parent / \"src\"))",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "",
    "pd.set_option(\"display.precision\", 4)",
    "pd.set_option(\"display.width\", 120)",
    "",
    "# Path to Tier 2 synthetic data (relative to this notebook)",
    "TIER2_DIR = Path.cwd().parent.parent / \"synthetic_data_generator\" / \"output\" / \"tier2\"",
    "TIER2_CSV = TIER2_DIR / \"returns.csv\"",
    "TIER2_COMPLETE = TIER2_DIR / \"returns_complete.csv\"",
    "TIER2_GT = TIER2_DIR / \"ground_truth.json\"",
    "assert TIER2_CSV.exists(), f\"Tier 2 fixture missing: {TIER2_CSV}\"",
]


# ---------------------------------------------------------------------------
# Notebook 1: Exploratory Data Analysis
# ---------------------------------------------------------------------------

def build_eda() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 01 · Exploratory Data Analysis",
            "",
            "Inspect the Tier 2 synthetic returns dataset — a regime-switching",
            "multivariate Gaussian DGP with 5 long-history assets and 3 short-history",
            "assets that start at row 3000.",
            "",
            "This notebook covers: data loading, missingness pattern, per-asset",
            "statistics, correlations, and rolling volatility.",
        ),
        code(*PREAMBLE),
        md("## Load the dataset"),
        code(
            "from backcast.data.loader import load_backcast_dataset",
            "",
            "ds = load_backcast_dataset(TIER2_CSV)",
            "",
            "print(f\"Total shape       : {ds.returns_full.shape}\")",
            "print(f\"Long  ({ds.n_long:>2}) assets : {ds.long_assets}\")",
            "print(f\"Short ({ds.n_short:>2}) assets : {ds.short_assets}\")",
            "print(f\"Overlap           : {ds.overlap_start.date()} → {ds.overlap_end.date()}  \"",
            "      f\"({ds.overlap_length} rows)\")",
            "print(f\"Backcast          : first {ds.backcast_length} rows\")",
            "print(f\"Short start indices: {ds.short_start_indices}\")",
        ),
        md(
            "## Missingness pattern",
            "",
            "The Tier 2 data has a classic monotone block: long assets observed for",
            "all 5000 rows, short assets starting simultaneously at row 3000.",
        ),
        code(
            "from backcast.visualization.plots import plot_missingness",
            "fig = plot_missingness(ds.returns_full)",
            "plt.show()",
        ),
        md("## Per-asset descriptive statistics"),
        code(
            "desc = ds.returns_full.describe(percentiles=[0.05, 0.5, 0.95]).T",
            "desc[\"ann_vol_%\"] = ds.returns_full.std() * np.sqrt(252) * 100",
            "desc[\"ann_mean_%\"] = ds.returns_full.mean() * 252 * 100",
            "desc[[\"count\", \"ann_mean_%\", \"ann_vol_%\", \"5%\", \"95%\"]].round(3)",
        ),
        md(
            "## Annualized volatility — bar chart",
            "",
            "Long assets (equities / bonds / gold) sit in the 5 – 20 % range; the",
            "short-history crypto assets are in the 60 – 70 % range by construction.",
        ),
        code(
            "ann_vol = (ds.returns_full.std() * np.sqrt(252) * 100)",
            "fig, ax = plt.subplots(figsize=(8, 3))",
            "colors = [\"C0\"] * ds.n_long + [\"C3\"] * ds.n_short",
            "ax.bar(range(len(ann_vol)), ann_vol.values, color=colors, edgecolor=\"k\")",
            "ax.set_xticks(range(len(ann_vol)))",
            "ax.set_xticklabels(ann_vol.index, rotation=30, ha=\"right\")",
            "ax.set_ylabel(\"Annualized vol (%)\")",
            "ax.set_title(\"Annualized volatility per asset (long=blue, short=red)\")",
            "ax.grid(axis=\"y\", alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Correlation matrix (overlap period)"),
        code(
            "overlap_corr = ds.overlap_matrix.corr()",
            "fig, ax = plt.subplots(figsize=(6, 5))",
            "im = ax.imshow(overlap_corr.values, cmap=\"RdBu_r\", vmin=-1, vmax=1)",
            "ax.set_xticks(range(len(overlap_corr)))",
            "ax.set_xticklabels(overlap_corr.columns, rotation=45, ha=\"right\")",
            "ax.set_yticks(range(len(overlap_corr)))",
            "ax.set_yticklabels(overlap_corr.columns)",
            "for i in range(len(overlap_corr)):",
            "    for j in range(len(overlap_corr)):",
            "        ax.text(j, i, f\"{overlap_corr.iat[i,j]:.2f}\",",
            "                ha=\"center\", va=\"center\", fontsize=7,",
            "                color=\"white\" if abs(overlap_corr.iat[i,j]) > 0.5 else \"black\")",
            "plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)",
            "ax.set_title(\"Overlap-period correlation matrix\")",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## Rolling volatility",
            "",
            "63-day rolling annualized vol for long and short assets.  Peaks visible",
            "in both groups mark the crisis-regime draws — this is exactly what the",
            "HMM should detect in the next notebook.",
        ),
        code(
            "rolling_vol = ds.returns_full.rolling(63).std() * np.sqrt(252) * 100",
            "fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)",
            "rolling_vol[ds.long_assets].plot(ax=axes[0], lw=0.8)",
            "rolling_vol[ds.short_assets].plot(ax=axes[1], lw=0.8)",
            "axes[0].set_title(\"Long assets — 63d rolling annualized vol (%)\")",
            "axes[1].set_title(\"Short assets — 63d rolling annualized vol (%)\")",
            "for ax in axes:",
            "    ax.grid(alpha=0.3)",
            "    ax.legend(fontsize=8, loc=\"upper right\")",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Cumulative returns"),
        code(
            "filled_forward = ds.returns_full.fillna(0.0)  # treat leading NaNs as zero-growth",
            "cum = (1.0 + filled_forward).cumprod()",
            "fig, ax = plt.subplots(figsize=(11, 4))",
            "for col in ds.returns_full.columns:",
            "    ax.plot(cum.index, cum[col], label=col, lw=0.8)",
            "ax.set_yscale(\"log\")",
            "ax.set_title(\"Cumulative returns (log scale)\")",
            "ax.legend(fontsize=8, ncol=4)",
            "ax.grid(alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## Summary",
            "",
            "- **Long assets** (equities, bonds, gold) have annualized vol 3 – 18 %,",
            "  plausible for those asset classes.",
            "- **Short assets** (crypto, alt) are high-vol by design (25 – 70 % ann.).",
            "- Missingness is monotone — no mid-series gaps — which is what the",
            "  backcast engine's Stambaugh EM is built for.",
            "- Rolling vol shows occasional regime-like spikes: the next notebook",
            "  will fit an HMM to identify these explicitly.",
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# Notebook 2: Model Comparison
# ---------------------------------------------------------------------------

def build_model_comparison() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 02 · Model Comparison",
            "",
            "Fit all three backcast models on Tier 2 and compare:",
            "",
            "- **Stambaugh EM** — unconditional mean and covariance",
            "- **Kalman TVP** — time-varying factor loadings",
            "- **Regime HMM** — latent regime detection",
            "",
            "The point of Tier 2 is that it has a known regime structure, so the",
            "regime HMM should win calibration checks.",
        ),
        code(*PREAMBLE),
        md("## Load"),
        code(
            "import json",
            "from backcast.data.loader import load_backcast_dataset",
            "",
            "ds = load_backcast_dataset(TIER2_CSV)",
            "complete = pd.read_csv(TIER2_COMPLETE, index_col=\"date\", parse_dates=True).astype(np.float64)",
            "with open(TIER2_GT) as fh:",
            "    gt = json.load(fh)",
            "print(f\"{ds.n_long} long + {ds.n_short} short, overlap={ds.overlap_length} rows\")",
        ),
        md("## 1.  Fit Stambaugh EM"),
        code(
            "from backcast.models.em_stambaugh import em_stambaugh",
            "from backcast.visualization.plots import plot_em_convergence",
            "",
            "em = em_stambaugh(ds.returns_full, max_iter=500, tolerance=1e-8,",
            "                   track_loglikelihood=True)",
            "print(f\"EM: {em.n_iter} iters, converged={em.converged}, \"",
            "      f\"final ΔΣ={em.final_delta:.3e}, log-L={em.log_likelihood_trace[-1]:.1f}\")",
            "",
            "fig = plot_em_convergence(em)",
            "plt.show()",
        ),
        md(
            "### Recovery vs ground truth",
            "",
            "Tier 2 ground truth gives us the true (unconditional) Σ.  EM should",
            "recover it to within a few percent on the long block.",
        ),
        code(
            "sigma_true = np.asarray(gt[\"sigma_daily\"])",
            "sigma_em = em.sigma",
            "rel_err = np.linalg.norm(sigma_em - sigma_true, \"fro\") / np.linalg.norm(sigma_true, \"fro\")",
            "print(f\"Frobenius ‖Σ_EM - Σ_true‖ / ‖Σ_true‖ = {rel_err:.4f}\")",
            "",
            "n_long = ds.n_long",
            "print(f\"  long-long block     : \"",
            "      f\"{np.linalg.norm(sigma_em[:n_long,:n_long] - sigma_true[:n_long,:n_long], 'fro') / np.linalg.norm(sigma_true[:n_long,:n_long], 'fro'):.4f}\")",
            "print(f\"  short-short block   : \"",
            "      f\"{np.linalg.norm(sigma_em[n_long:,n_long:] - sigma_true[n_long:,n_long:], 'fro') / np.linalg.norm(sigma_true[n_long:,n_long:], 'fro'):.4f}\")",
        ),
        md("## 2.  Fit Kalman TVP (per short asset)"),
        code(
            "from backcast.models.kalman_tvp import fit_kalman_all",
            "from backcast.visualization.plots import plot_kalman_betas",
            "",
            "kalman = fit_kalman_all(",
            "    ds.overlap_matrix, ds.long_assets, ds.short_assets,",
            "    state_noise_scale=0.01, use_smoother=True,",
            "    backcast_beta_method=\"mean_first_k\", backcast_beta_k=63,",
            ")",
            "print(\"Backcast loading matrix:\")",
            "print(kalman.backcast_matrix.round(3).to_string())",
            "",
            "fig = plot_kalman_betas(kalman)",
            "plt.show()",
        ),
        md(
            "The Kalman β paths are nearly flat — Tier 2's DGP has **constant** betas",
            "(the regime change scales vols, not loadings), so the Kalman correctly",
            "infers that the time-varying state barely drifts from its initial",
            "estimate.  This is the robustness check behaving as designed.",
        ),
        md("## 3.  Fit regime HMM with BIC model selection"),
        code(
            "from backcast.models.regime_hmm import fit_and_select_hmm",
            "from backcast.visualization.plots import plot_regime_timeline",
            "",
            "sel = fit_and_select_hmm(",
            "    ds.returns_full[ds.long_assets],",
            "    n_regimes_candidates=(2, 3, 4), criterion=\"bic\", seed=0,",
            ")",
            "print(f\"Selected K = {sel.best_n_regimes} via {sel.criterion.upper()}\")",
            "print(\"BIC scores:\")",
            "for K, s in sorted(sel.scores.items()):",
            "    marker = \" ←\" if K == sel.best_n_regimes else \"\"",
            "    print(f\"    K={K}:  {s:.1f}{marker}\")",
            "",
            "hmm = sel.best",
            "print(f\"\\nTransition matrix:\\n{np.array2string(hmm.transition_matrix, precision=3)}\")",
        ),
        md("### Regime-label accuracy vs ground truth"),
        code(
            "true_labels = np.asarray(gt[\"regime_labels\"])",
            "acc = (hmm.regime_labels == true_labels).mean()",
            "print(f\"Viterbi accuracy vs ground truth: {acc*100:.2f}%\")",
            "",
            "fig = plot_regime_timeline(hmm.regime_labels, ds.returns_full.index)",
            "plt.show()",
        ),
        md("## 4.  Point-imputation RMSE: EM vs regime-conditional HMM"),
        code(
            "from backcast.imputation.single_impute import single_impute",
            "from backcast.models.regime_hmm import compute_regime_params, regime_conditional_impute",
            "",
            "em_filled = single_impute(ds, em)",
            "overlap_labels = hmm.regime_labels[-ds.overlap_length:]",
            "regime_params = compute_regime_params(ds.overlap_matrix, overlap_labels)",
            "rc_filled = regime_conditional_impute(ds.returns_full, hmm.regime_labels, regime_params)",
            "",
            "backcast_dates = ds.returns_full.index[:-ds.overlap_length]",
            "short = ds.short_assets",
            "resid_em = (em_filled.loc[backcast_dates, short] - complete.loc[backcast_dates, short]).values",
            "resid_rc = (rc_filled.loc[backcast_dates, short] - complete.loc[backcast_dates, short]).values",
            "",
            "rmse_em = np.sqrt((resid_em ** 2).mean())",
            "rmse_rc = np.sqrt((resid_rc ** 2).mean())",
            "print(f\"Backcast-period RMSE (short assets only):\")",
            "print(f\"  unconditional EM      : {rmse_em:.6f}\")",
            "print(f\"  regime-conditional RC : {rmse_rc:.6f}\")",
            "print(f\"  Δ                     : {100 * (1 - rmse_rc/rmse_em):+.3f}%\")",
        ),
        md(
            "The regime-conditional RMSE is only slightly better on point estimates —",
            "as expected, because Tier 2's uniform `vol_multiplier` leaves",
            "β = Σ₂₁·Σ₁₁⁻¹ invariant.  The **real** gain of regime-conditional",
            "imputation is in variance calibration — that's covered in notebook 03.",
        ),
        md(
            "## Summary",
            "",
            "| Model | Convergence | Key output | Ground-truth check |",
            "|---|---|---|---|",
            "| Stambaugh EM | ~22 iters | (μ̂, Σ̂, β) | Σ recovery within a few % |",
            "| Kalman TVP | exact (closed form) | smoothed β paths | nearly flat → static-β DGP correctly inferred |",
            "| Regime HMM | ~200 Baum-Welch iters | regimes + transitions | Viterbi accuracy ~99 % |",
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# Notebook 3: Holdout Validation
# ---------------------------------------------------------------------------

def build_validation() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 03 · Holdout Validation",
            "",
            "Walk-forward holdout validation on the Tier 2 overlap period.  For each",
            "window we mask the short assets, refit EM, impute, and measure the error",
            "against the hidden truth.",
            "",
            "The nominal 95 % prediction interval should cover ~95 % of the actual",
            "values — this is the key calibration check.",
        ),
        code(*PREAMBLE),
        code(
            "from backcast.data.loader import load_backcast_dataset",
            "from backcast.validation.holdout import run_holdout_validation",
            "from backcast.visualization.plots import plot_holdout_scatter, plot_pit_histogram",
            "",
            "ds = load_backcast_dataset(TIER2_CSV)",
        ),
        md("## Run walk-forward holdout (3 × 504-day windows)"),
        code(
            "report = run_holdout_validation(",
            "    ds, holdout_days=504, n_windows=3, coverage_level=0.95,",
            ")",
            "print(f\"Overall coverage        : {report.overall_coverage:.4f} (nominal 0.95)\")",
            "print(f\"Correlation Frob error  : {report.overall_correlation_error:.4f}\")",
            "print(f\"Config                  : {report.config}\")",
        ),
        md("## Per-window summary"),
        code(
            "rows = []",
            "for w in report.windows:",
            "    rows.append({",
            "        \"window\": w.window_idx,",
            "        \"start\": w.start_date.date(),",
            "        \"end\": w.end_date.date(),",
            "        \"n_rows\": w.n_rows,",
            "        \"em_iter\": w.em_n_iter,",
            "        \"coverage\": w.coverage,",
            "        \"corr_err\": w.correlation_error,",
            "    })",
            "pd.DataFrame(rows).set_index(\"window\")",
        ),
        md("## Per-asset averaged metrics"),
        code("report.per_asset_mean.round(4)"),
        md(
            "### Per-window heatmap of RMSE",
            "",
            "Rows = asset, columns = window — helps spot any single-window anomalies.",
        ),
        code(
            "rmse_mat = pd.concat(",
            "    [w.per_asset[\"rmse\"].rename(f\"W{w.window_idx}\") for w in report.windows],",
            "    axis=1,",
            ")",
            "fig, ax = plt.subplots(figsize=(6, 3))",
            "im = ax.imshow(rmse_mat.values, cmap=\"viridis\", aspect=\"auto\")",
            "ax.set_xticks(range(rmse_mat.shape[1]))",
            "ax.set_xticklabels(rmse_mat.columns)",
            "ax.set_yticks(range(rmse_mat.shape[0]))",
            "ax.set_yticklabels(rmse_mat.index)",
            "for i in range(rmse_mat.shape[0]):",
            "    for j in range(rmse_mat.shape[1]):",
            "        ax.text(j, i, f\"{rmse_mat.iat[i,j]:.4f}\", ha=\"center\", va=\"center\",",
            "                fontsize=9, color=\"white\")",
            "plt.colorbar(im, ax=ax, label=\"RMSE\")",
            "ax.set_title(\"Per-window per-asset RMSE\")",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Predicted vs actual scatter (pooled across windows)"),
        code(
            "fig = plot_holdout_scatter(report)",
            "plt.show()",
        ),
        md(
            "## PIT histogram",
            "",
            "Transform each actual value through the predicted Gaussian CDF",
            "``N(α̂ + β̂·r_obs, diag(Σ̂_{M|O}))``.  Under a correctly-specified model",
            "the PIT values are uniform on (0, 1).",
        ),
        code(
            "from backcast.validation.metrics import pit_values",
            "",
            "all_pits = []",
            "for w in report.windows:",
            "    cond_std = w.cond_std  # (n_short,)",
            "    for i, name in enumerate(report.short_assets):",
            "        actual = w.actual[name].to_numpy()",
            "        pred = w.predicted[name].to_numpy()",
            "        pit = pit_values(actual, pred, np.full_like(actual, cond_std[i]))",
            "        all_pits.append(pit)",
            "pits = np.concatenate(all_pits)",
            "",
            "fig = plot_pit_histogram(pits, bins=20)",
            "plt.show()",
            "print(f\"PIT mean   : {pits.mean():.4f}  (nominal 0.5)\")",
            "print(f\"PIT std    : {pits.std():.4f}  (nominal ≈ 0.289)\")",
        ),
        md(
            "## Residual diagnostics",
            "",
            "Pool the residuals across all 3 windows and check normality (JB) +",
            "autocorrelation (LB).",
        ),
        code(
            "from backcast.validation.diagnostics import summarise_residual_diagnostics",
            "",
            "all_residuals = pd.concat([w.actual - w.predicted for w in report.windows], axis=0)",
            "summarise_residual_diagnostics(all_residuals).round(4)",
        ),
        md(
            "- `jb_pvalue > 0.05` → residuals pass the Jarque-Bera normality test",
            "- `lb_pvalue > 0.05` → no significant autocorrelation at lag 10",
            "- `mean ≈ 0`  → unbiased predictions",
        ),
        md(
            "## Summary",
            "",
            "Well-specified Gaussian DGP + Stambaugh EM gives:",
            "",
            "- Overall coverage very close to 95 %",
            "- Uniform PIT histogram (mean ≈ 0.5, std ≈ 0.29)",
            "- Gaussian, uncorrelated residuals",
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# Notebook 4: Downstream analytics
# ---------------------------------------------------------------------------

def build_downstream() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 04 · Downstream Analytics",
            "",
            "From ``M`` imputed histories, produce:",
            "",
            "1. Multiple covariance estimators (EM, Rubin combined, Ledoit-Wolf,",
            "   Marchenko-Pastur denoised)",
            "2. Uncertainty sets (ellipsoidal and box) for robust optimisation",
            "3. Portfolio-risk distribution for an equal-weight allocation",
            "4. Backtest fan charts across 4 strategies",
            "5. Copula-based scenario simulation",
        ),
        code(*PREAMBLE),
        code(
            "from backcast.data.loader import load_backcast_dataset",
            "from backcast.imputation.multiple_impute import multiple_impute_regime",
            "from backcast.models.em_stambaugh import em_stambaugh",
            "from backcast.models.regime_hmm import compute_regime_params, fit_regime_hmm",
            "",
            "ds = load_backcast_dataset(TIER2_CSV)",
            "em = em_stambaugh(ds.returns_full, max_iter=500, tolerance=1e-8,",
            "                   track_loglikelihood=False)",
            "hmm = fit_regime_hmm(ds.returns_full[ds.long_assets], n_regimes=2, seed=0)",
        ),
        md("## Multiple imputation (regime-conditional, M=50)"),
        code(
            "overlap_labels = hmm.regime_labels[-ds.overlap_length:]",
            "regime_params = compute_regime_params(ds.overlap_matrix, overlap_labels)",
            "mi = multiple_impute_regime(",
            "    ds, hmm.regime_labels, regime_params,",
            "    n_imputations=50, seed=42,",
            ")",
            "print(f\"Drew {mi.n_imputations} imputations — method={mi.method}\")",
        ),
        md("## Covariance comparison"),
        code(
            "from backcast.downstream.covariance import (",
            "    combined_covariance, denoise_covariance, from_em_result, shrink_covariance,",
            ")",
            "",
            "cov_em = from_em_result(em)",
            "cov_comb = combined_covariance(mi.imputations)",
            "cov_shrunk = shrink_covariance(mi.imputations[0])",
            "cov_denoised = denoise_covariance(mi.imputations[0])",
            "",
            "table = pd.DataFrame({",
            "    \"method\": [\"EM direct\", \"Rubin combined\", cov_shrunk.method, cov_denoised.method],",
            "    \"condition_number\": [",
            "        cov_em.condition_number, cov_comb.condition_number,",
            "        cov_shrunk.condition_number, cov_denoised.condition_number,",
            "    ],",
            "    \"min_eig\": [",
            "        cov_em.eigenvalues[-1], cov_comb.eigenvalues[-1],",
            "        cov_shrunk.eigenvalues[-1], cov_denoised.eigenvalues[-1],",
            "    ],",
            "    \"max_eig\": [",
            "        cov_em.eigenvalues[0], cov_comb.eigenvalues[0],",
            "        cov_shrunk.eigenvalues[0], cov_denoised.eigenvalues[0],",
            "    ],",
            "})",
            "table.round(6)",
        ),
        md("### Eigenvalue spectrum vs Marchenko-Pastur bounds"),
        code(
            "from backcast.visualization.plots import plot_eigenvalue_spectrum",
            "fig = plot_eigenvalue_spectrum(cov_comb.covariance, T=len(ds.returns_full))",
            "plt.show()",
        ),
        md(
            "### Rubin within / between / total variance of the Σ entries",
            "",
            "How much of each covariance element's variance is driven by",
            "between-imputation uncertainty (i.e., missing information)?",
        ),
        code(
            "within = cov_comb.within_variance",
            "between = cov_comb.between_variance",
            "total = cov_comb.total_variance",
            "rel_incr = (1 + 1/cov_comb.n_imputations) * between / np.where(within > 0, within, 1e-30)",
            "",
            "fig, axes = plt.subplots(1, 3, figsize=(12, 4))",
            "for ax, mat, title in zip(axes, [within, between, rel_incr],",
            "                          [\"Within\", \"Between\", \"(1+1/M)·B / W\"]):",
            "    im = ax.imshow(mat, cmap=\"viridis\", aspect=\"auto\")",
            "    ax.set_title(title)",
            "    plt.colorbar(im, ax=ax, fraction=0.04)",
            "    ax.set_xticks(range(mat.shape[1]))",
            "    ax.set_xticklabels(ds.returns_full.columns, rotation=45, fontsize=7)",
            "    ax.set_yticks(range(mat.shape[0]))",
            "    ax.set_yticklabels(ds.returns_full.columns, fontsize=7)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Uncertainty sets"),
        code(
            "from backcast.downstream.uncertainty import (",
            "    box_uncertainty, ellipsoidal_uncertainty, portfolio_risk_distribution,",
            ")",
            "from backcast.visualization.plots import plot_uncertainty_ellipses",
            "",
            "ellipse = ellipsoidal_uncertainty(mi.imputations, confidence=0.95)",
            "box = box_uncertainty(mi.imputations, confidence=0.95)",
            "",
            "print(f\"Ellipsoid κ = {ellipse.kappa:.2f}   (confidence = {ellipse.confidence:.0%})\")",
            "print(f\"μ_center = {ellipse.mu_center}\")",
            "print(\"\\nBox uncertainty on μ:\")",
            "mu_box = pd.DataFrame({\"lower\": box.mu_lower, \"upper\": box.mu_upper},",
            "                       index=ds.returns_full.columns)",
            "mu_box.round(5)",
        ),
        code(
            "fig = plot_uncertainty_ellipses(ellipse, pair=(ds.short_assets[0], ds.long_assets[0]))",
            "plt.show()",
        ),
        md("## Equal-weight portfolio-risk distribution"),
        code(
            "N = ds.n_total",
            "w_eq = np.full(N, 1.0 / N)",
            "risk_dist = portfolio_risk_distribution(w_eq, mi.imputations)",
            "annualiser = np.sqrt(252) * 100",
            "print(f\"Annualized portfolio vol (%):\")",
            "print(f\"  median  : {risk_dist.median_risk * annualiser:.2f}\")",
            "print(f\"  5-95%   : [{risk_dist.percentile_5 * annualiser:.2f}, \"",
            "      f\"{risk_dist.percentile_95 * annualiser:.2f}]\")",
            "",
            "fig, ax = plt.subplots(figsize=(7, 3.5))",
            "ax.hist(risk_dist.portfolio_risks * annualiser, bins=30, edgecolor=\"k\", alpha=0.7)",
            "ax.axvline(risk_dist.median_risk * annualiser, color=\"r\", lw=1.5, ls=\"--\",",
            "           label=f\"median = {risk_dist.median_risk * annualiser:.2f}%\")",
            "ax.set_xlabel(\"Annualized portfolio vol (%)\")",
            "ax.set_ylabel(\"Frequency across imputations\")",
            "ax.set_title(f\"Equal-weight risk across M={mi.n_imputations} imputations\")",
            "ax.legend()",
            "ax.grid(alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Backtests — 4 strategies"),
        code(
            "from backcast.downstream.backtest import run_backtest",
            "from backcast.visualization.plots import plot_backtest_fan",
            "",
            "backtests = {}",
            "for strat in (\"equal_weight\", \"inverse_volatility\", \"min_variance\", \"risk_parity\"):",
            "    backtests[strat] = run_backtest(",
            "        mi.imputations, strategy=strat,",
            "        lookback=63, rebalance_freq=21,",
            "    )",
            "",
            "summary = pd.DataFrame({",
            "    strat: {",
            "        \"sharpe_median\": np.median(bt.sharpe_distribution),",
            "        \"sharpe_p05\": np.percentile(bt.sharpe_distribution, 5),",
            "        \"sharpe_p95\": np.percentile(bt.sharpe_distribution, 95),",
            "        \"mdd_median\": np.median(bt.max_drawdown_distribution),",
            "        \"total_return_median\": np.median(bt.total_return_distribution),",
            "    }",
            "    for strat, bt in backtests.items()",
            "}).T",
            "summary.round(3)",
        ),
        md(
            "### Fan charts",
            "",
            "Each panel shows the median cumulative path and the 5–95 % band across",
            "the M imputed histories.",
        ),
        code(
            "fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)",
            "for ax, (strat, bt) in zip(axes.flat, backtests.items()):",
            "    idx = bt.cumulative_median.index",
            "    ax.plot(idx, bt.cumulative_median.values, \"C0-\", lw=1.2, label=\"median\")",
            "    ax.fill_between(idx, bt.cumulative_p05, bt.cumulative_p95,",
            "                    color=\"C0\", alpha=0.2, label=\"5-95 %\")",
            "    ax.set_title(f\"{strat}  (Sharpe med = \"",
            "                 f\"{np.median(bt.sharpe_distribution):+.2f})\")",
            "    ax.grid(alpha=0.3)",
            "    ax.legend(fontsize=8)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("## Copula-based simulation"),
        code(
            "from backcast.imputation.copula_sim import fit_copula, fit_marginals, simulate_copula",
            "from backcast.imputation.single_impute import single_impute",
            "",
            "filled = single_impute(ds, em)",
            "marg = fit_marginals(filled, candidates=(\"normal\", \"student_t\"))",
            "print(\"Selected marginal per asset:\")",
            "for name, mf in marg.items():",
            "    extra = f\"(df={mf.params['df']:.1f})\" if mf.distribution == \"student_t\" else \"\"",
            "    print(f\"  {name:10s}  {mf.distribution}  {extra}\")",
        ),
        code(
            "cop_t = fit_copula(filled, marg, copula_type=\"student_t\")",
            "print(f\"Student-t copula ν = {cop_t.df:.1f}\")",
            "print(f\"Copula correlation (off-diagonal max |ρ|) = \"",
            "      f\"{np.max(np.abs(cop_t.correlation - np.eye(cop_t.correlation.shape[0]))):.3f}\")",
            "",
            "sim = simulate_copula(cop_t, marg, n_simulations=500, horizon=252, seed=7)",
            "print(f\"Simulated scenarios shape: {sim.simulated_returns.shape}\")",
        ),
        md(
            "### Marginal reconstruction check",
            "",
            "The simulated per-asset distribution should line up with the fitted",
            "marginal.  Here we overlay the density of 500 × 252 simulated draws",
            "against the historical imputed returns.",
        ),
        code(
            "asset = ds.short_assets[0]",
            "asset_idx = sim.asset_names.index(asset)",
            "sim_flat = sim.simulated_returns[:, :, asset_idx].ravel()",
            "",
            "fig, ax = plt.subplots(figsize=(8, 3.5))",
            "ax.hist(filled[asset], bins=60, density=True, alpha=0.5, label=\"imputed\")",
            "ax.hist(sim_flat[::50], bins=60, density=True, alpha=0.5, label=\"simulated\")",
            "ax.set_xlabel(f\"Daily return — {asset}\")",
            "ax.set_ylabel(\"Density\")",
            "ax.set_title(f\"Marginal density: historical vs simulated ({asset})\")",
            "ax.legend()",
            "ax.grid(alpha=0.3)",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## Summary",
            "",
            "- Rubin-combined covariance carries **within**, **between**, and **total**",
            "  variance components — the between term quantifies imputation-driven",
            "  uncertainty per covariance entry.",
            "- Ledoit-Wolf and Marchenko-Pastur both reduce the condition number vs",
            "  the naive sample covariance.",
            "- The ellipsoidal μ-uncertainty set is directly consumable by robust",
            "  optimisers (`cvxpy`, `scipy.optimize`, etc.).",
            "- Running each strategy across M imputations yields a **distribution**",
            "  of Sharpe / MDD / total return, not a single point — this is the",
            "  actionable downstream output.",
            "- Copula simulation extends the history forward while preserving both",
            "  marginal tails and joint dependence.",
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "01_eda.ipynb": build_eda,
    "02_model_comparison.ipynb": build_model_comparison,
    "03_validation.ipynb": build_validation,
    "04_downstream.ipynb": build_downstream,
}


def _write(nb: nbf.NotebookNode, path: Path) -> None:
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    nbf.write(nb, path)


def _execute(path: Path, timeout: int = 600) -> bool:
    """Execute a notebook in place; return True on success."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                             resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"  ✗ EXECUTION ERROR in {path.name}: {exc}", file=sys.stderr)
        nbf.write(nb, path)
        return False
    nbf.write(nb, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true",
                        help="Write notebooks but don't execute them.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset of notebook filenames to build.")
    args = parser.parse_args()

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    # Also write a README describing the notebooks
    readme = NOTEBOOK_DIR / "README.md"
    readme.write_text(
        "# Notebooks\n\n"
        "Demo notebooks covering the backcast engine end-to-end on the Tier 2\n"
        "synthetic dataset generated by the sibling `synthetic_data_generator` project.\n\n"
        "| Notebook | Topic |\n"
        "|---|---|\n"
        "| `01_eda.ipynb` | Dataset structure, missingness, correlations, rolling vol |\n"
        "| `02_model_comparison.ipynb` | EM vs Kalman TVP vs regime HMM |\n"
        "| `03_validation.ipynb` | Walk-forward holdout, PIT histogram, residuals |\n"
        "| `04_downstream.ipynb` | Covariance, uncertainty sets, backtests, copula |\n\n"
        "## Running\n\n"
        "```bash\n"
        "# build + execute all four\n"
        "python scripts/build_notebooks.py\n\n"
        "# just build, don't execute\n"
        "python scripts/build_notebooks.py --no-run\n"
        "```\n\n"
        "The notebooks add `backcast_engine/src` to `sys.path` at the top so\n"
        "they don't require an editable install of the package.\n"
    )

    targets = NOTEBOOKS.items()
    if args.only:
        targets = [(k, v) for k, v in NOTEBOOKS.items() if k in args.only]

    errors = 0
    for fname, builder in targets:
        path = NOTEBOOK_DIR / fname
        nb = builder()
        _write(nb, path)
        print(f"  ✓ wrote  {path.relative_to(PROJ_ROOT)}")
        if not args.no_run:
            ok = _execute(path)
            if ok:
                size_kb = path.stat().st_size / 1024
                print(f"  ✓ ran    {path.name}  ({size_kb:.0f} KB)")
            else:
                errors += 1
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
