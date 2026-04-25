"""End-to-end backcast pipeline orchestration.

Loads a YAML config (or accepts a dict), then threads the following steps:

1. ``load_data``       — CSV → :class:`BackcastDataset`.
2. ``fit_models``      — EM + Kalman TVP + regime HMM.
3. ``validate``        — walk-forward holdout on the overlap period.
4. ``impute``          — multiple imputation (unconditional-EM by default;
   ``regime_conditional`` when config asks).
5. ``compute_downstream`` — covariance, uncertainty, backtest.
6. ``export``          — write plots + JSON/Parquet summaries.

``BackcastPipeline.run`` executes steps 1-5 in order and returns a
:class:`FullResults` bundle; :meth:`export` flushes everything to disk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from backcast.data.loader import BackcastDataset, load_backcast_dataset
from backcast.downstream.backtest import BacktestResult, run_backtest
from backcast.downstream.covariance import (
    CovarianceResult, combined_covariance, denoise_covariance, from_em_result,
    shrink_covariance,
)
from backcast.downstream.uncertainty import (
    BoxUncertaintySet, EllipsoidalUncertaintySet, PortfolioRiskDistribution,
    box_uncertainty, ellipsoidal_uncertainty, portfolio_risk_distribution,
)
from backcast.imputation.multiple_impute import (
    MultipleImputationResult, multiple_impute, multiple_impute_regime,
    prediction_intervals,
)
from backcast.imputation.single_impute import single_impute
from backcast.models.em_stambaugh import EMResult, em_stambaugh
from backcast.models.kalman_tvp import KalmanMultiAssetResult, fit_kalman_all
from backcast.models.model_selector import ModelSelectionResult, select_model_cv
from backcast.models.regime_hmm import (
    HMMResult, HMMSelectionResult, compute_regime_params, fit_and_select_hmm,
)
from backcast.validation.holdout import HoldoutReport, run_holdout_validation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------

@dataclass
class DownstreamResults:
    """Downstream analytics derived from the imputations."""

    covariance_em: CovarianceResult
    covariance_combined: CovarianceResult
    covariance_shrunk: Optional[CovarianceResult]
    covariance_denoised: Optional[CovarianceResult]
    ellipsoidal_mu: EllipsoidalUncertaintySet
    box_uncertainty: BoxUncertaintySet
    equal_weight_risk: PortfolioRiskDistribution
    backtests: dict   # strategy_name -> BacktestResult


@dataclass
class FullResults:
    """Every artefact produced by :meth:`BackcastPipeline.run`."""

    dataset: BackcastDataset
    em_result: EMResult
    kalman: Optional[KalmanMultiAssetResult]
    hmm: Optional[HMMResult]
    hmm_selection: Optional[HMMSelectionResult]
    holdout: HoldoutReport
    imputation: MultipleImputationResult
    downstream: DownstreamResults
    config: dict
    model_selection: Optional[ModelSelectionResult] = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"
)


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BackcastPipeline:
    """Orchestrate the end-to-end backcast workflow.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config.  If None and ``config_dict`` is also None,
        :data:`_DEFAULT_CONFIG_PATH` is used.
    config_dict : dict, optional
        Already-parsed config (overrides ``config_path`` when both are given).
    log_level : str
        Python logging level applied at init.

    Attributes
    ----------
    config : dict
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        config_dict: dict | None = None,
        log_level: str = "INFO",
    ):
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(levelname)s %(name)s: %(message)s",
        )
        if config_dict is not None:
            self.config = dict(config_dict)
        else:
            self.config = _load_yaml(config_path or _DEFAULT_CONFIG_PATH)
        self.seed = int(self.config.get("random_seed", 42))
        self.rng = np.random.default_rng(self.seed)
        logger.info("Pipeline initialised with seed=%d", self.seed)

    # -------------------------------------------------------------------
    # Step 1: data
    # -------------------------------------------------------------------

    def load_data(self, csv_path: str | Path) -> BackcastDataset:
        cfg = self.config.get("data", {})
        ds = load_backcast_dataset(
            csv_path,
            date_column=cfg.get("date_column", "date"),
            min_overlap_days=int(cfg.get("min_overlap_days", 0)),
        )
        logger.info(
            "Loaded %s: %d long, %d short, overlap=%d rows, backcast=%d rows",
            csv_path, ds.n_long, ds.n_short, ds.overlap_length, ds.backcast_length,
        )
        return ds

    # -------------------------------------------------------------------
    # Step 2: models
    # -------------------------------------------------------------------

    def fit_models(self, dataset: BackcastDataset) -> dict[str, Any]:
        """Fit EM, Kalman, HMM.  Each is independent so a failure in one
        doesn't prevent the others from running."""
        out: dict[str, Any] = {}

        # EM
        em_cfg = self.config.get("em", {})
        em = em_stambaugh(
            dataset.returns_full,
            max_iter=int(em_cfg.get("max_iterations", 500)),
            tolerance=float(em_cfg.get("tolerance", 1e-8)),
            track_loglikelihood=bool(em_cfg.get("track_loglikelihood", True)),
        )
        out["em"] = em
        logger.info("EM: %d iters, converged=%s", em.n_iter, em.converged)

        # Kalman TVP (needs fully-observed overlap)
        kcfg = self.config.get("kalman", {})
        try:
            kalman = fit_kalman_all(
                dataset.overlap_matrix,
                dataset.long_assets,
                dataset.short_assets,
                state_noise_scale=float(kcfg.get("state_noise_scale", 0.01)),
                initial_state_cov_scale=float(kcfg.get("initial_state_cov_scale", 1.0)),
                use_smoother=bool(kcfg.get("use_smoother", True)),
                backcast_beta_method=kcfg.get("backcast_beta_method", "earliest_smoothed"),
                backcast_beta_k=int(kcfg.get("backcast_beta_k", 63)),
            )
            out["kalman"] = kalman
        except Exception as exc:
            logger.warning("Kalman fit skipped: %s", exc)
            out["kalman"] = None

        # HMM (on long assets)
        hcfg = self.config.get("hmm", {})
        try:
            sel = fit_and_select_hmm(
                dataset.returns_full[dataset.long_assets],
                n_regimes_candidates=tuple(hcfg.get("n_regimes_candidates", (2, 3, 4))),
                criterion=hcfg.get("selection_criterion", "bic"),
                max_iter=int(hcfg.get("max_iterations", 200)),
                tolerance=float(hcfg.get("tolerance", 1e-4)),
                seed=self.seed,
            )
            out["hmm_selection"] = sel
            out["hmm"] = sel.best
            logger.info(
                "HMM selected K=%d via %s", sel.best_n_regimes, sel.criterion,
            )
        except Exception as exc:
            logger.warning("HMM fit skipped: %s", exc)
            out["hmm"] = None
            out["hmm_selection"] = None

        return out

    # -------------------------------------------------------------------
    # Step 3: validation
    # -------------------------------------------------------------------

    def validate(self, dataset: BackcastDataset) -> HoldoutReport:
        vcfg = self.config.get("validation", {})
        report = run_holdout_validation(
            dataset,
            holdout_days=int(vcfg.get("holdout_days", 504)),
            n_windows=int(vcfg.get("n_windows", 3)),
            coverage_level=float(vcfg.get("coverage_level", 0.95)),
        )
        logger.info(
            "Holdout: %d windows, overall coverage=%.3f",
            len(report.windows), report.overall_coverage,
        )
        return report

    # -------------------------------------------------------------------
    # Step 4a: model selection via cross-validation
    # -------------------------------------------------------------------

    def select_model(self, dataset: BackcastDataset) -> ModelSelectionResult:
        """Cross-validated selection among imputation methods.

        Reads its parameters from ``config['model_selection']`` (if present);
        otherwise falls back to sensible defaults.  Returns a
        :class:`ModelSelectionResult` whose ``best_method`` attribute can be
        fed back into the config to drive :meth:`impute`.
        """
        ms_cfg = self.config.get("model_selection", {})
        v_cfg = self.config.get("validation", {})
        h_cfg = self.config.get("hmm", {})
        em_cfg = self.config.get("em", {})
        candidates = tuple(ms_cfg.get("candidates",
                                       ("unconditional_em", "regime_conditional")))
        criterion = ms_cfg.get("criterion", "combined")
        result = select_model_cv(
            dataset,
            candidates=candidates,
            criterion=criterion,
            holdout_days=int(v_cfg.get("holdout_days", 504)),
            n_windows=int(v_cfg.get("n_windows", 3)),
            coverage_level=float(v_cfg.get("coverage_level", 0.95)),
            em_max_iter=int(em_cfg.get("max_iterations", 500)),
            em_tolerance=float(em_cfg.get("tolerance", 1e-8)),
            hmm_n_regimes=int(ms_cfg.get("hmm_n_regimes", 2)),
            hmm_max_iter=int(h_cfg.get("max_iterations", 200)),
            hmm_tolerance=float(h_cfg.get("tolerance", 1e-4)),
            hmm_seed=self.seed,
        )
        logger.info(
            "Model selection (%s): best=%s  ranking=%s",
            criterion, result.best_method, result.ranking,
        )
        return result

    # -------------------------------------------------------------------
    # Step 4: multiple imputation
    # -------------------------------------------------------------------

    def impute(
        self,
        dataset: BackcastDataset,
        em: EMResult,
        hmm: HMMResult | None = None,
    ) -> MultipleImputationResult:
        icfg = self.config.get("imputation", {})
        method = icfg.get("method", "unconditional_em")
        n_imp = int(icfg.get("n_imputations", 50))
        if method == "regime_conditional":
            if hmm is None:
                logger.warning("regime_conditional requested but HMM is None; "
                               "falling back to unconditional_em")
                return multiple_impute(dataset, em, n_imputations=n_imp, seed=self.seed)
            overlap_labels = hmm.regime_labels[-dataset.overlap_length:]
            regime_params = compute_regime_params(dataset.overlap_matrix, overlap_labels)
            return multiple_impute_regime(
                dataset, hmm.regime_labels, regime_params,
                n_imputations=n_imp, seed=self.seed,
            )
        return multiple_impute(dataset, em, n_imputations=n_imp, seed=self.seed)

    # -------------------------------------------------------------------
    # Step 5: downstream
    # -------------------------------------------------------------------

    def compute_downstream(
        self,
        dataset: BackcastDataset,
        em: EMResult,
        mi: MultipleImputationResult,
    ) -> DownstreamResults:
        dcfg = self.config.get("downstream", {})

        cov_em = from_em_result(em)
        cov_comb = combined_covariance(mi.imputations)

        cov_shrunk: Optional[CovarianceResult] = None
        cov_denoised: Optional[CovarianceResult] = None
        try:
            if dcfg.get("covariance_shrinkage", True):
                cov_shrunk = shrink_covariance(mi.imputations[0])
            if dcfg.get("denoise_eigenvalues", True):
                cov_denoised = denoise_covariance(mi.imputations[0])
        except Exception as exc:
            logger.warning("Advanced covariance estimation skipped: %s", exc)

        conf = float(dcfg.get("uncertainty_confidence", 0.95))
        ellipse = ellipsoidal_uncertainty(mi.imputations, confidence=conf)
        box = box_uncertainty(mi.imputations, confidence=conf)

        # Portfolio risk under the equal-weight portfolio, as a canonical example
        N = dataset.n_total
        w_eq = np.full(N, 1.0 / N)
        eq_risk = portfolio_risk_distribution(w_eq, mi.imputations)

        backtests: dict[str, BacktestResult] = {}
        lookback = int(dcfg.get("backtest_lookback", 63))
        reb = int(dcfg.get("backtest_rebalance_freq", 21))
        for strat in dcfg.get("backtest_strategies", ["equal_weight"]):
            try:
                backtests[strat] = run_backtest(
                    mi.imputations, strategy=strat,
                    lookback=lookback, rebalance_freq=reb,
                )
            except Exception as exc:
                logger.warning("Backtest %s failed: %s", strat, exc)

        return DownstreamResults(
            covariance_em=cov_em,
            covariance_combined=cov_comb,
            covariance_shrunk=cov_shrunk,
            covariance_denoised=cov_denoised,
            ellipsoidal_mu=ellipse,
            box_uncertainty=box,
            equal_weight_risk=eq_risk,
            backtests=backtests,
        )

    # -------------------------------------------------------------------
    # End-to-end
    # -------------------------------------------------------------------

    def run(self, csv_path: str | Path) -> FullResults:
        dataset = self.load_data(csv_path)
        models = self.fit_models(dataset)
        report = self.validate(dataset)

        # Optional CV-based model selection when method is 'auto' (or the
        # config explicitly requests `model_selection.enabled: True`).
        model_sel: Optional[ModelSelectionResult] = None
        icfg = self.config.get("imputation", {})
        ms_cfg = self.config.get("model_selection", {})
        method_requested = str(icfg.get("method", "unconditional_em")).lower()
        if method_requested == "auto" or ms_cfg.get("enabled"):
            model_sel = self.select_model(dataset)
            # Patch the method in-place so impute() picks it up
            icfg["method"] = model_sel.best_method
            self.config["imputation"] = icfg

        mi = self.impute(dataset, models["em"], models.get("hmm"))
        downstream = self.compute_downstream(dataset, models["em"], mi)
        return FullResults(
            dataset=dataset,
            em_result=models["em"],
            kalman=models.get("kalman"),
            hmm=models.get("hmm"),
            hmm_selection=models.get("hmm_selection"),
            holdout=report,
            imputation=mi,
            downstream=downstream,
            config=dict(self.config),
            model_selection=model_sel,
        )

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export(self, results: FullResults, output_dir: str | Path) -> dict[str, Path]:
        """Write plots (PNG), JSON summary, and optionally the imputed histories.

        Returns
        -------
        dict[str, Path]
            Keys identify the artefact; values are full paths on disk.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_fmt = self.config.get("output", {}).get("plot_format", "png")
        plot_dpi = int(self.config.get("output", {}).get("plot_dpi", 120))
        save_imp = bool(self.config.get("output", {}).get("save_imputations", False))

        artefacts: dict[str, Path] = {}

        # --- plots ---
        from backcast.visualization.plots import (
            plot_backcast_fan, plot_backtest_fan, plot_correlation_comparison,
            plot_eigenvalue_spectrum, plot_em_convergence, plot_holdout_scatter,
            plot_kalman_betas, plot_missingness, plot_regime_timeline,
            plot_uncertainty_ellipses,
        )
        import matplotlib.pyplot as plt

        def _save(fig, name):
            path = out / f"{name}.{plot_fmt}"
            fig.savefig(path, dpi=plot_dpi, bbox_inches="tight")
            plt.close(fig)
            artefacts[name] = path

        _save(plot_missingness(results.dataset.returns_full), "01_missingness")
        _save(plot_em_convergence(results.em_result), "02_em_convergence")
        if results.kalman is not None:
            _save(plot_kalman_betas(results.kalman), "03_kalman_betas")
        if results.hmm is not None:
            _save(plot_regime_timeline(
                results.hmm.regime_labels,
                results.dataset.returns_full.index,
            ), "04_regime_timeline")
        _save(plot_backcast_fan(
            results.imputation,
            actual=None,
            asset=results.dataset.short_assets[0],
        ), "05_backcast_fan")
        overlap_corr = np.corrcoef(
            results.dataset.overlap_matrix.to_numpy(), rowvar=False,
        )
        # Use the combined-covariance correlation for the backcast-era estimate
        _save(plot_correlation_comparison(
            overlap_corr,
            results.downstream.covariance_combined.correlation,
            labels=list(results.dataset.returns_full.columns),
            titles=("Overlap (sample)", "Combined (Rubin)"),
        ), "06_correlation_comparison")
        _save(plot_holdout_scatter(results.holdout), "07_holdout_scatter")
        _save(plot_eigenvalue_spectrum(
            results.downstream.covariance_combined.covariance,
            T=len(results.dataset.returns_full),
        ), "09_eigenvalue_spectrum")
        _save(plot_uncertainty_ellipses(
            results.downstream.ellipsoidal_mu,
        ), "10_uncertainty_ellipse")
        for name, bt in results.downstream.backtests.items():
            _save(plot_backtest_fan(bt), f"11_backtest_{name}")

        # --- summary JSON ---
        summary_path = out / "summary.json"
        summary = _build_summary(results)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)
        artefacts["summary"] = summary_path

        # --- optional: imputed histories ---
        if save_imp:
            imp_dir = out / "imputations"
            imp_dir.mkdir(exist_ok=True)
            for i, df in enumerate(results.imputation.imputations):
                p = imp_dir / f"imputation_{i:03d}.parquet"
                try:
                    df.to_parquet(p)
                except Exception:
                    p = p.with_suffix(".csv")
                    df.to_csv(p)
                artefacts[f"imputation_{i:03d}"] = p

        logger.info("Exported %d artefacts to %s", len(artefacts), out)
        return artefacts


# ---------------------------------------------------------------------------
# Summary serialisation
# ---------------------------------------------------------------------------

def _build_summary(results: FullResults) -> dict:
    """Serialise a compact subset of results to plain Python for JSON."""
    ds = results.dataset
    em = results.em_result
    mi = results.imputation
    ds_out = {
        "n_long": ds.n_long,
        "n_short": ds.n_short,
        "n_total": ds.n_total,
        "overlap_length": ds.overlap_length,
        "backcast_length": ds.backcast_length,
        "asset_names": list(ds.returns_full.columns),
        "long_assets": list(ds.long_assets),
        "short_assets": list(ds.short_assets),
        "overlap_start": str(ds.overlap_start.date()) if ds.overlap_start is not None else None,
        "overlap_end": str(ds.overlap_end.date()) if ds.overlap_end is not None else None,
    }
    em_out = {
        "n_iter": em.n_iter,
        "converged": em.converged,
        "final_delta": em.final_delta,
        "log_likelihood_final": em.log_likelihood_trace[-1] if em.log_likelihood_trace else None,
    }
    hmm_out = None
    if results.hmm is not None:
        hmm_out = {
            "n_regimes": results.hmm.n_regimes,
            "log_likelihood": results.hmm.log_likelihood,
            "bic": results.hmm.bic,
            "aic": results.hmm.aic,
            "transition_matrix": results.hmm.transition_matrix.tolist(),
        }
    selection_out = None
    if results.hmm_selection is not None:
        selection_out = {
            "candidates": list(results.hmm_selection.candidates),
            "best_n_regimes": results.hmm_selection.best_n_regimes,
            "criterion": results.hmm_selection.criterion,
            "scores": {int(k): float(v) for k, v in results.hmm_selection.scores.items()},
        }
    ho_out = {
        "overall_coverage": results.holdout.overall_coverage,
        "overall_correlation_error": results.holdout.overall_correlation_error,
        "per_asset_mean": results.holdout.per_asset_mean.to_dict(),
    }
    down = results.downstream
    cov_out = {
        "em_condition_number": down.covariance_em.condition_number,
        "combined_condition_number": down.covariance_combined.condition_number,
        "shrunk_method": down.covariance_shrunk.method if down.covariance_shrunk else None,
        "shrunk_condition_number": down.covariance_shrunk.condition_number if down.covariance_shrunk else None,
        "denoised_condition_number": down.covariance_denoised.condition_number if down.covariance_denoised else None,
    }
    uncertainty_out = {
        "ellipsoid_kappa": down.ellipsoidal_mu.kappa,
        "ellipsoid_confidence": down.ellipsoidal_mu.confidence,
        "mu_center": down.ellipsoidal_mu.mu_center.tolist(),
        "equal_weight_risk": {
            "median": down.equal_weight_risk.median_risk,
            "p05": down.equal_weight_risk.percentile_5,
            "p95": down.equal_weight_risk.percentile_95,
        },
    }
    backtest_out = {}
    for name, bt in down.backtests.items():
        backtest_out[name] = {
            "n_imputations": bt.n_imputations,
            "sharpe_median": float(np.median(bt.sharpe_distribution)),
            "sharpe_p05": float(np.percentile(bt.sharpe_distribution, 5)),
            "sharpe_p95": float(np.percentile(bt.sharpe_distribution, 95)),
            "total_return_median": float(np.median(bt.total_return_distribution)),
            "max_drawdown_median": float(np.median(bt.max_drawdown_distribution)),
        }

    model_sel_out = None
    if results.model_selection is not None:
        ms = results.model_selection
        model_sel_out = {
            "criterion": ms.criterion,
            "candidates": list(ms.candidates),
            "best_method": ms.best_method,
            "ranking": list(ms.ranking),
            "scores": {k: float(v) for k, v in ms.scores.items()},
            "per_method": {
                k: {
                    "rmse_overall": r.rmse_overall,
                    "coverage_overall": r.coverage_overall,
                    "coverage_error": r.coverage_error,
                    "correlation_error": r.correlation_error,
                } for k, r in ms.per_method.items()
            },
        }

    return {
        "dataset": ds_out,
        "em": em_out,
        "hmm": hmm_out,
        "hmm_selection": selection_out,
        "holdout": ho_out,
        "imputation": {
            "n_imputations": mi.n_imputations,
            "method": mi.method,
            "seed": mi.seed,
        },
        "model_selection": model_sel_out,
        "downstream": {
            "covariance": cov_out,
            "uncertainty": uncertainty_out,
            "backtests": backtest_out,
        },
        "config": results.config,
    }
