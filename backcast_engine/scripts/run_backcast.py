#!/usr/bin/env python3
"""Backcast pipeline CLI entry point.

Usage
-----
    python scripts/run_backcast.py -i returns.csv -o ./out
    python scripts/run_backcast.py -i data.csv -o ./out --config custom.yaml
    python scripts/run_backcast.py -i data.csv -o ./out \\
        --method regime_conditional --n-imputations 100 --seed 7

Loads a returns CSV, runs the end-to-end :class:`BackcastPipeline`
(load → fit EM + Kalman + HMM → holdout → multiple impute → downstream
covariance / uncertainty / backtests), then exports all plots + a
``summary.json`` to the requested output directory.

Exit codes
----------
- 0 : success
- 2 : input file missing
- 3 : configuration error
- 4 : runtime error while running the pipeline
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

# Make the package importable when running from a source checkout.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backcast.pipeline import BackcastPipeline, FullResults

logger = logging.getLogger("run_backcast")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="run_backcast",
        description=(
            "Backcast missing short-history returns and run downstream "
            "covariance / uncertainty / backtest analytics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to the returns CSV (first column = date, remaining columns = assets).",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Directory where plots, summary.json, and optional imputations are written.",
    )
    p.add_argument(
        "--config", "-c", default=None,
        help="YAML config path (defaults to the packaged config/default_config.yaml).",
    )
    # Inline overrides
    p.add_argument("--seed", type=int, default=None,
                   help="Override random_seed from the config.")
    p.add_argument("--n-imputations", type=int, default=None,
                   help="Override imputation.n_imputations from the config.")
    p.add_argument(
        "--method",
        choices=["unconditional_em", "regime_conditional", "auto"], default=None,
        help="Override imputation.method from the config "
             "('auto' triggers CV-based model selection).",
    )
    p.add_argument(
        "--save-imputations", action="store_true",
        help="Also write every imputed history as parquet/CSV.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging verbosity (default INFO).",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Only print the final summary — shortcut for --log-level WARNING.",
    )
    return p


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

def apply_overrides(pipe: BackcastPipeline, args: argparse.Namespace) -> None:
    """Patch ``pipe.config`` in-place with any CLI overrides."""
    if args.seed is not None:
        pipe.config["random_seed"] = int(args.seed)
        pipe.seed = int(args.seed)
        import numpy as np
        pipe.rng = np.random.default_rng(pipe.seed)
    if args.n_imputations is not None:
        pipe.config.setdefault("imputation", {})["n_imputations"] = int(args.n_imputations)
    if args.method is not None:
        pipe.config.setdefault("imputation", {})["method"] = args.method
    if args.save_imputations:
        pipe.config.setdefault("output", {})["save_imputations"] = True


# ---------------------------------------------------------------------------
# Final one-line summary
# ---------------------------------------------------------------------------

def _print_summary(results: FullResults, artefacts: dict[str, Path], output_dir: Path) -> None:
    """Emit a compact stdout summary that's grep-friendly for scripting."""
    em = results.em_result
    cov = results.downstream.covariance_combined
    ho = results.holdout
    mi = results.imputation
    lines = [
        "",
        "=" * 70,
        "BACKCAST PIPELINE RESULT",
        "=" * 70,
        f"  dataset           : {results.dataset.n_long} long + {results.dataset.n_short} short"
        f"  (overlap {results.dataset.overlap_length} rows,"
        f" backcast {results.dataset.backcast_length} rows)",
        f"  EM                : {em.n_iter} iters,  converged={em.converged},"
        f"  final ΔΣ={em.final_delta:.2e}",
        f"  HMM               : "
        + (f"K={results.hmm.n_regimes} regimes," if results.hmm else "skipped")
        + (f"  log-L={results.hmm.log_likelihood:.0f}" if results.hmm else ""),
        f"  holdout coverage  : {ho.overall_coverage:.4f}  (nominal 0.95)",
        f"  imputations       : M={mi.n_imputations},  method={mi.method}",
        f"  cov cond (comb.)  : {cov.condition_number:.1f}",
        f"  artefacts written : {len(artefacts)} in {output_dir}",
        "=" * 70,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the backcast pipeline end-to-end.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = build_parser().parse_args(argv)

    if args.quiet:
        args.log_level = "WARNING"

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        pipe = BackcastPipeline(config_path=args.config, log_level=args.log_level)
    except Exception as exc:
        print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
        return 3

    apply_overrides(pipe, args)

    try:
        results = pipe.run(input_path)
        artefacts = pipe.export(results, output_dir)
    except Exception as exc:
        print(f"ERROR: pipeline failed: {exc}", file=sys.stderr)
        if args.log_level == "DEBUG":
            traceback.print_exc()
        return 4

    _print_summary(results, artefacts, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
