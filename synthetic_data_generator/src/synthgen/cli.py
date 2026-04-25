"""Command-line interface for the synthetic financial time-series generator.

Usage examples
--------------
Generate Tier 1 with defaults::

    python -m synthgen --tier 1 --output ./data/tier1

Generate Tier 2 with 3 regimes::

    python -m synthgen --tier 2 --n-regimes 3 --output ./data/tier2

Generate Tier 2 adversarial variant::

    python -m synthgen --tier 2 --adversarial --output ./data/tier2_adv

Generate Tier 3 with Student-t(4) innovations::

    python -m synthgen --tier 3 --df 4 --output ./data/tier3

Generate all Tier 4 stress scenarios (each to a subdirectory)::

    python -m synthgen --tier 4 --scenario all --output ./data/tier4

Custom dimensions::

    python -m synthgen --tier 1 --n-long 8 --n-short 5 --t-total 7500 --seed 123
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from synthgen.config import (
    SyntheticConfig,
    Tier2Config,
    Tier3Config,
    Tier4Config,
)
from synthgen.io import save_dataset
from synthgen.tier1_stationary import generate_tier1
from synthgen.tier2_regime import generate_tier2
from synthgen.tier3_realistic import generate_tier3
from synthgen.tier4_stress import (
    SCENARIO_NAMES,
    generate_tier4,
    generate_tier4_all,
)

logger = logging.getLogger("synthgen")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        prog="synthgen",
        description="Synthetic financial time-series generator.",
    )
    p.add_argument("--tier", type=int, choices=[1, 2, 3, 4], required=True,
                   help="DGP tier (1–4).")
    p.add_argument("--output", type=str, default="./synthetic_output",
                   help="Output directory.")
    p.add_argument("--seed", type=int, default=42, help="Master random seed.")

    # Dimensions
    p.add_argument("--n-long", type=int, default=None,
                   help="Number of long-history assets.")
    p.add_argument("--n-short", type=int, default=None,
                   help="Number of short-history assets.")
    p.add_argument("--t-total", type=int, default=None,
                   help="Total number of trading days.")
    p.add_argument("--short-start-day", type=int, default=None,
                   help="Row index where short-history assets begin.")

    # Calendar
    p.add_argument("--start-date", type=str, default=None,
                   help="First trading day (YYYY-MM-DD).")

    # Correlation structure
    p.add_argument("--correlation-method", type=str, default=None,
                   choices=["factor_model", "random", "manual"],
                   help="Correlation/covariance build method.")
    p.add_argument("--n-factors", type=int, default=None,
                   help="Number of latent factors (factor_model only).")

    # Output control
    p.add_argument("--no-complete", action="store_true",
                   help="Do not write returns_complete.csv.")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity.")

    # Tier 2 options
    p.add_argument("--n-regimes", type=int, default=None,
                   help="(Tier 2) number of regimes.")
    p.add_argument("--adversarial", action="store_true",
                   help="(Tier 2) adversarial regime variant.")

    # Tier 3 options
    p.add_argument("--df", type=float, default=None,
                   help="(Tier 3) Student-t degrees of freedom.")
    p.add_argument("--innovation", type=str, default=None,
                   choices=["student_t", "gaussian"],
                   help="(Tier 3) innovation distribution.")

    # Tier 4 options
    p.add_argument("--scenario", type=str, default=None,
                   choices=list(SCENARIO_NAMES) + ["all"],
                   help="(Tier 4) stress scenario selector.")
    return p


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> SyntheticConfig:
    """Translate parsed CLI ``args`` into a :class:`SyntheticConfig`.

    Parameters
    ----------
    args : argparse.Namespace
        Output of :func:`build_parser().parse_args`.

    Returns
    -------
    SyntheticConfig
    """
    kwargs: dict[str, Any] = {
        "tier": args.tier,
        "seed": args.seed,
        "output_dir": args.output,
    }
    # Generic overrides
    for name, dest in [
        ("n_long", "n_long_assets"),
        ("n_short", "n_short_assets"),
        ("t_total", "t_total"),
        ("short_start_day", "short_start_day"),
        ("start_date", "start_date"),
        ("correlation_method", "correlation_method"),
        ("n_factors", "n_factors"),
    ]:
        val = getattr(args, name, None)
        if val is not None:
            kwargs[dest] = val

    if args.no_complete:
        kwargs["save_complete_returns"] = False

    # Tier-specific sub-configs
    if args.tier == 2:
        kwargs["tier2_config"] = Tier2Config(
            n_regimes=args.n_regimes if args.n_regimes is not None else 2,
            adversarial=args.adversarial,
        )
    if args.tier == 3:
        t3_kwargs: dict[str, Any] = {}
        if args.df is not None:
            t3_kwargs["degrees_of_freedom"] = args.df
        if args.innovation is not None:
            t3_kwargs["innovation_distribution"] = args.innovation
        kwargs["tier3_config"] = Tier3Config(**t3_kwargs)
    if args.tier == 4:
        kwargs["tier4_config"] = Tier4Config(
            scenario=args.scenario if args.scenario is not None else "short_overlap"
        )

    return SyntheticConfig(**kwargs)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run(cfg: SyntheticConfig) -> list[Path]:
    """Generate the configured dataset(s) and write them to disk.

    Parameters
    ----------
    cfg : SyntheticConfig

    Returns
    -------
    list[Path]
        Top-level output directory for each scenario that was written.
    """
    output_dirs: list[Path] = []

    if cfg.tier == 1:
        masked, complete, gt = generate_tier1(cfg)
        paths = save_dataset(
            cfg.output_dir, masked, gt,
            complete if cfg.save_complete_returns else None,
        )
        output_dirs.append(paths["returns"].parent)

    elif cfg.tier == 2:
        masked, complete, gt = generate_tier2(cfg)
        paths = save_dataset(
            cfg.output_dir, masked, gt,
            complete if cfg.save_complete_returns else None,
        )
        output_dirs.append(paths["returns"].parent)

    elif cfg.tier == 3:
        masked, complete, gt = generate_tier3(cfg)
        paths = save_dataset(
            cfg.output_dir, masked, gt,
            complete if cfg.save_complete_returns else None,
        )
        output_dirs.append(paths["returns"].parent)

    elif cfg.tier == 4:
        scen = cfg.tier4_config.scenario if cfg.tier4_config else "short_overlap"
        if scen == "all":
            results = generate_tier4_all(cfg)
            for scen_name, (m, c, g) in results.items():
                subdir = Path(cfg.output_dir) / scen_name
                save_dataset(
                    subdir, m, g,
                    c if cfg.save_complete_returns else None,
                )
                output_dirs.append(subdir)
        else:
            masked, complete, gt = generate_tier4(cfg)
            paths = save_dataset(
                cfg.output_dir, masked, gt,
                complete if cfg.save_complete_returns else None,
            )
            output_dirs.append(paths["returns"].parent)
    else:
        raise ValueError(f"Unknown tier {cfg.tier}")

    return output_dirs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : sequence of str or None
        Argument list.  If None, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = build_config(args)
    dirs = run(cfg)
    for d in dirs:
        logger.info("Dataset written to %s", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
