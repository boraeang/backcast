#!/usr/bin/env python3
"""Generate one synthetic dataset per tier (1, 2, 3, and all four Tier 4 scenarios).

The script writes everything under ``examples/output/`` (next to this file) so
it stays out of the way of the canonical fixtures in
``synthetic_data_generator/output/``.

Layout produced::

    examples/output/
    ├── tier1/                   stationary Gaussian
    ├── tier2/                   regime-switching, K=2
    ├── tier2_adversarial/       crisis regime never appears in overlap
    ├── tier3/                   GARCH + Student-t + TVP betas
    └── tier4/
        ├── short_overlap/
        ├── high_dimension/
        ├── near_singular/
        └── staggered_heavy/

Usage
-----
    python examples/generate_all_tiers.py
    python examples/generate_all_tiers.py --output ./elsewhere
    python examples/generate_all_tiers.py --tiers 1 2          # subset
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make the synthgen package importable when running from a source checkout.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synthgen.config import (  # noqa: E402  (deliberate after sys.path tweak)
    SyntheticConfig, Tier2Config, Tier3Config, Tier4Config,
)
from synthgen.io import save_dataset
from synthgen.tier1_stationary import generate_tier1
from synthgen.tier2_regime import generate_tier2
from synthgen.tier3_realistic import generate_tier3
from synthgen.tier4_stress import SCENARIO_NAMES, generate_tier4

logger = logging.getLogger("generate_all_tiers")


def _save(masked, complete, gt, out_dir: Path) -> None:
    paths = save_dataset(out_dir, masked, gt, complete)
    sizes = ", ".join(
        f"{p.name}={p.stat().st_size/1024:.1f}KB" for p in paths.values()
    )
    logger.info("  → %s  (%s)", out_dir, sizes)


def _run_tier1(out_dir: Path, seed: int) -> None:
    logger.info("Tier 1 — stationary Gaussian (default 5+3 assets, T=5000)")
    cfg = SyntheticConfig(seed=seed, tier=1)
    masked, complete, gt = generate_tier1(cfg)
    _save(masked, complete, gt, out_dir / "tier1")


def _run_tier2(out_dir: Path, seed: int) -> None:
    logger.info("Tier 2 — regime-switching (K=2, default transitions)")
    cfg = SyntheticConfig(seed=seed, tier=2, tier2_config=Tier2Config(n_regimes=2))
    masked, complete, gt = generate_tier2(cfg)
    _save(masked, complete, gt, out_dir / "tier2")

    logger.info("Tier 2 (adversarial) — crisis regime confined to backcast period")
    cfg_adv = SyntheticConfig(
        seed=seed, tier=2,
        tier2_config=Tier2Config(n_regimes=2, adversarial=True),
    )
    masked, complete, gt = generate_tier2(cfg_adv)
    _save(masked, complete, gt, out_dir / "tier2_adversarial")


def _run_tier3(out_dir: Path, seed: int) -> None:
    logger.info("Tier 3 — GARCH + Student-t(5) + TVP betas")
    cfg = SyntheticConfig(seed=seed, tier=3, tier3_config=Tier3Config())
    masked, complete, gt = generate_tier3(cfg)
    _save(masked, complete, gt, out_dir / "tier3")


def _run_tier4(out_dir: Path, seed: int) -> None:
    for i, scen in enumerate(SCENARIO_NAMES):
        logger.info("Tier 4 — scenario %s", scen)
        cfg = SyntheticConfig(
            seed=seed + i, tier=4, tier4_config=Tier4Config(scenario=scen),
        )
        masked, complete, gt = generate_tier4(cfg)
        _save(masked, complete, gt, out_dir / "tier4" / scen)


# ---------------------------------------------------------------------------

_RUNNERS = {
    1: _run_tier1,
    2: _run_tier2,
    3: _run_tier3,
    4: _run_tier4,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_all_tiers",
        description="Generate one synthetic dataset per Tier (1–4).",
    )
    parser.add_argument(
        "--output", "-o", default=str(_HERE / "output"),
        help="Top-level output directory (default: examples/output/).",
    )
    parser.add_argument(
        "--tiers", "-t", type=int, nargs="*", default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Subset of tiers to generate (default: all four).",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Master random seed (Tier 4 scenarios use seed+i).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: %s", out_dir)

    t0 = time.time()
    for tier in sorted(set(args.tiers)):
        _RUNNERS[tier](out_dir, args.seed)
    elapsed = time.time() - t0
    logger.info("Done — total elapsed %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
