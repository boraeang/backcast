"""synthgen — Synthetic Financial Time Series Generator."""
from synthgen.config import SyntheticConfig, Tier2Config, Tier3Config, Tier4Config
from synthgen.tier1_stationary import generate_tier1
from synthgen.tier2_regime import generate_tier2
from synthgen.tier3_realistic import generate_tier3
from synthgen.tier4_stress import generate_tier4, generate_tier4_all

__all__ = [
    "SyntheticConfig",
    "Tier2Config",
    "Tier3Config",
    "Tier4Config",
    "generate_tier1",
    "generate_tier2",
    "generate_tier3",
    "generate_tier4",
    "generate_tier4_all",
]
