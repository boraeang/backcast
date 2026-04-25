# Backcast Project

This project has two independent sub-projects. Each has its own 
specification file in the docs/ folder.

## Sub-project 1: Synthetic Data Generator
- Spec: docs/synthetic_data_generator_prompt.md
- Code goes in: synthetic_data_generator/
- Build this FIRST
- Standalone, no dependencies on the backcast engine

## Sub-project 2: Backcast Engine
- Spec: docs/backcast_prompt.md
- Code goes in: backcast_engine/
- Build this SECOND
- Consumes CSV files produced by the synthetic data generator

## Rules (apply to both sub-projects)
- Python 3.10+ compatibility
- Type hints on all functions
- NumPy-style docstrings
- Use numpy.random.Generator with explicit seed, never global random state
- Use scipy.linalg.cho_factor/cho_solve, never np.linalg.inv
- Run pytest after implementing each module
