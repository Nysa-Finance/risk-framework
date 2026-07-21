"""Nysa risk parameter engine.

Top-level namespace. See module docstrings for the exact section of
`docs/nysa-market-risk-framework.md` and `docs/nysa-lb-caps.md` each
piece implements.
"""

from .config import load_universe

__all__ = ["load_universe"]
