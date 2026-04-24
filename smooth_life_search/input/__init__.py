"""Input parsing, configuration loading, and objective ingestion."""

from .objectives import ObjectiveSpec, resolve_objective

__all__ = ["ObjectiveSpec", "resolve_objective"]
