"""Basin eligibility helpers for the simplified AGSLS controller."""

from __future__ import annotations

from ..core import Basin
from .config import AGSLSConfig


def eligible_basins(basins: list[Basin], config: AGSLSConfig) -> list[Basin]:
    """Return basins large and alive enough for a zoom decision."""

    return [
        basin
        for basin in basins
        if basin.area >= config.min_basin_cells and basin.alive_density >= config.min_alive_density
    ]


def select_basin(basins: list[Basin]) -> Basin:
    """Select the highest-ranked basin."""

    if not basins:
        raise ValueError("cannot select from an empty basin list")
    return max(basins, key=lambda basin: float(basin.combined_score))
