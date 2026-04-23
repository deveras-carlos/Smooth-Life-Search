"""Basin scoring for Adaptive Grid Smooth Life Search."""

from __future__ import annotations

import numpy as np

from ..results import Basin
from .config import AGSLSConfig


def score_basins(basins: list[Basin], config: AGSLSConfig) -> list[Basin]:
    """Populate combined scores for candidate basins."""

    if not basins:
        return []
    max_mass = max( basin.support_mass for basin in basins )
    max_area = max( max( basin.area, 1 ) for basin in basins )
    for basin in basins:
        norm_mass = basin.support_mass / max( max_mass, 1e-12 )
        norm_area = basin.area / max( max_area, 1 )
        basin.combined_score = (
            config.mass_weight * norm_mass
            + config.alive_density_weight * basin.alive_density
            + config.objective_weight * basin.objective_score
            + config.stability_weight * basin.stability_score
            - config.area_penalty * norm_area
        )
    basins.sort( key=lambda basin: basin.combined_score, reverse=True )
    return basins
