"""Small basin scoring helper kept for callers that rank AGSLS basins directly."""

from __future__ import annotations

from typing import Literal

from ..core import Basin
from .config import AGSLSConfig

PhaseName = Literal["commit", "exploitation"]


def score_basins(basins: list[Basin], config: AGSLSConfig, *, phase: PhaseName = "commit") -> list[Basin]:
    """Populate combined scores using the new phase priorities."""

    if not basins:
        return []
    max_mass = max(float(basin.support_mass) for basin in basins)
    max_area = max(max(int(basin.area), 1) for basin in basins)
    for basin in basins:
        norm_mass = float(basin.support_mass) / max(max_mass, 1e-12)
        norm_area = float(basin.area) / max(max_area, 1)
        if phase == "commit":
            basin.combined_score = (
                config.commit_mass_weight * norm_mass
                + config.commit_density_weight * float(basin.alive_density)
                + config.commit_stability_weight * float(basin.stability_score)
                + config.commit_objective_weight * float(basin.objective_score)
                - config.commit_area_penalty * norm_area
            )
        else:
            incumbent_bonus = 2.0 if basin.incumbent_in_envelope else 0.0
            best_score = float(basin.best_objective_score) if basin.evaluated_count > 0 else 0.0
            basin.combined_score = incumbent_bonus + best_score + 0.25 * norm_mass + 0.25 * float(basin.stability_score)
    basins.sort(key=lambda basin: float(basin.combined_score), reverse=True)
    return basins
