"""Basin eligibility and leader selection for AGSLS."""

from __future__ import annotations

import numpy as np

from ..core import Basin
from ..smoothlife.state import SmoothLifeState
from .config import AGSLSConfig


def eligible_basins(basins: list[Basin], config: AGSLSConfig) -> list[Basin]:
    """Filter basins that are large and alive enough for a zoom decision."""

    return [
        basin
        for basin in basins
        if basin.area >= config.min_basin_cells and basin.alive_density >= config.min_alive_density
    ]


def should_choose_leader(
    basins: list[Basin],
    explored_in_stage: int,
    config: AGSLSConfig,
    state: SmoothLifeState,
) -> tuple[bool, str]:
    """Return whether the current ranked basins are decisive enough."""

    if not basins:
        return False, "no_group"
    if len(basins) == 1:
        return True, "single_group"
    score_gap = float(basins[0].combined_score - basins[1].combined_score)
    if score_gap >= config.dominance_margin:
        return True, "dominant_group"
    if score_gap <= config.similarity_margin:
        if explored_in_stage <= 0 and config.candidate_probe_evaluations > 0:
            return False, "similar_groups_probe"
        return True, "similar_groups"
    if explored_in_stage >= config.undecided_stage_max_evaluations:
        return True, "exploration_cap"
    union_mask = np.zeros_like(basins[0].mask, dtype=bool)
    for basin in basins:
        union_mask |= basin.mask
    if not np.any(union_mask & ~state.evaluated_mask):
        return True, "no_unexplored_cells"
    return False, "continue_exploring"


def select_basin(basins: list[Basin], config: AGSLSConfig) -> Basin:
    """Select one basin from ranked candidates, breaking near-ties by quality."""

    if len(basins) <= 1:
        return basins[0]
    top_score = float(basins[0].combined_score)
    contenders = [
        basin
        for basin in basins
        if top_score - float(basin.combined_score) <= config.similarity_margin
    ]
    if len(contenders) <= 1:
        return basins[0]
    return max(
        contenders,
        key=lambda basin: (
            float(basin.best_objective_score) if basin.evaluated_count > 0 else -np.inf,
            float(basin.stability_score),
            -int(basin.area),
            float(basin.combined_score),
        ),
    )
