"""Phase cadence helpers for AGSLS."""

from __future__ import annotations

from typing import Literal

from .config import AGSLSConfig

PhaseName = Literal["exploration", "commit", "exploitation"]


def steps_for_phase(config: AGSLSConfig, phase: PhaseName) -> int:
    """Return the SmoothLife step batch used by a phase."""

    if phase == "exploration":
        return int(config.exploration_steps_per_tick)
    if phase == "commit":
        return int(config.commit_steps_per_zoom)
    return int(config.exploitation_steps_per_zoom)
