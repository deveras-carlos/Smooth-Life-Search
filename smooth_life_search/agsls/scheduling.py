"""Schedules that make AGSLS zoom more frequently over time."""

from __future__ import annotations

from .config import AGSLSConfig


def steps_for_zoom_cycle(config: AGSLSConfig, zoom_index: int) -> int:
    """Return the number of local simulator steps before the next zoom."""

    raw = round(config.initial_steps_per_zoom * (config.zoom_decay**zoom_index))
    return max(config.min_steps_per_zoom, int(raw))
