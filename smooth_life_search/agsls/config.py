"""Configuration for Adaptive Grid Smooth Life Search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AGSLSConfig:
    """Policy parameters for adaptive basin zooming."""

    max_zoom_cycles: int = 5
    initial_steps_per_zoom: int = 32
    min_steps_per_zoom: int = 8
    zoom_decay: float = 0.75
    basin_quantile: float = 0.88
    min_basin_cells: int = 24
    zoom_padding: float = 0.20
    min_side_fraction: float = 1e-64
    max_evaluations: int | None = None
    mass_weight: float = 1.0
    alive_density_weight: float = 0.75
    objective_weight: float = 1.5
    stability_weight: float = 0.75
    area_penalty: float = 0.20
    alive_core_threshold: float = 0.30
    min_alive_density: float = 0.20
    cluster_eps_pixels: float = 2.5
    cluster_min_samples: int = 6
    dominance_margin: float = 0.20
    similarity_margin: float = 0.05
    undecided_stage_max_evaluations: int = 64
    candidate_probe_evaluations: int = 16

    def __post_init__(self) -> None:
        if self.max_zoom_cycles <= 0:
            raise ValueError("max_zoom_cycles must be positive")
        if self.initial_steps_per_zoom <= 0 or self.min_steps_per_zoom <= 0:
            raise ValueError("steps_per_zoom values must be positive")
        if self.min_steps_per_zoom > self.initial_steps_per_zoom:
            raise ValueError("min_steps_per_zoom cannot exceed initial_steps_per_zoom")
        if not 0.0 < self.zoom_decay <= 1.0:
            raise ValueError("zoom_decay must be in (0, 1]")
        if not 0.0 < self.basin_quantile < 1.0:
            raise ValueError("basin_quantile must be in (0, 1)")
        if self.min_basin_cells <= 0:
            raise ValueError("min_basin_cells must be positive")
        if self.zoom_padding < 0.0:
            raise ValueError("zoom_padding must be non-negative")
        if not 0.0 < self.min_side_fraction <= 1.0:
            raise ValueError("min_side_fraction must be in (0, 1]")
        if not 0.0 <= self.alive_core_threshold <= 1.0:
            raise ValueError("alive_core_threshold must be in [0, 1]")
        if not 0.0 <= self.min_alive_density <= 1.0:
            raise ValueError("min_alive_density must be in [0, 1]")
        if self.cluster_eps_pixels <= 0.0:
            raise ValueError("cluster_eps_pixels must be positive")
        if self.cluster_min_samples <= 0:
            raise ValueError("cluster_min_samples must be positive")
        if self.undecided_stage_max_evaluations < 0:
            raise ValueError("undecided_stage_max_evaluations must be non-negative")
        if self.candidate_probe_evaluations < 0:
            raise ValueError("candidate_probe_evaluations must be non-negative")
