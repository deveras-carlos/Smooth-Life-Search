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
    zoom_cycles_budget_baseline: int | None = 800
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
    basin_envelope_quantile_offset: float = 0.08
    basin_envelope_growth_pixels: int = 1
    min_zoom_cells: int = 6
    incumbent_overlap_bonus: float = 0.20
    incumbent_exclusion_penalty: float = 0.15
    edge_risk_fraction: float = 0.20
    late_stage_zoom_fraction_threshold: float = 0.60
    late_stage_plateau_threshold: float = 1e-4
    late_stage_min_shrink_ratio: float = 0.85
    late_stage_max_rounds: int = 2
    late_stage_eval_batch: int = 8
    late_stage_elite_k: int = 4
    late_stage_focus_radius_cells: int = 3
    late_stage_microgrid_enabled: bool = False
    late_stage_microgrid_resolution: int = 5
    late_stage_microgrid_centers: int = 3
    late_stage_microgrid_side_fraction: float = 0.25
    late_stage_translation_enabled: bool = False
    late_stage_translation_step_fraction: float = 0.25
    late_stage_translation_min_offset_fraction: float = 0.10
    late_stage_exploiter: str = "microgrid"
    late_stage_periodic_local_search_enabled: bool = True
    late_stage_periodic_local_search_interval_steps: int = 8
    late_stage_periodic_local_search_initial_step_fraction: float = 0.10
    late_stage_periodic_local_search_min_step_fraction: float = 0.005
    late_stage_periodic_local_search_shrink: float = 0.5
    late_stage_periodic_local_search_max_iterations: int = 12
    late_stage_periodic_local_search_max_evaluations: int = 48
    late_stage_pattern_search_initial_step_fraction: float = 0.15
    late_stage_pattern_search_min_step_fraction: float = 0.005
    late_stage_pattern_search_shrink: float = 0.5
    late_stage_pattern_search_max_iterations: int = 12
    late_stage_pattern_search_max_evaluations: int = 20
    late_stage_pattern_search_reuse_tolerance_cells: float = 0.5
    final_polish_enabled: bool = True
    final_polish_max_evaluations: int = 512
    inter_zoom_polish_enabled: bool = True
    inter_zoom_polish_max_evaluations: int = 64

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
        if not 0.0 <= self.basin_envelope_quantile_offset <= 0.49:
            raise ValueError("basin_envelope_quantile_offset must be in [0, 0.49]")
        if self.basin_envelope_growth_pixels < 0:
            raise ValueError("basin_envelope_growth_pixels must be non-negative")
        if self.min_zoom_cells <= 0:
            raise ValueError("min_zoom_cells must be positive")
        if self.incumbent_overlap_bonus < 0.0:
            raise ValueError("incumbent_overlap_bonus must be non-negative")
        if self.incumbent_exclusion_penalty < 0.0:
            raise ValueError("incumbent_exclusion_penalty must be non-negative")
        if not 0.0 <= self.edge_risk_fraction <= 0.5:
            raise ValueError("edge_risk_fraction must be in [0, 0.5]")
        if self.zoom_cycles_budget_baseline is not None and self.zoom_cycles_budget_baseline <= 0:
            raise ValueError("zoom_cycles_budget_baseline must be positive when set")
        if not 0.0 <= self.late_stage_zoom_fraction_threshold <= 1.0:
            raise ValueError("late_stage_zoom_fraction_threshold must be in [0, 1]")
        if self.late_stage_plateau_threshold < 0.0:
            raise ValueError("late_stage_plateau_threshold must be non-negative")
        if not 0.0 < self.late_stage_min_shrink_ratio <= 1.0:
            raise ValueError("late_stage_min_shrink_ratio must be in (0, 1]")
        if self.late_stage_max_rounds < 0:
            raise ValueError("late_stage_max_rounds must be non-negative")
        if self.late_stage_eval_batch <= 0:
            raise ValueError("late_stage_eval_batch must be positive")
        if self.late_stage_elite_k <= 0:
            raise ValueError("late_stage_elite_k must be positive")
        if self.late_stage_focus_radius_cells <= 0:
            raise ValueError("late_stage_focus_radius_cells must be positive")
        if self.late_stage_microgrid_resolution <= 0:
            raise ValueError("late_stage_microgrid_resolution must be positive")
        if self.late_stage_microgrid_centers <= 0:
            raise ValueError("late_stage_microgrid_centers must be positive")
        if not 0.0 < self.late_stage_microgrid_side_fraction <= 1.0:
            raise ValueError("late_stage_microgrid_side_fraction must be in (0, 1]")
        if not 0.0 < self.late_stage_translation_step_fraction <= 1.0:
            raise ValueError("late_stage_translation_step_fraction must be in (0, 1]")
        if not 0.0 < self.late_stage_translation_min_offset_fraction <= 1.0:
            raise ValueError("late_stage_translation_min_offset_fraction must be in (0, 1]")
        if self.late_stage_exploiter not in ("microgrid", "pattern_search", "none"):
            raise ValueError("late_stage_exploiter must be one of 'microgrid', 'pattern_search', 'none'")
        if self.late_stage_periodic_local_search_interval_steps <= 0:
            raise ValueError("late_stage_periodic_local_search_interval_steps must be positive")
        if not 0.0 < self.late_stage_periodic_local_search_initial_step_fraction <= 1.0:
            raise ValueError("late_stage_periodic_local_search_initial_step_fraction must be in (0, 1]")
        if not 0.0 < self.late_stage_periodic_local_search_min_step_fraction <= self.late_stage_periodic_local_search_initial_step_fraction:
            raise ValueError("late_stage_periodic_local_search_min_step_fraction must be in (0, initial_step_fraction]")
        if not 0.0 < self.late_stage_periodic_local_search_shrink < 1.0:
            raise ValueError("late_stage_periodic_local_search_shrink must be in (0, 1)")
        if self.late_stage_periodic_local_search_max_iterations <= 0:
            raise ValueError("late_stage_periodic_local_search_max_iterations must be positive")
        if self.late_stage_periodic_local_search_max_evaluations <= 0:
            raise ValueError("late_stage_periodic_local_search_max_evaluations must be positive")
        if not 0.0 < self.late_stage_pattern_search_initial_step_fraction <= 1.0:
            raise ValueError("late_stage_pattern_search_initial_step_fraction must be in (0, 1]")
        if not 0.0 < self.late_stage_pattern_search_min_step_fraction <= self.late_stage_pattern_search_initial_step_fraction:
            raise ValueError("late_stage_pattern_search_min_step_fraction must be in (0, initial_step_fraction]")
        if not 0.0 < self.late_stage_pattern_search_shrink < 1.0:
            raise ValueError("late_stage_pattern_search_shrink must be in (0, 1)")
        if self.late_stage_pattern_search_max_iterations <= 0:
            raise ValueError("late_stage_pattern_search_max_iterations must be positive")
        if self.late_stage_pattern_search_max_evaluations <= 0:
            raise ValueError("late_stage_pattern_search_max_evaluations must be positive")
        if not 0.0 <= self.late_stage_pattern_search_reuse_tolerance_cells <= 1.0:
            raise ValueError("late_stage_pattern_search_reuse_tolerance_cells must be in [0, 1]")
        if self.final_polish_max_evaluations <= 0:
            raise ValueError("final_polish_max_evaluations must be positive")
        if self.inter_zoom_polish_max_evaluations <= 0:
            raise ValueError("inter_zoom_polish_max_evaluations must be positive")
