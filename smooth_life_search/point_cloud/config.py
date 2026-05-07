"""Configuration for archive-centered point-cloud SmoothLife search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PointCloudSearchConfig:
    """Policy parameters for point-cloud proposal search."""

    max_evaluations: int | None = None
    batch_size: int = 32
    initial_design_size: int = 32
    max_batches: int | None = None
    density_grid_shape: tuple[int, int] = (64, 64)
    density_elite_fraction: float = 0.25
    density_sigma_fraction: float = 0.12
    density_smooth_steps: int = 3
    portfolio_size: int = 4
    elite_fraction: float = 0.20
    trust_regions_enabled: bool = True
    region_initial_radius_fraction: float = 0.25
    region_min_radius_fraction: float = 1e-10
    region_max_radius_fraction: float = 0.75
    region_expand_factor: float = 1.35
    region_shrink_factor: float = 0.50
    region_candidate_fraction: float = 0.30
    density_candidate_fraction: float = 0.30
    global_candidate_fraction: float = 0.20
    exploit_candidate_fraction: float = 0.20
    early_stop_enabled: bool = False
    early_stop_value: float | None = None
    region_stall_patience: int = 3
    region_cooldown_batches: int = 4
    global_exploration_floor: float = 0.10
    region_stencil_fraction: float = 0.35
    anisotropic_regions_enabled: bool = True
    region_anisotropy_max: float = 25.0
    region_geometry_min_samples: int = 8
    active_subspace_size: int = 8
    surrogate_enabled: bool = True
    surrogate_min_samples: int = 12
    surrogate_max_samples: int = 96
    surrogate_full_quadratic_max_dimension: int = 6
    surrogate_regularization: float = 1e-10
    surrogate_max_condition: float = 1e10
    projection_axes: tuple[int, int] | None = None
    projection_ensemble_enabled: bool = True
    projection_ensemble_size: int = 4
    projection_ensemble_refresh_batches: int = 4
    local_refinement_enabled: bool = True
    local_refinement_start_evaluations: int = 12
    local_refinement_max_evaluations: int = 512
    local_refinement_gradient_tolerance: float = 1e-7
    local_refinement_step_fraction: float = 0.10
    local_refinement_method: str = "hybrid"
    local_refinement_damping: float = 1e-6
    high_dimensional_refinement_enabled: bool = True
    high_dimensional_min_dimension: int = 12
    block_refinement_overlap: int = 2
    block_refinement_blocks_per_pass: int = 2
    dimension_scaled_batches_enabled: bool = True
    dimension_scaled_batch_max: int = 48
    coherent_probes_enabled: bool = True
    surrogate_ranking_enabled: bool = True
    candidate_pool_multiplier: int = 3
    surrogate_ranking_neighbor_count: int = 32
    probe_recenter_enabled: bool = True
    probe_recenter_max_restarts: int = 8
    basin_polishing_enabled: bool = True
    basin_polishing_min_dimension: int = 12
    basin_polishing_activation_ratio: float = 0.25
    successful_direction_memory_size: int = 24
    direction_refinement_enabled: bool = True
    direction_refinement_max_evaluations: int = 128
    direction_line_search_mode: str = "bracketed"
    direction_line_search_max_steps: int = 4
    direction_line_search_min_step_fraction: float = 1e-10
    linkage_blocks_enabled: bool = True
    linkage_update_interval_batches: int = 8
    linkage_neighbor_count: int = 3
    cross_block_lbfgs_enabled: bool = True
    cross_block_lbfgs_memory_size: int = 16
    cooperative_refinement_enabled: bool = True
    cooperative_min_dimension: int = 50
    cooperative_group_size: int | None = None
    cooperative_groups_per_batch: int = 4
    cooperative_frontier_enabled: bool = True
    cooperative_frontier_fraction: float = 0.35
    axis_coverage_pressure: float = 0.20
    active_set_max_fraction: float = 0.25
    active_set_expand_interval_batches: int = 4
    surrogate_reliability_enabled: bool = True
    surrogate_rank_weight_min: float = 0.50
    surrogate_rank_weight_max: float = 0.75
    best_improvement_tolerance: float = 0.0
    snapshot_interval_batches: int = 16

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when set")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.initial_design_size <= 0:
            raise ValueError("initial_design_size must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when set")
        if len(self.density_grid_shape) != 2 or min(self.density_grid_shape) < 8:
            raise ValueError("density_grid_shape must be at least 8x8")
        for field_name in ("density_elite_fraction", "elite_fraction"):
            value = float(getattr(self, field_name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
        if self.density_sigma_fraction <= 0.0:
            raise ValueError("density_sigma_fraction must be positive")
        if self.density_smooth_steps < 0:
            raise ValueError("density_smooth_steps must be non-negative")
        if self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        if not 0.0 < self.region_min_radius_fraction <= self.region_initial_radius_fraction <= self.region_max_radius_fraction <= 1.0:
            raise ValueError("require min <= initial <= max region radius fractions in (0, 1]")
        if not 0.0 < self.region_shrink_factor < 1.0:
            raise ValueError("region_shrink_factor must be in (0, 1)")
        if self.region_expand_factor <= 1.0:
            raise ValueError("region_expand_factor must be greater than 1")
        for field_name in (
            "region_candidate_fraction",
            "density_candidate_fraction",
            "global_candidate_fraction",
            "exploit_candidate_fraction",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if (
            self.region_candidate_fraction
            + self.density_candidate_fraction
            + self.global_candidate_fraction
            + self.exploit_candidate_fraction
        ) <= 0.0:
            raise ValueError("at least one candidate fraction must be positive")
        if self.region_stall_patience <= 0:
            raise ValueError("region_stall_patience must be positive")
        if self.region_cooldown_batches < 0:
            raise ValueError("region_cooldown_batches must be non-negative")
        if not 0.0 <= self.global_exploration_floor <= 1.0:
            raise ValueError("global_exploration_floor must be in [0, 1]")
        if not 0.0 <= self.region_stencil_fraction <= 1.0:
            raise ValueError("region_stencil_fraction must be in [0, 1]")
        if self.region_anisotropy_max < 1.0:
            raise ValueError("region_anisotropy_max must be >= 1")
        if self.region_geometry_min_samples <= 0:
            raise ValueError("region_geometry_min_samples must be positive")
        if self.active_subspace_size <= 0:
            raise ValueError("active_subspace_size must be positive")
        if self.surrogate_min_samples <= 0 or self.surrogate_max_samples <= 0:
            raise ValueError("surrogate sample counts must be positive")
        if self.surrogate_max_samples < self.surrogate_min_samples:
            raise ValueError("surrogate_max_samples must be >= surrogate_min_samples")
        if self.surrogate_full_quadratic_max_dimension < 2:
            raise ValueError("surrogate_full_quadratic_max_dimension must be >= 2")
        if self.surrogate_regularization < 0.0:
            raise ValueError("surrogate_regularization must be non-negative")
        if self.surrogate_max_condition <= 0.0:
            raise ValueError("surrogate_max_condition must be positive")
        if self.projection_axes is not None:
            if len(self.projection_axes) != 2:
                raise ValueError("projection_axes must contain exactly two axes")
            first, second = int(self.projection_axes[0]), int(self.projection_axes[1])
            if first < 0 or second < 0 or first == second:
                raise ValueError("projection_axes must contain two distinct non-negative axes")
        if self.projection_ensemble_size <= 0:
            raise ValueError("projection_ensemble_size must be positive")
        if self.projection_ensemble_refresh_batches <= 0:
            raise ValueError("projection_ensemble_refresh_batches must be positive")
        if self.local_refinement_start_evaluations < 0:
            raise ValueError("local_refinement_start_evaluations must be non-negative")
        if self.local_refinement_max_evaluations < 0:
            raise ValueError("local_refinement_max_evaluations must be non-negative")
        if self.local_refinement_gradient_tolerance < 0.0:
            raise ValueError("local_refinement_gradient_tolerance must be non-negative")
        if not 0.0 < self.local_refinement_step_fraction <= 1.0:
            raise ValueError("local_refinement_step_fraction must be in (0, 1]")
        if self.local_refinement_method not in {"bfgs", "levenberg-marquardt", "hybrid"}:
            raise ValueError("local_refinement_method must be bfgs, levenberg-marquardt, or hybrid")
        if self.local_refinement_damping <= 0.0:
            raise ValueError("local_refinement_damping must be positive")
        if self.high_dimensional_min_dimension < 2:
            raise ValueError("high_dimensional_min_dimension must be >= 2")
        if self.block_refinement_overlap < 0:
            raise ValueError("block_refinement_overlap must be non-negative")
        if self.block_refinement_blocks_per_pass <= 0:
            raise ValueError("block_refinement_blocks_per_pass must be positive")
        if self.dimension_scaled_batch_max < self.batch_size:
            raise ValueError("dimension_scaled_batch_max must be >= batch_size")
        if self.candidate_pool_multiplier <= 0:
            raise ValueError("candidate_pool_multiplier must be positive")
        if self.surrogate_ranking_neighbor_count <= 0:
            raise ValueError("surrogate_ranking_neighbor_count must be positive")
        if self.probe_recenter_max_restarts < 0:
            raise ValueError("probe_recenter_max_restarts must be non-negative")
        if self.basin_polishing_min_dimension < 2:
            raise ValueError("basin_polishing_min_dimension must be >= 2")
        if self.basin_polishing_activation_ratio <= 0.0:
            raise ValueError("basin_polishing_activation_ratio must be positive")
        if self.successful_direction_memory_size <= 0:
            raise ValueError("successful_direction_memory_size must be positive")
        if self.direction_refinement_max_evaluations < 0:
            raise ValueError("direction_refinement_max_evaluations must be non-negative")
        if self.direction_line_search_mode not in {"opportunistic", "bracketed"}:
            raise ValueError("direction_line_search_mode must be opportunistic or bracketed")
        if self.direction_line_search_max_steps <= 0:
            raise ValueError("direction_line_search_max_steps must be positive")
        if self.direction_line_search_min_step_fraction <= 0.0:
            raise ValueError("direction_line_search_min_step_fraction must be positive")
        if self.linkage_update_interval_batches <= 0:
            raise ValueError("linkage_update_interval_batches must be positive")
        if self.linkage_neighbor_count <= 0:
            raise ValueError("linkage_neighbor_count must be positive")
        if self.cross_block_lbfgs_memory_size <= 0:
            raise ValueError("cross_block_lbfgs_memory_size must be positive")
        if self.cooperative_min_dimension < 2:
            raise ValueError("cooperative_min_dimension must be >= 2")
        if self.cooperative_group_size is not None and self.cooperative_group_size <= 0:
            raise ValueError("cooperative_group_size must be positive when set")
        if self.cooperative_groups_per_batch <= 0:
            raise ValueError("cooperative_groups_per_batch must be positive")
        if not 0.0 <= self.cooperative_frontier_fraction <= 1.0:
            raise ValueError("cooperative_frontier_fraction must be in [0, 1]")
        if not 0.0 <= self.axis_coverage_pressure <= 1.0:
            raise ValueError("axis_coverage_pressure must be in [0, 1]")
        if not 0.0 < self.active_set_max_fraction <= 1.0:
            raise ValueError("active_set_max_fraction must be in (0, 1]")
        if self.active_set_expand_interval_batches <= 0:
            raise ValueError("active_set_expand_interval_batches must be positive")
        if not 0.0 <= self.surrogate_rank_weight_min <= self.surrogate_rank_weight_max <= 1.0:
            raise ValueError("require 0 <= surrogate_rank_weight_min <= surrogate_rank_weight_max <= 1")
        if self.best_improvement_tolerance < 0.0:
            raise ValueError("best_improvement_tolerance must be non-negative")
        if self.snapshot_interval_batches <= 0:
            raise ValueError("snapshot_interval_batches must be positive")
