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
    surrogate_enabled: bool = True
    surrogate_min_samples: int = 12
    surrogate_max_samples: int = 96
    surrogate_regularization: float = 1e-10
    surrogate_max_condition: float = 1e10
    local_refinement_enabled: bool = True
    local_refinement_start_evaluations: int = 12
    local_refinement_max_evaluations: int = 512
    local_refinement_gradient_tolerance: float = 1e-7
    local_refinement_step_fraction: float = 0.10
    best_improvement_tolerance: float = 0.0
    snapshot_interval_batches: int = 1

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
        if self.surrogate_min_samples <= 0 or self.surrogate_max_samples <= 0:
            raise ValueError("surrogate sample counts must be positive")
        if self.surrogate_max_samples < self.surrogate_min_samples:
            raise ValueError("surrogate_max_samples must be >= surrogate_min_samples")
        if self.surrogate_regularization < 0.0:
            raise ValueError("surrogate_regularization must be non-negative")
        if self.surrogate_max_condition <= 0.0:
            raise ValueError("surrogate_max_condition must be positive")
        if self.local_refinement_start_evaluations < 0:
            raise ValueError("local_refinement_start_evaluations must be non-negative")
        if self.local_refinement_max_evaluations < 0:
            raise ValueError("local_refinement_max_evaluations must be non-negative")
        if self.local_refinement_gradient_tolerance < 0.0:
            raise ValueError("local_refinement_gradient_tolerance must be non-negative")
        if not 0.0 < self.local_refinement_step_fraction <= 1.0:
            raise ValueError("local_refinement_step_fraction must be in (0, 1]")
        if self.best_improvement_tolerance < 0.0:
            raise ValueError("best_improvement_tolerance must be non-negative")
        if self.snapshot_interval_batches <= 0:
            raise ValueError("snapshot_interval_batches must be positive")
