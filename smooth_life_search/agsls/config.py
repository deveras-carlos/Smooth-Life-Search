"""Configuration for the three-phase Adaptive Grid Smooth Life Search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AGSLSConfig:
    """Policy parameters for phase-driven basin zooming."""

    max_evaluations: int | None = None
    max_zoom_cycles: int = 16
    exploration_fraction: float = 0.05
    commit_fraction: float = 0.7
    exploration_steps_per_tick: int = 16
    commit_steps_per_zoom: int = 256
    exploitation_steps_per_zoom: int = 72
    basin_quantile: float = 0.88
    min_basin_cells: int = 24
    min_alive_density: float = 0.20
    alive_core_threshold: float = 0.30
    cluster_eps_pixels: float = 2.5
    cluster_min_samples: int = 6
    basin_envelope_quantile_offset: float = 0.01
    basin_envelope_growth_pixels: int = 1
    commit_zoom_padding: float = 0.05
    commit_min_shrink_fraction: float = 0.45
    commit_min_explored_fraction: float = 0.08
    commit_incumbent_padding_fraction: float = 0.02
    exploitation_shrink_fraction: float = 1e-12
    exploration_objective_gamma: float = 1.0
    commit_objective_gamma: float = 1.0
    exploitation_objective_gamma: float = 1.5
    exploration_support_ema_alpha: float = 0.0
    commit_support_ema_alpha: float = 0.05
    exploitation_support_ema_alpha: float = 0.20
    commit_guidance_top_k: int = 64
    commit_guidance_sigma: float = 0.12
    commit_guidance_temperature: float = 0.20
    commit_uncertainty_weight: float = 0.10
    commit_drift_strength: float = 0.06
    exploitation_guidance_top_k: int = 32
    exploitation_guidance_sigma: float = 0.04
    exploitation_guidance_temperature: float = 0.10
    exploitation_uncertainty_weight: float = 0.0
    exploitation_drift_strength: float = 0.02
    commit_surrogate_enabled: bool = True
    commit_surrogate_min_samples: int = 12
    commit_surrogate_max_samples: int = 96
    commit_surrogate_regularization: float = 1e-8
    commit_surrogate_min_predicted_improvement: float = 0.0
    commit_surrogate_max_condition: float = 1e8
    commit_surrogate_valley_expand: float = 1.25
    commit_surrogate_cross_shrink: float = 0.65
    commit_surrogate_support_weight: float = 0.50
    exploitation_valley_tracking_enabled: bool = True
    exploitation_valley_probe_evaluations: int = 8
    exploitation_valley_step_fraction: float = 0.15
    exploitation_valley_step_decay: float = 0.50
    exploitation_valley_min_step_fraction: float = 1e-4
    exploitation_valley_surrogate_min_samples: int = 12
    exploitation_valley_surrogate_max_samples: int = 96
    trust_region_enabled: bool = True
    commit_trust_region_evaluations: int = 8
    exploitation_trust_region_evaluations: int = 16
    trust_region_candidate_pool_size: int = 128
    trust_region_initial_radius_fraction: float = 0.25
    trust_region_min_radius_fraction: float = 1e-14
    trust_region_shrink_factor: float = 0.50
    trust_region_expand_factor: float = 1.40
    commit_acquisition_uncertainty_weight: float = 0.30
    exploitation_acquisition_uncertainty_weight: float = 0.05
    trust_region_support_weight: float = 0.25
    commit_mass_weight: float = 1.35
    commit_density_weight: float = 1.20
    commit_stability_weight: float = 1.00
    commit_objective_weight: float = 0.45
    commit_area_penalty: float = 0.15

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when set")
        if self.max_zoom_cycles <= 0:
            raise ValueError("max_zoom_cycles must be positive")
        if not 0.0 <= self.exploration_fraction < self.commit_fraction < 1.0:
            raise ValueError("require 0 <= exploration_fraction < commit_fraction < 1")
        if self.exploration_steps_per_tick <= 0:
            raise ValueError("exploration_steps_per_tick must be positive")
        if self.commit_steps_per_zoom <= 0:
            raise ValueError("commit_steps_per_zoom must be positive")
        if self.exploitation_steps_per_zoom <= 0:
            raise ValueError("exploitation_steps_per_zoom must be positive")
        if not 0.0 < self.basin_quantile < 1.0:
            raise ValueError("basin_quantile must be in (0, 1)")
        if self.min_basin_cells <= 0:
            raise ValueError("min_basin_cells must be positive")
        if not 0.0 <= self.min_alive_density <= 1.0:
            raise ValueError("min_alive_density must be in [0, 1]")
        if not 0.0 <= self.alive_core_threshold <= 1.0:
            raise ValueError("alive_core_threshold must be in [0, 1]")
        if self.cluster_eps_pixels <= 0.0:
            raise ValueError("cluster_eps_pixels must be positive")
        if self.cluster_min_samples <= 0:
            raise ValueError("cluster_min_samples must be positive")
        if not 0.0 <= self.basin_envelope_quantile_offset <= 0.49:
            raise ValueError("basin_envelope_quantile_offset must be in [0, 0.49]")
        if self.basin_envelope_growth_pixels < 0:
            raise ValueError("basin_envelope_growth_pixels must be non-negative")
        if self.commit_zoom_padding < 0.0:
            raise ValueError("commit_zoom_padding must be non-negative")
        if not 0.0 < self.commit_min_shrink_fraction <= 1.0:
            raise ValueError("commit_min_shrink_fraction must be in (0, 1]")
        if not 0.0 <= self.commit_min_explored_fraction <= 1.0:
            raise ValueError("commit_min_explored_fraction must be in [0, 1]")
        if self.commit_incumbent_padding_fraction < 0.0:
            raise ValueError("commit_incumbent_padding_fraction must be non-negative")
        if not 0.0 < self.exploitation_shrink_fraction < 1.0:
            raise ValueError("exploitation_shrink_fraction must be in (0, 1)")
        for field_name in (
            "exploration_objective_gamma",
            "commit_objective_gamma",
            "exploitation_objective_gamma",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "exploration_support_ema_alpha",
            "commit_support_ema_alpha",
            "exploitation_support_ema_alpha",
        ):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        for field_name in (
            "commit_guidance_top_k",
            "exploitation_guidance_top_k",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "commit_guidance_sigma",
            "commit_guidance_temperature",
            "exploitation_guidance_sigma",
            "exploitation_guidance_temperature",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "commit_uncertainty_weight",
            "commit_drift_strength",
            "exploitation_uncertainty_weight",
            "exploitation_drift_strength",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "commit_surrogate_min_samples",
            "commit_surrogate_max_samples",
            "exploitation_valley_surrogate_min_samples",
            "exploitation_valley_surrogate_max_samples",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.commit_surrogate_max_samples < self.commit_surrogate_min_samples:
            raise ValueError("commit_surrogate_max_samples must be >= commit_surrogate_min_samples")
        if self.exploitation_valley_surrogate_max_samples < self.exploitation_valley_surrogate_min_samples:
            raise ValueError("exploitation_valley_surrogate_max_samples must be >= exploitation_valley_surrogate_min_samples")
        for field_name in (
            "commit_surrogate_regularization",
            "commit_surrogate_min_predicted_improvement",
            "commit_surrogate_max_condition",
            "commit_surrogate_valley_expand",
            "commit_surrogate_cross_shrink",
            "commit_surrogate_support_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.exploitation_valley_probe_evaluations < 0:
            raise ValueError("exploitation_valley_probe_evaluations must be non-negative")
        if not 0.0 < self.exploitation_valley_step_fraction <= 1.0:
            raise ValueError("exploitation_valley_step_fraction must be in (0, 1]")
        if not 0.0 < self.exploitation_valley_step_decay < 1.0:
            raise ValueError("exploitation_valley_step_decay must be in (0, 1)")
        if not 0.0 < self.exploitation_valley_min_step_fraction <= self.exploitation_valley_step_fraction:
            raise ValueError("exploitation_valley_min_step_fraction must be in (0, step_fraction]")
        for field_name in (
            "commit_trust_region_evaluations",
            "exploitation_trust_region_evaluations",
            "trust_region_candidate_pool_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 < self.trust_region_initial_radius_fraction <= 1.0:
            raise ValueError("trust_region_initial_radius_fraction must be in (0, 1]")
        if not 0.0 < self.trust_region_min_radius_fraction <= self.trust_region_initial_radius_fraction:
            raise ValueError("trust_region_min_radius_fraction must be in (0, initial_radius_fraction]")
        if not 0.0 < self.trust_region_shrink_factor < 1.0:
            raise ValueError("trust_region_shrink_factor must be in (0, 1)")
        if self.trust_region_expand_factor <= 1.0:
            raise ValueError("trust_region_expand_factor must be greater than 1")
        for field_name in (
            "commit_acquisition_uncertainty_weight",
            "exploitation_acquisition_uncertainty_weight",
            "trust_region_support_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.commit_surrogate_cross_shrink <= 0.0:
            raise ValueError("commit_surrogate_cross_shrink must be positive")
        if self.commit_surrogate_valley_expand <= 0.0:
            raise ValueError("commit_surrogate_valley_expand must be positive")
        if self.commit_surrogate_max_condition <= 0.0:
            raise ValueError("commit_surrogate_max_condition must be positive")
        for field_name in (
            "commit_mass_weight",
            "commit_density_weight",
            "commit_stability_weight",
            "commit_objective_weight",
            "commit_area_penalty",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
