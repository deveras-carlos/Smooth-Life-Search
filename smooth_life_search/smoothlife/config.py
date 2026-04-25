"""Configuration for the literal 2D SmoothLife simulator/search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TimeMode = Literal["discrete", "continuous"]
RunMode = Literal["simulation", "search"]


@dataclass(slots=True)
class SmoothLifeConfig:
    """Parameters for a dense 2D SmoothLife field."""

    grid_shape: tuple[int, int] = (128, 128)
    inner_radius: float = 7.0
    outer_radius: float = 21.0
    anti_alias_radius: float = 1.0
    birth_low: float = 0.278
    birth_high: float = 0.365
    death_low: float = 0.267
    death_high: float = 0.445
    alpha_n: float = 0.028
    alpha_m: float = 0.147
    dt: float = 0.25
    diffusion: float = 0.10
    objective_coupling: float = 0.30
    objective_gamma: float = 1.0
    support_ema_alpha: float = 0.0
    field_floor: float = -1.0
    field_ceiling: float = 1.0
    initial_field_center: float = 0.0
    initial_field_noise: float = 0.10
    evaluations_per_step: int = 16
    snapshot_interval: int = 2
    time_mode: TimeMode = "discrete"
    run_mode: RunMode = "search"
    maximize: bool = False
    preset: str | None = None
    store_all_snapshots: bool = True
    subpixel_best_point: bool = True
    subpixel_confirm: bool = True
    subpixel_confirm_candidates: int = 1
    exploitation_score_late_stage: bool = False

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2:
            raise ValueError("grid_shape must be a 2D shape")
        if self.grid_shape[0] < 16 or self.grid_shape[1] < 16:
            raise ValueError("grid_shape must be at least 16x16")
        if self.inner_radius <= 0.0 or self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be greater than inner_radius > 0")
        if self.anti_alias_radius <= 0.0:
            raise ValueError("anti_alias_radius must be positive")
        if not 0.0 < self.alpha_n <= 1.0 or not 0.0 < self.alpha_m <= 1.0:
            raise ValueError("alpha_n and alpha_m must be in (0, 1]")
        if not 0.0 < self.dt <= 1.0:
            raise ValueError("dt must be in (0, 1]")
        if self.diffusion < 0.0:
            raise ValueError("diffusion must be non-negative")
        if not 0.0 <= self.objective_coupling <= 1.0:
            raise ValueError("objective_coupling must be in [0, 1]")
        if self.objective_gamma <= 0.0:
            raise ValueError("objective_gamma must be positive")
        if not 0.0 <= self.support_ema_alpha <= 1.0:
            raise ValueError("support_ema_alpha must be in [0, 1]")
        if self.field_floor >= self.field_ceiling:
            raise ValueError("field_floor must be less than field_ceiling")
        if self.evaluations_per_step <= 0:
            raise ValueError("evaluations_per_step must be positive")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.subpixel_confirm_candidates < 1:
            raise ValueError("subpixel_confirm_candidates must be at least 1")
