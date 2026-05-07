"""Configuration for Matrix SmoothLife optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MatrixSmoothLifeConfig:
    """Policy parameters for SmoothLife-as-genome optimization."""

    max_evaluations: int | None = None
    max_steps: int | None = None
    matrix_shape: tuple[int, int] | None = None
    decoder: str = "local_sparse"
    decoder_gain: float = 1.0
    projection_seed_offset: int = 7919
    cell_alive_threshold: float = 0.005
    local_decoder_block_size: int | None = None
    local_decoder_overlap: int = 2
    local_credit_enabled: bool = True
    local_credit_strength: float = 0.35
    local_credit_learning_rate: float = 0.5
    local_credit_decay: float = 0.96
    local_credit_clip: float = 5.0
    patch_probe_enabled: bool = True
    patch_probe_interval_evaluations: int = 32
    patch_probe_count: int = 4
    patch_probe_step: float = 0.08
    steps_per_evaluation: int = 1
    elite_pull_strength: float = 0.08
    failure_damping: float = 0.04
    reward_decay: float = 0.95
    reward_boost: float = 0.35
    mutation_noise: float = 0.015
    mutation_decay: float = 0.995
    advantage_strength: float = 0.20
    advantage_decay: float = 0.95
    direction_strength: float = 0.15
    direction_decay: float = 0.90
    temperature_init: float = 1.0
    temperature_decay: float = 0.995
    temperature_reheat: float = 0.10
    stagnation_reheat_evaluations: int = 64
    matrix_line_search_enabled: bool = True
    matrix_line_search_alphas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    early_stop_enabled: bool = False
    early_stop_value: float | None = None
    snapshot_interval: int = 8
    store_all_snapshots: bool = True
    best_improvement_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when set")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.matrix_shape is not None:
            if len(self.matrix_shape) != 2:
                raise ValueError("matrix_shape must contain two dimensions")
            if self.matrix_shape[0] < 16 or self.matrix_shape[1] < 16:
                raise ValueError("matrix_shape must be at least 16x16")
        if self.decoder not in {"local_sparse", "random_projection"}:
            raise ValueError("decoder must be local_sparse or random_projection")
        if self.decoder_gain <= 0.0:
            raise ValueError("decoder_gain must be positive")
        if self.cell_alive_threshold < 0.0:
            raise ValueError("cell_alive_threshold must be non-negative")
        if self.local_decoder_block_size is not None and self.local_decoder_block_size <= 0:
            raise ValueError("local_decoder_block_size must be positive when set")
        if self.local_decoder_overlap < 0:
            raise ValueError("local_decoder_overlap must be non-negative")
        if self.local_credit_strength < 0.0:
            raise ValueError("local_credit_strength must be non-negative")
        if self.local_credit_learning_rate < 0.0:
            raise ValueError("local_credit_learning_rate must be non-negative")
        if not 0.0 <= self.local_credit_decay <= 1.0:
            raise ValueError("local_credit_decay must be in [0, 1]")
        if self.local_credit_clip <= 0.0:
            raise ValueError("local_credit_clip must be positive")
        if self.patch_probe_interval_evaluations <= 0:
            raise ValueError("patch_probe_interval_evaluations must be positive")
        if self.patch_probe_count < 0:
            raise ValueError("patch_probe_count must be non-negative")
        if self.patch_probe_step < 0.0:
            raise ValueError("patch_probe_step must be non-negative")
        if self.steps_per_evaluation <= 0:
            raise ValueError("steps_per_evaluation must be positive")
        if not 0.0 <= self.elite_pull_strength <= 1.0:
            raise ValueError("elite_pull_strength must be in [0, 1]")
        if not 0.0 <= self.failure_damping <= 1.0:
            raise ValueError("failure_damping must be in [0, 1]")
        if not 0.0 <= self.reward_decay <= 1.0:
            raise ValueError("reward_decay must be in [0, 1]")
        if self.reward_boost < 0.0:
            raise ValueError("reward_boost must be non-negative")
        if self.mutation_noise < 0.0:
            raise ValueError("mutation_noise must be non-negative")
        if not 0.0 < self.mutation_decay <= 1.0:
            raise ValueError("mutation_decay must be in (0, 1]")
        if self.advantage_strength < 0.0:
            raise ValueError("advantage_strength must be non-negative")
        if not 0.0 <= self.advantage_decay <= 1.0:
            raise ValueError("advantage_decay must be in [0, 1]")
        if self.direction_strength < 0.0:
            raise ValueError("direction_strength must be non-negative")
        if not 0.0 <= self.direction_decay <= 1.0:
            raise ValueError("direction_decay must be in [0, 1]")
        if self.temperature_init <= 0.0:
            raise ValueError("temperature_init must be positive")
        if not 0.0 <= self.temperature_decay <= 1.0:
            raise ValueError("temperature_decay must be in [0, 1]")
        if self.temperature_reheat < 0.0:
            raise ValueError("temperature_reheat must be non-negative")
        if self.stagnation_reheat_evaluations <= 0:
            raise ValueError("stagnation_reheat_evaluations must be positive")
        if not self.matrix_line_search_alphas:
            raise ValueError("matrix_line_search_alphas must not be empty")
        if any(alpha <= 0.0 for alpha in self.matrix_line_search_alphas):
            raise ValueError("matrix_line_search_alphas must contain positive values")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.best_improvement_tolerance < 0.0:
            raise ValueError("best_improvement_tolerance must be non-negative")
