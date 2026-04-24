"""Field dynamics for the SmoothLife simulator."""

from __future__ import annotations

import numpy as np

from .config import SmoothLifeConfig
from .kernels import periodic_convolve2d
from .transition import smoothlife_transition


def initial_field(config: SmoothLifeConfig, rng: np.random.Generator) -> np.ndarray:
    """Create the randomized signed field used at reset."""

    height, width = config.grid_shape
    field = config.initial_field_center + rng.uniform(-0.75, 0.75, size=(height, width))
    rows = np.linspace(-1.0, 1.0, height, dtype=float)[:, None]
    cols = np.linspace(-1.0, 1.0, width, dtype=float)[None, :]
    field += 0.20 * np.sin(3.0 * np.pi * rows) * np.cos(2.0 * np.pi * cols)
    field += rng.normal(scale=config.initial_field_noise, size=(height, width))
    center_row = height // 2
    center_col = width // 2
    row0 = max(0, center_row - 8)
    row1 = min(height, center_row + 8)
    col0 = max(0, center_col - 8)
    col1 = min(width, center_col + 8)
    field[row0:row1, col0:col1] *= 0.25
    return np.clip(field, config.field_floor, config.field_ceiling)


def vitality(field: np.ndarray) -> np.ndarray:
    """Convert signed field values into SmoothLife vitality."""

    return np.clip(1.0 - np.abs(np.asarray(field, dtype=float)), 0.0, 1.0)


def laplacian(field: np.ndarray) -> np.ndarray:
    """Return a four-neighbor periodic Laplacian."""

    neighbor_sum = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
    )
    return 0.25 * neighbor_sum - field


def refresh_dynamics_fields(
    field: np.ndarray,
    objective_field: np.ndarray,
    config: SmoothLifeConfig,
    inner_kernel: np.ndarray,
    outer_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute inner fill, outer fill, and transition target."""

    live = vitality(field)
    inner_fill = periodic_convolve2d(live, inner_kernel)
    outer_fill = periodic_convolve2d(live, outer_kernel)
    transition, _target_vitality = smoothlife_transition(
        field,
        inner_fill,
        outer_fill,
        config,
        objective_field,
    )
    return np.clip(inner_fill, 0.0, 1.0), np.clip(outer_fill, 0.0, 1.0), transition


def exploration_score_field(field: np.ndarray, transition_field: np.ndarray) -> np.ndarray:
    """Score unexplored pixels by current vitality and transition interest."""

    transition_interest = np.clip(1.0 - np.abs(transition_field), 0.0, 1.0)
    return 0.5 * vitality(field) + 0.5 * transition_interest
