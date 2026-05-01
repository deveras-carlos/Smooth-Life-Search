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


def _bilinear_sample_clipped(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    height, width = field.shape
    clipped_rows = np.clip(rows, 0.0, float(height - 1))
    clipped_cols = np.clip(cols, 0.0, float(width - 1))
    row0 = np.floor(clipped_rows).astype(int)
    col0 = np.floor(clipped_cols).astype(int)
    row1 = np.minimum(row0 + 1, height - 1)
    col1 = np.minimum(col0 + 1, width - 1)
    row_weight = clipped_rows - row0
    col_weight = clipped_cols - col0
    top = (1.0 - col_weight) * field[row0, col0] + col_weight * field[row0, col1]
    bottom = (1.0 - col_weight) * field[row1, col0] + col_weight * field[row1, col1]
    return (1.0 - row_weight) * top + row_weight * bottom


def objective_drift_field(field: np.ndarray, objective_field: np.ndarray, config: SmoothLifeConfig) -> np.ndarray:
    """Advect the signed field slightly up the objective-support gradient."""

    strength = float(config.objective_drift_strength)
    clip = float(config.objective_drift_clip)
    if strength <= 0.0 or clip <= 0.0 or config.run_mode == "simulation":
        return field
    support = np.asarray(objective_field, dtype=float)
    if support.shape != field.shape or not np.any(np.isfinite(support)):
        return field
    support = np.nan_to_num(support, nan=0.5, posinf=1.0, neginf=0.0)
    grad_y, grad_x = np.gradient(support)
    norm = np.hypot(grad_y, grad_x)
    if float(np.max(norm)) <= 1e-15:
        return field
    velocity_y = np.divide(grad_y, norm, out=np.zeros_like(grad_y), where=norm > 1e-15)
    velocity_x = np.divide(grad_x, norm, out=np.zeros_like(grad_x), where=norm > 1e-15)
    height, width = field.shape
    rows, cols = np.indices(field.shape, dtype=float)
    departure_rows = rows - strength * velocity_y
    departure_cols = cols - strength * velocity_x
    advected = _bilinear_sample_clipped(np.asarray(field, dtype=float), departure_rows, departure_cols)
    delta = np.clip(advected - field, -clip, clip)
    return np.clip(field + delta, config.field_floor, config.field_ceiling)


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


def _four_neighbor_boundary(mask: np.ndarray) -> np.ndarray:
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    dilated = mask | up | down | left | right
    eroded = mask & up & down & left & right
    return dilated & ~eroded


def exploitation_score_field(
    field: np.ndarray,
    transition_field: np.ndarray,
    objective_field: np.ndarray,
    evaluated_mask: np.ndarray,
) -> np.ndarray:
    """Score late-stage pixels by support uncertainty and basin boundaries."""

    objective = np.asarray(objective_field, dtype=float)
    if not np.any(evaluated_mask) or float(np.nanmax(objective) - np.nanmin(objective)) <= 1e-12:
        return exploration_score_field(field, transition_field)

    live = vitality(field)
    support = live * np.clip(objective, 0.0, 1.0)
    uncertainty = np.clip(1.0 - np.abs(objective - 0.5) * 2.0, 0.0, 1.0)
    finite_support = support[np.isfinite(support)]
    if finite_support.size == 0:
        return exploration_score_field(field, transition_field)

    threshold = float(np.quantile(finite_support, 0.75))
    high_support = support >= threshold
    if not np.any(high_support) or np.all(high_support):
        boundary_bonus = np.zeros_like(support, dtype=float)
    else:
        boundary_bonus = _four_neighbor_boundary(high_support).astype(float)
    return np.clip(uncertainty * support + 0.25 * boundary_bonus, 0.0, 1.25)
