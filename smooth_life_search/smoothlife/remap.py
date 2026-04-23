"""Field remapping helpers used when AGSLS zooms into a basin."""

from __future__ import annotations

import numpy as np


def remap_field_to_bounds(
    field: np.ndarray,
    old_bounds: np.ndarray,
    new_bounds: np.ndarray,
) -> np.ndarray:
    """Resample a field into a new bounding box with bilinear interpolation."""

    height, width = field.shape
    local_x = (np.arange(width, dtype=float) + 0.5) / width
    local_y = (np.arange(height, dtype=float) + 0.5) / height
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="xy")
    world_x = new_bounds[0, 0] + grid_x * (new_bounds[0, 1] - new_bounds[0, 0])
    world_y = new_bounds[1, 0] + grid_y * (new_bounds[1, 1] - new_bounds[1, 0])

    x_old = (world_x - old_bounds[0, 0]) / max(old_bounds[0, 1] - old_bounds[0, 0], 1e-12)
    y_old = (world_y - old_bounds[1, 0]) / max(old_bounds[1, 1] - old_bounds[1, 0], 1e-12)
    x_old = np.clip(x_old, 0.0, 1.0)
    y_old = np.clip(y_old, 0.0, 1.0)

    x_pos = x_old * width - 0.5
    y_pos = y_old * height - 0.5
    x0 = np.floor(x_pos).astype(int)
    y0 = np.floor(y_pos).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    dx = x_pos - x0
    dy = y_pos - y0

    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]
    return (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v01 + (1.0 - dx) * dy * v10 + dx * dy * v11
