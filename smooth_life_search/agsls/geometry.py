"""Zoom-box geometry helpers for AGSLS."""

from __future__ import annotations

import numpy as np

from .config import AGSLSConfig


def minimum_zoom_widths(current_bounds: np.ndarray, grid_shape: tuple[int, int], config: AGSLSConfig) -> np.ndarray:
    """Return the minimum zoom widths implied by grid cell size."""

    height, width = grid_shape
    current_widths = np.asarray(current_bounds, dtype=float)[:, 1] - np.asarray(current_bounds, dtype=float)[:, 0]
    cell_widths = np.asarray([current_widths[0] / width, current_widths[1] / height], dtype=float)
    return config.min_zoom_cells * cell_widths


def expand_bounds_to_min_widths(
    bounds: np.ndarray,
    center: np.ndarray,
    min_widths: np.ndarray,
    current_bounds: np.ndarray,
) -> np.ndarray:
    """Expand bounds around a center until each side satisfies minimum widths."""

    expanded = np.asarray(bounds, dtype=float).copy()
    center = np.asarray(center, dtype=float)
    for idx in range(2):
        width = float(expanded[idx, 1] - expanded[idx, 0])
        if width >= float(min_widths[idx]):
            continue
        half_width = 0.5 * float(min_widths[idx])
        resolved_center = float(np.clip(center[idx], current_bounds[idx, 0], current_bounds[idx, 1]))
        lower = resolved_center - half_width
        upper = resolved_center + half_width
        if lower < current_bounds[idx, 0]:
            upper += current_bounds[idx, 0] - lower
            lower = current_bounds[idx, 0]
        if upper > current_bounds[idx, 1]:
            lower -= upper - current_bounds[idx, 1]
            upper = current_bounds[idx, 1]
        expanded[idx, 0] = max(current_bounds[idx, 0], lower)
        expanded[idx, 1] = min(current_bounds[idx, 1], upper)
    return expanded


def enforce_min_side_fraction(
    bounds: np.ndarray,
    current_bounds: np.ndarray,
    original_bounds: np.ndarray,
    config: AGSLSConfig,
) -> np.ndarray:
    """Prevent zoom boxes from shrinking below configured original-box fractions."""

    new_bounds = np.asarray(bounds, dtype=float).copy()
    min_widths = config.min_side_fraction * (original_bounds[:, 1] - original_bounds[:, 0])
    widths = new_bounds[:, 1] - new_bounds[:, 0]
    for idx in range(2):
        if widths[idx] >= min_widths[idx]:
            continue
        center = 0.5 * (new_bounds[idx, 0] + new_bounds[idx, 1])
        half_width = 0.5 * min_widths[idx]
        lower = max(current_bounds[idx, 0], center - half_width)
        upper = min(current_bounds[idx, 1], center + half_width)
        if upper - lower < min_widths[idx]:
            if lower <= current_bounds[idx, 0]:
                upper = min(current_bounds[idx, 1], lower + min_widths[idx])
            else:
                lower = max(current_bounds[idx, 0], upper - min_widths[idx])
        new_bounds[idx, 0] = lower
        new_bounds[idx, 1] = upper
    return new_bounds


def centered_bounds(center: np.ndarray, widths: np.ndarray, current_bounds: np.ndarray) -> np.ndarray:
    """Return bounds with the given width centered and clipped to current bounds."""

    resolved_center = np.asarray(center, dtype=float)
    resolved_widths = np.asarray(widths, dtype=float)
    bounds = np.column_stack((resolved_center - 0.5 * resolved_widths, resolved_center + 0.5 * resolved_widths))
    bounds[:, 0] = np.maximum(bounds[:, 0], current_bounds[:, 0])
    bounds[:, 1] = np.minimum(bounds[:, 1], current_bounds[:, 1])
    return expand_bounds_to_min_widths(bounds, resolved_center, resolved_widths, current_bounds)


def finalize_zoom_bounds(
    bounds: np.ndarray,
    center: np.ndarray,
    current_bounds: np.ndarray,
    original_bounds: np.ndarray,
    grid_shape: tuple[int, int],
    config: AGSLSConfig,
    *,
    anchor_point: np.ndarray | None = None,
    incumbent_point: np.ndarray | None = None,
) -> np.ndarray:
    """Apply minimum widths, padding, edge risk, and incumbent retention."""

    min_widths = minimum_zoom_widths(current_bounds, grid_shape, config)
    resolved_center = np.asarray(center, dtype=float)
    new_bounds = expand_bounds_to_min_widths(np.asarray(bounds, dtype=float), resolved_center, min_widths, current_bounds)
    anchor = resolved_center if anchor_point is None else np.asarray(anchor_point, dtype=float)
    widths = new_bounds[:, 1] - new_bounds[:, 0]
    directional_padding = config.zoom_padding * widths
    for idx in range(2):
        edge_band = config.edge_risk_fraction * max(widths[idx], 1e-12)
        if anchor[idx] - new_bounds[idx, 0] <= edge_band:
            new_bounds[idx, 0] -= directional_padding[idx]
        if new_bounds[idx, 1] - anchor[idx] <= edge_band:
            new_bounds[idx, 1] += directional_padding[idx]
    if incumbent_point is not None and np.all(np.isfinite(incumbent_point)):
        resolved_incumbent = np.asarray(incumbent_point, dtype=float)
        for idx in range(2):
            if resolved_incumbent[idx] < current_bounds[idx, 0] or resolved_incumbent[idx] > current_bounds[idx, 1]:
                continue
            width = max(float(new_bounds[idx, 1] - new_bounds[idx, 0]), float(min_widths[idx]))
            edge_band = config.edge_risk_fraction * width
            if resolved_incumbent[idx] < new_bounds[idx, 0]:
                new_bounds[idx, 0] = max(current_bounds[idx, 0], resolved_incumbent[idx] - edge_band)
            elif resolved_incumbent[idx] > new_bounds[idx, 1]:
                new_bounds[idx, 1] = min(current_bounds[idx, 1], resolved_incumbent[idx] + edge_band)
    widths = new_bounds[:, 1] - new_bounds[:, 0]
    padding = config.zoom_padding * widths
    new_bounds[:, 0] -= padding
    new_bounds[:, 1] += padding
    new_bounds[:, 0] = np.maximum(new_bounds[:, 0], current_bounds[:, 0])
    new_bounds[:, 1] = np.minimum(new_bounds[:, 1], current_bounds[:, 1])
    new_bounds = expand_bounds_to_min_widths(new_bounds, resolved_center, min_widths, current_bounds)
    return enforce_min_side_fraction(new_bounds, current_bounds, original_bounds, config)
