"""Kernel construction and periodic convolution for SmoothLife."""

from __future__ import annotations

import numpy as np


def _smooth_disk_indicator(distances: np.ndarray, radius: float, anti_alias_radius: float) -> np.ndarray:
    inner = max(0.0, radius - anti_alias_radius)
    outer = radius + anti_alias_radius
    weights = np.zeros_like(distances, dtype=float)
    weights[distances <= inner] = 1.0
    band = (distances > inner) & (distances < outer)
    if np.any(band):
        span = max(outer - inner, 1e-12)
        phase = (distances[band] - inner) / span
        weights[band] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return weights


def build_disk_kernel(radius: float, anti_alias_radius: float) -> np.ndarray:
    """Build an anti-aliased disk kernel normalized to unit mass."""

    half_extent = int(np.ceil(radius + anti_alias_radius))
    coords = np.arange(-half_extent, half_extent + 1, dtype=float)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    distances = np.sqrt(xx * xx + yy * yy)
    kernel = _smooth_disk_indicator(distances, radius, anti_alias_radius)
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("disk kernel must have positive mass")
    return kernel / total


def build_ring_kernel(inner_radius: float, outer_radius: float, anti_alias_radius: float) -> np.ndarray:
    """Build an anti-aliased ring kernel normalized to unit mass."""

    half_extent = int(np.ceil(outer_radius + anti_alias_radius))
    coords = np.arange(-half_extent, half_extent + 1, dtype=float)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    distances = np.sqrt(xx * xx + yy * yy)
    outer = _smooth_disk_indicator(distances, outer_radius, anti_alias_radius)
    inner = _smooth_disk_indicator(distances, inner_radius, anti_alias_radius)
    kernel = np.clip(outer - inner, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("ring kernel must have positive mass")
    return kernel / total


def periodic_convolve2d(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution matching the toroidal SmoothLife domain."""

    if field.ndim != 2 or kernel.ndim != 2:
        raise ValueError("field and kernel must be 2D arrays")
    padded = np.zeros_like(field, dtype=float)
    kh, kw = kernel.shape
    row_idx = (np.arange(kh) - kh // 2) % field.shape[0]
    col_idx = (np.arange(kw) - kw // 2) % field.shape[1]
    for kernel_row, field_row in enumerate(row_idx):
        np.add.at(padded[field_row], col_idx, kernel[kernel_row])
    fft_field = np.fft.rfft2(field)
    fft_kernel = np.fft.rfft2(padded)
    return np.fft.irfft2(fft_field * fft_kernel, s=field.shape)
