"""Persistent sample archive for point-cloud search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import PointCloudSample


@dataclass(slots=True)
class PointCloudArchive:
    """Unique objective samples keyed by exact floating-point coordinates."""

    samples: list[PointCloudSample]
    _keys: dict[tuple[str, ...], int]

    @classmethod
    def empty(cls) -> "PointCloudArchive":
        return cls(samples=[], _keys={})

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def key(point: np.ndarray) -> tuple[str, ...]:
        resolved = np.asarray(point, dtype=float)
        return tuple(float(value).hex() for value in resolved)

    def get(self, point: np.ndarray) -> PointCloudSample | None:
        index = self._keys.get(self.key(point))
        if index is None:
            return None
        return self.samples[index]

    def add(self, point: np.ndarray, value: float, *, source: str, batch_index: int) -> PointCloudSample | None:
        resolved_point = np.asarray(point, dtype=float)
        resolved_value = float(value)
        if resolved_point.ndim != 1 or not np.all(np.isfinite(resolved_point)) or not np.isfinite(resolved_value):
            return None
        key = self.key(resolved_point)
        if key in self._keys:
            return None
        sample = PointCloudSample(
            point=resolved_point.copy(),
            value=resolved_value,
            source=str(source),
            batch_index=int(batch_index),
        )
        self._keys[key] = len(self.samples)
        self.samples.append(sample)
        return sample

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
        return (
            np.vstack([sample.point for sample in self.samples]).astype(float, copy=False),
            np.asarray([sample.value for sample in self.samples], dtype=float),
        )

    def best_index(self, *, maximize: bool) -> int | None:
        if not self.samples:
            return None
        values = np.asarray([sample.value for sample in self.samples], dtype=float)
        return int(np.argmax(values) if maximize else np.argmin(values))

    def elite_indices(self, *, maximize: bool, fraction: float, minimum: int = 1) -> np.ndarray:
        if not self.samples:
            return np.empty((0,), dtype=int)
        values = np.asarray([sample.value for sample in self.samples], dtype=float)
        count = max(int(np.ceil(float(fraction) * values.size)), int(minimum))
        count = min(count, values.size)
        order = np.argsort(values)
        if maximize:
            order = order[::-1]
        return np.asarray(order[:count], dtype=int)

    def nearest_distance(self, points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        archive_points, _values = self.arrays()
        candidates = np.asarray(points, dtype=float)
        if candidates.ndim != 2:
            return np.empty((0,), dtype=float)
        if archive_points.size == 0:
            return np.ones(candidates.shape[0], dtype=float)
        widths = np.maximum(np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0], 1e-12)
        normalized_archive = (archive_points - bounds[:, 0]) / widths
        normalized_candidates = (candidates - bounds[:, 0]) / widths
        delta = normalized_candidates[:, None, :] - normalized_archive[None, :, :]
        return np.min(np.linalg.norm(delta, axis=2), axis=1)
