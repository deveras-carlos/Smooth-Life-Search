"""SmoothLife-native optimizer where the whole matrix decodes to one N-D point."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from ..core import SearchRun, normalize_bounds_nd
from ..smoothlife.config import SmoothLifeConfig
from ..smoothlife.dynamics import initial_field, laplacian, refresh_dynamics_fields, vitality
from ..smoothlife.kernels import build_disk_kernel, build_ring_kernel
from ..smoothlife.presets import apply_preset
from .config import MatrixSmoothLifeConfig
from .models import MatrixSmoothLifeSample, MatrixSmoothLifeSnapshot

Objective = Callable[[np.ndarray], float]


class MatrixSmoothLifeSearch:
    """Optimize an N-D objective by evolving a 2D SmoothLife genome matrix."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[tuple[float, float]],
        smoothlife_config: SmoothLifeConfig | None = None,
        matrix_config: MatrixSmoothLifeConfig | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = normalize_bounds_nd(bounds, owner="MatrixSmoothLifeSearch")
        self.dimension = int(self.original_bounds.shape[0])
        self.config = matrix_config or MatrixSmoothLifeConfig()
        self.matrix_shape = self._resolve_matrix_shape(self.dimension, self.config.matrix_shape)
        self.smoothlife_config = replace(
            apply_preset(smoothlife_config or SmoothLifeConfig(preset="search")),
            grid_shape=self.matrix_shape,
        )
        self.display_bounds = np.asarray(
            [(0.0, float(self.matrix_shape[1])), (0.0, float(self.matrix_shape[0]))],
            dtype=float,
        )
        self.rng = np.random.default_rng()
        self.field = np.zeros(self.matrix_shape, dtype=float)
        self.reward_field = np.zeros(self.matrix_shape, dtype=float)
        self.advantage_field = np.zeros(self.matrix_shape, dtype=float)
        self.direction_field = np.zeros(self.matrix_shape, dtype=float)
        self.local_credit_field = np.zeros(self.matrix_shape, dtype=float)
        self.neighborhood_credit = np.zeros(self.matrix_shape, dtype=float)
        self.temperature_field = np.ones(self.matrix_shape, dtype=float) * float(self.config.temperature_init)
        self.best_field = np.zeros(self.matrix_shape, dtype=float)
        self.inner_fill = np.zeros(self.matrix_shape, dtype=float)
        self.outer_fill = np.zeros(self.matrix_shape, dtype=float)
        self.transition_field = np.zeros(self.matrix_shape, dtype=float)
        self.projection = np.zeros((self.dimension, int(np.prod(self.matrix_shape))), dtype=float)
        self.local_decoder_axes = np.zeros((int(np.prod(self.matrix_shape)), 1), dtype=int)
        self.local_decoder_weights = np.zeros((int(np.prod(self.matrix_shape)), 1, 8), dtype=float)
        self.local_decoder_axis_counts = np.ones(self.dimension, dtype=float)
        self._local_block_size = 1
        self.archive: list[MatrixSmoothLifeSample] = []
        self._field_values: dict[tuple[str, ...], float] = {}
        self.snapshots: list[MatrixSmoothLifeSnapshot] = []
        self.best_point = np.mean(self.original_bounds, axis=1)
        self.best_value = self._worst_value()
        self.local_best_point = self.best_point.copy()
        self.local_best_value = self.best_value
        self.box_best_point = self.best_point.copy()
        self.box_best_value = self.best_value
        self.step_index = 0
        self._active_max_evaluations: int | None = None
        self._stop_reason = "not_started"
        self._current_noise = float(self.config.mutation_noise)
        self._last_improved = False
        self._last_improvement_amount = 0.0
        self._rng_seed = 0
        self._best_source = "none"
        self._stagnation_evaluations = 0
        self._line_search_improvements = 0
        self._patch_probe_count = 0
        self._patch_probe_improvements = 0
        self._next_patch_probe_evaluation = int(self.config.patch_probe_interval_evaluations)
        self._last_successful_field_step: np.ndarray | None = None
        self._last_successful_point_step: np.ndarray | None = None
        self._last_evaluated_field: np.ndarray | None = None
        self._last_evaluated_value: float | None = None
        self.inner_kernel = build_disk_kernel(self.smoothlife_config.inner_radius, self.smoothlife_config.anti_alias_radius)
        self.outer_kernel = build_ring_kernel(
            self.smoothlife_config.inner_radius,
            self.smoothlife_config.outer_radius,
            self.smoothlife_config.anti_alias_radius,
        )
        self.reset()

    @staticmethod
    def _resolve_matrix_shape(dimension: int, configured: tuple[int, int] | None) -> tuple[int, int]:
        if configured is not None:
            if int(configured[0]) * int(configured[1]) < int(dimension):
                raise ValueError("matrix_shape must contain at least dimension cells")
            return int(configured[0]), int(configured[1])
        target_cells = max(256, 4 * int(dimension))
        height = max(16, int(np.ceil(np.sqrt(target_cells))))
        width = max(16, int(np.ceil(target_cells / height)))
        while height * width < target_cells:
            width += 1
        return height, width

    def reset(self, seed: int | None = None) -> None:
        """Reset the matrix genome, archive, projection decoder, and diagnostics."""

        self._rng_seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)
        self.projection = self._build_projection(self._rng_seed + int(self.config.projection_seed_offset))
        self._build_local_decoder(self._rng_seed + int(self.config.projection_seed_offset) + 104729)
        self.field = np.zeros(self.matrix_shape, dtype=float)
        self.reward_field = np.zeros(self.matrix_shape, dtype=float)
        self.advantage_field = np.zeros(self.matrix_shape, dtype=float)
        self.direction_field = np.zeros(self.matrix_shape, dtype=float)
        self.local_credit_field = np.zeros(self.matrix_shape, dtype=float)
        self.neighborhood_credit = np.zeros(self.matrix_shape, dtype=float)
        self.temperature_field = np.ones(self.matrix_shape, dtype=float) * float(self.config.temperature_init)
        self.best_field = np.zeros(self.matrix_shape, dtype=float)
        self.inner_fill = np.zeros(self.matrix_shape, dtype=float)
        self.outer_fill = np.zeros(self.matrix_shape, dtype=float)
        self.transition_field = np.zeros(self.matrix_shape, dtype=float)
        self.archive = []
        self._field_values = {}
        self.snapshots = []
        self.best_point = self.decode_field(self.field)
        self.best_value = self._worst_value()
        self.local_best_point = self.best_point.copy()
        self.local_best_value = self.best_value
        self.box_best_point = self.best_point.copy()
        self.box_best_value = self.best_value
        self.step_index = 0
        self._active_max_evaluations = None
        self._stop_reason = "not_started"
        self._current_noise = float(self.config.mutation_noise)
        self._last_improved = False
        self._last_improvement_amount = 0.0
        self._best_source = "none"
        self._stagnation_evaluations = 0
        self._line_search_improvements = 0
        self._patch_probe_count = 0
        self._patch_probe_improvements = 0
        self._next_patch_probe_evaluation = int(self.config.patch_probe_interval_evaluations)
        self._last_successful_field_step = None
        self._last_successful_point_step = None
        self._last_evaluated_field = None
        self._last_evaluated_value = None
        self._refresh_dynamics()

    def _build_projection(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(self.dimension, int(np.prod(self.matrix_shape))))
        norms = np.linalg.norm(projection, axis=1, keepdims=True)
        return projection / np.maximum(norms, 1e-12)

    def _resolve_local_block_size(self) -> int:
        configured = self.config.local_decoder_block_size
        if configured is None:
            return max(1, min(8, self.dimension))
        return max(1, min(int(configured), self.dimension))

    def _build_local_decoder(self, seed: int) -> None:
        cell_count = int(np.prod(self.matrix_shape))
        block_size = self._resolve_local_block_size()
        overlap = min(int(self.config.local_decoder_overlap), max(block_size - 1, 0))
        stride = max(1, block_size - overlap)
        starts = (np.arange(cell_count, dtype=int) * stride) % self.dimension
        axes = (starts[:, None] + np.arange(block_size, dtype=int)[None, :]) % self.dimension

        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(cell_count, block_size, 8))
        norms = np.linalg.norm(weights, axis=2, keepdims=True)
        weights = weights / np.maximum(norms, 1e-12)

        axis_counts = np.zeros(self.dimension, dtype=float)
        np.add.at(axis_counts, axes.ravel(), 1.0)
        self.local_decoder_axes = axes.astype(int, copy=False)
        self.local_decoder_weights = weights.astype(float, copy=False)
        self.local_decoder_axis_counts = np.maximum(axis_counts, 1.0)
        self._local_block_size = int(block_size)

    @staticmethod
    def _neighbor_offsets() -> tuple[tuple[int, int], ...]:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    def _neighborhood_features(self, field: np.ndarray) -> np.ndarray:
        matrix = np.asarray(field, dtype=float)
        height, width = self.matrix_shape
        features = np.zeros((height, width, 8), dtype=float)
        threshold = float(self.config.cell_alive_threshold)
        for index, (dy, dx) in enumerate(self._neighbor_offsets()):
            src_rows = slice(max(0, -dy), min(height, height - dy))
            dst_rows = slice(max(0, dy), min(height, height + dy))
            src_cols = slice(max(0, -dx), min(width, width - dx))
            dst_cols = slice(max(0, dx), min(width, width + dx))
            values = matrix[src_rows, src_cols]
            features[dst_rows, dst_cols, index] = np.where(np.abs(values) >= threshold, values, 0.0)
        return features

    def _neighborhood_credit_to_cell_field(self) -> np.ndarray:
        height, width = self.matrix_shape
        accum = np.zeros(self.matrix_shape, dtype=float)
        counts = np.zeros(self.matrix_shape, dtype=float)
        credit = np.asarray(self.neighborhood_credit, dtype=float)
        for dy, dx in self._neighbor_offsets():
            src_rows = slice(max(0, -dy), min(height, height - dy))
            dst_rows = slice(max(0, dy), min(height, height + dy))
            src_cols = slice(max(0, -dx), min(width, width - dx))
            dst_cols = slice(max(0, dx), min(width, width + dx))
            accum[src_rows, src_cols] += credit[dst_rows, dst_cols]
            counts[src_rows, src_cols] += 1.0
        field = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0.0)
        clip = float(self.config.local_credit_clip)
        return np.clip(field / max(clip, 1e-12), -1.0, 1.0)

    def _worst_value(self) -> float:
        return -np.inf if self.smoothlife_config.maximize else np.inf

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        tolerance = float(self.config.best_improvement_tolerance)
        if not np.isfinite(incumbent):
            return True
        if self.smoothlife_config.maximize:
            return float(candidate) > float(incumbent) + tolerance
        return float(candidate) < float(incumbent) - tolerance

    def _target(self, value: float) -> float:
        return -float(value) if self.smoothlife_config.maximize else float(value)

    def _improvement_amount(self, before: float, after: float) -> float:
        if not np.isfinite(before) or not np.isfinite(after):
            return 0.0
        return max(0.0, self._target(before) - self._target(after))

    def _remaining(self) -> int:
        if self._active_max_evaluations is None:
            if self.config.max_evaluations is None:
                raise ValueError("MatrixSmoothLifeSearch requires max_evaluations or run(evaluations=...)")
            self._active_max_evaluations = int(self.config.max_evaluations)
        return max(int(self._active_max_evaluations) - len(self.archive), 0)

    def decode_field(self, field: np.ndarray) -> np.ndarray:
        """Decode a full matrix field into one bounded N-D candidate point."""

        if self.config.decoder == "local_sparse":
            return self._decode_local_sparse(field)
        flat = np.asarray(field, dtype=float).ravel()
        raw = float(self.config.decoder_gain) * (self.projection @ flat)
        normalized = 0.5 + 0.5 * np.tanh(raw)
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        return lower + normalized * (upper - lower)

    def _decode_local_sparse(self, field: np.ndarray) -> np.ndarray:
        features = self._neighborhood_features(field).reshape(-1, 8)
        contributions = np.einsum("cbn,cn->cb", self.local_decoder_weights, features, optimize=True)
        raw = np.zeros(self.dimension, dtype=float)
        np.add.at(raw, self.local_decoder_axes.ravel(), contributions.ravel())
        raw = raw / self.local_decoder_axis_counts
        normalized = 0.5 + 0.5 * np.tanh(float(self.config.decoder_gain) * raw)
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        return lower + normalized * (upper - lower)

    @staticmethod
    def _field_key(field: np.ndarray) -> tuple[str, ...]:
        return tuple(f"{value:.12g}" for value in np.asarray(field, dtype=float).ravel())

    def _refresh_dynamics(self) -> None:
        self.inner_fill, self.outer_fill, self.transition_field = refresh_dynamics_fields(
            self.field,
            self.reward_field,
            self.smoothlife_config,
            self.inner_kernel,
            self.outer_kernel,
            self.advantage_field,
            self.direction_field,
            self.local_credit_field if self.config.local_credit_enabled else None,
            self.config.advantage_strength,
            self.config.direction_strength,
            self.config.local_credit_strength if self.config.local_credit_enabled else 0.0,
        )

    def _normalized_point(self, point: np.ndarray) -> np.ndarray:
        lower = self.original_bounds[:, 0]
        span = np.maximum(self.original_bounds[:, 1] - lower, 1e-12)
        return (np.asarray(point, dtype=float) - lower) / span

    def _matrix_direction_from_decoded_delta(self, decoded_delta: np.ndarray) -> np.ndarray:
        if self.config.decoder == "local_sparse":
            return self._local_matrix_signal_from_decoded_delta(decoded_delta, normalize=True)
        raw = self.projection.T @ np.asarray(decoded_delta, dtype=float)
        matrix = raw.reshape(self.matrix_shape)
        scale = max(float(np.max(np.abs(matrix))), 1e-12)
        return np.clip(matrix / scale, -1.0, 1.0)

    def _local_matrix_signal_from_decoded_delta(self, decoded_delta: np.ndarray, *, normalize: bool) -> np.ndarray:
        desired = np.asarray(decoded_delta, dtype=float)
        block_targets = desired[self.local_decoder_axes]
        feature_signal = np.einsum("cb,cbn->cn", block_targets, self.local_decoder_weights, optimize=True)
        feature_signal = feature_signal.reshape((*self.matrix_shape, 8))
        matrix = np.zeros(self.matrix_shape, dtype=float)
        counts = np.zeros(self.matrix_shape, dtype=float)
        height, width = self.matrix_shape
        for index, (dy, dx) in enumerate(self._neighbor_offsets()):
            src_rows = slice(max(0, -dy), min(height, height - dy))
            dst_rows = slice(max(0, dy), min(height, height + dy))
            src_cols = slice(max(0, -dx), min(width, width - dx))
            dst_cols = slice(max(0, dx), min(width, width + dx))
            matrix[src_rows, src_cols] += feature_signal[dst_rows, dst_cols, index]
            counts[src_rows, src_cols] += 1.0
        matrix = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0.0)
        scale = max(float(np.max(np.abs(matrix))), 1e-12) if normalize else max(1.0, float(np.max(np.abs(matrix))))
        return np.clip(matrix / scale, -1.0, 1.0)

    @staticmethod
    def _field_step_signal(field_step: np.ndarray) -> np.ndarray:
        step = np.asarray(field_step, dtype=float)
        scale = max(float(np.max(np.abs(step))), 1e-12)
        return np.clip(step / scale, -1.0, 1.0)

    def _credit_scale(self, previous_value: float, value: float) -> float:
        improvement = self._improvement_amount(previous_value, value)
        target_scale = max(abs(float(self._target(previous_value))), 1.0)
        return float(np.clip(improvement / target_scale, 1e-4, 1.0))

    def _signed_target_delta(self, previous_value: float, value: float) -> float:
        if not np.isfinite(previous_value) or not np.isfinite(value):
            return 0.0
        target_scale = max(abs(float(self._target(previous_value))), 1.0)
        return float(np.clip((self._target(previous_value) - self._target(value)) / target_scale, -1.0, 1.0))

    def _refresh_local_credit_field(self) -> None:
        if not self.config.local_credit_enabled:
            self.local_credit_field.fill(0.0)
            return
        self.local_credit_field = self._neighborhood_credit_to_cell_field()

    def _update_local_credit_from_evaluation(
        self,
        previous_field: np.ndarray | None,
        previous_value: float | None,
        field: np.ndarray,
        value: float,
    ) -> None:
        if not self.config.local_credit_enabled:
            return
        self.neighborhood_credit *= float(self.config.local_credit_decay)
        if previous_field is not None and previous_value is not None and np.isfinite(previous_value):
            previous_features = self._neighborhood_features(previous_field)
            current_features = self._neighborhood_features(field)
            change = np.sum(np.abs(current_features - previous_features), axis=2)
            change_max = float(np.max(change))
            signed_delta = self._signed_target_delta(float(previous_value), value)
            if change_max > 1e-12 and abs(signed_delta) > 0.0:
                local_signal = change / change_max
                self.neighborhood_credit += (
                    float(self.config.local_credit_learning_rate) * signed_delta * local_signal
                )
        clip = float(self.config.local_credit_clip)
        self.neighborhood_credit = np.clip(self.neighborhood_credit, -clip, clip)
        self._refresh_local_credit_field()

    def _apply_direct_patch_credit(self, row: int, col: int, previous_value: float, value: float) -> None:
        if not self.config.local_credit_enabled or not np.isfinite(previous_value):
            return
        signed_delta = self._signed_target_delta(previous_value, value)
        if abs(signed_delta) <= 0.0:
            return
        self.neighborhood_credit[int(row), int(col)] += float(self.config.local_credit_learning_rate) * signed_delta
        clip = float(self.config.local_credit_clip)
        self.neighborhood_credit = np.clip(self.neighborhood_credit, -clip, clip)
        self._refresh_local_credit_field()

    def _record_improvement_credit(
        self,
        *,
        previous_field: np.ndarray,
        previous_point: np.ndarray,
        previous_value: float,
        field: np.ndarray,
        point: np.ndarray,
        value: float,
    ) -> None:
        if not np.isfinite(previous_value):
            return
        field_step = np.asarray(field, dtype=float) - np.asarray(previous_field, dtype=float)
        decoded_delta = self._normalized_point(point) - self._normalized_point(previous_point)
        field_signal = self._field_step_signal(field_step)
        direction_signal = self._matrix_direction_from_decoded_delta(decoded_delta)
        scale = self._credit_scale(previous_value, value)
        self.advantage_field = np.clip(
            float(self.config.advantage_decay) * self.advantage_field + scale * field_signal,
            -1.0,
            1.0,
        )
        self.direction_field = np.clip(
            float(self.config.direction_decay) * self.direction_field + scale * direction_signal,
            -1.0,
            1.0,
        )
        local_cooling = np.clip(0.5 * np.abs(field_signal) + 0.5 * vitality(field), 0.0, 1.0)
        self.temperature_field = np.clip(self.temperature_field * (1.0 - 0.25 * scale * local_cooling), 0.05, 3.0)
        self._stagnation_evaluations = 0
        self._last_successful_field_step = field_step.copy()
        self._last_successful_point_step = np.asarray(point, dtype=float) - np.asarray(previous_point, dtype=float)

    def _record_non_improvement_feedback(self) -> None:
        self._stagnation_evaluations += 1
        self.advantage_field *= float(self.config.advantage_decay)
        self.direction_field *= float(self.config.direction_decay)
        if self.config.local_credit_enabled:
            self.neighborhood_credit *= float(self.config.local_credit_decay)
            self._refresh_local_credit_field()
        self.temperature_field *= float(self.config.temperature_decay)
        if self._stagnation_evaluations >= int(self.config.stagnation_reheat_evaluations):
            certainty = np.clip(0.5 * (np.abs(self.advantage_field) + np.abs(self.direction_field)), 0.0, 1.0)
            self.temperature_field = np.clip(
                self.temperature_field + float(self.config.temperature_reheat) * (1.0 - certainty),
                0.05,
                3.0,
            )

    def _reward_support(self) -> np.ndarray:
        support = np.asarray(self.reward_field, dtype=float)
        if not np.any(np.isfinite(support)):
            return np.zeros(self.matrix_shape, dtype=float)
        support = np.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0)
        maximum = float(np.max(support))
        if maximum <= 1e-12:
            return np.zeros(self.matrix_shape, dtype=float)
        return np.clip(support / maximum, 0.0, 1.0)

    def _evaluate_current_field(self, source: str) -> bool:
        if self._remaining() <= 0:
            return False
        key = self._field_key(self.field)
        if key in self._field_values:
            self._last_improved = False
            self._last_improvement_amount = 0.0
            return False
        point = self.decode_field(self.field)
        previous_best = float(self.best_value)
        previous_best_field = self.best_field.copy()
        previous_best_point = self.best_point.copy()
        previous_evaluated_field = None if self._last_evaluated_field is None else self._last_evaluated_field.copy()
        previous_evaluated_value = self._last_evaluated_value
        value = float(self.objective(point))
        sample = MatrixSmoothLifeSample(
            point=point.copy(),
            value=value,
            source=str(source),
            evaluation=len(self.archive) + 1,
            step_index=int(self.step_index),
            field_key=key,
        )
        self.archive.append(sample)
        self._field_values[key] = value
        self._update_local_credit_from_evaluation(previous_evaluated_field, previous_evaluated_value, self.field, value)
        self._last_evaluated_field = self.field.copy()
        self._last_evaluated_value = value
        improved = self._is_better(value, self.best_value)
        self._last_improved = bool(improved)
        self._last_improvement_amount = self._improvement_amount(previous_best, value)
        if improved:
            self._record_improvement_credit(
                previous_field=previous_best_field,
                previous_point=previous_best_point,
                previous_value=previous_best,
                field=self.field,
                point=point,
                value=value,
            )
            self.best_point = point.copy()
            self.best_value = value
            self.local_best_point = point.copy()
            self.local_best_value = value
            self.box_best_point = point.copy()
            self.box_best_value = value
            self.best_field = self.field.copy()
            self._best_source = str(source)
        else:
            self._record_non_improvement_feedback()
        return True

    def _field_for_decoder_raw(self, raw: np.ndarray) -> np.ndarray:
        if self.config.decoder == "local_sparse":
            matrix = self._local_matrix_signal_from_decoded_delta(np.asarray(raw, dtype=float), normalize=False)
            return np.clip(matrix, self.smoothlife_config.field_floor, self.smoothlife_config.field_ceiling)
        field = self.projection.T @ np.asarray(raw, dtype=float)
        scale = max(1.0, float(np.max(np.abs(field))))
        return np.clip(field / scale, self.smoothlife_config.field_floor, self.smoothlife_config.field_ceiling).reshape(
            self.matrix_shape
        )

    def _evaluate_initial_matrix_design(self) -> None:
        if self._remaining() <= 0:
            return
        pulse_levels = (
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            -0.1,
            -0.05,
            -0.025,
            -0.01,
            -0.005,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
        )
        for level in pulse_levels:
            if self._remaining() <= 0:
                return
            self.field = self._field_for_decoder_raw(np.full(self.dimension, float(level), dtype=float))
            self._refresh_dynamics()
            if self._evaluate_current_field("matrix_pulse") and self._last_improved:
                self._apply_feedback(self.best_field, True)
                self._run_matrix_line_search()

        axis_count = min(self.dimension, 32)
        for level in (0.01, -0.01):
            for axis in range(axis_count):
                if self._remaining() <= 0:
                    return
                raw = np.zeros(self.dimension, dtype=float)
                raw[axis] = float(level)
                self.field = self._field_for_decoder_raw(raw)
                self._refresh_dynamics()
                if self._evaluate_current_field("matrix_axis") and self._last_improved:
                    self._apply_feedback(self.best_field, True)
                    self._run_matrix_line_search()

        random_count = min(self._remaining(), max(4, min(24, self.dimension)))
        for _ in range(int(random_count)):
            if self._remaining() <= 0:
                return
            self.field = self.rng.uniform(
                self.smoothlife_config.field_floor,
                self.smoothlife_config.field_ceiling,
                size=self.matrix_shape,
            )
            self._refresh_dynamics()
            if self._evaluate_current_field("matrix_random") and self._last_improved:
                self._apply_feedback(self.best_field, True)
                self._run_matrix_line_search()

    def _apply_feedback(self, previous_field: np.ndarray, improved: bool) -> None:
        movement = np.abs(self.field - previous_field)
        movement_max = max(float(np.max(movement)), 1e-12)
        signal = 0.65 * vitality(self.field) + 0.35 * np.clip(movement / movement_max, 0.0, 1.0)
        self.reward_field *= float(self.config.reward_decay)
        if improved:
            self.reward_field += float(self.config.reward_boost) * signal
            self._current_noise *= float(self.config.mutation_decay)
            return
        if np.isfinite(self.best_value):
            self.field = (
                (1.0 - float(self.config.elite_pull_strength)) * self.field
                + float(self.config.elite_pull_strength) * self.best_field
            )
        self.field -= float(self.config.failure_damping) * (self.field - previous_field)
        self.field = np.clip(self.field, self.smoothlife_config.field_floor, self.smoothlife_config.field_ceiling)

    def _run_matrix_line_search(self) -> None:
        if not self.config.matrix_line_search_enabled or self._last_successful_field_step is None:
            return
        if self._remaining() <= 0:
            return
        direction = self._last_successful_field_step.copy()
        for alpha in self.config.matrix_line_search_alphas:
            if self._remaining() <= 0:
                return
            previous_best_field = self.best_field.copy()
            candidate = np.clip(
                self.best_field + float(alpha) * direction,
                self.smoothlife_config.field_floor,
                self.smoothlife_config.field_ceiling,
            )
            if self._field_key(candidate) in self._field_values:
                continue
            self.field = candidate
            self._refresh_dynamics()
            added = self._evaluate_current_field("matrix_line_search")
            if added and self._last_improved:
                self._line_search_improvements += 1
                self._apply_feedback(previous_best_field, True)
                if self._last_successful_field_step is not None:
                    direction = self._last_successful_field_step.copy()
            elif added:
                self._apply_feedback(previous_best_field, False)
        if np.isfinite(self.best_value):
            self.field = self.best_field.copy()
            self._refresh_dynamics()

    def _selected_patch_centers(self) -> list[tuple[int, int]]:
        count = min(int(self.config.patch_probe_count), self._remaining())
        if count <= 0:
            return []
        change_pressure = np.abs(self.transition_field - self.field)
        score = (
            np.clip(self.temperature_field, 0.0, 3.0)
            + 0.5 * change_pressure
            + 0.25 * np.abs(self.local_credit_field)
        )
        flat_order = np.lexsort((np.arange(score.size), -score.ravel()))
        centers: list[tuple[int, int]] = []
        width = self.matrix_shape[1]
        for flat in flat_order:
            row = int(flat) // width
            col = int(flat) % width
            centers.append((row, col))
            if len(centers) >= count:
                break
        return centers

    def _patch_probe_candidate(self, row: int, col: int) -> np.ndarray:
        candidate = self.field.copy()
        step = float(self.config.patch_probe_step)
        if step <= 0.0:
            return candidate
        for index, (dy, dx) in enumerate(self._neighbor_offsets()):
            rr = int(row) + dy
            cc = int(col) + dx
            if rr < 0 or rr >= self.matrix_shape[0] or cc < 0 or cc >= self.matrix_shape[1]:
                continue
            direction = float(self.transition_field[rr, cc] - self.field[rr, cc])
            if abs(direction) <= 1e-12:
                direction = 1.0 if (row + col + index) % 2 == 0 else -1.0
            candidate[rr, cc] = np.clip(
                candidate[rr, cc] + step * np.sign(direction),
                self.smoothlife_config.field_floor,
                self.smoothlife_config.field_ceiling,
            )
        return candidate

    def _maybe_run_patch_probes(self) -> None:
        if not self.config.patch_probe_enabled or self._remaining() <= 0:
            return
        if len(self.archive) < self._next_patch_probe_evaluation:
            return
        centers = self._selected_patch_centers()
        if not centers:
            self._next_patch_probe_evaluation += int(self.config.patch_probe_interval_evaluations)
            return
        original_field = self.field.copy()
        original_transition = self.transition_field.copy()
        for row, col in centers:
            if self._remaining() <= 0:
                break
            previous_value = float(self.best_value)
            candidate = self._patch_probe_candidate(row, col)
            if self._field_key(candidate) in self._field_values:
                continue
            self.field = candidate
            self._refresh_dynamics()
            added = self._evaluate_current_field("patch_probe")
            if not added:
                continue
            self._patch_probe_count += 1
            value = float(self.archive[-1].value)
            self._apply_direct_patch_credit(row, col, previous_value, value)
            if self._last_improved:
                self._patch_probe_improvements += 1
                self._apply_feedback(original_field, True)
                original_field = self.best_field.copy()
                original_transition = self.transition_field.copy()
            else:
                self.field = original_field.copy()
                self.transition_field = original_transition.copy()
                self._refresh_dynamics()
        self._next_patch_probe_evaluation += int(self.config.patch_probe_interval_evaluations)

    def _initialize_noisy_field(self) -> None:
        self.field = initial_field(self.smoothlife_config, self.rng)
        if self.field.shape != self.matrix_shape:
            self.field = np.asarray(self.field, dtype=float).reshape(self.matrix_shape)

    def _evolve_once(self) -> None:
        previous = self.field.copy()
        self._refresh_dynamics()
        update = (
            (1.0 - float(self.smoothlife_config.dt)) * self.field
            + float(self.smoothlife_config.dt) * self.transition_field
            + float(self.smoothlife_config.diffusion) * laplacian(self.field)
        )
        if self._current_noise > 0.0:
            local_noise = float(self._current_noise) * np.clip(self.temperature_field, 0.05, 3.0)
            update += self.rng.normal(0.0, local_noise, size=self.matrix_shape)
        if not np.all(np.isfinite(update)):
            update = self.best_field.copy() if np.isfinite(self.best_value) else np.zeros(self.matrix_shape, dtype=float)
        self.field = np.clip(update, self.smoothlife_config.field_floor, self.smoothlife_config.field_ceiling)
        self.step_index += 1
        self._apply_feedback(previous, self._last_improved)
        self._refresh_dynamics()

    def _early_stop_reached(self) -> bool:
        if not self.config.early_stop_enabled or self.config.early_stop_value is None:
            return False
        if self.smoothlife_config.maximize:
            return float(self.best_value) >= float(self.config.early_stop_value)
        return float(self.best_value) <= float(self.config.early_stop_value)

    def _capture_snapshot(self, *, force: bool = False) -> None:
        if not self.config.store_all_snapshots and not force:
            return
        if not force and self.step_index % int(self.config.snapshot_interval) != 0:
            return
        snapshot = MatrixSmoothLifeSnapshot(
            step_index=int(self.step_index),
            bounds=self.display_bounds.copy(),
            field=self.field.copy(),
            inner_fill=self.inner_fill.copy(),
            outer_fill=self.outer_fill.copy(),
            objective_field=self._reward_support(),
            transition_field=self.transition_field.copy(),
            evaluated_mask=np.ones(self.matrix_shape, dtype=bool),
            best_point=self.best_point.copy(),
            best_value=float(self.best_value),
            local_best_point=self.local_best_point.copy(),
            local_best_value=float(self.local_best_value),
            box_best_point=self.box_best_point.copy(),
            box_best_value=float(self.box_best_value),
            metadata={
                "mode": "matrix",
                "evaluations": int(len(self.archive)),
                "archive_size": int(len(self.archive)),
                "matrix_shape": [int(self.matrix_shape[0]), int(self.matrix_shape[1])],
                "decoder": self.config.decoder,
                "local_sparse_block_size": int(self._local_block_size),
                "mutation_noise": float(self._current_noise),
                "advantage_norm": float(np.linalg.norm(self.advantage_field)),
                "direction_norm": float(np.linalg.norm(self.direction_field)),
                "local_credit_norm": float(np.linalg.norm(self.local_credit_field)),
                "local_credit_mean": float(np.mean(self.local_credit_field)),
                "temperature_mean": float(np.mean(self.temperature_field)),
                "temperature_max": float(np.max(self.temperature_field)),
                "stagnation_count": int(self._stagnation_evaluations),
                "line_search_improvements": int(self._line_search_improvements),
                "patch_probe_count": int(self._patch_probe_count),
                "patch_probe_improvements": int(self._patch_probe_improvements),
                "last_improved": bool(self._last_improved),
                "last_improvement_amount": float(self._last_improvement_amount),
            },
        )
        self.snapshots.append(snapshot)

    def run(self, evaluations: int | None = None) -> SearchRun:
        """Run until the evaluation budget, max step cap, or explicit target is reached."""

        limit = self.config.max_evaluations if evaluations is None else int(evaluations)
        if limit is None:
            raise ValueError("MatrixSmoothLifeSearch requires max_evaluations or run(evaluations=...)")
        self._active_max_evaluations = int(limit)
        self._stop_reason = "running"
        self._evaluate_current_field("neutral")
        self._capture_snapshot(force=True)
        if self._early_stop_reached():
            self._stop_reason = "early_stop_value"
        if self._remaining() > 0 and self._stop_reason == "running":
            self._evaluate_initial_matrix_design()
            self._capture_snapshot(force=True)
        if self._remaining() > 0 and self._stop_reason == "running":
            self._initialize_noisy_field()
            if np.isfinite(self.best_value):
                self.field = np.clip(
                    0.75 * self.best_field + 0.25 * self.field,
                    self.smoothlife_config.field_floor,
                    self.smoothlife_config.field_ceiling,
                )
            self._refresh_dynamics()
        while self._remaining() > 0 and self._stop_reason == "running":
            if self.config.max_steps is not None and self.step_index >= int(self.config.max_steps):
                self._stop_reason = "max_steps"
                break
            for _ in range(int(self.config.steps_per_evaluation)):
                self._evolve_once()
            added = self._evaluate_current_field("matrix")
            if added:
                self._apply_feedback(self.best_field if self._last_improved else self.field, self._last_improved)
                if self._last_improved:
                    self._run_matrix_line_search()
            self._maybe_run_patch_probes()
            self._refresh_dynamics()
            self._capture_snapshot()
            if self._early_stop_reached():
                self._stop_reason = "early_stop_value"
                break
        if self._stop_reason == "running":
            self._stop_reason = "budget_exhausted" if self._remaining() <= 0 else "stopped"
        if not self.snapshots:
            self._capture_snapshot(force=True)
        source_counts: dict[str, int] = {}
        for sample in self.archive:
            source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
        return SearchRun(
            best_point=self.best_point.copy(),
            best_value=float(self.best_value),
            evaluations=int(len(self.archive)),
            bounds=self.original_bounds.copy(),
            snapshots=list(self.snapshots),
            zoom_events=[],
            metadata={
                "mode": "matrix",
                "steps": int(self.step_index),
                "archive_size": int(len(self.archive)),
                "matrix_shape": [int(self.matrix_shape[0]), int(self.matrix_shape[1])],
                "decoder": self.config.decoder,
                "decoder_gain": float(self.config.decoder_gain),
                "local_sparse_block_size": int(self._local_block_size),
                "stop_reason": self._stop_reason,
                "mutation_noise": float(self._current_noise),
                "reward_max": float(np.max(self.reward_field)) if self.reward_field.size else 0.0,
                "best_source": self._best_source,
                "line_search_improvements": int(self._line_search_improvements),
                "patch_probe_count": int(self._patch_probe_count),
                "patch_probe_improvements": int(self._patch_probe_improvements),
                "advantage_norm": float(np.linalg.norm(self.advantage_field)),
                "direction_norm": float(np.linalg.norm(self.direction_field)),
                "local_credit_norm": float(np.linalg.norm(self.local_credit_field)),
                "local_credit_mean": float(np.mean(self.local_credit_field)),
                "temperature_mean": float(np.mean(self.temperature_field)),
                "temperature_max": float(np.max(self.temperature_field)),
                "stagnation_count": int(self._stagnation_evaluations),
                "source_counts": source_counts,
                "samples": [
                    {
                        "evaluation": int(sample.evaluation),
                        "step_index": int(sample.step_index),
                        "source": sample.source,
                        "value": float(sample.value),
                    }
                    for sample in self.archive
                ],
            },
        )
