from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import Basin
from smooth_life_search.agsls.surrogate import fit_commit_surrogate, surrogate_valley_tangent
from smooth_life_search.smoothlife.evaluation import evaluation_points


def _values(shape: tuple[int, int], bounds: np.ndarray, fn) -> np.ndarray:
    rows, cols = np.indices(shape)
    points = evaluation_points(rows.ravel(), cols.ravel(), bounds, shape)
    return np.asarray([fn(point) for point in points], dtype=float).reshape(shape)


def _basin(shape: tuple[int, int], bounds: np.ndarray, values: np.ndarray, *, maximize: bool = False) -> Basin:
    mask = np.ones(shape, dtype=bool)
    flat = int(np.argmax(values) if maximize else np.argmin(values))
    row, col = np.unravel_index(flat, shape)
    best_point = evaluation_points(
        np.asarray([row], dtype=int),
        np.asarray([col], dtype=int),
        bounds,
        shape,
    )[0]
    return Basin(
        mask=mask,
        centroid_grid=np.asarray([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0], dtype=float),
        centroid_world=np.mean(bounds, axis=1),
        bbox_grid=(0, 0, shape[0] - 1, shape[1] - 1),
        bbox_world=bounds.copy(),
        support_mass=float(np.count_nonzero(mask)),
        objective_score=1.0,
        stability_score=1.0,
        alive_density=1.0,
        basin_best_point=best_point,
        basin_best_value=float(values[row, col]),
        evaluated_count=int(values.size),
        best_objective_score=1.0,
        mean_objective_score=0.5,
    )


class TestCommitSurrogate(unittest.TestCase):
    def _fit(
        self,
        values: np.ndarray,
        bounds: np.ndarray,
        *,
        maximize: bool = False,
        evaluated_mask: np.ndarray | None = None,
    ):
        shape = values.shape
        evaluated = np.ones(shape, dtype=bool) if evaluated_mask is None else evaluated_mask
        basin = _basin(shape, bounds, values, maximize=maximize)
        return fit_commit_surrogate(
            objective_values=values,
            evaluated_mask=evaluated,
            support_field=np.ones(shape, dtype=float),
            basin=basin,
            bounds=bounds,
            grid_shape=shape,
            maximize=maximize,
            min_samples=12,
            max_samples=200,
            regularization=1e-10,
            min_predicted_improvement=0.0,
            max_condition=1e10,
            support_weight=0.0,
        )

    def test_recovers_synthetic_quadratic_minimizer(self) -> None:
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        values = _values(bounds=bounds, shape=(11, 11), fn=lambda point: (point[0] - 0.31) ** 2 + 2.0 * (point[1] - 0.68) ** 2)

        result = self._fit(values, bounds)

        self.assertTrue(result.accepted, msg=result.reason)
        self.assertTrue(np.allclose(result.point, np.asarray([0.31, 0.68]), atol=1e-2))
        self.assertTrue(np.isfinite(result.predicted_value))
        self.assertTrue(np.isfinite(result.condition))

    def test_rejects_insufficient_samples(self) -> None:
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        values = _values(bounds=bounds, shape=(8, 8), fn=lambda point: np.sum(point * point))
        evaluated = np.zeros(values.shape, dtype=bool)
        evaluated[:2, :2] = True

        result = self._fit(values, bounds, evaluated_mask=evaluated)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "insufficient_samples")
        self.assertEqual(result.sample_count, 4)

    def test_rejects_indefinite_quadratic(self) -> None:
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        values = _values(bounds=bounds, shape=(11, 11), fn=lambda point: (point[0] - 0.5) ** 2 - (point[1] - 0.5) ** 2)

        result = self._fit(values, bounds)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "nonconvex_surrogate")

    def test_maximization_uses_sign_converted_fit(self) -> None:
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        values = _values(bounds=bounds, shape=(11, 11), fn=lambda point: -((point[0] - 0.40) ** 2 + 2.0 * (point[1] - 0.60) ** 2))

        result = self._fit(values, bounds, maximize=True)

        self.assertTrue(result.accepted, msg=result.reason)
        self.assertTrue(np.allclose(result.point, np.asarray([0.40, 0.60]), atol=1e-2))

    def test_valley_tangent_uses_low_curvature_direction(self) -> None:
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        values = _values(bounds=bounds, shape=(15, 15), fn=lambda point: 0.05 * (point[0] - 0.40) ** 2 + 2.0 * (point[1] - 0.60) ** 2)

        result = self._fit(values, bounds)
        tangent = surrogate_valley_tangent(result)

        self.assertTrue(result.accepted, msg=result.reason)
        self.assertIsNotNone(tangent)
        self.assertGreater(abs(float(tangent[0])), 0.95)
        self.assertLess(abs(float(tangent[1])), 0.10)


if __name__ == "__main__":
    unittest.main()
