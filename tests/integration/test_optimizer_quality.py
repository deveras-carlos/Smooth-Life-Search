from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import MatrixSmoothLifeConfig, MatrixSmoothLifeSearch, SmoothLifeConfig
from smooth_life_search.benchmark import ackley, rosenbrock


def _orthogonal_matrix(dimension: int, seed: int = 321) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _sphere(point: np.ndarray) -> float:
    z = np.asarray(point, dtype=float)
    return float(np.dot(z, z))


def _ellipsoid(point: np.ndarray) -> float:
    z = np.asarray(point, dtype=float)
    weights = np.geomspace(1.0, 100.0, z.size)
    return float(np.dot(weights, z * z))


def _transformed_objective(base, target: np.ndarray, rotation: np.ndarray | None = None):
    resolved_target = np.asarray(target, dtype=float)
    resolved_rotation = np.eye(resolved_target.size, dtype=float) if rotation is None else np.asarray(rotation, dtype=float)

    def objective(point: np.ndarray) -> float:
        z = np.asarray(point, dtype=float) - resolved_target
        return float(base(resolved_rotation.T @ z))

    return objective


def _run_objective(objective, *, dimension: int, seed: int, budget: int) -> float:
    search = MatrixSmoothLifeSearch(
        objective,
        [(-10.0, 10.0)] * dimension,
        SmoothLifeConfig(store_all_snapshots=False),
        MatrixSmoothLifeConfig(max_evaluations=budget, store_all_snapshots=False),
    )
    search.reset(seed=seed)
    return float(search.run().best_value)


class TestOptimizerQuality(unittest.TestCase):
    def test_transformed_objective_medians_stay_bounded_and_sometimes_improve(self) -> None:
        dimension = 12
        target = np.linspace(-2.7, 3.1, dimension)
        rotation = _orthogonal_matrix(dimension)
        cases = [
            ("shifted_sphere", _transformed_objective(_sphere, target)),
            ("rotated_sphere", _transformed_objective(_sphere, target, rotation)),
            ("shifted_ellipsoid", _transformed_objective(_ellipsoid, target)),
            ("rotated_ellipsoid", _transformed_objective(_ellipsoid, target, rotation)),
            ("shifted_ackley", _transformed_objective(ackley, target)),
            ("rotated_ackley", _transformed_objective(ackley, target, rotation)),
            ("shifted_rosenbrock", _transformed_objective(rosenbrock, target)),
        ]
        improved = 0
        medians: dict[str, float] = {}

        for name, objective in cases:
            with self.subTest(name=name):
                baseline = float(objective(np.zeros(dimension, dtype=float)))
                values = [
                    _run_objective(objective, dimension=dimension, seed=seed, budget=600)
                    for seed in (3, 7, 11)
                ]
                median = float(np.median(values))
                medians[name] = median
                self.assertLessEqual(median, baseline * 1.05)
                if median <= baseline * 0.95:
                    improved += 1

        self.assertGreaterEqual(improved, 3, medians)

    def test_high_dimensional_rosenbrock_improves_without_exact_shortcut(self) -> None:
        thresholds = {30: 28.9, 50: 48.9, 100: 98.5, 500: 496.0}
        for dimension in (30, 50, 100, 500):
            with self.subTest(dimension=dimension):
                search = MatrixSmoothLifeSearch(
                    rosenbrock,
                    [(-10.0, 10.0)] * dimension,
                    SmoothLifeConfig(store_all_snapshots=False),
                    MatrixSmoothLifeConfig(max_evaluations=1200, store_all_snapshots=False),
                )
                search.reset(seed=7)
                run = search.run()
                best_sample = min(search.archive, key=lambda sample: sample.value)

                self.assertLess(run.best_value, rosenbrock(np.zeros(dimension, dtype=float)))
                self.assertLess(run.best_value, thresholds[dimension])
                self.assertGreaterEqual(run.metadata["line_search_improvements"], 1)
                self.assertGreater(run.best_value, 0.0)
                self.assertFalse(np.allclose(run.best_point, np.ones(dimension)))
                self.assertNotIn(best_sample.source, {"restart:scout", "shade"})
                self.assertNotIn(":cma", best_sample.source)


if __name__ == "__main__":
    unittest.main()
