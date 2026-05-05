from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig
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
    search = PointCloudSmoothLifeSearch(
        objective,
        [(-10.0, 10.0)] * dimension,
        SmoothLifeConfig(store_all_snapshots=False),
        PointCloudSearchConfig(
            max_evaluations=budget,
            batch_size=32,
            snapshot_interval_batches=999,
        ),
    )
    search.reset(seed=seed)
    return float(search.run().best_value)


class TestOptimizerQuality(unittest.TestCase):
    def test_transformed_objective_medians_improve_over_fixed_baselines(self) -> None:
        dimension = 12
        target = np.linspace(-2.7, 3.1, dimension)
        rotation = _orthogonal_matrix(dimension)
        # Fixed post-scout-fix medians for budget=600, before proposal metadata,
        # bounded source credit, surrogate preselection, and coherent high-D probes.
        cases = [
            ("shifted_sphere", _transformed_objective(_sphere, target), 5.0e-2),
            ("rotated_sphere", _transformed_objective(_sphere, target, rotation), 1.015630170421772e-1),
            ("shifted_ellipsoid", _transformed_objective(_ellipsoid, target), 4.026924605451464e1),
            ("rotated_ellipsoid", _transformed_objective(_ellipsoid, target, rotation), 6.263872304865816e1),
            ("shifted_ackley", _transformed_objective(ackley, target), 5.154713870860779),
            ("rotated_ackley", _transformed_objective(ackley, target, rotation), 5.741437610985212),
            ("shifted_rosenbrock", _transformed_objective(rosenbrock, target), 2.903412252300103e2),
        ]
        improved = 0
        medians: dict[str, float] = {}

        for name, objective, baseline in cases:
            with self.subTest(name=name):
                values = [
                    _run_objective(objective, dimension=dimension, seed=seed, budget=600)
                    for seed in (3, 7, 11)
                ]
                median = float(np.median(values))
                medians[name] = median
                self.assertLessEqual(median, baseline * 1.10)
                if median <= baseline * 0.85:
                    improved += 1

        self.assertGreaterEqual(improved, 5, medians)

    def test_high_dimensional_rosenbrock_targets_are_met_without_exact_shortcut(self) -> None:
        cases = [(30, 1.0), (50, 0.05), (100, 90.0), (500, 490.0)]
        for dimension, threshold in cases:
            with self.subTest(dimension=dimension):
                search = PointCloudSmoothLifeSearch(
                    rosenbrock,
                    [(-10.0, 10.0)] * dimension,
                    SmoothLifeConfig(store_all_snapshots=False),
                    PointCloudSearchConfig(
                        max_evaluations=6400,
                        batch_size=32,
                        snapshot_interval_batches=999,
                    ),
                )
                search.reset(seed=7)
                run = search.run()
                best_sample = min(search.archive.samples, key=lambda sample: sample.value)

                self.assertLess(run.best_value, threshold)
                self.assertGreater(run.best_value, 0.0)
                self.assertFalse(np.allclose(run.best_point, np.ones(dimension)))
                self.assertNotEqual(best_sample.source, "restart:scout")


if __name__ == "__main__":
    unittest.main()
