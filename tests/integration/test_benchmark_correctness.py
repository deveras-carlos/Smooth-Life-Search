from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig
from smooth_life_search.benchmark import ackley, rosenbrock


def _orthogonal_matrix(dimension: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _transformed_objective(base, target: np.ndarray, rotation: np.ndarray | None = None):
    resolved_target = np.asarray(target, dtype=float)
    resolved_rotation = np.eye(resolved_target.size, dtype=float) if rotation is None else np.asarray(rotation, dtype=float)

    def objective(point: np.ndarray) -> float:
        z = np.asarray(point, dtype=float) - resolved_target
        return float(base(resolved_rotation.T @ z))

    return objective


def _ellipsoid(point: np.ndarray) -> float:
    z = np.asarray(point, dtype=float)
    weights = np.linspace(1.0, 50.0, z.size)
    return float(np.dot(weights, z * z))


def _anchor_baseline(objective, bounds: list[tuple[float, float]]) -> float:
    search_bounds = np.asarray(bounds, dtype=float)
    dimension = int(search_bounds.shape[0])
    center = np.mean(search_bounds, axis=1)
    anchors = [center, np.zeros(dimension, dtype=float), search_bounds[:, 0], search_bounds[:, 1]]
    for axis in range(dimension):
        low = center.copy()
        high = center.copy()
        low[axis] = search_bounds[axis, 0]
        high[axis] = search_bounds[axis, 1]
        anchors.extend([low, high])
    return float(min(objective(point) for point in anchors))


def _run_case(objective, *, dimension: int, seed: int, budget: int = 1600):
    search = PointCloudSmoothLifeSearch(
        objective,
        [(-10.0, 10.0)] * dimension,
        SmoothLifeConfig(store_all_snapshots=False),
        PointCloudSearchConfig(max_evaluations=budget, batch_size=32),
    )
    search.reset(seed=seed)
    run = search.run()
    best_sample = min(search.archive.samples, key=lambda sample: sample.value)
    return run, search, best_sample


class TestBenchmarkCorrectness(unittest.TestCase):
    def test_adversarial_transforms_improve_without_anchor_or_removed_source_provenance(self) -> None:
        dimension = 12
        target = np.linspace(-2.7, 3.1, dimension)
        rotation = _orthogonal_matrix(dimension)
        cases = [
            (
                "shifted_sphere",
                _transformed_objective(lambda z: float(np.dot(z, z)), target),
                0.10,
            ),
            (
                "rotated_ellipsoid",
                _transformed_objective(_ellipsoid, target, rotation),
                0.25,
            ),
            (
                "shifted_ackley",
                _transformed_objective(ackley, target),
                0.75,
            ),
            (
                "shifted_rosenbrock",
                _transformed_objective(rosenbrock, target),
                0.25,
            ),
        ]
        forbidden_exact_sources = {"center", "anchor", "restart:scout", "shade"}

        for name, objective, improvement_ratio in cases:
            with self.subTest(name=name):
                baseline = _anchor_baseline(objective, [(-10.0, 10.0)] * dimension)
                values: list[float] = []
                best_samples = []
                for seed in (3, 7, 11):
                    run, _search, best_sample = _run_case(objective, dimension=dimension, seed=seed)
                    values.append(float(run.best_value))
                    best_samples.append(best_sample)
                    if abs(float(best_sample.value)) <= 1e-14:
                        self.assertNotIn(best_sample.source, forbidden_exact_sources)

                self.assertLess(float(np.median(values)), baseline * improvement_ratio)

    def test_removed_evolutionary_sources_are_absent_on_high_d_rosenbrock(self) -> None:
        dimension = 30
        center_baseline = rosenbrock(np.zeros(dimension, dtype=float))
        search = PointCloudSmoothLifeSearch(
            rosenbrock,
            [(-10.0, 10.0)] * dimension,
            SmoothLifeConfig(store_all_snapshots=False),
            PointCloudSearchConfig(max_evaluations=1200, batch_size=32),
        )
        search.reset(seed=7)
        run = search.run()
        best_sample = min(search.archive.samples, key=lambda sample: sample.value)
        sources = {sample.source for sample in search.archive.samples}

        self.assertLess(run.best_value, center_baseline)
        self.assertGreater(run.best_value, 0.0)
        self.assertFalse(np.allclose(run.best_point, np.ones(dimension)))
        self.assertFalse(best_sample.source == "restart:scout" and abs(best_sample.value) <= 1e-14)
        self.assertFalse(any(source == "restart:scout" or source == "shade" or ":cma" in source for source in sources))


if __name__ == "__main__":
    unittest.main()
