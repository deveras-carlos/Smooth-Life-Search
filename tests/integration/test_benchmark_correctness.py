from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import MatrixSmoothLifeConfig, MatrixSmoothLifeSearch, SmoothLifeConfig
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


def _run_case(objective, *, dimension: int, seed: int, budget: int = 600):
    search = MatrixSmoothLifeSearch(
        objective,
        [(-10.0, 10.0)] * dimension,
        SmoothLifeConfig(store_all_snapshots=False),
        MatrixSmoothLifeConfig(max_evaluations=budget, store_all_snapshots=False),
    )
    search.reset(seed=seed)
    run = search.run()
    best_sample = min(search.archive, key=lambda sample: sample.value)
    return run, search, best_sample


class TestBenchmarkCorrectness(unittest.TestCase):
    def test_adversarial_transforms_improve_without_forbidden_provenance(self) -> None:
        dimension = 12
        target = np.linspace(-2.7, 3.1, dimension)
        rotation = _orthogonal_matrix(dimension)
        cases = [
            ("shifted_sphere", _transformed_objective(lambda z: float(np.dot(z, z)), target)),
            ("rotated_ellipsoid", _transformed_objective(_ellipsoid, target, rotation)),
            ("shifted_ackley", _transformed_objective(ackley, target)),
            ("shifted_rosenbrock", _transformed_objective(rosenbrock, target)),
        ]
        materially_improved = 0

        for name, objective in cases:
            with self.subTest(name=name):
                baseline = float(objective(np.zeros(dimension, dtype=float)))
                values: list[float] = []
                best_samples = []
                for seed in (3, 7, 11):
                    run, _search, best_sample = _run_case(objective, dimension=dimension, seed=seed)
                    values.append(float(run.best_value))
                    best_samples.append(best_sample)
                    if abs(float(best_sample.value)) <= 1e-14:
                        self.assertNotEqual(best_sample.source, "neutral")

                median = float(np.median(values))
                self.assertLessEqual(median, baseline * 1.01)
                if median <= baseline * 0.95:
                    materially_improved += 1

        self.assertGreaterEqual(materially_improved, 2)

    def test_no_point_cloud_evolutionary_shortcut_sources_exist_on_high_d_rosenbrock(self) -> None:
        dimension = 30
        run, search, best_sample = _run_case(rosenbrock, dimension=dimension, seed=7, budget=1200)
        sources = {sample.source for sample in search.archive}

        self.assertLess(run.best_value, rosenbrock(np.zeros(dimension, dtype=float)))
        self.assertGreater(run.best_value, 0.0)
        self.assertFalse(np.allclose(run.best_point, np.ones(dimension)))
        self.assertFalse(best_sample.source == "restart:scout" and abs(best_sample.value) <= 1e-14)
        self.assertFalse(any(source == "restart:scout" or source == "shade" or ":cma" in source for source in sources))


if __name__ == "__main__":
    unittest.main()
