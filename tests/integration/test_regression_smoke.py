from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.benchmark import rosenbrock
from smooth_life_search.input.cli import build_parser, run_matrix_command


def _run_cli_case(
    objective: str,
    budget: int,
    extra_args: list[str] | None = None,
    *,
    seed: int = 7,
    dimension: int = 2,
) -> dict:
    parser = build_parser()
    args = parser.parse_args(
        [
            "matrix",
            "--objective",
            objective,
            "--dimension",
            str(dimension),
            "--seed",
            str(seed),
            "--budget",
            str(budget),
            "--no-store-all-snapshots",
            "--json",
            *(extra_args or []),
        ]
    )
    return run_matrix_command(args)


class TestMatrixRegressionSmoke(unittest.TestCase):
    def test_symmetric_origin_objectives_hit_center_anchor(self) -> None:
        for objective in ("ackley", "sphere", "rastrigin", "griewank"):
            with self.subTest(objective=objective):
                payload = _run_cli_case(objective, 64, ["--target-value", "0.0"], dimension=5)
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0] * 5)
                self.assertEqual(payload["evaluations"], 1)
                self.assertEqual(payload["stop_reason"], "early_stop_value")

    def test_shifted_sphere_and_ackley_improve_over_neutral_center(self) -> None:
        center = np.asarray([3.0] * 5, dtype=float)
        sphere_center = float(np.dot(center, center))
        for objective, center_value in (("sphere", sphere_center), ("ackley", 9.0)):
            with self.subTest(objective=objective):
                payload = _run_cli_case(
                    objective,
                    1000,
                    ["--lower", "-7", "--upper", "13"],
                    dimension=5,
                )
                self.assertLess(payload["best_value"], center_value)
                self.assertEqual(payload["evaluations"], 1000)

    def test_matrix_runs_are_deterministic_by_seed(self) -> None:
        first = _run_cli_case("ackley", 300, ["--lower", "-7", "--upper", "13"], seed=11, dimension=5)
        second = _run_cli_case("ackley", 300, ["--lower", "-7", "--upper", "13"], seed=11, dimension=5)

        self.assertEqual(first["best_value"], second["best_value"])
        self.assertEqual(first["best_point"], second["best_point"])

    def test_rosenbrock_10d_and_30d_improve_below_center_baseline(self) -> None:
        thresholds = {10: 8.95, 30: 28.9}
        for dimension in (10, 30):
            with self.subTest(dimension=dimension):
                payload = _run_cli_case("rosenbrock", 1200, dimension=dimension)
                self.assertLess(payload["best_value"], rosenbrock(np.zeros(dimension, dtype=float)))
                self.assertLess(payload["best_value"], thresholds[dimension])
                self.assertGreaterEqual(payload["line_search_improvements"], 1)
                self.assertGreater(payload["best_value"], 0.0)
                self.assertNotEqual(payload["best_point"], [1.0] * dimension)

    def test_rosenbrock_100d_credit_steering_improves_below_center(self) -> None:
        payload = _run_cli_case("rosenbrock", 1200, dimension=100)

        self.assertLess(payload["best_value"], rosenbrock(np.zeros(100, dtype=float)))
        self.assertLess(payload["best_value"], 98.5)
        self.assertGreaterEqual(payload["line_search_improvements"], 1)
        self.assertGreater(payload["advantage_norm"], 0.0)
        self.assertGreater(payload["direction_norm"], 0.0)

    def test_rosenbrock_500d_remains_bounded_and_below_center_baseline(self) -> None:
        dimension = 500
        payload = _run_cli_case("rosenbrock", 1200, dimension=dimension)

        self.assertLess(payload["best_value"], rosenbrock(np.zeros(dimension, dtype=float)))
        self.assertGreater(payload["best_value"], 0.0)
        self.assertNotEqual(payload["best_point"], [1.0] * dimension)


if __name__ == "__main__":
    unittest.main()
