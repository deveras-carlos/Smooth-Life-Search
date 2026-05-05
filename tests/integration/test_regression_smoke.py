from __future__ import annotations

import unittest
import numpy as np

from smooth_life_search.input.cli import build_parser, run_point_cloud_command


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
            "point-cloud",
            "--objective",
            objective,
            "--dimension",
            str(dimension),
            "--seed",
            str(seed),
            "--budget",
            str(budget),
            "--json",
            *(extra_args or []),
        ]
    )
    return run_point_cloud_command(args)


class TestPointCloudRegressionSmoke(unittest.TestCase):
    def test_rosenbrock_seed7_6400_reaches_target(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, ["--early-stop-enabled", "--early-stop-value", "1e-2"])

        self.assertLess(payload["best_value"], 1e-2)
        self.assertTrue(payload["batch_events"])
        self.assertTrue(payload["trust_region_events"])

    def test_ackley_seed7_12800_still_reaches_origin(self) -> None:
        payload = _run_cli_case("ackley", 12800, ["--early-stop-enabled", "--early-stop-value", "0.0"])

        self.assertEqual(payload["best_value"], 0.0)
        self.assertEqual(payload["best_point"], [0.0, 0.0])

    def test_rosenbrock_seed7_target_value_stops_after_local_refinement(self) -> None:
        payload = _run_cli_case("rosenbrock", 20000, ["--target-value", "1e-11"])

        self.assertLess(payload["best_value"], 1e-11)
        self.assertLess(payload["evaluations"], 200)
        self.assertEqual(payload["stop_reason"], "early_stop_value")
        self.assertAlmostEqual(payload["target_value"], 1e-11)
        self.assertTrue(payload["trust_region_events"])
        self.assertGreater(
            sum(int(event.get("trust_region_improvements", 0)) for event in payload["trust_region_events"]),
            0,
        )

    def test_rosenbrock_seed7_target_value_stops_cloud_only_run(self) -> None:
        payload = _run_cli_case("rosenbrock", 20000, ["--no-local-refinement", "--target-value", "1e-11"])

        self.assertLess(payload["best_value"], 1e-11)
        self.assertLess(payload["evaluations"], 8000)
        self.assertEqual(payload["stop_reason"], "early_stop_value")
        self.assertAlmostEqual(payload["target_value"], 1e-11)

    def test_rosenbrock_seed7_default_run_exhausts_budget(self) -> None:
        payload = _run_cli_case("rosenbrock", 20000)

        self.assertLess(payload["best_value"], 1e-11)
        self.assertEqual(payload["evaluations"], 20000)
        self.assertEqual(payload["stop_reason"], "budget_exhausted")
        self.assertIsNone(payload["target_value"])

    def test_no_trust_regions_disables_region_event_path(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, ["--no-trust-regions", "--early-stop-enabled", "--early-stop-value", "1e-2"])

        self.assertLess(payload["best_value"], 1e-2)
        self.assertEqual(payload["region_events"], [])

    def test_ackley_sphere_and_rastrigin_seed7_30000_reach_origin(self) -> None:
        for objective in ("ackley", "sphere", "rastrigin"):
            with self.subTest(objective=objective):
                payload = _run_cli_case(objective, 30000, ["--early-stop-enabled", "--early-stop-value", "0.0"])
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0, 0.0])

    def test_shifted_ackley_seed0_20000_finds_global_basin(self) -> None:
        payload = _run_cli_case(
            "ackley",
            20000,
            ["--lower", "-7", "--upper", "13"],
            seed=0,
        )

        self.assertLess(payload["best_value"], 1e-2)
        self.assertLess(float(np.linalg.norm(np.asarray(payload["best_point"], dtype=float))), 0.05)

    def test_rosenbrock_seed0_20000_without_local_refinement_uses_cloud_stencils(self) -> None:
        payload = _run_cli_case("rosenbrock", 20000, ["--no-local-refinement"], seed=0)

        self.assertLess(payload["best_value"], 1e-4)
        self.assertTrue(
            any(
                "exploit_stencil" in event.get("source_improvements", {})
                for event in payload["batch_events"]
            )
        )

    def test_sphere_5d_and_10d_reach_origin(self) -> None:
        for dimension in (5, 10):
            with self.subTest(dimension=dimension):
                payload = _run_cli_case(
                    "sphere",
                    5000,
                    ["--target-value", "0.0"],
                    dimension=dimension,
                )
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0] * dimension)

    def test_ackley_and_rastrigin_5d_reach_origin(self) -> None:
        for objective in ("ackley", "rastrigin"):
            with self.subTest(objective=objective):
                payload = _run_cli_case(
                    objective,
                    5000,
                    ["--target-value", "0.0"],
                    dimension=5,
                )
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0] * 5)

    def test_rosenbrock_5d_reaches_practical_target(self) -> None:
        payload = _run_cli_case(
            "rosenbrock",
            20000,
            ["--target-value", "1e-6"],
            dimension=5,
        )

        self.assertLess(payload["best_value"], 1e-6)
        self.assertEqual(payload["stop_reason"], "early_stop_value")

    def test_rosenbrock_10d_seed7_6400_remains_solved(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, dimension=10)

        self.assertLess(payload["best_value"], 1e-10)

    def test_rosenbrock_30d_and_50d_seed7_6400_improve_without_scout_shortcut(self) -> None:
        cases = [(30, 1.0), (50, 0.05), (100, 1.0)]
        for dimension, threshold in cases:
            with self.subTest(dimension=dimension):
                payload = _run_cli_case("rosenbrock", 6400, dimension=dimension)
                sources = {
                    source
                    for event in payload["batch_events"]
                    for source in event.get("source_counts", {})
                }

                self.assertLess(payload["best_value"], threshold)
                self.assertGreater(payload["best_value"], 0.0)
                self.assertFalse(np.allclose(payload["best_point"], [1.0] * dimension))
                self.assertIn("shade", sources)
                self.assertIn("restart:scout", sources)
                self.assertTrue(any(":cma" in source for source in sources))

    def test_rosenbrock_500d_seed7_6400_beats_center_baseline(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, dimension=500)

        self.assertLess(payload["best_value"], 450.0)
        self.assertGreater(payload["best_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
