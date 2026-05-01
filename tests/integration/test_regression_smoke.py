from __future__ import annotations

import unittest

from smooth_life_search.input.cli import build_parser, run_point_cloud_command


def _run_cli_case(objective: str, budget: int, extra_args: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(
        [
            "point-cloud",
            "--objective",
            objective,
            "--dimension",
            "2",
            "--seed",
            "7",
            "--budget",
            str(budget),
            "--json",
            *(extra_args or []),
        ]
    )
    return run_point_cloud_command(args)


class TestPointCloudRegressionSmoke(unittest.TestCase):
    def test_rosenbrock_seed7_6400_reaches_target(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400)

        self.assertLess(payload["best_value"], 1e-2)
        self.assertTrue(payload["batch_events"])
        self.assertTrue(payload["trust_region_events"])

    def test_ackley_seed7_12800_still_reaches_origin(self) -> None:
        payload = _run_cli_case("ackley", 12800)

        self.assertEqual(payload["best_value"], 0.0)
        self.assertEqual(payload["best_point"], [0.0, 0.0])

    def test_rosenbrock_seed7_30000_reaches_precision_target(self) -> None:
        payload = _run_cli_case("rosenbrock", 30000)

        self.assertLess(payload["best_value"], 1e-11)
        self.assertTrue(payload["trust_region_events"])
        self.assertGreater(
            sum(int(event.get("trust_region_improvements", 0)) for event in payload["trust_region_events"]),
            0,
        )

    def test_no_trust_regions_disables_region_event_path(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, ["--no-trust-regions"])

        self.assertLess(payload["best_value"], 1e-2)
        self.assertEqual(payload["region_events"], [])

    def test_ackley_sphere_and_rastrigin_seed7_30000_reach_origin(self) -> None:
        for objective in ("ackley", "sphere", "rastrigin"):
            with self.subTest(objective=objective):
                payload = _run_cli_case(objective, 30000)
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
