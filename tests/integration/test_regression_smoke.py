from __future__ import annotations

import unittest

from smooth_life_search.input.cli import build_parser, run_agsls_command


def _run_cli_case(objective: str, budget: int, extra_args: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(
        [
            "agsls",
            "--objective",
            objective,
            "--dimension",
            "2",
            "--seed",
            "7",
            "--budget",
            str(budget),
            "--zoom-cycles",
            "20",
            "--json",
            *(extra_args or []),
        ]
    )
    return run_agsls_command(args)


class TestAGSLSRegressionSmoke(unittest.TestCase):
    def test_rosenbrock_seed7_6400_uses_surrogate_and_reaches_target(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400)
        surrogate_commit_zooms = [
            event
            for event in payload["zoom_events"]
            if event["diagnostics"].get("phase") == "commit" and event["diagnostics"].get("surrogate_used")
        ]

        self.assertLess(payload["best_value"], 1e-2)
        self.assertTrue(surrogate_commit_zooms)

    def test_ackley_seed7_12800_still_reaches_origin(self) -> None:
        payload = _run_cli_case("ackley", 12800)

        self.assertEqual(payload["best_value"], 0.0)
        self.assertEqual(payload["best_point"], [0.0, 0.0])

    def test_trust_region_rosenbrock_seed7_30000_reaches_precision_target(self) -> None:
        payload = _run_cli_case(
            "rosenbrock",
            30000,
            [
                "--commit-fraction",
                "0.9",
                "--commit-steps",
                "96",
                "--exploitation-steps",
                "32",
                "--exploitation-shrink-fraction",
                "0.0001",
            ],
        )

        self.assertLess(payload["best_value"], 1e-11)
        self.assertTrue(payload["trust_region_events"])
        self.assertGreater(
            sum(int(event.get("trust_region_improvements", 0)) for event in payload["trust_region_events"]),
            0,
        )

    def test_no_trust_region_restores_zoom_first_path(self) -> None:
        payload = _run_cli_case("rosenbrock", 6400, ["--no-trust-region"])

        self.assertLess(payload["best_value"], 1e-2)
        self.assertEqual(payload["trust_region_events"], [])

    def test_trust_region_ackley_sphere_and_rastrigin_seed7_30000_reach_origin(self) -> None:
        extra = [
            "--commit-fraction",
            "0.9",
            "--commit-steps",
            "96",
            "--exploitation-steps",
            "32",
            "--exploitation-shrink-fraction",
            "0.0001",
        ]

        for objective in ("ackley", "sphere", "rastrigin"):
            with self.subTest(objective=objective):
                payload = _run_cli_case(objective, 30000, extra)
                self.assertEqual(payload["best_value"], 0.0)
                self.assertEqual(payload["best_point"], [0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
