from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig, sphere


class TestPointCloudSmoothLifeSearch(unittest.TestCase):
    def _search(self, seed: int = 0, budget: int = 120) -> PointCloudSmoothLifeSearch:
        search = PointCloudSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=budget,
                initial_design_size=16,
                batch_size=12,
                density_grid_shape=(24, 24),
                local_refinement_max_evaluations=48,
            ),
        )
        search.reset(seed=seed)
        return search

    def test_center_anchor_is_evaluated_first_and_hits_symmetric_origin(self) -> None:
        search = self._search(seed=4, budget=40)
        run = search.run()

        self.assertEqual(search.archive.samples[0].source, "center")
        self.assertEqual(run.best_value, 0.0)
        self.assertEqual(run.best_point.tolist(), [0.0, 0.0])

    def test_budget_is_never_exceeded(self) -> None:
        search = self._search(seed=3, budget=37)
        run = search.run()

        self.assertLessEqual(run.evaluations, 37)
        self.assertEqual(run.evaluations, len(search.archive))

    def test_seeded_runs_are_deterministic(self) -> None:
        first = self._search(seed=8, budget=90).run()
        second = self._search(seed=8, budget=90).run()

        self.assertEqual(first.best_value, second.best_value)
        self.assertTrue(np.allclose(first.best_point, second.best_point))
        self.assertEqual(first.metadata["archive_size"], second.metadata["archive_size"])

    def test_density_view_is_finite_and_portfolio_is_populated(self) -> None:
        search = self._search(seed=9, budget=120)
        run = search.run()
        snapshot = run.snapshots[-1]

        self.assertTrue(np.all(np.isfinite(snapshot.density_field)))
        self.assertTrue(np.all(np.isfinite(snapshot.objective_field)))
        self.assertGreaterEqual(len(run.metadata["portfolio"]), 1)
        self.assertGreaterEqual(run.metadata["archive_size"], len(run.metadata["portfolio"]))

    def test_region_radius_updates_are_recorded(self) -> None:
        search = self._search(seed=11, budget=160)
        run = search.run()
        events = run.metadata["region_events"]

        self.assertTrue(events)
        self.assertTrue(all("radius_before" in event and "radius_after" in event for event in events))

    def test_local_refinement_stall_does_not_stop_default_run(self) -> None:
        search = self._search(seed=4, budget=90)
        run = search.run()

        self.assertEqual(run.metadata["stop_reason"], "budget_exhausted")
        self.assertTrue(run.metadata["local_refinement_stalled"])
        self.assertEqual(run.evaluations, 90)

    def test_multiple_separated_regions_survive_dominant_incumbent(self) -> None:
        search = self._search(seed=12, budget=180)
        run = search.run()
        centers = np.asarray([region["center"] for region in run.metadata["portfolio"]], dtype=float)

        self.assertGreaterEqual(centers.shape[0], 2)
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        self.assertGreater(float(np.max(distances)), 1.0)

    def test_disabled_local_refinement_still_emits_stencil_candidates(self) -> None:
        search = PointCloudSmoothLifeSearch(
            lambda point: float((point[0] - 1.0) ** 2 + 2.0 * (point[1] - 1.0) ** 2),
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=120,
                initial_design_size=16,
                batch_size=12,
                density_grid_shape=(24, 24),
                local_refinement_enabled=False,
            ),
        )
        search.reset(seed=5)
        run = search.run()
        sources = {
            source
            for event in run.metadata["batch_events"]
            for source in event["source_counts"]
        }

        self.assertTrue(any(source.startswith("region:") and source.endswith(":stencil") for source in sources))
        self.assertIn("exploit_stencil", sources)

    def test_python_api_requires_budget(self) -> None:
        search = PointCloudSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(max_evaluations=None),
        )
        search.reset(seed=0)
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            search.run()


if __name__ == "__main__":
    unittest.main()
