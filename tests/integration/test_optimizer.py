from __future__ import annotations

import unittest

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig, sphere


class TestPointCloudIntegration(unittest.TestCase):
    def test_point_cloud_run_records_archive_and_batch_metadata(self) -> None:
        smoothlife = SmoothLifeConfig(
            grid_shape=(32, 32),
            evaluations_per_step=4,
            snapshot_interval=1,
            preset="search",
            subpixel_confirm=False,
        )
        config = PointCloudSearchConfig(
            max_evaluations=180,
            initial_design_size=16,
            batch_size=16,
            density_grid_shape=(24, 24),
            portfolio_size=3,
            local_refinement_max_evaluations=48,
        )
        search = PointCloudSmoothLifeSearch(sphere, [(-5.0, 5.0), (-5.0, 5.0)], smoothlife, config)
        search.reset(seed=3)
        run = search.run()

        self.assertGreater(run.evaluations, 0)
        self.assertLessEqual(run.evaluations, config.max_evaluations)
        self.assertEqual(run.metadata["mode"], "point-cloud")
        self.assertEqual(run.metadata["archive_size"], run.evaluations)
        self.assertTrue(run.metadata["batch_events"])
        self.assertEqual(run.zoom_events, [])

    def test_python_api_requires_point_cloud_budget(self) -> None:
        search = PointCloudSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(grid_shape=(32, 32), evaluations_per_step=4, preset="search"),
            PointCloudSearchConfig(max_evaluations=None),
        )
        search.reset(seed=0)
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            search.run()


if __name__ == "__main__":
    unittest.main()
