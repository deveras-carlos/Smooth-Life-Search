from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import AGSLSConfig, AdaptiveGridSmoothLifeSearch, SmoothLifeConfig, sphere


class TestThreePhaseIntegration(unittest.TestCase):
    def test_agsls_run_records_phase_metadata(self) -> None:
        smoothlife = SmoothLifeConfig(
            grid_shape=(32, 32),
            evaluations_per_step=4,
            snapshot_interval=1,
            preset="search",
            subpixel_confirm=False,
        )
        agsls = AGSLSConfig(
            max_evaluations=260,
            max_zoom_cycles=4,
            exploration_fraction=0.25,
            commit_fraction=0.60,
            exploration_steps_per_tick=2,
            commit_steps_per_zoom=2,
            exploitation_steps_per_zoom=1,
            min_basin_cells=4,
            cluster_min_samples=2,
        )
        search = AdaptiveGridSmoothLifeSearch(sphere, [(-5.0, 5.0), (-5.0, 5.0)], smoothlife, agsls)
        search.reset(seed=3)
        run = search.run()

        self.assertGreater(run.evaluations, 0)
        self.assertLessEqual(run.evaluations, agsls.max_evaluations)
        self.assertIn("exploration", run.metadata["phase_counts"])
        self.assertTrue(
            all(event.diagnostics["phase"] in {"commit", "exploitation"} for event in run.zoom_events)
        )
        self.assertFalse(
            any(event.diagnostics["phase"] == "exploration" for event in run.zoom_events),
            msg="exploration must never zoom",
        )

    def test_python_api_requires_agsls_budget(self) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(grid_shape=(32, 32), evaluations_per_step=4, preset="search"),
            AGSLSConfig(max_evaluations=None),
        )
        search.reset(seed=0)
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            search.run()


if __name__ == "__main__":
    unittest.main()
