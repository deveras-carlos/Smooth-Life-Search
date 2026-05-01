from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import AGSLSConfig, AdaptiveGridSmoothLifeSearch, Basin, SmoothLifeConfig, sphere
from smooth_life_search.agsls.scoring import score_basins
from smooth_life_search.smoothlife.evaluation import evaluation_points


def _basin(
    *,
    support_mass: float,
    alive_density: float,
    objective_score: float,
    bbox: np.ndarray | None = None,
) -> Basin:
    mask = np.ones((4, 4), dtype=bool)
    bbox_world = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float) if bbox is None else bbox
    return Basin(
        mask=mask,
        centroid_grid=np.asarray([1.5, 1.5], dtype=float),
        centroid_world=np.asarray(
            [0.5 * (bbox_world[0, 0] + bbox_world[0, 1]), 0.5 * (bbox_world[1, 0] + bbox_world[1, 1])],
            dtype=float,
        ),
        bbox_grid=(0, 0, 3, 3),
        bbox_world=bbox_world,
        support_mass=support_mass,
        objective_score=objective_score,
        stability_score=0.5,
        alive_density=alive_density,
        basin_best_point=np.mean(bbox_world, axis=1),
        basin_best_value=objective_score,
        evaluated_count=4,
        best_objective_score=objective_score,
        mean_objective_score=objective_score,
    )


def _matching_grid_basin(
    shape: tuple[int, int],
    bounds: np.ndarray,
    values: np.ndarray,
    bbox_grid: tuple[int, int, int, int] | None = None,
) -> Basin:
    if bbox_grid is None:
        bbox_grid = (0, 0, shape[0] - 1, shape[1] - 1)
    row_min, col_min, row_max, col_max = bbox_grid
    mask = np.zeros(shape, dtype=bool)
    mask[row_min : row_max + 1, col_min : col_max + 1] = True
    masked_values = np.where(mask, values, np.inf)
    flat = int(np.argmin(masked_values))
    row, col = np.unravel_index(flat, shape)
    mask_rows, mask_cols = np.nonzero(mask)
    mask_points = evaluation_points(mask_rows, mask_cols, bounds, shape)
    best_point = evaluation_points(
        np.asarray([row], dtype=int),
        np.asarray([col], dtype=int),
        bounds,
        shape,
    )[0]
    height, width = shape
    bbox_world = np.asarray(
        [
            [
                bounds[0, 0] + (col_min / width) * (bounds[0, 1] - bounds[0, 0]),
                bounds[0, 0] + ((col_max + 1) / width) * (bounds[0, 1] - bounds[0, 0]),
            ],
            [
                bounds[1, 0] + (row_min / height) * (bounds[1, 1] - bounds[1, 0]),
                bounds[1, 0] + ((row_max + 1) / height) * (bounds[1, 1] - bounds[1, 0]),
            ],
        ],
        dtype=float,
    )
    return Basin(
        mask=mask,
        centroid_grid=np.asarray([np.mean(mask_rows), np.mean(mask_cols)], dtype=float),
        centroid_world=np.mean(mask_points, axis=0),
        bbox_grid=bbox_grid,
        bbox_world=bbox_world,
        support_mass=float(np.count_nonzero(mask)),
        objective_score=1.0,
        stability_score=1.0,
        alive_density=1.0,
        basin_best_point=best_point,
        basin_best_value=float(values[row, col]),
        evaluated_count=int(values.size),
        best_objective_score=1.0,
        mean_objective_score=0.5,
    )


class TestThreePhaseAGSLS(unittest.TestCase):
    def _search(self, config: AGSLSConfig | None = None) -> AdaptiveGridSmoothLifeSearch:
        smoothlife = SmoothLifeConfig(
            grid_shape=(32, 32),
            evaluations_per_step=4,
            snapshot_interval=1,
            preset="search",
            subpixel_confirm=False,
        )
        return AdaptiveGridSmoothLifeSearch(
            sphere,
            [(-8.0, 8.0), (-8.0, 8.0)],
            smoothlife,
            config or AGSLSConfig(max_evaluations=240, min_basin_cells=4, cluster_min_samples=2),
        )

    def _valley_tracking_fixture(
        self,
        *,
        config: AGSLSConfig | None = None,
        start: np.ndarray | None = None,
    ) -> tuple[AdaptiveGridSmoothLifeSearch, Basin]:
        def objective(point: np.ndarray) -> float:
            return float(0.05 * (point[0] - 0.40) ** 2 + 2.0 * (point[1] - 0.50) ** 2)

        resolved_config = config or AGSLSConfig(
            max_evaluations=240,
            min_basin_cells=4,
            cluster_min_samples=2,
            exploitation_valley_probe_evaluations=6,
            exploitation_valley_step_fraction=0.15,
        )
        smoothlife = SmoothLifeConfig(
            grid_shape=(32, 32),
            evaluations_per_step=4,
            snapshot_interval=1,
            preset="search",
            subpixel_confirm=False,
        )
        search = AdaptiveGridSmoothLifeSearch(
            objective,
            [(0.0, 1.0), (0.0, 1.0)],
            smoothlife,
            resolved_config,
        )
        search.reset(seed=0)
        state = search.engine.state
        bounds = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        shape = search.engine.config.grid_shape
        rows, cols = np.indices(shape)
        points = evaluation_points(rows.ravel(), cols.ravel(), bounds, shape)
        values = np.asarray([objective(point) for point in points], dtype=float).reshape(shape)
        best_point = np.asarray([0.60, 0.50], dtype=float) if start is None else np.asarray(start, dtype=float)
        best_value = objective(best_point)
        state.bounds = bounds.copy()
        state.evaluated_mask[:, :] = True
        state.objective_values[:, :] = values
        state.objective_field[:, :] = 1.0
        state.best_point = best_point.copy()
        state.best_value = best_value
        state.local_best_point = best_point.copy()
        state.local_best_value = best_value
        state.box_best_point = best_point.copy()
        state.box_best_value = best_value
        state.evaluations = 100
        search.active_max_evaluations = resolved_config.max_evaluations
        search.engine.active_max_evaluations = resolved_config.max_evaluations
        basin = _matching_grid_basin(shape, bounds, values)
        return search, basin

    def test_exploration_phase_runs_smoothlife_without_zooming(self) -> None:
        config = AGSLSConfig(
            max_evaluations=160,
            exploration_fraction=0.40,
            exploration_steps_per_tick=2,
            min_basin_cells=4,
            cluster_min_samples=2,
        )
        search = self._search(config)
        search.reset(seed=11)
        start_bounds = search.engine.current_bounds()
        search.active_max_evaluations = config.max_evaluations
        search.engine.active_max_evaluations = config.max_evaluations
        try:
            search._run_exploration_phase(config.max_evaluations)
        finally:
            search.active_max_evaluations = None
            search.engine.active_max_evaluations = None

        self.assertEqual(search.zoom_events, [])
        self.assertTrue(np.allclose(search.engine.current_bounds(), start_bounds))
        self.assertGreaterEqual(search.engine.state.evaluations, int(config.max_evaluations * config.exploration_fraction * 0.9))

    def test_commit_scoring_prioritizes_group_density_and_support(self) -> None:
        config = AGSLSConfig(
            max_evaluations=100,
            commit_mass_weight=2.0,
            commit_density_weight=2.0,
            commit_objective_weight=0.1,
            commit_stability_weight=0.0,
            commit_area_penalty=0.0,
        )
        dense = _basin(support_mass=10.0, alive_density=0.9, objective_score=0.2)
        objective_only = _basin(support_mass=2.0, alive_density=0.1, objective_score=1.0)

        ranked = score_basins([objective_only, dense], config, phase="commit")

        self.assertIs(ranked[0], dense)

    def test_phase_parameters_apply_guidance_defaults(self) -> None:
        config = AGSLSConfig(max_evaluations=100)
        search = self._search(config)
        search.reset(seed=0)

        search._apply_phase_parameters("exploration")
        self.assertEqual(search.engine.config.objective_guidance_mode, "sampled")
        self.assertEqual(search.engine.config.objective_drift_strength, 0.0)

        search._apply_phase_parameters("commit")
        self.assertEqual(search.engine.config.objective_guidance_mode, "rbf")
        self.assertEqual(search.engine.config.objective_rbf_top_k, config.commit_guidance_top_k)
        self.assertAlmostEqual(search.engine.config.objective_rbf_sigma, config.commit_guidance_sigma)
        self.assertAlmostEqual(search.engine.config.objective_drift_strength, config.commit_drift_strength)

        search._apply_phase_parameters("exploitation")
        self.assertEqual(search.engine.config.objective_guidance_mode, "rbf")
        self.assertEqual(search.engine.config.objective_rbf_top_k, config.exploitation_guidance_top_k)
        self.assertAlmostEqual(search.engine.config.objective_rbf_sigma, config.exploitation_guidance_sigma)
        self.assertAlmostEqual(search.engine.config.objective_drift_strength, config.exploitation_drift_strength)

    def test_commit_zoom_keeps_conservative_width_floor(self) -> None:
        config = AGSLSConfig(max_evaluations=100, commit_min_shrink_fraction=0.45)
        search = self._search(config)
        search.reset(seed=0)
        current_bounds = np.asarray([[-10.0, 10.0], [-10.0, 10.0]], dtype=float)
        tiny = _basin(
            support_mass=10.0,
            alive_density=1.0,
            objective_score=0.5,
            bbox=np.asarray([[-0.1, 0.1], [-0.1, 0.1]], dtype=float),
        )

        bounds, _details = search._commit_bounds(tiny, current_bounds)

        self.assertIsNotNone(bounds)
        widths = bounds[:, 1] - bounds[:, 0]
        self.assertTrue(np.all(widths >= 0.45 * (current_bounds[:, 1] - current_bounds[:, 0]) - 1e-12))

    def test_commit_zoom_retains_global_best(self) -> None:
        config = AGSLSConfig(max_evaluations=100, commit_min_shrink_fraction=0.45)
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        state.best_point = np.asarray([8.0, 8.0], dtype=float)
        state.best_value = 0.0
        current_bounds = np.asarray([[-10.0, 10.0], [-10.0, 10.0]], dtype=float)
        distant = _basin(
            support_mass=10.0,
            alive_density=1.0,
            objective_score=0.5,
            bbox=np.asarray([[-9.0, -8.0], [-9.0, -8.0]], dtype=float),
        )

        bounds, details = search._commit_bounds(distant, current_bounds)

        self.assertIsNotNone(bounds)
        self.assertTrue(bool(details["retained_global_best"]))
        self.assertTrue(np.all(state.best_point >= bounds[:, 0]))
        self.assertTrue(np.all(state.best_point <= bounds[:, 1]))

    def test_commit_surrogate_center_overrides_basin_best_when_valid(self) -> None:
        config = AGSLSConfig(
            max_evaluations=100,
            commit_min_shrink_fraction=0.45,
            commit_surrogate_support_weight=0.0,
        )
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        current_bounds = np.asarray([[-1.0, 2.0], [-1.0, 2.0]], dtype=float)
        optimum = np.asarray([0.37, 0.81], dtype=float)
        shape = search.engine.config.grid_shape
        rows, cols = np.indices(shape)
        points = evaluation_points(rows.ravel(), cols.ravel(), current_bounds, shape)
        values = np.asarray([np.sum((point - optimum) ** 2) for point in points], dtype=float).reshape(shape)
        state.bounds = current_bounds.copy()
        state.evaluated_mask[:, :] = True
        state.objective_values[:, :] = values
        state.objective_field[:, :] = 1.0
        state.best_point = optimum.copy()
        state.best_value = 0.0
        basin = _matching_grid_basin(shape, current_bounds, values, bbox_grid=(8, 8, 23, 23))

        bounds, details = search._commit_bounds(basin, current_bounds)

        self.assertIsNotNone(bounds)
        self.assertTrue(bool(details["surrogate_used"]), msg=details.get("surrogate_reason"))
        self.assertTrue(np.allclose(details["zoom_center"], optimum, atol=1e-2))
        self.assertIn("surrogate_condition", details)

    def test_commit_surrogate_rejection_falls_back_with_diagnostics(self) -> None:
        config = AGSLSConfig(
            max_evaluations=100,
            commit_surrogate_min_samples=4096,
            commit_surrogate_max_samples=4096,
        )
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        current_bounds = np.asarray([[-1.0, 2.0], [-1.0, 2.0]], dtype=float)
        shape = search.engine.config.grid_shape
        rows, cols = np.indices(shape)
        points = evaluation_points(rows.ravel(), cols.ravel(), current_bounds, shape)
        values = np.asarray([np.sum(point * point) for point in points], dtype=float).reshape(shape)
        state.bounds = current_bounds.copy()
        state.evaluated_mask[:, :] = True
        state.objective_values[:, :] = values
        state.objective_field[:, :] = 1.0
        basin = _matching_grid_basin(shape, current_bounds, values, bbox_grid=(8, 8, 23, 23))

        bounds, details = search._commit_bounds(basin, current_bounds)

        self.assertIsNotNone(bounds)
        self.assertFalse(bool(details["surrogate_used"]))
        self.assertEqual(details["surrogate_reason"], "insufficient_samples")

    def test_commit_surrogate_can_fall_back_to_incumbent_local_fit(self) -> None:
        config = AGSLSConfig(
            max_evaluations=100,
            commit_min_shrink_fraction=0.45,
            commit_surrogate_support_weight=0.0,
        )
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        current_bounds = np.asarray([[-1.0, 2.0], [-1.0, 2.0]], dtype=float)
        optimum = np.asarray([0.37, 0.81], dtype=float)
        shape = search.engine.config.grid_shape
        rows, cols = np.indices(shape)
        points = evaluation_points(rows.ravel(), cols.ravel(), current_bounds, shape)
        values = np.asarray([np.sum((point - optimum) ** 2) for point in points], dtype=float).reshape(shape)
        best_flat = int(np.argmin(values))
        best_row, best_col = np.unravel_index(best_flat, shape)
        best_point = evaluation_points(
            np.asarray([best_row], dtype=int),
            np.asarray([best_col], dtype=int),
            current_bounds,
            shape,
        )[0]
        state.bounds = current_bounds.copy()
        state.evaluated_mask[:, :] = True
        state.objective_values[:, :] = values
        state.objective_field[:, :] = 1.0
        state.best_point = best_point
        state.best_value = float(values[best_row, best_col])
        distant = _basin(
            support_mass=10.0,
            alive_density=1.0,
            objective_score=0.5,
            bbox=np.asarray([[1.4, 1.8], [-0.8, -0.4]], dtype=float),
        )
        distant.basin_best_value = 10.0

        bounds, details = search._commit_bounds(distant, current_bounds)

        self.assertIsNotNone(bounds)
        self.assertTrue(bool(details["surrogate_used"]), msg=details.get("surrogate_reason"))
        self.assertEqual(details["surrogate_source"], "incumbent")
        self.assertTrue(np.allclose(details["zoom_center"], optimum, atol=1e-2))

    def test_commit_surrogate_bounds_retain_incumbent_and_floor(self) -> None:
        config = AGSLSConfig(
            max_evaluations=100,
            commit_min_shrink_fraction=0.45,
            commit_surrogate_support_weight=0.0,
        )
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        current_bounds = np.asarray([[-2.0, 2.0], [-2.0, 2.0]], dtype=float)
        shape = search.engine.config.grid_shape
        rows, cols = np.indices(shape)
        points = evaluation_points(rows.ravel(), cols.ravel(), current_bounds, shape)
        values = np.asarray([point[0] ** 2 + 3.0 * point[1] ** 2 for point in points], dtype=float).reshape(shape)
        state.bounds = current_bounds.copy()
        state.evaluated_mask[:, :] = True
        state.objective_values[:, :] = values
        state.objective_field[:, :] = 1.0
        state.best_point = np.asarray([1.9, 1.9], dtype=float)
        state.best_value = -1.0
        basin = _matching_grid_basin(shape, current_bounds, values, bbox_grid=(8, 8, 23, 23))

        bounds, details = search._commit_bounds(basin, current_bounds)

        self.assertIsNotNone(bounds)
        self.assertTrue(bool(details["surrogate_used"]), msg=details.get("surrogate_reason"))
        self.assertTrue(bool(details["retained_global_best"]))
        self.assertTrue(np.all(state.best_point >= bounds[:, 0]))
        self.assertTrue(np.all(state.best_point <= bounds[:, 1]))
        widths = bounds[:, 1] - bounds[:, 0]
        self.assertTrue(np.all(widths >= 0.45 * (current_bounds[:, 1] - current_bounds[:, 0]) - 1e-12))

    def test_commit_persistence_explores_minimum_active_box_fraction(self) -> None:
        config = AGSLSConfig(
            max_evaluations=1000,
            commit_min_explored_fraction=0.10,
            commit_steps_per_zoom=1,
            min_basin_cells=4,
            cluster_min_samples=2,
        )
        search = self._search(config)
        search.reset(seed=5)
        search.active_max_evaluations = config.max_evaluations
        search.engine.active_max_evaluations = config.max_evaluations
        try:
            search._persistence_map(1, min_explored_fraction=config.commit_min_explored_fraction)
        finally:
            search.active_max_evaluations = None
            search.engine.active_max_evaluations = None

        self.assertGreaterEqual(float(np.mean(search.engine.state.evaluated_mask)), config.commit_min_explored_fraction)

    def test_exploitation_centers_on_global_best_when_available(self) -> None:
        config = AGSLSConfig(max_evaluations=100, exploitation_shrink_fraction=0.25)
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        state.bounds = np.asarray([[-4.0, 4.0], [-4.0, 4.0]], dtype=float)
        state.best_point = np.asarray([1.0, -2.0], dtype=float)
        state.best_value = 0.0

        bounds, details = search._exploitation_bounds(None, state.bounds)

        self.assertIsNotNone(bounds)
        self.assertTrue(details["anchored_on_global_best"])
        self.assertTrue(np.allclose(np.mean(bounds, axis=1), state.best_point))

    def test_exploitation_valley_tracking_updates_global_best(self) -> None:
        search, basin = self._valley_tracking_fixture()
        state = search.engine.state
        before = float(state.best_value)

        details = search._track_exploitation_valley(basin, state.bounds.copy())

        self.assertTrue(bool(details["valley_tracking_used"]), msg=details["valley_tracking_reason"])
        self.assertGreater(int(details["valley_tracking_probes"]), 0)
        self.assertGreater(int(details["valley_tracking_improvements"]), 0)
        self.assertLess(float(state.best_value), before)
        self.assertLess(float(state.best_point[0]), 0.60)

    def test_exploitation_valley_tracking_keeps_best_when_probes_are_worse(self) -> None:
        search, basin = self._valley_tracking_fixture(start=np.asarray([0.40, 0.50], dtype=float))
        state = search.engine.state
        before_point = state.best_point.copy()
        before_value = float(state.best_value)

        details = search._track_exploitation_valley(basin, state.bounds.copy())

        self.assertTrue(bool(details["valley_tracking_used"]), msg=details["valley_tracking_reason"])
        self.assertEqual(int(details["valley_tracking_improvements"]), 0)
        self.assertTrue(np.allclose(state.best_point, before_point))
        self.assertEqual(float(state.best_value), before_value)

    def test_exploitation_valley_tracking_respects_probe_reserve(self) -> None:
        config = AGSLSConfig(
            max_evaluations=120,
            exploitation_valley_probe_evaluations=8,
            exploitation_valley_step_fraction=0.15,
        )
        search, basin = self._valley_tracking_fixture(config=config)
        state = search.engine.state
        state.evaluations = config.max_evaluations - search.engine.config.evaluations_per_step

        details = search._track_exploitation_valley(basin, state.bounds.copy())

        self.assertFalse(bool(details["valley_tracking_used"]))
        self.assertEqual(details["valley_tracking_reason"], "no_probe_budget")
        self.assertEqual(int(details["valley_tracking_probes"]), 0)
        self.assertEqual(int(state.evaluations), config.max_evaluations - search.engine.config.evaluations_per_step)

    def test_exploitation_valley_tracking_disabled_is_noop(self) -> None:
        config = AGSLSConfig(max_evaluations=240, exploitation_valley_tracking_enabled=False)
        search, basin = self._valley_tracking_fixture(config=config)
        state = search.engine.state
        before_point = state.best_point.copy()
        before_value = float(state.best_value)

        details = search._track_exploitation_valley(basin, state.bounds.copy())

        self.assertFalse(bool(details["valley_tracking_used"]))
        self.assertEqual(details["valley_tracking_reason"], "disabled")
        self.assertTrue(np.allclose(state.best_point, before_point))
        self.assertEqual(float(state.best_value), before_value)

    def test_exploitation_zoom_centers_on_valley_improved_best(self) -> None:
        config = AGSLSConfig(
            max_evaluations=240,
            exploitation_shrink_fraction=0.25,
            exploitation_valley_probe_evaluations=4,
            exploitation_valley_step_fraction=0.15,
        )
        search, basin = self._valley_tracking_fixture(config=config)
        state = search.engine.state

        valley_details = search._track_exploitation_valley(basin, state.bounds.copy())
        bounds, details = search._exploitation_bounds(basin, state.bounds.copy())
        details.update(valley_details)

        self.assertIsNotNone(bounds)
        self.assertGreater(int(details["valley_tracking_improvements"]), 0)
        self.assertTrue(np.allclose(np.mean(bounds, axis=1), state.best_point))

    def test_exploitation_valley_tracking_rejected_surrogate_falls_back(self) -> None:
        config = AGSLSConfig(
            max_evaluations=240,
            exploitation_valley_surrogate_min_samples=4000,
            exploitation_valley_surrogate_max_samples=4000,
        )
        search, basin = self._valley_tracking_fixture(config=config)
        state = search.engine.state
        before_point = state.best_point.copy()

        details = search._track_exploitation_valley(basin, state.bounds.copy())
        bounds, zoom_details = search._exploitation_bounds(basin, state.bounds.copy())

        self.assertFalse(bool(details["valley_tracking_used"]))
        self.assertTrue(str(details["valley_tracking_reason"]).startswith("surrogate_"))
        self.assertIsNotNone(bounds)
        self.assertTrue(zoom_details["anchored_on_global_best"])
        self.assertTrue(np.allclose(np.mean(bounds, axis=1), before_point))

    def test_exploitation_can_shrink_below_current_pixel_size(self) -> None:
        config = AGSLSConfig(max_evaluations=100, exploitation_shrink_fraction=0.20)
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        state.best_point = np.asarray([0.0, 0.0], dtype=float)
        current = np.asarray([[-8.0, 8.0], [-8.0, 8.0]], dtype=float)
        original_cell_width = (current[0, 1] - current[0, 0]) / search.engine.config.grid_shape[1]

        for _ in range(3):
            next_bounds, _details = search._exploitation_bounds(None, current)
            self.assertIsNotNone(next_bounds)
            current = next_bounds

        self.assertLess(float(current[0, 1] - current[0, 0]), original_cell_width)

    def test_exploitation_progress_ignores_shared_area_floor(self) -> None:
        config = AGSLSConfig(max_evaluations=100, exploitation_shrink_fraction=1e-6)
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        state.best_point = np.asarray([0.0, 0.0], dtype=float)
        current = np.asarray([[-1e-13, 1e-13], [-2e-13, 2e-13]], dtype=float)

        bounds, details = search._exploitation_bounds(None, current)

        self.assertIsNotNone(bounds)
        self.assertEqual(details["zoom_reason"], "exploitation_global_best")
        self.assertLess(float(bounds[0, 1] - bounds[0, 0]), float(current[0, 1] - current[0, 0]))

    def test_exploitation_uses_representable_floor_around_nonzero_best(self) -> None:
        config = AGSLSConfig(max_evaluations=100, exploitation_shrink_fraction=1e-26)
        search = self._search(config)
        search.reset(seed=0)
        state = search.engine.state
        state.best_point = np.asarray([1.0043639203712165, 1.0086010958425873], dtype=float)
        current = np.asarray(
            [[0.4017116880416868, 1.031345936745257], [0.0511081186294553, 1.0517908766555786]],
            dtype=float,
        )

        bounds, details = search._exploitation_bounds(None, current)

        self.assertIsNotNone(bounds)
        self.assertTrue(bool(details["representable_floor"]))
        self.assertTrue(np.all(bounds[:, 1] > bounds[:, 0]))
        self.assertTrue(np.all(state.best_point >= bounds[:, 0]))
        self.assertTrue(np.all(state.best_point <= bounds[:, 1]))

    def test_exploitation_persistence_reserves_remap_budget(self) -> None:
        config = AGSLSConfig(max_evaluations=40, exploitation_steps_per_zoom=100, min_basin_cells=4, cluster_min_samples=2)
        search = self._search(config)
        search.reset(seed=0)
        search.active_max_evaluations = config.max_evaluations
        search.engine.active_max_evaluations = config.max_evaluations
        try:
            search._persistence_map(100, reserve_evaluations=search.engine.config.evaluations_per_step)
        finally:
            search.active_max_evaluations = None
            search.engine.active_max_evaluations = None

        self.assertGreaterEqual(config.max_evaluations - int(search.engine.state.evaluations), search.engine.config.evaluations_per_step)


if __name__ == "__main__":
    unittest.main()
