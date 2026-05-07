from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig, sphere
from smooth_life_search.point_cloud.search import CandidateProposal


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

    def _high_dimensional_search(self, dimension: int = 30, seed: int = 0) -> PointCloudSmoothLifeSearch:
        search = PointCloudSmoothLifeSearch(
            sphere,
            [(-10.0, 10.0)] * dimension,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=1000,
                initial_design_size=48,
                batch_size=32,
                density_grid_shape=(24, 24),
                local_refinement_enabled=False,
            ),
        )
        search.reset(seed=seed)
        search._batch_index = 1
        search._evaluate_candidates(search._initial_design())
        search._update_portfolio()
        search._refresh_views()
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
        points = np.asarray([sample.point for sample in search.archive.samples], dtype=float)

        self.assertTrue(any(source.startswith("region:") and "stencil" in source for source in sources))
        self.assertTrue(any(source.startswith("region:") and source.endswith(":rotated_stencil") for source in sources))
        self.assertIn("exploit_stencil", sources)
        self.assertTrue(np.all(points >= -5.0))
        self.assertTrue(np.all(points <= 5.0))

    def test_levenberg_marquardt_refinement_records_diagnostics(self) -> None:
        search = PointCloudSmoothLifeSearch(
            lambda point: float((point[0] - 1.25) ** 2 + 3.0 * (point[1] + 0.5) ** 2),
            [(-4.0, 4.0), (-4.0, 4.0)],
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=180,
                initial_design_size=16,
                batch_size=12,
                density_grid_shape=(24, 24),
                local_refinement_method="levenberg-marquardt",
                early_stop_enabled=True,
                early_stop_value=1e-8,
            ),
        )
        search.reset(seed=6)
        run = search.run()
        local_events = [event for event in run.metadata["batch_events"] if event["kind"] == "local_refinement"]

        self.assertLess(run.best_value, 1e-8)
        self.assertTrue(local_events)
        self.assertEqual(local_events[-1]["local_refinement_method"], "levenberg-marquardt")
        self.assertGreater(local_events[-1]["lm_accepted_steps"], 0)
        self.assertIn("damping_final", local_events[-1])
        self.assertIn("fallback_count", local_events[-1])

    def test_5d_run_keeps_candidates_in_bounds_and_emits_pattern_probes(self) -> None:
        target = np.asarray([1.0, -1.0, 0.5, -0.5, 0.25], dtype=float)
        search = PointCloudSmoothLifeSearch(
            lambda point: float(np.sum(np.square(point - target))),
            [(-3.0, 3.0)] * 5,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=800,
                initial_design_size=24,
                batch_size=16,
                density_grid_shape=(24, 24),
                local_refinement_enabled=False,
                early_stop_enabled=True,
                early_stop_value=1e-6,
            ),
        )
        search.reset(seed=4)
        run = search.run()
        points = np.asarray([sample.point for sample in search.archive.samples], dtype=float)
        sources = {
            source
            for event in run.metadata["batch_events"]
            for source in event["source_counts"]
        }

        self.assertLess(run.best_value, 1e-6)
        self.assertEqual(run.best_point.shape, (5,))
        self.assertTrue(np.all(points >= -3.0))
        self.assertTrue(np.all(points <= 3.0))
        self.assertIn("exploit_pattern", sources)

    def test_5d_bfgs_refinement_records_local_events(self) -> None:
        target = np.asarray([0.5, -0.75, 1.0, -1.25, 0.25], dtype=float)
        search = PointCloudSmoothLifeSearch(
            lambda point: float(np.sum(np.square(point - target))),
            [(-3.0, 3.0)] * 5,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=300,
                initial_design_size=24,
                batch_size=16,
                density_grid_shape=(24, 24),
                local_refinement_method="bfgs",
                early_stop_enabled=True,
                early_stop_value=1e-8,
            ),
        )
        search.reset(seed=2)
        run = search.run()
        local_events = [event for event in run.metadata["batch_events"] if event["kind"] == "local_refinement"]

        self.assertLess(run.best_value, 1e-8)
        self.assertTrue(local_events)
        self.assertEqual(local_events[-1]["local_refinement_method"], "bfgs")

    def test_high_dimensional_effective_batch_size_scales_and_caps(self) -> None:
        cases = [(2, 32), (30, 60), (50, 100), (500, 128)]
        for dimension, expected in cases:
            with self.subTest(dimension=dimension):
                search = PointCloudSmoothLifeSearch(
                    sphere,
                    [(-5.0, 5.0)] * dimension,
                    SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
                    PointCloudSearchConfig(batch_size=32, dimension_scaled_batch_max=128),
                )
                self.assertEqual(search._effective_batch_size(), expected)

    def test_cloud_only_direction_refinement_uses_compact_batch_floor(self) -> None:
        search = PointCloudSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 2,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(batch_size=32, local_refinement_enabled=False),
        )

        self.assertEqual(search._effective_batch_size(), 48)

    def test_deterministic_stage_allocator_emits_lean_sources(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=1)

        exploration = search._candidate_counts(100)
        repeated_exploration = search._candidate_counts(100)
        search._anchor_baseline_target = 100.0
        search.best_value = 10.0
        polishing = search._candidate_counts(100)

        self.assertEqual(set(exploration), {"global", "smoothlife_density", "region", "coherent"})
        self.assertEqual(exploration, repeated_exploration)
        self.assertGreater(polishing["exploit"], exploration.get("exploit", 0))
        self.assertGreater(polishing["region"], 0)
        self.assertGreater(polishing["coherent"], 0)
        self.assertNotIn("shade", polishing)
        self.assertNotIn("cma", polishing)
        self.assertNotIn("restart", polishing)

    def test_lean_source_stats_are_simple_evaluation_accounting(self) -> None:
        search = self._high_dimensional_search(dimension=100, seed=2)
        candidates = search._candidate_batch(24)
        event = search._evaluate_candidates(candidates)

        stats = search._source_stats_payload()

        self.assertIn("source_stats", event.diagnostics)
        self.assertTrue(stats)
        for payload in stats.values():
            self.assertIn("attempts", payload)
            self.assertIn("evaluations", payload)
            self.assertIn("improvements", payload)
            self.assertIn("improvement_sum", payload)
            self.assertIn("evaluation_share", payload)
            self.assertNotIn("allocation_weight", payload)
            self.assertNotIn("relative_wins", payload)

    def test_high_dimensional_polishing_preserves_coherent_and_cooperative_floors(self) -> None:
        search = self._high_dimensional_search(dimension=500, seed=2)
        search._anchor_baseline_target = 500.0
        search.best_value = 10.0

        counts = search._candidate_counts(128)

        self.assertGreaterEqual(counts["coherent"], 20)
        self.assertGreaterEqual(counts["cooperative"], 16)
        self.assertGreater(counts["coherent"], counts["global"])
        self.assertGreater(counts["cooperative"], counts["global"])

    def test_no_removed_evolutionary_sources_are_emitted(self) -> None:
        first = self._high_dimensional_search(dimension=30, seed=3)
        second = self._high_dimensional_search(dimension=30, seed=3)

        first_candidates = first._candidate_batch(32)
        second_candidates = second._candidate_batch(32)
        first_points = np.asarray([candidate.point for candidate in first_candidates], dtype=float)
        second_points = np.asarray([candidate.point for candidate in second_candidates], dtype=float)
        sources = {candidate.source for candidate in first_candidates}

        self.assertTrue(np.allclose(first_points, second_points))
        self.assertTrue(np.all(first_points >= -10.0))
        self.assertTrue(np.all(first_points <= 10.0))
        keys = {tuple(point) for point in first_points}
        self.assertEqual(len(keys), len(first_points))
        self.assertFalse(any(source == "shade" or source == "restart:scout" or ":cma" in source for source in sources))

    def test_candidate_proposal_metadata_survives_generation_dedup_and_evaluation(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=4)
        proposals = search._candidate_batch(12)

        self.assertTrue(all(isinstance(candidate, CandidateProposal) for candidate in proposals))
        self.assertTrue(any(candidate.parent_key is not None for candidate in proposals))

        ranked = search._rank_candidate_pool([*proposals, *proposals], 6, search._candidate_counts(6))
        event = search._evaluate_candidates(ranked)

        self.assertLessEqual(event.candidate_count, 6)
        self.assertIn("source_stats", event.diagnostics)
        self.assertIn("surrogate_ranked_candidates", event.diagnostics)

    def test_surrogate_preselection_is_deterministic_bounded_and_budget_exact(self) -> None:
        first = self._high_dimensional_search(dimension=30, seed=10)
        second = self._high_dimensional_search(dimension=30, seed=10)

        first_candidates = first._candidate_batch(20)
        second_candidates = second._candidate_batch(20)
        first_points = np.asarray([candidate.point for candidate in first_candidates], dtype=float)
        second_points = np.asarray([candidate.point for candidate in second_candidates], dtype=float)
        before = len(first.archive)
        event = first._evaluate_candidates(first_candidates)

        self.assertEqual(len(first_candidates), 20)
        self.assertTrue(np.allclose(first_points, second_points))
        self.assertTrue(np.all(first_points >= -10.0))
        self.assertTrue(np.all(first_points <= 10.0))
        self.assertTrue(any(candidate.predicted_score is not None for candidate in first_candidates))
        self.assertLessEqual(event.evaluations_after - before, 20)

    def test_cooperative_candidates_modify_only_selected_group_axes(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=7)
        search.config.cooperative_min_dimension = 30
        search.best_point = np.linspace(-2.0, 2.0, 30)
        candidates = search._cooperative_candidates(12)

        self.assertTrue(candidates)
        for candidate in candidates:
            axes = set(candidate.axes or ())
            changed = set(np.flatnonzero(np.abs(candidate.point - search.best_point) > 1e-12))
            self.assertTrue(changed.issubset(axes))
            self.assertIn(candidate.source, {"cooperative:group", "cooperative:line_search"})

    def test_coherent_candidates_are_bounded_jittered_and_parent_labeled(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=13)
        search.best_point = np.linspace(-2.0, 2.0, 30)
        candidates = search._coherent_candidates(6)
        points = np.asarray([candidate.point for candidate in candidates], dtype=float)
        normalized = (points + 10.0) / 20.0

        self.assertTrue(candidates)
        self.assertEqual({candidate.source for candidate in candidates}, {"coherent"})
        self.assertTrue(all(candidate.parent_key is not None for candidate in candidates))
        self.assertTrue(np.all(points >= -10.0))
        self.assertTrue(np.all(points <= 10.0))
        self.assertTrue(np.all(np.var(normalized, axis=1) > 0.0))

    def test_coherent_probes_can_be_disabled(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=13)
        search.config.coherent_probes_enabled = False

        self.assertEqual(search._coherent_candidates(6), [])
        self.assertEqual(search._candidate_counts(64).get("coherent", 0), 0)

    def test_high_dimensional_ecology_density_view_is_finite_and_nonblank(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=9)
        before = np.zeros(30, dtype=float)
        after = np.zeros(30, dtype=float)
        after[[5, 11]] = 1.0
        search._record_successful_direction(before, after)
        search._axis_coverage[:] = 4.0
        search._axis_coverage[[17, 23]] = 0.0
        density, objective, evaluated = search._density_view()
        candidates = search._density_candidates(12)

        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(np.isfinite(objective)))
        self.assertGreater(float(np.max(density)), 0.0)
        self.assertGreater(int(np.count_nonzero(evaluated)), 0)
        self.assertGreaterEqual(search._last_projection_frame_count, 2)
        self.assertEqual(len(candidates), 12)
        self.assertTrue(all(candidate.source == "density" or candidate.source == "global" for candidate in candidates))
        self.assertTrue(all(np.all(candidate.point >= -10.0) and np.all(candidate.point <= 10.0) for candidate in candidates))

    def test_projection_ensemble_can_be_disabled(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=9)
        search.config.projection_ensemble_enabled = False
        frames = search._density_projection_frames()

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0], "coordinate")

    def test_probe_improvement_recenters_block_gradient(self) -> None:
        target = np.ones(12, dtype=float)
        search = PointCloudSmoothLifeSearch(
            lambda point: float(np.sum(np.square(point - target))),
            [(-10000.0, 10000.0)] * 12,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=120,
                initial_design_size=16,
                batch_size=16,
                probe_recenter_max_restarts=1,
            ),
        )
        search.reset(seed=14)
        search._batch_index = 1
        search._evaluate_point(np.zeros(12, dtype=float), source="center")
        search._anchor_baseline_target = 100.0

        result = search._finite_difference_block_gradient(search.best_point, np.arange(4, dtype=int))

        self.assertIsNotNone(result.gradient)
        self.assertTrue(result.recentered)
        self.assertEqual(result.improvement_count, 1)
        self.assertGreater(float(np.linalg.norm(result.point)), 0.0)
        self.assertLess(search.best_value, 12.0)

    def test_basin_polishing_allocation_activates_after_baseline_ratio(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=15)
        search._anchor_baseline_target = 100.0
        search.best_value = 40.0

        inactive = search._candidate_counts(100)
        search.best_value = 10.0
        active = search._candidate_counts(100)

        self.assertFalse(search._target(40.0) <= 0.25 * search._anchor_baseline_target)
        self.assertLess(active["global"], inactive["global"])
        self.assertGreaterEqual(active["exploit"], inactive.get("exploit", 0))
        self.assertTrue(search._basin_polishing_active())

    def test_successful_direction_memory_is_bounded_and_normalized(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=16)
        search.config.successful_direction_memory_size = 2

        for offset in range(4):
            before = np.zeros(30, dtype=float)
            after = np.zeros(30, dtype=float)
            after[offset] = 1.0
            search._record_successful_direction(before, after)

        self.assertEqual(len(search._successful_directions), 2)
        self.assertTrue(all(np.isclose(np.linalg.norm(direction), 1.0) for direction in search._successful_directions))

    def test_direction_line_search_improves_quadratic_without_local_refinement(self) -> None:
        target = np.ones(12, dtype=float)
        search = PointCloudSmoothLifeSearch(
            lambda point: float(np.sum(np.square(point - target))),
            [(-5.0, 5.0)] * 12,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=120,
                initial_design_size=16,
                batch_size=16,
                local_refinement_enabled=False,
                direction_refinement_max_evaluations=24,
            ),
        )
        search.reset(seed=17)
        search._batch_index = 1
        search._evaluate_point(np.zeros(12, dtype=float), source="center")
        search._anchor_baseline_target = 100.0
        search._record_successful_direction(-np.ones(12, dtype=float), np.zeros(12, dtype=float))
        before = float(search.best_value)

        event = search._run_direction_refinement()

        self.assertIsNotNone(event)
        self.assertLess(search.best_value, before)
        self.assertEqual(event.diagnostics["direction_line_search_mode"], "bracketed")
        self.assertIn("direction_quadratic_steps", event.diagnostics)
        self.assertGreater(event.diagnostics["direction_line_search_improvements"], 0)
        self.assertIn("direction_line_search", event.source_counts)

    def test_surrogate_reliability_gates_candidate_rank_weight(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=27)
        proposals = search._candidate_batch(20)

        self.assertTrue(any(candidate.predicted_score is not None for candidate in proposals))
        self.assertGreaterEqual(search._last_surrogate_reliability, 0.0)
        self.assertLessEqual(search._last_surrogate_reliability, 1.0)
        self.assertGreaterEqual(search._last_surrogate_rank_weight, search.config.surrogate_rank_weight_min)
        self.assertLessEqual(search._last_surrogate_rank_weight, search.config.surrogate_rank_weight_max)

    def test_cooperative_frontier_groups_follow_recent_improved_axes(self) -> None:
        search = self._high_dimensional_search(dimension=120, seed=28)
        search._last_cooperative_improved_axes = (40,)
        search._axis_coverage[:] = 5.0
        search._axis_coverage[[39, 41]] = 0.0

        groups = search._cooperative_groups(4)

        self.assertTrue(groups)
        self.assertTrue(any(39 in group.tolist() or 41 in group.tolist() for group in groups))

    def test_linkage_scores_produce_coupled_blocks(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=18)
        search.config.linkage_neighbor_count = 1
        before = np.zeros(30, dtype=float)
        after = np.zeros(30, dtype=float)
        after[[3, 7]] = 1.0
        search._record_successful_direction(before, after)
        search._axis_activity[3] = 10.0

        block = search._linked_block()

        self.assertIsNotNone(block)
        self.assertIn(3, block.tolist())
        self.assertIn(7, block.tolist())

    def test_cross_block_lbfgs_direction_accepts_descent_and_rejects_bad_curvature(self) -> None:
        search = self._high_dimensional_search(dimension=30, seed=19)
        step = np.zeros(30, dtype=float)
        step[0] = 1.0
        gradient_delta = np.zeros(30, dtype=float)
        gradient_delta[0] = 2.0

        search._store_lbfgs_pair(step, gradient_delta)
        direction = search._lbfgs_direction(gradient_delta)

        self.assertEqual(len(search._lbfgs_pairs), 1)
        self.assertIsNotNone(direction)
        self.assertLess(float(np.dot(direction, gradient_delta)), 0.0)

        bad = self._high_dimensional_search(dimension=30, seed=20)
        bad._store_lbfgs_pair(step, -gradient_delta)
        self.assertEqual(len(bad._lbfgs_pairs), 0)

    def test_cooperative_active_set_scores_are_finite_and_deterministic(self) -> None:
        first = self._high_dimensional_search(dimension=120, seed=21)
        second = self._high_dimensional_search(dimension=120, seed=21)
        first._axis_activity[[3, 9]] = [2.0, 1.0]
        second._axis_activity[[3, 9]] = [2.0, 1.0]

        first_scores = first._cooperative_axis_scores()
        second_scores = second._cooperative_axis_scores()
        first_axes = first._active_set_axes()
        second_axes = second._active_set_axes()

        self.assertTrue(np.all(np.isfinite(first_scores)))
        self.assertTrue(np.allclose(first_scores, second_scores))
        self.assertTrue(np.array_equal(first_axes, second_axes))
        self.assertLessEqual(first_axes.size, int(np.ceil(0.25 * 120)))

    def test_cooperative_active_set_expansion_preserves_coverage_pressure(self) -> None:
        search = self._high_dimensional_search(dimension=120, seed=22)
        search._axis_coverage[:] = 10.0
        search._axis_coverage[[17, 43]] = 0.0
        search._batch_index = 10

        axes = search._active_set_axes()

        self.assertIn(17, axes.tolist())
        self.assertIn(43, axes.tolist())
        self.assertGreater(search._active_set_expansions, 0)

    def test_cooperative_candidates_are_bounded_and_modify_only_group_axes(self) -> None:
        search = self._high_dimensional_search(dimension=120, seed=23)
        search.best_point = np.linspace(-1.0, 1.0, 120)
        candidates = search._cooperative_candidates(16)

        self.assertTrue(candidates)
        for proposal in candidates:
            self.assertIsNotNone(proposal.axes)
            self.assertTrue(proposal.source.startswith("cooperative:") or proposal.source == "global")
            self.assertTrue(np.all(proposal.point >= -10.0))
            self.assertTrue(np.all(proposal.point <= 10.0))
            if proposal.axes is None or proposal.source == "global":
                continue
            changed = np.flatnonzero(np.abs(proposal.point - search.best_point) > 1e-12)
            self.assertTrue(set(changed).issubset(set(proposal.axes)))

    def test_cooperative_groups_use_linkage_and_coverage(self) -> None:
        search = self._high_dimensional_search(dimension=120, seed=24)
        search.config.linkage_neighbor_count = 1
        before = np.zeros(120, dtype=float)
        after = np.zeros(120, dtype=float)
        after[[5, 77]] = 1.0
        search._record_successful_direction(before, after)
        search._axis_activity[5] = 10.0
        search._axis_coverage[:] = 5.0
        search._axis_coverage[91] = 0.0

        groups = search._cooperative_groups(4)
        flattened = {int(axis) for group in groups for axis in group}

        self.assertTrue(any(5 in group.tolist() and 77 in group.tolist() for group in groups))
        self.assertIn(91, flattened)

    def test_cooperative_refinement_improves_shifted_quadratic_without_local_refinement(self) -> None:
        target = np.ones(120, dtype=float)
        search = PointCloudSmoothLifeSearch(
            lambda point: float(np.sum(np.square(point - target))),
            [(-5.0, 5.0)] * 120,
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            PointCloudSearchConfig(
                max_evaluations=260,
                initial_design_size=32,
                batch_size=32,
                local_refinement_enabled=False,
                cooperative_min_dimension=100,
            ),
        )
        search.reset(seed=25)
        search._batch_index = 1
        search._evaluate_point(np.zeros(120, dtype=float), source="center")
        before = float(search.best_value)

        event = search._run_cooperative_refinement()

        self.assertIsNotNone(event)
        self.assertLess(search.best_value, before)
        self.assertGreater(event.diagnostics["cooperative_improvements"], 0)
        self.assertIn("cooperative:line_search", event.source_counts)

    def test_disabling_cooperative_refinement_removes_large_d_allocation(self) -> None:
        enabled = self._high_dimensional_search(dimension=120, seed=26)
        disabled = self._high_dimensional_search(dimension=120, seed=26)
        enabled._anchor_baseline_target = 100.0
        enabled.best_value = 10.0
        disabled._anchor_baseline_target = 100.0
        disabled.best_value = 10.0
        disabled.config.cooperative_refinement_enabled = False

        enabled_counts = enabled._candidate_counts(100)
        disabled_counts = disabled._candidate_counts(100)

        self.assertGreater(enabled_counts.get("cooperative", 0), 0)
        self.assertEqual(disabled_counts.get("cooperative", 0), 0)

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
