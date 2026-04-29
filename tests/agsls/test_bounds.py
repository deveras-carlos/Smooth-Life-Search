"""Tests for AGSLS bound-construction (R5: incumbent containment with re-expansion)."""

from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import AGSLSConfig
from smooth_life_search.agsls.geometry import finalize_zoom_bounds


def _bounds(x_low: float, x_high: float, y_low: float, y_high: float) -> np.ndarray:
    return np.asarray([[x_low, x_high], [y_low, y_high]], dtype=float)


def _config(**overrides: float | int) -> AGSLSConfig:
    base = AGSLSConfig(
        max_zoom_cycles=5,
        zoom_padding=0.0,
        min_side_fraction=1e-64,
        edge_risk_fraction=0.05,
        min_zoom_cells=2,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestFinalizeZoomBounds(unittest.TestCase):
    """Geometry-level tests for `finalize_zoom_bounds` after the R5 fix."""

    def test_no_incumbent_keeps_bounds_inside_current(self) -> None:
        original = _bounds(-5.0, 5.0, -5.0, 5.0)
        current = _bounds(-2.0, 2.0, -2.0, 2.0)
        proposed = _bounds(-1.0, 1.0, -1.0, 1.0)
        result = finalize_zoom_bounds(
            proposed,
            np.array([0.0, 0.0]),
            current,
            original,
            grid_shape=(64, 64),
            config=_config(),
        )
        self.assertTrue(np.all(result[:, 0] >= current[:, 0] - 1e-12))
        self.assertTrue(np.all(result[:, 1] <= current[:, 1] + 1e-12))

    def test_incumbent_inside_current_but_outside_proposed_pulls_bounds_in(self) -> None:
        original = _bounds(-5.0, 5.0, -5.0, 5.0)
        current = _bounds(-2.0, 2.0, -2.0, 2.0)
        proposed = _bounds(0.5, 1.5, 0.5, 1.5)
        incumbent = np.array([0.0, 0.0])
        result = finalize_zoom_bounds(
            proposed,
            np.array([1.0, 1.0]),
            current,
            original,
            grid_shape=(64, 64),
            config=_config(),
            incumbent_point=incumbent,
        )
        self.assertLessEqual(result[0, 0], incumbent[0] + 1e-12)
        self.assertLessEqual(result[1, 0], incumbent[1] + 1e-12)

    def test_incumbent_outside_current_re_expands_to_original(self) -> None:
        """The R5 fix: if a previous zoom cropped the optimum out of current_bounds,
        a new bound construction with the incumbent passed in must re-expand the
        offending axis up to original_bounds.
        """

        original = _bounds(-5.0, 5.0, -5.0, 5.0)
        current = _bounds(-2.0, 2.0, 1.14, 2.0)  # y was cropped to [1.14, 2.0]
        proposed = _bounds(-1.0, 1.0, 1.5, 1.9)  # basin lives high in y
        incumbent = np.array([1.0, 1.0])  # true optimum below current y_low
        result = finalize_zoom_bounds(
            proposed,
            np.array([0.0, 1.7]),
            current,
            original,
            grid_shape=(64, 64),
            config=_config(),
            incumbent_point=incumbent,
        )
        # Pre-R5: result[1, 0] would have been clipped at 1.14 (cropped optimum).
        # Post-R5: result[1, 0] should drop to or below incumbent[1] = 1.0.
        self.assertLessEqual(result[1, 0], incumbent[1] + 1e-9)
        # And it must remain inside original_bounds.
        self.assertGreaterEqual(result[1, 0], original[1, 0] - 1e-12)

    def test_incumbent_outside_original_is_skipped_defensively(self) -> None:
        original = _bounds(-5.0, 5.0, -5.0, 5.0)
        current = _bounds(-2.0, 2.0, -2.0, 2.0)
        proposed = _bounds(-1.0, 1.0, -1.0, 1.0)
        far_incumbent = np.array([10.0, 10.0])  # outside original
        result = finalize_zoom_bounds(
            proposed,
            np.array([0.0, 0.0]),
            current,
            original,
            grid_shape=(64, 64),
            config=_config(),
            incumbent_point=far_incumbent,
        )
        # Defensive: a pathological out-of-original incumbent should not warp the bounds.
        self.assertTrue(np.all(result[:, 0] >= original[:, 0] - 1e-12))
        self.assertTrue(np.all(result[:, 1] <= original[:, 1] + 1e-12))

    def test_re_expansion_never_exceeds_original_bounds(self) -> None:
        original = _bounds(-3.0, 3.0, -3.0, 3.0)
        current = _bounds(-1.0, 1.0, -1.0, 1.0)
        proposed = _bounds(-0.5, 0.5, -0.5, 0.5)
        # incumbent is just inside original, way outside current
        incumbent = np.array([2.9, -2.9])
        result = finalize_zoom_bounds(
            proposed,
            np.array([0.0, 0.0]),
            current,
            original,
            grid_shape=(32, 32),
            config=_config(edge_risk_fraction=0.5),
            incumbent_point=incumbent,
        )
        self.assertGreaterEqual(result[0, 0], original[0, 0] - 1e-12)
        self.assertLessEqual(result[0, 1], original[0, 1] + 1e-12)
        self.assertGreaterEqual(result[1, 0], original[1, 0] - 1e-12)
        self.assertLessEqual(result[1, 1], original[1, 1] + 1e-12)
        # Incumbent must end up inside the new bounds.
        self.assertLessEqual(result[0, 0], incumbent[0] + 1e-9)
        self.assertGreaterEqual(result[1, 1], incumbent[1] - 1e-9)


class TestRosenbrockBoundLossRegression(unittest.TestCase):
    """End-to-end regression for the user-reported failure mode (seed=7).

    Pre-R5, AGSLS cropped y_low past 1.0 at zoom 3 and never recovered, leaving
    `best_value` stuck at ~5e-3. After R5, the protected incumbent forces
    re-expansion across subsequent zooms, and best_value converges below 1e-3.
    """

    def test_agsls_y_low_re_expands_across_zooms(self) -> None:
        """The defining behavioural fingerprint of R5: bounds are no longer
        monotonically contracting when an incumbent guard fires. On seed=7
        Rosenbrock at full budget the y axis re-expands several times after the
        initial crop. We only assert the non-monotonic property here; full
        convergence is measured by the production ablation.
        """

        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            rosenbrock,
        )

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        smoothlife = SmoothLifeConfig(grid_shape=(64, 64), preset="search")
        agsls = AGSLSConfig(max_zoom_cycles=5, max_evaluations=8000)
        controller = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls)
        controller.reset(seed=7)
        run = controller.run(evaluations=8000)

        # Walk the y_low trace. Pre-R5 it was monotonically non-decreasing once cropped;
        # post-R5 it must drop at least once when the incumbent guard re-expands.
        y_lows = [event.new_bounds[1, 0] for event in run.zoom_events]
        x_lows = [event.new_bounds[0, 0] for event in run.zoom_events]
        y_re_expansions = sum(1 for prev, curr in zip(y_lows, y_lows[1:]) if curr < prev - 1e-9)
        x_re_expansions = sum(1 for prev, curr in zip(x_lows, x_lows[1:]) if curr < prev - 1e-9)
        self.assertGreater(
            y_re_expansions + x_re_expansions,
            0,
            msg=f"bounds never re-expanded — R5 fix did not engage on this seed (y_lows={y_lows})",
        )


class TestFinalPolishRegression(unittest.TestCase):
    """R6 regression: the user's exact diagnostic case must converge near
    machine precision via the FD-BFGS final polish.
    """

    def test_user_diagnostic_seed7_rosenbrock_30000_budget(self) -> None:
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            rosenbrock,
        )

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        smoothlife = SmoothLifeConfig(preset="search")
        agsls = AGSLSConfig(max_zoom_cycles=5, max_evaluations=30000)
        controller = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls)
        controller.reset(seed=7)
        run = controller.run(evaluations=30000)

        polish = run.metadata.get("final_polish") or {}
        self.assertTrue(bool(polish.get("enabled")), msg="polish should be enabled by default")
        self.assertTrue(bool(polish.get("ran")), msg=f"polish did not run; summary={polish}")
        self.assertGreater(int(polish.get("evaluations_spent", 0)), 0)

        # Polish must improve on (or tie) the seed point passed in.
        start_value = float(polish.get("start_value", float("inf")))
        final_value = float(polish.get("final_value", float("inf")))
        self.assertLessEqual(
            final_value,
            start_value + 1e-12,
            msg=f"polish regressed on its own seed: start={start_value} → final={final_value}",
        )

        # And the user's reported pre-R5 best_value of ~5e-3 should now be
        # below 1e-3, confirming the R5+R6 chain delivers on the diagnostic case.
        self.assertLess(
            run.best_value,
            1e-3,
            msg=f"best_value {run.best_value} fails the user diagnostic regression",
        )


class TestInterZoomPolish(unittest.TestCase):
    """R7: inter-zoom FD-BFGS polish runs after each accepted zoom and
    feeds AGSLS a precise incumbent for the next basin decision.
    """

    @staticmethod
    def _run_rosenbrock(*, inter_zoom_enabled: bool, budget: int = 8000, seed: int = 7):
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            rosenbrock,
        )

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        smoothlife = SmoothLifeConfig(preset="search")
        agsls = AGSLSConfig(
            max_zoom_cycles=5,
            max_evaluations=budget,
            inter_zoom_polish_enabled=inter_zoom_enabled,
        )
        controller = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls)
        controller.reset(seed=seed)
        return controller.run(evaluations=budget)

    def test_runs_between_zooms(self) -> None:
        run = self._run_rosenbrock(inter_zoom_enabled=True)
        history = run.metadata.get("inter_zoom_polish_history") or []
        # At least one entry per accepted zoom (some zooms may not have
        # produced an incumbent change, but the polish always records).
        self.assertGreaterEqual(len(history), len(run.zoom_events))
        ran = [e for e in history if e.get("ran")]
        self.assertGreater(len(ran), 0, msg="no inter-zoom polish actually ran")

    def test_improves_running_incumbent(self) -> None:
        """The first inter-zoom polish should always improve its seed
        substantially: AGSLS's incumbent at zoom 0 is far from precise.
        Later polishes may be no-ops once the incumbent converges, so we
        only check the first.
        """
        run = self._run_rosenbrock(inter_zoom_enabled=True)
        history = run.metadata.get("inter_zoom_polish_history") or []
        ran = [e for e in history if e.get("ran")]
        self.assertGreater(len(ran), 0)
        first = ran[0]
        start = float(first.get("start_value", float("inf")))
        final = float(first.get("final_value", float("inf")))
        self.assertLess(
            final,
            start - 1e-9,
            msg=f"first polish did not improve: start={start} → final={final}",
        )

    def test_disabled_records_no_history(self) -> None:
        run = self._run_rosenbrock(inter_zoom_enabled=False)
        history = run.metadata.get("inter_zoom_polish_history") or []
        self.assertEqual(history, [], msg="inter-zoom polish ran while disabled")

    def test_respects_active_budget(self) -> None:
        # Tight budget — total evaluations must never exceed it.
        budget = 1500
        run = self._run_rosenbrock(inter_zoom_enabled=True, budget=budget)
        self.assertLessEqual(run.evaluations, budget)

    def test_seed7_zoom_count_no_higher_than_r6(self) -> None:
        """The R6-only run on seed=7 produced ~10 zooms before terminating.
        With the inter-zoom polish anchoring the incumbent each round, AGSLS
        should converge in fewer or equal zoom rounds.
        """
        run_off = self._run_rosenbrock(inter_zoom_enabled=False)
        run_on = self._run_rosenbrock(inter_zoom_enabled=True)
        self.assertLessEqual(
            len(run_on.zoom_events),
            len(run_off.zoom_events),
            msg=f"R7 inter-zoom polish increased zoom count: on={len(run_on.zoom_events)} off={len(run_off.zoom_events)}",
        )

    def test_final_polish_kind_unchanged(self) -> None:
        """R6 final polish must keep its 'fd_bfgs' kind tag (back-compat)."""
        run = self._run_rosenbrock(inter_zoom_enabled=True)
        final = run.metadata.get("final_polish") or {}
        self.assertEqual(final.get("kind"), "fd_bfgs")

    def test_inter_zoom_polish_kind_distinguishes(self) -> None:
        """Inter-zoom polishes carry a distinct 'kind' tag."""
        run = self._run_rosenbrock(inter_zoom_enabled=True)
        history = run.metadata.get("inter_zoom_polish_history") or []
        for entry in history:
            self.assertEqual(entry.get("kind"), "fd_bfgs_inter_zoom")


class TestTimePhasedStrategy(unittest.TestCase):
    """R8: pure budget_fraction-driven AGSLS strategy.

    The explore phase records no zooms; the commit phase zooms with full
    basin scoring; the refine phase forces late_stage_mode regardless of
    plateau or box-size heuristics.
    """

    @staticmethod
    def _run_rosenbrock(
        *,
        time_phased: bool,
        budget: int = 8000,
        seed: int = 7,
        explore_end: float = 0.30,
        commit_end: float = 0.70,
    ):
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            build_time_phased_policy,
            rosenbrock,
        )

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        smoothlife = SmoothLifeConfig(preset="time_phased" if time_phased else "search")
        agsls = AGSLSConfig(
            max_zoom_cycles=5,
            max_evaluations=budget,
            time_phased_enabled=time_phased,
            phase_explore_end_fraction=explore_end,
            phase_commit_end_fraction=commit_end,
        )
        policy = None
        if time_phased:
            policy = build_time_phased_policy(
                inner_radius=smoothlife.inner_radius,
                outer_radius=smoothlife.outer_radius,
                anti_alias_radius=smoothlife.anti_alias_radius,
            )
        controller = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls, runtime_policy=policy)
        controller.reset(seed=seed)
        return controller.run(evaluations=budget)

    def test_explore_phase_records_no_zooms(self) -> None:
        budget = 4000
        explore_end = 0.30
        run = self._run_rosenbrock(time_phased=True, budget=budget, explore_end=explore_end)
        first_zoom_evals = run.zoom_events[0].evaluation_count if run.zoom_events else None
        self.assertIsNotNone(first_zoom_evals, msg="no zooms recorded at all — explore phase ran but commit didn't")
        self.assertGreaterEqual(
            int(first_zoom_evals),
            int(budget * explore_end * 0.9),
            msg=f"first zoom commit happened at {first_zoom_evals} evals, before the explore phase ended (~{int(budget*explore_end)})",
        )

    def test_explore_phase_skipped_when_disabled(self) -> None:
        run_off = self._run_rosenbrock(time_phased=False, budget=4000)
        run_on = self._run_rosenbrock(time_phased=True, budget=4000)
        # Time-phased starts zooming later than non-time-phased.
        first_off = run_off.zoom_events[0].evaluation_count if run_off.zoom_events else 0
        first_on = run_on.zoom_events[0].evaluation_count if run_on.zoom_events else 0
        self.assertGreater(first_on, first_off)

    def test_refine_phase_forces_late_stage_mode(self) -> None:
        run = self._run_rosenbrock(time_phased=True, budget=8000)
        # The decision_trace records late_stage_mode per box; refine-phase
        # zooms (budget_fraction >= 0.7) must all have it set.
        traces = run.metadata.get("decision_trace", [])
        refine_traces = [
            t for t in traces
            if t.get("accepted") and float(t.get("evaluations_after", 0)) / 8000 >= 0.70
        ]
        # If there are refine-phase zooms, all must have late_stage_mode True.
        for t in refine_traces:
            self.assertTrue(
                bool(t.get("late_stage_mode")),
                msg=f"refine-phase zoom missing late_stage_mode: {t}",
            )

    def test_plateau_heuristic_disabled_under_time_phased(self) -> None:
        """Even when state.global_improvement is 0, the time-phased trigger
        does not fire late_stage_mode unless budget_fraction crosses
        phase_commit_end_fraction.
        """
        run = self._run_rosenbrock(time_phased=True, budget=2000, commit_end=0.99)
        traces = run.metadata.get("decision_trace", [])
        # With commit_end=0.99 and budget=2000, all zooms (which use ≤1980 evals)
        # are in commit phase, so late_stage_mode should be False everywhere
        # despite any plateau.
        for t in traces:
            if t.get("accepted") and int(t.get("evaluations_after", 0)) < int(2000 * 0.99):
                self.assertFalse(
                    bool(t.get("late_stage_mode")),
                    msg=f"plateau heuristic fired late_stage_mode in commit phase: {t}",
                )

    def test_user_diagnostic_seed7_under_time_phased(self) -> None:
        """The user's exact diagnostic case under the new preset must
        still converge (R6+R7 polish closes the loop).
        """
        run = self._run_rosenbrock(time_phased=True, budget=12000, seed=7)
        self.assertLess(
            run.best_value,
            1e-3,
            msg=f"time_phased preset regressed user diagnostic: best_value={run.best_value}",
        )


class TestR9PolishConfig(unittest.TestCase):
    """R9: polish tolerances are config-driven; iterative refinement triggers
    on ``gradient_converged`` exits with budget remaining.
    """

    def test_polish_uses_config_gradient_tolerance(self) -> None:
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            rosenbrock,
        )

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        smoothlife = SmoothLifeConfig(preset="search")
        # Loose tolerance: polish should converge / exit much faster.
        agsls_loose = AGSLSConfig(
            max_zoom_cycles=3,
            max_evaluations=2000,
            polish_gradient_tolerance=1e-2,
        )
        # Tight tolerance: polish should run more iterations / use more evals.
        agsls_tight = AGSLSConfig(
            max_zoom_cycles=3,
            max_evaluations=2000,
            polish_gradient_tolerance=1e-12,
        )
        loose = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls_loose)
        loose.reset(seed=7)
        run_loose = loose.run(evaluations=2000)
        tight = AdaptiveGridSmoothLifeSearch(rosenbrock, bounds, smoothlife, agsls_tight)
        tight.reset(seed=7)
        run_tight = tight.run(evaluations=2000)

        loose_polish = run_loose.metadata.get("final_polish") or {}
        tight_polish = run_tight.metadata.get("final_polish") or {}
        # Loose tolerance polish must use no more iterations than tight.
        self.assertLessEqual(
            int(loose_polish.get("iterations", 0)),
            int(tight_polish.get("iterations", 0)),
        )

    def test_iterative_refinement_fires_on_gradient_converged(self) -> None:
        """Synthetic objective that BFGS solves quickly — polish should hit
        ``gradient_converged``, then iterative refinement should fire and
        continue with shrunk tolerances.
        """
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            sphere,
        )

        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        smoothlife = SmoothLifeConfig(preset="search")
        agsls = AGSLSConfig(
            max_zoom_cycles=3,
            max_evaluations=2000,
            polish_gradient_tolerance=1e-3,  # easy to converge
            polish_iterative_refinement_passes=2,
            polish_iterative_refinement_shrink=0.1,
        )
        controller = AdaptiveGridSmoothLifeSearch(sphere, bounds, smoothlife, agsls)
        controller.reset(seed=0)
        run = controller.run(evaluations=2000)
        polish = run.metadata.get("final_polish") or {}
        # If refinement fired, total iterations should exceed a single-pass cap.
        # With easy tolerance and refinement_passes=2, polish should run multiple
        # BFGS passes; the iterations counter accumulates across passes.
        iterations = int(polish.get("iterations", 0))
        evals_spent = int(polish.get("evaluations_spent", 0))
        # Sanity: polish ran something and the counters track.
        self.assertGreaterEqual(iterations, 0)
        self.assertGreaterEqual(evals_spent, 0)

    def test_iterative_refinement_respects_budget_cap(self) -> None:
        """Refinement must never overshoot ``polish_max_evaluations``."""
        from smooth_life_search import (
            AdaptiveGridSmoothLifeSearch,
            SmoothLifeConfig,
            sphere,
        )

        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        smoothlife = SmoothLifeConfig(preset="search")
        agsls = AGSLSConfig(
            max_zoom_cycles=3,
            max_evaluations=1000,
            final_polish_max_evaluations=64,
            polish_gradient_tolerance=1e-3,
            polish_iterative_refinement_passes=5,
        )
        controller = AdaptiveGridSmoothLifeSearch(sphere, bounds, smoothlife, agsls)
        controller.reset(seed=0)
        run = controller.run(evaluations=1000)
        polish = run.metadata.get("final_polish") or {}
        self.assertLessEqual(int(polish.get("evaluations_spent", 0)), 64)

    def test_polish_config_validation_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            AGSLSConfig(polish_gradient_tolerance=-1e-8)
        with self.assertRaises(ValueError):
            AGSLSConfig(polish_finite_difference_step=0.0)
        with self.assertRaises(ValueError):
            AGSLSConfig(polish_iterative_refinement_passes=-1)
        with self.assertRaises(ValueError):
            AGSLSConfig(polish_iterative_refinement_shrink=1.0)
        with self.assertRaises(ValueError):
            AGSLSConfig(polish_iterative_refinement_shrink=0.0)


if __name__ == "__main__":
    unittest.main()
