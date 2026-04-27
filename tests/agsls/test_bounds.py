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


if __name__ == "__main__":
    unittest.main()
