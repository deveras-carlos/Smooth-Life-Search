from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import RenderOptions
from smooth_life_search.core import SearchRun, SmoothLifeSnapshot
from smooth_life_search.visualization import render_run_frames, snapshot_to_image


def _snapshot() -> SmoothLifeSnapshot:
    shape = (16, 16)
    field = np.linspace(-1.0, 1.0, shape[0] * shape[1], dtype=float).reshape(shape)
    objective = np.flipud(np.linspace(0.0, 1.0, shape[0] * shape[1], dtype=float).reshape(shape))
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    return SmoothLifeSnapshot(
        step_index=3,
        bounds=bounds,
        field=field,
        inner_fill=np.zeros(shape, dtype=float),
        outer_fill=np.ones(shape, dtype=float) * 0.25,
        objective_field=objective,
        transition_field=-field,
        evaluated_mask=np.ones(shape, dtype=bool),
        best_point=np.asarray([0.0, 0.0], dtype=float),
        best_value=0.0,
        local_best_point=np.asarray([0.2, 0.2], dtype=float),
        local_best_value=0.1,
        box_best_point=np.asarray([-0.2, -0.2], dtype=float),
        box_best_value=0.2,
        metadata={"evaluations": 16, "explored_fraction": 1.0},
    )


class TestVisualizationRendering(unittest.TestCase):
    def test_snapshot_render_is_deterministic_size_and_nonblank(self) -> None:
        image = snapshot_to_image(_snapshot(), scale=1)
        pixels = np.asarray(image, dtype=np.uint8)
        self.assertEqual(image.size, (116, 208))
        self.assertGreater(int(np.max(pixels) - np.min(pixels)), 0)

    def test_render_options_validate_and_drive_run_frames(self) -> None:
        with self.assertRaises(ValueError):
            RenderOptions(scale=0)
        snapshot = _snapshot()
        run = SearchRun(
            best_point=snapshot.best_point,
            best_value=snapshot.best_value,
            evaluations=16,
            bounds=snapshot.bounds,
            snapshots=[snapshot],
            zoom_events=[],
            metadata={"mode": "test"},
        )
        frame = render_run_frames(run, options=RenderOptions(scale=2))[0]
        self.assertEqual(frame.size, (164, 240))


if __name__ == "__main__":
    unittest.main()
