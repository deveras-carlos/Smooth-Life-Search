"""Reusable dashboard panel builders."""

from __future__ import annotations

from PIL import Image, ImageDraw

import numpy as np

from ..core import SearchRun, SmoothLifeSnapshot

PANEL_BACKGROUND = (16, 20, 28)
PANEL_BORDER = (58, 66, 84)
TEXT_PRIMARY = (236, 236, 238)
TEXT_SECONDARY = (172, 180, 190)
GLOBAL_COLOR = (60, 255, 190)
LOCAL_COLOR = (96, 192, 255)
BOX_COLOR = (255, 150, 64)
ZOOM_BOX = (255, 104, 92)
DEFERRED = (160, 160, 160)


def support_field(snapshot: SmoothLifeSnapshot) -> np.ndarray:
    """Return the displayed support field for one snapshot."""

    if str(snapshot.metadata.get("mode", "")) == "point-cloud" and hasattr(snapshot, "density_field"):
        return np.asarray(getattr(snapshot, "density_field"), dtype=float)
    return np.clip(1.0 - np.abs(snapshot.field), 0.0, 1.0) * snapshot.objective_field


def _plot_transform(values: np.ndarray) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros_like(values, dtype=float)
    fill = float(finite_values[0])
    cleaned = np.where(np.isfinite(values), values, fill)
    if np.all(cleaned >= 0.0):
        return np.log1p(cleaned)
    return np.sign(cleaned) * np.log1p(np.abs(cleaned))


def convergence_panel(run: SearchRun | None, frame_index: int, panel_size: tuple[int, int]) -> Image.Image:
    """Render the run-history convergence plot."""

    panel_width, panel_height = panel_size
    image = Image.new("RGB", panel_size, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(image)
    left = max(8, int(round(panel_width * 0.16)))
    top = 14
    right = max(left + 8, panel_width - 12)
    bottom = max(top + 8, panel_height - 24)
    draw.rectangle((left, top, right, bottom), outline=PANEL_BORDER, width=1)

    if run is None or not run.snapshots:
        draw.text((16, 16), "no run history", fill=TEXT_PRIMARY)
        return image

    x_values = np.asarray(
        [snapshot.metadata.get("evaluations", snapshot.step_index) for snapshot in run.snapshots],
        dtype=float,
    )
    x_label = "evaluations"
    if len(np.unique(x_values)) <= 2:
        x_values = np.asarray([snapshot.step_index for snapshot in run.snapshots], dtype=float)
        x_label = "step"
    series = [
        np.asarray([snapshot.best_value for snapshot in run.snapshots], dtype=float),
        np.asarray([snapshot.local_best_value for snapshot in run.snapshots], dtype=float),
        np.asarray([snapshot.box_best_value for snapshot in run.snapshots], dtype=float),
    ]
    transformed = [_plot_transform(values) for values in series]
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    transformed_all = np.concatenate(transformed)
    y_min = float(np.min(transformed_all))
    y_max = float(np.max(transformed_all))
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    for step in range(5):
        frac = step / 4.0
        x = left + frac * (right - left)
        y = top + frac * (bottom - top)
        draw.line((x, top, x, bottom), fill=(38, 44, 57), width=1)
        draw.line((left, y, right, y), fill=(38, 44, 57), width=1)

    def build_line(values: np.ndarray) -> list[tuple[float, float]]:
        return [
            (
                float(left + (x_value - x_min) / (x_max - x_min) * (right - left)),
                float(bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)),
            )
            for x_value, y_value in zip(x_values, values)
        ]

    colors = (GLOBAL_COLOR, LOCAL_COLOR, BOX_COLOR)
    widths = (3, 2, 2)
    lines = [build_line(values) for values in transformed]
    for line, color, width in zip(lines, colors, widths):
        if len(line) >= 2:
            draw.line(line, fill=color, width=width)

    current = max(0, min(frame_index, len(lines[0]) - 1))
    for point, color, radius in (
        (lines[0][current], GLOBAL_COLOR, 4),
        (lines[1][current], LOCAL_COLOR, 3),
        (lines[2][current], BOX_COLOR, 3),
    ):
        x_value, y_value = point
        draw.ellipse((x_value - radius, y_value - radius, x_value + radius, y_value + radius), fill=color, outline=(10, 10, 10))

    for index, snapshot in enumerate(run.snapshots):
        if index > frame_index:
            continue
        decision = str(snapshot.metadata.get("zoom_decision", ""))
        if not decision:
            continue
        x_value = x_values[index]
        x = left + (x_value - x_min) / (x_max - x_min) * (right - left)
        color = ZOOM_BOX if decision == "accepted" else DEFERRED
        draw.line((x, top, x, bottom), fill=color, width=1)

    draw.text((left, panel_height - 20), x_label, fill=TEXT_SECONDARY)
    draw.text((10, 8), "best value", fill=TEXT_SECONDARY)
    draw.text((left, top + 4), "global / local / box", fill=TEXT_PRIMARY)
    return image
