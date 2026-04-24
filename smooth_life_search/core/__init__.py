"""Shared core types and helpers."""

from .bounds import BoundsLike, bounds_area, normalize_bounds_2d, point_in_bounds
from .models import Basin, Bounds2D, Objective, SearchResult, SearchRun, SearchRunner, SmoothLifeSnapshot, ZoomEvent
from .scheduling import (
    AGSLS_PER_DECISION_FIELDS,
    KERNEL_PARAMETER_FIELDS,
    RUN_CONSTANT_FIELDS,
    SMOOTHLIFE_PER_STEP_FIELDS,
    ZOOM_BOUNDARY_FIELDS,
    FieldSchedule,
    RuntimeSignals,
    SchedulePolicy,
    build_kernel_shrink_policy,
)

__all__ = [
    "AGSLS_PER_DECISION_FIELDS",
    "Basin",
    "Bounds2D",
    "BoundsLike",
    "FieldSchedule",
    "KERNEL_PARAMETER_FIELDS",
    "Objective",
    "RUN_CONSTANT_FIELDS",
    "RuntimeSignals",
    "SMOOTHLIFE_PER_STEP_FIELDS",
    "SearchResult",
    "SearchRun",
    "SearchRunner",
    "SchedulePolicy",
    "SmoothLifeSnapshot",
    "ZOOM_BOUNDARY_FIELDS",
    "ZoomEvent",
    "bounds_area",
    "build_kernel_shrink_policy",
    "normalize_bounds_2d",
    "point_in_bounds",
]
