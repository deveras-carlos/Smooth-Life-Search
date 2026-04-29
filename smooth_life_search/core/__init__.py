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
    build_ema_alpha_ramp_policy,
    build_gamma_ramp_policy,
    build_kernel_shrink_policy,
    build_time_phased_policy,
    combine_schedule_policies,
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
    "build_ema_alpha_ramp_policy",
    "build_gamma_ramp_policy",
    "build_kernel_shrink_policy",
    "build_time_phased_policy",
    "combine_schedule_policies",
    "normalize_bounds_2d",
    "point_in_bounds",
]
