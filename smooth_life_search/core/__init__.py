"""Shared core types and helpers."""

from .bounds import BoundsLike, bounds_area, normalize_bounds_2d, point_in_bounds
from .models import Basin, Bounds2D, Objective, SearchResult, SearchRun, SearchRunner, SmoothLifeSnapshot, ZoomEvent

__all__ = [
    "Basin",
    "Bounds2D",
    "BoundsLike",
    "Objective",
    "SearchResult",
    "SearchRun",
    "SearchRunner",
    "SmoothLifeSnapshot",
    "ZoomEvent",
    "bounds_area",
    "normalize_bounds_2d",
    "point_in_bounds",
]
