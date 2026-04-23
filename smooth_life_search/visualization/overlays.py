"""Overlay helpers for visualization."""

from __future__ import annotations

from typing import Iterable

from PIL import ImageDraw

import numpy as np


def world_to_image_xy(
    point_world: np.ndarray,
    bounds: np.ndarray,
    image_size: tuple[ int, int ],
) -> tuple[ float, float ]:
    """Project a 2D world point into image coordinates."""

    width, height = image_size
    x = (
        ( point_world[ 0 ] - bounds[ 0, 0 ] )
        / max( bounds[ 0, 1 ] - bounds[ 0, 0 ], 1e-12 )
        * width
    )
    y = (
        ( point_world[ 1 ] - bounds[ 1, 0 ] )
        / max( bounds[ 1, 1 ] - bounds[ 1, 0 ], 1e-12 )
        * height
    )
    return float( x ), float( y )


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox_world: np.ndarray,
    bounds: np.ndarray,
    image_size: tuple[ int, int ],
    color: tuple[ int, int, int ],
) -> None:
    """Draw a world-coordinate bounding box on a rendered frame."""

    x0, y0 = world_to_image_xy( bbox_world[ :, 0 ], bounds, image_size )
    x1, y1 = world_to_image_xy( bbox_world[ :, 1 ], bounds, image_size )
    draw.rectangle( ( x0, y0, x1, y1 ), outline=color, width=2 )


def draw_point_marker(
    draw: ImageDraw.ImageDraw,
    point_world: np.ndarray,
    bounds: np.ndarray,
    image_size: tuple[ int, int ],
    *,
    color: tuple[ int, int, int ],
    radius: int = 5,
) -> None:
    """Draw a point marker as a circle with a crosshair."""

    x, y = world_to_image_xy( point_world, bounds, image_size )
    draw.ellipse(
        ( x - radius, y - radius, x + radius, y + radius ),
        outline=color,
        width=2,
    )
    draw.line( ( x - radius - 3, y, x + radius + 3, y ), fill=color, width=2 )
    draw.line( ( x, y - radius - 3, x, y + radius + 3 ), fill=color, width=2 )


def draw_world_path(
    draw: ImageDraw.ImageDraw,
    points_world: Iterable[ np.ndarray ],
    bounds: np.ndarray,
    image_size: tuple[ int, int ],
    *,
    color: tuple[ int, int, int ],
    width: int = 2,
) -> None:
    """Draw a path through world coordinates that fall inside the current bounds."""

    projected: list[ tuple[ float, float ] ] = []
    for point in points_world:
        arr = np.asarray( point, dtype=float )
        if np.any( arr < bounds[ :, 0 ] ) or np.any( arr > bounds[ :, 1 ] ):
            continue
        projected.append( world_to_image_xy( arr, bounds, image_size ) )
    if len( projected ) >= 2:
        draw.line( projected, fill=color, width=width )
