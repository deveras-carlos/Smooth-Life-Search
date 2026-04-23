"""Convert simulator snapshots into dashboard images."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from ..results import SearchRun, SmoothLifeSnapshot
from .overlays import draw_bbox, draw_point_marker, draw_world_path, world_to_image_xy

_PANEL_GAP = 16
_PANEL_MARGIN = 18
_HEADER_HEIGHT = 54
_FOOTER_HEIGHT = 88
_BACKGROUND = ( 7, 9, 12 )
_PANEL_BACKGROUND = ( 16, 20, 28 )
_PANEL_BORDER = ( 58, 66, 84 )
_TEXT_PRIMARY = ( 236, 236, 238 )
_TEXT_SECONDARY = ( 172, 180, 190 )
_HIGHLIGHT = ( 255, 184, 48 )
_GLOBAL_COLOR = ( 60, 255, 190 )
_LOCAL_COLOR = ( 96, 192, 255 )
_BOX_COLOR = ( 255, 150, 64 )
_PATH_COLOR = ( 255, 255, 255 )
_ZOOM_BOX = ( 255, 104, 92 )
_DEFERRED = ( 160, 160, 160 )
_UNEXPLORED = ( 30, 34, 40 )

_PALETTES: dict[ str, tuple[ tuple[ int, int, int ], ... ] ] = {
    "signed": (
        ( 9, 23, 57 ),
        ( 55, 94, 151 ),
        ( 208, 224, 246 ),
        ( 246, 212, 168 ),
        ( 176, 79, 44 ),
    ),
    "objective": (
        ( 12, 9, 32 ),
        ( 71, 18, 99 ),
        ( 164, 44, 122 ),
        ( 239, 115, 62 ),
        ( 252, 232, 126 ),
    ),
    "support": (
        ( 5, 11, 24 ),
        ( 29, 59, 96 ),
        ( 84, 141, 154 ),
        ( 217, 179, 95 ),
        ( 255, 240, 182 ),
    ),
    "vitality": (
        ( 7, 18, 26 ),
        ( 26, 88, 83 ),
        ( 88, 171, 145 ),
        ( 205, 239, 198 ),
    ),
}


def _normalize_to_unit( values: np.ndarray ) -> np.ndarray:
    arr = np.asarray( values, dtype=float )
    finite = np.isfinite( arr )
    if not np.any( finite ):
        return np.zeros_like( arr, dtype=float )
    clipped = arr.copy()
    clipped[ ~finite ] = 0.0
    minimum = float( np.min( clipped[ finite ] ) )
    maximum = float( np.max( clipped[ finite ] ) )
    span = max( maximum - minimum, 1e-12 )
    normalized = ( clipped - minimum ) / span
    return np.clip( normalized, 0.0, 1.0 )


def _normalize_signed( values: np.ndarray ) -> np.ndarray:
    arr = np.asarray( values, dtype=float )
    normalized = 0.5 * ( np.clip( arr, -1.0, 1.0 ) + 1.0 )
    return np.clip( normalized, 0.0, 1.0 )


@lru_cache( maxsize=None )
def _palette_table( name: str ) -> np.ndarray:
    stops = _PALETTES[ name ]
    positions = np.linspace( 0.0, 1.0, len( stops ) )
    ramp = np.linspace( 0.0, 1.0, 256 )
    palette = np.empty( ( 256, 3 ), dtype=np.uint8 )
    for channel in range( 3 ):
        palette[ :, channel ] = np.round(
            np.interp( ramp, positions, [ color[ channel ] for color in stops ] )
        ).astype( np.uint8 )
    return palette


def _palette_image( values: np.ndarray, name: str, *, signed: bool = False ) -> Image.Image:
    normalized = _normalize_signed( values ) if signed else _normalize_to_unit( values )
    indices = np.round( normalized * 255.0 ).astype( np.uint8 )
    rgb = _palette_table( name )[ indices ]
    return Image.fromarray( rgb, mode="RGB" )


def _masked_palette_image(
    values: np.ndarray,
    name: str,
    valid_mask: np.ndarray,
    masked_color: tuple[ int, int, int ],
) -> Image.Image:
    image = _palette_image( values, name )
    rgb = np.asarray( image ).copy()
    rgb[ ~np.asarray( valid_mask, dtype=bool ) ] = np.asarray( masked_color, dtype=np.uint8 )
    return Image.fromarray( rgb, mode="RGB" )


def _resized_panel( image: Image.Image, scale: int ) -> Image.Image:
    return image.resize(
        ( image.size[ 0 ] * scale, image.size[ 1 ] * scale ),
        Image.Resampling.NEAREST,
    )


def _panel_box( panel_size: tuple[ int, int ], row: int, col: int ) -> tuple[ int, int, int, int ]:
    panel_width, panel_height = panel_size
    x0 = _PANEL_MARGIN + col * ( panel_width + _PANEL_GAP )
    y0 = _HEADER_HEIGHT + _PANEL_MARGIN + row * ( panel_height + _PANEL_GAP )
    return x0, y0, x0 + panel_width, y0 + panel_height


def _panel_title( draw: ImageDraw.ImageDraw, box: tuple[ int, int, int, int ], title: str ) -> None:
    x0, y0, _, _ = box
    draw.text( ( x0, y0 - 20 ), title, fill=_TEXT_PRIMARY )


def _support_field( snapshot: SmoothLifeSnapshot ) -> np.ndarray:
    return np.clip( 1.0 - np.abs( snapshot.field ), 0.0, 1.0 ) * snapshot.objective_field


def _local_best_history_points( run: SearchRun | None, frame_index: int ) -> list[ np.ndarray ]:
    if run is None:
        return [ ]
    return [ previous.local_best_point.copy() for previous in run.snapshots[ : frame_index + 1 ] ]


def _plot_transform( values: np.ndarray ) -> np.ndarray:
    if np.all( values >= 0.0 ):
        return np.log1p( values )
    return np.sign( values ) * np.log1p( np.abs( values ) )


def _convergence_panel( run: SearchRun | None, frame_index: int, panel_size: tuple[ int, int ] ) -> Image.Image:
    panel_width, panel_height = panel_size
    image = Image.new( "RGB", panel_size, _PANEL_BACKGROUND )
    draw = ImageDraw.Draw( image )
    left = max( 8, int( round( panel_width * 0.16 ) ) )
    top = 14
    right = max( left + 8, panel_width - 12 )
    bottom = max( top + 8, panel_height - 24 )
    draw.rectangle( ( left, top, right, bottom ), outline=_PANEL_BORDER, width=1 )

    if run is None or not run.snapshots:
        draw.text( ( 16, 16 ), "no run history", fill=_TEXT_PRIMARY )
        return image

    x_values = np.asarray(
        [ snapshot.metadata.get( "evaluations", snapshot.step_index ) for snapshot in run.snapshots ],
        dtype=float,
    )
    x_label = "evaluations"
    if len( np.unique( x_values ) ) <= 2:
        x_values = np.asarray( [ snapshot.step_index for snapshot in run.snapshots ], dtype=float )
        x_label = "step"
    global_values = np.asarray( [ snapshot.best_value for snapshot in run.snapshots ], dtype=float )
    local_values = np.asarray( [ snapshot.local_best_value for snapshot in run.snapshots ], dtype=float )
    box_values = np.asarray( [ snapshot.box_best_value for snapshot in run.snapshots ], dtype=float )
    transformed_global = _plot_transform( global_values )
    transformed_local = _plot_transform( local_values )
    transformed_box = _plot_transform( box_values )
    x_min = float( np.min( x_values ) )
    x_max = float( np.max( x_values ) )
    if abs( x_max - x_min ) < 1e-12:
        x_max = x_min + 1.0
    transformed_all = np.concatenate( [ transformed_global, transformed_local, transformed_box ] )
    y_min = float( np.min( transformed_all ) )
    y_max = float( np.max( transformed_all ) )
    if abs( y_max - y_min ) < 1e-12:
        y_max = y_min + 1.0

    for step in range( 5 ):
        frac = step / 4.0
        x = left + frac * ( right - left )
        y = top + frac * ( bottom - top )
        draw.line( ( x, top, x, bottom ), fill=( 38, 44, 57 ), width=1 )
        draw.line( ( left, y, right, y ), fill=( 38, 44, 57 ), width=1 )

    def build_line( series: np.ndarray ) -> list[ tuple[ float, float ] ]:
        points: list[ tuple[ float, float ] ] = [ ]
        for x_value, y_value in zip( x_values, series ):
            x = left + ( x_value - x_min ) / ( x_max - x_min ) * ( right - left )
            y = bottom - ( y_value - y_min ) / ( y_max - y_min ) * ( bottom - top )
            points.append( ( float( x ), float( y ) ) )
        return points

    global_line = build_line( transformed_global )
    local_line = build_line( transformed_local )
    box_line = build_line( transformed_box )
    if len( global_line ) >= 2:
        draw.line( global_line, fill=_GLOBAL_COLOR, width=3 )
    if len( local_line ) >= 2:
        draw.line( local_line, fill=_LOCAL_COLOR, width=2 )
    if len( box_line ) >= 2:
        draw.line( box_line, fill=_BOX_COLOR, width=2 )

    current = max( 0, min( frame_index, len( global_line ) - 1 ) )
    for point, color, radius in (
        ( global_line[ current ], _GLOBAL_COLOR, 4 ),
        ( local_line[ current ], _LOCAL_COLOR, 3 ),
        ( box_line[ current ], _BOX_COLOR, 3 ),
    ):
        x_value, y_value = point
        draw.ellipse(
            ( x_value - radius, y_value - radius, x_value + radius, y_value + radius ),
            fill=color,
            outline=( 10, 10, 10 ),
        )

    for index, snapshot in enumerate( run.snapshots ):
        if index > frame_index:
            continue
        decision = str( snapshot.metadata.get( "zoom_decision", "" ) )
        if not decision:
            continue
        x_value = x_values[ index ]
        x = left + ( x_value - x_min ) / ( x_max - x_min ) * ( right - left )
        color = _ZOOM_BOX if decision == "accepted" else _DEFERRED
        draw.line( ( x, top, x, bottom ), fill=color, width=1 )

    draw.text( ( left, panel_height - 20 ), x_label, fill=_TEXT_SECONDARY )
    draw.text( ( 10, 8 ), "best value", fill=_TEXT_SECONDARY )
    draw.text(
        ( left, top + 4 ),
        "global / local / box",
        fill=_TEXT_PRIMARY,
    )
    return image


def _footer_text( snapshot: SmoothLifeSnapshot, frame_index: int ) -> str:
    bounds = snapshot.bounds
    evals = int( snapshot.metadata.get( "evaluations", 0 ) )
    explored_fraction = float( snapshot.metadata.get( "explored_fraction", 0.0 ) )
    decision = str( snapshot.metadata.get( "zoom_decision", "none" ) )
    best_text = format( snapshot.best_value, ".6f" )
    local_text = format( snapshot.local_best_value, ".6f" )
    box_text = format( snapshot.box_best_value, ".6f" )
    explored_text = format( explored_fraction, ".3f" )
    x0_text = format( bounds[ 0, 0 ], ".4f" )
    x1_text = format( bounds[ 0, 1 ], ".4f" )
    y0_text = format( bounds[ 1, 0 ], ".4f" )
    y1_text = format( bounds[ 1, 1 ], ".4f" )
    return (
        f"frame { frame_index + 1 }   step={ snapshot.step_index }   evals={ evals }   explored={ explored_text }   decision={ decision }\n"
        f"global best={ best_text }   local best={ local_text }   box best={ box_text }    "
        f"x = [ { x0_text }, { x1_text } ]    y = [ { y0_text }, { y1_text } ]"
    )


def _draw_group_overlays( draw: ImageDraw.ImageDraw, snapshot: SmoothLifeSnapshot, image_size: tuple[ int, int ] ) -> None:
    dense_groups = snapshot.metadata.get( "dense_groups", [ ] )
    for entry in dense_groups[ : 3 ]:
        bbox_world = np.asarray( entry.get( "bbox", [ ] ), dtype=float )
        if bbox_world.shape != ( 2, 2 ):
            continue
        draw_bbox( draw, bbox_world, snapshot.bounds, image_size, color=_ZOOM_BOX )
        x, y = world_to_image_xy( np.asarray( [ bbox_world[ 0, 0 ], bbox_world[ 1, 0 ] ], dtype=float ), snapshot.bounds, image_size )
        rank_text = int( round( float( entry.get( "rank", 0.0 ) ) ) )
        density_text = format( float( entry.get( "alive_density", 0.0 ) ), ".2f" )
        draw.text( ( x + 4, y + 4 ), f"#{ rank_text } d={ density_text }", fill=_TEXT_PRIMARY )


def snapshot_to_image(
    snapshot: SmoothLifeSnapshot,
    scale: int = 2,
    *,
    run: SearchRun | None = None,
    frame_index: int | None = None,
) -> Image.Image:
    """Render one snapshot as a multi-panel dashboard image."""

    resolved_index = 0 if frame_index is None else int( frame_index )
    field_panel = _resized_panel( _palette_image( snapshot.field, "signed", signed=True ), scale )
    objective_panel = _resized_panel(
        _masked_palette_image( snapshot.objective_field, "objective", snapshot.evaluated_mask, _UNEXPLORED ),
        scale,
    )
    support_panel = _resized_panel( _palette_image( _support_field( snapshot ), "support" ), scale )
    transition_panel = _resized_panel( _palette_image( snapshot.transition_field, "signed", signed=True ), scale )
    vitality_panel = _resized_panel( _palette_image( np.clip( 1.0 - np.abs( snapshot.field ), 0.0, 1.0 ), "vitality" ), scale )

    history_points = _local_best_history_points( run, resolved_index )
    field_draw = ImageDraw.Draw( field_panel )
    draw_world_path(
        field_draw,
        history_points,
        snapshot.bounds,
        field_panel.size,
        color=_PATH_COLOR,
        width=2,
    )
    draw_point_marker(
        field_draw,
        snapshot.local_best_point,
        snapshot.bounds,
        field_panel.size,
        color=_LOCAL_COLOR,
        radius=5,
    )
    draw_point_marker(
        field_draw,
        snapshot.box_best_point,
        snapshot.bounds,
        field_panel.size,
        color=_BOX_COLOR,
        radius=4,
    )
    if snapshot.selected_basin_bbox is not None:
        draw_bbox( field_draw, snapshot.selected_basin_bbox, snapshot.bounds, field_panel.size, color=_ZOOM_BOX )

    support_draw = ImageDraw.Draw( support_panel )
    _draw_group_overlays( support_draw, snapshot, support_panel.size )
    draw_point_marker(
        support_draw,
        snapshot.best_point,
        snapshot.bounds,
        support_panel.size,
        color=_GLOBAL_COLOR,
        radius=5,
    )

    convergence_panel = _convergence_panel( run, resolved_index, field_panel.size )

    panel_size = field_panel.size
    canvas_width = _PANEL_MARGIN * 2 + panel_size[ 0 ] * 3 + _PANEL_GAP * 2
    canvas_height = _HEADER_HEIGHT + _PANEL_MARGIN + panel_size[ 1 ] * 2 + _PANEL_GAP + _FOOTER_HEIGHT
    canvas = Image.new( "RGB", ( canvas_width, canvas_height ), _BACKGROUND )
    draw = ImageDraw.Draw( canvas )
    font = ImageFont.load_default()

    title = "SmoothLife dashboard"
    if run is not None:
        mode = run.metadata.get( "mode" )
        objective = run.metadata.get( "objective" )
        parts = [ "SmoothLife dashboard" ]
        if mode:
            parts.append( str( mode ) )
        if objective:
            parts.append( str( objective ) )
        title = "  |  ".join( parts )
    draw.text( ( _PANEL_MARGIN, 14 ), title, fill=_TEXT_PRIMARY, font=font )
    draw.text(
        ( _PANEL_MARGIN, 32 ),
        "Signed field, explored objective values, dense support groups, and convergence are shown together.",
        fill=_TEXT_SECONDARY,
        font=font,
    )

    panel_specs = [
        ( field_panel, "signed field + accepted path", 0, 0 ),
        ( objective_panel, "explored objective desirability", 0, 1 ),
        ( support_panel, "support + dense groups", 0, 2 ),
        ( transition_panel, "transition target", 1, 0 ),
        ( vitality_panel, "vitality", 1, 1 ),
        ( convergence_panel, "convergence", 1, 2 ),
    ]

    for panel_image, panel_title, row, col in panel_specs:
        box = _panel_box( panel_size, row, col )
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(
            ( x0 - 4, y0 - 4, x1 + 4, y1 + 4 ),
            radius=10,
            fill=_PANEL_BACKGROUND,
            outline=_PANEL_BORDER,
            width=1,
        )
        canvas.paste( panel_image, ( x0, y0 ) )
        _panel_title( draw, box, panel_title )

    footer_top = canvas_height - _FOOTER_HEIGHT + 10
    draw.text( ( _PANEL_MARGIN, footer_top ), _footer_text( snapshot, resolved_index ), fill=_TEXT_PRIMARY, font=font )
    draw.text(
        ( _PANEL_MARGIN, footer_top + 34 ),
        "Unexplored pixels are dark in the objective panel. Red boxes show dense groups; gray convergence markers indicate deferred zoom decisions.",
        fill=_TEXT_SECONDARY,
        font=font,
    )
    return canvas
