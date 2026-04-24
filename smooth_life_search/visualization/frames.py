"""Convert simulator snapshots into dashboard images."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from ..core import SearchRun, SmoothLifeSnapshot
from .overlays import draw_bbox, draw_point_marker, draw_world_path, world_to_image_xy
from .palettes import palette_table
from .panels import convergence_panel as build_convergence_panel
from .panels import support_field

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


def _palette_image( values: np.ndarray, name: str, *, signed: bool = False ) -> Image.Image:
    normalized = _normalize_signed( values ) if signed else _normalize_to_unit( values )
    indices = np.round( normalized * 255.0 ).astype( np.uint8 )
    rgb = palette_table( name )[ indices ]
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
    return support_field( snapshot )


def _local_best_history_points( run: SearchRun | None, frame_index: int ) -> list[ np.ndarray ]:
    if run is None:
        return [ ]
    return [ previous.local_best_point.copy() for previous in run.snapshots[ : frame_index + 1 ] ]


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

    convergence_panel = build_convergence_panel( run, resolved_index, field_panel.size )

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
