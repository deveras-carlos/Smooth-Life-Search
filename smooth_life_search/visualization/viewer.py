"""Tk viewer for SmoothLife and AGSLS animations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ..core import SearchRun, SmoothLifeSnapshot
from .frames import snapshot_to_image
from .options import RenderOptions


def _viewer_title( run: SearchRun, base_title: str ) -> str:
    objective = run.metadata.get( "objective" )
    mode = run.metadata.get( "mode" )
    parts = [ base_title ]
    if mode:
        parts.append( str( mode ) )
    if objective:
        parts.append( str( objective ) )
    return " | ".join( parts )


def _snapshot_summary(
    snapshot: SmoothLifeSnapshot,
    frame_index: int,
    frame_count: int,
) -> str:
    bounds = snapshot.bounds
    evaluations = int( snapshot.metadata.get( "evaluations", 0 ) )
    explored_fraction = float( snapshot.metadata.get( "explored_fraction", 0.0 ) )
    decision = str( snapshot.metadata.get( "zoom_decision", "none" ) )
    best_text = format( snapshot.best_value, ".6f" )
    local_best_text = format( snapshot.local_best_value, ".6f" )
    box_best_text = format( snapshot.box_best_value, ".6f" )
    explored_text = format( explored_fraction, ".3f" )
    x0_text = format( bounds[ 0, 0 ], ".4f" )
    x1_text = format( bounds[ 0, 1 ], ".4f" )
    y0_text = format( bounds[ 1, 0 ], ".4f" )
    y1_text = format( bounds[ 1, 1 ], ".4f" )
    return (
        f"frame { frame_index + 1 } / { frame_count }    "
        f"step = { snapshot.step_index }    evals = { evaluations }    "
        f"explored = { explored_text }    decision = { decision }\n"
        f"global best = { best_text }    local best = { local_best_text }    box best = { box_best_text }\n"
        f"x = [ { x0_text }, { x1_text } ]    "
        f"y = [ { y0_text }, { y1_text } ]    "
        f"shortcuts: space play/pause, left/right step, home/end jump"
    )


def render_run_frames( run: SearchRun, scale: int = 2, *, options: RenderOptions | None = None ) -> list[ Image.Image ]:
    """Render all snapshots in a run into dashboard frames."""

    if not run.snapshots:
        raise ValueError( "run must contain at least one snapshot" )
    resolved = options or RenderOptions( scale=scale )
    return [
        snapshot_to_image( snapshot, scale=resolved.scale, run=run, frame_index=index )
        for index, snapshot in enumerate( run.snapshots )
    ]


def open_run_viewer(
    run: SearchRun,
    *,
    scale: int = 2,
    duration_ms: int = 90,
    options: RenderOptions | None = None,
    title: str = "SmoothLife Animation",
) -> None:
    """Open a GUI window that plays a SearchRun animation."""

    resolved = options or RenderOptions( scale=scale, duration_ms=duration_ms )
    frames = render_run_frames( run, options=resolved )
    snapshots = run.snapshots

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError( "unable to open GUI viewer; no display is available" ) from exc

    root.title( _viewer_title( run, title ) )

    state = {
        "index": 0,
        "playing": True,
        "after_id": None,
        "duration_ms": max( 10, int( resolved.duration_ms ) ),
        "photo": None,
    }

    container = ttk.Frame( root, padding=10 )
    container.grid( row=0, column=0, sticky="nsew" )
    root.columnconfigure( 0, weight=1 )
    root.rowconfigure( 0, weight=1 )
    container.columnconfigure( 0, weight=1 )
    container.rowconfigure( 0, weight=1 )

    image_label = ttk.Label( container )
    image_label.grid( row=0, column=0, columnspan=7, sticky="nsew" )

    info_var = tk.StringVar()
    info_label = ttk.Label( container, textvariable=info_var, justify="left" )
    info_label.grid( row=1, column=0, columnspan=7, sticky="w", pady=( 8, 6 ) )

    frame_var = tk.IntVar( value=0 )

    def update_frame( index: int ) -> None:
        clamped = max( 0, min( len( frames ) - 1, int( index ) ) )
        state[ "index" ] = clamped
        frame_var.set( clamped )
        photo = ImageTk.PhotoImage( frames[ clamped ] )
        state[ "photo" ] = photo
        image_label.configure( image=photo )
        info_var.set( _snapshot_summary( snapshots[ clamped ], clamped, len( frames ) ) )

    def cancel_after() -> None:
        after_id = state[ "after_id" ]
        if after_id is not None:
            root.after_cancel( after_id )
            state[ "after_id" ] = None

    def schedule_next() -> None:
        cancel_after()
        if not state[ "playing" ]:
            return
        state[ "after_id" ] = root.after( state[ "duration_ms" ], advance_frame )

    def advance_frame() -> None:
        update_frame( ( state[ "index" ] + 1 ) % len( frames ) )
        schedule_next()

    def previous_frame() -> None:
        cancel_after()
        update_frame( ( state[ "index" ] - 1 ) % len( frames ) )
        schedule_next()

    def next_frame() -> None:
        cancel_after()
        update_frame( ( state[ "index" ] + 1 ) % len( frames ) )
        schedule_next()

    def jump_start() -> None:
        cancel_after()
        update_frame( 0 )
        schedule_next()

    def jump_end() -> None:
        cancel_after()
        update_frame( len( frames ) - 1 )
        schedule_next()

    def toggle_play() -> None:
        state[ "playing" ] = not state[ "playing" ]
        play_button.configure( text="Pause" if state[ "playing" ] else "Play" )
        schedule_next()

    def on_slider( value: str ) -> None:
        cancel_after()
        update_frame( int( float( value ) ) )
        schedule_next()

    def on_speed_change( value: str ) -> None:
        fps = max( 1.0, float( value ) )
        state[ "duration_ms" ] = max( 10, int( round( 1000.0 / fps ) ) )
        schedule_next()

    play_button = ttk.Button( container, text="Pause", command=toggle_play )
    play_button.grid( row=2, column=0, padx=( 0, 6 ), pady=( 0, 8 ), sticky="ew" )

    prev_button = ttk.Button( container, text="Prev", command=previous_frame )
    prev_button.grid( row=2, column=1, padx=6, pady=( 0, 8 ), sticky="ew" )

    next_button = ttk.Button( container, text="Next", command=next_frame )
    next_button.grid( row=2, column=2, padx=6, pady=( 0, 8 ), sticky="ew" )

    start_button = ttk.Button( container, text="Start", command=jump_start )
    start_button.grid( row=2, column=3, padx=6, pady=( 0, 8 ), sticky="ew" )

    end_button = ttk.Button( container, text="End", command=jump_end )
    end_button.grid( row=2, column=4, padx=6, pady=( 0, 8 ), sticky="ew" )

    speed_label = ttk.Label( container, text="FPS" )
    speed_label.grid( row=2, column=5, padx=( 12, 6 ), pady=( 0, 8 ), sticky="e" )

    initial_fps = max( 1.0, round( 1000.0 / state[ "duration_ms" ], 2 ) )
    speed = tk.DoubleVar( value=initial_fps )
    speed_scale = ttk.Scale(
        container,
        from_=1.0,
        to=30.0,
        variable=speed,
        command=on_speed_change,
    )
    speed_scale.grid( row=2, column=6, pady=( 0, 8 ), sticky="ew" )

    slider = ttk.Scale(
        container,
        from_=0,
        to=max( 0, len( frames ) - 1 ),
        orient="horizontal",
        variable=frame_var,
        command=on_slider,
    )
    slider.grid( row=3, column=0, columnspan=7, sticky="ew" )

    root.bind( "<space>", lambda _event: toggle_play() )
    root.bind( "<Left>", lambda _event: previous_frame() )
    root.bind( "<Right>", lambda _event: next_frame() )
    root.bind( "<Home>", lambda _event: jump_start() )
    root.bind( "<End>", lambda _event: jump_end() )

    def on_close() -> None:
        cancel_after()
        root.destroy()

    root.protocol( "WM_DELETE_WINDOW", on_close )
    update_frame( 0 )
    schedule_next()
    root.mainloop()
