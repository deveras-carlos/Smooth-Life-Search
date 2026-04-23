"""Animation export helpers."""

from __future__ import annotations

from pathlib import Path

from ..results import SearchRun
from .viewer import render_run_frames


def save_run_animation(
    run: SearchRun,
    output_path: str | Path,
    scale: int = 2,
    duration_ms: int = 90,
) -> Path:
    """Export a run as an animated GIF."""

    path = Path( output_path )
    if not run.snapshots:
        raise ValueError( "run must contain at least one snapshot" )
    frames = render_run_frames( run, scale=scale )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[ 1 : ],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return path
