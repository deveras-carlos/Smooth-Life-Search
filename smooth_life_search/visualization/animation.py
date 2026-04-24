"""Animation export helpers."""

from __future__ import annotations

from pathlib import Path

from ..core import SearchRun
from .options import RenderOptions
from .viewer import render_run_frames


def save_run_animation(
    run: SearchRun,
    output_path: str | Path,
    scale: int = 2,
    duration_ms: int = 90,
    options: RenderOptions | None = None,
) -> Path:
    """Export a run as an animated GIF."""

    path = Path( output_path )
    if not run.snapshots:
        raise ValueError( "run must contain at least one snapshot" )
    resolved = options or RenderOptions( scale=scale, duration_ms=duration_ms )
    frames = render_run_frames( run, options=resolved )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[ 1 : ],
        duration=resolved.duration_ms,
        loop=0,
        optimize=False,
    )
    return path
