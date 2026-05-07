"""SmoothLife simulation and Matrix SmoothLife optimization."""

from .benchmark import DEFAULT_BOUNDS, OBJECTIVES, BenchmarkSummary, run_seeded_trials, summarize_results
from .benchmark import ackley, griewank, himmelblau, rastrigin, rosenbrock, sphere
from .core import Basin, SearchResult, SearchRun, SmoothLifeSnapshot, ZoomEvent
from .matrix import MatrixSmoothLifeConfig, MatrixSmoothLifeSample, MatrixSmoothLifeSearch, MatrixSmoothLifeSnapshot
from .smoothlife import SmoothLifeConfig, SmoothLifeSearch
from .visualization import RenderOptions, open_run_viewer, render_run_frames, save_run_animation, snapshot_to_image

__all__ = [
    "DEFAULT_BOUNDS",
    "OBJECTIVES",
    "Basin",
    "BenchmarkSummary",
    "MatrixSmoothLifeConfig",
    "MatrixSmoothLifeSample",
    "MatrixSmoothLifeSearch",
    "MatrixSmoothLifeSnapshot",
    "RenderOptions",
    "SearchResult",
    "SearchRun",
    "SmoothLifeConfig",
    "SmoothLifeSearch",
    "SmoothLifeSnapshot",
    "ZoomEvent",
    "ackley",
    "griewank",
    "himmelblau",
    "open_run_viewer",
    "rastrigin",
    "render_run_frames",
    "rosenbrock",
    "run_seeded_trials",
    "save_run_animation",
    "snapshot_to_image",
    "sphere",
    "summarize_results",
]
