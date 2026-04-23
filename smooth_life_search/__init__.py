"""SmoothLife Search and Adaptive Grid Smooth Life Search for 2D optimization."""

from .agsls import AGSLSConfig, AdaptiveGridSmoothLifeSearch
from .benchmarking import BenchmarkSummary, run_seeded_trials, summarize_results
from .benchmarks import ackley, griewank, himmelblau, rastrigin, rosenbrock, sphere
from .results import Basin, SearchResult, SearchRun, SmoothLifeSnapshot, ZoomEvent
from .smoothlife import SmoothLifeConfig, SmoothLifeSearch
from .visualization import open_run_viewer, render_run_frames, save_run_animation, snapshot_to_image

__all__ = [
    "AGSLSConfig",
    "AdaptiveGridSmoothLifeSearch",
    "Basin",
    "BenchmarkSummary",
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
