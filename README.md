# Smooth-Life-Search

Smooth-Life-Search is a small Python package for 2D SmoothLife-inspired optimization.

- `SmoothLifeSearch` is the dense SmoothLife-style simulator. It keeps the literal field evolution, disk/ring neighborhoods, lazy objective evaluation, objective-aware transition dynamics, and optional guided objective support.
- `AdaptiveGridSmoothLifeSearch` (`AGSLS`) is a deliberately small controller around that simulator. It runs SmoothLife, detects basins, and zooms the active bounds through three phases.

## What It Implements

- Literal 2D SmoothLife-style state evolution with disk and ring neighborhoods.
- Objective-aware transition dynamics where the optimization landscape can influence the update rule.
- RBF/softmin objective guidance and mild gradient drift for AGSLS commit/exploitation phases.
- Trust-region acquisition batches that let SmoothLife propose basins while off-grid probes improve the incumbent before remaps.
- Exploitation valley tracking remains available as the fallback path when trust-region acquisition is disabled.
- Three-phase AGSLS:
  - exploration: SmoothLife runs without zooming and without objective drift.
  - commit: guided basin support, density, stability, group evidence, and trust-region probes drive conservative remaps.
  - exploitation: the global best anchors aggressive zooms whenever it is inside the active box, after trust-region acquisition improves or confirms the incumbent.
- GIF animation export for the whole search trajectory.
- Built-in benchmark objectives and repeated seeded benchmark runs.
- CLI/config/objective ingestion for JSON/TOML config files, Python-callable objectives, and CSV sampled-surface objectives.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Installed CLI:

```bash
smooth-life-search --help
```

## Run From The CLI

You can run either the installed CLI or `python3 main.py`.

All commands accept a top-level config file:

```bash
smooth-life-search --config run.toml simulate --objective sphere --steps 40
```

Config values are used as defaults; explicit CLI flags override them.

AGSLS optimization run:

```bash
smooth-life-search agsls --objective ackley --dimension 2 --seed 7 --budget 20000 --zoom-cycles 8 --gif agsls.gif
```

Watch the same run in a GUI window:

```bash
smooth-life-search agsls --objective ackley --dimension 2 --seed 7 --budget 20000 --zoom-cycles 8 --show
```

Literal SmoothLife simulation:

```bash
smooth-life-search simulate --objective sphere --dimension 2 --seed 7 --steps 40 --preset paper_glider --gif smoothlife.gif
```

Seeded benchmark sweep:

```bash
smooth-life-search benchmark --objective ackley --dimension 2 --trials 20 --budget 20000 --success-threshold 0.1
```

Use `--json` on any subcommand for machine-readable output.

Trust-region acquisition is enabled by default for AGSLS. Disable it with `--no-trust-region`, or tune it with `--trust-region-commit-evals`, `--trust-region-exploitation-evals`, `--trust-region-candidates`, and `--trust-region-initial-radius-fraction`.

Disable the fallback exploitation valley tracking with `--no-exploitation-valley-tracking`, or tune it with `--exploitation-valley-probes` and `--exploitation-valley-step-fraction`.

## Quick Start From Python

```python
from smooth_life_search import (
    AGSLSConfig,
    AdaptiveGridSmoothLifeSearch,
    SmoothLifeConfig,
    ackley,
)

search = AdaptiveGridSmoothLifeSearch(
    objective=ackley,
    bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    smoothlife_config=SmoothLifeConfig(preset="search"),
    agsls_config=AGSLSConfig(max_evaluations=20_000, max_zoom_cycles=8),
)
search.reset(seed=7)
run = search.run()
print(run.best_value)
print(run.best_point)
```

Package-level modules are organized by responsibility:

- `smooth_life_search.smoothlife`: literal SmoothLife search engine.
- `smooth_life_search.agsls`: three-phase adaptive grid SmoothLife search.
- `smooth_life_search.benchmark`: objective registry and seeded trials.
- `smooth_life_search.visualization`: frame rendering, GIF export, and Tk viewer.
- `smooth_life_search.input`: CLI/config/objective ingestion.
- `smooth_life_search.core`: shared dataclasses, protocols, bounds, and runtime helpers.

Built-in and external objectives can be resolved through the input layer:

```python
from smooth_life_search.input import ObjectiveSpec, resolve_objective

objective = resolve_objective(ObjectiveSpec.builtin("ackley"))
external = resolve_objective({
    "kind": "import_path",
    "import_path": "my_package.objectives:custom_objective",
})
surface = resolve_objective({"kind": "csv_surface", "csv_path": "surface.csv"})
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- The current implementation is intentionally **2D only**.
- AGSLS runs require an objective evaluation budget, either in `AGSLSConfig(max_evaluations=...)` or `run(evaluations=...)`.
- For maximization, set `maximize=True` in `SmoothLifeConfig`.
- The convenience public API lives in `smooth_life_search/__init__.py`; subsystem APIs live in their package `__init__.py` files.
