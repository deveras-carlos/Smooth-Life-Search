# Smooth-Life-Search

Smooth-Life-Search is a small Python package for two related 2D algorithms:

- `SmoothLifeSearch`: a literal dense SmoothLife-style simulator that can couple an optimization objective directly into the state transition.
- `AdaptiveGridSmoothLifeSearch` (`AGSLS`): a controller that uses `SmoothLifeSearch` as its local search engine and repeatedly zooms the whole grid into the most promising basin.

## What It Implements

- Literal 2D SmoothLife-style state evolution with disk and ring neighborhoods.
- Objective-aware transition dynamics where the optimization landscape is part of the update rule.
- Adaptive basin detection and full-grid zooming through `AGSLS`.
- GIF animation export for the whole search trajectory.
- A `benchmark` package for objective functions, repeated seeded runs, tuning studies, exploitation studies, and artifact/report IO.
- An `input` package for CLI parsing, JSON/TOML config loading, Python-callable objective loading, and CSV sampled-surface objectives.
- A `core` package for shared models, bounds helpers, objective protocols, and runtime scheduling.

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
smooth-life-search --config run.toml simulate --steps 40
```

Config values are used as defaults; explicit CLI flags override them.

AGSLS optimization run:

```bash
smooth-life-search agsls --objective ackley --dimension 2 --seed 7 --budget 20000 --zoom-cycles 5 --gif agsls.gif
```

Watch the same run in a GUI window:

```bash
smooth-life-search agsls --objective ackley --dimension 2 --seed 7 --budget 20000 --zoom-cycles 5 --show
```

Literal SmoothLife simulation:

```bash
smooth-life-search simulate --objective sphere --dimension 2 --seed 7 --steps 40 --preset paper_glider --gif smoothlife.gif
```

Seeded benchmark sweep:

```bash
smooth-life-search benchmark --objective ackley --dimension 2 --trials 20 --budget 20000 --success-threshold 0.1
```

Use `--json` on any subcommand if you want machine-readable output.

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
    agsls_config=AGSLSConfig(max_zoom_cycles=5),
)
search.reset(seed=7)
run = search.run()
print(run.best_value)
print(run.best_point)
```

Package-level modules are organized by responsibility:

- `smooth_life_search.smoothlife`: literal SmoothLife search engine.
- `smooth_life_search.agsls`: adaptive grid SmoothLife search.
- `smooth_life_search.benchmark`: objective registry, seeded trials, tuning, exploitation studies, artifacts, reports.
- `smooth_life_search.visualization`: frame rendering, GIF export, and Tk viewer.
- `smooth_life_search.input`: CLI/config/objective ingestion.
- `smooth_life_search.core`: shared dataclasses, protocols, bounds, and scheduling.

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
- For maximization, set `maximize=True` in `SmoothLifeConfig`.
- The project now exposes the class-based APIs directly rather than the old compatibility wrappers.
- The convenience public API lives in `smooth_life_search/__init__.py`; subsystem APIs live in their package `__init__.py` files.
