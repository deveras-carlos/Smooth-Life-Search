# Smooth-Life-Search

Smooth-Life-Search is a small Python package for two related 2D algorithms:

- `SmoothLifeSearch`: a literal dense SmoothLife-style simulator that can couple an optimization objective directly into the state transition.
- `AdaptiveGridSmoothLifeSearch` (`AGSLS`): a controller that uses `SmoothLifeSearch` as its local search engine and repeatedly zooms the whole grid into the most promising basin.

## What It Implements

- Literal 2D SmoothLife-style state evolution with disk and ring neighborhoods.
- Objective-aware transition dynamics where the optimization landscape is part of the update rule.
- Adaptive basin detection and full-grid zooming through `AGSLS`.
- GIF animation export for the whole search trajectory.
- Benchmark helpers for repeated seeded runs and aggregate summaries.

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

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- The current implementation is intentionally **2D only**.
- For maximization, set `maximize=True` in `SmoothLifeConfig`.
- The project now exposes the class-based APIs directly rather than the old compatibility wrappers.
- The public API lives in `smooth_life_search/__init__.py`.
