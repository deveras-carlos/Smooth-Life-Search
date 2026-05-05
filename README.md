# Smooth-Life-Search

Smooth-Life-Search is a small Python package for 2D SmoothLife-inspired simulation and N-D archive-centered optimization.

- `SmoothLifeSearch` is the literal dense SmoothLife-style simulator. It keeps field evolution, disk/ring neighborhoods, lazy objective evaluation, objective-aware transition dynamics, and optional guided objective support.
- `PointCloudSmoothLifeSearch` is the optimizer. It keeps a persistent N-D archive of evaluated points, derives temporary 2D SmoothLife-style ecology projections from that archive or from local active subspaces, and proposes new candidates from global low-discrepancy samples, density samples, anisotropic adaptive trust regions, SHADE-style archive differences, low-rank CMA-style region samples, coherent coordinate probes, restart scouts, local quadratic surrogates, rotated/axis stencils, pattern probes, and incumbent-centered refinement probes.

## What It Implements

- Literal 2D SmoothLife-style state evolution with disk and ring neighborhoods.
- Objective-aware transition dynamics for dense SmoothLife simulation.
- Archive-first N-D point-cloud optimization where every objective evaluation is stored once.
- Adaptive proposal-region portfolios with radius expansion/shrink feedback, stall cooldown, archive-derived anisotropic geometry, and active/sleeping diagnostics.
- Derived 2D SmoothLife-style ecology projections for candidate proposal and 2D visualization, blending fitness support, novelty, uncertainty, crowding, and improvement support.
- High-dimensional evolutionary proposal sources with adaptive source credit allocation, live-population SHADE-style archive differences, low-rank CMA-style region sampling, coherent coordinate probes, surrogate-ranked proposal pools, and restart/scout proposals.
- Dimension-aware quadratic/diagonal surrogate candidates, derivative-free pattern probes, and hybrid BFGS/Levenberg-Marquardt local refinement.
- GIF animation export for 2D simulation and 2D point-cloud search trajectories.
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

Point-cloud optimization run:

```bash
smooth-life-search point-cloud --objective ackley --dimension 2 --seed 7 --budget 20000 --gif point-cloud.gif
```

N-D point-cloud optimization run:

```bash
smooth-life-search point-cloud --objective rosenbrock --dimension 5 --seed 7 --budget 20000 --target-value 1e-6
```

High-dimensional evolutionary point-cloud run:

```bash
smooth-life-search point-cloud --objective rosenbrock --dimension 30 --seed 7 --budget 6400
```

Watch the same run in a GUI window:

```bash
smooth-life-search point-cloud --objective ackley --dimension 2 --seed 7 --budget 20000 --show
```

Literal SmoothLife simulation:

```bash
smooth-life-search simulate --objective sphere --dimension 2 --seed 7 --steps 40 --preset paper_glider --gif smoothlife.gif
```

`simulate`, `--gif`, and `--show` rendering remain 2D. `point-cloud --dimension N` works for `N >= 2` when no visualization output is requested.

Seeded benchmark sweep:

```bash
smooth-life-search benchmark --objective ackley --dimension 2 --trials 20 --budget 20000 --success-threshold 0.1
```

Use `--json` on any subcommand for machine-readable output.

Point-cloud trust regions are enabled by default. Disable region candidate batches with `--no-trust-regions`, disable local surrogates with `--no-surrogate`, or tune the portfolio with `--portfolio-size`, `--region-initial-radius-fraction`, `--region-expand-factor`, `--region-shrink-factor`, `--region-stall-patience`, and `--region-cooldown-batches`.

Anisotropic proposal regions are enabled by default. Disable rotated archive-derived region geometry with `--no-anisotropic-regions`, or tune it with `--region-anisotropy-max` and `--region-geometry-min-samples`.

For N-D runs, tune the local active subspace with `--active-subspace-size`, the surrogate model cutoff with `--surrogate-full-quadratic-max-dimension`, and the 2D proposal-density projection with `--projection-axes I J`.

For high-dimensional runs, evolutionary proposal sources turn on at `--high-dimensional-min-dimension` by default. Restart scouts use scrambled low-discrepancy points with per-axis variation, so they explore broadly without injecting constant diagonal benchmark optima. Tune adaptive allocation with `--source-credit-temperature`, `--source-exploration-floor`, `--relative-success-credit`, `--evolutionary-population-size`, and `--evolutionary-population-max`; tune surrogate preselection with `--candidate-pool-multiplier` and `--surrogate-ranking-neighbor-count`; disable individual engines with `--no-source-adaptation`, `--no-shade`, `--no-cma-region`, `--no-restart-strategy`, or `--no-surrogate-ranking`.

High-dimensional local polishing is also enabled by default. Once the archive
has moved materially beyond deterministic anchors, basin-polishing allocation
spends more evaluations on exploit, region, CMA, SHADE, and direction-set
refinement. Finite-difference probes can recenter on meaningful incumbent
improvements, successful steps feed derivative-free line searches, linkage
scores form coupled refinement blocks, and cross-block L-BFGS memory reuses
accepted curvature. Tune these with `--basin-polishing-activation-ratio`,
`--successful-direction-memory-size`, `--direction-refinement-max-evaluations`,
`--linkage-neighbor-count`, and `--cross-block-lbfgs-memory-size`; disable them
with `--no-probe-recenter`, `--no-basin-polishing`,
`--no-direction-refinement`, `--no-linkage-blocks`, or
`--no-cross-block-lbfgs`.

Disable finite-difference incumbent refinement with `--no-local-refinement`, or tune it with `--local-refinement-start-evaluations`, `--local-refinement-max-evaluations`, `--local-refinement-step-fraction`, `--local-refinement-method`, and `--local-refinement-damping`.

By default, local refinement can report a local stall but does not stop the whole point-cloud run; default optimization runs spend the requested budget for stronger global-search diagnostics. Use `--target-value VALUE` when you only need a run to stop once it reaches an explicit objective threshold. The longer form `--early-stop-enabled --early-stop-value VALUE` is still supported.

```bash
smooth-life-search point-cloud --objective rosenbrock --budget 20000 --target-value 1e-11
smooth-life-search point-cloud --objective rosenbrock --budget 20000 --no-local-refinement --target-value 1e-11
```

## Quick Start From Python

```python
from smooth_life_search import (
    PointCloudSearchConfig,
    PointCloudSmoothLifeSearch,
    SmoothLifeConfig,
    ackley,
)

search = PointCloudSmoothLifeSearch(
    objective=ackley,
    bounds=[(-10.0, 10.0)] * 5,
    smoothlife_config=SmoothLifeConfig(preset="search"),
    point_cloud_config=PointCloudSearchConfig(max_evaluations=20_000),
)
search.reset(seed=7)
run = search.run()
print(run.best_value)
print(run.best_point)
```

Package-level modules are organized by responsibility:

- `smooth_life_search.smoothlife`: literal SmoothLife simulator.
- `smooth_life_search.point_cloud`: archive-centered point-cloud optimizer.
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

- The shipped optimizer supports point-cloud search in N dimensions; dense SmoothLife simulation and rendering are intentionally **2D stable**.
- Point-cloud optimization runs require an objective evaluation budget, either in `PointCloudSearchConfig(max_evaluations=...)` or `run(evaluations=...)`.
- For maximization, set `maximize=True` in `SmoothLifeConfig`.
- The convenience public API lives in `smooth_life_search/__init__.py`; subsystem APIs live in their package `__init__.py` files.
