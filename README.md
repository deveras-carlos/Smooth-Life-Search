# Smooth-Life-Search

Smooth-Life-Search is a small Python package for 2D SmoothLife-inspired simulation and N-D SmoothLife-native black-box optimization.

- `SmoothLifeSearch` is the literal dense 2D SmoothLife-style simulator.
- `MatrixSmoothLifeSearch` is the optimizer. It treats a whole 2D SmoothLife matrix as one evolving genome, decodes local 8-neighbor cell neighborhoods into one N-D candidate, evaluates that candidate once, and feeds global improvement credit back into the matrix field.

## What It Implements

- Literal 2D SmoothLife-style state evolution with disk and ring neighborhoods.
- Objective-aware transition dynamics for dense SmoothLife simulation.
- Matrix-native N-D optimization with exact archive-based evaluation accounting.
- Local sparse divide-and-conquer decoding from 8-neighbor matrix patches to bounded N-D points.
- Neutral zero-field evaluation, which decodes to the domain center.
- Matrix reward/support feedback from global objective evaluations.
- Signed cell credit, local neighborhood credit, decoded-direction pressure, and per-cell temperature for matrix credit steering.
- Archive-accounted local patch probes for sharper objective-credit assignment.
- Small archive-accounted matrix-space line search after improvements.
- Elite-field pull, failure damping, bounded mutation, and SmoothLife transition updates.
- Typed matrix samples and snapshots for diagnostics and rendering.
- GIF animation export for 2D simulation and matrix-genome dashboards.
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

All commands accept a top-level config file:

```bash
smooth-life-search --config run.toml matrix --objective sphere --budget 1000
```

Config values are used as defaults; explicit CLI flags override them.

Matrix optimization run:

```bash
smooth-life-search matrix --objective ackley --dimension 5 --seed 7 --budget 20000
```

High-dimensional matrix run:

```bash
smooth-life-search matrix --objective rosenbrock --dimension 100 --seed 7 --budget 6400
```

Render the evolving matrix genome:

```bash
smooth-life-search matrix --objective ackley --dimension 5 --seed 7 --budget 1000 --gif matrix.gif
```

Literal SmoothLife simulation remains 2D:

```bash
smooth-life-search simulate --objective sphere --dimension 2 --seed 7 --steps 40 --preset paper_glider --gif smoothlife.gif
```

Seeded benchmark sweep:

```bash
smooth-life-search benchmark --objective ackley --dimension 5 --trials 20 --budget 20000 --success-threshold 0.1
```

Use `--json` on any subcommand for machine-readable output.

Matrix optimizer controls include:

- `--matrix-height` and `--matrix-width` for the genome matrix shape.
- `--decoder`, `--decoder-gain`, and `--projection-seed-offset` for deterministic decoding; `local_sparse` is the default and `random_projection` is available for ablation.
- `--cell-alive-threshold`, `--local-decoder-block-size`, and `--local-decoder-overlap` for local sparse neighborhood decoding.
- `--no-local-credit`, `--local-credit-strength`, `--local-credit-learning-rate`, `--local-credit-decay`, and `--local-credit-clip` for objective-derived neighborhood credit.
- `--no-patch-probes`, `--patch-probe-interval-evaluations`, `--patch-probe-count`, and `--patch-probe-step` for local counterfactual probes.
- `--steps-per-evaluation` for SmoothLife evolution cadence.
- `--elite-pull-strength`, `--failure-damping`, `--reward-decay`, `--reward-boost`, `--mutation-noise`, and `--mutation-decay` for feedback dynamics.
- `--advantage-strength`, `--advantage-decay`, `--direction-strength`, and `--direction-decay` for signed credit steering.
- `--temperature-init`, `--temperature-decay`, `--temperature-reheat`, and `--stagnation-reheat-evaluations` for local exploration temperature.
- `--matrix-line-search-alphas` and `--no-matrix-line-search` for matrix-space line search.
- `--target-value VALUE` for opt-in early stopping, equivalent to `--early-stop-enabled --early-stop-value VALUE`.
- `--matrix-snapshot-interval` and `--no-store-all-snapshots` for snapshot storage.

The old `point-cloud` command and public `PointCloud*` API are intentionally removed on this branch.

## Quick Start From Python

```python
from smooth_life_search import (
    MatrixSmoothLifeConfig,
    MatrixSmoothLifeSearch,
    SmoothLifeConfig,
    ackley,
)

search = MatrixSmoothLifeSearch(
    objective=ackley,
    bounds=[(-10.0, 10.0)] * 5,
    smoothlife_config=SmoothLifeConfig(preset="search"),
    matrix_config=MatrixSmoothLifeConfig(max_evaluations=20_000),
)
search.reset(seed=7)
run = search.run()
print(run.best_value)
print(run.best_point)
```

Package-level modules are organized by responsibility:

- `smooth_life_search.smoothlife`: literal SmoothLife simulator.
- `smooth_life_search.matrix`: SmoothLife matrix-genome optimizer.
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

- Matrix optimization supports `N >= 2`; dense SmoothLife simulation remains intentionally 2D.
- Matrix optimization runs require an objective evaluation budget, either in `MatrixSmoothLifeConfig(max_evaluations=...)` or `run(evaluations=...)`.
- The neutral zero matrix always decodes to the domain center. This is useful for symmetric benchmarks, but shifted/rotated tests are needed to measure real search quality.
- For maximization, set `maximize=True` in `SmoothLifeConfig`.
