# Point-Cloud SmoothLife Search

This branch replaces the public AGSLS optimizer with `PointCloudSmoothLifeSearch`.
The dense `SmoothLifeSearch` simulator remains available for literal field
simulation; optimization is now archive-centered.

## Core Loop

1. Initialize a persistent `PointCloudArchive` and evaluate the domain center.
2. Add a deterministic/global initial design, respecting the evaluation budget.
3. Derive temporary 2D views from the archive:
   - a SmoothLife-style proposal density rasterized from elite samples,
   - an objective desirability field,
   - an evaluated-cell mask used only for rendering.
4. Build a portfolio of adaptive proposal regions from elite archive samples.
5. Evaluate mixed candidate batches from global low-discrepancy samples,
   density samples with a global exploration floor, region trust samples,
   region stencils, local quadratic surrogate optima, and incumbent probes.
6. Expand or shrink region radii based on whether region-sourced candidates
   improve the incumbent; stalled regions cool down instead of deleting the
   rest of the portfolio.
7. Run finite-difference local refinement around the incumbent when enough
   archive evidence exists. A local stall is diagnostic by default and does
   not certify global convergence.
8. Emit typed batch, region, trust-region, stop-reason, active/sleeping
   region, and point-cloud snapshot diagnostics.

The sample archive is the source of truth. Grid arrays are derived views for
proposal density and visualization; they are not objective caches and are not
remapped as an optimization action.

## Public Surface

- Python: `PointCloudSmoothLifeSearch`, `PointCloudSearchConfig`.
- CLI: `smooth-life-search point-cloud`.
- Benchmark sweeps use point-cloud search by default.
- The old public AGSLS class/config/CLI surface is intentionally removed.
- Explicit early stopping is opt-in through `early_stop_enabled` and
  `early_stop_value`.

## Acceptance Targets

- The full test suite must pass with `python -m unittest discover -s tests -v`.
- Symmetric benchmark objectives evaluate the center first, so Ackley, Sphere,
  and Rastrigin hit the exact origin on symmetric default bounds.
- Rosenbrock seed-7 smoke runs must remain below the previous precision targets.
