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
5. Derive local region geometry from weighted archive covariance in normalized
   coordinates. The geometry falls back to identity when too few finite nearby
   samples are available.
6. Evaluate mixed candidate batches from global low-discrepancy samples,
   density samples with a global exploration floor, anisotropic region trust
   samples, rotated/axis region stencils, local quadratic surrogate optima,
   pattern probes, and incumbent probes.
7. Expand or shrink region radii based on whether region-sourced candidates
   improve the incumbent; stalled regions cool down instead of deleting the
   rest of the portfolio.
8. Run finite-difference local refinement around the incumbent when enough
   archive evidence exists. The default `hybrid` method uses BFGS-style steps
   and only tries damped Levenberg-Marquardt/Newton steps when the local model
   is stable. A local stall is diagnostic by default and does not certify
   global convergence.
9. Emit typed batch, region, trust-region, stop-reason, active/sleeping region,
   anisotropy, local-refinement, and point-cloud snapshot diagnostics.

The sample archive is the source of truth. Grid arrays are derived views for
proposal density and visualization; they are not objective caches and are not
remapped as an optimization action.

## Public Surface

- Python: `PointCloudSmoothLifeSearch`, `PointCloudSearchConfig`.
- CLI: `smooth-life-search point-cloud`.
- Benchmark sweeps use point-cloud search by default.
- The old public AGSLS class/config/CLI surface is intentionally removed.
- Explicit early stopping is opt-in through `early_stop_enabled` and
  `early_stop_value`; the CLI shorthand is `--target-value VALUE`.
- Anisotropic regions are controlled by `anisotropic_regions_enabled`,
  `region_anisotropy_max`, and `region_geometry_min_samples`.
- Local refinement methods are `bfgs`, `levenberg-marquardt`, and `hybrid`.
- Default CLI runs exhaust their requested budget for global-search
  diagnostics. Use `--target-value 1e-11` for Rosenbrock-style runs where
  reaching a known precision target is the desired stopping condition.

## Acceptance Targets

- The full test suite must pass with `python -m unittest discover -s tests -v`.
- Symmetric benchmark objectives evaluate the center first, so Ackley, Sphere,
  and Rastrigin hit the exact origin on symmetric default bounds.
- Rosenbrock seed-7 smoke runs must remain below the previous precision targets.
