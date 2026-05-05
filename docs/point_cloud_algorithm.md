# Point-Cloud SmoothLife Search

This branch replaces the public AGSLS optimizer with `PointCloudSmoothLifeSearch`.
The dense `SmoothLifeSearch` simulator remains available for literal 2D field
simulation; optimization is now archive-centered and supports N-D point clouds.

## Core Loop

1. Initialize a persistent `PointCloudArchive` and evaluate the domain center.
2. Add a deterministic/global initial design, respecting the evaluation budget.
3. Derive temporary 2D projection views from the N-D archive:
   - a SmoothLife-style ecology field rasterized from elite samples on a
     local active-subspace frame when available,
   - configured coordinate axes when `projection_axes` is set,
   - fallback coordinate axes `(0, 1)` otherwise,
   - an objective desirability field,
   - novelty, uncertainty, crowding, and improvement-support components used
     by high-dimensional evolutionary proposal allocation,
   - an evaluated-cell mask used only for rendering.
4. Build a portfolio of adaptive proposal regions from elite archive samples.
5. Derive local region geometry from weighted archive covariance in normalized
   N-D coordinates. Small dimensions use the full eigenbasis; larger dimensions
   use a capped active subspace. The geometry falls back to identity when too
   few finite nearby samples are available.
6. Evaluate mixed candidate batches from global low-discrepancy samples,
   SmoothLife ecology density samples with a global exploration floor,
   anisotropic region trust samples, rotated/axis region stencils, SHADE-style
   current-to-pbest archive-difference proposals, low-rank CMA-style region
   proposals, coherent coordinate probes, scrambled restart/scout proposals,
   local quadratic surrogate optima, pattern probes, and incumbent probes.
7. Expand or shrink region radii based on whether region-sourced candidates
   improve the incumbent; stalled regions cool down instead of deleting the
   rest of the portfolio.
8. Track source attempts, evaluations, global-best improvements,
   parent-relative improvements, region-relative improvements, improvement
   magnitudes, and bounded EMA credit. In high-dimensional runs, candidate
   counts are adapted from source credit with a configured exploration floor.
   SHADE memory and CMA region state are updated from relative success, not
   only from global incumbent replacement.
9. Run finite-difference local refinement around the incumbent when enough
   archive evidence exists. The default `hybrid` method uses BFGS-style steps
   and only tries damped Levenberg-Marquardt/Newton steps when the local model
   is stable. A local stall is diagnostic by default and does not certify
   global convergence.
10. Emit typed batch, source-credit, SHADE-memory, CMA-region, restart,
    trust-region, stop-reason, active/sleeping region, anisotropy,
    local-refinement, and point-cloud snapshot diagnostics.

High-dimensional batches can generate a larger proposal pool than the remaining
evaluation allotment for that batch. A NumPy-only surrogate preselection pass
ranks the pool from local archive neighbors, novelty, source quotas, and source
credit, then evaluates only selected archive-deduplicated proposals. This does
not add objective evaluations.

The sample archive is the source of truth. Grid arrays are derived 2D views for
proposal density and visualization; they are not objective caches and are not
remapped as an optimization action. In N-D runs, density samples are lifted back
into the full coordinate space around the current incumbent or active region.

## Public Surface

- Python: `PointCloudSmoothLifeSearch`, `PointCloudSearchConfig`.
- CLI: `smooth-life-search point-cloud`.
- Benchmark sweeps use point-cloud search by default.
- The old public AGSLS class/config/CLI surface is intentionally removed.
- Explicit early stopping is opt-in through `early_stop_enabled` and
  `early_stop_value`; the CLI shorthand is `--target-value VALUE`.
- Anisotropic regions are controlled by `anisotropic_regions_enabled`,
  `region_anisotropy_max`, `region_geometry_min_samples`, and
  `active_subspace_size`.
- N-D density projection and surrogate behavior are controlled by
  `projection_axes` and `surrogate_full_quadratic_max_dimension`.
- High-dimensional evolutionary proposals are controlled by
  `source_adaptation_enabled`, `source_credit_temperature`,
  `source_exploration_floor`, `evolutionary_population_size`,
  `evolutionary_population_max`, `relative_success_credit`, `shade_enabled`,
  `shade_memory_size`, `shade_pbest_fraction`, `shade_archive_fraction`,
  `cma_region_enabled`, `cma_direction_memory_size`, `cma_sigma_init`,
  `restart_strategy_enabled`, and `restart_stall_batches`.
- Surrogate preselection is controlled by `surrogate_ranking_enabled`,
  `candidate_pool_multiplier`, and `surrogate_ranking_neighbor_count`.
- Basin-polishing controls are `probe_recenter_enabled`,
  `probe_recenter_max_restarts`, `basin_polishing_enabled`,
  `basin_polishing_min_dimension`, `basin_polishing_activation_ratio`,
  `successful_direction_memory_size`, `direction_refinement_enabled`,
  `direction_refinement_max_evaluations`, `linkage_blocks_enabled`,
  `linkage_update_interval_batches`, `linkage_neighbor_count`,
  `cross_block_lbfgs_enabled`, and `cross_block_lbfgs_memory_size`.
- Large-D cooperative propagation is controlled by
  `cooperative_refinement_enabled`, `cooperative_min_dimension`,
  `cooperative_group_size`, `cooperative_groups_per_batch`,
  `active_set_max_fraction`, and `active_set_expand_interval_batches`.
- Local refinement methods are `bfgs`, `levenberg-marquardt`, and `hybrid`.
- Default CLI runs exhaust their requested budget for global-search
  diagnostics. Use `--target-value 1e-11` for Rosenbrock-style runs where
  reaching a known precision target is the desired stopping condition.

## N-D Phase 1 Scope

- `smooth-life-search point-cloud --dimension N` accepts `N >= 2`.
- `smooth-life-search simulate` remains restricted to `--dimension 2`.
- Himmelblau remains restricted to 2D.
- GIF and GUI rendering remain 2D because snapshots still use projected density
  dashboards.

## Large-D Propagation Checkpoint

- The 30D and 50D Rosenbrock cases are already basin-polishing problems; they
  stay covered by the existing local/block/direction regression checks.
- 100D and 500D need propagation through many coordinate groups. Cooperative
  active-set refinement keeps a capped set of promising and under-covered axes,
  adds frontier coordinate windows, and runs derivative-free group probes in the
  current incumbent context.
- Coherent high-D probes are archive-derived and jittered, so they can test
  broad coordinate-level hypotheses without reintroducing exact constant
  diagonal restart scouts.

Recommended no-GIF smoke commands:

```bash
smooth-life-search point-cloud --objective rosenbrock --dimension 100 --seed 7 --budget 6400
smooth-life-search point-cloud --objective rosenbrock --dimension 500 --seed 7 --budget 6400
```

Latest checkpoint values for those commands are below `1.0` for 100D and
approximately `1.9e1` for 500D; both remain budget-exhausting diagnostic runs.
- Full quadratic surrogates are used only up to the configured dimension
  cutoff. Higher-dimensional fits use a diagonal/linear fallback and are
  rejected safely when the fit is underdetermined or ill-conditioned.

## Acceptance Targets

- The full test suite must pass with `python -m unittest discover -s tests -v`.
- Symmetric benchmark objectives evaluate the center first, so Ackley, Sphere,
  and Rastrigin hit the exact origin on symmetric default bounds.
- Rosenbrock seed-7 smoke runs must remain below the previous precision targets.
- Phase 1 N-D smoke runs cover Sphere 5D/10D, Ackley 5D, Rastrigin 5D, and a
  practical Rosenbrock 5D target.
- High-dimensional Rosenbrock smoke runs cover 10D, 30D, 50D, 100D, and a
  diagnostic 500D run. The seed-7 budget-6400 targets assert 30D below 1.0,
  50D below 0.05, 100D below 90.0, and 500D below 490.0 while rejecting exact
  diagonal scout shortcuts.
- Adversarial transformed benchmark tests cover shifted/rotated objectives so
  center/origin anchors and constant-diagonal scouts cannot masquerade as
  optimizer correctness.
