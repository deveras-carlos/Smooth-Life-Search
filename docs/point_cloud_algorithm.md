# Point-Cloud SmoothLife Search

This branch replaces the public AGSLS optimizer with `PointCloudSmoothLifeSearch`.
The dense `SmoothLifeSearch` simulator remains available for literal 2D field
simulation; optimization is now archive-centered and supports N-D point clouds.

## Core Loop

1. Initialize a persistent `PointCloudArchive` and evaluate the domain center.
2. Add a deterministic/global initial design, respecting the evaluation budget.
3. Derive a small deterministic ensemble of temporary 2D projection views from
   the N-D archive:
   - configured coordinate axes when `projection_axes` is set,
   - local active-subspace frames from archive covariance when available,
   - successful-direction axes,
   - under-covered coordinate axes for novelty pressure,
   - fallback coordinate axes `(0, 1)` otherwise,
   - an objective desirability field,
   - novelty, uncertainty, crowding, and improvement-support components used
     by high-dimensional coherent/cooperative proposal allocation,
   - an evaluated-cell mask used only for rendering.
4. Build a portfolio of adaptive proposal regions from elite archive samples.
5. Derive local region geometry from weighted archive covariance in normalized
   N-D coordinates. Small dimensions use the full eigenbasis; larger dimensions
   use a capped active subspace. The geometry falls back to identity when too
   few finite nearby samples are available.
6. Evaluate mixed candidate batches from a deterministic stage allocator:
   exploration uses global low-discrepancy samples, SmoothLife ecology density,
   anisotropic region trust samples, and coherent coordinate probes; basin
   polishing adds local surrogate optima, pattern/stencil probes, and
   incumbent-centered proposals; large-D propagation reserves budget for
   coherent probes and cooperative frontier coordinate groups.
7. Expand or shrink region radii based on whether region-sourced candidates
   improve the incumbent; stalled regions cool down instead of deleting the
   rest of the portfolio.
8. Track source attempts, evaluations, incumbent improvements, improvement
   magnitudes, evaluation share, active/sleeping regions, cooperative group
   activity, and polishing diagnostics. Candidate counts are stage-deterministic
   rather than governed by source-credit feedback.
9. Run finite-difference local refinement around the incumbent when enough
   archive evidence exists. The default `hybrid` method uses BFGS-style steps
   and only tries damped Levenberg-Marquardt/Newton steps when the local model
   is stable. A local stall is diagnostic by default and does not certify
   global convergence.
10. Emit typed batch, source-accounting, trust-region, stop-reason,
    active/sleeping region, anisotropy,
    local-refinement, and point-cloud snapshot diagnostics.

Candidate batches can generate a larger proposal pool than the remaining
evaluation allotment for that batch. A NumPy-only surrogate preselection pass
ranks the pool from local archive neighbors, novelty, and source quotas. A
small reliability gate estimates local rank consistency and softly reduces
surrogate weight when the model looks untrustworthy. The optimizer then
evaluates only selected archive-deduplicated proposals. This does not add
objective evaluations. The default large-D candidate batch cap is kept compact
so discovery batches interleave frequently with one scheduled polishing pass:
local/block refinement, bracketed direction-set refinement, or cooperative
frontier refinement.

The sample archive is the source of truth. Grid arrays are derived 2D views for
proposal density and visualization; they are not objective caches and are not
remapped as an optimization action. In N-D runs, density samples are lifted back
into the full coordinate space around the current incumbent, configured
coordinate frame, or active-subspace origin.

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
  `projection_axes`, `projection_ensemble_enabled`,
  `projection_ensemble_size`, `projection_ensemble_refresh_batches`, and
  `surrogate_full_quadratic_max_dimension`.
- Coherent probes are controlled by `coherent_probes_enabled`.
- Surrogate preselection is controlled by `surrogate_ranking_enabled`,
  `candidate_pool_multiplier`, `surrogate_ranking_neighbor_count`,
  `surrogate_reliability_enabled`, `surrogate_rank_weight_min`, and
  `surrogate_rank_weight_max`.
- Basin-polishing controls are `probe_recenter_enabled`,
  `probe_recenter_max_restarts`, `basin_polishing_enabled`,
  `basin_polishing_min_dimension`, `basin_polishing_activation_ratio`,
  `successful_direction_memory_size`, `direction_refinement_enabled`,
  `direction_refinement_max_evaluations`, `direction_line_search_mode`,
  `direction_line_search_max_steps`,
  `direction_line_search_min_step_fraction`, `linkage_blocks_enabled`,
  `linkage_update_interval_batches`, `linkage_neighbor_count`,
  `cross_block_lbfgs_enabled`, and `cross_block_lbfgs_memory_size`.
- Large-D cooperative propagation is controlled by
  `cooperative_refinement_enabled`, `cooperative_min_dimension`,
  `cooperative_group_size`, `cooperative_groups_per_batch`,
  `cooperative_frontier_enabled`, `cooperative_frontier_fraction`,
  `axis_coverage_pressure`, `active_set_max_fraction`, and
  `active_set_expand_interval_batches`.
- Local refinement methods are `bfgs`, `levenberg-marquardt`, and `hybrid`.
- Default CLI runs exhaust their requested budget for global-search
  diagnostics. Use `--target-value 1e-11` for Rosenbrock-style runs where
  reaching a known precision target is the desired stopping condition.
- `tools/point_cloud_ablation.py` runs deterministic no-GIF ablation matrices
  for high-D Rosenbrock, transformed 12D guardrails, shifted Ackley, and
  symmetric anchor sanity cases. Use it to measure proposal-source
  contribution before changing optimizer policy.

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
  reserves a fraction of groups for frontier propagation from recently
  improved axes, and runs derivative-free group probes in the current incumbent
  context.
- Coherent high-D probes are archive-derived and jittered, so they can test
  broad coordinate-level hypotheses without reintroducing exact constant
  diagonal restart scouts.

Recommended no-GIF smoke commands:

```bash
smooth-life-search point-cloud --objective rosenbrock --dimension 100 --seed 7 --budget 6400
smooth-life-search point-cloud --objective rosenbrock --dimension 500 --seed 7 --budget 6400
```

Latest lean checkpoint values for seed 7 budget 6400 are about `6.77e-2` for
30D, `5.71e-1` for 50D, `5.55e-2` for 100D, and `2.18e1` for 500D; all remain
budget-exhausting diagnostic runs.

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
  diagnostic 500D run. The seed-7 budget-6400 lean targets assert 30D below
  1.0, 50D below 1.0, 100D below 5.0, and 500D below 75.0 while rejecting
  removed SHADE/CMA/restart shortcuts.
- Adversarial transformed benchmark tests cover shifted/rotated objectives so
  center/origin anchors and constant-diagonal scouts cannot masquerade as
  optimizer correctness.
