"""Density-driven basin detection for 2D SmoothLife support fields."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..core import Basin

_NEIGHBOR_OFFSETS = tuple( ( dr, dc ) for dr in ( -1, 0, 1 ) for dc in ( -1, 0, 1 ) if dr != 0 or dc != 0 )


def _grid_to_world( bounds: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple[ int, int ] ) -> np.ndarray:
    height, width = shape
    local_x = ( cols.astype( float ) + 0.5 ) / width
    local_y = ( rows.astype( float ) + 0.5 ) / height
    x = bounds[ 0, 0 ] + local_x * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] )
    y = bounds[ 1, 0 ] + local_y * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] )
    return np.column_stack( ( x, y ) )


def _bbox_world_from_grid(
    bounds: np.ndarray,
    row_min: int,
    col_min: int,
    row_max: int,
    col_max: int,
    shape: tuple[ int, int ],
) -> np.ndarray:
    height, width = shape
    return np.array(
        [
            [
                bounds[ 0, 0 ] + ( col_min / width ) * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] ),
                bounds[ 0, 0 ] + ( ( col_max + 1 ) / width ) * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] ),
            ],
            [
                bounds[ 1, 0 ] + ( row_min / height ) * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] ),
                bounds[ 1, 0 ] + ( ( row_max + 1 ) / height ) * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] ),
            ],
        ],
        dtype=float,
    )


def _dbscan_labels( points: np.ndarray, eps: float, min_samples: int ) -> np.ndarray:
    if points.size == 0:
        return np.asarray( [ ], dtype=int )
    diff = points[ :, None, : ] - points[ None, :, : ]
    sq_distances = np.sum( diff * diff, axis=2 )
    neighbors = sq_distances <= float( eps ) ** 2
    neighbor_counts = np.sum( neighbors, axis=1 )
    core_mask = neighbor_counts >= int( min_samples )
    labels = np.full( points.shape[ 0 ], -1, dtype=int )
    queued = np.zeros( points.shape[ 0 ], dtype=bool )
    cluster_id = 0
    for index in range( points.shape[ 0 ] ):
        if labels[ index ] != -1 or not core_mask[ index ]:
            continue
        queue: deque[ int ] = deque( [ index ] )
        labels[ index ] = cluster_id
        queued[ index ] = True
        while queue:
            current = queue.popleft()
            neighbor_indices = np.flatnonzero( neighbors[ current ] )
            for neighbor in neighbor_indices:
                if labels[ neighbor ] == -1:
                    labels[ neighbor ] = cluster_id
                if core_mask[ neighbor ] and labels[ neighbor ] == cluster_id and not queued[ neighbor ]:
                    queue.append( int( neighbor ) )
                    queued[ neighbor ] = True
        cluster_id += 1
    return labels


def _label_components( candidate_mask: np.ndarray ) -> np.ndarray:
    labels = np.full( candidate_mask.shape, -1, dtype=int )
    height, width = candidate_mask.shape
    component_id = 0
    for start_row, start_col in zip( *np.nonzero( candidate_mask ) ):
        if labels[ start_row, start_col ] >= 0:
            continue
        labels[ start_row, start_col ] = component_id
        queue: deque[ tuple[ int, int ] ] = deque( [ ( int( start_row ), int( start_col ) ) ] )
        while queue:
            row, col = queue.popleft()
            for row_offset, col_offset in _NEIGHBOR_OFFSETS:
                next_row = row + row_offset
                next_col = col + col_offset
                if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                    continue
                if labels[ next_row, next_col ] >= 0 or not candidate_mask[ next_row, next_col ]:
                    continue
                labels[ next_row, next_col ] = component_id
                queue.append( ( next_row, next_col ) )
        component_id += 1
    return labels


def _component_for_seeds( labels: np.ndarray, seed_rows: np.ndarray, seed_cols: np.ndarray ) -> np.ndarray:
    seed_labels = labels[ seed_rows.astype( int ), seed_cols.astype( int ) ]
    seed_labels = seed_labels[ seed_labels >= 0 ]
    if seed_labels.size == 0:
        return np.zeros_like( labels, dtype=bool )
    unique_labels, counts = np.unique( seed_labels, return_counts=True )
    component_label = int( unique_labels[ int( np.argmax( counts ) ) ] )
    return labels == component_label


def _dilate_mask( mask: np.ndarray, iterations: int ) -> np.ndarray:
    result = np.asarray( mask, dtype=bool ).copy()
    height, width = result.shape
    for _ in range( int( iterations ) ):
        padded = np.pad( result, 1, mode="constant", constant_values=False )
        expanded = result.copy()
        for row_offset in range( 3 ):
            for col_offset in range( 3 ):
                expanded |= padded[ row_offset : row_offset + height, col_offset : col_offset + width ]
        result = expanded
    return result


def _point_in_mask( bounds: np.ndarray, point: np.ndarray | None, mask: np.ndarray ) -> bool:
    if point is None:
        return False
    point = np.asarray( point, dtype=float )
    if point.shape != ( 2, ):
        return False
    if np.any( point < bounds[ :, 0 ] ) or np.any( point > bounds[ :, 1 ] ):
        return False
    height, width = mask.shape
    span = bounds[ :, 1 ] - bounds[ :, 0 ]
    local_x = ( point[ 0 ] - bounds[ 0, 0 ] ) / max( span[ 0 ], 1e-12 )
    local_y = ( point[ 1 ] - bounds[ 1, 0 ] ) / max( span[ 1 ], 1e-12 )
    if local_x < 0.0 or local_x > 1.0 or local_y < 0.0 or local_y > 1.0:
        return False
    col = min( width - 1, max( 0, int( np.floor( local_x * width ) ) ) )
    row = min( height - 1, max( 0, int( np.floor( local_y * height ) ) ) )
    return bool( mask[ row, col ] )


def _is_better_or_equal( candidate: float | None, incumbent: float | None, *, maximize: bool ) -> bool:
    if candidate is None or incumbent is None or not np.isfinite( incumbent ):
        return False
    if maximize:
        return float( candidate ) >= float( incumbent ) - 1e-12
    return float( candidate ) <= float( incumbent ) + 1e-12


def detect_basins(
    support_field: np.ndarray,
    objective_field: np.ndarray,
    bounds: np.ndarray,
    alive_mask: np.ndarray,
    evaluated_mask: np.ndarray,
    objective_values: np.ndarray,
    maximize: bool,
    threshold_quantile: float,
    min_cells: int,
    cluster_eps_pixels: float,
    cluster_min_samples: int,
    *,
    basin_envelope_quantile_offset: float = 0.08,
    basin_envelope_growth_pixels: int = 1,
    global_best_point: np.ndarray | None = None,
    global_best_value: float | None = None,
) -> list[ Basin ]:
    """Detect dense alive groups with tight cores and safer connected envelopes."""

    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError( "threshold_quantile must be in ( 0, 1 )" )
    threshold = float( np.quantile( support_field, threshold_quantile ) )
    envelope_quantile = max( 0.50, float( threshold_quantile ) - float( basin_envelope_quantile_offset ) )
    envelope_threshold = float( np.quantile( support_field, envelope_quantile ) )
    support = np.asarray( support_field, dtype=float )
    alive = np.asarray( alive_mask, dtype=bool )
    core_candidate_mask = alive & ( support >= threshold )
    envelope_candidate_mask = alive & ( support >= envelope_threshold )
    rows, cols = np.nonzero( core_candidate_mask )
    if rows.size == 0:
        return [ ]
    points = np.column_stack( ( rows.astype( float ), cols.astype( float ) ) )
    labels = _dbscan_labels( points, cluster_eps_pixels, cluster_min_samples )
    envelope_labels = _label_components( envelope_candidate_mask )
    basins: list[ Basin ] = [ ]
    seen_envelopes: set[ int ] = set()
    for cluster_id in sorted( label for label in np.unique( labels ) if label >= 0 ):
        member_mask = labels == cluster_id
        cluster_rows = rows[ member_mask ]
        cluster_cols = cols[ member_mask ]
        if cluster_rows.size < int( min_cells ):
            continue
        core_component = np.zeros_like( core_candidate_mask, dtype=bool )
        core_component[ cluster_rows, cluster_cols ] = True
        seed_labels = envelope_labels[ cluster_rows, cluster_cols ]
        seed_labels = seed_labels[ seed_labels >= 0 ]
        if seed_labels.size == 0:
            continue
        unique_seed_labels, seed_label_counts = np.unique( seed_labels, return_counts=True )
        envelope_label = int( unique_seed_labels[ int( np.argmax( seed_label_counts ) ) ] )
        if envelope_label in seen_envelopes:
            continue
        seen_envelopes.add( envelope_label )
        envelope_component = _component_for_seeds( envelope_labels, cluster_rows, cluster_cols )
        envelope_component |= core_component
        envelope_component = _dilate_mask( envelope_component, basin_envelope_growth_pixels )
        envelope_rows, envelope_cols = np.nonzero( envelope_component )
        if envelope_rows.size == 0:
            continue
        weights = np.asarray( support_field[ core_component ], dtype=float )
        if not np.any( np.isfinite( weights ) ):
            continue
        points_world = _grid_to_world( bounds, cluster_rows, cluster_cols, support_field.shape )
        centroid_world = np.average( points_world, axis=0, weights=weights )
        centroid_grid = np.average( np.column_stack( ( cluster_cols, cluster_rows ) ), axis=0, weights=weights )
        core_row_min = int( cluster_rows.min() )
        core_row_max = int( cluster_rows.max() )
        core_col_min = int( cluster_cols.min() )
        core_col_max = int( cluster_cols.max() )
        row_min = int( envelope_rows.min() )
        row_max = int( envelope_rows.max() )
        col_min = int( envelope_cols.min() )
        col_max = int( envelope_cols.max() )
        bbox_world = _bbox_world_from_grid( bounds, row_min, col_min, row_max, col_max, support_field.shape )
        core_bbox_world = _bbox_world_from_grid(
            bounds,
            core_row_min,
            core_col_min,
            core_row_max,
            core_col_max,
            support_field.shape,
        )
        alive_density = float( np.mean( alive[ envelope_component ] ) ) if np.any( envelope_component ) else 0.0
        explored_component = envelope_component & np.asarray( evaluated_mask, dtype=bool )
        evaluated_count = int( np.count_nonzero( explored_component ) )
        best_objective_score = 0.0
        mean_objective_score = 0.0
        if evaluated_count > 0:
            evaluated_scores = np.asarray( objective_field[ explored_component ], dtype=float )
            best_objective_score = float( np.max( evaluated_scores ) )
            mean_objective_score = float( np.mean( evaluated_scores ) )
            cluster_objective = 0.7 * best_objective_score + 0.3 * mean_objective_score
        else:
            cluster_objective = 0.25
        unexplored_fraction = float( np.mean( ~np.asarray( evaluated_mask, dtype=bool )[ envelope_component ] ) )
        basin_best_point: np.ndarray | None = None
        basin_best_value: float | None = None
        if np.any( explored_component ):
            flat_component = explored_component.ravel()
            flat_values = objective_values.ravel()
            explored_indices = np.flatnonzero( flat_component )
            explored_values = flat_values[ explored_indices ]
            if maximize:
                best_offset = int( np.argmax( explored_values ) )
            else:
                best_offset = int( np.argmin( explored_values ) )
            best_flat = int( explored_indices[ best_offset ] )
            best_row, best_col = np.unravel_index( best_flat, support_field.shape )
            basin_best_point = _grid_to_world(
                bounds,
                np.asarray( [ best_row ], dtype=int ),
                np.asarray( [ best_col ], dtype=int ),
                support_field.shape,
            )[ 0 ]
            basin_best_value = float( flat_values[ best_flat ] )
        incumbent_in_envelope = _point_in_mask( bounds, global_best_point, envelope_component )
        better_than_incumbent = _is_better_or_equal( basin_best_value, global_best_value, maximize=maximize )
        basin = Basin(
            mask=envelope_component,
            centroid_grid=np.asarray( centroid_grid, dtype=float ),
            centroid_world=np.asarray( centroid_world, dtype=float ),
            bbox_grid=( row_min, col_min, row_max, col_max ),
            bbox_world=bbox_world,
            support_mass=float( np.sum( weights ) ),
            objective_score=cluster_objective,
            stability_score=0.0,
            alive_density=alive_density,
            basin_best_point=None if basin_best_point is None else np.asarray( basin_best_point, dtype=float ),
            basin_best_value=basin_best_value,
            core_mask=core_component,
            core_bbox_grid=( core_row_min, core_col_min, core_row_max, core_col_max ),
            core_bbox_world=core_bbox_world,
            evaluated_count=evaluated_count,
            best_objective_score=best_objective_score,
            mean_objective_score=mean_objective_score,
            unexplored_fraction=unexplored_fraction,
            incumbent_in_envelope=incumbent_in_envelope,
            better_than_incumbent=better_than_incumbent,
        )
        basins.append( basin )
    basins.sort( key=lambda basin: basin.support_mass, reverse=True )
    return basins
