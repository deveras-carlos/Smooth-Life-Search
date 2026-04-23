"""Density-driven basin detection for 2D SmoothLife support fields."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..results import Basin


def _grid_to_world( bounds: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple[ int, int ] ) -> np.ndarray:
    height, width = shape
    local_x = ( cols.astype( float ) + 0.5 ) / width
    local_y = ( rows.astype( float ) + 0.5 ) / height
    x = bounds[ 0, 0 ] + local_x * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] )
    y = bounds[ 1, 0 ] + local_y * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] )
    return np.column_stack( ( x, y ) )


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
) -> list[ Basin ]:
    """Detect dense alive groups from a support field using DBSCAN-style clustering."""

    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError( "threshold_quantile must be in ( 0, 1 )" )
    threshold = float( np.quantile( support_field, threshold_quantile ) )
    candidate_mask = np.asarray( alive_mask, dtype=bool ) & ( np.asarray( support_field, dtype=float ) >= threshold )
    rows, cols = np.nonzero( candidate_mask )
    if rows.size == 0:
        return [ ]
    points = np.column_stack( ( rows.astype( float ), cols.astype( float ) ) )
    labels = _dbscan_labels( points, cluster_eps_pixels, cluster_min_samples )
    basins: list[ Basin ] = [ ]
    for cluster_id in sorted( label for label in np.unique( labels ) if label >= 0 ):
        member_mask = labels == cluster_id
        cluster_rows = rows[ member_mask ]
        cluster_cols = cols[ member_mask ]
        if cluster_rows.size < int( min_cells ):
            continue
        component = np.zeros_like( candidate_mask, dtype=bool )
        component[ cluster_rows, cluster_cols ] = True
        weights = np.asarray( support_field[ component ], dtype=float )
        if not np.any( np.isfinite( weights ) ):
            continue
        points_world = _grid_to_world( bounds, cluster_rows, cluster_cols, support_field.shape )
        centroid_world = np.average( points_world, axis=0, weights=weights )
        centroid_grid = np.average( np.column_stack( ( cluster_cols, cluster_rows ) ), axis=0, weights=weights )
        row_min = int( cluster_rows.min() )
        row_max = int( cluster_rows.max() )
        col_min = int( cluster_cols.min() )
        col_max = int( cluster_cols.max() )
        height, width = support_field.shape
        bbox_world = np.array(
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
        alive_window = alive_mask[ row_min : row_max + 1, col_min : col_max + 1 ]
        alive_density = float( np.mean( alive_window ) ) if alive_window.size else 0.0
        cluster_objective = float( np.mean( objective_field[ component ] ) )
        explored_component = component & np.asarray( evaluated_mask, dtype=bool )
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
        basin = Basin(
            mask=component,
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
        )
        basins.append( basin )
    basins.sort( key=lambda basin: basin.support_mass, reverse=True )
    return basins
