"""Diagnostic payload schemas for AGSLS decisions and late-stage strategies."""

from __future__ import annotations


def empty_microgrid_summary() -> dict[str, object]:
    return {
        "microgrid_ran": False,
        "microgrid_candidate_centers": [],
        "microgrid_sample_count_total": 0,
        "microgrid_sample_count_per_center": 0,
        "microgrid_winning_center_kind": "",
        "microgrid_winning_point": [],
        "microgrid_winning_value": None,
        "microgrid_refined_bounds": [],
    }


def empty_translation_summary() -> dict[str, object]:
    return {
        "translation_ran": False,
        "translation_trigger_reason": "",
        "translation_target_kind": "",
        "translation_source_point": [],
        "translation_target_point": [],
        "translation_applied_vector": [],
        "translation_refined_bounds": [],
    }


def empty_pattern_search_summary() -> dict[str, object]:
    return {
        "pattern_search_ran": False,
        "pattern_search_seed_kind": "",
        "pattern_search_iterations": 0,
        "pattern_search_evaluations_spent": 0,
        "pattern_search_evaluations_reused": 0,
        "pattern_search_final_step": [],
        "pattern_search_final_point": [],
        "pattern_search_final_value": None,
        "pattern_search_refined_bounds": [],
        "pattern_search_improved": False,
    }


def empty_periodic_local_search_summary() -> dict[str, object]:
    return {
        "periodic_local_search_ran": False,
        "periodic_local_search_runs": 0,
        "periodic_local_search_total_evaluations_spent": 0,
        "periodic_local_search_last_step_index": None,
        "periodic_local_search_last_seed_count": 0,
        "periodic_local_search_last_best_point": [],
        "periodic_local_search_last_best_value": None,
        "periodic_local_search_last_improved": False,
        "periodic_local_search_last_evaluations_spent": 0,
    }


def merge_summary(defaults: dict[str, object], summary: dict[str, object] | None) -> dict[str, object]:
    """Merge an optional strategy summary over schema defaults."""

    payload = dict(defaults)
    if summary is not None:
        payload.update(summary)
    return payload
