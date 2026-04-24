from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

import smooth_life_search.benchmark.tuning as tuning_module
from smooth_life_search import (
    AGSLSConfig,
    AdaptiveGridSmoothLifeSearch,
    FieldSchedule,
    RuntimeSignals,
    SchedulePolicy,
    SmoothLifeConfig,
    SmoothLifeSearch,
    StudySpec,
    ackley,
    run_tuning_study,
    sphere,
    summarize_results,
)
from smooth_life_search.core import SearchRun


def _trial_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tiny_spec(output_dir: Path, workers: int, *, resume: bool = True) -> StudySpec:
    return StudySpec(
        output_dir=output_dir,
        objectives=("sphere", "ackley"),
        baseline_budgets=(40,),
        static_budgets=(40,),
        adaptive_screen_budgets=(40,),
        adaptive_confirmation_budgets=(60,),
        interaction_budgets=(60,),
        final_confirmation_budgets=(80,),
        stage1_seeds=1,
        stage2_seeds=1,
        stage3_seeds=1,
        stage4_seeds=1,
        stage5_seeds=1,
        stage6_seeds=1,
        interaction_top_k=2,
        finalist_limit=2,
        workers=workers,
        resume=resume,
        keep_snapshots_in_finalists=False,
        parameter_families=("transition_window", "diffusion", "objective_guidance", "basin_quantile", "late_stage_microgrid", "late_stage_translation"),
        profile="test",
        preflight=False,
    )


def _artifact_texts(summary: tuning_module.StudySummary) -> dict[str, str]:
    return {
        "per_objective": summary.per_objective_leaderboard_path.read_text(encoding="utf-8"),
        "overall": summary.overall_rank_path.read_text(encoding="utf-8"),
        "parameter_effects": summary.parameter_effects_path.read_text(encoding="utf-8"),
        "adaptive_required": summary.adaptive_required_path.read_text(encoding="utf-8"),
        "adaptive_paired_effects": summary.adaptive_paired_effects_path.read_text(encoding="utf-8"),
        "finalists": summary.finalists_path.read_text(encoding="utf-8"),
        "family_manifest": summary.family_manifest_path.read_text(encoding="utf-8"),
        "report": summary.report_path.read_text(encoding="utf-8"),
    }


def _paired_row(
    *,
    variant: str,
    family: str,
    family_scope: str,
    objective: str,
    seed: int,
    improvement: float,
) -> dict[str, object]:
    constant_best = 5.0
    adaptive_best = constant_best - improvement
    return {
        "variant": variant,
        "family": family,
        "family_scope": family_scope,
        "objective": objective,
        "budget": 3200,
        "seed": seed,
        "maximize": False,
        "best_constant_config_id": f"{family}__constant",
        "best_adaptive_config_id": f"{family}__adaptive",
        "best_adaptive_schedule_kind": "budget_sigmoid",
        "constant_best_value": constant_best,
        "adaptive_best_value": adaptive_best,
        "paired_improvement": float(improvement),
        "adaptive_win": bool(improvement > 0.0),
    }


class TestBenchmarkingSummary(unittest.TestCase):
    def test_maximize_success_threshold_uses_greater_equal(self) -> None:
        results = [
            SearchRun(best_point=np.zeros(2), best_value=10.0, evaluations=1, bounds=np.zeros((2, 2)), snapshots=[], zoom_events=[], metadata={"seed": 0}),
            SearchRun(best_point=np.zeros(2), best_value=12.0, evaluations=1, bounds=np.zeros((2, 2)), snapshots=[], zoom_events=[], metadata={"seed": 1}),
        ]
        summary = summarize_results(results, success_threshold=9.0, maximize=True)
        self.assertAlmostEqual(summary.success_rate, 1.0)


class TestRuntimePolicies(unittest.TestCase):
    def test_remap_resets_local_best_but_preserves_global_best(self) -> None:
        config = SmoothLifeConfig(grid_shape=(48, 48), evaluations_per_step=4, snapshot_interval=1, preset="search")
        search = SmoothLifeSearch(sphere, bounds=[(-10.0, 10.0), (-10.0, 10.0)], config=config)
        search.reset(seed=7)
        search.step(4)
        before_global = float(search.state.best_value)
        before_local = float(search.state.local_best_value)
        search.remap_to_bounds(np.asarray([[5.0, 10.0], [5.0, 10.0]], dtype=float))
        self.assertLessEqual(float(search.state.best_value), before_global + 1e-12)
        self.assertGreater(float(search.state.local_best_value), before_local)

    def test_per_run_evaluation_limit_does_not_mutate_shared_agsls_config(self) -> None:
        smoothlife = SmoothLifeConfig(grid_shape=(24, 24), evaluations_per_step=3, preset="search")
        agsls = AGSLSConfig(max_zoom_cycles=2, max_evaluations=200)
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            smoothlife_config=smoothlife,
            agsls_config=agsls,
        )
        search.reset(seed=2)
        search.run(evaluations=50)
        self.assertEqual(agsls.max_evaluations, 200)
        self.assertEqual(search.agsls_config.max_evaluations, 200)
        search.reset(seed=3)
        result = search.run()
        self.assertLessEqual(result.evaluations, 200)

    def test_kernel_rebuild_happens_only_for_zoom_boundary_schedules(self) -> None:
        policy = SchedulePolicy(
            schedules=(
                FieldSchedule("diffusion", "budget_sigmoid", base_value=0.0, low_value=0.0, high_value=0.2),
                FieldSchedule("inner_radius", "zoom_linear", base_value=7.0, low_value=6.0, high_value=8.0),
                FieldSchedule("outer_radius", "zoom_linear", base_value=21.0, low_value=18.0, high_value=24.0),
            ),
            family="timing",
            schedule_kind="mixed",
        )
        config = SmoothLifeConfig(grid_shape=(32, 32), evaluations_per_step=4, snapshot_interval=1, preset="search")
        search = SmoothLifeSearch(ackley, bounds=[(-10.0, 10.0), (-10.0, 10.0)], config=config, runtime_policy=policy)
        search.reset(seed=5)
        original_rebuild = search._rebuild_kernels
        search._rebuild_kernels = MagicMock(wraps=original_rebuild)
        search.run(evaluations=40)
        self.assertEqual(search._rebuild_kernels.call_count, 1)
        self.assertGreater(float(search.config.diffusion), 0.0)

    def test_late_stage_reactive_allowed_in_relevant_scopes(self) -> None:
        kinds = tuning_module.FIELD_SCOPE_SCHEDULE_KINDS
        self.assertIn("late_stage_reactive", kinds["sls_per_step"])
        self.assertIn("late_stage_reactive", kinds["agsls_decision"])

    def test_cadence_families_registered(self) -> None:
        registry = tuning_module.FAMILY_REGISTRY
        self.assertIn("smoothlife_cadence", registry)
        self.assertIn("agsls_cadence", registry)
        self.assertIn("late_stage_microgrid", registry)
        self.assertIn("late_stage_translation", registry)
        self.assertEqual(registry["smoothlife_cadence"].family_scope, "sls_per_step")
        self.assertEqual(registry["agsls_cadence"].family_scope, "agsls_decision")
        self.assertEqual(registry["late_stage_microgrid"].family_scope, "static_only")
        self.assertEqual(registry["late_stage_translation"].family_scope, "static_only")
        self.assertEqual(registry["smoothlife_cadence"].applicable_variants, ("agsls",))
        self.assertEqual(registry["agsls_cadence"].applicable_variants, ("agsls",))
        self.assertEqual(registry["late_stage_microgrid"].applicable_variants, ("agsls",))
        self.assertEqual(registry["late_stage_translation"].applicable_variants, ("agsls",))
        self.assertIn("late_stage_reactive", registry["smoothlife_cadence"].schedule_kinds)
        self.assertIn("late_stage_reactive", registry["agsls_cadence"].schedule_kinds)

    def test_decision_gap_reactive_schedule_uses_basin_count_and_gap(self) -> None:
        schedule = FieldSchedule(
            "candidate_probe_evaluations",
            "decision_gap_reactive",
            base_value=16,
            low_value=4,
            high_value=32,
            plateau_threshold=0.05,
        )
        baseline = RuntimeSignals(
            step_index=0,
            zoom_index=0,
            evaluations=0,
            remaining_budget=100,
            total_budget=100,
            explored_fraction=0.0,
            current_box_widths=(1.0, 1.0),
            best_value=1.0,
            local_best_value=1.0,
            global_improvement=0.0,
            stage_improvement=0.0,
            basin_count=1,
            top_basin_score_gap=0.0,
            max_zoom_cycles=5,
        )
        crowded = replace(baseline, basin_count=3, top_basin_score_gap=0.01)
        separated = replace(baseline, basin_count=3, top_basin_score_gap=0.20)
        self.assertEqual(schedule.evaluate(baseline), 16)
        self.assertEqual(schedule.evaluate(crowded), 32)
        self.assertEqual(schedule.evaluate(separated), 4)


class TestTuningStudy(unittest.TestCase):
    def test_ranking_ignores_wall_time_for_ordering(self) -> None:
        rows = [
            {
                "config_id": "b",
                "median_best_value": 1.0,
                "success_rate": 0.5,
                "iqr_width": 0.25,
                "mean_wall_time_s": 0.01,
                "maximize": False,
            },
            {
                "config_id": "a",
                "median_best_value": 1.0,
                "success_rate": 0.5,
                "iqr_width": 0.25,
                "mean_wall_time_s": 999.0,
                "maximize": False,
            },
        ]
        ranked = tuning_module._rank_bucket_rows(rows, "variant_rank")
        self.assertEqual([row["config_id"] for row in ranked], ["a", "b"])

    def test_resume_false_restarts_study_without_reusing_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir)
            initial = run_tuning_study(_tiny_spec(output_dir, workers=1))
            restarted = run_tuning_study(_tiny_spec(output_dir, workers=1, resume=False))
            self.assertGreater(initial.total_trials, 0)
            self.assertEqual(restarted.skipped_trials, 0)
            self.assertGreater(restarted.completed_trials, 0)
            self.assertEqual(len(_trial_rows(restarted.trials_path)), restarted.total_trials)
            self.assertEqual(restarted.total_trials, initial.total_trials)

    def test_family_manifest_covers_all_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            summary = run_tuning_study(_tiny_spec(Path(tempdir), workers=1))
            manifest = _json_file(summary.family_manifest_path)
            self.assertEqual(set(manifest["smoothlife"]), set(SmoothLifeConfig.__dataclass_fields__))
            self.assertEqual(set(manifest["agsls"]), set(AGSLSConfig.__dataclass_fields__))
            self.assertIn("late_stage_microgrid", manifest["families"])
            self.assertIn("late_stage_translation", manifest["families"])
            for group_name in ("smoothlife", "agsls"):
                for payload in manifest[group_name].values():
                    self.assertIn("scope", payload)
                    self.assertIn("compatible_schedule_kinds", payload)
                    self.assertIn("adaptive_variants", payload)

    def test_sls_adaptive_screen_excludes_non_runtime_families(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            summary = run_tuning_study(_tiny_spec(Path(tempdir), workers=1))
            trials = _trial_rows(summary.trials_path)
            sls_stage3_families = {
                str(row["family"])
                for row in trials
                if str(row["stage_name"]) == "stage3_adaptive_screen" and str(row["variant"]) == "sls"
            }
            self.assertIn("diffusion", sls_stage3_families)
            self.assertIn("objective_guidance", sls_stage3_families)
            self.assertNotIn("transition_window", sls_stage3_families)
            self.assertNotIn("basin_quantile", sls_stage3_families)

    def test_adaptive_classification_covers_required_helpful_constant_and_not_runtime(self) -> None:
        spec = StudySpec(
            output_dir=Path("unused"),
            adaptive_confirmation_budgets=(3200,),
            parameter_families=("diffusion", "objective_guidance", "basin_quantile", "transition_window"),
            preflight=False,
        )
        paired_trials: list[dict[str, object]] = []
        for objective in spec.objectives:
            for seed in range(2):
                paired_trials.append(
                    _paired_row(
                        variant="sls",
                        family="diffusion",
                        family_scope="sls_per_step",
                        objective=objective,
                        seed=seed,
                        improvement=0.5,
                    )
                )
        for objective in spec.objectives:
            for seed, improvement in enumerate((0.2, 0.2, -0.01, -0.01)):
                paired_trials.append(
                    _paired_row(
                        variant="agsls",
                        family="basin_quantile",
                        family_scope="agsls_decision",
                        objective=objective,
                        seed=seed,
                        improvement=improvement,
                    )
                )
        for objective in spec.objectives:
            for seed in range(2):
                paired_trials.append(
                    _paired_row(
                        variant="sls",
                        family="objective_guidance",
                        family_scope="sls_per_step",
                        objective=objective,
                        seed=seed,
                        improvement=-0.1,
                    )
                )
        rows = tuning_module._adaptive_required_rows(spec, paired_trials)
        by_key = {(str(row["variant"]), str(row["family"])): row for row in rows}
        self.assertEqual(by_key[("sls", "diffusion")]["classification"], "adaptive_required")
        self.assertEqual(by_key[("agsls", "basin_quantile")]["classification"], "adaptive_helpful")
        self.assertEqual(by_key[("sls", "objective_guidance")]["classification"], "constant_sufficient")
        self.assertEqual(by_key[("sls", "transition_window")]["classification"], "not_runtime_adaptable")

    def test_parallel_and_serial_studies_match_all_non_efficiency_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
            serial_summary = run_tuning_study(_tiny_spec(Path(serial_dir), workers=1))
            parallel_summary = run_tuning_study(_tiny_spec(Path(parallel_dir), workers=2))

            serial_trials = {row["trial_key"]: row for row in _trial_rows(serial_summary.trials_path)}
            parallel_trials = {row["trial_key"]: row for row in _trial_rows(parallel_summary.trials_path)}
            self.assertEqual(set(serial_trials), set(parallel_trials))
            for trial_key in sorted(serial_trials):
                self.assertAlmostEqual(float(serial_trials[trial_key]["best_value"]), float(parallel_trials[trial_key]["best_value"]), places=12)
                self.assertEqual(int(serial_trials[trial_key]["evaluations"]), int(parallel_trials[trial_key]["evaluations"]))
                self.assertEqual(bool(serial_trials[trial_key]["success"]), bool(parallel_trials[trial_key]["success"]))

            self.assertEqual(_artifact_texts(serial_summary), _artifact_texts(parallel_summary))
            self.assertTrue(serial_summary.runtime_efficiency_path.exists())
            self.assertTrue(parallel_summary.runtime_efficiency_path.exists())


if __name__ == "__main__":
    unittest.main()
