"""Reviewer-facing contracts for matcher and self-training audits."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.ablations import (
    COMPONENT_ABLATION_ARMS,
    _arm_cell_job,
    _arm_suite_specs,
    run_arm_major_suite,
    REPRESENTATION_ABLATION_ARMS,
    RETENTION_ABLATION_ARMS,
    COMMIT_FULL,
    COMMIT_PROMOTION,
    COMMIT_TENTATIVE,
    MatcherSettings,
    GraphMatcher,
    LATENT_STRATEGY_ABLATION_ARMS,
    MEMORY_PREDICTOR_ABLATION_CELLS,
    MEMORY_PREDICTOR_ABLATION_METRICS,
    LOCAL_ROUTE,
    SHARED_ROUTE,
    build_partial_order_graph,
    parse_commit_records,
    default_matchers,
    generate_match_cases,
    multiset_jaccard,
    partial_order_similarity,
    run_matcher_ablation,
    run_all_ablations,
    run_routing_ablation,
    latent_strategy_ablation_design,
    summarize_commit_decisions,
    summarize_latent_strategy_ablation,
    summarize_memory_predictor_ablation,
    summarize_routing_ablation,
    teaching_metrics,
    _trend,
)


ABLATION_SCENARIOS = ("homogeneous", "heterogeneous", "holdout")


def _ablation_payload(name, seeds, scenarios=ABLATION_SCENARIOS):
    """A minimal suite output that satisfies the collection validator."""
    arms = {arm.name for arm in LATENT_STRATEGY_ABLATION_ARMS}
    if name == "representation":
        return {
            "rows": [
                {"scenario": scenario, "seed": seed, "arm": arm.name}
                for scenario in scenarios for seed in seeds
                for arm in REPRESENTATION_ABLATION_ARMS
            ],
            "summary": {"memory_policy_held_fixed": True},
        }
    if name == "matcher":
        return {"summary_by_matcher": {"graph": {}}}
    if name == "routing":
        return {
            "rows": [
                {"scenario": scenario, "seed": seed, "trend": [{}]}
                for scenario in scenarios for seed in seeds
            ]
        }
    if name == "memory":
        return {
            "rows": [
                {
                    "scenario": scenario, "seed": seed,
                    "arm": str(cell["arm"]),
                    "baseline": str(cell["baseline"]),
                    "memory_level": str(cell["memory"]),
                    "memory_policy": (
                        "adaptive"
                        if cell["memory"] == "adaptive_pinned" else "none"
                    ),
                    "latest_pin_enabled": (
                        cell["memory"] == "adaptive_pinned"
                    ),
                }
                for scenario in scenarios for seed in seeds
                for cell in MEMORY_PREDICTOR_ABLATION_CELLS
            ],
            "summary": {
                "memory_levels_are_internally_consistent": True,
                "maxent_cells_share_predictor_support": True,
            },
        }
    if name == "components":
        return {
            "rows": [
                {"scenario": scenario, "seed": seed, "arm": arm.name}
                for scenario in scenarios for seed in seeds
                for arm in COMPONENT_ABLATION_ARMS
            ],
            "summary": {
                "declared_factors_were_applied": True,
                "retention_held_constant": True,
            },
        }
    if name == "retention":
        return {
            "rows": [
                {"scenario": scenario, "seed": seed, "arm": arm.name}
                for scenario in scenarios for seed in seeds
                for arm in RETENTION_ABLATION_ARMS
            ],
            "summary": {"predictor_support_held_constant": True},
        }
    return {
        "rows": [
            {"scenario": scenario, "seed": seed, "arm": arm}
            for scenario in scenarios for seed in seeds for arm in arms
        ]
    }


class AllAblationRunnerTests(unittest.TestCase):
    def test_collection_runner_owns_outputs_manifest_and_validation(self):
        from src.evaluation import PAPER_SEEDS

        scenarios = ABLATION_SCENARIOS
        seeds = tuple(int(seed) for seed in PAPER_SEEDS)

        def complete(name, command, run_dir, _environment):
            payload = _ablation_payload(name, seeds, scenarios)
            output = Path(run_dir) / f"{name}.json"
            output.write_text(json.dumps(payload), encoding="utf-8")
            return {
                "name": name,
                "command": list(command),
                "started_at_utc": "2026-01-01T00:00:00Z",
                "completed_at_utc": "2026-01-01T00:00:01Z",
                "return_code": 0,
                "wall_s": 1.0,
                "output": str(output),
                "stdout": str(Path(run_dir) / f"{name}.stdout.log"),
                "stderr": str(Path(run_dir) / f"{name}.stderr.log"),
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "src.ablations._run_ablation_process", side_effect=complete,
        ):
            result = run_all_ablations(directory, workers=1)
            manifest = json.loads(Path(result["manifest"]).read_text())

        self.assertEqual(result["state"], "complete")
        expected_suites = {
            "matcher", "routing", "latent", "memory", "representation",
            "components", "retention",
        }
        self.assertEqual(set(result["outputs"]), expected_suites)
        self.assertEqual(manifest["state"], "complete")
        self.assertEqual(set(manifest["jobs"]), expected_suites)
        self.assertEqual(manifest["longitudinal_scenarios"], list(scenarios))
        self.assertEqual(manifest["longitudinal_seeds"], list(seeds))

    def test_suites_run_one_at_a_time_over_the_full_paired_seed_grid(self):
        """Each suite gets the machine to itself for its own seed grid.

        Running all four suites at once would put unrelated suites in
        contention for the same performance cores, and the per-arm wall-clock
        numbers these suites report are meant to be comparable.
        """
        from src.evaluation import PAPER_SEEDS

        seeds = tuple(int(seed) for seed in PAPER_SEEDS)
        seed_csv = ",".join(str(seed) for seed in seeds)
        order: list = []
        live: list = []
        overlaps: list = []

        def record(name, command, run_dir, _environment):
            live.append(name)
            if len(live) > 1:
                overlaps.append(tuple(live))
            order.append((name, list(command)))
            payload = _ablation_payload(name, seeds)
            output = Path(run_dir) / f"{name}.json"
            output.write_text(json.dumps(payload), encoding="utf-8")
            live.remove(name)
            return {
                "name": name, "command": list(command),
                "started_at_utc": "2026-01-01T00:00:00Z",
                "completed_at_utc": "2026-01-01T00:00:01Z",
                "return_code": 0, "wall_s": 1.0,
                "output": str(output),
                "stdout": str(Path(run_dir) / f"{name}.stdout.log"),
                "stderr": str(Path(run_dir) / f"{name}.stderr.log"),
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "src.ablations._run_ablation_process", side_effect=record,
        ):
            result = run_all_ablations(directory, workers=8)
            manifest = json.loads(Path(result["manifest"]).read_text())

        self.assertEqual(overlaps, [], "suites overlapped instead of running in turn")
        self.assertEqual(manifest["suite_execution"], "sequential")
        self.assertEqual(manifest["suite_parallelism"], 1)
        self.assertEqual(manifest["longitudinal_workers_per_suite"], 8)

        commands = dict(order)
        for suite, flag in (
            ("routing", "--routing-seeds"),
            ("latent", "--latent-seeds"),
            ("memory", "--memory-seeds"),
            ("components", "--components-seeds"),
            ("retention", "--retention-seeds"),
        ):
            command = commands[suite]
            self.assertEqual(command[command.index(flag) + 1], seed_csv)
            self.assertEqual(command[command.index("--workers") + 1], "8")
        # The matcher suite generates one stress dataset rather than a paired
        # seed grid, so it keeps its single generation seed.
        self.assertNotIn("--routing-seeds", commands["matcher"])
        self.assertIn("--seed", commands["matcher"])

    def test_worker_count_may_span_the_whole_scenario_seed_grid(self):
        from src.evaluation import PAPER_SEEDS, SCENARIOS

        maximum = len(SCENARIOS) * len(PAPER_SEEDS)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, f"between 1 and {maximum}"):
                run_all_ablations(directory, workers=maximum + 1)


class MemoryPredictorFactorialAblationTests(unittest.TestCase):
    def test_cells_cross_both_factors_exactly_once(self):
        cells = MEMORY_PREDICTOR_ABLATION_CELLS
        self.assertEqual(len(cells), 4)
        self.assertEqual(
            {(cell["predictor"], cell["memory"]) for cell in cells},
            {
                ("maxent", "adaptive_pinned"),
                ("maxent", "retain_all"),
                ("behavior_cloning", "adaptive_pinned"),
                ("behavior_cloning", "retain_all"),
            },
        )
        # 'full' must lead: it defines the shared interaction route the other
        # three cells replay, so the comparison stays paired.
        self.assertEqual(cells[0]["arm"], "full")
        # Two cells share the 'full' baseline and differ only in their
        # settings overrides, so the arm name is the cell identity.
        self.assertEqual(
            {cell["arm"] for cell in cells},
            {"full", "bc_adaptive", "maxent_retain_all", "bc"},
        )
        self.assertEqual(
            {cell["baseline"] for cell in cells},
            {"full", "bc_adaptive", "bc"},
        )

    def test_both_maxent_cells_keep_fulls_predictor_support(self):
        """The repair: the memory factor must not move the semantic components.

        The MaxEnt retain-all cell used to be the `no_decay` baseline, which is
        built through `_without_proposed_components` and therefore also dropped
        the semantic fallback and the latent residual. That left the MaxEnt
        simple effect and the interaction term confounded.
        """
        from dataclasses import replace as replace_settings
        from src.models import DEFAULT_SETTINGS

        maxent = [
            cell for cell in MEMORY_PREDICTOR_ABLATION_CELLS
            if cell["predictor"] == "maxent"
        ]
        self.assertEqual(len(maxent), 2)
        for cell in maxent:
            with self.subTest(arm=cell["arm"]):
                settings = replace_settings(
                    DEFAULT_SETTINGS, **dict(cell.get("overrides") or {}),
                )
                self.assertTrue(settings.semantic_fallback_enabled)
                self.assertTrue(settings.latent_strategy_enabled)

    def test_registered_cells_agree_with_the_agents_they_name(self):
        """The design is only meaningful if the arms behave as labelled."""
        from dataclasses import replace
        from src.baselines import BASELINE_AGENTS
        from src.models import Settings

        config = Settings(
            verbose=False, irl_cold_steps=2, irl_warm_steps=1,
            bc_cold_epochs=2, bc_warm_epochs=1,
        )
        expected_policy = {"adaptive_pinned": "adaptive", "retain_all": "none"}
        for cell in MEMORY_PREDICTOR_ABLATION_CELLS:
            with self.subTest(arm=cell["arm"]):
                # A cell is its baseline *plus* its settings overrides; the
                # MaxEnt row distinguishes its two cells that way.
                cell_config = replace(config, **dict(cell.get("overrides") or {}))
                if cell["baseline"] == "full":
                    from src.adaptive_agent import AdaptiveAgent
                    agent = AdaptiveAgent(settings=cell_config)
                else:
                    agent = BASELINE_AGENTS[cell["baseline"]](cell_config)
                self.assertEqual(
                    agent.replay.policy, expected_policy[cell["memory"]],
                )
                self.assertEqual(
                    agent.settings.pin_latest,
                    cell["memory"] == "adaptive_pinned",
                )
                self.assertEqual(
                    agent.predictor_name().startswith("behavior_cloning"),
                    cell["predictor"] == "behavior_cloning",
                )

    def test_summary_signs_every_effect_so_positive_favours_the_treatment(self):
        def row(scenario, seed, arm, memory, top_1, load):
            return {
                "scenario": scenario, "seed": seed, "arm": arm,
                "memory_level": memory,
                "memory_policy": (
                    "adaptive" if memory == "adaptive_pinned" else "none"
                ),
                "latest_pin_enabled": memory == "adaptive_pinned",
                "teacher_forced_top_1": top_1,
                "normalized_human_action_load": load,
            }

        # Memory helps both predictors by the same amount, so the interaction
        # is zero even though the cloner is the more accurate model here.
        rows = [
            row("homogeneous", 7, "full", "adaptive_pinned", 0.90, 0.50),
            row("homogeneous", 7, "maxent_retain_all", "retain_all", 0.85, 0.55),
            row("homogeneous", 7, "bc_adaptive", "adaptive_pinned", 0.95, 0.45),
            row("homogeneous", 7, "bc", "retain_all", 0.90, 0.50),
        ]
        summary = summarize_memory_predictor_ablation(rows)
        effects = {
            entry["name"]: entry["metrics"]
            for entry in summary["paired_seed_simple_effects"]
        }

        # Higher-is-better metric passes through unchanged.
        self.assertAlmostEqual(
            effects["memory_effect_within_maxent"]["teacher_forced_top_1"][
                "mean_treatment_advantage"
            ], 0.05,
        )
        # Lower-is-better metric is flipped, so a drop in human load is a gain.
        self.assertAlmostEqual(
            effects["memory_effect_within_behavior_cloning"][
                "normalized_human_action_load"
            ]["mean_treatment_advantage"], 0.05,
        )
        # BC is ahead under a matched memory policy, so Full's simple effect
        # on the predictor factor is negative rather than silently clipped.
        self.assertAlmostEqual(
            effects["predictor_effect_under_adaptive_memory"][
                "teacher_forced_top_1"
            ]["mean_treatment_advantage"], -0.05,
        )

        interaction = summary["memory_by_predictor_interaction"][0]["metrics"]
        for metric, _direction in MEMORY_PREDICTOR_ABLATION_METRICS[:1]:
            self.assertAlmostEqual(
                interaction[metric]["mean_interaction"], 0.0,
            )
        self.assertTrue(summary["memory_levels_are_internally_consistent"])

    def test_summary_flags_a_cell_that_did_not_run_its_declared_policy(self):
        rows = [
            {
                "scenario": "homogeneous", "seed": 7, "baseline": "full",
                "memory_level": "adaptive_pinned",
                "memory_policy": "adaptive", "latest_pin_enabled": True,
            },
            {
                "scenario": "homogeneous", "seed": 7, "baseline": "bc_adaptive",
                "memory_level": "adaptive_pinned",
                # Regression guard: the arm silently kept BC's old storage.
                "memory_policy": "none", "latest_pin_enabled": False,
            },
        ]
        summary = summarize_memory_predictor_ablation(rows)
        self.assertFalse(summary["memory_levels_are_internally_consistent"])


class LatentStrategyComponentAblationTests(unittest.TestCase):
    def test_four_arms_encode_the_controlled_component_design(self):
        arms = {arm.name: arm for arm in LATENT_STRATEGY_ABLATION_ARMS}

        self.assertEqual(
            set(arms),
            {
                "maxent_only",
                "latent_timing_lightweight",
                "latent_timing_capacity_matched",
                "latent_timing_trajectory_hybrid",
            },
        )
        self.assertFalse(arms["maxent_only"].latent_strategy_enabled)
        self.assertEqual(
            arms["latent_timing_lightweight"].model_overrides(),
            {
                "latent_strategy_enabled": True,
                "latent_strategy_rank": 8,
                "latent_strategy_knn": 3,
                "latent_strategy_strength": 1.0,
                "latent_strategy_sequence_weight": 0.0,
            },
        )
        timing = arms["latent_timing_capacity_matched"].model_overrides()
        hybrid = arms["latent_timing_trajectory_hybrid"].model_overrides()
        self.assertEqual(
            {
                key: value for key, value in timing.items()
                if key != "latent_strategy_sequence_weight"
            },
            {
                key: value for key, value in hybrid.items()
                if key != "latent_strategy_sequence_weight"
            },
        )
        self.assertEqual(timing["latent_strategy_sequence_weight"], 0.0)
        self.assertEqual(hybrid["latent_strategy_sequence_weight"], 0.5)

    def test_design_marks_only_capacity_matched_trajectory_contrast(self):
        design = latent_strategy_ablation_design()
        contrast = next(
            row for row in design["planned_contrasts"]
            if row["name"] == "trajectory_alignment_contribution"
        )

        self.assertEqual(
            contrast["reference"], "latent_timing_capacity_matched",
        )
        self.assertEqual(
            contrast["treatment"], "latent_timing_trajectory_hybrid",
        )
        self.assertIn("Do not attribute", design["invalid_causal_contrast"])

    def test_summary_uses_paired_treatment_minus_reference_deltas(self):
        rows = []
        for seed, timing, hybrid in ((1, 0.60, 0.65), (2, 0.70, 0.73)):
            for arm, score in (
                ("maxent_only", timing - 0.04),
                ("latent_timing_lightweight", timing - 0.02),
                ("latent_timing_capacity_matched", timing),
                ("latent_timing_trajectory_hybrid", hybrid),
            ):
                rows.append({
                    "scenario": "homogeneous",
                    "seed": seed,
                    "arm": arm,
                    "teacher_forced_top_1": score,
                })

        summary = summarize_latent_strategy_ablation(rows)
        trajectory = next(
            row for row in summary["paired_seed_contrasts"]
            if row["name"] == "trajectory_alignment_contribution"
        )

        self.assertEqual(trajectory["n_paired_seeds"], 2)
        self.assertAlmostEqual(
            trajectory["metrics"]["teacher_forced_top_1"]
            ["mean_treatment_minus_reference"],
            0.04,
        )


class AblationTrendTests(unittest.TestCase):
    def test_trend_logs_each_ladder_phase(self):
        stream = SimpleNamespace(
            episode_rows=[
                {
                    "phase_index": 0, "event_index": 0, "mode": "observe",
                    "recipe_steps": 4,
                },
                {
                    "phase_index": 0, "event_index": 1, "mode": "assist",
                    "teacher_forced_correct_count": 2,
                    "teacher_forced_prediction_count": 4,
                    "teacher_forced_total_nll": 3.0,
                },
                {
                    "phase_index": 1, "event_index": 2, "mode": "assist",
                    "teacher_forced_correct_count": 3,
                    "teacher_forced_prediction_count": 4,
                    "teacher_forced_total_nll": 2.0,
                },
            ],
            frozen_rows=[
                {"event_index": 1, "checkpoint": "event_1", "top_1": 0.5},
                {"event_index": 2, "checkpoint": "event_2", "top_1": 0.75},
            ],
            memory_rows=[
                {"event_index": 1, "active_variants": 1, "pruned_variants": 0},
                {"event_index": 2, "active_variants": 2, "pruned_variants": 0},
            ],
        )
        trend = _trend(stream)
        self.assertEqual([row["phase"] for row in trend], [0, 1])
        self.assertEqual(trend[0]["phase_top_1"], 0.5)
        self.assertEqual(trend[1]["phase_top_1"], 0.75)
        self.assertEqual(trend[1]["top_1_seen"], 0.625)
        self.assertEqual(trend[1]["active_variants"], 2)

class TeachingBurdenRoutingAblationTests(unittest.TestCase):
    def test_burden_separates_planned_demos_extra_demos_and_corrections(self):
        episodes = [
            {
                "mode": "observe",
                "requested_mode": "observe",
                "recipe_steps": 10,
            },
            {
                "mode": "observe",
                "requested_mode": "assist",
                "recipe_steps": 5,
            },
            {
                "mode": "assist",
                "requested_mode": "assist",
                "recipe_steps": 8,
                "hrc_human_correction_count": 2,
                "hrc_human_turn_count": 2,
                "hrc_robot_turn_count": 6,
                "hrc_robot_correct_count": 4,
                "teacher_forced_prediction_count": 8,
                "teacher_forced_correct_count": 6,
            },
        ]

        metrics = teaching_metrics(
            episodes,
            [
                {"checkpoint": "phase_1", "top_1": 0.4, "top_k": 0.7,
                 "recipe_steps": 5, "hrc_human_turn_count": 1,
                 "hrc_human_correction_count": 3},
                {"checkpoint": "final", "top_1": 0.8, "top_k": 0.9,
                 "recipe_steps": 5, "hrc_human_turn_count": 2,
                 "hrc_human_correction_count": 1},
                {"checkpoint": "pre_event_2", "probe_phase": "pre_event",
                 "top_1": 0.0, "top_k": 0.0,
                 "recipe_steps": 5, "hrc_human_turn_count": 0,
                 "hrc_human_correction_count": 5},
            ],
        )

        self.assertEqual(metrics["n_planned_demonstration_actions"], 10.0)
        self.assertEqual(metrics["n_extra_demonstration_actions"], 5.0)
        self.assertEqual(metrics["n_corrective_teaching_actions"], 2.0)
        self.assertEqual(metrics["n_total_explicit_teaching_actions"], 17.0)
        self.assertAlmostEqual(metrics["fixed_checkpoint_top_1"], 0.6)
        self.assertAlmostEqual(
            metrics["fixed_checkpoint_corrections_per_recipe_step"], 0.4,
        )
        self.assertAlmostEqual(
            metrics["fixed_checkpoint_normalized_human_action_load"], 0.7,
        )
        self.assertAlmostEqual(
            metrics["fixed_checkpoint_mean_corrections_per_task"], 2.0,
        )
        self.assertEqual(metrics["fixed_checkpoint_n_rows"], 2)
        self.assertAlmostEqual(
            metrics["assist_live_top_1_selection_affected"], 4.0 / 6.0,
        )
        self.assertAlmostEqual(metrics["assist_teacher_forced_top_1"], 0.75)
        self.assertAlmostEqual(
            metrics["assist_normalized_human_action_load"], 0.5,
        )
        self.assertAlmostEqual(
            metrics["assist_mean_corrections_per_task"], 2.0,
        )
        self.assertAlmostEqual(
            metrics["assist_corrections_per_recipe_step"], 0.25,
        )

    def test_summary_reports_matched_local_minus_shared_burden(self):
        rows = []
        for seed in (1, 2):
            common = {
                "scenario": "test",
                "seed": seed,
                "baseline": "bc",
                "n_corrective_teaching_actions": 2.0,
            }
            rows.append({
                **common,
                "routing_condition": SHARED_ROUTE,
                "n_total_explicit_teaching_actions": 20.0,
                "n_extra_demonstration_actions": 0.0,
                "fixed_checkpoint_top_1": 0.5,
            })
            rows.append({
                **common,
                "routing_condition": LOCAL_ROUTE,
                "n_total_explicit_teaching_actions": 30.0,
                "n_extra_demonstration_actions": 10.0,
                "fixed_checkpoint_top_1": 0.6,
            })

        summary = summarize_routing_ablation(rows)
        contrast = summary["paired_seed_local_minus_shared"][0]

        self.assertEqual(contrast["n_paired_seeds"], 2)
        self.assertEqual(
            contrast["metrics"]["n_extra_demonstration_actions"]
            ["mean_local_minus_shared"],
            10.0,
        )
        self.assertAlmostEqual(
            contrast["metrics"]["fixed_checkpoint_top_1"]
            ["mean_local_minus_shared"],
            0.1,
        )
        self.assertAlmostEqual(
            contrast[
                "mean_fixed_checkpoint_top_1_gain_per_additional_demonstration_action"
            ],
            0.01,
        )

    def test_frozen_deployment_references_are_not_valid_local_teaching_arms(self):
        with self.assertRaisesRegex(ValueError, "frozen deployment references"):
            run_routing_ablation(
                baselines=("full", "frozen"),
            )


class DiscriminationAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = MatcherSettings(
            recipe_count=6,
            case_limit=60,
            overlap_repeats=2,
            include_prefixes=False,
            calibrate=False,
        )
        cls.cases = generate_match_cases(cls.config)

    def test_executable_task_variations_are_state_replayed_and_valid(self):
        families = {
            "same_recipe_optional_action_omitted",
            "new_recipe_one_action_added_valid",
            "new_recipe_one_action_substitution_valid",
        }
        for family in families:
            rows = [case for case in self.cases if case.family == family]
            self.assertTrue(rows, family)
            self.assertTrue(all(case.metadata["valid_ordering"] for case in rows), family)
            self.assertTrue(all(case.query_actions for case in rows))

    def test_recognition_and_execution_noise_are_distinct(self):
        recognition = [
            case for case in self.cases
            if case.metadata.get("case_category") == "recognition_corruption"
        ]
        execution = [
            case for case in self.cases
            if case.family == "same_recipe_execution_repeat"
        ]
        self.assertTrue(recognition)
        self.assertTrue(execution)
        self.assertTrue(all(case.metadata.get("physical_execution_valid") for case in recognition))
        self.assertTrue(all(case.metadata.get("expected_redundant_or_no_effect_action") for case in execution))

    def test_controlled_overlaps_are_exact_distinct_boundary_probes(self):
        controlled = [
            case for case in self.cases
            if case.family == "controlled_overlap_new_recipe"
        ]
        self.assertEqual(
            len(controlled),
            len(self.config.overlap_targets) * self.config.overlap_repeats,
        )
        for case in controlled:
            reference = case.library[0]
            actual = multiset_jaccard(case.query_actions, reference.actions)
            self.assertAlmostEqual(actual, case.metadata["actual_overlap"], places=12)
            self.assertLess(actual, 1.0)
            self.assertNotEqual(case.query_actions, reference.actions)
            self.assertFalse(case.metadata["primary_accuracy"])

    def test_preference_family_is_balanced_across_selected_recipes(self):
        recipes = {
            case.metadata["source_recipe"]
            for case in self.cases
            if case.family == "same_recipe_reordered_preference"
        }
        self.assertEqual(len(recipes), self.config.recipe_count)

    def test_all_requested_matcher_families_are_runnable(self):
        names = {matcher.name for matcher in default_matchers()}
        self.assertTrue({
            "current_jaccard_kendall",
            "edit_distance",
            "lcs",
            "probabilistic_recipe_belief",
            "partial_order_task",
        }.issubset(names))

    def test_partial_order_graph_penalizes_dependency_inversion(self):
        transfer = "transfer (tomato, from=storage, to=prep_station)"
        cut = "cut (tomato, prep_station)"
        independent = "transfer (pot, from=storage, to=cooking_station)"
        candidate = build_partial_order_graph((transfer, cut, independent))
        independent_reorder = build_partial_order_graph((independent, transfer, cut))
        dependency_inversion = build_partial_order_graph((cut, transfer, independent))
        support_ok, precedence_ok = partial_order_similarity(independent_reorder, candidate)
        support_bad, precedence_bad = partial_order_similarity(dependency_inversion, candidate)
        self.assertEqual(support_ok, 1.0)
        self.assertEqual(precedence_ok, 1.0)
        self.assertEqual(support_bad, 1.0)
        self.assertLess(precedence_bad, precedence_ok)
        self.assertIsInstance(GraphMatcher(), GraphMatcher)

    def test_primary_summary_excludes_controlled_and_prefix_cases(self):
        result = run_matcher_ablation(self.config)
        expected_primary = sum(
            bool(case.metadata.get("primary_accuracy", True))
            and not bool(case.metadata.get("prefix_case"))
            for case in self.cases
        )
        self.assertEqual(result["n_primary_cases"], expected_primary)
        self.assertEqual(result["n_controlled_boundary_cases"], 12)

    def test_baseline_operating_points_use_independent_calibration_cases(self):
        config = MatcherSettings(
            recipe_count=3,
            case_limit=2,
            overlap_repeats=1,
            include_prefixes=False,
            calibrate=True,
            calibration_case_limit=2,
        )
        result = run_matcher_ablation(config)
        calibration = result["baseline_calibration"]
        self.assertEqual(calibration["status"], "completed")
        self.assertNotEqual(calibration["seed"], config.seed)
        self.assertEqual(
            calibration["reports"]["current_jaccard_kendall"]["selection_data"],
            "fixed_production_operating_point",
        )
        self.assertEqual(
            calibration["reports"]["edit_distance"]["selection_data"],
            "independent_seed_primary_nonprefix_cases",
        )


class SemanticActionAndCommitAuditTests(unittest.TestCase):
    def test_discrimination_cases_use_semantic_actions_directly(self):
        cases = generate_match_cases(MatcherSettings(
            recipe_count=3,
            case_limit=2,
            overlap_repeats=1,
            include_prefixes=False,
            calibrate=False,
        ))
        case = next(
            case for case in cases
            if case.metadata.get("case_category") == "natural_task"
        )
        self.assertTrue(all("(" in action for action in case.query_actions))
        variant = case.library[0]
        self.assertEqual(variant.to_known_variant().ordering, variant.actions)

    def test_commit_metrics_cover_promotion_recovery_pin_and_calibration(self):
        rows = [
            {
                "event_index": 4,
                "mode": "assist",
                "commit_decision_id": "bad_tentative",
                "commit_decision": COMMIT_TENTATIVE,
                "commit_applied": True,
                "self_training_opportunity": True,
                "expected_recipe_id": "R1",
                "expected_variant_id": "v1",
                "commit_candidate_recipe_id": "R2",
                "commit_variant_id": "wrong",
                "commit_candidate_correct": False,
                "commit_confidence": 0.60,
                "latest_pinned": False,
            },
            {
                "event_index": 6,
                "mode": "assist",
                "commit_decision_id": "correct_promotion",
                "commit_decision": COMMIT_PROMOTION,
                "commit_applied": True,
                "self_training_opportunity": True,
                "expected_recipe_id": "R1",
                "expected_variant_id": "v1",
                "commit_candidate_recipe_id": "R1",
                "commit_variant_id": "v1",
                "commit_candidate_correct": True,
                "commit_confidence": 0.90,
                "promoted_from_tentative": True,
                "promotion_delay_demos": 2,
                "latest_pinned": True,
                "latest_pin_correct": True,
            },
            {
                "event_index": 7,
                "mode": "assist",
                "commit_decision": COMMIT_FULL,
                "commit_applied": True,
                "self_training_opportunity": True,
                "expected_recipe_id": "R3",
                "expected_variant_id": "v3",
                "commit_candidate_recipe_id": "R4",
                "commit_variant_id": "wrong",
                "commit_candidate_correct": False,
                "commit_confidence": 0.80,
                "latest_pinned": True,
                "latest_pin_correct": False,
            },
        ]
        summary = summarize_commit_decisions(parse_commit_records(rows))
        self.assertAlmostEqual(summary["commit_precision"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["commit_recall"], 1.0 / 3.0)
        self.assertEqual(summary["tentative_to_full_promotion_accuracy"], 1.0)
        self.assertEqual(summary["false_promotion_rate"], 0.0)
        self.assertEqual(summary["false_latest_pin_rate"], 0.5)
        self.assertEqual(summary["mean_time_to_promotion"], 2.0)
        self.assertEqual(summary["recovery_after_mistaken_commit_rate"], 0.5)
        self.assertEqual(summary["mean_recovery_steps_after_mistaken_commit"], 2.0)
        self.assertEqual(summary["calibration"]["status"], "completed")
        self.assertEqual(summary["calibration"]["n"], 3)


if __name__ == "__main__":
    unittest.main()


class ArmMajorAblationSchedulingTests(unittest.TestCase):
    """Suites run arm by arm; that must not move a single ablation number."""

    TIMING = re.compile(r"(wall_s|_at_utc|elapsed_s|latency|_per_s|_time|gflops_s)$")

    @classmethod
    def _without_timings(cls, value):
        if isinstance(value, dict):
            return {
                key: cls._without_timings(item)
                for key, item in value.items()
                if not cls.TIMING.search(key)
            }
        if isinstance(value, list):
            return [cls._without_timings(item) for item in value]
        return value

    @staticmethod
    def _config(experiment):
        from src.evaluation import EvalSettings, ScheduleSettings

        return EvalSettings(
            seeds=(1337,), scenarios=("homogeneous",), baselines=("full",),
            include_oracle=False, show_eta=False, recipe_count=2,
            schedule=ScheduleSettings(
                panel_size=2, phases=2, demos=24, min_recipes=1, max_recipes=2,
            ),
            audit_period=0, workers=1, experiment=experiment,
            model_settings={
                "irl_cold_steps": 1, "irl_warm_steps": 1,
                "bc_cold_epochs": 1, "bc_warm_epochs": 1,
            },
        )

    @classmethod
    def _cell_major(cls, suite, arms, config):
        """The previous nesting: one cell at a time, every arm inside it."""
        spec = _arm_suite_specs()[suite]
        rows = []
        for scenario in config.scenarios:
            for seed in config.seeds:
                route = None
                for arm in arms:
                    row, realized = _arm_cell_job(
                        (suite, str(scenario), int(seed), config, arm, route)
                    )
                    if realized is not None:
                        route = realized
                    rows.append(row)
        order = {spec.name(arm): index for index, arm in enumerate(arms)}
        rows.sort(key=lambda row: (
            str(row["scenario"]), int(row["seed"]), order[str(row["arm"])],
        ))
        return rows

    def test_arm_major_scheduling_matches_cell_major_scheduling(self):
        suites = (
            ("component_ablation", tuple(COMPONENT_ABLATION_ARMS)),
            ("latent_strategy_ablation", tuple(LATENT_STRATEGY_ABLATION_ARMS)),
            ("memory_predictor_ablation",
             tuple(dict(cell) for cell in MEMORY_PREDICTOR_ABLATION_CELLS)),
            ("representation_ablation", tuple(REPRESENTATION_ABLATION_ARMS)),
        )
        for suite, arms in suites:
            with self.subTest(suite=suite):
                config = self._config(suite)
                self.assertEqual(
                    self._without_timings(self._cell_major(suite, arms, config)),
                    self._without_timings(run_arm_major_suite(suite, arms, config)),
                )

    def test_each_arm_gets_its_own_result_file(self):
        arms = tuple(COMPONENT_ABLATION_ARMS)
        config = self._config("component_ablation")
        with tempfile.TemporaryDirectory() as directory:
            arms_dir = Path(directory) / "arms"
            rows = run_arm_major_suite(
                "component_ablation", arms, config, arms_dir=arms_dir,
            )
            written = {path.stem for path in arms_dir.iterdir()}
            self.assertEqual(written, {arm.name for arm in arms})
            split = [
                row
                for path in sorted(arms_dir.iterdir())
                for row in json.loads(path.read_text())["rows"]
            ]
            self.assertCountEqual(split, rows)
            # Exactly one arm defines the route the others replay.
            reference = [
                json.loads(path.read_text())["arm"]
                for path in arms_dir.iterdir()
                if json.loads(path.read_text())["reference_arm"]
            ]
            self.assertEqual(reference, ["full"])

    def test_an_arm_name_that_is_not_a_safe_filename_is_rejected(self):
        from src.ablations import _safe_arm_component

        self.assertEqual(_safe_arm_component("full_no_pin"), "full_no_pin")
        for bad in ("../escape", "with/slash", ""):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    _safe_arm_component(bad)
