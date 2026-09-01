"""Reviewer-facing contracts for matcher and self-training audits."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ablations import (
    COMMIT_FULL,
    COMMIT_PROMOTION,
    COMMIT_TENTATIVE,
    MatcherSettings,
    GraphMatcher,
    LATENT_STRATEGY_ABLATION_ARMS,
    LOCAL_ROUTE,
    SHARED_ROUTE,
    build_partial_order_graph,
    parse_commit_records,
    default_matchers,
    generate_match_cases,
    multiset_jaccard,
    partial_order_similarity,
    run_matcher_ablation,
    run_routing_ablation,
    latent_strategy_ablation_design,
    summarize_commit_decisions,
    summarize_latent_strategy_ablation,
    summarize_routing_ablation,
    teaching_metrics,
    _trend,
)


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
