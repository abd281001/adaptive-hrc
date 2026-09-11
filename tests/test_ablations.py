"""Contracts for the matcher stress suite and the longitudinal arm grid."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.ablations import (
    ARMS,
    CONTRASTS,
    CORE_METRICS,
    GROUP_METRICS,
    INVARIANTS,
    LOCAL,
    PUBLISH,
    REPLAY,
    Arm,
    COMMIT_FULL,
    COMMIT_PROMOTION,
    COMMIT_TENTATIVE,
    MatcherSettings,
    GraphMatcher,
    arm_config,
    arms_by_name,
    build_partial_order_graph,
    check_invariants,
    declared_arms,
    default_matchers,
    generate_match_cases,
    group_arms,
    group_contrasts,
    group_metrics,
    groups,
    multiset_jaccard,
    paired_deltas,
    parse_commit_records,
    partial_order_similarity,
    roster_arms,
    run_matcher_ablation,
    summarize_commit_decisions,
    summarize_group,
    teaching_metrics,
    _holm,
    _validate_tables,
)


ABLATION_SCENARIOS = ("homogeneous", "heterogeneous", "holdout")


class ArmTableTests(unittest.TestCase):
    """The tables are the design; they must not be able to lie about it."""

    def test_tables_validate_at_import(self):
        _validate_tables()

    def test_no_declared_arm_duplicates_the_deployable_roster(self):
        from src.evaluation import DEFAULT_BASELINES, MEMORY_ORACLE

        roster = set(DEFAULT_BASELINES) | {MEMORY_ORACLE}
        self.assertEqual(set(arms_by_name()) & roster, set())

    def test_no_two_arms_are_the_same_condition(self):
        conditions = {}
        for arm in ARMS:
            key = (arm.agent, arm.route, tuple(sorted(arm.overrides.items())))
            self.assertNotIn(key, conditions, f"{arm.name} repeats {conditions.get(key)}")
            conditions[key] = arm.name

    def test_an_arm_shared_by_two_groups_is_declared_once(self):
        """The cross-group duplicates the old suites each recomputed."""
        shared = {arm.name: arm.groups for arm in ARMS if len(arm.groups) > 1}
        self.assertEqual(shared["full_no_latent_residual"], ("components", "latent"))
        self.assertEqual(shared["bc_adaptive"], ("memory", "representation"))

    def test_every_group_declares_exactly_one_primary_contrast(self):
        for group in groups():
            primary = [c.name for c in group_contrasts(group) if c.primary]
            self.assertEqual(len(primary), 1, f"{group}: {primary}")

    def test_contrasts_only_name_arms_that_exist(self):
        from src.evaluation import DEFAULT_BASELINES, MEMORY_ORACLE

        known = set(arms_by_name()) | set(DEFAULT_BASELINES) | {MEMORY_ORACLE}
        for contrast in CONTRASTS:
            self.assertIn(contrast.treatment, known, contrast.name)
            self.assertIn(contrast.reference, known, contrast.name)

    def test_a_metric_reads_one_direction_everywhere(self):
        directions = dict(CORE_METRICS)
        for extras in GROUP_METRICS.values():
            for metric, direction in extras:
                self.assertEqual(
                    directions.setdefault(metric, direction), direction, metric,
                )

    def test_the_latent_group_is_a_complete_two_by_two(self):
        """The deployed residual is a corner of the design, not beside it.

        Capacity x sequence alignment has four cells. 'full' is the shipped
        one -- low capacity with alignment on -- and it comes from the
        roster, so completing the design costs no extra run.
        """
        from src.models import DEFAULT_SETTINGS as D

        corners = {
            (arm.facets["capacity"], arm.facets["sequence_alignment"])
            for arm in ARMS if "latent" in arm.groups and "capacity" in arm.facets
        }
        self.assertEqual(corners, {("low", "off"), ("high", "off"), ("high", "on")})
        # The fourth corner is Full as deployed.
        self.assertTrue(D.latent_strategy_enabled)
        self.assertEqual(
            (D.latent_strategy_rank, D.latent_strategy_knn,
             D.latent_strategy_sequence_weight), (8, 3, 0.5),
        )
        self.assertIn("full", group_arms("latent"))

    def test_only_arms_that_can_retire_a_variant_get_a_local_route(self):
        """A retain-all method cannot route differently, so it has no local arm."""
        local = {arm.agent for arm in ARMS if arm.route == LOCAL}
        self.assertEqual(local, {"unpinned", "latest", "fixed"})

    def test_roster_arms_are_referenced_and_never_declared(self):
        referenced = set(roster_arms(groups()))
        self.assertEqual(referenced, {"full", "unpinned", "bc", "latest", "fixed"})
        self.assertEqual(referenced & set(arms_by_name()), set())


class ArmConfigTests(unittest.TestCase):
    def _config(self):
        from src.evaluation import EvalSettings

        return EvalSettings(seeds=(1337,), scenarios=("homogeneous",),
                            baselines=("full",), model_settings={"irl_l2": 0.5})

    def test_overrides_are_layered_onto_the_shared_model_settings(self):
        arm = arms_by_name()["full_no_pin"]
        config = arm_config(arm, self._config())
        self.assertEqual(config.model_settings["irl_l2"], 0.5)
        self.assertIs(config.model_settings["pin_latest"], False)

    def test_a_replaying_arm_keeps_shared_routing(self):
        config = arm_config(arms_by_name()["full_no_pin"], self._config())
        self.assertTrue(config.shared_routing)
        self.assertFalse(config.allow_repeat_observation)

    def test_a_local_arm_is_free_to_request_its_own_observations(self):
        config = arm_config(arms_by_name()["unpinned_local"], self._config())
        self.assertFalse(config.shared_routing)
        self.assertTrue(config.observe_missing_recipes)
        self.assertTrue(config.allow_repeat_observation)

    def test_nothing_that_determines_the_plan_is_changed(self):
        base = self._config()
        for arm in ARMS:
            with self.subTest(arm=arm.name):
                config = arm_config(arm, base)
                for field in ("seeds", "scenarios", "recipe_count", "schedule"):
                    self.assertEqual(getattr(config, field), getattr(base, field))


class ContrastSummaryTests(unittest.TestCase):
    @staticmethod
    def _rows(values):
        return [
            {"arm": arm, "scenario": "homogeneous", "seed": seed, **metrics}
            for arm, per_seed in values.items()
            for seed, metrics in per_seed.items()
        ]

    def test_deltas_are_signed_so_positive_always_favours_the_treatment(self):
        contrast = next(c for c in CONTRASTS if c.name == "cost_of_removing_latest_pin")
        rows = self._rows({
            "full": {1: {"teacher_forced_top_1": 0.9, "normalized_human_action_load": 0.5},
                     2: {"teacher_forced_top_1": 0.9, "normalized_human_action_load": 0.5}},
            "full_no_pin": {1: {"teacher_forced_top_1": 0.8, "normalized_human_action_load": 0.6},
                            2: {"teacher_forced_top_1": 0.8, "normalized_human_action_load": 0.6}},
        })
        entry = paired_deltas(rows, contrast, CORE_METRICS, "homogeneous")
        # Higher-is-better passes through.
        self.assertAlmostEqual(
            entry["metrics"]["teacher_forced_top_1"]["mean_treatment_advantage"], 0.1)
        # Lower-is-better is flipped: Full needing less human work is a gain.
        self.assertAlmostEqual(
            entry["metrics"]["normalized_human_action_load"]["mean_treatment_advantage"], 0.1)

    def test_only_seeds_present_in_both_arms_are_paired(self):
        contrast = next(c for c in CONTRASTS if c.name == "cost_of_removing_latest_pin")
        rows = self._rows({
            "full": {1: {"live_top_1": 0.9}, 2: {"live_top_1": 0.9}, 3: {"live_top_1": 0.9}},
            "full_no_pin": {1: {"live_top_1": 0.8}, 2: {"live_top_1": 0.8}},
        })
        entry = paired_deltas(rows, contrast, CORE_METRICS, "homogeneous")
        self.assertEqual(entry["n_paired_seeds"], 2)
        self.assertEqual(entry["metrics"]["live_top_1"]["n_paired_seeds"], 2)

    def test_secondary_contrasts_share_one_multiplicity_family(self):
        rows = self._rows({
            arm: {seed: {metric: 0.5 for metric, _d in group_metrics("components")}
                  for seed in range(1, 9)}
            for arm in group_arms("components")
        })
        summary = summarize_group("components", rows, ("homogeneous",))
        self.assertEqual(summary["primary_contrast"], "cost_of_removing_latest_pin")
        for entry in summary["contrasts"]:
            for metric in entry["metrics"].values():
                if entry["primary"]:
                    self.assertNotIn("sign_flip_p_two_sided_holm", metric)
                else:
                    self.assertIn("sign_flip_p_two_sided_holm", metric)

    def test_holm_is_monotone_and_never_reduces_a_p_value(self):
        raw = [0.001, 0.02, 0.04, 0.5]
        adjusted = _holm(raw)
        self.assertEqual(adjusted, sorted(adjusted))
        for before, after in zip(raw, adjusted):
            self.assertGreaterEqual(after, before)


class InvariantTests(unittest.TestCase):
    @staticmethod
    def _component_rows(**overrides):
        state = {
            "full": (True, True, True),
            "full_no_pin": (False, True, True),
            "full_no_semantic_fallback": (True, False, True),
            "full_no_latent_residual": (True, True, False),
            "unpinned": (False, False, False),
        }
        state.update(overrides)
        return [
            {"arm": arm, "scenario": "homogeneous", "seed": seed,
             "latest_pin_enabled": pin, "semantic_fallback_enabled": semantic,
             "latent_strategy_enabled": latent, "memory_policy": "adaptive"}
            for arm, (pin, semantic, latent) in state.items() for seed in (1, 2)
        ]

    def test_an_arm_that_moved_its_declared_component_passes(self):
        checks = check_invariants("components", self._component_rows())
        self.assertTrue(all(check["holds"] for check in checks), checks)

    def test_an_arm_that_moved_the_wrong_component_is_caught(self):
        rows = self._component_rows(full_no_pin=(True, True, True))
        checks = {c["name"]: c for c in check_invariants("components", rows)}
        failed = checks["each_component_arm_moved_exactly_its_own_component"]
        self.assertFalse(failed["holds"])
        self.assertTrue(any("full_no_pin" in p for p in failed["problems"]))

    def test_arms_that_must_agree_are_checked_against_each_other(self):
        rows = [
            {"arm": "full", "scenario": "homogeneous", "seed": 1,
             "memory_policy": "adaptive", "latest_pin_enabled": True},
            {"arm": "bc_adaptive", "scenario": "homogeneous", "seed": 1,
             "memory_policy": "none", "latest_pin_enabled": True},
        ]
        checks = {c["name"]: c for c in check_invariants("memory", rows)}
        self.assertFalse(checks["adaptive_memory_level_is_internally_consistent"]["holds"])


class ArmReuseTests(unittest.TestCase):
    """An arm is run once and read by every group that names it."""

    @staticmethod
    def _config(root, **overrides):
        from src.evaluation import EvalSettings, ScheduleSettings

        values = dict(
            seeds=(1337,), scenarios=("homogeneous",),
            baselines=("full", "unpinned"), include_oracle=False,
            recipe_count=2,
            schedule=ScheduleSettings(panel_size=2, phases=2, demos=24,
                                      min_recipes=1, max_recipes=2),
            audit_period=0, show_eta=False, workers=1, output=str(root),
            model_settings={"irl_cold_steps": 1, "irl_warm_steps": 1},
        )
        values.update(overrides)
        return EvalSettings(**values)

    def test_ablation_arms_join_the_standard_run_and_reuse_the_roster(self):
        from src.ablations import run_ablations
        from src.evaluation import run_evaluation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = run_evaluation(self._config(root, experiment="standard"))
            run_dir = Path(suite["run_dir"])
            roster_cells = {
                path.relative_to(run_dir).parts[1]
                for path in run_dir.glob("baselines/*/scenarios/*/seeds/*/summary.json")
            }
            self.assertEqual(roster_cells, {"full", "unpinned"})

            report = run_ablations(
                ("components",), self._config(root, experiment="ablation"),
                workers=1, progress=False,
            )
            # The same directory: the arm list is not part of the run identity.
            self.assertEqual(report["execution"]["run"], run_dir.name)
            # Full and unpinned are read, never re-run.
            self.assertEqual(
                set(report["execution"]["roster_arms_reused"]), {"full", "unpinned"},
            )
            self.assertNotIn("full", report["execution"]["cells_executed"])
            self.assertNotIn("unpinned", report["execution"]["cells_executed"])
            self.assertEqual(
                set(report["execution"]["cells_executed"]),
                {"full_no_pin", "full_no_semantic_fallback", "full_no_latent_residual"},
            )
            self.assertTrue(report["complete"])
            self.assertTrue(report["invariants_hold"])

            # Running again executes nothing at all.
            again = run_ablations(
                ("components",), self._config(root, experiment="ablation"),
                workers=1, progress=False,
            )
            self.assertEqual(
                set(again["execution"]["cells_executed"].values()), {0},
            )

    def test_ablating_without_the_standard_run_fails_loudly(self):
        from src.ablations import run_ablations

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "no evaluation run matches"):
                run_ablations(
                    ("components",), self._config(Path(directory)),
                    workers=1, progress=False,
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




if __name__ == "__main__":
    unittest.main()
