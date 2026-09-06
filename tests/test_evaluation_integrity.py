"""Scientific and serialization contracts for the longitudinal evaluator."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.evaluation import (
    HOLDOUT_AXIS,
    HOLDOUT_SOURCES,
    HOLDOUT_TARGETS,
    HOLDOUT_VALUE,
    COMMIT_SENSITIVITY_SPECS,
    PAPER_SEEDS,
    TaskVariant,
    SCENARIOS,
    HOLDOUT,
    HETEROGENEOUS,
    HOMOGENEOUS,
    EvalSettings,
    Lifecycle,
    ScheduleSettings,
    Event,
    Plan,
    _climb_groups,
    _post_mismatch_accuracy,
    _evolve_active_set,
    _memory_state,
    _pair_key,
    _preference_cohort_tags,
    _sample_hetero_gap,
    _scope_metrics,
    summarize_adaptation,
    aggregate_episodes,
    assist_demo,
    build_settings,
    build_plan,
    summarize_recurrence,
    summarize_frozen,
    summarize_frozen_by,
    build_task,
    summarize_oracle,
    summarize_pairs,
    parse_args,
    summarize_cohorts,
    summarize_reentry,
    run_plan,
)
from src.models import Settings
from src.preferences import PREFERENCES
from src.plotting import (
    _make_commit_reliability_figure,
    _make_compute_phase_figure,
    _make_hrc_phase_figure,
    _make_phase_figure,
    _make_switch_aligned_figure,
    _switch_aligned_records,
    _switch_metric_curve,
)
from src.representations import observe_actions


def _lifecycle_config(
    operation: str, *, reentry_rate: float = 0.0,
) -> ScheduleSettings:
    names = ("retention", "addition", "removal", "swap")
    probabilities = tuple(float(name == operation) for name in names)
    return ScheduleSettings(
        phases=2,
        demos=21,
        min_recipes=1,
        max_recipes=1,
        transition_min=0.20,
        transition_max=0.25,
        gap_allocation=0.8,
        active_size_weights=(1.0, 0.0, 0.0),
        lifecycle_weights=probabilities,
        reentry_rate=reentry_rate,
        recipe_skew=0.7,
        pair_skew=0.7,
        holdout_start=0.5,
        holdout_demos=1,
    )


def _small_config() -> EvalSettings:
    return EvalSettings(
        recipe_count=8,
        schedule=replace(
            ScheduleSettings(),
            panel_size=8,
            phases=5,
            demos=120,
            min_recipes=2,
            max_recipes=4,
        ),
    )


class LifecycleTests(unittest.TestCase):
    def test_default_distribution_is_one_explicit_categorical_draw(self):
        config = EvalSettings()
        self.assertEqual(
            config.schedule.lifecycle_weights,
            (0.40, 0.20, 0.20, 0.20),
        )
        self.assertEqual(config.schedule.reentry_rate, 0.05)
        self.assertEqual(config.schedule.hetero_gap_mean, 15)
        self.assertEqual(config.schedule.hetero_gap_min, 3)
        self.assertEqual(config.schedule.hetero_gap_max, 40)
        self.assertEqual(config.schedule.hetero_gap_shape, 0.35)
        self.assertEqual(config.schedule.climb_decay, 0.50)
        self.assertEqual(config.recipe_count, 20)
        self.assertEqual(config.schedule.panel_size, 20)
        self.assertEqual(config.schedule.min_recipes, 5)
        self.assertEqual(config.schedule.max_recipes, 8)
        self.assertEqual(
            config.schedule.active_size_weights,
            (0.90, 0.08, 0.02),
        )

    def test_addition_and_removal_are_literal(self):
        add_state = Lifecycle(active={"p1"}, ever={"p1", "p3"}, removed={"p3"})
        active, delta, operation = _evolve_active_set(
            add_state,
            ("p1", "p2", "p3"),
            (),
            _lifecycle_config("addition"),
            random.Random(8),
        )
        self.assertEqual(operation, "addition")
        self.assertEqual(active, {"p1", "p2"})
        self.assertEqual(delta["added"], {"p2"})
        self.assertFalse(delta["removed"])

        remove_state = Lifecycle(active={"p1", "p2"}, ever={"p1", "p2"})
        active, delta, operation = _evolve_active_set(
            remove_state,
            ("p1", "p2", "p3"),
            (),
            _lifecycle_config("removal"),
            random.Random(8),
        )
        self.assertEqual(operation, "removal")
        self.assertEqual(len(active), 1)
        self.assertEqual(len(delta["removed"]), 1)
        self.assertFalse(delta["added"])

    def test_swap_replaces_one_preference_without_a_silent_add(self):
        state = Lifecycle(active={"p1"}, ever={"p1"})
        active, delta, operation = _evolve_active_set(
            state,
            ("p1", "p2"),
            (),
            _lifecycle_config("swap"),
            random.Random(4),
        )
        self.assertEqual(operation, "swap")
        self.assertEqual(active, {"p2"})
        self.assertEqual(delta["added"], {"p2"})
        self.assertEqual(delta["removed"], {"p1"})
        self.assertFalse(delta["reintroduced"])

    def test_reintroduction_is_a_swap_subtype(self):
        state = Lifecycle(
            active={"p1", "p3"},
            ever={"p1", "p2", "p3"},
            removed={"p2"},
        )
        active, delta, operation = _evolve_active_set(
            state,
            ("p1", "p2", "p3"),
            (),
            _lifecycle_config("swap", reentry_rate=1.0),
            random.Random(4),
        )
        self.assertEqual(operation, "swap")
        self.assertIn("p2", active)
        self.assertEqual(len(active), 2)
        self.assertEqual(delta["reintroduced"], {"p2"})
        self.assertEqual(len(delta["removed"]), 1)

    def test_infeasible_draws_fail_instead_of_changing_operation(self):
        with self.assertRaisesRegex(RuntimeError, "sampled removal is infeasible"):
            _evolve_active_set(
                Lifecycle(active={"p1"}, ever={"p1"}),
                ("p1", "p2"),
                (),
                _lifecycle_config("removal"),
                random.Random(4),
            )

    def test_forced_introduction_at_capacity_is_an_explicit_swap(self):
        state = Lifecycle(active={"p1"}, ever={"p1"})
        active, delta, operation = _evolve_active_set(
            state,
            ("p1", "p2"),
            ("p2",),
            _lifecycle_config("addition"),
            random.Random(4),
            max_active_size=1,
        )
        self.assertEqual(operation, "swap")
        self.assertEqual(active, {"p2"})
        self.assertEqual(delta["added"], {"p2"})
        self.assertEqual(delta["removed"], {"p1"})

    def test_generator_validation_enforces_budget_and_probabilities(self):
        config = _lifecycle_config("addition")
        config.validate(1)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            replace(
                config,
                lifecycle_weights=(0.4, 0.2, 0.2, 0.1),
            ).validate(1)
        with self.assertRaisesRegex(ValueError, "0.20 <= min <= max <= 0.25"):
            replace(config, transition_max=0.26).validate(1)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            replace(config, demos=20).validate(1)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            replace(config, climb_decay=1.0).validate(1)

    def test_climb_group_size_has_geometrically_decreasing_probability(self):
        rng = random.Random(91)
        recipes = tuple(f"r{index}" for index in range(1, 9))
        counts = Counter(
            len(_climb_groups(recipes, 0.5, rng)[0])
            for _ in range(20_000)
        )
        self.assertTrue(all(counts[size] > counts[size + 1] for size in range(1, 8)))
        self.assertAlmostEqual(counts[1] / 20_000, 0.502, delta=0.02)
        self.assertAlmostEqual(counts[2] / 20_000, 0.251, delta=0.02)

    def test_heterogeneous_delta_t_is_heavy_tailed_with_exact_mean(self):
        config = EvalSettings().schedule
        allocations = [
            _sample_hetero_gap(
                config,
                random.Random(seed),
                [config.hetero_gap_min] * 28,
            )
            for seed in range(5)
        ]
        for allocation in allocations:
            self.assertEqual(
                sum(allocation),
                config.hetero_gap_mean * len(allocation),
            )
        pooled = [value for allocation in allocations for value in allocation]
        self.assertEqual(min(pooled), config.hetero_gap_min)
        self.assertEqual(max(pooled), config.hetero_gap_max)


class ScheduleTests(unittest.TestCase):
    @staticmethod
    def _assert_phase_contract(plan: Plan, config: EvalSettings) -> None:
        metadata = plan.metadata
        structure = plan.events[0].tags["deployment_structure"]
        heterogeneous = structure == "heterogeneous"
        expected_demos = (
            3 * sum(phase["gap"] for phase in metadata["phases"])
            if heterogeneous else config.schedule.demos
        )
        assert metadata["totals"]["demos"] == expected_demos
        assert len(plan.events) == expected_demos
        assert metadata["leakage_audit"]["passed"]

        by_phase: dict[int, list[Event]] = defaultdict(list)
        for index, event in enumerate(plan.events):
            assert event.tags["event_index"] == index
            by_phase[event.tags["phase_index"]].append(event)
        assert len(by_phase) == len(metadata["phases"])
        if heterogeneous:
            assert len(by_phase) >= config.schedule.phases
            gap = [phase["gap"] for phase in metadata["phases"]]
            assert sum(gap) == config.schedule.hetero_gap_mean * len(gap)
            assert min(gap) >= config.schedule.hetero_gap_min
            assert max(gap) <= config.schedule.hetero_gap_max
        else:
            assert len(by_phase) == config.schedule.phases

        climb_total = 0
        for phase in metadata["phases"]:
            index = phase["index"]
            events = by_phase[index]
            climb = [event for event in events if event.tags["phase_role"] == "climb"]
            settled = [event for event in events if event.tags["phase_role"] == "settled"]
            assert events == climb + settled
            assert len(events) == phase["demo_count"] == 3 * phase["gap"]
            assert Counter(event.pair.label for event in climb) == Counter(
                phase["climb_pairs"]
            )
            assert Counter(event.pair.label for event in settled) == Counter(
                phase["settled_pair_counts"]
            )
            assert all(event.tags["gap"] == phase["gap"] for event in events)
            operations = {
                transition["operation"]
                for transition in phase["transitions"].values()
            }
            assert operations <= {
                "initialization", "retention", "addition", "removal", "swap",
            }
            if heterogeneous:
                climb_recipes = set(phase["climb_recipes"])
                assert 1 <= len(climb_recipes) <= config.schedule.max_recipes
                assert set(phase["transitions"]) == climb_recipes
                assert {
                    event.pair.recipe_name for event in climb
                } == climb_recipes
                active = {
                    f"{recipe}/{preference}"
                    for recipe, preferences in phase["active_preferences"].items()
                    for preference in preferences
                }
                assert set(phase["settled_pair_counts"]) == active
            else:
                assert config.schedule.min_recipes <= len(phase["recipes"])
                assert len(phase["recipes"]) <= config.schedule.max_recipes
            for label in phase["reintroduced_pairs"]:
                recipe = label.split("/", 1)[0]
                assert phase["transitions"][recipe]["operation"] == "swap"
            climb_total += len(climb)

        fraction = climb_total / len(plan.events)
        if not heterogeneous:
            assert config.schedule.transition_min <= fraction
            assert fraction <= config.schedule.transition_max
        assert abs(fraction - metadata["transition_fraction"]) < 1e-12
        assert sum(phase["gap"] for phase in metadata["phases"]) == expected_demos // 3

    def test_default_protocol_has_five_paired_seeds_and_three_scenarios(self):
        self.assertEqual(len(PAPER_SEEDS), 5)
        self.assertEqual(parse_args([]).seeds, PAPER_SEEDS)
        self.assertEqual(
            SCENARIOS,
            (
                HOMOGENEOUS,
                HETEROGENEOUS,
                HOLDOUT,
            ),
        )

    def test_default_panel_uses_all_twenty_candidates(self):
        plan = build_plan(
            HETEROGENEOUS,
            EvalSettings(),
            1337,
        )
        metadata = plan.metadata
        self.assertEqual(len(metadata["candidates"]), 20)
        self.assertEqual(len(metadata["recipe_panel"]), 20)
        self.assertFalse(set(metadata["candidates"]) - set(metadata["recipe_panel"]))
        self.assertEqual(set(plan.selected_recipes), set(metadata["recipe_panel"]))
        self.assertEqual(
            tuple(metadata["generator"]["lifecycle_weights"]),
            (0.4, 0.2, 0.2, 0.2),
        )

    def test_ordinary_scenarios_share_one_phase_contract(self):
        config = _small_config()
        for scenario in (
            HOMOGENEOUS,
            HETEROGENEOUS,
        ):
            with self.subTest(scenario=scenario):
                self._assert_phase_contract(
                    build_plan(scenario, config, 17),
                    config,
                )

    def test_same_seed_reproduces_and_different_seed_changes_plan(self):
        config = _small_config()
        first = build_plan(
            HETEROGENEOUS, config, 17,
        )
        repeated = build_plan(
            HETEROGENEOUS, config, 17,
        )
        different = build_plan(
            HETEROGENEOUS, config, 18,
        )
        signature = lambda plan: [
            (event.mode, event.pair.label, event.tags["phase_role"])
            for event in plan.events
        ]
        self.assertEqual(signature(first), signature(repeated))
        self.assertNotEqual(signature(first), signature(different))
        self.assertEqual(first.metadata, repeated.metadata)

    def test_each_recipe_is_observed_once_then_always_assisted(self):
        config = _small_config()
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                plan = build_plan(scenario, config, 17)
                seen_recipes: set[str] = set()
                observation_counts: Counter[str] = Counter()
                for event in plan.events:
                    recipe = event.pair.recipe_name
                    expected_mode = (
                        "observe" if recipe not in seen_recipes else "assist"
                    )
                    self.assertEqual(event.mode, expected_mode)
                    observation_counts[recipe] += int(event.mode == "observe")
                    seen_recipes.add(recipe)
                self.assertEqual(
                    observation_counts,
                    Counter({recipe: 1 for recipe in seen_recipes}),
                )

    def test_seed_profile_is_shared_across_scenarios(self):
        config = _small_config()
        plans = {
            scenario: build_plan(scenario, config, 17)
            for scenario in SCENARIOS
        }
        popularity = {
            tuple(sorted(plan.metadata["recipe_popularity"].items()))
            for plan in plans.values()
        }
        self.assertEqual(len(popularity), 1)
        self.assertEqual(
            len({tuple(plan.metadata["recipe_panel"]) for plan in plans.values()}),
            1,
        )

    def test_homogeneous_uses_one_preference_set_per_phase(self):
        plan = build_plan(
            HOMOGENEOUS,
            _small_config(),
            17,
        )
        for phase in plan.metadata["phases"]:
            active_sets = {
                tuple(preferences)
                for preferences in phase["active_preferences"].values()
            }
            self.assertEqual(len(active_sets), 1)

    def test_heterogeneous_allows_recipe_specific_preference_sets(self):
        observed_diversity = False
        for seed in (17, 18, 19):
            plan = build_plan(
                HETEROGENEOUS,
                _small_config(),
                seed,
            )
            observed_diversity |= any(
                len({
                    tuple(preferences)
                    for preferences in phase["active_preferences"].values()
                }) > 1
                for phase in plan.metadata["phases"]
            )
        self.assertTrue(observed_diversity)

    def test_heterogeneous_groups_recipe_climbs_and_settles_full_active_panel(self):
        config = _small_config()
        plan = build_plan(
            HETEROGENEOUS,
            config,
            17,
        )
        phases = plan.metadata["phases"]
        gap = [phase["gap"] for phase in phases]

        self.assertGreaterEqual(len(phases), config.schedule.phases)
        self.assertEqual(
            sum(gap),
            config.schedule.hetero_gap_mean * len(phases),
        )
        self.assertLessEqual(min(gap), 5)
        self.assertGreaterEqual(max(gap) - min(gap), 20)
        for phase in phases:
            climb_recipes = set(phase["climb_recipes"])
            self.assertEqual(
                {label.split("/", 1)[0] for label in phase["climb_pairs"]},
                climb_recipes,
            )
            self.assertEqual(set(phase["transitions"]), climb_recipes)
            self.assertEqual(
                set(phase["settled_pair_counts"]),
                {
                    f"{recipe}/{preference}"
                    for recipe, preferences in phase["active_preferences"].items()
                    for preference in preferences
                },
            )
        by_macro: dict[int, set[str]] = defaultdict(set)
        requested: dict[int, int] = {}
        for phase in phases:
            macro = phase["stage"]
            by_macro[macro].update(phase["climb_recipes"])
            requested[macro] = phase["requested_recipes"]
        self.assertEqual(set(by_macro), set(range(config.schedule.phases)))
        for macro, recipes in by_macro.items():
            self.assertEqual(len(recipes), requested[macro])
            self.assertGreaterEqual(len(recipes), config.schedule.min_recipes)
            self.assertLessEqual(len(recipes), config.schedule.max_recipes)

    def test_removed_pairs_stay_absent_until_explicit_reintroduction(self):
        plan = build_plan(
            HETEROGENEOUS,
            EvalSettings(),
            1337,
        )
        removed: set[str] = set()
        for phase in plan.metadata["phases"]:
            active = {
                f"{recipe}/{preference}"
                for recipe, preferences in phase["active_preferences"].items()
                for preference in preferences
            }
            reintroduced = set(phase["reintroduced_pairs"])
            self.assertFalse((removed & active) - reintroduced)
            removed.update(phase["removed_pairs"])
            removed.difference_update(reintroduced)

    def test_holdout_uses_the_supervisor_specified_source_and_targets(self):
        config = _small_config()
        plan = build_plan(HOLDOUT, config, 17)
        holdout = plan.metadata["holdout"]
        phases = plan.metadata["phases"]

        self.assertEqual(tuple(holdout["source_preferences"]), HOLDOUT_SOURCES)
        self.assertEqual(tuple(holdout["target_preferences"]), HOLDOUT_TARGETS)
        self.assertEqual(holdout["axis"], HOLDOUT_AXIS)
        self.assertEqual(holdout["axis_value"], HOLDOUT_VALUE)
        self.assertTrue(all(
            PREFERENCES[preference].as_dict()[HOLDOUT_AXIS]
            != HOLDOUT_VALUE
            for preference in HOLDOUT_SOURCES
        ))
        self.assertTrue(all(
            PREFERENCES[preference].as_dict()[HOLDOUT_AXIS]
            == HOLDOUT_VALUE
            for preference in HOLDOUT_TARGETS
        ))
        self.assertEqual(len(phases), len(HOLDOUT_SOURCES + HOLDOUT_TARGETS))
        self.assertEqual(
            [next(iter(phase["active_preferences"].values()))[0] for phase in phases],
            list(HOLDOUT_SOURCES + HOLDOUT_TARGETS),
        )
        self.assertTrue(plan.metadata["leakage_audit"]["passed"])
        self.assertEqual(plan.metadata["leakage_audit"]["axis_value_leak_count"], 0)

        target_events = [
            event for event in plan.events
            if event.pair.preference_name in HOLDOUT_TARGETS
        ]
        self.assertTrue(target_events)
        self.assertEqual(
            target_events[0].tags["phase_index"], len(HOLDOUT_SOURCES),
        )
        self.assertEqual(target_events[0].tags["phase_role"], "climb")
        self.assertTrue(target_events[0].tags["is_first_holdout_exposure"])
        self.assertEqual(target_events[0].tags["holdout_target_type"], "isolated_axis")
        self.assertTrue(all(
            event.tags["holdout_partition"] == "heldout"
            for event in target_events
        ))

    def test_recurrence_gaps_use_demonstration_indices(self):
        plan = build_plan(
            HETEROGENEOUS,
            _small_config(),
            17,
        )
        last_pair: dict[str, int] = {}
        history: dict[str, set[str]] = defaultdict(set)
        for index, event in enumerate(plan.events):
            label = event.pair.label
            tags = event.tags
            self.assertEqual(
                tags["demos_since_pair_seen"],
                None if label not in last_pair else index - last_pair[label],
            )
            self.assertEqual(
                tags["preference_history_depth_before"],
                len(history[event.pair.recipe_name]),
            )
            history[event.pair.recipe_name].add(event.pair.preference_name)
            self.assertEqual(
                tags["preference_history_depth_after"],
                len(history[event.pair.recipe_name]),
            )
            last_pair[label] = index

    def test_commit_sensitivity_only_changes_named_fields(self):
        for _name, overrides in COMMIT_SENSITIVITY_SPECS:
            config = build_settings(17, EvalSettings(model_settings=overrides))
            for field, value in overrides.items():
                self.assertEqual(getattr(config, field), value)


class ResultLayoutTests(unittest.TestCase):
    def test_post_mismatch_accuracy_is_explicitly_conditional(self):
        eligible = _post_mismatch_accuracy(
            [True, False, True, False, True], 1,
        )
        self.assertEqual(eligible["first_mismatch_robot_turn"], 1.0)
        self.assertEqual(eligible["conditional_post_mismatch_top_1_w1"], 1.0)
        self.assertEqual(eligible["conditional_post_mismatch_top_1_w2"], 0.5)
        self.assertEqual(eligible["conditional_post_mismatch_eligible_w3"], 1.0)
        self.assertNotIn("primary_adaptation_recovery_top_1", eligible)

        no_error = _post_mismatch_accuracy([True, True], -1)
        self.assertEqual(no_error["first_mismatch_rate"], 0.0)
        self.assertEqual(no_error["conditional_post_mismatch_eligible_w1"], 0.0)

    def test_exposure_summary_only_includes_post_initialization_changes(self):
        def row(operation: str, changed: bool, exposure: int) -> dict:
            return {
                "mode": "assist",
                "lifecycle_operation": operation,
                "recipe_changed_this_phase": changed,
                "exposure_index_since_recipe_change": exposure,
                "hrc_robot_turn_count": 1,
                "hrc_robot_correct_count": 1,
                "hrc_robot_top_k_hit_count": 1,
                "recipe_steps": 1,
            }

        summary = _scope_metrics(
            [
                row("initialization", True, 1),
                row("retention", False, 4),
                row("swap", True, 1),
                row("addition", True, 2),
            ],
            {},
        )

        self.assertNotIn("by_exposure_index", summary)
        self.assertEqual(
            set(summary["by_post_change_exposure_index"]), {"1", "2"},
        )

    def test_adaptation_speed_uses_one_exposure_and_model_specific_baseline(self):
        def episode(
            event_index: int,
            top_1: float,
            *,
            changed: bool = False,
            exposure: int = 0,
            corrections: int = 0,
        ) -> dict:
            return {
                "mode": "assist",
                "event_index": event_index,
                "phase_id": "phase_01" if changed else "phase_00",
                "phase_start": 3 if changed else 0,
                "recipe": "recipe",
                "pair": "recipe/new" if changed else "recipe/old",
                "recipe_changed_this_phase": changed,
                "lifecycle_operation": "swap" if changed else "initialization",
                "exposure_index_since_recipe_change": exposure,
                "teacher_forced_prediction_count": 10,
                "teacher_forced_top_1": top_1,
                "hrc_human_correction_count": corrections,
            }

        retained_recovery = episode(4, 0.9, changed=True, exposure=2)
        retained_recovery.update({
            "phase_id": "phase_02",
            "recipe_changed_this_phase": False,
            "lifecycle_operation": "retention",
        })
        episodes = [
            episode(0, 0.8),
            episode(1, 0.9),
            episode(2, 1.0),
            episode(3, 0.75, changed=True, exposure=1, corrections=2),
            retained_recovery,
        ]
        turns = [
            {"event_index": 3, "recipe_step": 0, "correct_top_1": False},
            {"event_index": 3, "recipe_step": 1, "correct_top_1": True},
        ]

        summary = summarize_adaptation(episodes, turns)

        self.assertEqual(summary["early_window_exposures"], 1)
        self.assertEqual(
            summary["first_post_switch_teacher_forced_decision_top_1"], 0.0,
        )
        self.assertEqual(
            summary["first_post_switch_exposure_teacher_forced_top_1"], 0.75,
        )
        self.assertEqual(
            summary["mean_cumulative_corrections_first_1_exposure"], 2.0,
        )
        self.assertAlmostEqual(
            summary["mean_pre_switch_teacher_forced_top_1"], 0.9,
        )
        self.assertEqual(summary["mean_exposures_to_recover_90pct"], 2.0)
        self.assertEqual(
            summary["mean_teacher_forced_decisions_to_recover_90pct"], 20.0,
        )
        self.assertNotIn("adaptation_auc", summary)

    def test_preference_cohorts_separate_current_recurrent_stale_and_holdout(self):
        pairs = {
            preference: TaskVariant(
                "recipe", preference, (f"action_{preference}",),
            )
            for preference in ("old", "current", "stale", "holdout")
        }
        plan = Plan(
            scenario="cohort_unit",
            seed=1,
            events=(
                Event(
                    "assist", pairs["old"], {"phase_index": 0},
                ),
                Event(
                    "assist", pairs["current"], {"phase_index": 1},
                ),
                Event(
                    "assist", pairs["old"], {"phase_index": 2},
                ),
            ),
            eval_pairs=tuple(pairs.values()),
            selected_recipes=("recipe",),
            selected_preferences=tuple(pairs),
            description="cohort test",
            metadata={
                "holdout": {
                    "preference": "holdout",
                    "target_preferences": ["holdout"],
                },
                "phases": [{
                    "index": 1,
                    "active_preferences": {
                        "recipe": ["current"],
                    },
                }],
            },
        )

        tags = _preference_cohort_tags(
            plan,
            1,
            {pairs["old"].label, pairs["stale"].label},
            before_event=False,
        )

        self.assertEqual(
            tags[pairs["current"].label]["preference_temporal_status"],
            "current_active",
        )
        self.assertEqual(
            tags[pairs["old"].label]["preference_temporal_status"],
            "old_expected_to_recur",
        )
        self.assertEqual(
            tags[pairs["stale"].label]["preference_temporal_status"],
            "intentionally_stale",
        )
        self.assertTrue(
            tags[pairs["holdout"].label]["heldout_never_learned"],
        )

        rows = [
            {
                "checkpoint": "test",
                "top_1": 1.0,
                "top_k": 1.0,
                "memory_state_before": "active_memory",
                **tag,
            }
            for tag in tags.values()
        ]
        cohorts = summarize_cohorts(rows)
        self.assertEqual(cohorts["current_active"]["n_rows"], 1)
        self.assertEqual(cohorts["old_expected_to_recur"]["n_rows"], 1)
        self.assertEqual(cohorts["intentionally_stale"]["n_rows"], 1)
        self.assertEqual(cohorts["heldout_never_learned"]["n_rows"], 1)
        self.assertEqual(
            cohorts["intentionally_stale"]["checkpoints"]["test"]
            ["memory_state_counts"],
            {"active_memory": 1},
        )

    def test_compact_results_omit_redundant_summary_copies(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        plan = Plan(
            scenario="layout_unit",
            seed=5,
            events=(Event("observe", pair, {"event_type": "unit"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=("default",),
            description="not serialized because it is derivable from the scenario",
            metadata={},
        )
        config = EvalSettings(
            baselines=(),
            include_oracle=False,
            shared_routing=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_plan(plan, config, Path(directory))
            saved_plan = json.loads(Path(directory, "plan.json").read_text())
            saved_summary = json.loads(Path(directory, "summary.json").read_text())

        self.assertEqual(saved_plan["eval_pairs"], [pair.label])
        self.assertEqual(summary["plan"], "plan.json")
        for redundant in (
            "description", "n_events", "n_eval_pairs", "scenario_metadata",
            "support_counts", "oracle_reference",
        ):
            self.assertNotIn(redundant, saved_summary)

    def test_frozen_and_delayed_audits_report_explicit_support(self):
        frozen = summarize_frozen([
            {
                "checkpoint": "final",
                "top_1": 0.75,
                "top_k": 1.0,
                "recipe_steps": 4,
                "hrc_human_turn_count": 1,
                "hrc_human_correction_count": 1,
            },
            {
                "checkpoint": "final",
                "top_1": 0.25,
                "top_k": 0.5,
                "recipe_steps": 4,
                "hrc_human_turn_count": 1,
                "hrc_human_correction_count": 3,
            },
        ])
        self.assertEqual(frozen["status"], "completed")
        self.assertEqual(frozen["n_rows"], 2)
        self.assertEqual(frozen["checkpoints"]["final"]["top_1"], 0.5)
        self.assertEqual(
            frozen["checkpoints"]["final"]["corrections_per_recipe_step"],
            0.5,
        )
        self.assertEqual(
            frozen["checkpoints"]["final"]["normalized_human_action_load"],
            0.75,
        )
        self.assertEqual(
            frozen["checkpoints"]["final"]["mean_corrections_per_task"],
            2.0,
        )
        self.assertEqual(summarize_frozen([])["status"], "not_run")
        self.assertEqual(
            set(summarize_frozen_by(
                [{"condition": "delayed", "checkpoint": "x", "top_1": 1.0}],
                "condition",
            )),
            {"delayed"},
        )

        delayed = summarize_recurrence([
            {
                "delayed_recurrence_probe": True,
                "delayed_recurrence_target_active_before": True,
                "delayed_recurrence_target_is_latest_before": True,
                "delayed_recurrence_all_known_conflicts_pruned_before": True,
                "delayed_recurrence_known_conflicting_variant_count_before": 3,
                "delayed_recurrence_conflicting_active_count_before": 0,
                "delayed_recurrence_conflicting_pruned_count_before": 3,
                "delayed_recurrence_intervening_event_count": 4,
                "delayed_recurrence_required_intervening_event_count": 4,
                "delayed_recurrence_actual_grace_horizon_before": 7,
            },
        ])
        self.assertEqual(delayed["n_probes"], 1)
        self.assertEqual(delayed["all_known_conflicts_pruned_before_rate"], 1.0)

    def test_reentry_metrics_separate_pruned_recovery_from_active_control(self):
        common = {
            "hrc_robot_turn_count": 2,
            "hrc_robot_correct_count": 1,
            "hrc_robot_top_k_hit_count": 2,
        }
        summary = summarize_reentry([
            {
                **common,
                "mode": "assist",
                "reentry_probe_target_state_before": "pruned_exact_variant",
                "actual_reentry_from_pruned": True,
            },
            {
                **common,
                "mode": "assist",
                "reentry_probe_target_state_before": "active_exact_variant",
                "actual_reentry_from_pruned": False,
            },
            {
                **common,
                "mode": "assist",
                "reentry_probe_target_state_before": "known_recipe_no_exact_variant",
                "actual_reentry_from_pruned": False,
            },
            # Observation episodes are excluded: nothing is predicted there.
            {
                **common,
                "mode": "observe",
                "reentry_probe_target_state_before": "pruned_exact_variant",
                "actual_reentry_from_pruned": False,
            },
        ])
        self.assertEqual(summary["n_assist_episodes"], 3)
        self.assertEqual(summary["n_target_pruned_before_episode"], 1)
        self.assertEqual(summary["n_target_active_before_episode_control"], 1)
        self.assertEqual(summary["n_target_known_recipe_new_variant"], 1)
        self.assertEqual(summary["n_confirmed_reentry_from_pruned"], 1)
        self.assertEqual(summary["confirmed_reentry_from_pruned"]["live_top_1"], 0.5)

    def test_reentry_metrics_report_no_pruned_episodes_for_a_retaining_agent(self):
        """An agent that never prunes must show an empty recovery cell, not a
        zero-probe diagnostic: the control cell still has to populate."""
        summary = summarize_reentry([
            {
                "mode": "assist",
                "reentry_probe_target_state_before": "active_exact_variant",
                "actual_reentry_from_pruned": False,
                "hrc_robot_turn_count": 2,
                "hrc_robot_correct_count": 2,
                "hrc_robot_top_k_hit_count": 2,
            },
        ])
        self.assertEqual(summary["n_target_pruned_before_episode"], 0)
        self.assertEqual(summary["n_target_active_before_episode_control"], 1)
        self.assertEqual(
            summary["target_active_before_episode_control"]["live_top_1"], 1.0,
        )

    def test_paired_bootstrap_preserves_pairs_and_metric_directions(self):
        def summary(
            seed: int,
            full_top_1: float,
            baseline_top_1: float,
            full_human_load: float,
            baseline_human_load: float,
        ) -> dict:
            def metrics(top_1: float, human_load: float) -> dict:
                return {"assist": {"overall": {
                    "live_top_1": top_1,
                    "teacher_forced_top_1": top_1,
                    "normalized_human_action_load": human_load,
                }}}
            return {
                "scenario": "unit",
                "seed": seed,
                "per_baseline": {
                    "full": metrics(full_top_1, full_human_load),
                    "unpinned": metrics(
                        baseline_top_1, baseline_human_load
                    ),
                },
            }

        statistics = summarize_pairs({
            "unit/seed_1": summary(1, 0.8, 0.5, 0.3, 0.6),
            "unit/seed_2": summary(2, 0.6, 0.4, 0.4, 0.7),
        }, n_samples=200)
        comparison = statistics["by_scenario"]["unit"][
            "comparisons_against_full"
        ]["unpinned"]
        self.assertEqual(comparison["teacher_forced_top_1"]["n_paired_seeds"], 2)
        self.assertGreater(comparison["teacher_forced_top_1"]["mean"], 0.0)
        self.assertGreater(
            comparison["normalized_human_action_load"]["mean"],
            0.0,
        )

    def test_turn_metrics_are_pooled_not_averaged_by_episode(self):
        metrics = aggregate_episodes([
            {
                "hrc_robot_turn_count": 0,
                "hrc_robot_top_k_hit_count": 0,
                "hrc_robot_correct_count": 0,
            },
            {
                "hrc_robot_turn_count": 10,
                "hrc_robot_top_k_hit_count": 8,
                "hrc_robot_correct_count": 8,
            },
        ])
        self.assertEqual(metrics["live_top_1"], 0.8)
        self.assertEqual(metrics["live_top_k"], 0.8)

    def test_oracle_summary_does_not_mix_metrics(self):
        summary = summarize_oracle([
            {
                "baseline": "full",
                "oracle_reference": "oracle",
                "metric": "live_top_1",
                "scope": "assist.overall",
                "oracle_advantage": 0.2,
            },
            {
                "baseline": "full",
                "oracle_reference": "oracle",
                "metric": "normalized_human_action_load",
                "scope": "assist.overall",
                "oracle_advantage": 0.1,
            },
        ])
        by_metric = summary["by_baseline"]["full"]["by_metric"]
        self.assertEqual(
            set(by_metric),
            {"live_top_1", "normalized_human_action_load"},
        )
        self.assertEqual(
            by_metric["live_top_1"]["mean_oracle_advantage"],
            0.2,
        )

    def test_memory_state_uses_pre_episode_active_or_pruned_membership(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
        ))
        agent.start_demo()
        for observation in observe_actions(pair.actions):
            agent.observe(observation)
        classification = agent.end_demo()
        assert classification.recipe_id is not None
        recipe_ids = {recipe_name: classification.recipe_id}
        key = _pair_key(agent, pair, recipe_ids)
        assert key is not None
        self.assertEqual(
            _memory_state(agent, pair, recipe_ids, {recipe_name}),
            "active_memory",
        )
        entry = agent.replay.active[key]
        agent.replay._prune_entry(
            key,
            entry,
            now=agent.demo_counter + 1,
            cycle=agent.retrain_cycle,
        )
        self.assertEqual(
            _memory_state(agent, pair, recipe_ids, {recipe_name}),
            "pruned_memory",
        )
        row = assist_demo(
            agent,
            pair,
            recipe_ids,
            config=EvalSettings(),
            observed_pairs={pair.label},
            observed_recipes={recipe_name},
        )
        self.assertEqual(row["memory_state_before"], "pruned_memory")
        self.assertEqual(row["memory_state_after"], "active_memory")


class SwitchAlignedMetricTests(unittest.TestCase):
    @staticmethod
    def _row(
        seed: int,
        event_index: int,
        phase: str,
        operation: str,
        *,
        changed: bool,
        recipe: str = "recipe",
    ) -> dict:
        return {
            "baseline": "full",
            "seed": seed,
            "event_index": event_index,
            "mode": "assist",
            "recipe": recipe,
            "phase_id": phase,
            "phase_start": event_index if changed else 0,
            "recipe_changed_this_phase": changed,
            "lifecycle_operation": operation,
            "teacher_forced_prediction_count": 2,
            "teacher_forced_correct_count": 1,
            "hrc_robot_turn_count": 1,
            "hrc_robot_correct_count": 1,
            "hrc_human_turn_count": 1,
            "hrc_human_correction_count": 0,
            "recipe_steps": 2,
        }

    def test_alignment_uses_first_post_switch_exposure_and_stops_at_next_switch(self):
        rows = [
            self._row(1, 0, "phase_00", "initialization", changed=False),
            self._row(1, 1, "phase_00", "retention", changed=False),
            self._row(1, 3, "phase_01", "swap", changed=True),
            self._row(1, 4, "phase_01", "swap", changed=True),
            self._row(1, 6, "phase_02", "swap", changed=True),
            self._row(1, 7, "phase_02", "swap", changed=True),
        ]
        # Every row in a changed phase carries the phase boundary, as in the
        # evaluator's episode logs.
        rows[3]["phase_start"] = 3
        rows[5]["phase_start"] = 6

        records = _switch_aligned_records(rows, "full", radius=3)
        first = {
            int(record["relative_t"]): int(record["row"]["event_index"])
            for record in records
            if record["switch_id"] == "1|phase_01|recipe"
        }
        self.assertEqual(first, {-2: 0, -1: 1, 0: 3, 1: 4})
        self.assertNotIn(6, first.values())
        second_pre = {
            int(record["row"]["event_index"])
            for record in records
            if record["switch_id"] == "1|phase_02|recipe"
            and int(record["relative_t"]) < 0
        }
        self.assertEqual(second_pre, {3, 4})

    def test_switch_curve_pools_counts_within_seed_before_averaging_seeds(self):
        def record(seed: int, switch: str, correct: int, total: int) -> dict:
            return {
                "seed": seed,
                "switch_id": switch,
                "relative_t": 0,
                "row": {"correct": correct, "total": total},
            }

        curve = _switch_metric_curve(
            (
                record(1, "a", 1, 1),
                record(1, "b", 0, 9),
                record(2, "c", 1, 1),
            ),
            ("correct",),
            "total",
            key="unit",
            radius=1,
        )
        point = next(row for row in curve if row["relative_t"] == 0)
        self.assertAlmostEqual(point["mean"], 0.55)
        self.assertEqual(point["n_seeds"], 2)
        self.assertEqual(point["n_switches"], 3)

    def test_switch_figure_has_three_unsmoothed_outcome_panels(self):
        rows = []
        for seed in (1, 2):
            rows.extend([
                self._row(seed, 0, "phase_00", "initialization", changed=False),
                self._row(seed, 1, "phase_00", "retention", changed=False),
                self._row(seed, 2, "phase_01", "swap", changed=True),
                self._row(seed, 3, "phase_01", "swap", changed=True),
            ])
            rows[-1]["phase_start"] = 2
        figure = _make_switch_aligned_figure(
            rows,
            HOMOGENEOUS,
            radius=1,
        )
        try:
            self.assertEqual(len(figure.axes), 3)
            for axis in figure.axes:
                vertical = [
                    line for line in axis.lines
                    if np.array_equal(line.get_xdata(), (0.0, 0.0))
                ]
                self.assertTrue(vertical)
        finally:
            plt.close(figure)

    def test_commit_reliability_figure_uses_logged_candidate_confidence(self):
        rows = [
            {
                "baseline": "full",
                "seed": 1,
                "event_index": 0,
                "mode": "assist",
                "commit_decision": "full",
                "commit_applied": True,
                "self_training_opportunity": True,
                "expected_recipe_id": "R1",
                "expected_variant_id": "v1",
                "commit_candidate_recipe_id": "R1",
                "commit_variant_id": "v1",
                "commit_candidate_correct": True,
                "commit_confidence": 0.85,
            },
        ]
        figure = _make_commit_reliability_figure(
            rows,
            HOMOGENEOUS,
        )
        try:
            self.assertEqual(len(figure.axes), 1)
            self.assertGreaterEqual(len(figure.axes[0].lines), 2)
        finally:
            plt.close(figure)


class PhaseFigureSeparationTests(unittest.TestCase):
    @staticmethod
    def _summaries() -> list[dict]:
        summaries = []
        for seed, offset in ((1, 0.0), (2, 0.05)):
            summaries.append({
                "seed": seed,
                "per_baseline": {
                    "full": {
                        "assist": {
                            "by_phase_role": {
                                "climb": {
                                    "teacher_forced_top_1": 0.80 + offset,
                                    "live_top_1": 0.70 + offset,
                                    "normalized_human_action_load": 0.45 - offset,
                                    "mean_corrections_per_task": 1.5 - offset,
                                    "testing_episode_wall_s": 2.0 + offset,
                                    "n_episodes": 4,
                                    "mean_prediction_wall_s": 0.01 + offset / 100.0,
                                },
                            },
                        },
                        "training": {
                            "per_phase_role": {
                                "climb": {
                                    "online_training_total_retrain_wall_s": 1.0 + offset,
                                    "online_training_estimated_fit_flops": 2.0e9 + offset,
                                },
                            },
                        },
                    },
                },
            })
        return summaries

    def test_phase_figures_separate_scientific_hrc_and_compute_metrics(self):
        summaries = self._summaries()
        scenario = HOMOGENEOUS
        figures = (
            _make_phase_figure(summaries, scenario, "climb", "Climb"),
            _make_hrc_phase_figure(summaries, scenario, "climb", "Climb"),
            _make_compute_phase_figure(summaries, scenario, "climb", "Climb"),
        )
        try:
            scientific, hrc, compute = figures
            self.assertEqual(len(scientific.axes), 3)
            self.assertEqual(
                {axis.get_title() for axis in scientific.axes},
                {
                    "Identical-prefix prediction",
                    "Closed-loop prediction",
                    "Normalized human action load",
                },
            )
            self.assertEqual(len(hrc.axes), 1)
            self.assertEqual(
                {axis.get_title() for axis in hrc.axes},
                {"Corrective burden"},
            )
            self.assertEqual(len(compute.axes), 4)
            self.assertEqual(
                {axis.get_title() for axis in compute.axes},
                {
                    "Measured online fitting time",
                    "Measured prediction latency",
                    "Measured evaluator episode runtime",
                    "Model-specific fitting operations",
                },
            )
        finally:
            for figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
