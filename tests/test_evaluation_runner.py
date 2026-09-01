"""Integration contracts for scenario execution, routing, and metrics."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.evaluation import (
    MEMORY_ORACLE,
    DEFAULT_BASELINES,
    EvalSettings,
    ScheduleSettings,
    TaskVariant,
    Event,
    Plan,
    HETEROGENEOUS,
    HOMOGENEOUS,
    SCENARIOS,
    _apply_oracle_pruning,
    _oracle_candidate_noninferior,
    _pair_key,
    _periodic_probe_due,
    _pre_event_probe_due,
    aggregate_episodes,
    assist_demo,
    evaluate_frozen,
    build_task,
    snapshot_memory,
    observe_demo,
    run_stream,
    run_plan,
    summarize_stream,
)
from src.models import Settings
from src.memory import MatchResult


def _fast_eval_config(**overrides):
    values = {
        "seeds": (19,),
        "baselines": ("full",),
        "include_oracle": False,
        "recipe_count": 2,
        "schedule": ScheduleSettings(
            panel_size=2,
            phases=2,
            demos=42,
            min_recipes=1,
            max_recipes=2,
        ),
        "audit_period": 0,
        "show_eta": False,
        "model_settings": {
            "irl_cold_steps": 1,
            "irl_warm_steps": 1,
        },
    }
    values.update(overrides)
    return EvalSettings(**values)


class EvaluationRunnerContractTests(unittest.TestCase):
    def _pair_and_agent(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
        ))
        return recipe_name, pair, agent

    def test_stream_summary_uses_compact_sections(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_schema",
            seed=23,
            events=(Event("observe", pair, {"event_type": "unit"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=("default",),
            description="schema contract",
        )
        stream = run_stream(
            "full", plan, _fast_eval_config(),
        )
        self.assertEqual(stream.active_audit_rows, [])
        summary = summarize_stream(stream)
        self.assertEqual(
            set(summary),
            {"assist", "episodes", "training", "diagnostics", "system"},
        )
        for redundant in ("assist_only", "all_episode_workload", "memory", "compute", "paper_hypothesis_views"):
            self.assertNotIn(redundant, summary)

    def test_observation_logs_no_robot_prediction_or_turn_records(self):
        recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}

        row = observe_demo(agent, pair, recipe_ids)

        self.assertEqual(row["mode"], "observe")
        self.assertEqual(row["n_steps"], 0)
        self.assertEqual(row["hrc_robot_turn_count"], 0)
        self.assertEqual(row["_turn_records"], [])
        self.assertIn(recipe_name, recipe_ids)
        self.assertIsNone(row["normalized_human_action_load"])
        self.assertIsNone(row["corrections_per_recipe_step"])
        for removed in (
            "human_correction_rate",
            "testing_total_action_time",
            "testing_human_only_action_time",
            "testing_normalized_interaction_cost",
        ):
            self.assertNotIn(removed, row)

    def test_compute_snapshot_pairs_fit_flops_with_fit_wall_time(self):
        _recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)

        snapshot = snapshot_memory(agent)

        self.assertIn("training_estimated_fit_flops", snapshot)
        self.assertEqual(snapshot["predictor"], "maxent")
        self.assertEqual(snapshot["irl_features"], "engineered")
        self.assertEqual(snapshot["memory_policy"], "adaptive")
        self.assertTrue(snapshot["latest_pin_enabled"])
        self.assertEqual(snapshot["adaptive_pair_horizon_count"], 1)
        self.assertEqual(len(snapshot["pair_grace_horizons_demos"]), 1)
        self.assertEqual(snapshot["pair_recurrence_gap_windows_demos"], {})
        self.assertEqual(snapshot["pair_recurrence_window"], 12)
        self.assertEqual(snapshot["pair_recurrence_min_samples"], 5)
        self.assertAlmostEqual(snapshot["pair_downward_half_life_samples"], 3.0)
        self.assertEqual(snapshot["recipe_recurrence_window"], 24)
        self.assertEqual(snapshot["global_recurrence_window"], 60)
        self.assertEqual(snapshot["demo_counter"], 1)
        self.assertEqual(snapshot["registry_recipes"], 1)
        self.assertEqual(snapshot["registry_size"], 1)
        self.assertEqual(snapshot["active_variants"], 1)
        self.assertEqual(snapshot["archived_variants"], 0)
        self.assertGreater(snapshot["registry_steps"], 0)
        self.assertEqual(len(snapshot["pair_horizon_evidence"]), 1)
        self.assertEqual(snapshot["training_flop_accounting_scope"], "fit_only_model_specific_arithmetic")
        self.assertFalse(snapshot["training_flop_cross_model_comparable"])
        self.assertEqual(snapshot["training_total_retrain_wall_scope"], "end_to_end_retrain_including_trajectory_build")
        self.assertGreater(snapshot["training_fit_wall_s"], 0.0)
        self.assertAlmostEqual(
            snapshot["training_fit_effective_gflops_s"],
            snapshot["training_estimated_fit_flops"] / (1.0e9 * snapshot["training_fit_wall_s"]),
        )

    def test_assistance_logs_robot_and_human_shadow_turns_separately(self):
        recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)

        row = assist_demo(
            agent,
            pair,
            recipe_ids,
            config=_fast_eval_config(),
            observed_pairs={pair.label},
            observed_recipes={recipe_name},
        )
        robot_turns = [turn for turn in row["_turn_records"] if turn["turn_kind"] == "robot"]
        shadow_turns = [turn for turn in row["_turn_records"] if turn["turn_kind"] == "human_shadow"]

        self.assertEqual(row["mode"], "assist")
        self.assertGreater(row["hrc_robot_turn_count"], 0)
        self.assertEqual(len(robot_turns), row["hrc_robot_turn_count"])
        self.assertEqual(len(shadow_turns), row["hrc_human_shadow_turn_count"])
        self.assertEqual(
            sorted(turn["recipe_step"] for turn in robot_turns + shadow_turns),
            list(range(len(pair.actions))),
        )
        self.assertEqual(
            row["teacher_forced_prediction_count"], len(pair.actions),
        )
        self.assertEqual(row["teacher_forced_position_coverage"], 1.0)
        self.assertEqual(
            row["teacher_forced_correct_count"],
            sum(int(turn["correct_top_1"]) for turn in robot_turns + shadow_turns),
        )
        self.assertEqual(row["n_steps"], row["hrc_robot_turn_count"])
        self.assertEqual(row["commit_registry_size"], 1)
        self.assertEqual(row["commit_variants_scored"], 1)
        self.assertGreaterEqual(row["commit_scoring_wall_s"], 0.0)
        self.assertTrue(all(turn["scheduled_actor"] == "robot" for turn in robot_turns))
        self.assertTrue(all(turn["scheduled_actor"] == "human" for turn in shadow_turns))
        self.assertEqual(
            row["corrections_per_recipe_step"],
            row["hrc_human_correction_count"] / len(pair.actions),
        )
        self.assertEqual(
            row["normalized_human_action_load"],
            (
                row["hrc_human_turn_count"]
                + row["hrc_human_correction_count"]
            ) / len(pair.actions),
        )
        self.assertNotIn("human_correction_rate", row)
        self.assertNotIn("testing_total_action_time", row)

    def test_assist_prediction_cannot_overwrite_evaluator_ground_truth(self):
        items = list(recipe_builders().items())[:2]
        first_pair = build_task(items[0][0], "default", items[0][1])
        second_pair = build_task(items[1][0], "default", items[1][1])
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
        ))
        recipe_ids = {}
        observe_demo(agent, first_pair, recipe_ids)
        observe_demo(agent, second_pair, recipe_ids)
        expected_mapping = dict(recipe_ids)
        wrong_recipe_id = recipe_ids[second_pair.recipe_name]
        agent._match_prefix = lambda _prefix: (
            MatchResult("preference_shift", wrong_recipe_id, None, 1.0, 0.0),
            False,
        )
        agent._commit_confidence = lambda _rid, _prefix, _cls: (
            1.0,
            {
                "recipe_jaccard": 1.0,
                "recipe_jaccard_margin": 1.0,
                "late_window_prediction_agreement": 1.0,
            },
        )

        row = assist_demo(
            agent,
            first_pair,
            recipe_ids,
            config=_fast_eval_config(),
            observed_pairs={first_pair.label, second_pair.label},
            observed_recipes={first_pair.recipe_name, second_pair.recipe_name},
        )

        self.assertEqual(recipe_ids, expected_mapping)
        self.assertTrue(row["commit_applied"])
        self.assertFalse(row["commit_candidate_recipe_correct"])
        self.assertFalse(row["commit_candidate_correct"])

    def test_authoritative_assist_mode_reports_protocol_error_when_memory_is_empty(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_routing",
            seed=23,
            events=(Event("assist", pair, {"event_type": "unit_absent_assist"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for safe routing of an unseen recipe.",
        )

        stream = run_stream("full", plan, _fast_eval_config())
        row = stream.episode_rows[0]

        self.assertEqual(row["requested_mode"], "assist")
        self.assertEqual(row["mode"], "assist")
        self.assertEqual(row["executed_mode"], "assist")
        self.assertEqual(row["mode_route_reason"], "user_selected")
        self.assertEqual(row["classification_kind"], "assist_unavailable")
        self.assertFalse(row["commit_applied"])
        self.assertEqual(row["user_observation_required"], 0.0)

    def test_oracle_retains_exact_future_preferences_and_removes_dead_ones(self):
        recipe_name = "tomato_onion_soup"
        builder = recipe_builders()[recipe_name]
        pair_a = build_task(recipe_name, "default", builder)
        pair_b = build_task(recipe_name, "prep_first", builder)
        self.assertNotEqual(pair_a.actions, pair_b.actions)
        plan = Plan(
            scenario="unit_recipe_preference_oracle",
            seed=23,
            events=(
                Event("observe", pair_a, {}),
                Event("assist", pair_b, {}),
                Event("assist", pair_a, {}),
            ),
            eval_pairs=(),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair_a.preference_name, pair_b.preference_name),
            description="Exact future recipe-preference retention contract.",
        )

        stream = run_stream(
            MEMORY_ORACLE,
            plan,
            _fast_eval_config(
                model_settings={
                    "initial_grace": 0,
                    "min_grace": 0,
                    "prune_delay": 1,
                    "irl_cold_steps": 1,
                    "irl_warm_steps": 1,
                },
            ),
        )

        self.assertEqual(stream.agent.replay.policy, "none")
        self.assertEqual(
            [row["active_variants"] for row in stream.memory_rows],
            [1, 1, 0],
        )
        self.assertEqual(
            [row["pruned_variants"] for row in stream.memory_rows],
            [0, 0, 0],
        )
        self.assertEqual(stream.memory_rows[0]["mean_active_weight"], 1.0)
        after = {
            row["event_index"]: row
            for row in stream.oracle_pruning_rows
            if row["timing"] == "after_event"
        }
        self.assertEqual(after[0]["oracle_pruned_active_variants"], 0)
        self.assertEqual(after[1]["oracle_pruned_active_variants"], 1)
        self.assertEqual(after[1]["oracle_future_pair_count"], 1)
        self.assertEqual(after[1]["oracle_future_variant_count"], 1)
        self.assertGreaterEqual(after[2]["oracle_pruned_active_variants"], 1)
        self.assertEqual(
            after[2]["oracle_pruned_active_variants"],
            after[2]["oracle_active_variants_before"],
        )
        self.assertEqual(after[2]["oracle_active_variants_after"], 0)
        self.assertEqual(
            after[2]["oracle_retention_policy"],
            "future_filtered_with_full_fallback",
        )

    def test_oracle_retains_shared_variant_until_last_equivalent_pair(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        equivalent = TaskVariant(
            recipe_name=recipe_name,
            preference_name="equivalent_identity",
            actions=pair.actions,
            values=pair.values,
        )
        plan = Plan(
            scenario="unit_equivalent_recipe_preference_oracle",
            seed=29,
            events=(
                Event("observe", pair, {}),
                Event("assist", equivalent, {}),
            ),
            eval_pairs=(),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name, equivalent.preference_name),
            description="Equivalent variant retention contract.",
        )

        stream = run_stream(
            MEMORY_ORACLE,
            plan,
            _fast_eval_config(),
        )

        self.assertEqual(
            [row["active_variants"] for row in stream.memory_rows],
            [1, 0],
        )
        after = [
            row for row in stream.oracle_pruning_rows
            if row["timing"] == "after_event"
        ]
        self.assertEqual(after[0]["oracle_future_pair_count"], 1)
        self.assertEqual(after[0]["oracle_future_variant_count"], 1)
        self.assertEqual(after[0]["oracle_pruned_active_variants"], 0)
        self.assertEqual(after[1]["oracle_pruned_active_variants"], 1)

    def test_oracle_deletes_dead_variants_from_pruned_replay_archive(self):
        recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)
        key = _pair_key(agent, pair, recipe_ids)
        self.assertIsNotNone(key)
        assert key is not None
        entry = agent.replay.active[key]
        agent.replay._prune_entry(
            key,
            entry,
            now=agent.demo_counter + 1,
            cycle=agent.retrain_cycle,
        )

        result = _apply_oracle_pruning(agent, (), recipe_ids)

        self.assertNotIn(key, agent.replay.active)
        self.assertNotIn(key, agent.replay.pruned)
        self.assertEqual(result["oracle_pruned_active_variants"], 0)
        self.assertEqual(result["oracle_pruned_archived_variants"], 1)
        # Identity memory remains separate so pruning replay does not erase
        # the robot's learned recipe-recognition representation.
        self.assertIn(key[1], agent.library.variants[key[0]])

    def test_oracle_restores_a_pruned_variant_needed_in_the_future(self):
        recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)
        key = _pair_key(agent, pair, recipe_ids)
        self.assertIsNotNone(key)
        assert key is not None
        entry = agent.replay.active[key]
        agent.replay._prune_entry(
            key,
            entry,
            now=agent.demo_counter + 1,
            cycle=agent.retrain_cycle,
        )
        retrains_before = len(agent.retrain_events)

        result = _apply_oracle_pruning(
            agent,
            (Event("assist", pair, {}),),
            recipe_ids,
        )

        self.assertIn(key, agent.replay.active)
        self.assertNotIn(key, agent.replay.pruned)
        self.assertEqual(agent.replay.active[key].weight, 1.0)
        self.assertEqual(result["oracle_restored_future_variants"], 1)
        self.assertGreater(len(agent.retrain_events), retrains_before)

    def test_oracle_resets_future_variant_weight_and_refits(self):
        _recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)
        key = _pair_key(agent, pair, recipe_ids)
        self.assertIsNotNone(key)
        assert key is not None
        agent.replay.active[key].weight = 0.25
        retrains_before = len(agent.retrain_events)

        result = _apply_oracle_pruning(
            agent,
            (Event("assist", pair, {}),),
            recipe_ids,
        )

        self.assertEqual(agent.replay.active[key].weight, 1.0)
        self.assertEqual(result["oracle_reset_future_weights"], 1)
        self.assertGreater(len(agent.retrain_events), retrains_before)

    def test_oracle_candidate_rejects_any_headline_regression(self):
        reference = {
            "teacher_forced_top_1": 0.8,
            "live_top_1": 0.75,
            "normalized_human_action_load": 0.5,
            "commit_correct": True,
        }
        candidate = dict(reference)
        candidate["live_top_1"] = 0.70

        accepted, regressions = _oracle_candidate_noninferior(
            candidate, reference,
        )

        self.assertFalse(accepted)
        self.assertEqual(regressions, ("live_top_1",))

    def test_full_realized_schedule_forces_baseline_to_the_same_interaction_mode(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_matched_route",
            seed=23,
            events=(Event("assist", pair, {"event_type": "unit_assist"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for canonical full interaction routing.",
        )
        config = _fast_eval_config(observe_missing_recipes=True)

        locally_routed = run_stream("bc", plan, config)
        matched = run_stream(
            "bc",
            plan,
            config,
            execution_mode_schedule=("assist",),
            mode_schedule_policy="matched_full_realized_execution_schedule",
        )

        self.assertEqual(locally_routed.episode_rows[0]["mode"], "observe")
        row = matched.episode_rows[0]
        self.assertEqual(row["mode"], "assist")
        self.assertEqual(row["natural_executed_mode"], "observe")
        self.assertTrue(row["mode_matches_full_schedule"])
        self.assertEqual(row["mode_route_reason"], "matched_full_realized_execution_schedule")

    def test_repeat_observation_is_available_only_in_local_recovery_ablation(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_repeat_observation",
            seed=23,
            events=(
                Event("observe", pair, {"event_type": "unit_onboarding"}),
                Event("assist", pair, {"event_type": "unit_reentry"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Ablation-only recovery after complete recipe loss.",
        )
        strict_config = _fast_eval_config(observe_missing_recipes=True)
        local_config = _fast_eval_config(
            observe_missing_recipes=True,
            allow_repeat_observation=True,
        )

        with patch(
            "src.evaluation._recipe_has_active_variant", return_value=False,
        ):
            strict = run_stream("full", plan, strict_config)
            local = run_stream("full", plan, local_config)

        strict_row = strict.episode_rows[1]
        self.assertEqual(strict_row["requested_mode"], "assist")
        self.assertEqual(strict_row["mode"], "assist")
        self.assertFalse(strict_row["repeat_observation"])

        local_row = local.episode_rows[1]
        self.assertEqual(local_row["requested_mode"], "assist")
        self.assertEqual(local_row["mode"], "observe")
        self.assertTrue(local_row["repeat_observation"])
        self.assertEqual(
            local_row["mode_route_reason"],
            "assist_routed_to_reobservation_recipe_absent_from_active_memory",
        )

    def test_every_default_method_executes_the_same_two_event_workload(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_all_methods",
            seed=31,
            events=(
                Event("observe", pair, {"event_type": "unit_onboarding"}),
                Event("assist", pair, {"event_type": "unit_assist"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Shared evaluator contract for every default method.",
        )
        config = _fast_eval_config()

        for baseline in DEFAULT_BASELINES:
            with self.subTest(baseline=baseline):
                stream = run_stream(
                    baseline,
                    plan,
                    config,
                    execution_mode_schedule=("observe", "assist"),
                    mode_schedule_policy="unit_shared_schedule",
                )
                self.assertEqual(
                    [row["mode"] for row in stream.episode_rows],
                    ["observe", "assist"],
                )
                self.assertEqual(len(stream.episode_rows), len(plan.events))
                self.assertTrue(all(
                    row["mode_matches_full_schedule"]
                    for row in stream.episode_rows
                ))

    def test_repeat_observation_for_the_same_recipe_is_rejected(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_invalid_repeat_observation",
            seed=31,
            events=(
                Event("observe", pair, {"event_type": "unit_onboarding"}),
                Event("observe", pair, {"event_type": "unit_invalid_repeat"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="A recipe may enter observation mode only once.",
        )

        with self.assertRaisesRegex(ValueError, "first stream exposure"):
            run_stream("bc", plan, _fast_eval_config())

    def test_oracle_uses_full_schedule_and_dominates_full_eventwise(self):
        recipe_name = "tomato_onion_soup"
        builder = recipe_builders()[recipe_name]
        old_pair = build_task(recipe_name, "default", builder)
        new_pair = build_task(recipe_name, "prep_first", builder)
        plan = Plan(
            scenario="unit_oracle_local_route",
            seed=23,
            events=(
                Event("observe", old_pair, {"event_type": "unit_onboarding"}),
                Event("assist", new_pair, {"event_type": "unit_shift"}),
                Event("assist", new_pair, {"event_type": "unit_repeat"}),
            ),
            eval_pairs=(old_pair, new_pair),
            selected_recipes=(recipe_name,),
            selected_preferences=(old_pair.preference_name, new_pair.preference_name),
            description="Oracle must observe a new preference after deleting the dead one.",
        )
        config = _fast_eval_config(
            baselines=("full",),
            include_oracle=True,
            shared_routing=True,
            observe_missing_recipes=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_plan(plan, config, Path(temp_dir))
            with gzip.open(Path(temp_dir) / "tables" / "episodes.jsonl.gz", "rt") as handle:
                rows = [json.loads(line) for line in handle]
            with gzip.open(Path(temp_dir) / "tables" / "frozen_probes.jsonl.gz", "rt") as handle:
                frozen_rows = [json.loads(line) for line in handle]

        full = [row for row in rows if row["baseline"] == "full"]
        oracle = [
            row for row in rows
            if row["baseline"] == MEMORY_ORACLE
        ]
        self.assertEqual(
            [row["mode"] for row in full],
            ["observe", "assist", "assist"],
        )
        self.assertEqual(
            [row["mode"] for row in oracle],
            ["observe", "assist", "assist"],
        )
        self.assertEqual(oracle[1]["requested_mode"], "assist")
        self.assertEqual(oracle[1]["natural_executed_mode"], "assist")
        self.assertEqual(
            oracle[1]["mode_schedule_policy"],
            "matched_full_realized_execution_schedule_oracle",
        )
        self.assertEqual(oracle[1]["full_realized_execution_mode"], "assist")
        self.assertTrue(oracle[1]["mode_matches_full_schedule"])
        self.assertEqual(oracle[1]["oracle_selection"], "full_for_unseen_pair")
        for full_row, oracle_row in zip(full, oracle):
            if full_row["mode"] != "assist":
                continue
            for metric in (
                "teacher_forced_top_1",
                "teacher_forced_top_k",
                "live_top_1",
                "live_top_k",
            ):
                self.assertGreaterEqual(oracle_row[metric], full_row[metric])
            for metric in (
                "teacher_forced_mean_nll",
                "mean_nll_per_robot_turn",
                "normalized_human_action_load",
                "corrections_per_recipe_step",
            ):
                self.assertLessEqual(oracle_row[metric], full_row[metric])
        unseen_metrics = (
            "teacher_forced_top_1",
            "teacher_forced_top_k",
            "teacher_forced_mean_nll",
            "live_top_1",
            "live_top_k",
            "mean_nll_per_robot_turn",
            "normalized_human_action_load",
            "corrections_per_recipe_step",
            "commit_applied",
            "commit_candidate_recipe_id",
            "commit_variant_id",
        )
        for metric in unseen_metrics:
            self.assertEqual(oracle[1].get(metric), full[1].get(metric), metric)
        self.assertEqual(
            summary["mode_schedule"]["policy"],
            "full_realized_shared_across_all_methods",
        )

        full_probes = {
            (row["checkpoint"], row["pair"]): row
            for row in frozen_rows if row["baseline"] == "full"
        }
        oracle_probes = {
            (row["checkpoint"], row["pair"]): row
            for row in frozen_rows if row["baseline"] == MEMORY_ORACLE
        }
        self.assertEqual(set(oracle_probes), set(full_probes))
        for key, oracle_row in oracle_probes.items():
            full_row = full_probes[key]
            for metric in (
                "top_1", "top_k",
                "closed_loop_live_top_1", "closed_loop_live_top_k",
            ):
                self.assertGreaterEqual(oracle_row[metric], full_row[metric])
            for metric in (
                "teacher_forced_mean_nll",
                "normalized_human_action_load",
                "corrections_per_recipe_step",
            ):
                self.assertLessEqual(oracle_row[metric], full_row[metric])

    def test_offline_pretrained_frozen_baseline_trains_once_then_never_updates(self):
        (first_name, first_builder), (second_name, second_builder) = list(recipe_builders().items())[:2]
        first_pair = build_task(first_name, "default", first_builder)
        second_pair = build_task(second_name, "prep_first", second_builder)
        plan = Plan(
            scenario="unit_offline_frozen",
            seed=23,
            events=(
                Event("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                Event("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            selected_recipes=(first_name, second_name),
            selected_preferences=("default", "prep_first"),
            description="Unit offline pretraining then frozen deployment schedule.",
        )

        stream = run_stream(
            "frozen",
            plan,
            _fast_eval_config(
                offline_recipe_fraction=0.50,
                offline_preference_fraction=0.50,
            ),
        )

        self.assertTrue(stream.agent._deployment_locked)
        self.assertEqual(len(stream.agent.retrain_events), 1)
        self.assertEqual(stream.episode_rows[0]["offline_training_pair_count"], 1)
        self.assertFalse(stream.episode_rows[0]["updates_allowed"])
        snapshot = snapshot_memory(stream.agent)
        offline = snapshot["offline_pretraining"]
        self.assertEqual(offline["train_count"], 1)
        self.assertFalse(offline["updates_allowed"])

    def test_offline_all_recipes_identity_baseline_covers_every_recipe_but_no_preference_axes(self):
        (first_name, first_builder), (second_name, second_builder) = list(recipe_builders().items())[:2]
        first_pair = build_task(first_name, "prep_first", first_builder)
        second_pair = build_task(second_name, "equipment_jit_serving_early", second_builder)
        plan = Plan(
            scenario="unit_all_recipes_identity_frozen",
            seed=23,
            events=(
                Event("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                Event("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            # Baseline protocol, not scenario order, supplies default training.
            selected_recipes=(first_name, second_name),
            selected_preferences=("prep_first", "equipment_jit_serving_early"),
            description="Unit all-recipes default-only offline training schedule.",
        )

        stream = run_stream(
            "offline_default",
            plan,
            _fast_eval_config(),
        )

        snapshot = snapshot_memory(stream.agent)
        offline = snapshot["offline_pretraining"]
        self.assertTrue(stream.agent._deployment_locked)
        self.assertEqual(offline["offline_training_design"], "all_recipes_default_only")
        self.assertEqual(offline["offline_training_recipe_count"], 2)
        self.assertEqual(offline["offline_training_preference_names"], ["default"])
        self.assertEqual(offline["offline_training_pair_count"], 2)
        self.assertEqual(len(stream.agent.retrain_events), 2)
        self.assertFalse(offline["updates_allowed"])

    def test_primary_assist_has_a_matched_nonmutating_pre_event_probe(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_pre_event_probe",
            seed=23,
            events=(
                Event("observe", pair, {"event_type": "unit_onboarding"}),
                Event("assist", pair, {"event_type": "unit_primary", "primary_probe": True}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for a matched frozen probe before live assistance.",
        )
        stream = run_stream(
            "full",
            plan,
            _fast_eval_config(pre_event_probes=True),
        )
        rows = [row for row in stream.frozen_rows if row.get("probe_phase") == "pre_event"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], pair.label)
        self.assertEqual(rows[0]["live_event_index"], 1)
        self.assertEqual(len(stream.episode_rows), len(plan.events), "frozen probe must not add a live episode")

    def test_batched_frozen_probe_matches_noncommitting_episode_and_restores_state(self):
        recipe_name, pair, agent = self._pair_and_agent()
        recipe_ids = {}
        observe_demo(agent, pair, recipe_ids)
        config = _fast_eval_config()
        observed_pairs = {pair.label}
        observed_recipes = {recipe_name}

        reference = assist_demo(
            agent,
            pair,
            recipe_ids,
            config=config,
            commit=False,
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        )
        before = agent._frozen_structural_digest()
        rows = evaluate_frozen(
            agent,
            (pair, pair),
            recipe_ids,
            config=config,
            checkpoint="unit",
            event_index=1,
            context={"baseline": "full", "scenario": "unit", "seed": 19},
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        )

        self.assertEqual(before, agent._frozen_structural_digest())
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["top_1"], reference["teacher_forced_top_1"])
            self.assertEqual(row["top_k"], reference["teacher_forced_top_k"])
            self.assertEqual(row["closed_loop_live_top_1"], reference["live_top_1"])
            self.assertEqual(
                row["normalized_human_action_load"],
                reference["normalized_human_action_load"],
            )
            self.assertEqual(
                row["corrections_per_recipe_step"],
                reference["corrections_per_recipe_step"],
            )

    def test_runner_exposes_two_deployments_and_one_holdout(self):
        self.assertEqual(
            SCENARIOS,
            (
                "homogeneous",
                "heterogeneous",
                "holdout",
            ),
        )

    def test_frozen_schedule_is_scenario_specific(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        config = _fast_eval_config()

        homogeneous = Plan(
            scenario=HOMOGENEOUS,
            seed=19,
            events=(
                Event("observe", pair, {"phase_id": "phase_00", "phase_role": "climb"}),
                Event("assist", pair, {"phase_id": "phase_00", "phase_role": "settled"}),
                Event("assist", pair, {"phase_id": "phase_00", "phase_role": "settled"}),
                Event("assist", pair, {"phase_id": "phase_01", "phase_role": "climb"}),
                Event("assist", pair, {"phase_id": "phase_01", "phase_role": "settled"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit homogeneous schedule.",
        )
        self.assertEqual(
            [index for index in range(len(homogeneous.events)) if _periodic_probe_due(homogeneous, index)],
            [2, 4],
        )

        heterogeneous = Plan(
            scenario=HETEROGENEOUS,
            seed=19,
            events=(
                Event("assist", pair, {"phase_id": "phase_01", "phase_role": "climb"}),
                Event("assist", pair, {"phase_id": "phase_01", "phase_role": "settled"}),
                Event("assist", pair, {"phase_id": "phase_01", "phase_role": "settled"}),
                Event("assist", pair, {"phase_id": "phase_02", "phase_role": "climb"}),
                Event("assist", pair, {"phase_id": "phase_02", "phase_role": "settled"}),
                Event("assist", pair, {"phase_id": "phase_02", "phase_role": "settled"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit heterogeneous schedule.",
        )
        self.assertEqual(
            [index for index in range(len(heterogeneous.events)) if _periodic_probe_due(heterogeneous, index)],
            [2, 5],
        )

        self.assertFalse(_pre_event_probe_due("assist", "assist", {}, config))
        self.assertTrue(_pre_event_probe_due("assist", "assist", {"primary_probe": True}, config))
        self.assertFalse(_pre_event_probe_due("assist", "observe", {}, config))

    def test_aggregation_uses_robot_turn_support_and_excludes_observation_rows(self):
        rows = [
            {
                "mode": "observe",
                "recipe_steps": 12,
                "hrc_robot_turn_count": 0,
                "hrc_robot_correct_count": 0,
                "hrc_robot_wrong_count": 0,
                "hrc_robot_top_k_hit_count": 0,
                "hrc_human_correction_count": 0,
                "later_action_errors": 0,
            },
            {
                "mode": "assist",
                "recipe_steps": 6,
                "hrc_robot_turn_count": 4,
                "hrc_robot_correct_count": 3,
                "hrc_robot_wrong_count": 1,
                "hrc_robot_top_k_hit_count": 4,
                "hrc_human_turn_count": 2,
                "hrc_human_correction_count": 1,
                "later_action_errors": 1,
                "teacher_forced_prediction_count": 6,
                "teacher_forced_correct_count": 5,
                "teacher_forced_top_k_hit_count": 6,
                "teacher_forced_total_nll": 3.0,
            },
        ]
        metrics = aggregate_episodes(rows)

        self.assertEqual(metrics["live_top_1"], 0.75)
        self.assertEqual(metrics["live_top_k"], 1.0)
        self.assertEqual(metrics["teacher_forced_top_1"], 5.0 / 6.0)
        self.assertEqual(metrics["teacher_forced_top_k"], 1.0)
        self.assertEqual(metrics["teacher_forced_mean_nll"], 0.5)
        self.assertEqual(metrics["teacher_forced_position_coverage"], 1.0)
        self.assertEqual(metrics["normalized_human_action_load"], 0.5)
        self.assertEqual(metrics["mean_corrections_per_task"], 1.0)
        self.assertEqual(metrics["corrections_per_recipe_step"], 1.0 / 6.0)
        self.assertEqual(
            metrics["later_action_error_share"], 1.0
        )
        self.assertNotIn("future_valid_wrong_rate", metrics)
        self.assertEqual(metrics["n_assist_recipe_steps"], 6.0)
        self.assertEqual(metrics["primary_prediction_metric"], "teacher_forced_top_1")
        self.assertEqual(
            metrics["primary_hrc_metric"], "normalized_human_action_load"
        )
        self.assertEqual(metrics["observation_mode_rate"], 0.5)

if __name__ == "__main__":
    unittest.main()
