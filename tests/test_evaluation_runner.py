"""Integration contracts for scenario execution, routing, and metrics."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
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
    _select_oracle_probe_rows,
    _phase_retrain_latency,
    _recipe_has_active_variant,
    _pair_key,
    _periodic_probe_due,
    _pre_event_probe_due,
    _preference_panel,
    _baseline_cell_dir,
    _run_baseline_cell_job,
    baseline_run_order,
    load_route,
    run_baseline_cell,
    save_route,
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
from src.preferences import PREFERENCE_IDS


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

    def test_completed_event_is_checkpointed_before_later_failure(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario="unit_partial_checkpoint",
            seed=23,
            events=(
                Event(
                    "observe",
                    pair,
                    {
                        "event_type": "unit",
                        "phase_id": "phase_00",
                        "stage": 0,
                    },
                ),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=("default",),
            description="completed events must survive an interrupted seed",
        )

        def interrupt_after_checkpoint(_progress):
            raise RuntimeError("planned interruption")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                run_plan(
                    plan,
                    _fast_eval_config(),
                    root,
                    progress_callback=interrupt_after_checkpoint,
                )
            checkpoint = json.loads((
                root / "partial" / "checkpoints" / "full"
                / "event_000000.json"
            ).read_text())
            partial = json.loads((root / "partial" / "summary.json").read_text())

        self.assertEqual(checkpoint["state"], "event_complete")
        self.assertEqual(checkpoint["events_completed"], 1)
        self.assertEqual(len(checkpoint["tables"]["episodes"]), 1)
        self.assertEqual(partial["state"], "running")
        self.assertEqual(partial["latest"]["event_index"], 0)

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
            "future_filtered_after_first_exposure",
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

    def test_recipe_active_variant_check_discriminates_by_recipe(self):
        """Regression: the comprehension variable must not shadow the target.

        Bound as ``recipe_id``, the test became ``recipe_id == recipe_id`` and
        returned True whenever active memory was non-empty. That is the only
        trigger for baseline-local re-observation, so the routing ablation
        recorded zero extra observations for every baseline in every run.
        """
        builders = recipe_builders()
        held = build_task("tomato_onion_soup", "default", builders["tomato_onion_soup"])
        absent = build_task("simple_salad", "default", builders["simple_salad"])
        agent = AdaptiveAgent(Settings(verbose=False, irl_cold_steps=1, irl_warm_steps=1))
        agent.replay.register("R1", "v1", ("a",), now=1, cycle=0, pin_latest=False)
        recipe_ids = {"tomato_onion_soup": "R1", "simple_salad": "R2"}

        self.assertTrue(_recipe_has_active_variant(agent, held, recipe_ids))
        self.assertFalse(_recipe_has_active_variant(agent, absent, recipe_ids))

        # Empty memory cannot hold any recipe.
        agent.replay.active.clear()
        self.assertFalse(_recipe_has_active_variant(agent, held, recipe_ids))

    def test_phase_retrain_latency_slices_fits_into_the_phase_they_ran_in(self):
        class _StubAgent:
            retrain_fit_wall_times = [1.0, 2.0, 3.0, 4.0, 100.0]

        rows = [
            {"event_index": 0, "phase_role": "climb", "phase_index": 0, "training_retrain_count": 2},
            {"event_index": 1, "phase_role": "climb", "phase_index": 0, "training_retrain_count": 4},
            {"event_index": 2, "phase_role": "settled", "phase_index": 1, "training_retrain_count": 5},
        ]
        by_phase, by_index = _phase_retrain_latency(_StubAgent(), rows, {"training_retrain_count": 0})

        self.assertEqual(by_phase["climb"]["online_retrain_fit_count"], 4.0)
        self.assertEqual(by_phase["climb"]["online_mean_retrain_fit_wall_s"], 2.5)
        self.assertEqual(by_phase["climb"]["online_max_retrain_fit_wall_s"], 4.0)
        # The outlier belongs to the settled phase and must not be averaged away.
        self.assertEqual(by_phase["settled"]["online_retrain_fit_count"], 1.0)
        self.assertEqual(by_phase["settled"]["online_p95_retrain_fit_wall_s"], 100.0)
        self.assertEqual(by_index["settled"]["phase_01"]["online_max_retrain_fit_wall_s"], 100.0)

    def test_phase_retrain_latency_ignores_events_with_no_new_fit(self):
        class _StubAgent:
            retrain_fit_wall_times = [1.0]

        rows = [
            {"event_index": 0, "phase_role": "climb", "phase_index": 0, "training_retrain_count": 1},
            # A skipped retrain leaves the cumulative count unchanged.
            {"event_index": 1, "phase_role": "climb", "phase_index": 0, "training_retrain_count": 1},
        ]
        by_phase, _ = _phase_retrain_latency(_StubAgent(), rows, {"training_retrain_count": 0})
        self.assertEqual(by_phase["climb"]["online_retrain_fit_count"], 1.0)

    def test_oracle_probe_rows_keep_the_future_filtered_outcome_when_worse(self):
        """The oracle must be allowed to lose; it is a reference, not a maximum.

        Selecting per event on realized metrics forced oracle_advantage >= 0 by
        construction, which is the one thing a bounding reference may not do.
        """
        reference = [{
            "pair": "soup/default",
            "teacher_forced_top_1": 0.8,
            "live_top_1": 0.75,
            "normalized_human_action_load": 0.5,
        }]
        candidate = [dict(reference[0], live_top_1=0.70, teacher_forced_top_1=0.60)]

        selected = _select_oracle_probe_rows(candidate, reference)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["oracle_probe_selection"], "future_filtered")
        self.assertEqual(selected[0]["live_top_1"], 0.70)
        self.assertEqual(selected[0]["teacher_forced_top_1"], 0.60)

    def test_oracle_probe_rows_reject_misaligned_pairs(self):
        reference = [{"pair": "soup/default", "live_top_1": 0.5}]
        candidate = [{"pair": "salad/default", "live_top_1": 0.5}]

        with self.assertRaises(RuntimeError):
            _select_oracle_probe_rows(candidate, reference)

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

    def test_oracle_uses_full_schedule_and_may_lose_after_first_exposure(self):
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
        # After a pair is learned the oracle keeps its own future-filtered
        # outcome and is never reverted to Full's. It is a reference curve, not
        # a per-event maximum over {Full, oracle}: constraining it to dominate
        # made oracle_advantage non-negative by construction.
        self.assertEqual(oracle[2]["oracle_selection"], "future_filtered")
        self.assertFalse(oracle[2]["oracle_full_fallback"])
        self.assertNotIn("oracle_regressed_metrics", oracle[2])
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
        for oracle_row in oracle_probes.values():
            self.assertEqual(oracle_row["oracle_probe_selection"], "future_filtered")
            self.assertNotIn("oracle_probe_regressed_metrics", oracle_row)

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

    def test_offline_default_baseline_covers_every_recipe_but_no_preference_axes(self):
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

    def test_offline_all_baseline_trains_on_the_whole_preference_library(self):
        """The corpus is the seed's frozen-probe panel, not its scheduled pairs.

        The schedule supplies one preference here; the arm must still pretrain
        on every behaviourally distinct preset of both recipes, because the
        candidate preference set does not vary by seed.
        """
        (first_name, first_builder), (second_name, second_builder) = list(recipe_builders().items())[:2]
        first_pair = build_task(first_name, "prep_first", first_builder)
        second_pair = build_task(second_name, "prep_first", second_builder)
        expected_pairs = sum(
            len(_preference_panel(name, builder, PREFERENCE_IDS)[0])
            for name, builder in ((first_name, first_builder), (second_name, second_builder))
        )
        plan = Plan(
            scenario="unit_all_pairs_frozen",
            seed=23,
            events=(
                Event("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                Event("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            selected_recipes=(first_name, second_name),
            selected_preferences=("prep_first",),
            description="Unit all-recipes all-preferences offline training schedule.",
        )

        stream = run_stream(
            "offline_all",
            plan,
            _fast_eval_config(),
        )

        snapshot = snapshot_memory(stream.agent)
        offline = snapshot["offline_pretraining"]
        self.assertTrue(stream.agent._deployment_locked)
        self.assertEqual(offline["offline_training_design"], "all_recipes_all_preferences")
        self.assertEqual(offline["offline_training_recipe_count"], 2)
        self.assertEqual(offline["offline_training_recipe_names"], sorted([first_name, second_name]))
        self.assertEqual(offline["offline_training_declared_preference_count"], len(PREFERENCE_IDS))
        # Far more than the single preference this schedule ever presents.
        self.assertGreater(offline["offline_training_preference_count"], 1)
        self.assertEqual(offline["offline_training_pair_count"], expected_pairs)
        self.assertEqual(len(stream.agent.retrain_events), expected_pairs)
        self.assertFalse(offline["updates_allowed"])

    def test_offline_all_baseline_strictly_contains_the_frozen_subset_corpus(self):
        """`frozen` is this arm at half coverage, so its pairs must nest here."""
        (first_name, first_builder), (second_name, second_builder) = list(recipe_builders().items())[:2]
        first_pair = build_task(first_name, "prep_first", first_builder)
        second_pair = build_task(second_name, "serving_early", second_builder)
        plan = Plan(
            scenario="unit_frozen_nesting",
            seed=23,
            events=(
                Event("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                Event("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            selected_recipes=(first_name, second_name),
            selected_preferences=("prep_first", "serving_early"),
            description="Unit nesting check for the two cross-product frozen arms.",
        )
        config = _fast_eval_config(
            offline_recipe_fraction=0.50,
            offline_preference_fraction=0.50,
        )

        subset = snapshot_memory(
            run_stream("frozen", plan, config).agent,
        )["offline_pretraining"]
        everything = snapshot_memory(
            run_stream("offline_all", plan, config).agent,
        )["offline_pretraining"]

        self.assertLessEqual(
            set(subset["offline_training_recipe_names"]),
            set(everything["offline_training_recipe_names"]),
        )
        self.assertLessEqual(
            set(subset["offline_training_preference_names"]),
            set(everything["offline_training_preference_names"]),
        )
        self.assertLess(
            subset["offline_training_pair_count"],
            everything["offline_training_pair_count"],
        )

    def test_offline_all_baseline_never_sees_a_recipe_outside_this_seed(self):
        """Unscheduled recipes stay out: no other arm is ever shown them."""
        names = list(recipe_builders().items())
        (first_name, first_builder), (second_name, _second_builder) = names[:2]
        first_pair = build_task(first_name, "default", first_builder)
        plan = Plan(
            scenario="unit_all_pairs_frozen_panel",
            seed=23,
            events=(
                Event("observe", first_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair,),
            selected_recipes=(first_name,),
            selected_preferences=("default",),
            description="Unit panel-restricted all-pairs offline training schedule.",
        )

        stream = run_stream("offline_all", plan, _fast_eval_config())

        offline = snapshot_memory(stream.agent)["offline_pretraining"]
        self.assertEqual(offline["offline_training_recipe_names"], [first_name])
        self.assertNotIn(second_name, offline["offline_training_recipe_names"])
        self.assertLess(
            offline["offline_training_recipe_count"], len(recipe_builders()),
        )

    # Wall-clock readings measure the machine, not the method, and never
    # reproduce across processes. Everything else must.
    _TIMING_FIELD = re.compile(
        r"(wall_s|_at_utc|elapsed_s|latency|_per_s|_time|gflops_s)$"
    )

    @classmethod
    def _without_timings(cls, value):
        if isinstance(value, dict):
            return {
                key: cls._without_timings(item)
                for key, item in value.items()
                if not cls._TIMING_FIELD.search(key)
            }
        if isinstance(value, list):
            return [cls._without_timings(item) for item in value]
        return value

    @classmethod
    def _canonical(cls, value):
        return json.dumps(cls._without_timings(value), sort_keys=True)

    @staticmethod
    def _gz_rows(path):
        if not Path(path).is_file():
            return []
        with gzip.open(path, "rt") as handle:
            return [json.loads(line) for line in handle]

    def test_running_one_arm_at_a_time_scores_exactly_as_running_them_together(self):
        """The suite runs arm-major; that must not change a single number.

        Every arm but ``full`` replays ``full``'s realized observe/assist
        schedule. Running the arms in separate cells only works if handing
        that schedule through the shared route is equivalent to holding it in
        memory across one plan, so this compares the two paths directly.
        """
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario=HOMOGENEOUS,
            seed=31,
            events=(
                Event("observe", pair, {"event_type": "unit_onboarding"}),
                Event("assist", pair, {"event_type": "unit_assist"}),
                Event("assist", pair, {"event_type": "unit_assist"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Arm-major execution must match arm-together execution.",
        )
        config = _fast_eval_config(
            baselines=("full", "no_decay", "bc"),
            include_oracle=True,
            shared_routing=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            together = run_plan(plan, config, root / "together")

            route = None
            apart = {}
            for baseline in baseline_run_order(config):
                summary = run_baseline_cell(
                    baseline, plan, config, root / "apart" / baseline, route=route,
                )
                if summary.get("realized_route"):
                    route = tuple(summary["realized_route"])
                apart[baseline] = summary["per_baseline"][baseline]

            self.assertEqual(
                sorted(together["per_baseline"]), sorted(apart),
            )
            for baseline in sorted(apart):
                with self.subTest(baseline=baseline):
                    self.assertEqual(
                        self._canonical(together["per_baseline"][baseline]),
                        self._canonical(apart[baseline]),
                    )
                    for table in (
                        "episodes", "turns", "frozen_probes", "axis_transfer",
                    ):
                        merged = [
                            row for row in self._gz_rows(
                                root / "together" / "tables" / f"{table}.jsonl.gz"
                            )
                            if row.get("baseline") == baseline
                        ]
                        cell = self._gz_rows(
                            root / "apart" / baseline / "tables" / f"{table}.jsonl.gz"
                        )
                        self.assertEqual(
                            self._canonical(merged),
                            self._canonical(cell),
                            f"{baseline}/{table}",
                        )

    def test_an_arm_cell_refuses_to_run_without_the_route_it_replays(self):
        """A cell must fail loudly rather than silently score its own route."""
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = Plan(
            scenario=HOMOGENEOUS,
            seed=31,
            events=(Event("observe", pair, {"event_type": "unit_onboarding"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="A replaying arm needs a published route.",
        )
        config = _fast_eval_config(
            baselines=("full", "no_decay"), shared_routing=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "no route"):
                _run_baseline_cell_job(
                    "no_decay", HOMOGENEOUS, 31, config, str(run_dir),
                )
            status = json.loads(
                (
                    _baseline_cell_dir(run_dir, "no_decay", HOMOGENEOUS, 31)
                    / "status.json"
                ).read_text()
            )
            self.assertEqual(status["state"], "failed")

    def test_a_published_route_survives_deleting_the_arm_that_consumed_it(self):
        """Re-running one arm must not require re-running 'full'."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            save_route(run_dir, HOMOGENEOUS, 31, ["observe", "assist"])
            self.assertEqual(
                load_route(run_dir, HOMOGENEOUS, 31), ("observe", "assist"),
            )
            self.assertIsNone(load_route(run_dir, HOMOGENEOUS, 99))

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


class CliDefaultsMatchDataclassDefaults(unittest.TestCase):
    """A CLI default that restates a dataclass default silently overrides it.

    `EvalSettings` is the documented source of truth for the evaluation
    configuration, but every flag parsed with its own literal default wins over
    it, so the two can drift apart and a config change appears to have no
    effect. This asserts they agree for every flag that maps onto a field.
    """

    def test_no_cli_flag_default_contradicts_its_dataclass_field(self):
        import dataclasses
        import re
        from pathlib import Path

        from src.evaluation import EvalSettings, ScheduleSettings

        source = (Path(__file__).resolve().parent.parent / "src" / "evaluation.py").read_text(encoding="utf-8")
        fields = {}
        for spec in (EvalSettings, ScheduleSettings):
            for field in dataclasses.fields(spec):
                if field.default is not dataclasses.MISSING:
                    fields[field.name] = field.default

        mismatched = []
        for match in re.finditer(
            r'add_argument\("--([a-z0-9-]+)"[^)]*?default=([^,)\s]+)', source,
        ):
            name = match.group(1).replace("-", "_")
            if name not in fields:
                continue
            literal = match.group(2).strip()
            try:
                value = eval(literal, {"EvalSettings": EvalSettings, "ScheduleSettings": ScheduleSettings}, {})
            except Exception:
                continue
            if value != fields[name]:
                mismatched.append((name, value, fields[name]))

        self.assertEqual(mismatched, [], f"CLI defaults contradict dataclass defaults: {mismatched}")


class WorkerCrashDiagnosticsTests(unittest.TestCase):
    """A worker that dies natively has to leave evidence and clean state."""

    def test_stale_checkpoints_from_an_earlier_attempt_are_cleared(self):
        from src.evaluation import _clear_stale_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            checkpoints = out_dir / "partial" / "checkpoints" / "in_context_llm"
            checkpoints.mkdir(parents=True)
            for index in (0, 71, 131):
                (checkpoints / f"event_{index:06d}.json").write_text("{}")
            keep = out_dir / "partial" / "summary.json"
            keep.write_text("{}")

            removed = _clear_stale_checkpoints(out_dir)

            self.assertEqual(removed, 3)
            self.assertEqual(list(checkpoints.glob("event_*.json")), [])
            # Only per-event checkpoints are stale; the summary is not.
            self.assertTrue(keep.is_file())

    def test_clearing_checkpoints_is_safe_before_any_exist(self):
        from src.evaluation import _clear_stale_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_clear_stale_checkpoints(Path(directory)), 0)

    def test_worker_fault_log_captures_a_native_stack(self):
        import faulthandler
        from src.evaluation import _open_worker_fault_log

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            previously_enabled = faulthandler.is_enabled()
            handle = _open_worker_fault_log(out_dir)
            try:
                self.assertIsNotNone(handle)
                self.assertTrue(faulthandler.is_enabled())
                faulthandler.dump_traceback(file=handle, all_threads=False)
            finally:
                if previously_enabled:
                    faulthandler.enable()
                else:
                    faulthandler.disable()

            written = (out_dir / "worker_fault.log").read_text()
            self.assertIn("test_worker_fault_log_captures_a_native_stack", written)


class EventResumeTests(unittest.TestCase):
    """Per-event resume must reproduce an uninterrupted run exactly."""

    @staticmethod
    def _config(tmp, **overrides):
        from src.evaluation import EvalSettings

        base = dict(
            baselines=("bc",),
            scenarios=("homogeneous",),
            seeds=(1337,),
            include_oracle=False,
            top_k=3,
            audit_period=0,
            output=str(tmp),
            event_resume=True,
        )
        base.update(overrides)
        return EvalSettings(**base)

    def _short_plan(self, config, events=6):
        from dataclasses import replace as dc_replace
        from src.evaluation import build_plan

        plan = build_plan("homogeneous", config, 1337)
        return dc_replace(plan, events=tuple(plan.events[:events]))

    def test_resumed_stream_matches_an_uninterrupted_stream(self):
        from src.evaluation import run_stream

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            plan = self._short_plan(config)

            reference = run_stream("bc", plan, config)

            # Run again, stopping after event 2 by way of the resume state the
            # stream writes, then continue in a fresh call.
            partial_dir = root / "partial_run"
            partial_dir.mkdir()
            stop_after = 2

            class Stop(RuntimeError):
                pass

            def stop(_state, event_index):
                if event_index >= stop_after:
                    raise Stop

            with self.assertRaises(Stop):
                run_stream(
                    "bc", plan, config,
                    event_progress=stop,
                    resume_dir=partial_dir,
                )
            self.assertTrue(
                (partial_dir / "partial" / "resume" / "bc.pickle").is_file()
            )

            resumed = run_stream("bc", plan, config, resume_dir=partial_dir)

            self.assertEqual(
                len(resumed.episode_rows), len(reference.episode_rows),
            )
            for key in ("teacher_forced_top_1", "live_top_1", "recipe", "preference"):
                self.assertEqual(
                    [row.get(key) for row in resumed.episode_rows],
                    [row.get(key) for row in reference.episode_rows],
                    msg=f"episode column {key} diverged after resume",
                )
            self.assertEqual(
                len(resumed.turn_rows), len(reference.turn_rows),
            )
            self.assertEqual(
                [row.get("predicted") for row in resumed.turn_rows],
                [row.get("predicted") for row in reference.turn_rows],
            )

    def test_resume_state_is_refused_when_the_plan_differs(self):
        from src.evaluation import _load_event_resume, _event_resume_path, run_stream

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            plan = self._short_plan(config, events=6)
            shorter = self._short_plan(config, events=4)

            class Stop(RuntimeError):
                pass

            with self.assertRaises(Stop):
                def stop(_state, event_index):
                    if event_index >= 1:
                        raise Stop
                run_stream("bc", plan, config, event_progress=stop, resume_dir=root)

            path = _event_resume_path(root, "bc")
            self.assertIsNotNone(_load_event_resume(path, "bc", plan))
            # A different event count, baseline or seed must not be continued.
            self.assertIsNone(_load_event_resume(path, "bc", shorter))
            self.assertIsNone(_load_event_resume(path, "unpinned", plan))

    def test_corrupt_resume_state_restarts_instead_of_failing(self):
        from src.evaluation import _load_event_resume, _event_resume_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            plan = self._short_plan(config, events=4)
            path = _event_resume_path(root, "bc")
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not a pickle")

            self.assertIsNone(_load_event_resume(path, "bc", plan))

    def test_checkpoints_past_the_resume_point_are_trimmed(self):
        from src.evaluation import _trim_checkpoints_after

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "partial" / "checkpoints" / "in_context_llm"
            checkpoints.mkdir(parents=True)
            for index in range(6):
                (checkpoints / f"event_{index:06d}.json").write_text("{}")

            removed = _trim_checkpoints_after(root, "in_context_llm", 3)

            self.assertEqual(removed, 2)
            self.assertEqual(
                sorted(p.name for p in checkpoints.glob("event_*.json")),
                [f"event_{i:06d}.json" for i in range(4)],
            )


class ResumableRunDiscoveryTests(unittest.TestCase):
    """Only a directory that explicit --resume would accept may be continued."""

    def _config(self, output):
        return EvalSettings(
            output=str(output),
            experiment="llm_single_seed_evaluation",
            baselines=("full", "in_context_llm"),
            scenarios=("homogeneous",),
            seeds=(1337,),
            include_oracle=False,
            event_resume=True,
        )

    def _make_run(
        self, runs, name, digest, *,
        config_hash=None, experiment="llm_single_seed_evaluation",
        state="failed", resume_state=True,
    ):
        run_dir = runs / name
        cell_dir = (
            run_dir
            / "baselines/in_context_llm/scenarios/homogeneous/seeds/0000001337"
        )
        (cell_dir / "partial/resume").mkdir(parents=True)
        if resume_state:
            (cell_dir / "partial/resume/in_context_llm.pickle").write_bytes(b"x")
        (run_dir / "manifest.json").write_text(json.dumps({
            "config_hash": config_hash or digest,
            "experiment": experiment,
        }))
        if state is not None:
            (run_dir / "status.json").write_text(json.dumps({"state": state}))
        return run_dir

    def test_only_matching_incomplete_runs_with_state_are_offered(self):
        from src.evaluation import _config_hash, find_resumable_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            digest = _config_hash(config)
            runs = root / "runs"
            runs.mkdir()

            self.assertIsNone(find_resumable_run(config))
            for name, kwargs in (
                ("wrong-hash", {"config_hash": "different"}),
                ("wrong-experiment", {"experiment": "other"}),
                ("already-complete", {"state": "complete"}),
                ("no-state", {"resume_state": False}),
            ):
                self._make_run(runs, name, digest, **kwargs)
                with self.subTest(rejected=name):
                    self.assertIsNone(find_resumable_run(config))

            self._make_run(runs, "genuine", digest)
            self.assertEqual(find_resumable_run(config), "genuine")

    def test_the_newest_matching_run_wins(self):
        from src.evaluation import _config_hash, find_resumable_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            digest = _config_hash(config)
            runs = root / "runs"
            runs.mkdir()
            older = self._make_run(runs, "older", digest)
            os.utime(older / "manifest.json", (1, 1))
            self._make_run(runs, "newer", digest)

            self.assertEqual(find_resumable_run(config), "newer")
