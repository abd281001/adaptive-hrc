"""Integration contracts for the active evaluation harness.

These checks intentionally exercise only ``src.evaluation`` public behavior:
scenario construction, online observation/assistance, routing, metric logging,
and on-disk evaluation artifacts.
"""
from __future__ import annotations

import unittest

from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.evaluation import (
    EvaluationConfig,
    ScenarioEvent,
    ScenarioPlan,
    SCENARIO_DEPLOYMENT_RANDOM,
    SCENARIO_LADDER_HETEROGENEOUS,
    SCENARIO_LADDER_HOMOGENEOUS,
    SCENARIOS,
    _should_run_periodic_frozen_eval,
    _should_run_pre_event_frozen_probe,
    aggregate_episode_metrics,
    assist_episode,
    frozen_eval,
    materialize_pair,
    memory_snapshot,
    observe_episode,
    run_event_stream_for_baseline,
)
from src.models import Config


def _fast_eval_config(**overrides):
    values = {
        "seeds": (19,),
        "baselines": ("full",),
        "include_clairvoyant_oracle": False,
        "n_recipes": 2,
        "ladder_rungs": 2,
        "frozen_eval_period": 0,
        "active_only_audit_period": 0,
        "print_eta": False,
        "model_overrides": {
            "maxent_iters_cold": 1,
            "maxent_iters_warm": 1,
        },
    }
    values.update(overrides)
    return EvaluationConfig(**values)


class EvaluationRunnerContractTests(unittest.TestCase):
    def _pair_and_agent(self):
        recipe_name, builder = next(iter(gen.recipe_library().items()))
        pair = materialize_pair(recipe_name, "identity", builder)
        agent = AdaptiveHRCAgent(Config(
            verbose=False,
            maxent_iters_cold=1,
            maxent_iters_warm=1,
        ))
        return recipe_name, pair, agent

    def test_observation_logs_no_robot_prediction_or_turn_records(self):
        recipe_name, pair, agent = self._pair_and_agent()
        name_to_rid = {}

        row = observe_episode(agent, pair, name_to_rid)

        self.assertEqual(row["mode"], "observe")
        self.assertEqual(row["n_steps"], 0)
        self.assertEqual(row["hrc_robot_turn_count"], 0)
        self.assertEqual(row["_turn_records"], [])
        self.assertIn(recipe_name, name_to_rid)
        self.assertEqual(row["testing_total_action_time"], row["testing_human_only_action_time"] + len(pair.actions))

    def test_compute_snapshot_pairs_fit_flops_with_fit_wall_time(self):
        _recipe_name, pair, agent = self._pair_and_agent()
        name_to_rid = {}
        observe_episode(agent, pair, name_to_rid)

        snapshot = memory_snapshot(agent)

        self.assertIn("training_estimated_fit_flops", snapshot)
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
        name_to_rid = {}
        observe_episode(agent, pair, name_to_rid)

        row = assist_episode(
            agent,
            pair,
            name_to_rid,
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
        self.assertEqual(row["n_steps"], row["hrc_robot_turn_count"])
        self.assertTrue(all(turn["scheduled_actor"] == "robot" for turn in robot_turns))
        self.assertTrue(all(turn["scheduled_actor"] == "human" for turn in shadow_turns))
        self.assertGreaterEqual(row["testing_total_action_time"], row["testing_human_only_action_time"])

    def test_absent_assist_request_is_routed_to_observation_mode(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = ScenarioPlan(
            scenario="unit_routing",
            seed=23,
            events=(ScenarioEvent("assist", pair, {"event_type": "unit_absent_assist"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for safe routing of an unseen recipe.",
        )

        stream = run_event_stream_for_baseline("full", plan, _fast_eval_config())
        row = stream.episode_rows[0]

        self.assertEqual(row["requested_mode"], "assist")
        self.assertEqual(row["mode"], "observe")
        self.assertEqual(row["executed_mode"], "observe")
        self.assertEqual(row["mode_route_reason"], "assist_routed_to_observe_recipe_absent_from_active_memory")

    def test_full_realized_schedule_forces_baseline_to_the_same_interaction_mode(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = ScenarioPlan(
            scenario="unit_matched_route",
            seed=23,
            events=(ScenarioEvent("assist", pair, {"event_type": "unit_assist"}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for canonical full interaction routing.",
        )
        config = _fast_eval_config()

        locally_routed = run_event_stream_for_baseline("bigram", plan, config)
        matched = run_event_stream_for_baseline(
            "bigram",
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

    def test_offline_pretrained_frozen_baseline_trains_once_then_never_updates(self):
        (first_name, first_builder), (second_name, second_builder) = list(gen.recipe_library().items())[:2]
        first_pair = materialize_pair(first_name, "identity", first_builder)
        second_pair = materialize_pair(second_name, "p1_mise_en_place", second_builder)
        plan = ScenarioPlan(
            scenario="unit_offline_frozen",
            seed=23,
            events=(
                ScenarioEvent("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                ScenarioEvent("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            selected_recipes=(first_name, second_name),
            selected_preferences=("identity", "p1_mise_en_place"),
            description="Unit offline pretraining then frozen deployment schedule.",
        )

        stream = run_event_stream_for_baseline(
            "offline_pretrained_frozen",
            plan,
            _fast_eval_config(
                offline_pretrained_recipe_fraction=0.50,
                offline_pretrained_preference_fraction=0.50,
            ),
        )

        self.assertTrue(stream.agent._deployment_locked)
        self.assertEqual(len(stream.agent.retrain_events), 1)
        self.assertEqual(stream.episode_rows[0]["offline_training_pair_count"], 1)
        self.assertFalse(stream.episode_rows[0]["deployment_updates_allowed"])
        snapshot = memory_snapshot(stream.agent)
        self.assertEqual(snapshot["offline_training_retrain_count"], 1)
        self.assertFalse(snapshot["deployment_updates_allowed"])

    def test_offline_all_recipes_identity_baseline_covers_every_recipe_but_no_preference_axes(self):
        (first_name, first_builder), (second_name, second_builder) = list(gen.recipe_library().items())[:2]
        first_pair = materialize_pair(first_name, "p1_mise_en_place", first_builder)
        second_pair = materialize_pair(second_name, "p9_equipment_jit_frontload_serving", second_builder)
        plan = ScenarioPlan(
            scenario="unit_all_recipes_identity_frozen",
            seed=23,
            events=(
                ScenarioEvent("observe", first_pair, {"event_type": "unit_deployment_observation"}),
                ScenarioEvent("observe", second_pair, {"event_type": "unit_deployment_observation"}),
            ),
            eval_pairs=(first_pair, second_pair),
            # Identity is intentionally absent here: the baseline's protocol,
            # rather than scenario ordering, defines its sole training preference.
            selected_recipes=(first_name, second_name),
            selected_preferences=("p1_mise_en_place", "p9_equipment_jit_frontload_serving"),
            description="Unit all-recipes identity-only offline training schedule.",
        )

        stream = run_event_stream_for_baseline(
            "offline_all_recipes_identity_frozen",
            plan,
            _fast_eval_config(),
        )

        snapshot = memory_snapshot(stream.agent)
        self.assertTrue(stream.agent._deployment_locked)
        self.assertEqual(snapshot["offline_training_design"], "all_selected_recipes_identity_only")
        self.assertEqual(snapshot["offline_training_recipe_count"], 2)
        self.assertEqual(snapshot["offline_training_preference_names"], ["identity"])
        self.assertEqual(snapshot["offline_training_pair_count"], 2)
        self.assertEqual(len(stream.agent.retrain_events), 2)
        self.assertFalse(snapshot["deployment_updates_allowed"])

    def test_primary_assist_has_a_matched_nonmutating_pre_event_probe(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        plan = ScenarioPlan(
            scenario="unit_pre_event_probe",
            seed=23,
            events=(
                ScenarioEvent("observe", pair, {"event_type": "unit_onboarding"}),
                ScenarioEvent("assist", pair, {"event_type": "unit_primary", "primary_probe": True}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit test for a matched frozen probe before live assistance.",
        )
        stream = run_event_stream_for_baseline(
            "full",
            plan,
            _fast_eval_config(pre_event_frozen_probes=True),
        )
        rows = [row for row in stream.frozen_rows if row.get("probe_phase") == "pre_event"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], pair.label)
        self.assertEqual(rows[0]["live_event_idx"], 1)
        self.assertEqual(len(stream.episode_rows), len(plan.events), "frozen probe must not add a live episode")

    def test_batched_frozen_probe_matches_noncommitting_episode_and_restores_state(self):
        recipe_name, pair, agent = self._pair_and_agent()
        name_to_rid = {}
        observe_episode(agent, pair, name_to_rid)
        config = _fast_eval_config()
        observed_pairs = {pair.label}
        observed_recipes = {recipe_name}

        reference = assist_episode(
            agent,
            pair,
            name_to_rid,
            config=config,
            commit=False,
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        )
        before = agent._frozen_structural_digest()
        rows = frozen_eval(
            agent,
            (pair, pair),
            name_to_rid,
            config=config,
            checkpoint="unit",
            event_idx=1,
            context={"baseline": "full", "scenario": "unit", "seed": 19},
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        )

        self.assertEqual(before, agent._frozen_structural_digest())
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["top1"], reference["live_top1"])
            self.assertEqual(row["topk"], reference["live_topk"])
            self.assertEqual(row["human_correction_rate"], reference["human_correction_rate"])

    def test_public_runner_exposes_the_three_primary_and_two_holdout_scenarios(self):
        self.assertEqual(
            SCENARIOS,
            (
                "ladder_heterogeneous",
                "ladder_homogeneous",
                "ladder_deployment_random",
                "axis_holdout",
                "preference_holdout",
            ),
        )

    def test_frozen_schedule_is_scenario_specific(self):
        recipe_name, pair, _agent = self._pair_and_agent()
        config = _fast_eval_config(frozen_eval_period=5)

        homogeneous = ScenarioPlan(
            scenario=SCENARIO_LADDER_HOMOGENEOUS,
            seed=19,
            events=(
                ScenarioEvent("observe", pair, {"rung_idx": 0}),
                ScenarioEvent("assist", pair, {"rung_idx": 1}),
                ScenarioEvent("assist", pair, {"rung_idx": 1}),
                ScenarioEvent("assist", pair, {"rung_idx": 2}),
                ScenarioEvent("assist", pair, {"rung_idx": 2}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit homogeneous schedule.",
        )
        self.assertEqual(
            [idx for idx in range(len(homogeneous.events)) if _should_run_periodic_frozen_eval(homogeneous, idx, config)],
            [2, 4],
        )

        heterogeneous = ScenarioPlan(
            scenario=SCENARIO_LADDER_HETEROGENEOUS,
            seed=19,
            events=(
                ScenarioEvent("assist", pair, {"rung_idx": 1, "phase_id": "rung_01", "phase_role": "climb"}),
                ScenarioEvent("assist", pair, {"rung_idx": 1, "phase_id": "rung_01", "phase_role": "settled"}),
                ScenarioEvent("assist", pair, {"rung_idx": 1, "phase_id": "rung_01", "phase_role": "settled"}),
                ScenarioEvent("assist", pair, {"rung_idx": 2, "phase_id": "rung_02", "phase_role": "climb"}),
                ScenarioEvent("assist", pair, {"rung_idx": 2, "phase_id": "rung_02", "phase_role": "settled"}),
                ScenarioEvent("assist", pair, {"rung_idx": 2, "phase_id": "rung_02", "phase_role": "settled"}),
            ),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit heterogeneous schedule.",
        )
        self.assertEqual(
            [idx for idx in range(len(heterogeneous.events)) if _should_run_periodic_frozen_eval(heterogeneous, idx, config)],
            [2, 5],
        )

        random_plan = ScenarioPlan(
            scenario=SCENARIO_DEPLOYMENT_RANDOM,
            seed=19,
            events=(ScenarioEvent("assist", pair, {}),),
            eval_pairs=(pair,),
            selected_recipes=(recipe_name,),
            selected_preferences=(pair.preference_name,),
            description="Unit randomized schedule.",
        )
        self.assertFalse(_should_run_periodic_frozen_eval(random_plan, 0, config))
        self.assertFalse(_should_run_pre_event_frozen_probe(random_plan, "assist", "assist", {}, config))
        self.assertTrue(_should_run_pre_event_frozen_probe(random_plan, "assist", "assist", {"primary_probe": True}, config))
        self.assertFalse(_should_run_pre_event_frozen_probe(random_plan, "assist", "observe", {}, config))
        self.assertFalse(_should_run_pre_event_frozen_probe(heterogeneous, "assist", "assist", {}, config))
        self.assertTrue(_should_run_pre_event_frozen_probe(heterogeneous, "assist", "assist", {"primary_probe": True}, config))

    def test_aggregation_uses_robot_turn_support_and_excludes_observation_rows(self):
        rows = [
            {"mode": "observe", "hrc_robot_turn_count": 0, "hrc_robot_correct_count": 0, "hrc_robot_topk_hit_count": 0},
            {"mode": "assist", "hrc_robot_turn_count": 4, "hrc_robot_correct_count": 3, "hrc_robot_topk_hit_count": 4},
        ]
        metrics = aggregate_episode_metrics(rows)

        self.assertEqual(metrics["live_top1"], 0.75)
        self.assertEqual(metrics["live_topk"], 1.0)
        self.assertEqual(metrics["observation_mode_rate"], 0.5)

if __name__ == "__main__":
    unittest.main()
