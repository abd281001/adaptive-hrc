import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src import experiments
from src import hrc_simulation
from src.adaptive_agent import AdaptiveHRCAgent
from src.baselines import BigramOnlyAgent, ExperienceReplayAgent, RecencyPrioritizedReplayAgent
from src.models import Config


class BaselineHealthRegressionTests(unittest.TestCase):
    def test_experience_replay_populates_buffer_without_latest_keys(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        agent = ExperienceReplayAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        name_to_rid = {}
        experiments.observe_episode(agent, pairs[0], name_to_rid)
        dist = agent.predict_next_tokens([])
        self.assertTrue(dist, "experience replay should emit predictions after one committed demo")

    def test_recency_prioritized_replay_populates_buffer_without_latest_keys(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        agent = RecencyPrioritizedReplayAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        name_to_rid = {}
        experiments.observe_episode(agent, pairs[0], name_to_rid)
        dist = agent.predict_next_tokens([])
        meta = agent.replay_buffer_metadata()
        self.assertTrue(dist, "recency replay should emit predictions after one committed demo")
        self.assertEqual(meta["policy"], "recency_prioritized")

    def test_retrain_flops_use_actual_fit_accounting(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        cfg = Config(
            verbose=False,
            maxent_iters_cold=2,
            maxent_iters_warm=1,
            maxent_mc_rollouts=2,
            bc_epochs_cold=5,
            bc_epochs_warm=3,
        )

        full = AdaptiveHRCAgent(cfg=cfg)
        experiments.observe_episode(full, pairs[0], {})
        full_event = [e for e in full.retrain_events if not e.get("skipped")][-1]
        full_stats = full_event.get("fit_stats", {})
        self.assertEqual(full_stats.get("model_family"), "maxent_irl")
        self.assertEqual(full_stats.get("iterations_run"), 2.0)
        self.assertIn("vi_sweeps", full_stats)
        self.assertGreater(full_event["flop_estimate"], 0.0)

        replay = ExperienceReplayAgent(cfg=cfg)
        experiments.observe_episode(replay, pairs[0], {})
        replay_event = [e for e in replay.retrain_events if not e.get("skipped")][-1]
        replay_stats = replay_event.get("fit_stats", {})
        self.assertEqual(replay_stats.get("model_family"), "behavior_cloning")
        self.assertEqual(replay_stats.get("epochs"), 5.0)
        self.assertEqual(replay_stats.get("warm_start"), 0.0)
        self.assertEqual(replay_event["flop_estimate"], replay_stats.get("estimated_flops"))

    def test_bigram_reports_assist_gate_when_predicting(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        agent = BigramOnlyAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        experiments.observe_episode(agent, pairs[0], {})
        event = [e for e in agent.retrain_events if not e.get("skipped")][-1]
        self.assertEqual(event["fit_stats"].get("model_family"), "bigram")
        self.assertEqual(event["flop_estimate"], 0.0)
        dist = agent.predict_next_tokens([])
        gate = agent.assist_gate_stats()
        self.assertTrue(dist)
        self.assertIn(gate.get("reason"), {"baseline_policy", "low_action_confidence"})
        self.assertIsNotNone(gate.get("action_confidence"))

    def test_assist_episode_uses_robot_only_alternating_turns(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        pair = pairs[0]
        agent = experiments.OracleCeilingAgent(cfg=Config(verbose=False))
        metrics, steps = experiments.assist_episode(
            agent,
            pair,
            {},
            run_config=experiments.RunConfig(topk=3),
            commit=False,
        )
        expected_recipe_steps = list(range(1, len(pair.actions), 2))
        self.assertEqual([int(s["recipe_step"]) for s in steps], expected_recipe_steps)
        self.assertEqual(int(metrics["n_steps"]), len(expected_recipe_steps))
        self.assertEqual(int(metrics["n_recipe_steps"]), len(pair.actions))
        self.assertEqual(metrics["live_top1"], 1.0)
        self.assertIn("total_robot_turn_nll", metrics)
        self.assertIn("mean_nll_per_robot_turn", metrics)
        self.assertIn("mean_nll_per_recipe_step", metrics)
        self.assertNotIn("cross_entropy", metrics)
        self.assertEqual(metrics["nll_scored_robot_turns"], metrics["hrc_robot_turn_count"])
        self.assertEqual(metrics["nll_normalization_recipe_steps"], metrics["n_recipe_steps"])
        self.assertAlmostEqual(
            metrics["mean_nll_per_robot_turn"],
            metrics["total_robot_turn_nll"] / max(1.0, metrics["nll_scored_robot_turns"]),
        )
        self.assertAlmostEqual(
            metrics["mean_nll_per_recipe_step"],
            metrics["total_robot_turn_nll"] / max(1.0, metrics["nll_normalization_recipe_steps"]),
        )
        self.assertEqual(metrics["hrc_human_correction_count"], 0)
        self.assertNotIn("hrc_robot_success_rate", metrics)
        self.assertNotIn("hrc_time_saved_vs_human_only", metrics)
        self.assertNotIn("hrc_total_time", metrics)
        self.assertNotIn("human_only_time", metrics)
        self.assertNotIn("normalized_interaction_cost", metrics)
        self.assertNotIn("hrc_time_penalty_vs_human_only", metrics)
        self.assertNotIn("avoidable_correction_cost", metrics)
        self.assertEqual(metrics["human_correction_rate"], 0.0)
        timing = metrics["timing_details"]
        expected_total_time = (
            metrics["hrc_human_turn_count"] * experiments.DEFAULT_HRC_TIMING.human_action_time
            + metrics["hrc_robot_correct_count"] * experiments.DEFAULT_HRC_TIMING.robot_correct_action_time
        )
        self.assertEqual(timing["hrc_total_time"], expected_total_time)
        self.assertGreater(timing["normalized_interaction_cost"], 1.0)
        self.assertGreater(timing["hrc_time_penalty_vs_human_only"], 0.0)
        expected_human_effort = metrics["hrc_human_turn_count"] * experiments.DEFAULT_HRC_TIMING.human_action_time
        self.assertEqual(metrics["human_effort_time"], expected_human_effort)
        self.assertAlmostEqual(metrics["human_effort_fraction"], expected_human_effort / timing["human_only_time"])
        self.assertNotIn("normalized_human_effort", metrics)
        self.assertNotIn("correction_recovery_penalty_time", metrics)
        self.assertEqual(metrics["primary_hrc_metric"], "human_action_fraction")
        self.assertEqual(metrics["primary_hrc_metric_value"], metrics["human_action_fraction"])
        self.assertIn("timing_details", metrics)
        self.assertIn("only when HRCTimingConfig is calibrated", timing["warning"])
        self.assertIn("human_effort_fraction", timing["warning"])
        self.assertIn("hrc_time_saved_vs_human_only", timing)
        self.assertLess(timing["hrc_time_saved_vs_human_only"], 0.0)
        self.assertAlmostEqual(
            metrics["human_action_fraction"],
            metrics["hrc_human_action_count"] / metrics["n_recipe_steps"],
        )
        self.assertGreater(metrics["human_workload_reduction_vs_human_only"], 0.0)
        self.assertTrue(all("assist_used" not in s for s in steps))
        self.assertTrue(all("policy_diagnostics" in s for s in steps))
        self.assertTrue(all("policy_confidence" in s["policy_diagnostics"] for s in steps))
        self.assertTrue(all(s["executed_by"] == "robot" for s in steps))
        self.assertTrue(all("human_only_time_so_far" not in s for s in steps))
        expected_human_actions_so_far = list(range(1, len(steps) + 1))
        self.assertEqual([int(s["human_actions_so_far"]) for s in steps], expected_human_actions_so_far)
        self.assertEqual(
            [float(s["human_effort_time_so_far"]) for s in steps],
            [n * experiments.DEFAULT_HRC_TIMING.human_action_time for n in expected_human_actions_so_far],
        )

    def test_prediction_wall_time_excludes_runner_distribution_copy(self):
        clock = {"t": 0.0}

        class SlowCopyDistribution:
            def __init__(self, values):
                self._values = dict(values)

            def keys(self):
                clock["t"] += 10.0
                return self._values.keys()

            def __getitem__(self, key):
                return self._values[key]

        def predict_distribution(prefix):
            self.assertEqual(tuple(prefix), ("human_action",))
            clock["t"] += 0.25
            return SlowCopyDistribution({"robot_action": 1.0})

        observed = []
        original_perf_counter = hrc_simulation.time.perf_counter
        hrc_simulation.time.perf_counter = lambda: clock["t"]
        try:
            trace = hrc_simulation.run_alternating_hrc_episode(
                observations=("obs_human", "obs_robot"),
                actual_tokens=("human_action", "robot_action"),
                actual_actions=("human_action", "robot_action"),
                current_prefix=lambda: ("human_action",),
                predict_distribution=predict_distribution,
                observe_ground_truth=lambda obs, distribution=None: observed.append((obs, distribution)),
                topk=1,
                prob_floor=1e-12,
            )
        finally:
            hrc_simulation.time.perf_counter = original_perf_counter

        self.assertEqual(len(trace.robot_turns), 1)
        self.assertAlmostEqual(trace.robot_turns[0].prediction_wall_s, 0.25)
        self.assertEqual(trace.robot_turns[0].distribution, {"robot_action": 1.0})
        self.assertGreater(clock["t"], 10.0, "copy cost should still occur after timing")
        self.assertIsNone(observed[0][1])
        self.assertEqual(observed[1][1], {"robot_action": 1.0})

    def test_assist_episode_reuses_non_oracle_robot_prediction_for_step_log(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        pair = pairs[0]

        class CountingAgent(AdaptiveHRCAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.target_tokens = []
                self.prediction_prefixes = []

            def predict_next_tokens(self, prefix=None):
                prefix = list(prefix) if prefix is not None else list(self.current_prefix)
                self.prediction_prefixes.append(tuple(prefix))
                idx = len(prefix)
                if idx >= len(self.target_tokens):
                    return {}
                return {self.target_tokens[idx]: 1.0}

        agent = CountingAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        name_to_rid = {}
        experiments.observe_episode(agent, pair, name_to_rid)
        agent.target_tokens = agent._tokens_from_action_labels(pair.actions)
        agent.prediction_prefixes = []

        metrics, _steps = experiments.assist_episode(
            agent,
            pair,
            name_to_rid,
            run_config=experiments.RunConfig(topk=3),
            commit=True,
        )

        self.assertEqual(metrics["live_top1"], 1.0)
        self.assertEqual(len(agent.prediction_prefixes), len(pair.actions))
        self.assertEqual(len(set(agent.prediction_prefixes)), len(agent.prediction_prefixes))

    def test_oracle_label_assist_keeps_non_oracle_step_log(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        pair = pairs[0]
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        name_to_rid = {}
        experiments.observe_episode(agent, pair, name_to_rid)
        actual_tokens = agent._tokens_from_action_labels(pair.actions)
        sentinel_prediction = "__non_oracle_prediction__"

        def oracle_predict(prefix, _oracle_recipe_id, _oracle_pref_id):
            idx = len(list(prefix))
            if idx >= len(actual_tokens):
                return {}
            return {actual_tokens[idx]: 1.0}

        def non_oracle_predict(prefix=None):
            return {sentinel_prediction: 1.0}

        agent.predict_with_oracle = oracle_predict
        agent.predict_next_tokens = non_oracle_predict
        log_start = len(agent.step_log)

        metrics, steps = experiments.assist_episode(
            agent,
            pair,
            name_to_rid,
            run_config=experiments.RunConfig(topk=3),
            commit=True,
            oracle_recipe_id=name_to_rid[pair.recipe_name],
        )

        new_logs = agent.step_log[log_start:]
        self.assertEqual(metrics["live_top1"], 1.0)
        self.assertEqual(steps[0]["predicted"], actual_tokens[1])
        self.assertEqual(new_logs[1].actual, actual_tokens[1])
        self.assertEqual(new_logs[1].predicted, sentinel_prediction)
        self.assertFalse(new_logs[1].correct)

    def test_future_valid_robot_action_is_corrected_without_advancing_turn(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        pair = pairs[0]

        class FutureActionTooEarlyAgent(experiments.OracleCeilingAgent):
            def set_oracle_target(self, actions):
                super().set_oracle_target(actions)
                self._future_injected = False

            def predict_next_tokens(self, prefix=None):
                if self._oracle_target is None:
                    return {}
                prefix = list(prefix) if prefix is not None else list(self.current_prefix)
                idx = len(prefix)
                if idx >= len(self._oracle_target):
                    return {}
                if (not self._future_injected) and idx == 1 and idx + 2 < len(self._oracle_target):
                    self._future_injected = True
                    return {self._oracle_target[idx + 2]: 1.0}
                return {"invalid_robot_action": 1.0}

        agent = FutureActionTooEarlyAgent(cfg=Config(verbose=False))
        name_to_rid = {}
        experiments.observe_episode(agent, pair, name_to_rid)
        metrics, steps = experiments.assist_episode(
            agent,
            pair,
            name_to_rid,
            run_config=experiments.RunConfig(topk=3),
            commit=False,
        )
        self.assertEqual([int(s["recipe_step"]) for s in steps], list(range(1, len(pair.actions))))
        self.assertTrue(steps[0]["future_valid_wrong"])
        self.assertEqual(steps[0]["predicted_future_offset"], 2)
        self.assertEqual(metrics["hrc_human_correction_count"], metrics["n_steps"])
        self.assertEqual(metrics["future_valid_wrong_rate"], 1.0 / metrics["n_steps"])
        timing = metrics["timing_details"]
        self.assertGreater(timing["hrc_total_time"], timing["human_only_time"])
        self.assertNotIn("hrc_time_saved_vs_human_only", metrics)
        self.assertNotIn("normalized_interaction_cost", metrics)
        self.assertNotIn("hrc_time_penalty_vs_human_only", metrics)
        self.assertNotIn("avoidable_correction_cost", metrics)
        self.assertEqual(metrics["human_correction_rate"], 1.0)
        self.assertGreater(timing["normalized_interaction_cost"], 1.0)
        self.assertGreater(timing["hrc_time_penalty_vs_human_only"], 0.0)
        expected_human_effort = (
            metrics["hrc_human_turn_count"] * experiments.DEFAULT_HRC_TIMING.human_action_time
            + metrics["hrc_human_correction_count"] * experiments.DEFAULT_HRC_TIMING.human_correction_time
        )
        self.assertEqual(metrics["human_effort_time"], expected_human_effort)
        self.assertAlmostEqual(metrics["human_effort_fraction"], expected_human_effort / timing["human_only_time"])
        self.assertNotIn("normalized_human_effort", metrics)
        self.assertNotIn("correction_recovery_penalty_time", metrics)
        self.assertEqual(
            timing["hrc_wrong_action_extra_penalty_time"],
            metrics["hrc_robot_wrong_count"] * experiments.DEFAULT_HRC_TIMING.wrong_action_extra_penalty,
        )
        self.assertTrue(all("human_only_time_so_far" not in s for s in steps))
        expected_human_actions_so_far = list(range(2, len(pair.actions) + 1))
        self.assertEqual([int(s["human_actions_so_far"]) for s in steps], expected_human_actions_so_far)
        self.assertEqual(
            [float(s["human_effort_time_so_far"]) for s in steps],
            [
                experiments.DEFAULT_HRC_TIMING.human_action_time
                + (idx + 1) * experiments.DEFAULT_HRC_TIMING.human_correction_time
                for idx, _s in enumerate(steps)
            ],
        )

    def test_aggregate_pools_correction_aware_human_effort(self):
        rows = [
            {
                "mode": "assist",
                "n_steps": 1,
                "n_recipe_steps": 2,
                "hrc_robot_turn_count": 1,
                "hrc_human_turn_count": 1,
                "hrc_human_correction_count": 0,
                "hrc_human_action_count": 1,
                "hrc_robot_correct_count": 1,
                "hrc_robot_wrong_count": 0,
                "human_effort_time": 4.0,
                "timing_details": {
                    "human_only_time": 8.0,
                    "hrc_total_time": 10.0,
                    "observation_mode_extra_time": 0.0,
                    "hrc_robot_correct_time": 6.0,
                    "hrc_robot_wrong_time": 0.0,
                    "hrc_human_action_time": 4.0,
                    "hrc_human_correction_time": 0.0,
                    "hrc_wrong_action_extra_penalty_time": 0.0,
                },
            },
            {
                "mode": "assist",
                "n_steps": 1,
                "n_recipe_steps": 2,
                "hrc_robot_turn_count": 1,
                "hrc_human_turn_count": 1,
                "hrc_human_correction_count": 1,
                "hrc_human_action_count": 2,
                "hrc_robot_correct_count": 0,
                "hrc_robot_wrong_count": 1,
                "human_effort_time": 8.0,
                "timing_details": {
                    "human_only_time": 8.0,
                    "hrc_total_time": 13.0,
                    "observation_mode_extra_time": 0.0,
                    "hrc_robot_correct_time": 0.0,
                    "hrc_robot_wrong_time": 2.0,
                    "hrc_human_action_time": 4.0,
                    "hrc_human_correction_time": 4.0,
                    "hrc_wrong_action_extra_penalty_time": 3.0,
                },
            },
        ]

        aggregate = experiments._aggregate_episode_metrics(rows)

        self.assertEqual(aggregate["human_effort_time"], 12.0)
        self.assertEqual(aggregate["human_action_fraction"], 0.75)
        self.assertEqual(aggregate["human_effort_fraction"], 0.75)
        self.assertNotIn("normalized_human_effort", aggregate)
        self.assertNotIn("correction_recovery_penalty_time", aggregate)
        self.assertNotIn("avoidable_correction_cost", aggregate)
        self.assertNotIn("hrc_total_time", aggregate)
        self.assertEqual(aggregate["timing_details"]["hrc_total_time"], 23.0)
        self.assertEqual(aggregate["timing_details"]["human_only_time"], 16.0)
        self.assertEqual(aggregate["timing_details"]["hrc_wrong_action_extra_penalty_time"], 3.0)
        self.assertNotIn("correction_recovery_penalty_time", aggregate["timing_details"])

    def test_aggregate_pools_nll_by_recipe_step_support(self):
        rows = [
            {
                "mode": "assist",
                "n_steps": 1,
                "n_recipe_steps": 10,
                "hrc_robot_turn_count": 1,
                "total_robot_turn_nll": 5.0,
                "nll_scored_robot_turns": 1,
                "nll_normalization_recipe_steps": 10,
                "mean_nll_per_robot_turn": 5.0,
                "mean_nll_per_recipe_step": 0.5,
            },
            {
                "mode": "assist",
                "n_steps": 5,
                "n_recipe_steps": 10,
                "hrc_robot_turn_count": 5,
                "total_robot_turn_nll": 20.0,
                "nll_scored_robot_turns": 5,
                "nll_normalization_recipe_steps": 10,
                "mean_nll_per_robot_turn": 4.0,
                "mean_nll_per_recipe_step": 2.0,
            },
            {
                "mode": "observe",
                "n_steps": 0,
                "n_recipe_steps": 10,
                "hrc_robot_turn_count": 0,
                "total_robot_turn_nll": 0.0,
                "nll_scored_robot_turns": 0,
                "nll_normalization_recipe_steps": 0,
                "mean_nll_per_robot_turn": 0.0,
                "mean_nll_per_recipe_step": 0.0,
            },
        ]

        aggregate = experiments._aggregate_episode_metrics(rows)

        self.assertEqual(aggregate["total_robot_turn_nll"], 25.0)
        self.assertEqual(aggregate["nll_scored_robot_turns"], 6.0)
        self.assertEqual(aggregate["nll_normalization_recipe_steps"], 20.0)
        self.assertAlmostEqual(aggregate["mean_nll_per_robot_turn"], 25.0 / 6.0)
        self.assertAlmostEqual(aggregate["mean_nll_per_recipe_step"], 25.0 / 20.0)
        self.assertNotIn("cross_entropy", aggregate)

    def test_adaptation_recovery_uses_fixed_windows_and_censoring(self):
        early_mismatch = {
            "mode": "assist",
            "n_steps": 5,
            "live_top1": 0.6,
            **experiments._adaptation_recovery_metrics(
                [True, False, True, True, False],
                1,
            ),
        }
        late_mismatch = {
            "mode": "assist",
            "n_steps": 3,
            "live_top1": 0.67,
            **experiments._adaptation_recovery_metrics(
                [True, True, False],
                2,
            ),
        }

        aggregate = experiments._aggregate_episode_metrics([early_mismatch, late_mismatch])
        self.assertAlmostEqual(aggregate["adaptation_recovery_top1_w3"], 2.0 / 3.0)
        self.assertAlmostEqual(aggregate["adaptation_recovery_eligible_rate_w3"], 0.5)
        self.assertAlmostEqual(aggregate["adaptation_recovery_censored_rate_w3"], 0.5)
        self.assertEqual(aggregate["n_adaptation_recovery_eligible_w3"], 1.0)
        self.assertTrue(aggregate["adaptation_recovery_window_curve"])
        self.assertTrue(aggregate["first_mismatch_position_curve"])


class ExperimentOutputContractTests(unittest.TestCase):
    def test_public_registry_uses_unified_experiment_list(self):
        self.assertIn("materiality_preflight", experiments.EXPERIMENTS)
        self.assertIn("deployment_stream", experiments.EXPERIMENTS)
        self.assertIn("confidence_threshold_accuracy", experiments.EXPERIMENTS)
        self.assertIn("single_shot_reuse", experiments.EXPERIMENTS)
        self.assertIn("demo_count_sample_efficiency", experiments.EXPERIMENTS)
        for baseline in ("bc", "ewc", "online_ewc"):
            self.assertIn(baseline, experiments.PAPER_BASELINES)
        self.assertEqual(
            experiments.TransferSuiteConfig().baselines,
            (
                "full",
                "latest_only",
                "bc",
                "experience_replay_bc",
                "recency_prioritized_replay",
                "ewc",
                "online_ewc",
                "bigram",
                "no_preference_prototype",
            ),
        )
        removed_redundant = (
            "materiality_audit",
            "cross_recipe_preference_transfer",
            "preference_axis_holdout",
            "novel_preference_composition",
            "adaptive_decay_reentry",
            "multi_user_continual_stream",
            "bounded_memory_tradeoff",
            "compute_tradeoff",
            "disambiguation_threshold_calibration",
            "action_gate_threshold_calibration",
            "boundary_degradation_disambiguation",
            "reentry_gap_neutral_vs_distractor",
            "active_pruned_decay_probe",
            "confidence_calibration_reliability",
        )
        suite_config = experiments.ExperimentSuiteConfig()
        for name in removed_redundant:
            self.assertNotIn(name, experiments.EXPERIMENTS)
            self.assertNotIn(name, experiments.EXPERIMENT_RUNNERS)
            self.assertFalse(hasattr(suite_config, name))
            self.assertFalse(hasattr(experiments, f"run_{name}"))
        for stress_name in (
            "memory_exhaustion_stress",
            "prefix_collision_stress",
            "preference_thrash_stress",
            "rare_reentry_stress",
            "late_distractor_stress",
        ):
            self.assertIn(stress_name, experiments.EXPERIMENT_RUNNERS)
            self.assertIn(stress_name, experiments.EXPERIMENTS)
        self.assertIn("adversarial_stress_suite", experiments.EXPERIMENTS)
        self.assertIn("baseline_hyperparameter_tuning", experiments.EXPERIMENTS)
        self.assertIn("deployment_ladder_heterogeneous_prefs", experiments.EXPERIMENTS)
        self.assertIn("deployment_ladder_shared_pref", experiments.EXPERIMENTS)
        for name in experiments.EXPERIMENTS:
            self.assertIn(name, experiments.EXPERIMENT_RUNNERS)
            self.assertTrue(hasattr(suite_config, name), f"missing suite config for {name}")

    def test_disambiguation_audit_uses_shared_validation_protocol(self):
        _recipes, pairs = experiments.build_pairs(17, 3, n_preferences=4)
        cfg = experiments.DisambiguationAuditConfig(
            n_recipes=3,
            thresholds=(0.85, 0.95),
            online_partial_thresholds=(0.20, 0.30),
            min_classify_lengths=(3, 6),
            min_classify_fractions=(0.35,),
            prefix_fractions=(0.25, 0.50, 1.00),
            validation_fraction=0.34,
        )
        calibration = experiments._disambiguation_shared_validation_protocol(
            17,
            experiments.RunConfig(seeds=(17,), workers=1),
            cfg,
            pairs,
        )

        self.assertFalse(hasattr(experiments, "_independent_disambiguation_validation"))
        self.assertEqual(calibration["protocol"], "shared_prefix_full_validation")
        self.assertTrue(calibration["validation_rows"])
        self.assertTrue(calibration["partial_precision_recall_curve"])
        self.assertTrue(calibration["full_precision_recall_curve"])
        self.assertTrue(calibration["joint_grid"])
        self.assertIn("transition_consistency", calibration["joint_grid"][0])
        self.assertIn("min_classify_rule", calibration["recommendations"])
        self.assertIn("selected_metrics", calibration["recommendations"])
        self.assertNotIn("online_partial_thresholds", calibration)
        self.assertNotIn("jaccard_thresholds", calibration)

    def test_active_experiment_harness_has_no_quick_mode(self):
        self.assertNotIn("quick", experiments.RunConfig.__dataclass_fields__)
        with self.assertRaises(TypeError):
            experiments.RunConfig(quick=True)

    def test_transfer_suite_uses_current_condition_switches(self):
        cfg = experiments.TransferSuiteConfig()
        self.assertTrue(cfg.include_cross_axis_value_transfer)
        self.assertTrue(cfg.include_axis_family_holdout)
        self.assertFalse(hasattr(cfg, "include_axis_holdout"))
        self.assertFalse(hasattr(experiments, "_axis_holdout_split"))

    def test_transfer_condition_builders_cover_axis_values(self):
        builders = experiments.select_recipe_builders(17, 8)
        axis_value_specs = experiments._cross_axis_value_transfer_specs(builders)
        family_specs = experiments._axis_family_holdout_specs(builders)
        self.assertTrue(axis_value_specs)
        self.assertTrue(family_specs)
        self.assertTrue({s["condition"] for s in axis_value_specs} <= {"cross_axis_value_transfer"})
        self.assertTrue({s["condition"] for s in family_specs} <= {"axis_family_holdout"})

    def test_seed_result_and_aggregate_are_single_json_outputs(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_exp_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(1337,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("bigram", "experience_replay_bc"),
            )
            suite_config = experiments.ExperimentSuiteConfig(
                single_shot_reuse=experiments.SingleShotReuseConfig(n_recipes=1, n_preferences=1, test_preferences=("identity",)),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("single_shot_reuse",))
            self.assertEqual(result["failed_experiments"], {})
            seed_dir = root / "single_shot_reuse" / "seed_1337"
            exp_dir = root / "single_shot_reuse"
            self.assertTrue((seed_dir / "result.json").exists())
            self.assertTrue((exp_dir / "aggregate" / "aggregate.json").exists())
            self.assertFalse((seed_dir / "metrics.json").exists())
            self.assertFalse((seed_dir / "episode_metrics.jsonl").exists())
            self.assertFalse((seed_dir / "figures").exists())
            data = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
            for key in ("metadata", "status", "config", "scenario", "baseline_health", "baselines", "metrics", "comparisons", "records", "warnings"):
                self.assertIn(key, data)
            self.assertIn("episode_steps", data["records"])
            self.assertLess(data["baseline_health"]["experience_replay_bc"]["empty_prediction_rate"], 1.0)
            self.assertNotIn("figures", data)
            self.assertFalse(list(seed_dir.glob("*.png")))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_deployment_stream_writes_logs_without_plot_views(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_deployment_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(1337,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("full", "bigram"),
            )
            suite_config = experiments.ExperimentSuiteConfig(
                deployment_stream=experiments.DeploymentStreamConfig(
                    n_recipes=2,
                    n_users=2,
                    n_phase_b_events=6,
                ),
                materiality_preflight=experiments.MaterialityPreflightConfig(
                    n_recipes=2,
                    n_preferences=3,
                    min_effective_preferences=2,
                    min_classify_length=100,
                    fail_on_duplicate_ordering=False,
                ),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("deployment_stream",))
            self.assertEqual(result["failed_experiments"], {})
            seed_dir = root / "deployment_stream" / "seed_1337"
            data = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("derived_views", data["metrics"])
            self.assertNotIn("memory_pareto", data["metrics"])
            removed_metrics = (
                "conditional_top1",
                "precision_when_assist",
                "useful_assistance_rate",
                "assist_correct_rate",
                "correct_prediction_assist_recall",
                "wrong_prediction_assist_rate",
                "hidden_correct_rate",
                "confidence_threshold_accuracy_curve",
                "action_gate_calibration",
            )
            for metric_name in removed_metrics:
                self.assertNotIn(metric_name, data["metrics"]["per_baseline"]["full"])
            self.assertEqual(data["metrics"]["per_baseline"]["full"]["primary_forgetting_metric"], "online_best_so_far_forgetting")
            self.assertIn("online_best_so_far_forgetting", data["metrics"]["per_baseline"]["full"])
            self.assertIn("worst_per_pair_histogram", data["metrics"]["per_baseline"]["full"]["online_best_so_far_forgetting"])
            self.assertEqual(data["metrics"]["primary_forgetting_metric"], "online_best_so_far_forgetting")
            self.assertIn("online_best_so_far_forgetting_summary", data["metrics"])
            self.assertIn("preference_reentry_recovery", data["metrics"]["per_baseline"]["full"])
            self.assertIn("user_switch_recovery", data["metrics"]["per_baseline"]["full"])
            self.assertNotIn("memory_efficiency", data["metrics"]["per_baseline"]["full"])
            self.assertIn("preference_decay_profile", data["metrics"])
            self.assertIn("valid_random_floor", data["metrics"])
            self.assertIn("valid_random_floor", data["metrics"]["per_baseline"]["full"])
            self.assertIn("coarse_exposure_cell", data["metrics"]["per_baseline"]["full"])
            self.assertIn("event_mix_sensitivity", data["metrics"])
            self.assertIn("oracle_references", data["metrics"])
            for oracle_row in data["metrics"]["oracle_references"].values():
                self.assertIn("mean", oracle_row)
                self.assertIn("ci95", oracle_row)
                self.assertEqual(oracle_row["reported_as"], "dashed_reference_not_deployable")
            self.assertIn("episode_steps", data["records"])
            self.assertIn("online_forgetting_trace", data["records"])
            self.assertIn("user_switch_recovery", data["records"])
            self.assertIn("preference_reentry_recovery", data["records"])
            self.assertIn("adaptation_recovery_top1_w3", data["metrics"]["per_baseline"]["full"])
            self.assertIn("adaptation_recovery_eligible_rate_w3", data["metrics"]["per_baseline"]["full"])
            self.assertIn("adaptation_recovery_censored_rate_w3", data["metrics"]["per_baseline"]["full"])
            self.assertIn("first_mismatch_position_curve", data["metrics"]["per_baseline"]["full"])
            self.assertIn("adaptation_recovery_window_curve", data["metrics"]["per_baseline"]["full"])
            self.assertFalse((seed_dir / "figures").exists())
            self.assertNotIn("figures", data)
            self.assertFalse(list(seed_dir.glob("*.png")))
            aggregate = json.loads((root / "deployment_stream" / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
            self.assertNotIn("figures", aggregate)
            self.assertFalse(list(seed_dir.glob("*.png")))
            self.assertFalse(list((root / "deployment_stream" / "aggregate").glob("*.png")))
            self.assertFalse(hasattr(experiments, "render_figures"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_deployment_ladder_streams_use_observe_then_assist_rungs(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_ladder_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(11,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("full", "bigram"),
            )
            ladder_cfg = experiments.DeploymentLadderConfig(
                n_recipes=2,
                n_rungs=2,
                settle_repeats_per_recipe=1,
                preferences=("identity", "p1_prep_first", "p2_frontload", "p3_clean_eager", "p4_prep_clean"),
            )
            suite_config = experiments.ExperimentSuiteConfig(
                deployment_ladder_heterogeneous_prefs=ladder_cfg,
                deployment_ladder_shared_pref=ladder_cfg,
            )
            result = experiments.run_suite(
                run_config=run_config,
                suite_config=suite_config,
                experiments=("deployment_ladder_heterogeneous_prefs", "deployment_ladder_shared_pref"),
            )
            self.assertEqual(result["failed_experiments"], {})
            for name in ("deployment_ladder_heterogeneous_prefs", "deployment_ladder_shared_pref"):
                seed_dir = root / name / "seed_11"
                data = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
                metrics = data["metrics"]
                events = data["scenario"]["events"]
                first_rung = [e for e in events if int(e.get("rung_idx", -1)) == 0 and e.get("ladder_stage") == "add"]
                later_add = [e for e in events if int(e.get("rung_idx", -1)) > 0 and e.get("ladder_stage") == "add"]
                settle = [e for e in events if e.get("ladder_stage") == "settle"]
                self.assertTrue(first_rung)
                self.assertTrue(later_add)
                self.assertTrue(settle)
                self.assertTrue(all(e["mode"] == "observe" for e in first_rung))
                self.assertTrue(all(e["mode"] == "assist" for e in later_add + settle))
                self.assertEqual(metrics["n_rungs"], 2)
                self.assertEqual(metrics["settle_events_per_rung"], 2)
                self.assertNotIn("derived_views", metrics)
                self.assertNotIn("memory_pareto", metrics)
                self.assertIn("per_rung", metrics["per_baseline"]["full"])
                self.assertIn("per_rung_stage", metrics["per_baseline"]["full"])
                self.assertIn("rung_memory_trace", metrics["per_baseline"]["full"])
                self.assertIn("per_rung_stage", metrics["per_baseline"]["full"])
                self.assertGreater(metrics["per_baseline"]["full"]["per_rung_stage"]["1"]["add"]["n_episodes"], 0.0)
                aggregate = json.loads((root / name / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
                self.assertNotIn("figures", data)
                self.assertNotIn("figures", aggregate)
                self.assertFalse(list(seed_dir.glob("*.png")))
                self.assertFalse(list((root / name / "aggregate").glob("*.png")))
                if name == "deployment_ladder_shared_pref":
                    for rung in metrics["rung_plan"]:
                        prefs = {pair["preference"] for pair in rung["pairs"]}
                        self.assertEqual(len(prefs), 1)
                else:
                    by_recipe = {}
                    for rung in metrics["rung_plan"]:
                        for pair in rung["pairs"]:
                            by_recipe.setdefault(pair["recipe"], []).append(pair["preference"])
                    for prefs in by_recipe.values():
                        self.assertEqual(len(prefs), len(set(prefs)))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_realistic_stream_contains_narrative_event_types(self):
        cfg = experiments.MultiUserStreamConfig(n_users=3, n_recipes=4, n_events=30, switch_probability=0.8)
        _recipes, _users, events = experiments._stream_events(1337, cfg)
        event_types = {e.tags.get("event_type") for e in events}
        self.assertIn("initial_recipe_observation", event_types)
        self.assertGreaterEqual(len(event_types), 3)
        self.assertTrue(any(e.tags.get("four_cell_before") == "seen_recipe_new_preference" for e in events))

    def test_supporting_experiments_are_logged_without_figures(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_supporting_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(7,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("bigram", "experience_replay_bc"),
            )
            suite_config = experiments.ExperimentSuiteConfig(
                demo_count_sample_efficiency=experiments.CapacitySweepConfig(demo_counts=(1, 2), n_preferences=2),
                zipf_usage_sweep=experiments.ZipfUsageConfig(n_recipes=2, n_preferences=3, n_events=4, alphas=(0.3, 0.9)),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("demo_count_sample_efficiency", "zipf_usage_sweep"))
            self.assertEqual(result["failed_experiments"], {})
            demo_seed = root / "demo_count_sample_efficiency" / "seed_7"
            zipf_seed = root / "zipf_usage_sweep" / "seed_7"
            demo_data = json.loads((demo_seed / "result.json").read_text(encoding="utf-8"))
            self.assertIn("preference_transfer_sample_efficiency", demo_data["metrics"])
            self.assertIn("preference_variant_sample_efficiency", demo_data["metrics"])
            self.assertIn("sample_efficiency_curves", demo_data["metrics"])
            self.assertIn("sample_efficiency_metric_curves", demo_data["metrics"])
            self.assertIn("preference_transfer", demo_data["metrics"]["sample_efficiency_curves"])
            self.assertIn("preference_variant", demo_data["metrics"]["sample_efficiency_curves"])
            self.assertIn("oracle_ceiling", demo_data["metrics"]["sample_efficiency_metric_curves"]["recipe_onboarding"]["live_top1"])
            self.assertNotIn("figures", demo_data)
            self.assertFalse(list(demo_seed.glob("*.png")))
            zipf_data = json.loads((zipf_seed / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("scenario_has_no_recorded_events", zipf_data["warnings"])
            self.assertGreater(zipf_data["metrics"]["n_workflows"], 2)
            first_alpha = next(iter(zipf_data["metrics"]["per_alpha"].values()))
            for row in first_alpha.values():
                self.assertGreater(row["n_eval_pairs"], row["n_unique_phase_b_pairs"])
            aggregate = json.loads((root / "zipf_usage_sweep" / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
            self.assertNotIn("figures", zipf_data)
            self.assertNotIn("figures", aggregate)
            self.assertFalse(list(zipf_seed.glob("*.png")))
            self.assertFalse(list((root / "zipf_usage_sweep" / "aggregate").glob("*.png")))
            self.assertFalse((zipf_seed / "figures").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_baseline_hyperparameter_tuning_reports_shared_protocol(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_baseline_tuning_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(5,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
            )
            suite_config = experiments.ExperimentSuiteConfig(
                baseline_hyperparameter_tuning=experiments.BaselineHyperparameterTuningConfig(
                    n_recipes=2,
                    n_preferences=2,
                    baselines=("bc", "experience_replay_bc", "recency_prioritized_replay", "ewc"),
                    max_grid_points_per_baseline=2,
                    max_eval_pairs=3,
                )
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("baseline_hyperparameter_tuning",))
            self.assertEqual(result["failed_experiments"], {})
            data = json.loads((root / "baseline_hyperparameter_tuning" / "seed_5" / "result.json").read_text(encoding="utf-8"))
            metrics = data["metrics"]
            self.assertTrue(metrics["shared_calibration_protocol"]["same_split_for_all_systems"])
            self.assertTrue(metrics["reported_only_does_not_mutate_default_config"])
            self.assertIn("grid_ranges", metrics)
            self.assertIn("recency_prioritized_replay", metrics["per_baseline"])
            recommendation = metrics["per_baseline"]["recency_prioritized_replay"]["recommendation"]
            self.assertIn("er_buffer_size", recommendation["selected_values"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
