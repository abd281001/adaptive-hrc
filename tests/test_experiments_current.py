import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src import experiments
from src.adaptive_agent import AdaptiveHRCAgent
from src.baselines import BigramOnlyAgent, ExperienceReplayAgent
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


class ExperimentOutputContractTests(unittest.TestCase):
    def test_action_gate_threshold_selection_prefers_safe_constrained_rows(self):
        curve = {
            "0.05": {
                "coverage": 0.90,
                "conditional_top1": 0.40,
                "wrong_prediction_assist_rate": 0.80,
                "net_assistance_value": 0.20,
                "useful_assistance_rate": 0.36,
            },
            "0.50": {
                "coverage": 0.40,
                "conditional_top1": 0.70,
                "wrong_prediction_assist_rate": 0.20,
                "net_assistance_value": 0.12,
                "useful_assistance_rate": 0.28,
            },
        }
        threshold, selection = experiments.select_action_gate_threshold(
            curve,
            min_conditional_top1=0.55,
            max_wrong_assist_rate=0.45,
        )
        self.assertEqual(threshold, 0.50)
        self.assertEqual(selection["selection_reason"], "max_hrc_assistance_utility_under_constraints")

    def test_action_gate_threshold_selection_can_choose_abstention_when_all_assistance_is_harmful(self):
        curve = {
            "0.10": {
                "coverage": 0.80,
                "conditional_top1": 0.20,
                "wrong_prediction_assist_rate": 0.90,
                "net_assistance_value": -0.50,
                "useful_assistance_rate": 0.16,
            },
            "1.01": {
                "coverage": 0.0,
                "conditional_top1": 0.0,
                "wrong_prediction_assist_rate": 0.0,
                "net_assistance_value": 0.0,
                "useful_assistance_rate": 0.0,
            },
        }
        threshold, selection = experiments.select_action_gate_threshold(curve)
        self.assertEqual(threshold, 1.01)
        self.assertEqual(selection["selection_reason"], "max_hrc_assistance_utility_unconstrained")

    def test_action_gate_threshold_selection_penalizes_hidden_correct_assists(self):
        curve = {
            "0.30": {
                "coverage": 0.80,
                "conditional_top1": 0.80,
                "wrong_prediction_assist_rate": 0.20,
                "useful_assistance_rate": 0.64,
                "hidden_correct_rate": 0.30,
            },
            "0.60": {
                "coverage": 0.72,
                "conditional_top1": 0.85,
                "wrong_prediction_assist_rate": 0.20,
                "useful_assistance_rate": 0.612,
                "hidden_correct_rate": 0.02,
            },
        }
        threshold, selection = experiments.select_action_gate_threshold(
            curve,
            min_conditional_top1=0.55,
            max_wrong_assist_rate=0.45,
            wrong_assist_penalty=1.0,
            hidden_correct_penalty=0.5,
        )
        self.assertEqual(threshold, 0.60)
        self.assertGreater(selection["selected_row"]["hrc_assistance_utility"], 0.0)

    def test_public_registry_uses_canonical_harness(self):
        self.assertEqual(
            experiments.MAIN_EXPERIMENTS,
            (
                "materiality_preflight",
                "deployment_stream",
                "cross_recipe_transfer",
                "decay_reentry",
                "disambiguation_audit",
            ),
        )
        self.assertIn("single_shot_reuse", experiments.LEGACY_EXPERIMENTS)
        self.assertNotIn("single_shot_reuse", experiments.ALL_EXPERIMENTS)
        self.assertNotIn("coverage_accuracy_curve", experiments.EXPERIMENT_RUNNERS)
        for stress_name in (
            "memory_exhaustion_stress",
            "prefix_collision_stress",
            "preference_thrash_stress",
            "rare_reentry_stress",
            "late_distractor_stress",
        ):
            self.assertNotIn(stress_name, experiments.EXPERIMENT_RUNNERS)
        suite_config = experiments.ExperimentSuiteConfig()
        for name in experiments.ALL_EXPERIMENTS:
            self.assertTrue(hasattr(suite_config, name), f"missing suite config for {name}")

    def test_axis_holdout_split_is_available_for_transfer_suite(self):
        for axis in experiments.WORKFLOW_AXES:
            train, holdout = experiments._axis_holdout_split(axis)
            self.assertTrue(train)
            self.assertTrue(holdout)
            self.assertFalse(set(train) & set(holdout))

    def test_seed_result_and_aggregate_are_single_json_outputs(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_exp_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(1337,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("bigram", "experience_replay_bc"),
                quick=True,
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
            for key in ("metadata", "status", "config", "scenario", "baseline_health", "baselines", "metrics", "comparisons", "records", "figures", "warnings"):
                self.assertIn(key, data)
            self.assertIn("episode_steps", data["records"])
            self.assertLess(data["baseline_health"]["experience_replay_bc"]["empty_prediction_rate"], 1.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_canonical_deployment_stream_writes_derived_views(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_canonical_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(1337,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("full", "bigram"),
                quick=True,
            )
            suite_config = experiments.ExperimentSuiteConfig(
                deployment_stream=experiments.DeploymentStreamConfig(
                    n_recipes=2,
                    n_users=2,
                    n_phase_b_events=6,
                    forgetting_checkpoint_interval=3,
                    calibration_thresholds=(0.0, 0.5, 1.01),
                    calibration_n_probe_pairs=3,
                ),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("deployment_stream",))
            self.assertEqual(result["failed_experiments"], {})
            seed_dir = root / "deployment_stream" / "seed_1337"
            data = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
            self.assertIn("derived_views", data["metrics"])
            self.assertIn("coverage_accuracy_curve", data["metrics"]["derived_views"])
            self.assertIn("action_gate_calibration", data["metrics"]["derived_views"])
            self.assertIn("full", data["metrics"]["derived_views"]["action_gate_calibration"])
            self.assertIn("bigram", data["metrics"]["derived_views"]["action_gate_calibration"])
            self.assertIn("episode_steps", data["records"])
            self.assertFalse((seed_dir / "figures").exists())
            f1 = seed_dir / "F1a_live_top1.png"
            self.assertTrue(f1.exists())
            self.assertTrue((seed_dir / "F5a_gate_calibration_selection.png").exists())
            self.assertTrue((seed_dir / "F5b_posthoc_coverage_accuracy.png").exists())
            self.assertTrue((seed_dir / "S1_fixed_vs_calibrated_gate_utility.png").exists())
            f1.unlink()
            experiments.render_figures(root)
            self.assertTrue(f1.exists(), "render_figures should rebuild F-series figures from result.json")
            self.assertTrue((root / "deployment_stream" / "aggregate" / "aggregate_live_top1.png").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_realistic_stream_contains_narrative_event_types(self):
        cfg = experiments.MultiUserStreamConfig(n_users=3, n_recipes=4, n_events=30, switch_probability=0.8)
        _recipes, _users, events = experiments._stream_events(1337, cfg)
        event_types = {e.tags.get("event_type") for e in events}
        self.assertIn("initial_recipe_observation", event_types)
        self.assertGreaterEqual(len(event_types), 3)
        self.assertTrue(any(e.tags.get("four_cell_before") == "seen_unseen" for e in events))

    def test_appendix_figures_are_flat_and_aggregated(self):
        root = Path(tempfile.mkdtemp(prefix="hrc_appendix_contract_"))
        try:
            run_config = experiments.RunConfig(
                seeds=(7,),
                output_root=str(root),
                timestamp_subdir=False,
                workers=1,
                baselines=("bigram", "experience_replay_bc"),
                quick=True,
            )
            suite_config = experiments.ExperimentSuiteConfig(
                demo_count_sample_efficiency=experiments.CapacitySweepConfig(demo_counts=(2, 3), n_preferences=2),
                zipf_usage_sweep=experiments.ZipfUsageConfig(n_recipes=2, n_preferences=3, n_events=4, alphas=(0.3, 0.9)),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("demo_count_sample_efficiency", "zipf_usage_sweep"))
            self.assertEqual(result["failed_experiments"], {})
            demo_seed = root / "demo_count_sample_efficiency" / "seed_7"
            zipf_seed = root / "zipf_usage_sweep" / "seed_7"
            self.assertTrue((demo_seed / "sample_efficiency_live_top1.png").exists())
            self.assertFalse(list(demo_seed.glob("n_*_live_top1.png")))
            self.assertFalse(list(demo_seed.glob("*_capacity.png")))
            self.assertTrue((zipf_seed / "zipf_utility_top1.png").exists())
            self.assertTrue((root / "zipf_usage_sweep" / "aggregate" / "aggregate_zipf_utility_top1.png").exists())
            zipf_data = json.loads((zipf_seed / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("scenario_has_no_recorded_events", zipf_data["warnings"])
            self.assertGreater(zipf_data["metrics"]["n_workflows"], 2)
            first_alpha = next(iter(zipf_data["metrics"]["per_alpha"].values()))
            for row in first_alpha.values():
                self.assertGreater(row["n_eval_pairs"], row["n_unique_phase_b_pairs"])
            aggregate = json.loads((root / "zipf_usage_sweep" / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
            self.assertTrue(aggregate.get("figures"))
            self.assertFalse((zipf_seed / "figures").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
