import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src import experiments
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

    def test_bigram_reports_assist_gate_when_predicting(self):
        _recipes, pairs = experiments.build_pairs(1337, 1, preferences=("identity",))
        self.assertTrue(pairs)
        agent = BigramOnlyAgent(cfg=Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        experiments.observe_episode(agent, pairs[0], {})
        dist = agent.predict_next_tokens([])
        gate = agent.assist_gate_stats()
        self.assertTrue(dist)
        self.assertIn(gate.get("reason"), {"baseline_policy", "low_action_confidence"})
        self.assertIsNotNone(gate.get("action_confidence"))


class ExperimentOutputContractTests(unittest.TestCase):
    def test_public_registry_uses_canonical_harness(self):
        self.assertEqual(
            experiments.MAIN_EXPERIMENTS,
            (
                "deployment_stream",
                "cross_recipe_transfer",
                "decay_reentry",
                "disambiguation_audit",
                "materiality_preflight",
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
                single_shot_reuse=experiments.SingleShotReuseConfig(n_recipes=1, preferences=("identity",)),
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
                ),
            )
            result = experiments.run_suite(run_config=run_config, suite_config=suite_config, experiments=("deployment_stream",))
            self.assertEqual(result["failed_experiments"], {})
            seed_dir = root / "deployment_stream" / "seed_1337"
            data = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
            self.assertIn("derived_views", data["metrics"])
            self.assertIn("coverage_accuracy_curve", data["metrics"]["derived_views"])
            self.assertIn("episode_steps", data["records"])
            self.assertFalse((seed_dir / "figures").exists())
            f1 = seed_dir / "F1a_live_top1.png"
            self.assertTrue(f1.exists())
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
                zipf_usage_sweep=experiments.ZipfUsageConfig(n_recipes=2, n_events=4, alphas=(0.3, 0.9)),
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
            aggregate = json.loads((root / "zipf_usage_sweep" / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
            self.assertTrue(aggregate.get("figures"))
            self.assertFalse((zipf_seed / "figures").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
