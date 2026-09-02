import json
from pathlib import Path
import tempfile
import unittest

from adaptive_hrc_burrito.evaluation import (
    FULL_ARM,
    load_config,
    run_experiment,
    validate_result,
)


BURRITO_ROOT = Path(__file__).resolve().parents[1]


class BurritoEvaluationTests(unittest.TestCase):
    def test_validation_config_covers_progressive_matrix(self):
        config = load_config(BURRITO_ROOT / "configs" / "validation-v1.json")
        self.assertGreaterEqual(len(config["seeds"]), 2)
        self.assertGreaterEqual(len(config["layouts"]), 2)
        self.assertEqual(set(config["arms"]), {
            "maxent_only",
            FULL_ARM,
            "latent_timing_capacity_matched",
            "latent_timing_trajectory_hybrid",
        })
        self.assertEqual(
            {condition["name"] for condition in config["conditions"]},
            {
                "observation_to_assist",
                "cross_recipe_semantic_transfer",
                "preference_switch",
                "long_gap_recurrence",
            },
        )

    def test_authoritative_ci_config_writes_complete_manifest(self):
        config = BURRITO_ROOT / "configs" / "ci-v1.json"
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(config, output_root=directory)
            validation = validate_result(result, config)
            run_dir = Path(result["run_dir"])
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            episodes = json.loads(
                (run_dir / "episodes.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["episode_count"], 2)
            self.assertFalse(manifest["repositories"]["adaptive_hrc"]["commit"] == "")
            self.assertIn("talents_zsc", manifest["repositories"])
            self.assertIn("overcooked_ai", manifest["repositories"])
            self.assertIn("reward", manifest["features"])
            self.assertEqual(
                manifest["strategy_roles"]["names"],
                [
                    "retrieve", "prepare", "start_cook", "stage",
                    "collect", "assemble", "serve",
                ],
            )
            self.assertIn(
                "learner_retained_payload_bytes",
                manifest["metric_definitions"],
            )
            self.assertIn("steak", manifest["macros"])
            self.assertIn("burrito_1-2_2p", manifest["parking_positions"])
            self.assertEqual(summary["delivery_rate"], 1.0)
            self.assertEqual(summary["failure_count"], 0)
            self.assertTrue(episodes)
            for episode in episodes:
                self.assertGreater(
                    episode["learner_retained_payload_bytes"], 0,
                )
                self.assertGreaterEqual(
                    episode["learner_retained_payload_bytes"],
                    episode["learner_dense_array_bytes"],
                )
                self.assertNotIn("model_storage_values", episode)
                self.assertNotIn("model_storage_values_per_macro", episode)
            self.assertIn(
                "learner_retained_payload_bytes", summary["groups"][0],
            )


if __name__ == "__main__":
    unittest.main()
