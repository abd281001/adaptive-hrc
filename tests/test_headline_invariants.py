"""Fast invariants for evaluator-facing, paper-level metrics."""
from __future__ import annotations

import math
import unittest

from src.evaluation import EvaluationConfig, RecipePreferencePair, _exposure_tags, aggregate_episode_metrics
from src.models import DEFAULT_CONFIG


class HeadlineInvariantTests(unittest.TestCase):
    def test_default_decay_horizon_matches_retention_spec(self):
        self.assertEqual(DEFAULT_CONFIG.decay_horizon_init, 15)
        self.assertEqual(DEFAULT_CONFIG.decay_horizon_floor, 6)
        self.assertEqual(DEFAULT_CONFIG.decay_after_grace_steps, 3)
        self.assertEqual(DEFAULT_CONFIG.decay_reuse_window, 3)

    def test_evaluator_default_uses_the_published_topk_setting(self):
        self.assertEqual(EvaluationConfig().topk, 3)
        self.assertEqual(EvaluationConfig().n_recipes, 15)
        self.assertEqual(EvaluationConfig().ladder_rungs, 9)

    def test_aggregate_ignores_nonfinite_metric_values(self):
        metrics = aggregate_episode_metrics([
            {
                "hrc_robot_turn_count": float("nan"),
                "hrc_robot_correct_count": float("inf"),
                "hrc_robot_topk_hit_count": 0,
                "testing_total_action_time": float("nan"),
                "testing_human_only_action_time": float("inf"),
            },
            {
                "hrc_robot_turn_count": 2,
                "hrc_robot_correct_count": 1,
                "hrc_robot_topk_hit_count": 2,
                "testing_total_action_time": 3.0,
                "testing_human_only_action_time": 2.0,
            },
        ])
        self.assertAlmostEqual(metrics["live_top1"], 0.5)
        self.assertAlmostEqual(metrics["live_topk"], 1.0)
        self.assertAlmostEqual(metrics["testing_normalized_interaction_cost"], 1.5)
        self.assertFalse(math.isnan(metrics["live_top1"]))

    def test_empty_aggregate_is_marked_not_run_not_zero_accuracy(self):
        metrics = aggregate_episode_metrics([])
        self.assertEqual(metrics["status"], "not_run")
        self.assertIsNone(metrics["live_top1"])
        self.assertIsNone(metrics["human_correction_rate"])

    def test_exposure_cell_marks_known_recipe_new_preference(self):
        pair = RecipePreferencePair("recipe", "alternative", ("opaque",))
        tags = _exposure_tags(
            pair,
            observed_recipes={"recipe"},
            observed_preferences=set(),
            observed_pairs=set(),
            preferences_by_recipe={"recipe": set()},
            axis_values_by_recipe={"recipe": set()},
        )
        self.assertEqual(tags["four_cell_before"], "seen_recipe_new_preference")
        self.assertEqual(tags["transfer_cell_before"], "seen_recipe_new_preference")


if __name__ == "__main__":
    unittest.main()
