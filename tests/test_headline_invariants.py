"""Fast invariants for evaluator-facing, paper-level metrics."""
from __future__ import annotations

import math
import unittest

from src.evaluation import EvalSettings, TaskVariant, _exposure_tags, aggregate_episodes
from src.models import DEFAULT_SETTINGS


class HeadlineInvariantTests(unittest.TestCase):
    def test_default_decay_horizon_matches_retention_spec(self):
        self.assertEqual(DEFAULT_SETTINGS.initial_grace, 50)
        self.assertEqual(DEFAULT_SETTINGS.min_grace, 6)
        self.assertEqual(DEFAULT_SETTINGS.prune_delay, 3)
        self.assertEqual(DEFAULT_SETTINGS.pair_gap_window, 12)
        self.assertEqual(DEFAULT_SETTINGS.parent_weight_samples, 5)
        self.assertAlmostEqual(DEFAULT_SETTINGS.pair_prior_half_life, 3.0)
        self.assertAlmostEqual(DEFAULT_SETTINGS.gap_quantile, 0.90)
        self.assertAlmostEqual(DEFAULT_SETTINGS.gap_iqr_scale, 1.50)
        self.assertEqual(DEFAULT_SETTINGS.recipe_gap_window, 24)
        self.assertEqual(DEFAULT_SETTINGS.global_gap_window, 60)
        self.assertTrue(DEFAULT_SETTINGS.pin_latest)

    def test_evaluator_default_uses_the_published_top_k_setting(self):
        self.assertEqual(EvalSettings().top_k, 3)
        self.assertEqual(EvalSettings().recipe_count, 20)
        self.assertEqual(EvalSettings().schedule.phases, 7)
        self.assertEqual(EvalSettings().schedule.demos, 210)
        self.assertEqual(EvalSettings().schedule.demos % 3, 0)

    def test_aggregate_ignores_nonfinite_metric_values(self):
        metrics = aggregate_episodes([
            {
                "mode": "assist",
                "recipe_steps": float("nan"),
                "hrc_robot_turn_count": float("nan"),
                "hrc_robot_correct_count": float("inf"),
                "hrc_robot_top_k_hit_count": 0,
                "hrc_human_turn_count": float("inf"),
                "hrc_human_correction_count": float("inf"),
            },
            {
                "mode": "assist",
                "recipe_steps": 2,
                "hrc_robot_turn_count": 2,
                "hrc_robot_correct_count": 1,
                "hrc_robot_top_k_hit_count": 2,
                "hrc_human_turn_count": 0,
                "hrc_human_correction_count": 1,
            },
        ])
        self.assertAlmostEqual(metrics["live_top_1"], 0.5)
        self.assertAlmostEqual(metrics["live_top_k"], 1.0)
        self.assertAlmostEqual(metrics["normalized_human_action_load"], 0.5)
        self.assertAlmostEqual(metrics["mean_corrections_per_task"], 0.5)
        self.assertAlmostEqual(metrics["corrections_per_recipe_step"], 0.5)
        self.assertFalse(math.isnan(metrics["live_top_1"]))
        self.assertNotIn("testing_normalized_interaction_cost", metrics)

    def test_empty_aggregate_is_marked_not_run_not_zero_accuracy(self):
        metrics = aggregate_episodes([])
        self.assertEqual(metrics["status"], "not_run")
        self.assertIsNone(metrics["live_top_1"])
        self.assertIsNone(metrics["normalized_human_action_load"])
        self.assertIsNone(metrics["mean_corrections_per_task"])
        self.assertIsNone(metrics["corrections_per_recipe_step"])
        self.assertIsNone(
            metrics["later_action_error_share"]
        )
        self.assertNotIn("human_action_fraction", metrics)
        self.assertNotIn("human_correction_rate", metrics)

    def test_exposure_cell_marks_known_recipe_new_preference(self):
        pair = TaskVariant("recipe", "alternative", ("opaque",))
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
