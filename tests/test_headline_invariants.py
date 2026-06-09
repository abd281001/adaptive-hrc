"""Fast invariants for paper-facing headline metrics.

These tests protect against silent regressions where headline metric support is
structurally invalid.
"""
from __future__ import annotations

import math
import unittest

from src.experiments import (
    _four_cell,
    bwt_zero_shot_checkpoints,
    calibration_curve_from_steps,
    mean_ci95,
)
from src.models import DEFAULT_CONFIG


class HeadlineInvariantTests(unittest.TestCase):
    def test_default_decay_horizon_is_ten_sessions(self):
        self.assertEqual(DEFAULT_CONFIG.decay_horizon_init, 10)
        self.assertAlmostEqual(DEFAULT_CONFIG.min_global_rate, 0.02)

    def test_mean_ci95_filters_nonfinite_values(self):
        mean, ci = mean_ci95([float("nan"), float("inf"), 1.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertGreater(ci, 0.0)
        empty_mean, empty_ci = mean_ci95([float("nan"), float("-inf")])
        self.assertTrue(math.isnan(empty_mean))
        self.assertEqual(empty_ci, 0.0)

    def test_bwt_zero_shot_requires_nonempty_holdout_support(self):
        phase_a = {
            "train": {"top1": 0.8},
            "heldout": {"top1": 0.4},
        }
        final = {
            "train": {"top1": 0.7},
            "heldout": {"top1": 0.6},
        }
        out = bwt_zero_shot_checkpoints(phase_a, final, phase_a_seen_labels=["train"])
        self.assertEqual(out["n_zero_shot_tasks"], 1)
        self.assertFalse(math.isnan(out["zero_shot_transfer"]))

    def test_calibration_curve_uses_step_confidence_support(self):
        rows = [
            {"policy_diagnostics": {"policy_confidence": i / 99}, "correct_top1": i % 2 == 0}
            for i in range(100)
        ]
        out = calibration_curve_from_steps(rows, n_bins=10)
        self.assertEqual(sum(int(row["n_steps"]) for row in out), 100)
        self.assertEqual(len(out), 10)

    def test_cross_product_offdiagonal_cell_is_known_recipe_new_preference(self):
        self.assertEqual(_four_cell(seen_recipe=True, seen_preference=False), "seen_recipe_new_preference")


if __name__ == "__main__":
    unittest.main()
