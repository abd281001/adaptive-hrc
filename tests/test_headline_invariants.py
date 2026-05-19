"""Fast invariants for paper-facing headline metrics.

These tests protect against silent regressions where a figure still renders but
the metric support is structurally invalid.
"""
from __future__ import annotations

import math
import unittest

from src.evaluations import (
    LiveEpisodeRecord,
    LiveStepRecord,
    bwt_fwt_checkpoints,
    four_cell_key,
    mean_ci95,
    pooled_calibration_from_episodes,
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

    def test_bwt_fwt_requires_nonempty_holdout_support(self):
        phase_a = {
            "train": {"top1": 0.8},
            "heldout": {"top1": 0.4},
        }
        final = {
            "train": {"top1": 0.7},
            "heldout": {"top1": 0.6},
        }
        out = bwt_fwt_checkpoints(phase_a, final, phase_a_seen_labels=["train"])
        self.assertEqual(out["n_fwt_tasks"], 1)
        self.assertFalse(math.isnan(out["fwt"]))

    def test_pooled_calibration_uses_quantile_support_not_episode_mean(self):
        steps = [
            LiveStepRecord(
                step=i,
                predicted="a",
                actual="a",
                correct_top1=(i % 2 == 0),
                correct_top3=True,
                topk=("a",),
                inferred_recipe="R",
                inferred_pref="P",
                posterior_confidence=i / 99,
            )
            for i in range(100)
        ]
        record = LiveEpisodeRecord(
            pair_label="R/P",
            recipe_id="R",
            preference_name="P",
            memory_state="active_memory",
            mode="online",
            steps=tuple(steps),
            live_top1=0.5,
            live_top3=1.0,
            n=len(steps),
            first_mismatch_step=1,
        )
        out = pooled_calibration_from_episodes([record])
        self.assertEqual(out["n_pooled"], 100)
        self.assertGreaterEqual(out["n_bins"], 3)
        self.assertLessEqual(out["n_bins"], 15)

    def test_cross_product_offdiagonal_cell_is_seen_recipe_unseen_pref(self):
        self.assertEqual(four_cell_key(seen_recipe=True, seen_preference=False), "seen_unseen")


if __name__ == "__main__":
    unittest.main()
