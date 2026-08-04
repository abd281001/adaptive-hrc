"""Tests for per-recipe moving-window retention horizons in DecayManager."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace
from src.memory import DecayManager
from src.models import Config, DEFAULT_CONFIG


KEY_A = ("R1", "variant_a")
KEY_A_ALT = ("R1", "variant_b")
KEY_B = ("R2", "variant_a")


def _record_gap(dm: DecayManager, key, gap: int) -> None:
    dm._record_reuse_gap(key, gap, step=gap)


class TestReuseGapWindow(unittest.TestCase):
    def test_cold_start_horizon_yields_to_moving_window_maximum(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, decay_horizon_init=15, decay_horizon_floor=6, decay_reuse_window=3))

        self.assertAlmostEqual(dm.recipe_horizon_for("R1"), 15.0)
        _record_gap(dm, KEY_A, 5)
        self.assertAlmostEqual(dm.recipe_horizon_for("R1"), 6.0)
        for gap in (20, 8, 4):
            _record_gap(dm, KEY_A, gap)
        self.assertEqual(dm.recipe_window_snapshot("R1"), [20, 8, 4])
        self.assertAlmostEqual(dm.recipe_horizon_for("R1"), 20.0)

        _record_gap(dm, KEY_A, 3)
        self.assertEqual(dm.recipe_window_snapshot("R1"), [8, 4, 3])
        self.assertAlmostEqual(dm.recipe_horizon_for("R1"), 8.0)

    def test_recipe_windows_are_independent(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, decay_horizon_init=21, decay_reuse_window=3))

        _record_gap(dm, KEY_A, 6)
        _record_gap(dm, KEY_B, 30)

        self.assertAlmostEqual(dm.horizon_for(KEY_A), 6.0)
        self.assertAlmostEqual(dm.horizon_for(KEY_B), 30.0)

    def test_preferences_of_same_recipe_share_horizon(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, decay_horizon_init=21, decay_reuse_window=3))

        _record_gap(dm, KEY_A, 5)

        self.assertAlmostEqual(dm.horizon_for(KEY_A), 6.0)
        self.assertAlmostEqual(dm.horizon_for(KEY_A_ALT), 6.0)

    def test_short_reuse_gaps_shorten_horizon_after_long_gap_expires(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, decay_horizon_init=21, decay_horizon_floor=6, decay_reuse_window=3))

        for gap in (20, 8, 4):
            _record_gap(dm, KEY_A, gap)
        self.assertAlmostEqual(dm.horizon_for(KEY_A), 20.0)
        _record_gap(dm, KEY_A, 3)
        self.assertAlmostEqual(dm.horizon_for(KEY_A), 8.0)
        _record_gap(dm, KEY_A, 1)
        self.assertAlmostEqual(dm.horizon_for(KEY_A), 6.0)

    def test_gap_larger_than_diagnostic_window_is_stored(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, mwr_window=30))

        _record_gap(dm, KEY_A, 38)

        self.assertEqual(dm.window_snapshot(), [38])
        self.assertEqual(dm.recipe_window_snapshot("R1"), [38])
        self.assertAlmostEqual(dm.horizon_for(KEY_A), 38.0)

    def test_diagnostic_window_length_is_bounded(self):
        cfg = replace(DEFAULT_CONFIG, mwr_window=4, decay_reuse_window=3)
        dm = DecayManager(cfg)
        for i in range(20):
            _record_gap(dm, KEY_A, i + 1)

        self.assertLessEqual(len(dm.window_snapshot()), 4)
        self.assertLessEqual(len(dm.recipe_window_snapshot("R1")), 3)

    def test_zero_gap_ignored(self):
        dm = DecayManager(replace(DEFAULT_CONFIG, decay_reuse_window=3))
        horizon_before = dm.horizon_for(KEY_A)

        _record_gap(dm, KEY_A, 0)

        self.assertEqual(dm.horizon_for(KEY_A), horizon_before)
        self.assertEqual(dm.window_snapshot(), [])
        self.assertEqual(dm.recipe_window_snapshot("R1"), [])

    def test_decay_reuse_window_is_explicit_config(self):
        cfg = Config(decay_reuse_window=2)
        self.assertEqual(cfg.decay_reuse_window, 2)
        dm = DecayManager(cfg)

        for gap in (10, 20, 40):
            _record_gap(dm, KEY_A, gap)

        self.assertEqual(dm.recipe_window_snapshot("R1"), [20, 40])
        self.assertAlmostEqual(dm.horizon_for(KEY_A), 40.0)


if __name__ == "__main__":
    unittest.main()
