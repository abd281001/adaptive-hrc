"""Tests for the max-over-window reuse-gap scheme in DecayManager."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace
from src.memory import DecayManager
from src.models import Config, DEFAULT_CONFIG


class TestMaxOverWindow(unittest.TestCase):

    def test_base_rate_tracks_max_over_window(self):
        """base_rate should track 1 / max(window) as gaps enter and evict."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=5, decay_init=0.5)
        dm = DecayManager(cfg)

        for g in (5, 20, 8, 40, 15):
            dm._adapt_base_from_reuse(g)
        # window is full: [5, 20, 8, 40, 15]; max=40
        self.assertAlmostEqual(dm.base_rate, 1.0 / 40, places=9)

        # push five gap=10 entries; evicts 5, 20, 8, 40, 15 one by one
        dm._adapt_base_from_reuse(10)   # window [20, 8, 40, 15, 10]; max=40
        self.assertAlmostEqual(dm.base_rate, 1.0 / 40, places=9)
        dm._adapt_base_from_reuse(10)   # [8, 40, 15, 10, 10]; max=40
        self.assertAlmostEqual(dm.base_rate, 1.0 / 40, places=9)
        dm._adapt_base_from_reuse(10)   # [40, 15, 10, 10, 10]; max=40
        self.assertAlmostEqual(dm.base_rate, 1.0 / 40, places=9)
        dm._adapt_base_from_reuse(10)   # [15, 10, 10, 10, 10]; max=15
        self.assertAlmostEqual(dm.base_rate, 1.0 / 15, places=9)
        dm._adapt_base_from_reuse(10)   # [10, 10, 10, 10, 10]; max=10
        self.assertAlmostEqual(dm.base_rate, 1.0 / 10, places=9)

    def test_window_eviction_removes_old_influence(self):
        """Filling with gap=100 pins rate low; flooding with gap=2 recovers it
        once the long entries evict."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=3, decay_init=0.5)
        dm = DecayManager(cfg)

        for _ in range(3):
            dm._adapt_base_from_reuse(100)
        self.assertAlmostEqual(dm.base_rate, 1.0 / 100, places=9)

        for _ in range(3):
            dm._adapt_base_from_reuse(2)
        self.assertAlmostEqual(dm.base_rate, 1.0 / 10, places=9)

    def test_window_size_1_tracks_instant_gap(self):
        """mwr_window=1 tracks the latest gap subject to the initial horizon floor."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=1, decay_init=0.2)
        dm = DecayManager(cfg)

        dm._adapt_base_from_reuse(5)
        self.assertAlmostEqual(dm.base_rate, 1.0 / 10, places=9)

        dm._adapt_base_from_reuse(40)
        self.assertAlmostEqual(dm.base_rate, 1.0 / 40, places=9)

    def test_gap_larger_than_window_is_stored(self):
        """mwr_window limits stored samples, not the gap magnitude."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=30, decay_init=0.1)
        dm = DecayManager(cfg)

        dm._adapt_base_from_reuse(38)
        self.assertEqual(dm.window_snapshot(), [38])
        self.assertAlmostEqual(dm.base_rate, 1.0 / 38, places=9)

    def test_short_reuse_gap_respects_initial_horizon(self):
        """Frequent repeats do not shorten retention below H_init=10."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=5, decay_init=0.1)
        dm = DecayManager(cfg)

        dm._adapt_base_from_reuse(2)
        self.assertAlmostEqual(dm.base_rate, 0.1, places=9)
        self.assertEqual(dm.base_rate, 1.0 / cfg.decay_horizon_init)

    def test_window_snapshot_length_bounded(self):
        """Window should never exceed mwr_window entries."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=4)
        dm = DecayManager(cfg)
        for i in range(20):
            dm._adapt_base_from_reuse(i + 1)
        snap = dm.window_snapshot()
        self.assertLessEqual(len(snap), 4)

    def test_zero_gap_ignored(self):
        """Gap of 0 should be silently ignored."""
        cfg = replace(DEFAULT_CONFIG, mwr_window=5, decay_init=0.3)
        dm = DecayManager(cfg)
        rate_before = dm.base_rate
        dm._adapt_base_from_reuse(0)
        self.assertEqual(dm.base_rate, rate_before)
        self.assertEqual(dm.window_snapshot(), [])

    def test_mwr_window_is_explicit_config(self):
        """MWR window size is configured directly."""
        from src.models import Config
        cfg = Config(decay_init=0.1, mwr_window=42)
        self.assertEqual(cfg.mwr_window, 42)

    def test_mwr_window_per_scenario_override(self):
        """A smaller window recovers faster after a pinning long gap evicts."""
        cfg_fast = replace(DEFAULT_CONFIG, mwr_window=3, decay_init=0.5)
        cfg_slow = replace(DEFAULT_CONFIG, mwr_window=30, decay_init=0.5)
        dm_fast = DecayManager(cfg_fast)
        dm_slow = DecayManager(cfg_slow)

        # Fill both with gap=100, pinning base_rate low.
        for _ in range(3):
            dm_fast._adapt_base_from_reuse(100)
        for _ in range(30):
            dm_slow._adapt_base_from_reuse(100)

        # Feed 5 short gaps of 5. The fast window (size=3) evicts all 100s
        # within 3 samples and should now track 1/5; the slow window still has
        # 100s dominating the max.
        for _ in range(5):
            dm_fast._adapt_base_from_reuse(5)
            dm_slow._adapt_base_from_reuse(5)

        self.assertGreater(
            dm_fast.base_rate,
            dm_slow.base_rate,
            "Smaller window should have recovered faster once long gaps evicted",
        )


if __name__ == "__main__":
    unittest.main()
