"""Tests for robust exact-pair retention horizons in ReplayMemory."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace
from src.memory import ReplayMemory
from src.models import Settings, DEFAULT_SETTINGS


KEY_A = ("R1", "variant_a")
KEY_A_ALT = ("R1", "variant_b")
KEY_B = ("R2", "variant_a")


def _record_gap(dm: ReplayMemory, key, gap: int) -> None:
    dm._record_reuse_gap(key, gap, step=gap)


def _robust_config(**overrides):
    values = {
        "initial_grace": 15,
        "min_grace": 0,
        "pair_gap_window": 5,
        "parent_weight_samples": 5,
        "pair_prior_half_life": 3.0,
        "gap_quantile": 0.90,
        "gap_iqr_scale": 1.50,
    }
    values.update(overrides)
    return replace(DEFAULT_SETTINGS, **values)


class TestReuseGapWindow(unittest.TestCase):
    def test_sparse_pair_evidence_is_partially_pooled_until_sufficient(self):
        dm = ReplayMemory(_robust_config(min_grace=6))

        sparse_horizons = []
        for gap in (4, 5, 4, 5):
            _record_gap(dm, KEY_A, gap)
            sparse_horizons.append(dm.horizon(KEY_A))

        self.assertTrue(all(6.0 < horizon < 15.0 for horizon in sparse_horizons))
        self.assertEqual(sparse_horizons, sorted(sparse_horizons, reverse=True))

        _record_gap(dm, KEY_A, 4)
        self.assertEqual(dm.pair_history(KEY_A), [4, 5, 4, 5, 4])
        self.assertAlmostEqual(dm.horizon(KEY_A), 9.0)

    def test_three_downward_samples_halve_the_error_to_the_pair_target(self):
        dm = ReplayMemory(_robust_config())
        for _ in range(3):
            _record_gap(dm, KEY_A, 5)

        diagnostics = dm.horizon_stats(KEY_A)

        self.assertAlmostEqual(diagnostics["pair_evidence_weight"], 0.5)
        self.assertEqual(diagnostics["pair_pooling_mode"], "exponential_downward")
        self.assertAlmostEqual(diagnostics["pair_robust_upper_demos"], 5.0)
        self.assertAlmostEqual(diagnostics["recipe_prior_horizon_demos"], 15.0)
        self.assertAlmostEqual(dm.horizon(KEY_A), 10.0)

    def test_one_long_gap_cannot_set_the_pair_horizon(self):
        dm = ReplayMemory(_robust_config())

        for gap in (5, 5, 5, 5, 50):
            _record_gap(dm, KEY_A, gap)

        self.assertAlmostEqual(dm.horizon(KEY_A), 9.0)

    def test_repeated_long_gaps_adapt_horizon_upward(self):
        dm = ReplayMemory(_robust_config())

        for gap in (5, 5, 30, 30, 30):
            _record_gap(dm, KEY_A, gap)

        self.assertAlmostEqual(dm.horizon(KEY_A), 30.0)

    def test_old_behavior_ages_out_and_horizon_adapts_downward(self):
        dm = ReplayMemory(_robust_config())
        for gap in (5, 5, 30, 30, 30):
            _record_gap(dm, KEY_A, gap)
        self.assertAlmostEqual(dm.horizon(KEY_A), 30.0)

        for _ in range(5):
            _record_gap(dm, KEY_A, 4)

        self.assertEqual(dm.pair_history(KEY_A), [4, 4, 4, 4, 4])
        self.assertAlmostEqual(dm.horizon(KEY_A), 8.0)

    def test_well_observed_preferences_of_same_recipe_have_independent_horizons(self):
        dm = ReplayMemory(_robust_config())
        for gap in (4, 5, 4, 5, 4):
            _record_gap(dm, KEY_A, gap)
        for gap in (18, 20, 22, 20, 19):
            _record_gap(dm, KEY_A_ALT, gap)

        self.assertAlmostEqual(dm.horizon(KEY_A), 11.0)
        self.assertAlmostEqual(dm.horizon(KEY_A_ALT), 22.0)
        self.assertEqual(dm.pair_history(KEY_B), [])
        # Sparse pairs borrow the robust user-level recurrence prior.
        self.assertGreater(dm.horizon(KEY_B), 15.0)
        self.assertLessEqual(dm.horizon(KEY_B), 22.0)

    def test_sparse_pair_borrows_long_recipe_recurrence_without_identity_leakage(self):
        dm = ReplayMemory(_robust_config())
        for gap in (24, 28, 30, 27, 29):
            _record_gap(dm, KEY_A, gap)

        diagnostics = dm.horizon_stats(KEY_A_ALT)

        self.assertEqual(diagnostics["pair_gap_samples"], 0)
        self.assertEqual(diagnostics["recipe_prior_gap_samples"], 5)
        self.assertGreaterEqual(dm.horizon(KEY_A_ALT), 29.0)

    def test_one_sparse_spike_is_bounded_and_then_ages_out(self):
        dm = ReplayMemory(_robust_config())
        _record_gap(dm, KEY_A, 100)
        one_spike_horizon = dm.horizon(KEY_A)

        self.assertGreater(one_spike_horizon, 15.0)
        self.assertLess(one_spike_horizon, 100.0)

        for _ in range(4):
            _record_gap(dm, KEY_A, 5)
        self.assertAlmostEqual(dm.horizon(KEY_A), 9.0)

    def test_register_measures_recurrence_of_exact_pair_not_recipe(self):
        dm = ReplayMemory(_robust_config())
        dm.register(*KEY_A, ("a",), now=0, cycle=0)
        dm.register(*KEY_A_ALT, ("b",), now=2, cycle=0)
        dm.register(*KEY_A, ("a",), now=10, cycle=1)

        self.assertEqual(dm.pair_history(KEY_A), [10])
        self.assertEqual(dm.pair_history(KEY_A_ALT), [])
        self.assertEqual(dm.reuse_gap_events, [(10, KEY_A, 10)])

    def test_diagnostic_and_pair_windows_are_independently_bounded(self):
        dm = ReplayMemory(_robust_config(diagnostic_gap_window=4, pair_gap_window=5))
        for gap in range(1, 21):
            _record_gap(dm, KEY_A, gap)

        self.assertEqual(dm.gap_history(), [17, 18, 19, 20])
        self.assertEqual(dm.pair_history(KEY_A), [16, 17, 18, 19, 20])

    def test_zero_gap_is_ignored(self):
        dm = ReplayMemory(_robust_config())
        horizon_before = dm.horizon(KEY_A)

        _record_gap(dm, KEY_A, 0)

        self.assertEqual(dm.horizon(KEY_A), horizon_before)
        self.assertEqual(dm.gap_history(), [])
        self.assertEqual(dm.pair_history(KEY_A), [])

    def test_pair_estimator_parameters_are_explicit_config(self):
        settings = Settings(
            pair_gap_window=7,
            parent_weight_samples=4,
            pair_prior_half_life=4.0,
            gap_quantile=0.8,
            gap_iqr_scale=2.0,
            recipe_gap_window=19,
            global_gap_window=41,
        )
        dm = ReplayMemory(settings)

        self.assertEqual(dm.reuse_window, 7)
        self.assertEqual(dm.reuse_min_samples, 4)
        self.assertAlmostEqual(dm.pair_downward_half_life, 4.0)
        self.assertAlmostEqual(dm.reuse_quantile, 0.8)
        self.assertAlmostEqual(dm.reuse_iqr_multiplier, 2.0)
        self.assertEqual(dm.recipe_reuse_window, 19)
        self.assertEqual(dm.global_reuse_window, 41)

    def test_downward_half_life_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "pair_prior_half_life"):
            Settings(pair_prior_half_life=0.0)


if __name__ == "__main__":
    unittest.main()
