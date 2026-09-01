import unittest

from src.memory import ReplayMemory
from src.models import Settings


TOMATO_ONION_SOUP = "tomato_onion_soup"
TOMATO_SOUP = "tomato_soup"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class ReplayMemoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(initial_grace=50, prune_delay=3, diagnostic_gap_window=10)
        self.dm = ReplayMemory(self.settings)

    def test_default_grace_horizon_matches_retention_spec(self):
        self.assertEqual(self.dm.default_grace_horizon, 50)
        self.assertEqual(self.dm.min_grace, 6)
        self.assertAlmostEqual(
            self.dm.horizon((TOMATO_ONION_SOUP, BASE_PREF)), 50.0,
        )

    def test_register_new_weight_one(self):
        e = self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b"), now=1, cycle=0)
        self.assertEqual(e.weight, 1.0)
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.active)

    def test_pair_horizon_enforces_floor(self):
        dm = ReplayMemory(Settings(initial_grace=0, min_grace=6))
        self.assertAlmostEqual(
            dm.horizon((TOMATO_ONION_SOUP, BASE_PREF)), 6.0,
        )

    def test_unpinned_variant_has_grace_then_three_step_decay(self):
        dm = ReplayMemory(Settings(
            initial_grace=2,
            min_grace=0,
            prune_delay=3,
            pin_latest=False,
            prune_threshold=1e-9,
        ))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)

        dm.step(1, cycle=0)
        self.assertAlmostEqual(dm.active[key].weight, 1.0)
        dm.step(2, cycle=0)
        self.assertAlmostEqual(dm.active[key].weight, 1.0)
        dm.step(3, cycle=0)
        self.assertAlmostEqual(dm.active[key].weight, 2.0 / 3.0)
        dm.step(4, cycle=0)
        self.assertAlmostEqual(dm.active[key].weight, 1.0 / 3.0)
        pruned = dm.step(5, cycle=0)

        self.assertEqual(pruned, [key])
        self.assertIn(key, dm.pruned)
        self.assertNotIn(key, dm.active)

    def test_latest_key_is_pinned_indefinitely(self):
        dm = ReplayMemory(Settings(initial_grace=0, min_grace=0, prune_delay=1))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        for t in range(1, 20):
            dm.step(t, cycle=0)
        self.assertIn(key, dm.active)
        self.assertNotIn(key, dm.pruned)
        self.assertAlmostEqual(dm.active[key].weight, 1.0)

    def test_old_latest_keeps_its_own_pair_horizon_after_new_latest_registered(self):
        dm = ReplayMemory(Settings(initial_grace=2, min_grace=0, prune_delay=3))
        old_key = (TOMATO_ONION_SOUP, BASE_PREF)
        new_key = (TOMATO_ONION_SOUP, ALT_PREF)
        dm.register(*old_key, ("a",), now=0, cycle=0)
        dm.register(*new_key, ("b",), now=1, cycle=0)

        self.assertNotIn(old_key, dm.latest_keys)
        self.assertIn(new_key, dm.latest_keys)
        self.assertAlmostEqual(dm.horizon(old_key), 2.0)
        self.assertAlmostEqual(dm.horizon(new_key), 2.0)
        dm.step(2, cycle=0)
        self.assertAlmostEqual(dm.active[old_key].weight, 1.0)
        dm.step(3, cycle=0)
        self.assertAlmostEqual(dm.active[old_key].weight, 2.0 / 3.0)
        self.assertAlmostEqual(dm.active[new_key].weight, 1.0)

    def test_reentry_restores_weight_and_records_exact_pair_gap(self):
        dm = ReplayMemory(Settings(
            initial_grace=2,
            min_grace=0,
            prune_delay=1,
            pin_latest=False,
            prune_threshold=1e-9,
        ))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        for t in range(1, 4):
            dm.step(t, cycle=0)
        self.assertIn(key, dm.pruned)

        e = dm.register(*key, ("a",), now=8, cycle=1)

        self.assertEqual(e.weight, 1.0)
        self.assertNotIn(key, dm.pruned)
        self.assertEqual(dm.pair_history(key), [8])
        # One return influences but cannot fully replace the cold-start prior.
        self.assertAlmostEqual(dm.horizon(key), 4.0)
        self.assertEqual(dm.reentry_events[-1], (8, key, 8))
        self.assertEqual(dm.reuse_gap_events[-1], (8, key, 8))

    def test_reuse_horizon_adapts_from_rolling_robust_pair_history(self):
        dm = ReplayMemory(Settings(
            initial_grace=21,
            min_grace=0,
            pair_gap_window=5,
            parent_weight_samples=5,
            pin_latest=False,
        ))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        dm.register(*key, ("a",), now=10, cycle=1)
        self.assertGreater(dm.horizon(key), 10.0)
        self.assertLess(dm.horizon(key), 21.0)
        self.assertEqual(dm.reentry_events, [])
        self.assertEqual(dm.reuse_gap_events, [(10, key, 10)])
        dm.register(*key, ("a",), now=20, cycle=2)
        dm.register(*key, ("a",), now=50, cycle=3)
        dm.register(*key, ("a",), now=80, cycle=4)
        dm.register(*key, ("a",), now=110, cycle=5)
        self.assertEqual(dm.pair_history(key), [10, 10, 30, 30, 30])
        self.assertAlmostEqual(dm.horizon(key), 30.0)

    def test_horizons_are_per_preference_within_recipe(self):
        dm = ReplayMemory(Settings(
            initial_grace=5,
            min_grace=0,
            prune_delay=3,
            pair_gap_window=5,
            parent_weight_samples=5,
            pin_latest=False,
        ))
        key_base = (TOMATO_ONION_SOUP, BASE_PREF)
        key_alt = (TOMATO_ONION_SOUP, ALT_PREF)
        key_other_recipe = (TOMATO_SOUP, BASE_PREF)
        dm.register(*key_base, ("a",), now=0, cycle=0)
        dm.register(*key_other_recipe, ("b",), now=0, cycle=0)
        dm.register(*key_alt, ("c",), now=2, cycle=1)
        for _ in range(5):
            dm._record_reuse_gap(key_base, 1)

        self.assertAlmostEqual(dm.horizon(key_base), 3.0)
        self.assertAlmostEqual(dm.horizon(key_alt), 5.0)
        self.assertAlmostEqual(dm.horizon(key_other_recipe), 5.0)
        dm.step(4, cycle=1)

        self.assertLess(dm.active[key_base].weight, 1.0)
        self.assertAlmostEqual(dm.active[key_alt].weight, 1.0)
        self.assertAlmostEqual(dm.active[key_other_recipe].weight, 1.0)

    def test_latest_protection_can_be_disabled(self):
        dm = ReplayMemory(Settings(
            initial_grace=0,
            min_grace=0,
            prune_delay=3,
            pin_latest=False,
        ))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        self.assertEqual(dm.latest_keys, set())
        dm.step(1, cycle=0)
        self.assertAlmostEqual(dm.active[key].weight, 2.0 / 3.0)

    def test_discard_refuses_latest_pin_by_default(self):
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        self.dm.register(*key, ("a",), now=0, cycle=0)
        with self.assertRaises(RuntimeError):
            self.dm.discard(*key)
        self.dm.discard(*key, allow_latest=True)
        self.assertNotIn(key, self.dm.active)


if __name__ == "__main__":
    unittest.main()
