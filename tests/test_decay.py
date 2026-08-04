import unittest

from src.memory import DecayManager
from src.models import Config


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"
TOMATO_SOUP = "tomato_soup"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class DecayManagerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(decay_horizon_init=15, decay_after_grace_steps=3, decay_reuse_window=3, mwr_window=10)
        self.dm = DecayManager(self.cfg)

    def test_default_grace_horizon_matches_retention_spec(self):
        self.assertEqual(self.dm.default_grace_horizon, 15)
        self.assertEqual(self.dm.decay_horizon_floor, 6)
        self.assertAlmostEqual(self.dm.recipe_horizon_for(TOMATO_ONION_SOUP), 15.0)

    def test_register_new_weight_one(self):
        e = self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b"), now=1, cycle=0)
        self.assertEqual(e.weight, 1.0)
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.active)

    def test_recipe_horizon_enforces_floor(self):
        dm = DecayManager(Config(decay_horizon_init=0, decay_horizon_floor=6))
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 6.0)

    def test_unpinned_variant_has_grace_then_three_step_decay(self):
        dm = DecayManager(Config(
            decay_horizon_init=2,
            decay_horizon_floor=0,
            decay_after_grace_steps=3,
            protect_latest_preference=False,
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
        dm = DecayManager(Config(decay_horizon_init=0, decay_horizon_floor=0, decay_after_grace_steps=1))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        for t in range(1, 20):
            dm.step(t, cycle=0)
        self.assertIn(key, dm.active)
        self.assertNotIn(key, dm.pruned)
        self.assertAlmostEqual(dm.active[key].weight, 1.0)

    def test_old_latest_uses_updated_recipe_horizon_after_new_latest_registered(self):
        dm = DecayManager(Config(decay_horizon_init=2, decay_horizon_floor=0, decay_after_grace_steps=3))
        old_key = (TOMATO_ONION_SOUP, BASE_PREF)
        new_key = (TOMATO_ONION_SOUP, ALT_PREF)
        dm.register(*old_key, ("a",), now=0, cycle=0)
        dm.register(*new_key, ("b",), now=1, cycle=0)

        self.assertNotIn(old_key, dm.latest_keys)
        self.assertIn(new_key, dm.latest_keys)
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 1.0)
        dm.step(2, cycle=0)
        self.assertAlmostEqual(dm.active[old_key].weight, 2.0 / 3.0)
        self.assertAlmostEqual(dm.active[new_key].weight, 1.0)

    def test_reentry_restores_weight_and_updates_that_recipe_horizon(self):
        dm = DecayManager(Config(
            decay_horizon_init=2,
            decay_horizon_floor=0,
            decay_after_grace_steps=1,
            protect_latest_preference=False,
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
        self.assertEqual(dm.recipe_window_snapshot(TOMATO_ONION_SOUP), [8])
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 8.0)
        self.assertEqual(dm.reentry_events[-1], (8, key, 8))
        self.assertEqual(dm.reuse_gap_events[-1], (8, key, 8))

    def test_reuse_horizon_is_moving_max_per_recipe(self):
        dm = DecayManager(Config(decay_horizon_init=21, decay_reuse_window=3, protect_latest_preference=False))
        key = (TOMATO_ONION_SOUP, BASE_PREF)
        dm.register(*key, ("a",), now=0, cycle=0)
        dm.register(*key, ("a",), now=10, cycle=1)
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 10.0)
        self.assertEqual(dm.reentry_events, [])
        self.assertEqual(dm.reuse_gap_events, [(10, key, 10)])
        dm.register(*key, ("a",), now=20, cycle=2)
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 10.0)
        dm.register(*key, ("a",), now=50, cycle=3)
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 30.0)
        dm.register(*key, ("a",), now=80, cycle=4)
        self.assertEqual(dm.recipe_window_snapshot(TOMATO_ONION_SOUP), [10, 30, 30])
        self.assertAlmostEqual(dm.recipe_horizon_for(TOMATO_ONION_SOUP), 30.0)

    def test_horizons_are_per_recipe_not_per_preference(self):
        dm = DecayManager(Config(decay_horizon_init=5, decay_horizon_floor=0, decay_after_grace_steps=3, protect_latest_preference=False))
        key_base = (TOMATO_ONION_SOUP, BASE_PREF)
        key_alt = (TOMATO_ONION_SOUP, ALT_PREF)
        key_other_recipe = (TOMATO_SOUP, BASE_PREF)
        dm.register(*key_base, ("a",), now=0, cycle=0)
        dm.register(*key_other_recipe, ("b",), now=0, cycle=0)
        dm.register(*key_alt, ("c",), now=2, cycle=1)

        self.assertAlmostEqual(dm.horizon_for(key_base), 2.0)
        self.assertAlmostEqual(dm.horizon_for(key_alt), 2.0)
        self.assertAlmostEqual(dm.horizon_for(key_other_recipe), 5.0)
        dm.step(3, cycle=1)

        self.assertLess(dm.active[key_base].weight, 1.0)
        self.assertAlmostEqual(dm.active[key_alt].weight, 1.0)
        self.assertAlmostEqual(dm.active[key_other_recipe].weight, 1.0)

    def test_latest_protection_can_be_disabled(self):
        dm = DecayManager(Config(
            decay_horizon_init=0,
            decay_horizon_floor=0,
            decay_after_grace_steps=3,
            protect_latest_preference=False,
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
