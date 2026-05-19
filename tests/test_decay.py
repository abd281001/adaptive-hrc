import unittest

from src.memory import DecayManager
from src.models import Config


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"
TOMATO_SOUP = "tomato_soup"
MUSHROOM_SOUP = "mushroom_soup"
SEASONED_MIXTURE_SOUP = "seasoned_mixture_soup"
GRILLED_STEAK = "grilled_steak"
BURGER = "burger"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class DecayManagerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(decay_init=0.1, mwr_window=10)
        self.dm = DecayManager(self.cfg)

    def test_register_new_weight_one(self):
        e = self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b"), now=1, cycle=0)
        self.assertEqual(e.weight, 1.0)
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.active)

    def test_linear_decay_until_prune(self):
        self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        self.dm.register(TOMATO_ONION_SOUP, ALT_PREF, ("b",), now=0, cycle=0)
        pruned = []
        for t in range(1, 30):
            pruned.extend(self.dm.step(t, cycle=0))
            if pruned:
                break
        self.assertTrue(pruned)
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.pruned)

    def test_reentry_restores_weight_to_one(self):
        self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        self.dm.register(TOMATO_ONION_SOUP, ALT_PREF, ("b",), now=0, cycle=0)
        for t in range(1, 20):
            self.dm.step(t, cycle=0)
            if not self.dm.active:
                break
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.pruned)
        e = self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=30, cycle=1)
        self.assertEqual(e.weight, 1.0)
        self.assertNotIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.pruned)

    def test_active_reuse_drives_base_rate(self):
        """Every same-recipe_pref reuse, including active reuse, drives base_rate."""
        dm = DecayManager(Config(
            decay_init=0.2,
            mwr_window=1,
            size_rescale_exponent=0.0,
        ))
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=5, cycle=0)
        rate_before = dm.base_rate
        e = dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=17, cycle=1)

        self.assertEqual(e.weight, 1.0)
        self.assertEqual(e.last_seen_step, 17)
        self.assertEqual(dm.window_snapshot(), [12])
        self.assertAlmostEqual(dm.base_rate, 1.0 / 12, places=9)
        self.assertLess(dm.base_rate, rate_before)
        # Legacy reentry-events log records all same-variant reuse gaps for metrics.
        self.assertEqual(dm.reentry_events[-1], (17, (TOMATO_ONION_SOUP, BASE_PREF), 12))

    def test_decay_rate_propagates_to_existing_entries(self):
        """When base_rate changes, every existing entry's deduction at the next
        step() reflects the NEW global_rate — there is no per-entry snapshot."""
        dm = DecayManager(Config(
            decay_init=0.1,
            decay_horizon_init=2,
            mwr_window=1,
            protect_latest_preference=False,
            size_rescale_exponent=0.0,
        ))
        # Register at the initial rate (decay_init=0.1).
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        e = dm.active[(TOMATO_ONION_SOUP, BASE_PREF)]
        self.assertAlmostEqual(e.weight, 1.0, places=9)
        # Force the global_rate down to 0.05 by recording a long re-entry gap.
        dm._adapt_base_from_reuse(20)  # backwards-compat shim: feeds reuse window
        dm._rescale_rate_by_size(step=1)
        self.assertAlmostEqual(dm.global_rate, 0.05, places=9)
        # Stepping deducts the CURRENT global_rate (0.05), not a snapshot at register time.
        dm.step(1, cycle=0)
        self.assertAlmostEqual(dm.active[(TOMATO_ONION_SOUP, BASE_PREF)].weight, 0.95, places=9)
        # Now move global_rate back up via a short gap. Existing entry must adapt.
        dm._adapt_base_from_reuse(2)
        dm._rescale_rate_by_size(step=2)
        self.assertAlmostEqual(dm.global_rate, 0.5, places=9)
        dm.step(2, cycle=0)
        self.assertAlmostEqual(dm.active[(TOMATO_ONION_SOUP, BASE_PREF)].weight, 0.45, places=9)

    def test_mwr_window_combines_active_reuse_and_pruned_reentry(self):
        """Active repeats and pruned restores feed the same cadence window."""
        dm = DecayManager(Config(
            decay_init=0.2,
            mwr_window=10,
            size_rescale_exponent=0.0,
            protect_latest_preference=False,
        ))
        # In-active reuse: register, then re-register before pruning.
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=3, cycle=0)
        self.assertEqual(dm.window_snapshot(), [3])
        # Now allow pruning then re-enter from the pruned dict.
        for t in range(4, 30):
            dm.step(t, cycle=0)
            if (TOMATO_ONION_SOUP, BASE_PREF) in dm.pruned:
                break
        last_seen = dm.pruned[(TOMATO_ONION_SOUP, BASE_PREF)].last_seen_step
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=50, cycle=1)
        self.assertEqual(dm.window_snapshot(), [3, 50 - last_seen])

    def test_reuse_gap_adapts_rate(self):
        dm = DecayManager(Config(
            decay_init=0.2,
            mwr_window=10,
            protect_latest_preference=False,
            size_rescale_exponent=0.0,
        ))
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        # prune it
        for t in range(1, 20):
            dm.step(t, cycle=0)
            if (TOMATO_ONION_SOUP, BASE_PREF) in dm.pruned:
                break
        rate_before = dm.global_rate
        # re-enter after a longer gap; rate tracks 1/(gap since previous demo),
        # even when the gap magnitude exceeds mwr_window.
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=40, cycle=1)
        rate_after = dm.global_rate
        self.assertNotAlmostEqual(rate_before, rate_after, places=6)
        # monotone direction: gap is larger than prior lifespan → rate drops
        self.assertLess(rate_after, rate_before)
        self.assertEqual(dm.window_snapshot(), [40])
        self.assertEqual(dm.reentry_events[-1], (40, (TOMATO_ONION_SOUP, BASE_PREF), 40))

    def test_latest_key_is_pinned(self):
        self.dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        for t in range(1, 20):
            self.dm.step(t, cycle=0)
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.active)
        self.assertNotIn((TOMATO_ONION_SOUP, BASE_PREF), self.dm.pruned)
        self.assertAlmostEqual(self.dm.active[(TOMATO_ONION_SOUP, BASE_PREF)].weight, 1.0)

    def test_latest_protection_can_be_disabled(self):
        dm = DecayManager(Config(
            decay_init=0.1,
            protect_latest_preference=False,
        ))
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        self.assertEqual(dm.latest_keys, set())
        dm.step(1, cycle=0)
        self.assertAlmostEqual(dm.active[(TOMATO_ONION_SOUP, BASE_PREF)].weight, 0.9)

    def test_rate_has_initial_horizon_floor(self):
        dm = DecayManager(Config(decay_init=0.1, mwr_window=1, size_rescale_exponent=0.0))
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        # Long gaps are bounded by the explicit deployment floor.
        dm._adapt_rate_from_reuse(10_000, step=10_000)
        dm._rescale_rate_by_size(step=10_000)
        self.assertAlmostEqual(dm.global_rate, dm.cfg.min_global_rate, places=9)
        # Short gaps cannot make retention shorter than H_init.
        dm._adapt_rate_from_reuse(1, step=10_001)
        dm._rescale_rate_by_size(step=10_001)
        self.assertAlmostEqual(dm.global_rate, 0.1, places=9)

    def test_size_rescaling_slows_with_more_recipes(self):
        dm = DecayManager(Config(decay_init=0.1))
        dm.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        rate1 = dm.global_rate
        for rid in (TOMATO_SOUP, MUSHROOM_SOUP, SEASONED_MIXTURE_SOUP, GRILLED_STEAK, BURGER):
            dm.register(rid, BASE_PREF, ("a",), now=0, cycle=0)
        self.assertLessEqual(dm.global_rate, rate1 + 1e-9)

    def test_size_rescaling_uses_recent_max_after_recipes_are_removed(self):
        dm = DecayManager(Config(decay_init=0.1))
        for rid in (TOMATO_ONION_SOUP, TOMATO_SOUP, MUSHROOM_SOUP, SEASONED_MIXTURE_SOUP):
            dm.register(rid, BASE_PREF, ("a",), now=0, cycle=0)
        crowded_rate = dm.global_rate
        for key in [(TOMATO_SOUP, BASE_PREF), (MUSHROOM_SOUP, BASE_PREF), (SEASONED_MIXTURE_SOUP, BASE_PREF)]:
            del dm.active[key]
        dm._rescale_rate_by_size(step=1)
        self.assertAlmostEqual(dm.global_rate, crowded_rate, places=9)

    def test_pruning_boost_uses_recent_largest_active_count(self):
        cfg = Config(
            decay_horizon_init=50,
            mwr_window=30,
            size_rescale_exponent=0.5,
            min_global_rate=0.0,
            max_decay_horizon=0,
        )
        dm = DecayManager(cfg)
        recipe_ids = (
            TOMATO_ONION_SOUP,
            TOMATO_SOUP,
            MUSHROOM_SOUP,
            SEASONED_MIXTURE_SOUP,
            GRILLED_STEAK,
            BURGER,
        )
        for rid in recipe_ids:
            dm.register(rid, BASE_PREF, ("a",), now=0, cycle=0)
        crowded_rate = dm.global_rate
        self.assertAlmostEqual(crowded_rate, dm.base_rate / (len(recipe_ids) ** cfg.size_rescale_exponent), places=9)
        self.assertEqual(max(dm.active_count_window_snapshot()), len(recipe_ids))

        for rid in recipe_ids[1:]:
            del dm.active[(rid, BASE_PREF)]
        dm._rescale_rate_by_size(step=1)

        self.assertAlmostEqual(dm.global_rate, crowded_rate, places=9)

    def test_size_rescale_exponent_knob(self):
        """exponent=0 disables size rescale; exponent=1 gives strict 1/d scaling."""
        # exponent=0.0 — rate independent of active-set size
        dm0 = DecayManager(Config(decay_init=0.1, size_rescale_exponent=0.0))
        for rid in (TOMATO_ONION_SOUP, TOMATO_SOUP, MUSHROOM_SOUP, SEASONED_MIXTURE_SOUP, GRILLED_STEAK):
            dm0.register(rid, BASE_PREF, ("a",), now=0, cycle=0)
        # effective rate == base_rate
        self.assertAlmostEqual(dm0.global_rate, dm0.base_rate, places=9)

        # exponent=1.0 — while current active count is the recent max, the
        # size term is 1/d relative to a one-entry baseline.
        dm1 = DecayManager(Config(
            decay_init=0.1, size_rescale_exponent=1.0,
        ))
        dm1.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        rate_at_1 = dm1.global_rate
        for rid in (TOMATO_SOUP, MUSHROOM_SOUP, SEASONED_MIXTURE_SOUP, GRILLED_STEAK):
            dm1.register(rid, BASE_PREF, ("a",), now=0, cycle=0)
        # After 5 entries total, rate ≈ base_rate / 5
        self.assertAlmostEqual(dm1.global_rate, rate_at_1 / 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
