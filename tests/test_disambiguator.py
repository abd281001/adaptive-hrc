import unittest

from src.memory import (
    Disambiguator,
    KnownVariant,
    jaccard,
    kendall_tau_distance,
    pref_hash,
)
from src.models import Config


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class JaccardTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(jaccard(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(jaccard(["a"], ["b"]), 0.0)
        self.assertAlmostEqual(jaccard(["a", "b", "c"], ["a", "b", "d"]), 0.5)


class TauTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(kendall_tau_distance(["a", "b", "c"], ["a", "b", "c"]), 0.0)

    def test_reverse(self):
        self.assertAlmostEqual(
            kendall_tau_distance(["a", "b", "c"], ["c", "b", "a"]), 1.0
        )

    def test_repeated_tokens_are_occurrence_sensitive(self):
        d = kendall_tau_distance(["a", "b", "a", "c"], ["a", "a", "b", "c"])
        self.assertGreater(d, 0.0)


class PrefHashOrderPreservingTests(unittest.TestCase):
    def test_raw_hash_distinguishes_orderings(self):
        h1 = pref_hash(["a(x)", "a(y)", "b"])
        h2 = pref_hash(["a(y)", "a(x)", "b"])
        self.assertNotEqual(h1, h2)

    def test_canonicalize_flag_no_longer_merges_same_type_swap(self):
        # Same action-type swaps can encode preferences, so the compatibility
        # flag must not collapse them.
        h1 = pref_hash(["a(x)", "a(y)", "b"], canonicalize=True)
        h2 = pref_hash(["a(y)", "a(x)", "b"], canonicalize=True)
        self.assertNotEqual(h1, h2)

    def test_order_preserving_hash_preserves_different_type_order(self):
        h1 = pref_hash(["a", "b", "c"], canonicalize=True)
        h2 = pref_hash(["b", "a", "c"], canonicalize=True)
        self.assertNotEqual(h1, h2)


class DisambiguatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(jaccard_threshold=0.6, ordering_tolerance=0.05)
        self.d = Disambiguator(self.cfg)

    def test_empty_library_is_new(self):
        c = self.d.classify(["a", "b"], [])
        self.assertEqual(c.kind, "new_recipe")

    def test_identical_is_known(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, pref_hash(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["a", "b", "c"], lib)
        self.assertEqual(c.kind, "known")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_same_set_different_order_is_pref_shift(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, pref_hash(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["c", "b", "a"], lib)
        self.assertEqual(c.kind, "preference_shift")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_disjoint_is_new(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, pref_hash(["a", "b"]), ("a", "b"))]
        c = self.d.classify(["x", "y", "z"], lib)
        self.assertEqual(c.kind, "new_recipe")

    def test_partial_scoring_prefers_matching_prefix(self):
        v1 = KnownVariant(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b", "c", "d"))
        v2 = KnownVariant(TOMATO_ONION_SOUP, ALT_PREF, ("a", "c", "b", "d"))
        ranked = self.d.score_partial(["a", "b"], [v1, v2])
        self.assertEqual(ranked[0][0].pref_hash, BASE_PREF)


if __name__ == "__main__":
    unittest.main()
