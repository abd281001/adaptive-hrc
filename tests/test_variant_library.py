import unittest

from src.models import Settings
from src.memory import VariantLibrary


TOMATO_ONION_SOUP = "tomato_onion_soup"


class VariantLibraryTests(unittest.TestCase):
    def setUp(self):
        self.library = VariantLibrary()

    def test_register_creates_variant(self):
        v = self.library.register(TOMATO_ONION_SOUP, ["a", "b", "c"], step=1)
        self.assertEqual(v.recipe_id, TOMATO_ONION_SOUP)
        self.assertEqual(self.library.latest[TOMATO_ONION_SOUP], v.variant_id)

    def test_coexistence_two_variants(self):
        first = self.library.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        second = self.library.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.assertNotEqual(first.variant_id, second.variant_id)
        self.assertEqual(len(self.library.variants[TOMATO_ONION_SOUP]), 2)
        self.assertEqual(self.library.latest[TOMATO_ONION_SOUP], second.variant_id)

    def test_promote_changes_latest(self):
        first = self.library.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        self.library.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.library.promote_latest(TOMATO_ONION_SOUP, first.variant_id, step=3)
        self.assertEqual(self.library.latest[TOMATO_ONION_SOUP], first.variant_id)

    def test_library_filters_allowed_keys(self):
        first = self.library.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        self.library.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        library = self.library.known_variants(allowed_keys={(TOMATO_ONION_SOUP, first.variant_id)})
        self.assertEqual(len(library), 1)
        self.assertEqual(library[0].variant_id, first.variant_id)

    def test_register_never_evicts_known_variants(self):
        memory = VariantLibrary()
        first = memory.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        second = memory.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.assertIn(first.variant_id, memory.variants[TOMATO_ONION_SOUP])
        self.assertIn(second.variant_id, memory.variants[TOMATO_ONION_SOUP])
        self.assertEqual(memory.latest[TOMATO_ONION_SOUP], second.variant_id)


if __name__ == "__main__":
    unittest.main()
