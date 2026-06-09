import unittest

from src.models import Config
from src.memory import VariantMemory


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"


class VariantMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = VariantMemory(Config())

    def test_register_creates_variant(self):
        v = self.memory.register(TOMATO_ONION_SOUP, ["a", "b", "c"], step=1)
        self.assertEqual(v.recipe_id, TOMATO_ONION_SOUP)
        self.assertEqual(self.memory.latest[TOMATO_ONION_SOUP], v.variant_hash)

    def test_coexistence_two_variants(self):
        v1 = self.memory.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        v2 = self.memory.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.assertNotEqual(v1.variant_hash, v2.variant_hash)
        self.assertEqual(len(self.memory.variants[TOMATO_ONION_SOUP]), 2)
        self.assertEqual(self.memory.latest[TOMATO_ONION_SOUP], v2.variant_hash)

    def test_promote_changes_latest(self):
        v1 = self.memory.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        self.memory.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.memory.promote_latest(TOMATO_ONION_SOUP, v1.variant_hash, step=3)
        self.assertEqual(self.memory.latest[TOMATO_ONION_SOUP], v1.variant_hash)

    def test_library_filters_allowed_keys(self):
        v1 = self.memory.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        self.memory.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        library = self.memory.library(allowed_keys={(TOMATO_ONION_SOUP, v1.variant_hash)})
        self.assertEqual(len(library), 1)
        self.assertEqual(library[0].variant_hash, v1.variant_hash)

    def test_register_never_evicts_known_variants(self):
        memory = VariantMemory(Config(max_variants_per_recipe=1))
        v1 = memory.register(TOMATO_ONION_SOUP, ["a", "b"], step=1)
        v2 = memory.register(TOMATO_ONION_SOUP, ["b", "a"], step=2)
        self.assertIn(v1.variant_hash, memory.variants[TOMATO_ONION_SOUP])
        self.assertIn(v2.variant_hash, memory.variants[TOMATO_ONION_SOUP])
        self.assertEqual(memory.latest[TOMATO_ONION_SOUP], v2.variant_hash)


if __name__ == "__main__":
    unittest.main()
