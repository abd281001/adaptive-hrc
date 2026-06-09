"""Phase 2: tests for the recipe prototype learner."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.posterior import RecipePrototypeLearner


class PrototypeBasicTests(unittest.TestCase):
    def test_update_grows_action_set(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b", "c"])
        proto = learner.get("R1")
        self.assertEqual(proto.action_set, frozenset({"a", "b", "c"}))
        self.assertEqual(proto.n_demos, 1)
        learner.update_from_demo("R1", ["a", "b", "d"])
        proto = learner.get("R1")
        self.assertEqual(proto.action_set, frozenset({"a", "b", "c", "d"}))
        self.assertEqual(proto.n_demos, 2)

    def test_terminal_signatures_recorded(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b"], terminal_state=(0, 1))
        learner.update_from_demo("R1", ["a", "c"], terminal_state=(1, 1))
        proto = learner.get("R1")
        self.assertEqual(proto.terminal_signatures, [(0, 1), (1, 1)])

    def test_recipe_match_higher_for_matching_prefix(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b", "c"])
        learner.update_from_demo("R2", ["x", "y", "z"])
        # Prefix from R1 should match R1 better than R2.
        self.assertGreater(
            learner.recipe_match(["a", "b"], "R1"),
            learner.recipe_match(["a", "b"], "R2"),
        )

    def test_recipe_match_zero_for_unknown_recipe(self):
        learner = RecipePrototypeLearner()
        self.assertEqual(learner.recipe_match(["a"], "R_missing"), 0.0)

    def test_recipe_match_weights_are_configurable(self):
        prefix = ["a", "c", "b"]
        token_only = RecipePrototypeLearner(
            SimpleNamespace(
                recipe_match_token_weight=1.0,
                recipe_match_precedence_weight=0.0,
            )
        )
        token_only.update_from_demo("R1", ["a", "b", "c"])
        precedence_only = RecipePrototypeLearner(
            SimpleNamespace(
                recipe_match_token_weight=0.0,
                recipe_match_precedence_weight=1.0,
            )
        )
        precedence_only.update_from_demo("R1", ["a", "b", "c"])

        self.assertAlmostEqual(token_only.recipe_match(prefix, "R1"), 1.0)
        self.assertLess(precedence_only.recipe_match(prefix, "R1"), 1.0)


class FrontierTests(unittest.TestCase):
    def test_frontier_aligns_with_variants(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b", "c", "d"])
        # Two variants: one with the natural ordering, one shuffled.
        v1 = (("a", "b", "c", "d"), 1.0)
        v2 = (("a", "b", "d", "c"), 1.0)
        dist = learner.frontier(["a", "b"], "R1", [v1, v2])
        # Both variants point to "c" or "d" after the prefix "a, b". The
        # distribution should split between them.
        self.assertEqual(set(dist), {"c", "d"})
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)

    def test_frontier_weighted_by_variant_weight(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b", "c", "d"])
        v1 = (("a", "b", "c", "d"), 1.0)
        v2 = (("a", "b", "d", "c"), 3.0)
        dist = learner.frontier(["a", "b"], "R1", [v1, v2])
        # v2 has 3x the weight of v1, so "d" should outweigh "c".
        self.assertGreater(dist["d"], dist["c"])

    def test_frontier_falls_back_to_unused(self):
        learner = RecipePrototypeLearner()
        learner.update_from_demo("R1", ["a", "b", "c"])
        # Empty variants -> fallback uniform over unused.
        dist = learner.frontier(["a"], "R1", [])
        # "b" and "c" are unused; "a" already in prefix.
        self.assertEqual(set(dist), {"b", "c"})
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)

    def test_frontier_returns_distribution_or_empty(self):
        learner = RecipePrototypeLearner()
        # No prototype, no variants.
        dist = learner.frontier(["x"], "R_missing", [])
        self.assertEqual(dist, {})


if __name__ == "__main__":
    unittest.main()
