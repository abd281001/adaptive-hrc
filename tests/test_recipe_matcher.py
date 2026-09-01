import unittest

from src.memory import (
    RecipeMatcher,
    KnownVariant,
    jaccard,
    kendall_tau_distance,
    make_variant_id,
)
from src.models import Settings
from src.representations import observe_actions
from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.preferences import apply_preset_actions


TOMATO_ONION_SOUP = "tomato_onion_soup"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class ConfigValidationTests(unittest.TestCase):
    def test_commit_thresholds_and_floors_are_ordered(self):
        with self.assertRaisesRegex(ValueError, "tentative threshold"):
            Settings(
                tentative_threshold=0.8,
                commit_threshold=0.7,
            )
        with self.assertRaisesRegex(ValueError, "known-score floor"):
            Settings(
                commit_threshold=0.95,
                known_score_floor=0.9,
            )

    def test_provisional_weight_cannot_exceed_its_cap(self):
        with self.assertRaisesRegex(ValueError, "weight cannot exceed"):
            Settings(provisional_weight=0.8, provisional_cap=0.7)


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


class VariantHashOrderPreservingTests(unittest.TestCase):
    def test_raw_hash_distinguishes_orderings(self):
        h1 = make_variant_id(["a(x)", "a(y)", "b"])
        h2 = make_variant_id(["a(y)", "a(x)", "b"])
        self.assertNotEqual(h1, h2)

    def test_order_preserving_hash_preserves_different_type_order(self):
        h1 = make_variant_id(["a", "b", "c"])
        h2 = make_variant_id(["b", "a", "c"])
        self.assertNotEqual(h1, h2)


class RecipeMatcherTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(match_threshold=0.6)
        self.d = RecipeMatcher(self.settings)

    def test_empty_library_is_new(self):
        c = self.d.classify(["a", "b"], [])
        self.assertEqual(c.kind, "new_recipe")

    def test_identical_is_known(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, make_variant_id(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["a", "b", "c"], lib)
        self.assertEqual(c.kind, "known")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_same_set_different_order_is_pref_shift(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, make_variant_id(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["c", "b", "a"], lib)
        self.assertEqual(c.kind, "preference_shift")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_disjoint_is_new(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, make_variant_id(["a", "b"]), ("a", "b"))]
        c = self.d.classify(["x", "y", "z"], lib)
        self.assertEqual(c.kind, "new_recipe")

    def test_partial_scoring_prefers_matching_prefix(self):
        first = KnownVariant(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b", "c", "d"))
        second = KnownVariant(TOMATO_ONION_SOUP, ALT_PREF, ("a", "c", "b", "d"))
        ranked = self.d.score_prefix(["a", "b"], [first, second])
        self.assertEqual(ranked[0][0].variant_id, BASE_PREF)

    def test_semantic_actions_recognize_reordered_workflow_variant(self):
        settings = Settings(match_threshold=0.8, match_margin=0.05)
        d = RecipeMatcher(settings)
        base = ("move_container (bowl, from=prep_station, to=plating_station)", "load (tomato, bowl, prep_station)")
        variant = ("load (tomato, bowl, prep_station)", "move_container (bowl, from=prep_station, to=plating_station)")
        lib = [KnownVariant(TOMATO_ONION_SOUP, make_variant_id(base), base)]
        cls = d.classify(variant, lib)
        self.assertEqual(cls.kind, "preference_shift")
        self.assertEqual(cls.recipe_id, TOMATO_ONION_SOUP)

    def test_recipe_margin_rejects_ambiguous_recipe(self):
        settings = Settings(match_threshold=0.8, match_margin=0.05)
        d = RecipeMatcher(settings)
        seq = ("action_a", "action_b")
        lib = [
            KnownVariant("R1", make_variant_id(seq), seq),
            KnownVariant("R2", "different_hash", seq),
        ]
        # Avoid exact-hash short-circuit while retaining equal recipe scores.
        cls = d.classify(("action_b", "action_a"), lib)
        self.assertEqual(cls.kind, "new_recipe")

    def test_batched_scores_match_singleton_classification(self):
        d = RecipeMatcher(Settings(profile=True, match_threshold=0.0))
        sequence = ("query_a", "query_b")
        library = [
            KnownVariant(
                f"R{index // 2}", make_variant_id((f"action_{index}",)),
                ("query_a", "query_b") if index % 3 == 0 else ("query_a", f"action_{index}"),
            )
            for index in range(12)
        ]
        expected = {}
        for variant in library:
            result = d.classify(sequence, [variant])
            previous = expected.get(variant.recipe_id)
            if previous is None or result.jaccard > previous[1]:
                expected[variant.recipe_id] = (
                    variant.variant_id, result.jaccard, result.order_distance,
                )

        d.profile.clear()
        actual = d.score(sequence, library)

        self.assertEqual(set(actual), set(expected))
        for recipe_id, (variant, score, distance) in actual.items():
            self.assertEqual(
                (variant.variant_id, score, distance), expected[recipe_id],
            )
        self.assertEqual(d.profile["score"][0], 1)

    def test_registry_sizes_use_one_batched_scoring_pass(self):
        d = RecipeMatcher(Settings(profile=True))
        for size in (8, 32, 128, 512):
            library = [
                KnownVariant(f"R{index}", f"h{index}", ("shared", f"action_{index}"))
                for index in range(size)
            ]
            d.profile.clear()
            scores = d.score(("shared", "query"), library)
            self.assertEqual(len(scores), size)
            self.assertEqual(d.profile["score"][0], 1)


class SemanticActionRepresentationTests(unittest.TestCase):
    def test_loaded_and_empty_container_moves_keep_the_same_semantic_action(self):
        empty_actions = [
            "transfer (bowl, from=storage, to=prep_station)",
            "move_container (bowl, from=prep_station, to=plating_station)",
        ]
        loaded_actions = [
            "transfer (bowl, from=storage, to=prep_station)",
            "transfer (tomato, from=storage, to=prep_station)",
            "load (tomato, bowl, prep_station)",
            "move_container (bowl, from=prep_station, to=plating_station)",
        ]
        empty_move = observe_actions(empty_actions)[-1]
        loaded_move = observe_actions(loaded_actions)[-1]
        self.assertEqual(empty_move.action, loaded_move.action)
        self.assertEqual(empty_move.action, empty_actions[-1])

    def test_default_ladder_variants_remain_known_among_all_recipes(self):
        """Regression for representative isolated and composition variants."""
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
        ))

        for recipe_id, recipe_fn in recipe_builders().items():
            agent.library.register(recipe_id, recipe_fn(), step=1)
        library = agent.library.known_variants()

        for preference in (
            "cook_start_late",
            "prep_loading_serving_cleanup",
            "prep_loading_cleanup",
        ):
            for recipe_id, recipe_fn in recipe_builders().items():
                actions = apply_preset_actions(recipe_fn(), preference)
                cls = agent.matcher.classify(actions, library)
                self.assertNotEqual(cls.kind, "new_recipe", f"{recipe_id}/{preference}")
                self.assertEqual(cls.recipe_id, recipe_id, f"{recipe_id}/{preference}")


if __name__ == "__main__":
    unittest.main()
