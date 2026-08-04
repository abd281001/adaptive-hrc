import unittest

from src.memory import (
    Disambiguator,
    KnownVariant,
    jaccard,
    kendall_tau_distance,
    variant_hash,
)
from src.models import Config
from src.representations import identity_token_from_observation, observations_from_actions
from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.preferences import materialize


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


class VariantHashOrderPreservingTests(unittest.TestCase):
    def test_raw_hash_distinguishes_orderings(self):
        h1 = variant_hash(["a(x)", "a(y)", "b"])
        h2 = variant_hash(["a(y)", "a(x)", "b"])
        self.assertNotEqual(h1, h2)

    def test_order_preserving_hash_preserves_different_type_order(self):
        h1 = variant_hash(["a", "b", "c"])
        h2 = variant_hash(["b", "a", "c"])
        self.assertNotEqual(h1, h2)


class DisambiguatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(jaccard_threshold=0.6)
        self.d = Disambiguator(self.cfg)

    def test_empty_library_is_new(self):
        c = self.d.classify(["a", "b"], [])
        self.assertEqual(c.kind, "new_recipe")

    def test_identical_is_known(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, variant_hash(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["a", "b", "c"], lib)
        self.assertEqual(c.kind, "known")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_same_set_different_order_is_pref_shift(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, variant_hash(["a", "b", "c"]), ("a", "b", "c"))]
        c = self.d.classify(["c", "b", "a"], lib)
        self.assertEqual(c.kind, "preference_shift")
        self.assertEqual(c.recipe_id, TOMATO_ONION_SOUP)

    def test_disjoint_is_new(self):
        lib = [KnownVariant(TOMATO_ONION_SOUP, variant_hash(["a", "b"]), ("a", "b"))]
        c = self.d.classify(["x", "y", "z"], lib)
        self.assertEqual(c.kind, "new_recipe")

    def test_partial_scoring_prefers_matching_prefix(self):
        v1 = KnownVariant(TOMATO_ONION_SOUP, BASE_PREF, ("a", "b", "c", "d"))
        v2 = KnownVariant(TOMATO_ONION_SOUP, ALT_PREF, ("a", "c", "b", "d"))
        ranked = self.d.score_partial(["a", "b"], [v1, v2])
        self.assertEqual(ranked[0][0].variant_hash, BASE_PREF)

    def test_canonical_identity_recognizes_reordered_context_variant(self):
        cfg = Config(identity_jaccard_threshold=0.8, identity_score_margin=0.05)
        d = Disambiguator(cfg)
        raw_base = ("raw_move_empty", "raw_load")
        raw_variant = ("raw_load", "raw_move_loaded")
        identity = ("identity_load", "identity_move_bowl")
        lib = [KnownVariant(TOMATO_ONION_SOUP, variant_hash(raw_base), raw_base, identity)]
        cls = d.classify(raw_variant, lib, identity_sequence=("identity_load", "identity_move_bowl"))
        self.assertEqual(cls.kind, "preference_shift")
        self.assertEqual(cls.recipe_id, TOMATO_ONION_SOUP)

    def test_identity_margin_rejects_ambiguous_recipe(self):
        cfg = Config(identity_jaccard_threshold=0.8, identity_score_margin=0.05)
        d = Disambiguator(cfg)
        seq = ("raw_new_a", "raw_new_b")
        lib = [
            KnownVariant("R1", variant_hash(("raw_a",)), ("raw_a",), ("identity_a", "identity_b")),
            KnownVariant("R2", variant_hash(("raw_b",)), ("raw_b",), ("identity_a", "identity_b")),
        ]
        cls = d.classify(seq, lib, identity_sequence=("identity_a", "identity_b"))
        self.assertEqual(cls.kind, "new_recipe")


class CanonicalIdentityTokenTests(unittest.TestCase):
    def test_loaded_and_empty_container_moves_have_the_same_identity_token(self):
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
        empty_move = observations_from_actions(empty_actions)[-1]
        loaded_move = observations_from_actions(loaded_actions)[-1]
        self.assertEqual(empty_move.action_vector, loaded_move.action_vector)
        self.assertEqual(
            identity_token_from_observation(empty_move),
            identity_token_from_observation(loaded_move),
        )

    def test_default_ladder_variants_remain_known_among_all_recipes(self):
        """Regression for representative isolated and composition variants."""
        agent = AdaptiveHRCAgent(Config(
            verbose=False,
            maxent_iters_cold=1,
            maxent_iters_warm=1,
        ))

        def encode(actions):
            raw, identity = [], []
            for observation in observations_from_actions(actions):
                raw.append(agent._token_for_vector(observation.action_vector))
                identity.append(identity_token_from_observation(observation))
            return raw, identity

        for recipe_id, recipe_fn in gen.recipe_library().items():
            raw, identity = encode(recipe_fn())
            agent.memory.register(recipe_id, raw, step=1, identity_ordering=identity)
        library = agent.memory.library()

        for preference in (
            "p6_deferred_cook_start",
            "p12_multi_stage_reorganization",
            "p14_mise_load_clean",
        ):
            for recipe_id, recipe_fn in gen.recipe_library().items():
                raw, identity = encode(materialize(recipe_fn(), preference))
                cls = agent.disambig.classify(raw, library, identity_sequence=identity)
                self.assertNotEqual(cls.kind, "new_recipe", f"{recipe_id}/{preference}")
                self.assertEqual(cls.recipe_id, recipe_id, f"{recipe_id}/{preference}")


if __name__ == "__main__":
    unittest.main()
