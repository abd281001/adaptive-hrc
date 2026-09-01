import unittest

import numpy as np

from src.environment import StateTracker
from src.models import (
    Settings,
    MaxEntIrl,
    build_features,
    feasible_actions,
    index_demos,
    top_actions,
    top_probability_tie_size,
)


def _demo(actions):
    tracker = StateTracker()
    trajectory = []
    for action in actions:
        trajectory.append((tuple(tracker.get_state_vector().tolist()), action))
        tracker.apply_action(action, enforce_preconditions=True)
    trajectory.append((tuple(tracker.get_state_vector().tolist()), "stop"))
    return trajectory


def _settings(**values):
    defaults = {
        "verbose": False,
        "irl_cold_steps": 2,
        "irl_warm_steps": 1,
        "irl_horizon": 8,
    }
    defaults.update(values)
    return Settings(**defaults)


class FeatureTests(unittest.TestCase):
    def test_state_action_maps_respect_given_order(self):
        demos = [[((0,), "a"), ((1,), "stop")], [((0,), "b"), ((2,), "stop")]]
        _, _, action_ids, actions = index_demos(
            demos, unique_actions=["b", "a", "stop"]
        )
        self.assertEqual(action_ids, {"b": 0, "a": 1, "stop": 2})
        self.assertEqual(actions[1], "a")

    def test_features_reuse_frozen_scale(self):
        tracker = StateTracker()
        first = tuple(tracker.get_state_vector().tolist())
        tracker.apply_action("transfer (pot, from=storage, to=cooking_station)")
        second = tuple(tracker.get_state_vector().tolist())
        matrix, mean, scale = build_features({0: first, 1: second})
        query, query_mean, query_scale = build_features(
            {0: second}, known_mean=mean, known_scale=scale
        )
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all(np.isfinite(query)))
        np.testing.assert_allclose(query_mean, mean)
        np.testing.assert_allclose(query_scale, scale)

    def test_raw_state_keeps_every_coordinate(self):
        state = tuple(StateTracker().get_state_vector().tolist())
        width = len(state)
        matrix, _, _ = build_features(
            {0: state},
            known_mean=np.zeros(width, dtype=np.float32),
            known_scale=np.ones(width, dtype=np.float32),
            feature_mode="raw_state",
        )
        self.assertEqual(matrix.shape, (1, width))
        np.testing.assert_array_equal(matrix[0], state)

    def test_modes_are_validated(self):
        with self.assertRaisesRegex(ValueError, "irl_features"):
            Settings(irl_features="invalid")
        with self.assertRaisesRegex(ValueError, "predictor"):
            Settings(predictor="invalid")
        with self.assertRaisesRegex(
            ValueError, "semantic_fallback_max_rms_distance",
        ):
            Settings(semantic_fallback_max_rms_distance=-0.1)
        with self.assertRaisesRegex(ValueError, "latent_strategy_rank"):
            Settings(latent_strategy_rank=0)
        with self.assertRaisesRegex(ValueError, "latent_strategy_knn"):
            Settings(latent_strategy_knn=0)
        with self.assertRaisesRegex(ValueError, "latent_strategy_strength"):
            Settings(latent_strategy_strength=-0.1)
        with self.assertRaisesRegex(ValueError, "latent_strategy_sequence_weight"):
            Settings(latent_strategy_sequence_weight=1.1)

    def test_semantic_features_mask_substitutable_ingredient_identity(self):
        tomato = StateTracker()
        mushroom = StateTracker()
        tomato.apply_action(
            "transfer (tomato, from=storage, to=prep_station)",
            enforce_preconditions=True,
        )
        mushroom.apply_action(
            "transfer (mushroom, from=storage, to=prep_station)",
            enforce_preconditions=True,
        )
        states = {
            0: tuple(tomato.get_state_vector()),
            1: tuple(mushroom.get_state_vector()),
        }
        semantic, _, _ = build_features(states, feature_mode="semantic")
        engineered, _, _ = build_features(states, feature_mode="engineered")
        np.testing.assert_allclose(semantic[0], semantic[1])
        self.assertFalse(np.allclose(engineered[0], engineered[1]))

class GaussianTieBreakingTests(unittest.TestCase):
    def test_exact_top_tie_uses_gaussian_draws_not_action_spelling(self):
        class FixedNormalDraws:
            def normal(self, *, loc, scale, size):
                self.parameters = (loc, scale, size)
                return np.asarray([-1.0, 2.0])

        distribution = {"alphabetically_first": 0.5, "zeta": 0.5}
        original = dict(distribution)
        rng = FixedNormalDraws()

        ranked = top_actions(distribution, k=2, rng=rng)

        self.assertEqual(ranked, ["zeta", "alphabetically_first"])
        self.assertEqual(rng.parameters, (0.0, 1.0, 2))
        self.assertEqual(distribution, original)
        self.assertEqual(top_probability_tie_size(distribution), 2)

    def test_seeded_tie_breaking_is_reproducible(self):
        distribution = {"a": 0.5, "b": 0.5, "c": 0.25}
        first = top_actions(
            distribution, k=3, rng=np.random.default_rng(29),
        )
        second = top_actions(
            distribution, k=3, rng=np.random.default_rng(29),
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(distribution))


class MaxEntIrlTests(unittest.TestCase):
    def setUp(self):
        self.demo = _demo([
            "transfer (pot, from=storage, to=cooking_station)",
            "turn_on (stove, cooking_station)",
        ])

    def test_fit_and_predict_semantic_action(self):
        model = MaxEntIrl(_settings())
        model.fit([self.demo], [1.0])
        dist = model.predict(self.demo[0][0])
        self.assertTrue(dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
        self.assertTrue(all("(" in action for action in dist))
        self.assertEqual(model.last_fit_stats["model_family"], "maxent_irl")
        self.assertEqual(
            model.last_fit_stats["occupancy_method"], "finite_horizon_dp"
        )
        self.assertEqual(model.last_fit_stats["irl_features"], "engineered")
        self.assertEqual(
            model.last_fit_stats["semantic_fallback_similarity_features"],
            "semantic_raw_rms",
        )
        self.assertGreater(
            model.features.shape[1], model.semantic_features.shape[1],
        )

    def test_default_uses_only_demonstrated_actions(self):
        model = MaxEntIrl(_settings())
        model.fit([self.demo])
        expansion = model.last_fit_stats["valid_action_expansion"]
        self.assertEqual(expansion["enabled"], 0.0)
        self.assertEqual(expansion["accepted"], 0.0)
        self.assertEqual(
            set(model.action_ids),
            {action for _state, action in self.demo},
        )

    def test_optional_expansion_accepts_semantic_actions(self):
        model = MaxEntIrl(_settings(expand_actions=True))
        model.fit([self.demo])
        expansion = model.last_fit_stats["valid_action_expansion"]
        self.assertEqual(expansion["enabled"], 1.0)
        self.assertGreater(expansion["accepted"], 0.0)

    def test_unseen_state_uses_semantic_value_interpolation(self):
        model = MaxEntIrl(_settings(
            semantic_fallback_max_rms_distance=10.0,
        ))
        model.fit([self.demo])
        tracker = StateTracker()
        tracker.apply_action("turn_on (sink, washing_station)")
        query = tuple(tracker.get_state_vector().tolist())
        self.assertNotIn(query, model.state_ids)
        self.assertTrue(model.predict(query))
        self.assertTrue(
            model.last_prediction_stats["semantic_fallback_attempted"]
        )
        self.assertTrue(
            model.last_prediction_stats["semantic_fallback_used"]
        )

    def test_semantic_fallback_can_be_disabled_for_irl_only_controls(self):
        model = MaxEntIrl(_settings(
            semantic_fallback_enabled=False,
            semantic_fallback_max_rms_distance=10.0,
            latent_strategy_enabled=False,
        ))
        model.fit([self.demo])
        tracker = StateTracker()
        tracker.apply_action("turn_on (sink, washing_station)")
        query = tuple(tracker.get_state_vector().tolist())

        distribution = model.predict(query)

        self.assertTrue(distribution)
        self.assertIsNone(model.semantic_features)
        self.assertFalse(model.last_fit_stats["semantic_fallback_enabled"])
        self.assertEqual(
            model.last_fit_stats["semantic_fallback_similarity_features"],
            "disabled",
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_enabled"]
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_attempted"]
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_used"]
        )
        self.assertFalse(
            model.last_prediction_stats["latent_strategy_eligible"]
        )

    def test_supported_state_action_never_uses_semantic_fallback(self):
        model = MaxEntIrl(_settings())
        model.fit([self.demo])
        state, action = self.demo[0]

        self.assertEqual(set(model.predict(state, [action])), {action})
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_attempted"]
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_used"]
        )

    def test_exact_state_learned_support_dominates_complete_distribution(self):
        demo = _demo([
            "transfer (pot, from=storage, to=cooking_station)",
            "transfer (pan, from=storage, to=cooking_station)",
        ])
        model = MaxEntIrl(_settings(
            semantic_fallback_max_rms_distance=10.0,
        ))
        model.fit([demo])
        state, supported_action = demo[0]
        unsupported_here = demo[1][1]

        distribution = model.predict(
            state, [supported_action, unsupported_here],
        )

        self.assertEqual(
            set(distribution), {supported_action, unsupported_here},
        )
        self.assertGreater(
            distribution[supported_action], distribution[unsupported_here],
        )
        self.assertEqual(model.last_prediction_stats["candidate_count"], 2)
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_attempted"]
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_used"]
        )

    def test_distant_semantic_neighbor_is_rejected_without_dropping_action(self):
        model = MaxEntIrl(_settings(
            semantic_fallback_max_rms_distance=0.0,
        ))
        model.fit([self.demo])
        tracker = StateTracker()
        tracker.apply_action("turn_on (sink, washing_station)")
        query = tuple(tracker.get_state_vector().tolist())

        distribution = model.predict(query)

        self.assertTrue(distribution)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=6)
        self.assertTrue(
            model.last_prediction_stats["semantic_fallback_attempted"]
        )
        self.assertFalse(
            model.last_prediction_stats["semantic_fallback_used"]
        )
        self.assertGreater(
            model.last_prediction_stats["semantic_fallback_rejected_actions"],
            0,
        )

    def test_feasibility_mask_keeps_all_valid_conflicting_actions(self):
        tracker = StateTracker()
        for action in (
            "transfer (bowl, from=storage, to=prep_station)",
            "transfer (tomato, from=storage, to=prep_station)",
            "transfer (onion, from=storage, to=prep_station)",
        ):
            tracker.apply_action(action, enforce_preconditions=True)
        state = tuple(tracker.get_state_vector())
        candidates = feasible_actions(state, (
            "cut (tomato, prep_station)",
            "cut (onion, prep_station)",
            "cook_contents (pot, cooking_station)",
        ))
        self.assertEqual(
            set(candidates),
            {
                "cut (tomato, prep_station)",
                "cut (onion, prep_station)",
            },
        )

    def test_raw_state_feature_mode(self):
        model = MaxEntIrl(_settings(irl_features="raw_state"))
        model.fit([self.demo])
        self.assertEqual(model.features.shape[1], StateTracker().n_features)
        self.assertEqual(model.last_fit_stats["irl_features"], "raw_state")

    def test_relative_weight_changes_reward(self):
        second = _demo([
            "transfer (pan, from=storage, to=cooking_station)",
            "turn_on (stove, cooking_station)",
        ])
        first = MaxEntIrl(_settings(seed=17, irl_cold_steps=4))
        weighted = MaxEntIrl(_settings(seed=17, irl_cold_steps=4))
        first.fit([self.demo, second], [1.0, 1.0])
        weighted.fit([self.demo, second], [1.0, 4.0])
        self.assertFalse(np.allclose(first.reward_weights, weighted.reward_weights))

    def test_seed_controls_reproducibility(self):
        first = MaxEntIrl(_settings(seed=7))
        second = MaxEntIrl(_settings(seed=7))
        other = MaxEntIrl(_settings(seed=8))
        first.fit([self.demo])
        second.fit([self.demo])
        other.fit([self.demo])
        np.testing.assert_array_equal(first.reward_weights, second.reward_weights)
        self.assertFalse(np.array_equal(first.reward_weights, other.reward_weights))


if __name__ == "__main__":
    unittest.main()
