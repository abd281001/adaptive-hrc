import hashlib
import unittest

import numpy as np

from src.environment import StateTracker
from src.models import (
    Settings,
    MaxEntIrl,
    build_features,
    feasible_actions,
    feature_names,
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

    def test_validated_lightweight_sequence_setting_is_the_default(self):
        self.assertEqual(Settings().latent_strategy_rank, 8)
        self.assertEqual(Settings().latent_strategy_knn, 3)
        self.assertEqual(Settings().latent_strategy_sequence_weight, 0.5)

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


class RewardFeaturePlanTests(unittest.TestCase):
    """Pin the reward representation so a learned weight keeps its meaning.

    ``reward_weights`` is indexed positionally, so silently reordering,
    inserting, or renaming a column would reinterpret every previously
    reported weight while leaving every accuracy number intact.  These
    hashes fail loudly instead.
    """

    EXPECTED = {
        "engineered": (199, "4fbf0803ddd69b8096b9ee08ca445927b33d35e3a088df65249545b4eca33637"),
        "semantic": (50, "5dd5f6a8dc3f0ac13253c87aec87bdaafbbf746031f36111fc0546955c8105cd"),
    }

    def test_feature_plan_order_and_width_are_frozen(self):
        for mode, (width, digest) in self.EXPECTED.items():
            with self.subTest(feature_mode=mode):
                names = feature_names(mode)
                self.assertEqual(len(names), width)
                self.assertEqual(len(set(names)), width, "feature names must be unique")
                self.assertEqual(
                    hashlib.sha256("\n".join(names).encode()).hexdigest(), digest,
                )

    def test_feature_names_align_with_built_columns(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (pot, from=storage, to=cooking_station)")
        state = tuple(tracker.get_state_vector().astype(int).tolist())
        for mode in ("engineered", "semantic", "raw_state"):
            with self.subTest(feature_mode=mode):
                raw, _mean, _scale = build_features(
                    {0: state}, feature_mode=mode, normalize=False,
                )
                self.assertEqual(raw.shape[1], len(feature_names(mode)))

    def test_named_columns_evaluate_to_their_definition(self):
        tracker = StateTracker()
        for action in (
            "transfer (pot, from=storage, to=cooking_station)",
            "turn_on (stove, cooking_station)",
        ):
            tracker.apply_action(action)
        state = tuple(tracker.get_state_vector().astype(int).tolist())
        raw, _mean, _scale = build_features(
            {0: state}, feature_mode="engineered", normalize=False,
        )
        column = dict(zip(feature_names("engineered"), raw[0].tolist()))
        self.assertEqual(column["stove_on"], 1.0)
        self.assertEqual(column["pot_at_cooking_station"], 1.0)
        # Interaction composite: stove on AND cookware present.
        self.assertEqual(column["stove_on_x_cookware_present"], 1.0)
        self.assertEqual(column["blender_on_x_glass_present"], 0.0)


if __name__ == "__main__":
    unittest.main()


class FisherTests(unittest.TestCase):
    """The Fisher must measure policy curvature, not feature magnitude.

    The previous estimator returned the replay-weighted mean of ``phi(s)**2``,
    which is the feature second moment: it is large wherever the state features
    are large, regardless of whether the reward weights influence the action
    choice there at all. EWC built on it anchored the wrong directions.
    """

    def _fitted(self, **overrides):
        actions = [
            "transfer (pot, from=storage, to=cooking_station)",
            "transfer (tomato, from=storage, to=prep_station)",
            "cut (tomato, prep_station)",
        ]
        alternative = [actions[1], actions[0], actions[2]]
        demos = [_demo(actions), _demo(alternative)]
        model = MaxEntIrl(_settings(**overrides))
        model.fit(demos)
        return model, demos

    def test_fisher_records_its_estimator_and_stays_finite_and_nonnegative(self):
        model, demos = self._fitted()
        fisher = model.fisher(demos)

        self.assertIsNotNone(fisher)
        self.assertEqual(fisher.shape, model.reward_weights.shape)
        self.assertTrue(np.all(np.isfinite(fisher)))
        self.assertTrue(np.all(fisher >= 0.0))
        self.assertLessEqual(float(fisher.max()), float(model.settings.fisher_cap))
        self.assertEqual(
            model.last_fisher_stats["fisher_estimator"],
            "diagonal_empirical_fisher_one_step_q_linearization",
        )
        self.assertGreater(model.last_fisher_stats["fisher_scored_visits"], 0.0)

    def test_fisher_is_not_the_feature_second_moment(self):
        model, demos = self._fitted()
        fisher = model.fisher(demos)

        # The quantity the old implementation returned.
        second_moment = np.zeros_like(fisher)
        total = 0.0
        for demo in demos:
            for state, _action in demo:
                state_id = model.state_ids.get(state)
                if state_id is None:
                    continue
                feature = model.features[state_id]
                second_moment += feature * feature
                total += 1.0
        if total:
            second_moment /= total

        self.assertFalse(np.allclose(fisher, second_moment))
        # Nor a rescaling of it: the two weight the feature directions
        # differently, which is the whole point.
        def unit(vector):
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm > 0.0 else vector

        self.assertFalse(np.allclose(unit(fisher), unit(second_moment), atol=1e-3))
        # Visits where the policy cannot discriminate contribute no curvature at
        # all, while the second moment charges them their full feature mass.
        self.assertGreater(model.last_fisher_stats["fisher_zero_score_visits"], 0.0)

    def test_fisher_scales_with_the_policy_temperature(self):
        """The score carries a 1/T factor, so the Fisher carries 1/T**2."""
        cold, demos = self._fitted(irl_temperature=0.25)
        warm, _ = self._fitted(irl_temperature=1.0)

        cold_fisher = cold.fisher(demos)
        warm_fisher = warm.fisher(demos)

        self.assertGreater(float(cold_fisher.sum()), float(warm_fisher.sum()))

    def test_fisher_returns_none_before_any_fit(self):
        self.assertIsNone(MaxEntIrl(_settings()).fisher([]))
