import unittest

import numpy as np

from src.environment import StateTracker
from src.models import (
    Config,
    MaxEntIRL2,
    NGramMarkov,
    StateTransitionGraph,
    build_demo_trajectory,
    create_feature_matrix_2_0,
    create_state_action_mappings,
    ensemble_predict,
    top_k,
)


class IRLHelperTests(unittest.TestCase):
    def test_build_demo_trajectory_preserves_actions_and_appends_stop(self):
        traj = build_demo_trajectory(["abstract_action_1", "abstract_action_2"])
        self.assertEqual([action for _, action in traj], ["abstract_action_1", "abstract_action_2", "stop"])
        self.assertTrue(all(len(state) == len(traj[0][0]) for state, _ in traj))

    def test_create_state_action_mappings_respects_unique_actions(self):
        demos = [[((0,), "a"), ((1,), "stop")], [((0,), "b"), ((2,), "stop")]]
        _, _, action_to_idx, idx_to_action = create_state_action_mappings(
            demos,
            unique_actions=["b", "a", "stop"],
        )
        self.assertEqual(action_to_idx, {"b": 0, "a": 1, "stop": 2})
        self.assertEqual(idx_to_action[1], "a")

    def test_create_feature_matrix_reuses_welford_statistics(self):
        tracker = StateTracker()
        s0 = tuple(tracker.get_state_vector().tolist())
        tracker.apply_action("transfer (pot, from=storage, to=cooking_station)")
        s1 = tuple(tracker.get_state_vector().tolist())
        fm, col_min, col_max = create_feature_matrix_2_0({0: s0, 1: s1})
        fm2, col_min2, col_max2 = create_feature_matrix_2_0(
            {0: s1},
            col_min_known=col_min,
            col_max_known=col_max,
        )
        self.assertEqual(fm.shape[0], 2)
        self.assertEqual(fm2.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(fm)))
        self.assertTrue(np.all(np.isfinite(fm2)))
        np.testing.assert_allclose(col_min2, col_min)
        np.testing.assert_allclose(col_max2, col_max)


class LearningHeadTests(unittest.TestCase):
    def test_ngram_markov_prefers_matching_context(self):
        demos = [build_demo_trajectory(["a", "b", "c"])]
        head = NGramMarkov(order=2, prob_floor=1e-6)
        head.fit(demos, weights=[1.0])
        dist = head.predict(demos[0][2][0], ["a", "b"])
        self.assertEqual(top_k(dist, k=1)[0], "c")

    def test_state_transition_graph_uses_previous_action_fallback(self):
        demos = [
            build_demo_trajectory(["a", "x"]),
            build_demo_trajectory(["b", "y"]),
        ]
        head = StateTransitionGraph(prob_floor=1e-6)
        head.fit(demos, weights=[1.0, 1.0])
        dist = head.predict(demos[0][1][0], ["a"])
        self.assertEqual(top_k(dist, k=1)[0], "x")

    def test_ensemble_predict_normalises_and_top_k_orders_descending(self):
        cfg = Config(
            verbose=False,
            ensemble_alpha=0.5,
            ensemble_beta=0.3,
            ensemble_delta=0.1,
            prob_floor=1e-6,
        )
        dist = ensemble_predict(
            {"a": 0.9, "b": 0.1},
            {"a": 0.8, "b": 0.2},
            {"a": 0.6, "b": 0.4},
            cfg=cfg,
        )
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
        self.assertEqual(top_k(dist, k=2), ["a", "b"])

    def test_maxent_irl_fit_predict_smoke(self):
        demo = build_demo_trajectory(
            [
                "transfer (pot, from=storage, to=cooking_station)",
                "turn_on (stove, cooking_station)",
            ]
        )
        cfg = Config(
            verbose=False,
            maxent_iters_cold=5,
            maxent_iters_warm=3,
            maxent_mc_rollouts=10,
        )
        model = MaxEntIRL2(cfg=cfg)
        model.fit([demo], demo_weights=[1.0])
        dist = model.predict(demo[0][0])
        self.assertTrue(dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)


class WeightedLossContractTests(unittest.TestCase):
    """The IRL head must respond to absolute weight magnitudes; the n-gram and
    graph heads must be invariant to a uniform multiplicative scaling of weights
    (per-context normalisation cancels constants)."""

    def _two_demos(self):
        return [
            build_demo_trajectory([
                "transfer (pot, from=storage, to=cooking_station)",
                "turn_on (stove, cooking_station)",
            ]),
            build_demo_trajectory([
                "transfer (pan, from=storage, to=cooking_station)",
                "turn_on (stove, cooking_station)",
            ]),
        ]

    def test_irl_responds_to_absolute_weight_scale(self):
        cfg = Config(verbose=False, maxent_iters_cold=8, maxent_iters_warm=4, maxent_mc_rollouts=8)
        demos = self._two_demos()
        m1 = MaxEntIRL2(cfg=cfg); m1.fit(demos, demo_weights=[1.0, 1.0])
        m2 = MaxEntIRL2(cfg=cfg); m2.fit(demos, demo_weights=[0.05, 0.05])
        # Same relative weighting; under the new contract, absolute scale matters.
        self.assertFalse(np.allclose(m1.theta, m2.theta), "IRL theta should change with absolute weight scale")

    def test_ngram_invariant_to_uniform_weight_scaling(self):
        demos = self._two_demos()
        h1 = NGramMarkov(order=2, prob_floor=1e-6); h1.fit(demos, weights=[1.0, 1.0])
        h2 = NGramMarkov(order=2, prob_floor=1e-6); h2.fit(demos, weights=[0.3, 0.3])
        d1 = h1.predict(demos[0][1][0], ["transfer (pot, from=storage, to=cooking_station)"])
        d2 = h2.predict(demos[0][1][0], ["transfer (pot, from=storage, to=cooking_station)"])
        self.assertEqual(set(d1), set(d2))
        for k in d1:
            self.assertAlmostEqual(d1[k], d2[k], places=9)

    def test_ngram_responds_to_relative_weights(self):
        demos = [
            [((0,), "ctx"), ((1,), "low"), ((2,), "stop")],
            [((0,), "ctx"), ((1,), "high"), ((3,), "stop")],
        ]
        head = NGramMarkov(order=1, prob_floor=1e-6)
        head.fit(demos, weights=[1.0, 4.0])
        dist = head.predict((1,), ["ctx"])
        self.assertGreater(dist["high"], dist["low"])
        self.assertEqual(top_k(dist, 1)[0], "high")

    def test_graph_invariant_to_uniform_weight_scaling(self):
        demos = self._two_demos()
        g1 = StateTransitionGraph(prob_floor=1e-6); g1.fit(demos, weights=[1.0, 1.0])
        g2 = StateTransitionGraph(prob_floor=1e-6); g2.fit(demos, weights=[0.4, 0.4])
        d1 = g1.predict(demos[0][1][0], ["transfer (pot, from=storage, to=cooking_station)"])
        d2 = g2.predict(demos[0][1][0], ["transfer (pot, from=storage, to=cooking_station)"])
        self.assertEqual(set(d1), set(d2))
        for k in d1:
            self.assertAlmostEqual(d1[k], d2[k], places=9)

    def test_graph_responds_to_relative_weights(self):
        demos = [
            [((0,), "ctx"), ((1,), "low"), ((2,), "stop")],
            [((0,), "ctx"), ((1,), "high"), ((3,), "stop")],
        ]
        graph = StateTransitionGraph(prob_floor=1e-6)
        graph.fit(demos, weights=[1.0, 4.0])
        dist = graph.predict((1,), ["ctx"])
        self.assertGreater(dist["high"], dist["low"])
        self.assertEqual(top_k(dist, 1)[0], "high")

    def test_ngram_excludes_zero_weight_demos(self):
        demos = self._two_demos()
        head = NGramMarkov(order=1, prob_floor=1e-6)
        head.fit(demos, weights=[1.0, 0.0])  # second demo's "pan" should not appear
        self.assertNotIn("transfer (pan, from=storage, to=cooking_station)", head.vocab)


class ReproducibilityTests(unittest.TestCase):
    """Same seed -> same theta. Per-Config rng/prng must isolate stochastic
    paths from the global random module."""

    def test_irl_reproducible(self):
        demos = [
            build_demo_trajectory([
                "transfer (pot, from=storage, to=cooking_station)",
                "turn_on (stove, cooking_station)",
            ]),
        ]
        cfg1 = Config(verbose=False, seed=1337, maxent_iters_cold=10, maxent_iters_warm=4, maxent_mc_rollouts=8)
        m1 = MaxEntIRL2(cfg=cfg1); m1.fit(demos, demo_weights=[1.0])
        cfg2 = Config(verbose=False, seed=1337, maxent_iters_cold=10, maxent_iters_warm=4, maxent_mc_rollouts=8)
        m2 = MaxEntIRL2(cfg=cfg2); m2.fit(demos, demo_weights=[1.0])
        np.testing.assert_array_equal(m1.theta, m2.theta)

    def test_different_seeds_diverge(self):
        demos = [
            build_demo_trajectory([
                "transfer (pot, from=storage, to=cooking_station)",
                "turn_on (stove, cooking_station)",
            ]),
        ]
        cfg1 = Config(verbose=False, seed=1, maxent_iters_cold=10, maxent_iters_warm=4, maxent_mc_rollouts=8)
        m1 = MaxEntIRL2(cfg=cfg1); m1.fit(demos, demo_weights=[1.0])
        cfg2 = Config(verbose=False, seed=2, maxent_iters_cold=10, maxent_iters_warm=4, maxent_mc_rollouts=8)
        m2 = MaxEntIRL2(cfg=cfg2); m2.fit(demos, demo_weights=[1.0])
        self.assertFalse(np.array_equal(m1.theta, m2.theta), "different seeds must produce different theta")


if __name__ == "__main__":
    unittest.main()
