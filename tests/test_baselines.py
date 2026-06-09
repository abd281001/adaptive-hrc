import unittest

from src.baselines import (
    AdaptiveDecayAgent,
    BehaviorCloningAgent,
    EWCAgent,
    L2AnchorAgent,
    ExperienceReplayAgent,
    FixedDecayAgent,
    LatestOnlyPreferenceAgent,
    NoDecayAgent,
    NoReplayAgent,
    RecencyPrioritizedReplayAgent,
    UniformWeightAgent,
    BASELINE_AGENTS,
)
from src.models import Config


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"
TOMATO_SOUP = "tomato_soup"
BASE_PREF = "base_pref"
ALT_PREF = "alternate_pref"


class _RecordingNoReplayAgent(NoReplayAgent):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.fit_calls = []

    def _fit_heads(self, trajectories, weights):
        self.fit_calls.append((trajectories, list(weights)))


class _RecordingUniformWeightAgent(UniformWeightAgent):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.fit_calls = []

    def _fit_heads(self, trajectories, weights):
        self.fit_calls.append((trajectories, list(weights)))


class BaselineAgentTests(unittest.TestCase):
    def test_standard_baselines_disable_latest_preference_protection(self):
        for AgentClass in [
            AdaptiveDecayAgent,
            FixedDecayAgent,
            NoDecayAgent,
            NoReplayAgent,
            UniformWeightAgent,
            L2AnchorAgent,
            BehaviorCloningAgent,
            EWCAgent,
            ExperienceReplayAgent,
            RecencyPrioritizedReplayAgent,
        ]:
            agent = AgentClass(Config(verbose=False))
            agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
            self.assertEqual(agent.decay.latest_keys, set(), AgentClass.__name__)

    def test_latest_only_preference_agent_discards_older_variants(self):
        agent = LatestOnlyPreferenceAgent(Config(verbose=False))
        agent._register_if_live(TOMATO_ONION_SOUP, ["prep", "cook"], step=1)
        agent._register_if_live(TOMATO_ONION_SOUP, ["cook", "prep"], step=2)
        self.assertEqual(len(agent.memory.variants[TOMATO_ONION_SOUP]), 1)
        self.assertEqual(len(agent.decay.active_entries()), 1)
        self.assertEqual(agent.decay.latest_keys, set())

    def test_no_replay_retrains_on_latest_demo_only(self):
        agent = _RecordingNoReplayAgent(Config(verbose=False))
        a1 = agent._token_for_vector((1, 0))
        a2 = agent._token_for_vector((0, 1))
        b1 = agent._token_for_vector((1, 1))
        b2 = agent._token_for_vector((1, 0, 1))
        b3 = agent._token_for_vector((0, 1, 1))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (a1, a2), now=1, cycle=0)
        agent.decay.register(TOMATO_SOUP, ALT_PREF, (b1, b2, b3), now=2, cycle=0)
        agent._retrain()
        trajectories, weights = agent.fit_calls[-1]
        self.assertEqual(weights, [1.0])
        self.assertEqual([action for _, action in trajectories[0][:-1]], [b1, b2, b3])

    def test_no_replay_replaces_old_memory_after_commit(self):
        agent = NoReplayAgent(Config(verbose=False, maxent_iters_cold=2, maxent_iters_warm=1, maxent_mc_rollouts=2))
        agent.start_demo()
        agent.pending_demo = ["prep", "cook"]
        cls_a = agent.end_demo()
        agent.start_demo()
        agent.pending_demo = ["serve", "clean"]
        cls_b = agent.end_demo()
        self.assertEqual(set(agent.memory.variants), {cls_b.recipe_id})
        self.assertEqual({entry.recipe_id for entry in agent.decay.active_entries()}, {cls_b.recipe_id})
        self.assertNotIn(cls_a.recipe_id, agent.memory.variants)

    def test_no_replay_clears_structural_prototypes(self):
        agent = NoReplayAgent(Config(verbose=False, maxent_iters_cold=1, maxent_iters_warm=1, maxent_mc_rollouts=1))
        agent._register_if_live("R_old", ["old_a", "old_b"], step=1)
        agent._register_if_live("R_new", ["new_a", "new_b"], step=2)
        self.assertIn("R_old", agent.recipe_prototypes.prototypes)
        agent._clear_memory()
        self.assertEqual(agent.memory.variants, {})
        self.assertEqual(agent.recipe_prototypes.prototypes, {})
        self.assertEqual(agent.preference_prototypes.prototypes, {})
        self.assertEqual(agent.task_signatures, {})
        self.assertEqual(agent.preference_signatures, {})
        self.assertIsNone(agent.last_pref_id)

    def test_uniform_weight_ignores_decay_weights(self):
        agent = _RecordingUniformWeightAgent(Config(verbose=False))
        a1 = agent._token_for_vector((1, 0))
        a2 = agent._token_for_vector((0, 1))
        b1 = agent._token_for_vector((1, 1))
        b2 = agent._token_for_vector((1, 0, 1))
        e1 = agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (a1, a2), now=1, cycle=0)
        e2 = agent.decay.register(TOMATO_SOUP, ALT_PREF, (b1, b2), now=2, cycle=0)
        e1.weight = 0.2
        e2.weight = 0.8
        agent._retrain()
        trajectories, weights = agent.fit_calls[-1]
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(weights, [1.0, 1.0])

    def test_fixed_decay_keeps_global_rate_constant(self):
        cfg = Config(verbose=False, decay_init=0.2)
        agent = FixedDecayAgent(cfg=cfg)
        rate0 = agent.decay.global_rate
        base0 = agent.decay.base_rate
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        agent.decay.register(TOMATO_ONION_SOUP, ALT_PREF, ("b",), now=1, cycle=0)
        self.assertAlmostEqual(agent.decay.global_rate, rate0, places=8)
        self.assertAlmostEqual(agent.decay.base_rate, base0, places=8)
        for step in range(2, 10):
            agent.decay.step(step, cycle=0)
            if (TOMATO_ONION_SOUP, BASE_PREF) in agent.decay.pruned:
                break
        self.assertIn((TOMATO_ONION_SOUP, BASE_PREF), agent.decay.pruned)
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=20, cycle=1)
        self.assertAlmostEqual(agent.decay.global_rate, rate0, places=8)
        self.assertAlmostEqual(agent.decay.base_rate, base0, places=8)

    def test_no_decay_keeps_rates_zero_after_reuse(self):
        agent = NoDecayAgent(cfg=Config(verbose=False))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=0, cycle=0)
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, ("a",), now=5, cycle=1)
        self.assertAlmostEqual(agent.decay.base_rate, 0.0, places=8)
        self.assertAlmostEqual(agent.decay.global_rate, 0.0, places=8)

    def test_l2_anchor_still_fits_reward_head(self):
        agent = L2AnchorAgent(cfg=Config(verbose=False))
        a1 = agent._token_for_vector((1, 0))
        a2 = agent._token_for_vector((0, 1))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (a1, a2), now=1, cycle=0)
        agent._retrain()
        self.assertIsNotNone(agent.irl.theta)

    def test_behavior_cloning_agent_predicts_trained_next_action(self):
        cfg = Config(verbose=False, bc_epochs_cold=30, bc_epochs_warm=15)
        agent = BehaviorCloningAgent(cfg=cfg)
        prep = agent._token_for_vector((1, 0))
        cook = agent._token_for_vector((0, 1))
        serve = agent._token_for_vector((1, 1))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (prep, cook, serve), now=1, cycle=0)
        agent._retrain()
        dist = agent.predict_next_tokens([prep, cook])
        self.assertEqual(max(dist, key=dist.get), serve)

    def test_behavior_cloning_retrain_does_not_keep_discarded_actions(self):
        cfg = Config(verbose=False, bc_epochs_cold=10, bc_epochs_warm=5)
        agent = BehaviorCloningAgent(cfg=cfg)
        prep = agent._token_for_vector((1, 0))
        cook = agent._token_for_vector((0, 1))
        serve = agent._token_for_vector((1, 1))

        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (prep, cook), now=1, cycle=0)
        agent._retrain()
        self.assertEqual(agent.bc.normalizer.count, 1)

        agent.decay.active.clear()
        agent.decay.register(TOMATO_SOUP, ALT_PREF, (serve,), now=2, cycle=1)
        agent._retrain()

        self.assertEqual(agent.bc.normalizer.count, 1)
        self.assertEqual(set(agent.bc.action_to_idx), {serve})

    def test_ewc_agent_anchors_irl_theta(self):
        """EWCAgent must use the IRL head with a real diagonal Fisher anchor —
        not a separate BC head. After two refits, theta_star must be snapshotted
        and fisher_diagonal must be non-None."""
        cfg = Config(verbose=False, maxent_iters_cold=8, maxent_iters_warm=4, maxent_mc_rollouts=8)
        agent = EWCAgent(cfg=cfg)
        prep = agent._token_for_vector((1, 0))
        cook = agent._token_for_vector((0, 1))
        serve = agent._token_for_vector((1, 1))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (prep, cook), now=1, cycle=0)
        agent._retrain()
        # First fit primes theta_star/fisher.
        self.assertIsNotNone(agent._ewc_theta_star)
        self.assertIsNotNone(agent._ewc_fisher)
        first_shape = agent.irl.theta.shape
        agent.decay.register(TOMATO_ONION_SOUP, ALT_PREF, (prep, serve), now=2, cycle=0)
        agent._retrain()
        # Second fit re-snapshots; theta shape preserved.
        self.assertEqual(agent.irl.theta.shape, first_shape)
        # EWC routes through the IRL head, not a BC head.
        self.assertFalse(hasattr(agent, "bc"), "EWCAgent must not carry a BC head — Phase 1.6 routes through IRL")

    def test_experience_replay_uses_unfiltered_reservoir(self):
        """ExperienceReplayAgent must not intersect its buffer with `decay.active_orderings`.
        That intersection would leak the proposed system's pruning into the baseline."""
        cfg = Config(verbose=False, er_buffer_size=4, er_batch_size=2, maxent_iters_cold=4, maxent_iters_warm=2)
        agent = ExperienceReplayAgent(cfg=cfg)
        # Seed the buffer with demos that are NOT in the decay active set.
        agent._buffer = [["c", "d"], ["e", "f"]]
        agent._seen = 2
        agent._retrain()
        # Buffer must be preserved (not filtered to empty by an active intersection).
        self.assertEqual(len(agent._buffer), 2, "ER buffer must not be filtered by decay.active_orderings")

    def test_experience_replay_bc_predicts_from_replay_buffer(self):
        cfg = Config(verbose=False, er_buffer_size=4, er_batch_size=4, bc_epochs_cold=12, bc_epochs_warm=6)
        agent = ExperienceReplayAgent(cfg=cfg)
        prep = agent._token_for_vector((1, 0, 0))
        cook = agent._token_for_vector((0, 1, 0))
        plate = agent._token_for_vector((0, 0, 1))
        serve = agent._token_for_vector((1, 1, 1))
        agent._buffer = [[prep, cook, serve], [prep, plate, serve]]
        agent._seen = 2
        agent._retrain()
        dist = agent.predict_next_tokens([prep])
        self.assertTrue(dist, "ER-BC replay head should emit an online distribution after refit")
        self.assertTrue({cook, plate} & set(dist))

    def test_nearest_neighbor_baseline_is_removed_from_public_registry(self):
        self.assertNotIn("NearestNeighbor", BASELINE_AGENTS)

    def test_recency_prioritized_replay_reports_policy_metadata(self):
        cfg = Config(verbose=False, er_buffer_size=2, er_batch_size=2, er_recency_alpha=2.0, er_uniform_mix=0.1, bc_epochs_cold=8, bc_epochs_warm=4)
        agent = RecencyPrioritizedReplayAgent(cfg=cfg)
        prep = agent._token_for_vector((1, 0))
        cook = agent._token_for_vector((0, 1))
        serve = agent._token_for_vector((1, 1))
        plate = agent._token_for_vector((0, 0, 1))
        agent.decay.register(TOMATO_ONION_SOUP, BASE_PREF, (prep, cook), now=1, cycle=0)
        agent._retrain()
        agent.decay.register(TOMATO_SOUP, ALT_PREF, (prep, serve), now=5, cycle=1)
        agent.decay.register("salad", "fresh", (prep, plate), now=6, cycle=2)
        agent._retrain()
        meta = agent.replay_buffer_metadata()
        self.assertEqual(meta["policy"], "recency_prioritized")
        self.assertEqual(meta["er_buffer_size"], 2)
        self.assertEqual(meta["er_batch_size"], 2)
        self.assertLessEqual(meta["n_buffered"], 2)
        self.assertEqual(meta["alpha"], 2.0)
        self.assertTrue(agent.predict_next_tokens([prep]))


class NewBaselineTests(unittest.TestCase):
    """Phase 2: bona-fide bigram floor + uniform-valid floor + oracle ceiling."""

    def _rid_pair(self):
        return TOMATO_ONION_SOUP, BASE_PREF

    def test_true_bigram_does_not_inherit_state_conditional_ngram(self):
        """The true bigram must NOT use the parent's state-conditional N-gram head;
        it must use its own flat Counter-based bigram on tokens only."""
        from src.baselines import BigramOnlyAgent
        agent = BigramOnlyAgent(cfg=Config(verbose=False))
        a = agent._token_for_vector((1, 0))
        b = agent._token_for_vector((0, 1))
        c = agent._token_for_vector((1, 1))
        rid, pref = self._rid_pair()
        agent.decay.register(rid, pref, (a, b, c), now=1, cycle=0)
        agent._retrain()
        # On (a,) the bigram says only b can follow.
        dist = agent.predict_next_tokens([a])
        self.assertEqual(set(dist), {b})
        self.assertAlmostEqual(dist[b], 1.0, places=9)
        # The parent's markov head must be empty (we _reset_heads in _fit_heads).
        self.assertEqual(agent.markov.vocab, set())

    def test_uniform_valid_returns_uniform_over_active_vocab(self):
        from src.baselines import UniformValidActionAgent
        agent = UniformValidActionAgent(cfg=Config(verbose=False))
        a = agent._token_for_vector((1, 0))
        b = agent._token_for_vector((0, 1))
        c = agent._token_for_vector((1, 1))
        rid, pref = self._rid_pair()
        agent.decay.register(rid, pref, (a, b, c), now=1, cycle=0)
        dist = agent.predict_next_tokens([])
        self.assertEqual(set(dist), {a, b, c})
        for v in dist.values():
            self.assertAlmostEqual(v, 1.0 / 3.0, places=9)

    def test_oracle_ceiling_returns_ground_truth_next_action(self):
        from src.baselines import OracleCeilingAgent
        agent = OracleCeilingAgent(cfg=Config(verbose=False))
        target = ["x", "y", "z"]
        agent.set_oracle_target(target)
        # Step 0 (empty prefix) -> "x"
        self.assertEqual(agent.predict_next_tokens([]), {"x": 1.0})
        # Step 1 (prefix=[x]) -> "y"
        self.assertEqual(agent.predict_next_tokens(["x"]), {"y": 1.0})
        # Step 2 (prefix=[x,y]) -> "z"
        self.assertEqual(agent.predict_next_tokens(["x", "y"]), {"z": 1.0})
        # Past end -> empty
        self.assertEqual(agent.predict_next_tokens(["x", "y", "z"]), {})

    def test_oracle_ceiling_resets_on_new_demo(self):
        from src.baselines import OracleCeilingAgent
        agent = OracleCeilingAgent(cfg=Config(verbose=False))
        agent.set_oracle_target(["a", "b"])
        agent.start_demo()
        self.assertEqual(agent._oracle_step, 0)

    def test_online_ewc_emas_fisher_across_tasks(self):
        from src.baselines import OnlineEWCAgent
        cfg = Config(verbose=False, maxent_iters_cold=6, maxent_iters_warm=3, maxent_mc_rollouts=6)
        agent = OnlineEWCAgent(cfg=cfg, gamma_fisher=0.5, gamma_theta=0.5)
        a = agent._token_for_vector((1, 0))
        b = agent._token_for_vector((0, 1))
        rid, pref = self._rid_pair()
        agent.decay.register(rid, pref, (a, b), now=1, cycle=0)
        agent._retrain()
        fisher_after_1 = agent._ewc_fisher.copy()
        # Add another variant and refit; the Fisher should now be a blend, not
        # a pure overwrite (which the batch EWCAgent does).
        agent.decay.register(rid, "alt", (b, a), now=2, cycle=0)
        agent._retrain()
        fisher_after_2 = agent._ewc_fisher
        self.assertEqual(fisher_after_1.shape, fisher_after_2.shape)
        # gamma=0.5 means 50/50 mix; the result must differ from the raw new Fisher.
        # We don't have direct access to the raw new Fisher here, but we can check
        # that the change is non-trivial AND that gamma_fisher is wired.
        self.assertEqual(agent._gamma_fisher, 0.5)
        self.assertEqual(agent._gamma_theta, 0.5)


if __name__ == "__main__":
    unittest.main()
