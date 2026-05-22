"""Tests for the freeze primitive on AdaptiveHRCAgent."""
import sys
import os
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adaptive_agent import AdaptiveHRCAgent
from src.representations import observations_from_actions
from src.models import DEFAULT_CONFIG


TOMATO_ONION_SOUP = "tomato_onion_soup_v1"


def _make_agent():
    return AdaptiveHRCAgent(cfg=DEFAULT_CONFIG, narrate=lambda _: None)


def _train_one(agent, seq, rid=TOMATO_ONION_SOUP, observe=True):
    if observe:
        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()
    else:
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs, ground_truth_recipe=rid)
        agent.end_demo()


class TestFreezePrimitive(unittest.TestCase):

    def test_set_frozen_blocks_retrain(self):
        agent = _make_agent()
        from src.environment import gen
        rl = gen.recipe_library()
        seq = list(rl[list(rl.keys())[0]]())

        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        cycle_before = agent.retrain_cycle
        agent.set_frozen(True)

        # end_demo while frozen should not retrain
        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        self.assertEqual(agent.retrain_cycle, cycle_before, "retrain_cycle should not change while frozen")

    def test_set_frozen_blocks_decay_step(self):
        agent = _make_agent()
        from src.environment import gen
        rl = gen.recipe_library()
        seq = list(rl[list(rl.keys())[0]]())

        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        session_before = agent.session_counter
        agent.set_frozen(True)

        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        self.assertEqual(agent.session_counter, session_before, "session_counter should not advance while frozen")

    def test_frozen_context_manager_no_mutation(self):
        agent = _make_agent()
        from src.environment import gen
        rl = gen.recipe_library()
        seq = list(rl[list(rl.keys())[0]]())

        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        snap = (
            agent.session_counter,
            agent.retrain_cycle,
            len(agent.decay.active),
            len(agent.decay.pruned),
            len(agent.memory.variants),
        )

        with agent.frozen():
            # Evaluate — should not mutate
            acc = agent.evaluate_sequence(seq)
            # Calling observe_action while frozen (prediction only, no state change)
            for obs in observations_from_actions(seq[:3]):
                agent.observe_observation(obs)

        snap_after = (
            agent.session_counter,
            agent.retrain_cycle,
            len(agent.decay.active),
            len(agent.decay.pruned),
            len(agent.memory.variants),
        )
        self.assertEqual(snap, snap_after, "Agent state changed inside frozen() context")

    def test_frozen_rollback_on_accidental_mutation(self):
        """Under the deepcopy contract, mutation inside frozen() is rolled back on exit.
        The contract is restore-on-exit, not raise-on-exit."""
        agent = _make_agent()
        sc_before = agent.session_counter
        agent.set_frozen(True)
        # Directly mutate session_counter (bypass all gates)
        agent.session_counter += 999
        agent.set_frozen(False)
        self.assertEqual(agent.session_counter, sc_before, "Mutation inside frozen window must be rolled back on exit")

    def test_frozen_rollback_covers_predictor_heads(self):
        """Mutating IRL theta, n-gram counts, or the codebook inside frozen()
        must be rolled back on exit. The structural-summary snapshot did not catch these;
        the deepcopy contract does."""
        import numpy as np

        agent = _make_agent()
        from src.environment import gen
        rl = gen.recipe_library()
        seq = list(rl[list(rl.keys())[0]]())
        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        # Snapshot the heads BEFORE entering frozen.
        theta_before = agent.irl.theta.copy() if hasattr(agent.irl, "theta") and agent.irl.theta is not None else None
        codebook_before = dict(agent.action_vector_to_token)
        ngram_size_before = sum(len(v) for v in getattr(agent.markov, "counts_state", {}).values())

        with agent.frozen():
            # Aggressively mutate every predictor head.
            if theta_before is not None:
                agent.irl.theta = np.zeros_like(agent.irl.theta)
            agent.action_vector_to_token[("__bogus__",)] = "act_bogus"
            agent.token_to_action_vector["act_bogus"] = ("__bogus__",)
            if hasattr(agent.markov, "counts_state"):
                agent.markov.counts_state.setdefault("__bogus_state__", Counter())["__bogus__"] = 99

        if theta_before is not None:
            self.assertTrue(np.array_equal(agent.irl.theta, theta_before), "IRL theta must be restored after frozen()")
        self.assertEqual(agent.action_vector_to_token, codebook_before, "codebook must be restored after frozen()")
        ngram_size_after = sum(len(v) for v in getattr(agent.markov, "counts_state", {}).values())
        self.assertEqual(ngram_size_after, ngram_size_before, "n-gram counts must be restored after frozen()")

    def test_evaluate_sequence_always_safe(self):
        agent = _make_agent()
        from src.environment import gen
        rl = gen.recipe_library()
        seq = list(rl[list(rl.keys())[0]]())
        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        agent.end_demo()

        sc1 = agent.session_counter
        rc1 = agent.retrain_cycle
        _ = agent.evaluate_sequence(seq)
        self.assertEqual(agent.session_counter, sc1)
        self.assertEqual(agent.retrain_cycle, rc1)

    def test_frozen_flag_resets_on_context_exit(self):
        agent = _make_agent()
        with agent.frozen():
            self.assertTrue(agent._frozen)
        self.assertFalse(agent._frozen)


if __name__ == "__main__":
    unittest.main()
