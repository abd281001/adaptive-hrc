"""Tests for the freeze primitive on AdaptiveAgent."""
import unittest

import numpy as np

from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.representations import observe_actions
from src.models import DEFAULT_SETTINGS

def _make_agent():
    return AdaptiveAgent(settings=DEFAULT_SETTINGS, narrate=lambda _: None)

def _first_recipe():
    return list(next(iter(recipe_builders().values()))())

class TestFreezePrimitive(unittest.TestCase):
    def test_set_frozen_blocks_retrain(self):
        agent = _make_agent()
        seq = _first_recipe()

        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        cycle_before = agent.retrain_cycle
        agent.set_frozen(True)

        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        self.assertEqual(agent.retrain_cycle, cycle_before, "retrain_cycle should not change while frozen")

    def test_set_frozen_blocks_decay_step(self):
        agent = _make_agent()
        seq = _first_recipe()

        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        demos_before = agent.demo_counter
        agent.set_frozen(True)

        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        self.assertEqual(agent.demo_counter, demos_before, "demo_counter should not advance while frozen")

    def test_frozen_context_manager_no_mutation(self):
        agent = _make_agent()
        seq = _first_recipe()

        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        snap = (
            agent.demo_counter,
            agent.retrain_cycle,
            len(agent.replay.active),
            len(agent.replay.pruned),
            len(agent.library.variants),
        )

        with agent.frozen():
            agent.evaluate(seq)
            for obs in observe_actions(seq[:3]):
                agent.observe(obs)

        snap_after = (
            agent.demo_counter,
            agent.retrain_cycle,
            len(agent.replay.active),
            len(agent.replay.pruned),
            len(agent.library.variants),
        )
        self.assertEqual(snap, snap_after, "Agent state changed inside frozen() context")

    def test_frozen_rollback_on_accidental_mutation(self):
        """Manual set_frozen() windows restore direct mutations on exit."""
        agent = _make_agent()
        demos_before = agent.demo_counter
        agent.set_frozen(True)
        agent.demo_counter += 999
        agent.set_frozen(False)
        self.assertEqual(agent.demo_counter, demos_before, "Mutation inside frozen window must be rolled back on exit")

    def test_frozen_context_reports_structural_mutation(self):
        agent = _make_agent()
        prefix_before = tuple(agent.current_prefix)
        with self.assertRaises(RuntimeError):
            with agent.frozen():
                agent.current_prefix.append("mutated_inside_frozen")
        self.assertEqual(tuple(agent.current_prefix), prefix_before)

    def test_frozen_rollback_covers_maxent_reward_state(self):
        """Frozen rollback covers the MaxEnt IRL reward state."""
        import numpy as np

        agent = _make_agent()
        seq = _first_recipe()
        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        weights_before = agent.maxent.reward_weights.copy() if hasattr(agent.maxent, "reward_weights") and agent.maxent.reward_weights is not None else None
        with agent.frozen():
            if weights_before is not None:
                agent.maxent.reward_weights = np.zeros_like(agent.maxent.reward_weights)

        if weights_before is not None:
            self.assertTrue(np.array_equal(agent.maxent.reward_weights, weights_before), "IRL reward_weights must be restored after frozen()")

    def test_evaluate_sequence_always_safe(self):
        agent = _make_agent()
        seq = _first_recipe()
        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        agent.end_demo()

        demos_before = agent.demo_counter
        rc1 = agent.retrain_cycle
        _ = agent.evaluate(seq)
        self.assertEqual(agent.demo_counter, demos_before)
        self.assertEqual(agent.retrain_cycle, rc1)

    def test_frozen_flag_resets_on_context_exit(self):
        agent = _make_agent()
        with agent.frozen():
            self.assertTrue(agent._frozen)
        self.assertFalse(agent._frozen)


class TestProbeLeavesNoTrace(unittest.TestCase):
    """A non-committing probe must leave the agent exactly as it found it.

    The easily-forgotten part is the random stream. `_tie_break_rng` decides
    every prediction tie, so a probe that leaves it advanced changes later
    predictions with nothing failing. Restoring "the state that matters" is
    therefore not a subset of fields -- it includes stream position -- and
    that is asserted here rather than left to emerge from blanket copying.
    """

    @staticmethod
    def _rng_state(agent):
        return repr(agent._tie_break_rng.bit_generator.state)

    def _trained_agent(self):
        agent = _make_agent()
        actions = _first_recipe()
        agent.start_demo()
        for observation in observe_actions(actions):
            agent.observe(observation)
        agent.end_demo()
        return agent, actions

    def test_a_probe_advances_the_tie_break_stream(self):
        """Guards the premise: if this stops being true the test below is vacuous."""
        agent, actions = self._trained_agent()
        before = self._rng_state(agent)
        prefix = []
        for observation in observe_actions(actions):
            distribution = agent.predict_actions(prefix)
            if distribution:
                agent.rank_actions(distribution, k=3)
            prefix.append(observation.action)
        self.assertNotEqual(before, self._rng_state(agent))

    def test_snapshot_restore_round_trip_restores_the_stream(self):
        agent, actions = self._trained_agent()
        checkpoint = agent.snapshot()
        before_rng = self._rng_state(agent)
        before_digest = agent._frozen_structural_digest()

        prefix = []
        for observation in observe_actions(actions):
            distribution = agent.predict_actions(prefix)
            if distribution:
                agent.rank_actions(distribution, k=3)
            agent.observe(observation)
            prefix.append(observation.action)

        agent.restore_from(checkpoint)
        self.assertEqual(before_rng, self._rng_state(agent))
        self.assertEqual(before_digest, agent._frozen_structural_digest())

    def test_repeated_restores_from_one_checkpoint_are_identical(self):
        """evaluate_frozen reuses a single checkpoint across many probes."""
        agent, actions = self._trained_agent()
        checkpoint = agent.snapshot()
        states = []
        for _ in range(3):
            prefix = []
            for observation in observe_actions(actions):
                distribution = agent.predict_actions(prefix)
                if distribution:
                    agent.rank_actions(distribution, k=3)
                agent.observe(observation)
                prefix.append(observation.action)
            agent.restore_from(checkpoint)
            states.append((self._rng_state(agent), agent._frozen_structural_digest()))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])

    def test_frozen_context_restores_the_stream(self):
        agent, actions = self._trained_agent()
        before = self._rng_state(agent)
        with agent.frozen():
            agent.evaluate_actions(actions, top_k=3)
        self.assertEqual(before, self._rng_state(agent))

    def test_capture_refuses_to_alias_a_live_generator(self):
        """The other half of the invariant: a capture must be independent.

        An aliased capture would make the restore check pass vacuously, so the
        two guards together cover both directions.
        """
        agent, _actions = self._trained_agent()
        captured = agent._capture_state()
        self.assertIsNot(captured["_tie_break_rng"], agent._tie_break_rng)
        self.assertIsNot(captured["_init_rng"], agent._init_rng)

    def test_restore_detects_an_unrestored_stream(self):
        """A subset restore that forgets the stream must fail loudly."""
        agent, _actions = self._trained_agent()
        state = agent._capture_state()
        # Simulate exactly the mistake the invariant exists to catch: the
        # caller restores everything except the tie-break generator.
        del state["_tie_break_rng"]
        agent._tie_break_rng.standard_normal(8)
        with self.assertRaisesRegex(RuntimeError, "random stream"):
            agent._apply_state(state)


if __name__ == "__main__":
    unittest.main()
