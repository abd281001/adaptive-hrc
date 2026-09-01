"""Tests for the freeze primitive on AdaptiveAgent."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
