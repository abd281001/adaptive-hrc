"""Contracts for the runnable publication comparison baselines."""
from __future__ import annotations

import unittest

from src.baselines import (
    AdaptiveDecayAgent,
    BASELINE_AGENTS,
    BehaviorCloningAgent,
    BigramOnlyAgent,
    EWCAgent,
    ExperienceReplayAgent,
    FixedDecayAgent,
    LatestOnlyPreferenceAgent,
    NoDecayAgent,
    OfflinePretrainedFrozenAgent,
)
from src.models import Config
from src.representations import ActionObservation


def _fast_config(**overrides):
    values = {
        "verbose": False,
        "maxent_iters_cold": 2,
        "maxent_iters_warm": 1,
        "bc_epochs_cold": 2,
        "bc_epochs_warm": 1,
    }
    values.update(overrides)
    return Config(**values)


class RunnableBaselineTests(unittest.TestCase):
    def test_public_registry_contains_exactly_the_runnable_baselines(self):
        self.assertEqual(
            set(BASELINE_AGENTS),
            {
                "offline_pretrained_frozen",
                "offline_all_recipes_identity_frozen",
                "adaptive_decay",
                "latest_only",
                "fixed_decay",
                "no_decay",
                "bc",
                "ewc",
                "experience_replay_bc",
                "bigram",
            },
        )
        self.assertIs(BASELINE_AGENTS["adaptive_decay"], AdaptiveDecayAgent)
        self.assertIs(BASELINE_AGENTS["offline_pretrained_frozen"], OfflinePretrainedFrozenAgent)
        self.assertIs(BASELINE_AGENTS["offline_all_recipes_identity_frozen"], OfflinePretrainedFrozenAgent)

    def test_all_runnable_controls_disable_fulls_latest_variant_pin(self):
        continual_controls = {
            name: agent_type
            for name, agent_type in BASELINE_AGENTS.items()
            if not name.startswith("offline_")
        }
        for agent_type in continual_controls.values():
            agent = agent_type(_fast_config())
            agent.decay.register("recipe", "variant", ("a",), now=0, cycle=0)
            self.assertFalse(agent.cfg.protect_latest_preference, agent_type.__name__)
            self.assertEqual(agent.decay.latest_keys, set(), agent_type.__name__)

    def test_offline_pretrained_frozen_agent_has_no_deployment_learning_path(self):
        agent = OfflinePretrainedFrozenAgent(_fast_config())
        known = agent._token_for_vector((1, 0))
        agent.decay.register("recipe", "variant", (known,), now=0, cycle=0)
        agent._retrain()
        metadata = agent.lock_deployment()

        before_codebook = dict(agent.action_vector_to_token)
        before_active = {
            key: (entry.ordering, entry.weight, entry.last_seen_step)
            for key, entry in agent.decay.active.items()
        }
        before_retrains = list(agent.retrain_events)
        before_cycle = agent.retrain_cycle
        before_step_counter = agent.step_counter
        before_session_counter = agent.session_counter
        before_theta = agent.irl.theta.tolist() if agent.irl.theta is not None else None

        agent.start_demo()
        agent.observe_observation(ActionObservation((0,), (1, 0), (1,)))
        agent.observe_observation(ActionObservation((1,), (0, 1), (0,)))
        cls = agent.end_demo()

        self.assertEqual(cls.kind, "offline_frozen")
        self.assertFalse(metadata["deployment_updates_allowed"])
        self.assertEqual(agent.action_vector_to_token, before_codebook)
        self.assertEqual(
            {
                key: (entry.ordering, entry.weight, entry.last_seen_step)
                for key, entry in agent.decay.active.items()
            },
            before_active,
        )
        self.assertEqual(agent.retrain_events, before_retrains)
        self.assertEqual(agent.retrain_cycle, before_cycle)
        self.assertEqual(agent.step_counter, before_step_counter)
        self.assertEqual(agent.session_counter, before_session_counter)
        self.assertEqual(agent.irl.theta.tolist() if agent.irl.theta is not None else None, before_theta)
        self.assertEqual(agent.current_prefix, [])

    def test_latest_only_keeps_one_variant_per_recipe(self):
        agent = LatestOnlyPreferenceAgent(_fast_config())
        agent._register_if_live("recipe", ["a", "b"], step=1)
        agent._register_if_live("recipe", ["b", "a"], step=2)

        self.assertEqual(len(agent.memory.variants["recipe"]), 1)
        self.assertEqual(len(agent.decay.active_entries()), 1)

    def test_fixed_decay_prunes_at_the_configured_fixed_rate(self):
        agent = FixedDecayAgent(_fast_config(decay_init=0.5))
        agent.decay.register("recipe", "variant", ("a",), now=0, cycle=0)
        agent.decay.step(now=1, cycle=1)
        agent.decay.step(now=2, cycle=2)

        self.assertIn(("recipe", "variant"), agent.decay.pruned)

    def test_no_decay_never_prunes(self):
        agent = NoDecayAgent(_fast_config())
        agent.decay.register("recipe", "variant", ("a",), now=0, cycle=0)
        self.assertEqual(agent.decay.step(now=10_000, cycle=1), [])
        self.assertIn(("recipe", "variant"), agent.decay.active)

    def test_behavior_cloning_emits_a_distribution_after_training(self):
        agent = BehaviorCloningAgent(_fast_config())
        first = agent._token_for_vector((1, 0))
        second = agent._token_for_vector((0, 1))
        agent.decay.register("recipe", "variant", (first, second), now=0, cycle=0)
        agent._retrain()

        self.assertTrue(agent.predict_next_tokens([first]))

    def test_ewc_records_a_fisher_anchor(self):
        agent = EWCAgent(_fast_config())
        first = agent._token_for_vector((1, 0))
        second = agent._token_for_vector((0, 1))
        agent.decay.register("recipe", "variant", (first, second), now=0, cycle=0)
        agent._retrain()

        self.assertIsNotNone(agent._ewc_theta_star)
        self.assertIsNotNone(agent._ewc_fisher)

    def test_experience_replay_trains_from_its_reservoir(self):
        agent = ExperienceReplayAgent(_fast_config(er_buffer_size=4, er_batch_size=4))
        first = agent._token_for_vector((1, 0))
        second = agent._token_for_vector((0, 1))
        agent._record_committed_replay_demo(
            "recipe", "variant", (first, second), session_step=1, action_step=1, source_mode="observe"
        )
        agent._retrain()

        self.assertTrue(agent.predict_next_tokens([first]))
        self.assertEqual(agent.replay_buffer_metadata()["n_buffered"], 1)

    def test_bigram_ignores_the_parent_irl_and_markov_heads(self):
        agent = BigramOnlyAgent(_fast_config())
        first = agent._token_for_vector((1, 0))
        second = agent._token_for_vector((0, 1))
        agent.decay.register("recipe", "variant", (first, second), now=0, cycle=0)
        agent._retrain()

        self.assertEqual(agent.predict_next_tokens([first]), {second: 1.0})
        self.assertEqual(agent.markov.vocab, set())


if __name__ == "__main__":
    unittest.main()
