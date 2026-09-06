"""Contracts for the runnable publication comparison baselines."""
from __future__ import annotations

import unittest

from src.baselines import (
    UnpinnedAgent,
    BehaviorCloner,
    BASELINE_AGENTS,
    BehaviorCloningAgent,
    MemoryMatchedBcAgent,
    EwcAgent,
    ReplayBcAgent,
    FixedDecayAgent,
    LatestAgent,
    NoDecayAgent,
    FrozenAgent,
)
from src.models import Settings
from src.representations import Observation, observe_actions
from src.llm_baseline import InContextLlmAgent


def _trace(*actions):
    """Observed transitions for `actions`, as the live protocol records them.

    Replay items must carry the transitions that were actually observed; the
    learner refuses to reconstruct states from action labels alone.
    """
    return tuple(
        (observation.state, observation.action, observation.next_state)
        for observation in observe_actions(actions)
    )


def _fast_config(**overrides):
    values = {
        "verbose": False,
        "irl_cold_steps": 2,
        "irl_warm_steps": 1,
        "bc_cold_epochs": 2,
        "bc_warm_epochs": 1,
    }
    values.update(overrides)
    return Settings(**values)


class RunnableBaselineTests(unittest.TestCase):
    def test_public_registry_contains_exactly_the_runnable_baselines(self):
        self.assertEqual(
            set(BASELINE_AGENTS),
            {
                "frozen",
                "offline_default",
                "unpinned",
                "latest",
                "fixed",
                "no_decay",
                "bc",
                "bc_adaptive",
                "ewc",
                "replay_bc",
                "in_context_llm",
            },
        )
        self.assertIs(BASELINE_AGENTS["bc"], BehaviorCloningAgent)
        self.assertIs(BASELINE_AGENTS["bc_adaptive"], MemoryMatchedBcAgent)
        self.assertIs(BASELINE_AGENTS["unpinned"], UnpinnedAgent)
        self.assertIs(BASELINE_AGENTS["frozen"], FrozenAgent)
        self.assertIs(BASELINE_AGENTS["offline_default"], FrozenAgent)
        self.assertIs(BASELINE_AGENTS["in_context_llm"], InContextLlmAgent)

    def test_publication_baselines_disable_fulls_latest_variant_pin(self):
        continual_controls = {
            name: agent_type
            for name, agent_type in BASELINE_AGENTS.items()
            if name in {
                "unpinned", "latest", "fixed", "no_decay",
                "bc", "ewc", "replay_bc",
            }
        }
        for agent_type in continual_controls.values():
            agent = agent_type(_fast_config())
            agent.replay.register("recipe", "variant", ("a",), now=0, cycle=0)
            self.assertFalse(agent.settings.pin_latest, agent_type.__name__)
            self.assertEqual(agent.replay.latest_keys, set(), agent_type.__name__)

    def test_architecture_baselines_use_no_decay_without_full_components(self):
        for agent_type in (
            BehaviorCloningAgent, ReplayBcAgent, EwcAgent,
        ):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(_fast_config())
                agent.replay.register(
                    "recipe", "variant", ("a",), now=0, cycle=0,
                )
                self.assertEqual(agent.replay.policy, "none")
                self.assertFalse(agent.settings.pin_latest)
                self.assertFalse(agent.settings.semantic_fallback_enabled)
                self.assertFalse(agent.settings.latent_strategy_enabled)
                self.assertEqual(agent.replay.latest_keys, set())

    def test_distinct_policy_references_are_unchanged(self):
        for agent_type in (FrozenAgent, InContextLlmAgent):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(_fast_config())
                self.assertEqual(agent.replay.policy, "adaptive")
                self.assertTrue(agent.settings.pin_latest)
                self.assertTrue(agent.settings.semantic_fallback_enabled)
                self.assertTrue(agent.settings.latent_strategy_enabled)

    def test_offline_pretrained_frozen_agent_has_no_deployment_learning_path(self):
        agent = FrozenAgent(_fast_config())
        known = "transfer (pot, from=storage, to=cooking_station)"
        agent.replay.register("recipe", "variant", (known,), now=0, cycle=0, transitions=_trace(known))
        agent._retrain()
        metadata = agent.lock_deployment()

        before_active = {
            key: (entry.ordering, entry.weight, entry.last_seen_step)
            for key, entry in agent.replay.active.items()
        }
        before_retrains = list(agent.retrain_events)
        before_cycle = agent.retrain_cycle
        before_step_counter = agent.step_counter
        before_demo_counter = agent.demo_counter
        weights_before = agent.maxent.reward_weights.tolist() if agent.maxent.reward_weights is not None else None

        agent.start_demo()
        agent.observe(Observation(
            (0,), "transfer (pan, from=storage, to=cooking_station)", (1,),
        ))
        agent.observe(Observation(
            (1,), "turn_on (stove, cooking_station)", (0,),
        ))
        cls = agent.end_demo()

        self.assertEqual(cls.kind, "offline_frozen")
        self.assertFalse(metadata["updates_allowed"])
        self.assertEqual(
            {
                key: (entry.ordering, entry.weight, entry.last_seen_step)
                for key, entry in agent.replay.active.items()
            },
            before_active,
        )
        self.assertEqual(agent.retrain_events, before_retrains)
        self.assertEqual(agent.retrain_cycle, before_cycle)
        self.assertEqual(agent.step_counter, before_step_counter)
        self.assertEqual(agent.demo_counter, before_demo_counter)
        self.assertEqual(agent.maxent.reward_weights.tolist() if agent.maxent.reward_weights is not None else None, weights_before)
        self.assertEqual(agent.current_prefix, [])

    def test_latest_only_keeps_one_variant_per_recipe(self):
        agent = LatestAgent(_fast_config())
        agent._register_if_live("recipe", ["a", "b"], step=1)
        agent._register_if_live("recipe", ["b", "a"], step=2)

        self.assertEqual(len(agent.library.variants["recipe"]), 1)
        self.assertEqual(len(agent.replay.active_items()), 1)

    def test_fixed_decay_prunes_at_the_configured_fixed_rate(self):
        agent = FixedDecayAgent(_fast_config(fixed_decay=0.5))
        agent.replay.register("recipe", "variant", ("a",), now=0, cycle=0)
        agent.replay.step(now=1, cycle=1)
        agent.replay.step(now=2, cycle=2)

        self.assertIn(("recipe", "variant"), agent.replay.pruned)

    def test_no_decay_never_prunes(self):
        agent = NoDecayAgent(_fast_config())
        agent.replay.register("recipe", "variant", ("a",), now=0, cycle=0)
        self.assertEqual(agent.replay.step(now=10_000, cycle=1), [])
        self.assertIn(("recipe", "variant"), agent.replay.active)

    def test_architecture_baselines_retain_unit_weight_without_decay(self):
        for agent_type in (
            BehaviorCloningAgent,
            ReplayBcAgent,
            EwcAgent,
        ):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(_fast_config())
                key = ("recipe", "variant")
                agent.replay.register(*key, ("a",), now=0, cycle=0)

                self.assertEqual(agent.replay.policy, "none")
                self.assertEqual(agent.replay.step(now=10_000, cycle=1), [])
                self.assertIn(key, agent.replay.active)
                self.assertNotIn(key, agent.replay.pruned)
                self.assertEqual(agent.replay.active[key].weight, 1.0)

    def test_memory_policy_ablation_agents_remain_distinct(self):
        self.assertEqual(UnpinnedAgent(_fast_config()).replay.policy, "adaptive")
        self.assertEqual(LatestAgent(_fast_config()).replay.policy, "adaptive")
        self.assertEqual(FixedDecayAgent(_fast_config()).replay.policy, "fixed")
        self.assertEqual(NoDecayAgent(_fast_config()).replay.policy, "none")
        self.assertEqual(BehaviorCloningAgent(_fast_config()).replay.policy, "none")
        self.assertEqual(EwcAgent(_fast_config()).replay.policy, "none")
        self.assertEqual(ReplayBcAgent(_fast_config()).replay.policy, "none")
        self.assertEqual(
            MemoryMatchedBcAgent(_fast_config()).replay.policy, "adaptive",
        )

    def test_irl_memory_controls_disable_both_semantic_components(self):
        for agent_type in (
            UnpinnedAgent, LatestAgent, FixedDecayAgent, NoDecayAgent, EwcAgent,
        ):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(_fast_config())
                self.assertFalse(agent.settings.semantic_fallback_enabled)
                self.assertFalse(agent.settings.latent_strategy_enabled)

    def test_baselines_report_the_predictor_they_actually_execute(self):
        expected = {
            BehaviorCloningAgent: ("behavior_cloning", "not_applicable"),
            ReplayBcAgent: (
                "experience_replay_behavior_cloning", "not_applicable",
            ),
            MemoryMatchedBcAgent: (
                "behavior_cloning_adaptive_memory", "not_applicable",
            ),
            EwcAgent: ("ewc_maxent", "engineered"),
            FrozenAgent: ("maxent", "engineered"),
        }
        for agent_type, (predictor, irl_features) in expected.items():
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(_fast_config())
                self.assertEqual(agent.predictor_name(), predictor)
                self.assertEqual(agent.irl_feature_name(), irl_features)

    def test_shared_baselines_use_threshold_three_without_changing_fixed_pin(self):
        for agent_type in (
            UnpinnedAgent,
            FixedDecayAgent,
            NoDecayAgent,
            BehaviorCloningAgent,
            EwcAgent,
            ReplayBcAgent,
        ):
            with self.subTest(agent=agent_type.__name__):
                self.assertEqual(agent_type(_fast_config()).retrain_policy.cold_after, 3)

        self.assertFalse(FixedDecayAgent(_fast_config()).settings.pin_latest)
        self.assertFalse(BehaviorCloningAgent(_fast_config()).settings.pin_latest)

    def test_memory_matched_bc_isolates_the_predictor_from_the_memory_policy(self):
        """``bc`` and ``bc_adaptive`` may differ only in how storage behaves.

        The pair is only interpretable as a predictor-vs-memory decomposition
        if the retention policy is the sole difference between this arm and
        Full, and the model family is the sole difference between this arm and
        ``bc``.  Both halves are asserted here so a later settings change
        cannot quietly reintroduce a second confound.
        """
        matched = MemoryMatchedBcAgent(_fast_config())
        plain = BehaviorCloningAgent(_fast_config())

        # Full's memory policy, restored.
        self.assertEqual(matched.replay.policy, "adaptive")
        self.assertTrue(matched.settings.pin_latest)

        # Still the cloner, not MaxEnt: the semantic components are reached
        # only through the MaxEnt predictor, so they stay off in both arms.
        self.assertIsInstance(matched.cloner, BehaviorCloner)
        self.assertEqual(matched.predictor_name(), plain.predictor_name() + "_adaptive_memory")
        self.assertEqual(matched.irl_feature_name(), "not_applicable")
        for agent in (matched, plain):
            self.assertFalse(agent.settings.semantic_fallback_enabled)
            self.assertFalse(agent.settings.latent_strategy_enabled)

        # Everything the cloner is fitted with is unchanged.
        self.assertEqual(
            {
                field: getattr(matched.settings, field)
                for field in vars(matched.settings)
                if field.startswith("bc_")
            },
            {
                field: getattr(plain.settings, field)
                for field in vars(plain.settings)
                if field.startswith("bc_")
            },
        )
        self.assertEqual(
            matched.retrain_policy.cold_after, plain.retrain_policy.cold_after,
        )

    def test_memory_matched_bc_decays_and_pins_the_way_full_does(self):
        first = "transfer (pot, from=storage, to=cooking_station)"
        second = "turn_on (stove, cooking_station)"
        matched = MemoryMatchedBcAgent(_fast_config())
        matched._register_if_live("recipe", [first, second], step=1)

        # Pinned, so the latest variant is protected from the decay above.
        self.assertEqual(len(matched.replay.latest_keys), 1)
        key = next(iter(matched.replay.latest_keys))
        self.assertIn(key, matched.replay.active)
        self.assertEqual(matched.replay.step(now=10_000, cycle=1), [])
        self.assertIn(key, matched.replay.active)

        # Unpinned entries still decay, unlike ``bc``'s nondecaying storage.
        matched.replay.register(
            "other", "stale", (first,), now=0, cycle=0, pin_latest=False,
        )
        for step in range(1, 200):
            matched.replay.step(now=10_000 + step, cycle=1 + step)
        self.assertIn(("other", "stale"), matched.replay.pruned)
        self.assertIn(key, matched.replay.active)

    def test_er_bc_default_reservoir_is_bounded_conservatively(self):
        agent = ReplayBcAgent(_fast_config())
        self.assertEqual(agent.settings.replay_capacity, 64)
        self.assertEqual(agent.settings.replay_batch, 64)

    def test_behavior_cloning_emits_a_distribution_after_training(self):
        agent = BehaviorCloningAgent(_fast_config())
        first = "transfer (pot, from=storage, to=cooking_station)"
        second = "turn_on (stove, cooking_station)"
        agent.replay.register("recipe", "variant", (first, second), now=0, cycle=0, transitions=_trace(first, second))
        agent._retrain()

        distribution = agent.predict_actions([first])
        self.assertEqual(set(distribution), {second})
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=6)
        self.assertTrue(
            agent.policy_stats()["action_mask_shared_across_predictors"]
        )
        self.assertFalse(
            agent.policy_stats()["action_mask_uses_recipe_hypothesis"]
        )

    def test_ewc_records_a_fisher_anchor(self):
        agent = EwcAgent(_fast_config())
        first = "transfer (pot, from=storage, to=cooking_station)"
        second = "turn_on (stove, cooking_station)"
        agent.replay.register("recipe", "variant", (first, second), now=0, cycle=0, transitions=_trace(first, second))
        agent._retrain()

        self.assertIsNotNone(agent._anchor)
        self.assertIsNotNone(agent._fisher)

    def test_experience_replay_trains_from_its_reservoir(self):
        agent = ReplayBcAgent(_fast_config(replay_capacity=4, replay_batch=4))
        first = "transfer (pot, from=storage, to=cooking_station)"
        second = "turn_on (stove, cooking_station)"
        agent.replay.register(
            "recipe", "variant", (first, second), now=0, cycle=0,
            transitions=_trace(first, second),
        )
        agent._record_demo(
            "recipe", "variant", (first, second), transitions=_trace(first, second),
            demo_step=1, action_step=1, source_mode="observe"
        )
        agent._retrain()

        self.assertTrue(agent.predict_actions([first]))
        self.assertEqual(agent.replay_stats()["n_buffered"], 1)

if __name__ == "__main__":
    unittest.main()
