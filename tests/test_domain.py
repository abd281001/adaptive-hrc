import unittest

import numpy as np

from src.adaptive_agent import AdaptiveAgent
from src.domain import DomainAdapter, SymbolicDomainAdapter, default_domain
from src.environment import StateTracker
from src.models import MaxEntIrl, Settings, build_features, feasible_actions
from src.representations import observe_actions


ACTIONS = (
    "transfer (pot, from=storage, to=cooking_station)",
    "turn_on (stove, cooking_station)",
)


def _trajectory():
    observations = observe_actions(ACTIONS)
    return [
        *((observation.state, observation.action) for observation in observations),
        (observations[-1].next_state, "stop"),
    ]


class DomainBoundaryTests(unittest.TestCase):
    def test_symbolic_adapter_is_the_default_protocol_implementation(self):
        self.assertIsInstance(default_domain(), SymbolicDomainAdapter)
        self.assertIsInstance(default_domain(), DomainAdapter)

    def test_symbolic_replay_matches_legacy_observation_states_exactly(self):
        adapter = SymbolicDomainAdapter()
        observations = observe_actions(ACTIONS)
        state = adapter.initial_state()
        self.assertEqual(state, observations[0].state)
        for observation in observations:
            state = adapter.replay_transition(state, observation.action)
            self.assertEqual(state, observation.next_state)
        self.assertEqual(adapter.state_from_actions(ACTIONS), state)

    def test_symbolic_features_and_legality_match_legacy_functions(self):
        adapter = SymbolicDomainAdapter()
        state = adapter.initial_state()
        legacy = build_features({0: state}, feature_mode="semantic")
        adapted = adapter.build_features({0: state}, feature_mode="semantic")
        for legacy_value, adapted_value in zip(legacy, adapted):
            np.testing.assert_array_equal(legacy_value, adapted_value)
        candidates = ACTIONS + ("stop",)
        self.assertEqual(
            adapter.legal_actions(state, candidates),
            feasible_actions(state, candidates),
        )
        reward = adapter.reward_features({0: state})
        semantic_legacy = build_features(
            {0: state}, feature_mode="semantic", normalize=False,
        )
        semantic = adapter.semantic_features({0: state})
        for expected, actual in zip(semantic_legacy, semantic):
            np.testing.assert_array_equal(expected, actual)
        engineered = build_features({0: state}, feature_mode="engineered")
        for expected, actual in zip(engineered, reward):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(adapter.canonical_action(ACTIONS[0]), ACTIONS[0])

    def test_explicit_symbolic_adapter_preserves_seeded_maxent_policy(self):
        settings = Settings(
            verbose=False,
            irl_cold_steps=3,
            irl_warm_steps=2,
            irl_horizon=8,
        )
        demonstration = _trajectory()
        default_model = MaxEntIrl(settings)
        explicit_model = MaxEntIrl(
            Settings(
                verbose=False,
                irl_cold_steps=3,
                irl_warm_steps=2,
                irl_horizon=8,
            ),
            domain=SymbolicDomainAdapter(),
        )
        default_model.fit([demonstration], [1.0])
        explicit_model.fit([demonstration], [1.0])
        np.testing.assert_array_equal(
            default_model.reward_weights, explicit_model.reward_weights,
        )
        self.assertEqual(
            default_model.predict(demonstration[0][0]),
            explicit_model.predict(demonstration[0][0]),
        )

    def test_agent_accepts_observed_state_without_prefix_reconstruction(self):
        class CountingDomain(SymbolicDomainAdapter):
            def __init__(self):
                self.state_key_calls = 0

            def state_key(self, state, *, actor_id=0):
                self.state_key_calls += 1
                return super().state_key(state, actor_id=actor_id)

        domain = CountingDomain()
        agent = AdaptiveAgent(Settings(verbose=False), domain=domain)
        state = StateTracker().get_state_vector()
        self.assertEqual(agent.predict_actions(state=state), {})
        self.assertEqual(domain.state_key_calls, 1)
