import unittest

from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.memory import variant_hash
from src.models import Config
from src.preferences import PRESET_PREFERENCES, WorkflowPreferenceModifier
from src.representations import observations_from_actions


RECIPE_LIBRARY = gen.recipe_library()
BASE_RECIPE_NAME = "tomato_onion_soup_v1"
BASE_RECIPE = RECIPE_LIBRARY[BASE_RECIPE_NAME]()


def _fast_config(**overrides):
    values = {
        "verbose": False,
        "seed": 42,
        "maxent_iters_cold": 2,
        "maxent_iters_warm": 1,
    }
    values.update(overrides)
    return Config(**values)


def _fast_decay_config(**overrides):
    values = {
        "decay_horizon_init": 0,
        "decay_after_grace_steps": 1,
        "prune_threshold": 0.6,
    }
    values.update(overrides)
    return _fast_config(**values)


def _material_preference_variants(actions):
    modifier = WorkflowPreferenceModifier()
    variants = []
    seen = set()
    candidates = [list(actions)] + [
        modifier.modify_recipe(actions, pref)
        for pref in PRESET_PREFERENCES.values()
    ]
    for seq in candidates:
        key = tuple(seq)
        if key in seen:
            continue
        variants.append(list(seq))
        seen.add(key)
    return variants


def observe_episode(agent, actions):
    agent.start_demo()
    for obs in observations_from_actions(actions):
        agent.observe_observation(obs)
    return agent.end_demo()


def run_online_episode(agent, actions, ground_truth_recipe=BASE_RECIPE_NAME):
    for obs in observations_from_actions(actions):
        agent.observe_observation(obs, ground_truth_recipe=ground_truth_recipe)
    return agent.end_demo()


def assert_latest_pin_invariant(agent):
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    recipes = {rid for rid, slot in agent.memory.variants.items() if slot}
    require(set(agent.memory.latest) == recipes, "memory.latest must cover every recipe with variants")
    require(set(agent.decay.latest_by_recipe) == recipes, "decay.latest_by_recipe must match memory recipes")
    for rid in recipes:
        latest_hash = agent.memory.latest[rid]
        key = (rid, latest_hash)
        require(agent.decay.latest_by_recipe[rid] == latest_hash, f"{rid}: decay latest must equal memory latest")
        require(latest_hash in agent.memory.variants[rid], f"{rid}: latest hash missing from full registry")
        require(key in agent.decay.active, f"{rid}: latest key missing from active replay")
        require(key not in agent.decay.pruned, f"{rid}: latest key must not be pruned")
        require(agent.decay.active[key].weight == 1.0, f"{rid}: latest key must have unit weight")
    expected = {(rid, agent.memory.latest[rid]) for rid in recipes}
    require(agent.decay.latest_keys == expected, "decay.latest_keys must equal the memory latest set")
    agent._assert_latest_pin_invariant()


def _tokens_for(agent, actions):
    return agent._tokens_from_action_labels(actions)


def _hash_for(agent, actions):
    return variant_hash(_tokens_for(agent, actions))


def _advance_sessions(agent, n):
    for _ in range(n):
        agent.session_counter += 1
        agent.decay.step(agent.session_counter, agent.retrain_cycle)


def _advance_until_pruned(agent, key, max_sessions=50):
    if key in agent.decay.pruned:
        return
    for _ in range(max_sessions):
        agent.session_counter += 1
        agent.decay.step(agent.session_counter, agent.retrain_cycle)
        if key in agent.decay.pruned:
            agent.refresh_model_from_memory()
            return
    raise AssertionError(f"{key} did not prune within {max_sessions} sessions")




class AdaptiveInvariantTests(unittest.TestCase):
    def test_latest_keys_invariant_over_stream(self):
        agent = AdaptiveHRCAgent(_fast_config())
        soup_variants = _material_preference_variants(BASE_RECIPE)
        stream = [
            ("observe", soup_variants[0], BASE_RECIPE_NAME),
            ("online", soup_variants[1], BASE_RECIPE_NAME),
            ("observe", RECIPE_LIBRARY["tomato_soup"](), "tomato_soup"),
            ("online", soup_variants[2], BASE_RECIPE_NAME),
            ("online", soup_variants[0], BASE_RECIPE_NAME),
        ]

        for mode, actions, recipe_name in stream:
            if mode == "observe":
                observe_episode(agent, actions)
            else:
                run_online_episode(agent, actions, recipe_name)
            assert_latest_pin_invariant(agent)



    def test_pruned_variant_recognized_and_restored(self):
        agent = AdaptiveHRCAgent(_fast_decay_config())
        p1, p2 = _material_preference_variants(BASE_RECIPE)[:2]
        cls0 = observe_episode(agent, p1)
        rid = cls0.recipe_id
        p1_hash = _hash_for(agent, p1)
        observe_episode(agent, p2)
        p1_key = (rid, p1_hash)

        _advance_until_pruned(agent, p1_key)

        self.assertNotIn(p1_key, agent.decay.active)
        self.assertIn(p1_key, agent.decay.pruned)
        self.assertIn(p1_hash, agent.memory.variants[rid])

        # Membership removal—not weight drift—must rebuild the deployed heads
        # from active entries only. The full registry retains archived variants
        # solely for re-entry bookkeeping.
        agent.refresh_model_from_memory()
        audit = agent.pruned_influence_audit(max_prefixes=8)
        self.assertTrue(audit["passed"])

        cls = run_online_episode(agent, p1, BASE_RECIPE_NAME)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertIn(p1_key, agent.decay.active)
        self.assertNotIn(p1_key, agent.decay.pruned)
        self.assertAlmostEqual(agent.decay.active[p1_key].weight, 1.0)
        assert_latest_pin_invariant(agent)

    def test_multiple_variants_remain_active_before_temporal_decay(self):
        agent = AdaptiveHRCAgent(_fast_config())
        variants = _material_preference_variants(BASE_RECIPE)[:3]
        expected_hashes = []
        rid = None

        for actions in variants:
            cls = observe_episode(agent, actions)
            rid = rid or cls.recipe_id
            expected_hashes.append(_hash_for(agent, actions))

        self.assertEqual(len(set(expected_hashes)), 3)
        self.assertEqual(len(agent.decay.active_entries_for(rid)), 3)
        for variant_hash in expected_hashes:
            self.assertIn(variant_hash, agent.memory.variants[rid])
        assert_latest_pin_invariant(agent)

    def test_latest_pin_survives_decay(self):
        agent = AdaptiveHRCAgent(_fast_decay_config())
        variants = _material_preference_variants(BASE_RECIPE)[:3]
        rid = None
        latest_hash = None

        for actions in variants:
            cls = observe_episode(agent, actions)
            rid = rid or cls.recipe_id
            latest_hash = _hash_for(agent, actions)

        latest_key = (rid, latest_hash)
        _advance_sessions(agent, 50)

        self.assertIn(latest_key, agent.decay.latest_keys)
        self.assertIn(latest_key, agent.decay.active)
        self.assertAlmostEqual(agent.decay.active[latest_key].weight, 1.0)
        self.assertIn(latest_hash, agent.memory.variants[rid])
        assert_latest_pin_invariant(agent)


if __name__ == "__main__":
    unittest.main()
