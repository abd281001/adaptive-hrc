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
        "maxent_mc_rollouts": 2,
        "bc_epochs_cold": 2,
        "bc_epochs_warm": 1,
    }
    values.update(overrides)
    return Config(**values)


def _fast_decay_config(**overrides):
    values = {
        "decay_init": 0.5,
        "decay_horizon_init": 2,
        "prune_threshold": 0.6,
        "size_rescale_exponent": 0.0,
        "max_global_rate": 1.0,
        "min_global_rate": 0.0,
        "max_decay_horizon": 0,
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


def _novel_recipe_requiring_observation(agent):
    min_len = max(int(agent.cfg.online_new_recipe_min_prefix) + 1, 6)
    for name, make_recipe in RECIPE_LIBRARY.items():
        if name == BASE_RECIPE_NAME:
            continue
        actions = make_recipe()
        tokens = _tokens_for(agent, actions)
        prefix = tokens[:min(len(tokens), min_len)]
        with agent.frozen():
            if agent._prefix_needs_observation(prefix):
                return name, actions, prefix
    raise AssertionError("no novel recipe produced an observation-required prefix")


class AdaptiveInvariantTests(unittest.TestCase):
    def test_latest_keys_invariant_over_stream(self):
        agent = AdaptiveHRCAgent(_fast_config(max_variants_per_recipe=4))
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

    def test_known_recipe_never_needs_observation(self):
        agent = AdaptiveHRCAgent(_fast_config(max_variants_per_recipe=16))
        observe_episode(agent, BASE_RECIPE)
        min_len = int(agent.cfg.online_new_recipe_min_prefix)

        for actions in _material_preference_variants(BASE_RECIPE):
            tokens = _tokens_for(agent, actions)
            with agent.frozen():
                for end in range(min_len, len(tokens) + 1):
                    prefix = tokens[:end]
                    self.assertFalse(
                        agent._prefix_needs_observation(prefix),
                        f"known recipe suppressed on prefix {prefix}",
                    )
            cls = run_online_episode(agent, actions, BASE_RECIPE_NAME)
            self.assertNotEqual(cls.kind, "needs_observation")
            assert_latest_pin_invariant(agent)

    def test_novel_recipe_requires_observation_with_nonempty_memory(self):
        agent = AdaptiveHRCAgent(_fast_config())
        cls0 = observe_episode(agent, BASE_RECIPE)
        known_rid = cls0.recipe_id
        before_variants = {
            rid: set(slot)
            for rid, slot in agent.memory.variants.items()
        }
        before_active = set(agent.decay.active)
        before_session = agent.session_counter

        novel_name, novel_actions, prefix = _novel_recipe_requiring_observation(agent)
        with agent.frozen():
            self.assertTrue(agent._prefix_needs_observation(prefix))

        cls = run_online_episode(agent, novel_actions, novel_name)
        self.assertEqual(cls.kind, "needs_observation")
        self.assertEqual(agent.session_counter, before_session)
        self.assertEqual({rid: set(slot) for rid, slot in agent.memory.variants.items()}, before_variants)
        self.assertEqual(set(agent.decay.active), before_active)
        self.assertEqual(set(agent.memory.variants), {known_rid})
        assert_latest_pin_invariant(agent)

    def test_pruned_variant_recognized_and_restored(self):
        agent = AdaptiveHRCAgent(_fast_decay_config(max_variants_per_recipe=2))
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

        cls = run_online_episode(agent, p1, BASE_RECIPE_NAME)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertIn(p1_key, agent.decay.active)
        self.assertNotIn(p1_key, agent.decay.pruned)
        self.assertAlmostEqual(agent.decay.active[p1_key].weight, 1.0)
        assert_latest_pin_invariant(agent)

    def test_variant_cap_does_not_destroy_archive(self):
        agent = AdaptiveHRCAgent(_fast_decay_config(max_variants_per_recipe=2))
        variants = _material_preference_variants(BASE_RECIPE)[:3]
        expected_hashes = []
        rid = None

        for actions in variants:
            cls = observe_episode(agent, actions)
            rid = rid or cls.recipe_id
            expected_hashes.append(_hash_for(agent, actions))

        self.assertEqual(len(set(expected_hashes)), 3)
        self.assertLessEqual(len(agent.decay.active_entries_for(rid)), 2)
        for variant_hash in expected_hashes:
            self.assertIn(variant_hash, agent.memory.variants[rid])
        assert_latest_pin_invariant(agent)

    def test_latest_pin_survives_cap_and_decay(self):
        agent = AdaptiveHRCAgent(_fast_decay_config(max_variants_per_recipe=2))
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
