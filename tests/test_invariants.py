import unittest

from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.memory import make_variant_id
from src.models import Settings
from src.preferences import PREFERENCES, apply_preference
from src.representations import observe_actions


RECIPE_LIBRARY = recipe_builders()
BASE_RECIPE_NAME = "tomato_onion_soup"
BASE_RECIPE = RECIPE_LIBRARY[BASE_RECIPE_NAME]()


def _fast_config(**overrides):
    values = {
        "verbose": False,
        "seed": 42,
        "irl_cold_steps": 2,
        "irl_warm_steps": 1,
    }
    values.update(overrides)
    return Settings(**values)


def _fast_decay_config(**overrides):
    values = {
        "initial_grace": 0,
        "prune_delay": 1,
        "prune_threshold": 0.6,
    }
    values.update(overrides)
    return _fast_config(**values)


def _material_preference_variants(actions):
    variants = []
    seen = set()
    candidates = [list(actions)] + [
        apply_preference(actions, pref).actions
        for pref in PREFERENCES.values()
    ]
    for seq in candidates:
        key = tuple(seq)
        if key in seen:
            continue
        variants.append(list(seq))
        seen.add(key)
    return variants


def observe_demo(agent, actions):
    agent.start_demo()
    for obs in observe_actions(actions):
        agent.observe(obs)
    return agent.end_demo()


def run_online_episode(agent, actions, ground_truth_recipe=BASE_RECIPE_NAME):
    for obs in observe_actions(actions):
        agent.observe(obs, ground_truth_recipe=ground_truth_recipe)
    return agent.end_demo()


def assert_latest_pin_invariant(agent):
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    recipes = {recipe_id for recipe_id, slot in agent.library.variants.items() if slot}
    require(set(agent.library.latest) == recipes, "memory.latest must cover every recipe with variants")
    require(set(agent.replay.latest_by_recipe) == recipes, "decay.latest_by_recipe must match memory recipes")
    for recipe_id in recipes:
        latest_hash = agent.library.latest[recipe_id]
        key = (recipe_id, latest_hash)
        require(agent.replay.latest_by_recipe[recipe_id] == latest_hash, f"{recipe_id}: decay latest must equal memory latest")
        require(latest_hash in agent.library.variants[recipe_id], f"{recipe_id}: latest hash missing from full registry")
        require(key in agent.replay.active, f"{recipe_id}: latest key missing from active replay")
        require(key not in agent.replay.pruned, f"{recipe_id}: latest key must not be pruned")
        require(agent.replay.active[key].weight == 1.0, f"{recipe_id}: latest key must have unit weight")
    expected = {(recipe_id, agent.library.latest[recipe_id]) for recipe_id in recipes}
    require(agent.replay.latest_keys == expected, "decay.latest_keys must equal the memory latest set")
    agent._check_latest()


def _hash_for(actions):
    return make_variant_id(actions)


def _advance_demos(agent, n):
    for _ in range(n):
        agent.demo_counter += 1
        agent.replay.step(agent.demo_counter, agent.retrain_cycle)


def _advance_until_pruned(agent, key, max_demos=50):
    if key in agent.replay.pruned:
        return
    for _ in range(max_demos):
        agent.demo_counter += 1
        agent.replay.step(agent.demo_counter, agent.retrain_cycle)
        if key in agent.replay.pruned:
            agent.refresh()
            return
    raise AssertionError(f"{key} did not prune within {max_demos} demonstrations")

class AdaptiveInvariantTests(unittest.TestCase):
    def test_latest_keys_invariant_over_stream(self):
        agent = AdaptiveAgent(_fast_config())
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
                observe_demo(agent, actions)
            else:
                run_online_episode(agent, actions, recipe_name)
            assert_latest_pin_invariant(agent)

    def test_pruned_variant_recognized_and_restored(self):
        agent = AdaptiveAgent(_fast_decay_config())
        first_actions, second_actions = _material_preference_variants(
            BASE_RECIPE,
        )[:2]
        cls0 = observe_demo(agent, first_actions)
        recipe_id = cls0.recipe_id
        first_variant_id = _hash_for(first_actions)
        observe_demo(agent, second_actions)
        first_key = (recipe_id, first_variant_id)

        _advance_until_pruned(agent, first_key)

        self.assertNotIn(first_key, agent.replay.active)
        self.assertIn(first_key, agent.replay.pruned)
        self.assertIn(first_variant_id, agent.library.variants[recipe_id])

        # Membership removal rebuilds predictors; the registry only supports reentry.
        agent.refresh()
        audit = agent.audit_pruning(max_prefixes=8)
        # The deployed model must reflect active memory only, which is the
        # path-dependence term: after a refresh it has no history to carry.
        self.assertLessEqual(
            audit["deployed_path_dependence_max_l1"], audit["tolerance"],
        )
        # The pruned variant is visible to the audit as a counterfactual.
        self.assertTrue(audit["pruned_available"])
        self.assertGreaterEqual(audit["n_pruned_variants"], 1)
        # The membership contract is the only pass/fail term, and it holds:
        # the fit received no pruned record.
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["active_only_training_inputs_verified"])
        # Redundancy is reported as a magnitude, not a verdict. These two
        # variants are deliberately materially different, so restoring the
        # pruned one moves the policy: the retention decision was not free.
        self.assertGreater(audit["redundancy_max_l1"], audit["tolerance"])

        cls = run_online_episode(agent, first_actions, BASE_RECIPE_NAME)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertIn(first_key, agent.replay.active)
        self.assertNotIn(first_key, agent.replay.pruned)
        self.assertAlmostEqual(agent.replay.active[first_key].weight, 1.0)
        assert_latest_pin_invariant(agent)

    def test_multiple_variants_remain_active_before_temporal_decay(self):
        agent = AdaptiveAgent(_fast_config())
        variants = _material_preference_variants(BASE_RECIPE)[:3]
        expected_hashes = []
        recipe_id = None

        for actions in variants:
            cls = observe_demo(agent, actions)
            recipe_id = recipe_id or cls.recipe_id
            expected_hashes.append(_hash_for(actions))

        self.assertEqual(len(set(expected_hashes)), 3)
        self.assertEqual(len(agent.replay.recipe_items(recipe_id)), 3)
        for variant_id in expected_hashes:
            self.assertIn(variant_id, agent.library.variants[recipe_id])
        assert_latest_pin_invariant(agent)

    def test_latest_pin_survives_decay(self):
        agent = AdaptiveAgent(_fast_decay_config())
        variants = _material_preference_variants(BASE_RECIPE)[:3]
        recipe_id = None
        latest_hash = None

        for actions in variants:
            cls = observe_demo(agent, actions)
            recipe_id = recipe_id or cls.recipe_id
            latest_hash = _hash_for(actions)

        latest_key = (recipe_id, latest_hash)
        _advance_demos(agent, 50)

        self.assertIn(latest_key, agent.replay.latest_keys)
        self.assertIn(latest_key, agent.replay.active)
        self.assertAlmostEqual(agent.replay.active[latest_key].weight, 1.0)
        self.assertIn(latest_hash, agent.library.variants[recipe_id])
        assert_latest_pin_invariant(agent)


if __name__ == "__main__":
    unittest.main()
