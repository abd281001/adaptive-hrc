import unittest

from src.adaptive_agent import AdaptiveAgent, MODE_ONLINE
from src.memory import MatchResult
from src.representations import observe_actions
from src.environment import recipe_builders
from src.models import Settings
from src.preferences import PREFERENCES, apply_preference, apply_preset_actions

def _trace(*actions):
    """Observed transitions for `actions`, as the live protocol records them."""
    return tuple(
        (observation.state, observation.action, observation.next_state)
        for observation in observe_actions(actions)
    )


RECIPE_TOMATO_ONION_SOUP = recipe_builders()["tomato_onion_soup"]()
RECIPE_TOMATO_SOUP = recipe_builders()["tomato_soup"]()
RECIPE_MUSHROOM_SOUP = recipe_builders()["mushroom_soup"]()


def _tomato_onion_preference():
    for name, pref in PREFERENCES.items():
        if name == "default":
            continue
        candidate = apply_preference(RECIPE_TOMATO_ONION_SOUP, pref).actions
        if candidate != RECIPE_TOMATO_ONION_SOUP:
            return candidate
    return list(RECIPE_TOMATO_ONION_SOUP)


RECIPE_TOMATO_ONION_SOUP_PREF = _tomato_onion_preference()


class AdaptiveAgentTests(unittest.TestCase):
    def _obs(self, agent, seq):
        agent.start_demo()
        for obs in observe_actions(seq):
            agent.observe(obs)
        return agent.end_demo()

    def _online(self, agent, seq, gt):
        hits = []
        for obs in observe_actions(seq):
            hits.append(agent.observe(obs, ground_truth_recipe=gt).correct)
        agent.end_demo()
        return hits

    @staticmethod
    def _demo(agent, actions):
        observations = observe_actions(actions)
        trajectory = [
            (observation.state, observation.action)
            for observation in observations
        ]
        trajectory.append((observations[-1].next_state, "stop"))
        return trajectory

    def test_full_has_only_the_maxent_predictor(self):
        actions = RECIPE_TOMATO_ONION_SOUP[:3]
        agent = AdaptiveAgent(Settings(
            verbose=False,
            predictor="maxent",
            irl_cold_steps=2,
            irl_warm_steps=1,
        ))
        trajectory = self._demo(agent, RECIPE_TOMATO_ONION_SOUP[:3])
        agent._fit_predictors([trajectory], [1.0], warm_start=False)
        self.assertTrue(agent.maxent.predict(trajectory[0][0]))
        self.assertIsNotNone(agent.maxent.reward_weights)
        self.assertFalse(hasattr(agent, "ngram"))
        self.assertTrue(agent._predictors_ready())

        for removed in ("ngram", "fusion"):
            with self.subTest(removed=removed):
                with self.assertRaisesRegex(ValueError, "predictor"):
                    Settings(predictor=removed)

    def test_observe_registers_new_recipe(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        self.assertIn(cls.recipe_id, ag.library.variants)
        self.assertEqual(ag.mode, MODE_ONLINE)

    def test_online_same_pref_single_maxent_produces_nontrivial_policy(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        hits = self._online(ag, RECIPE_TOMATO_ONION_SOUP, "tomato_onion_soup")
        self.assertTrue(hits)
        self.assertGreater(sum(hits), 0)

    def test_maxent_residual_never_masks_unseen_target_preference_action(self):
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=3,
            irl_warm_steps=2,
        ))
        preference = "prep_first_cleanup"
        self._obs(agent, RECIPE_TOMATO_ONION_SOUP)
        self._obs(
            agent,
            apply_preset_actions(RECIPE_MUSHROOM_SOUP, preference),
        )

        prefix = []
        target = apply_preset_actions(
            RECIPE_TOMATO_ONION_SOUP, preference,
        )
        for action in target:
            with self.subTest(step=len(prefix), action=action):
                distribution = agent.predict_actions(prefix)
                self.assertIn(action, distribution)
                self.assertTrue(
                    agent.policy_stats()["action_mask_preference_neutral"]
                )
                self.assertFalse(
                    agent.policy_stats()["action_mask_uses_recipe_hypothesis"]
                )
            prefix.append(action)

        self.assertFalse(hasattr(agent.maxent, "strategy_components"))
        self.assertFalse(agent._latent_strategy_confirmed)
        self.assertFalse(
            agent.maxent.last_prediction_stats["latent_strategy_eligible"]
        )
        self.assertEqual(
            agent.maxent.last_prediction_stats["model_structure"],
            "maxent_irl_with_latent_strategy_residual",
        )
        self.assertTrue(
            agent.maxent.last_prediction_stats[
                "semantic_fallback_enabled"
            ]
        )
        self.assertNotIn("strategy_component_flops", agent.maxent.last_fit_stats)
        self.assertEqual(
            agent.maxent.last_fit_stats["latent_strategy_rank"], 2,
        )
        self.assertGreater(
            agent.maxent.last_fit_stats["latent_strategy_parameter_count"], 0,
        )

    def test_action_mask_depends_on_state_but_not_prefix_or_recipe_hypothesis(self):
        agent = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=2,
            irl_warm_steps=1,
        ))
        self._obs(agent, RECIPE_TOMATO_ONION_SOUP)
        self._obs(agent, RECIPE_MUSHROOM_SOUP)
        state = agent._replay_prefix(())

        without_prefix = agent._conditioned_actions(state)
        agent.current_prefix = list(RECIPE_MUSHROOM_SOUP[:4])
        with_prefix = agent._conditioned_actions(state)

        self.assertEqual(without_prefix, with_prefix)
        self.assertTrue(
            agent.policy_stats()["action_mask_shared_across_predictors"]
        )
        self.assertFalse(
            agent.policy_stats()["action_mask_uses_recipe_hypothesis"]
        )

    def test_online_prediction_skips_first_action_then_scores_before_registering_current_action(self):
        ag = AdaptiveAgent(Settings(verbose=False))
        first_obs, second_obs = observe_actions(RECIPE_TOMATO_ONION_SOUP[:2])
        seen_during_predict = []
        prefixes = []

        def spy_predict(prefix=None):
            prefixes.append(tuple(prefix or ()))
            seen_during_predict.append(second_obs.action in ag.current_prefix)
            return {}

        ag.predict_actions = spy_predict
        first = ag.observe(first_obs, ground_truth_recipe="tomato_onion_soup")
        second = ag.observe(second_obs, ground_truth_recipe="tomato_onion_soup")

        self.assertEqual(len(prefixes), 1)
        self.assertTrue(prefixes[0])
        self.assertEqual(seen_during_predict, [False])
        self.assertEqual(ag.current_prefix, [first_obs.action, second_obs.action])
        self.assertFalse(first.correct)
        self.assertFalse(second.correct)

    def test_action_policy_has_no_confidence_threshold_gate(self):
        ag = AdaptiveAgent(Settings(verbose=False))

        self.assertFalse(hasattr(ag, "_action_gate_allows"))

    def test_commit_confidence_scores_the_registry_once(self):
        agent = AdaptiveAgent(Settings(verbose=False, profile=True))
        first = None
        for index in range(32):
            variant = agent._register_if_live(
                f"R{index % 4}", (f"action_{index}", "shared"), step=index + 1,
            )
            first = first or variant
        assert first is not None
        agent.matcher.profile.clear()

        agent._commit_confidence(
            "R0", first.ordering,
            MatchResult("known", "R0", first.variant_id, 1.0, 0.0),
        )

        diagnostics = agent.last_commit_stats
        self.assertEqual(agent.matcher.profile["score"][0], 1)
        self.assertNotIn("classify", agent.matcher.profile)
        self.assertEqual(diagnostics["registry_size"], 32)
        self.assertEqual(diagnostics["registry_recipes"], 4)
        self.assertEqual(diagnostics["variants_scored"], 32)
        self.assertGreaterEqual(diagnostics["scoring_wall_s"], 0.0)

    def test_retrain_rebuilds_predictors_from_active_set_only(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=2,
                irl_warm_steps=1,
            )
        )
        a = "transfer (pot, from=storage, to=cooking_station)"
        b = "turn_on (stove, cooking_station)"
        c = "transfer (pan, from=storage, to=cooking_station)"

        ag.replay.register("R1", "pref_a", (a, b), now=1, cycle=0, transitions=_trace(a, b))
        ag._retrain()
        self.assertEqual(ag.maxent.normalizer.count, len(ag.maxent.state_vectors))

        ag.replay.active.clear()
        ag.replay.register("R2", "pref_b", (c,), now=2, cycle=1, transitions=_trace(c))
        ag._retrain()

        self.assertEqual(ag.maxent.normalizer.count, len(ag.maxent.state_vectors))
        self.assertEqual(set(ag.maxent.action_ids), {c, "stop"})

    def test_online_pref_shift_promotes_variant(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
                commit_threshold=0.0,
                tentative_threshold=0.0,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls.recipe_id
        self._online(ag, RECIPE_TOMATO_ONION_SOUP_PREF, "tomato_onion_soup")
        self.assertGreaterEqual(len(ag.library.variants[recipe_id]), 2)
        latest = ag.library.latest_variant(recipe_id)
        self.assertEqual(list(latest.ordering), RECIPE_TOMATO_ONION_SOUP_PREF)
        self.assertEqual(latest.last_seen_step, ag.step_counter)

    def test_online_multi_stage_reorganization_stays_assistive_for_known_recipe(self):
        ag = AdaptiveAgent(Settings(
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
        ))
        base = RECIPE_TOMATO_SOUP
        first = self._obs(ag, base)
        variant = apply_preset_actions(base, "prep_loading_serving_cleanup")
        for observation in observe_actions(variant):
            ag.observe(observation)
        cls = ag.end_demo()

        self.assertIn(
            cls.kind, {"preference_shift", "tentative_preference_shift"}
        )
        self.assertEqual(cls.recipe_id, first.recipe_id)
        self.assertFalse(ag._needs_observation)


    def test_new_recipe_creates_entry(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        self.assertIn(cls_a.recipe_id, ag.library.variants)
        self.assertIn(cls_b.recipe_id, ag.library.variants)
        self.assertNotEqual(cls_a.recipe_id, cls_b.recipe_id)

    def test_decay_active_tracks_demos(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        ids = {e.recipe_id for e in ag.replay.active_items()}
        self.assertEqual(ids, {cls_a.recipe_id, cls_b.recipe_id})

    def test_multiple_variants_remain_active_until_temporal_decay(self):
        ag = AdaptiveAgent(Settings(verbose=False))
        ag._register_if_live("R1", ["a", "b"], step=1)
        ag._register_if_live("R1", ["b", "a"], step=2)
        ag._register_if_live("R1", ["c", "a"], step=3)
        memory_keys = {
            (recipe_id, variant_id)
            for recipe_id, slot in ag.library.variants.items()
            for variant_id in slot
        }
        self.assertEqual(len(memory_keys), 3)
        self.assertEqual(set(ag.replay.active), memory_keys)
        self.assertIn(("R1", ag.library.latest["R1"]), ag.replay.latest_keys)

    def test_discard_latest_requires_explicit_escape_hatch(self):
        ag = AdaptiveAgent(Settings(verbose=False))
        v = ag._register_if_live("R1", ["a", "b"], step=1)
        with self.assertRaises(RuntimeError):
            ag.replay.discard("R1", v.variant_id)
        ag.replay.discard("R1", v.variant_id, allow_latest=True)
        self.assertNotIn(("R1", v.variant_id), ag.replay.active)

    def test_refresh_after_prune_clears_stale_predictors(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                initial_grace=0,
                min_grace=0,
                prune_delay=1,
                prune_threshold=0.6,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls.recipe_id
        self.assertIsNotNone(ag.maxent.reward_weights)
        latest_h = ag.library.latest[recipe_id]
        ag.replay.unmark_latest(recipe_id, latest_h)
        ag.demo_counter += 1
        ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        ag.refresh()
        self.assertEqual(ag.replay.active_items(), [])
        self.assertIsNone(ag.maxent.reward_weights)
        self.assertEqual(ag.predict_actions([]), {})
        self.assertEqual(ag.evaluate(RECIPE_TOMATO_ONION_SOUP), 0.0)
        row = ag.observe(observe_actions(RECIPE_TOMATO_ONION_SOUP[:1])[0])
        self.assertIsNone(row.predicted)


    def test_pruned_variant_reenters_only_after_session_commit(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                initial_grace=0,
                min_grace=0,
                prune_delay=1,
                prune_threshold=0.6,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls0 = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls0.recipe_id
        latest_h = ag.library.latest[recipe_id]
        ag.replay.unmark_latest(recipe_id, latest_h)
        ag.demo_counter += 1
        ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        ag.refresh()

        rows = [
            ag.observe(obs, ground_truth_recipe="tomato_onion_soup")
            for obs in observe_actions(RECIPE_TOMATO_ONION_SOUP)
        ]
        self.assertTrue(all(row.predicted is None for row in rows))
        self.assertEqual(ag.replay.active_items(), [])

        cls = ag.end_demo()
        self.assertEqual(cls.recipe_id, recipe_id)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertEqual({entry.recipe_id for entry in ag.replay.active_items()}, {recipe_id})
        self.assertGreater(ag.evaluate(RECIPE_TOMATO_ONION_SOUP), 0.0)

    def test_latest_variant_is_never_decayed(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                initial_grace=0,
                min_grace=0,
                prune_delay=3,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls.recipe_id
        latest_h = ag.library.latest[recipe_id]
        for _ in range(5):
            ag.demo_counter += 1
            ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        self.assertIn((recipe_id, latest_h), ag.replay.active)
        self.assertAlmostEqual(ag.replay.active[(recipe_id, latest_h)].weight, 1.0)

    def test_old_latest_decays_after_new_latest_registered(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                initial_grace=0,
                min_grace=0,
                prune_delay=3,
                irl_cold_steps=30,
                irl_warm_steps=12,
                commit_threshold=0.0,
                tentative_threshold=0.0,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls.recipe_id
        old_h = ag.library.latest[recipe_id]
        self._online(ag, RECIPE_TOMATO_ONION_SOUP_PREF, "tomato_onion_soup")
        new_h = ag.library.latest[recipe_id]
        self.assertNotEqual(old_h, new_h)

        ag.demo_counter += 1
        ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        weights = ag.replay.weights()
        self.assertLess(weights[(recipe_id, old_h)], 1.0)
        self.assertAlmostEqual(weights[(recipe_id, new_h)], 1.0)

    def test_weight_only_decay_skips_fit_without_advancing_membership_counter(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                initial_grace=0,
                min_grace=0,
                prune_delay=3,
                prune_threshold=1e-9,
                irl_cold_steps=5,
                irl_warm_steps=2,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls.recipe_id
        old_h = ag.library.latest[recipe_id]
        self._obs(ag, RECIPE_TOMATO_ONION_SOUP_PREF)
        new_h = ag.library.latest[recipe_id]
        old_key = (recipe_id, old_h)

        self.assertNotEqual(old_h, new_h)
        self.assertIn(old_key, ag.replay.active)
        fit_count = len(ag.retrain_fit_wall_times)
        skipped_count = ag.skipped_trains
        count = ag.cold_change_count

        ag.demo_counter += 1
        ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        self.assertLess(ag.replay.active[old_key].weight, 1.0)
        ag.refresh()

        self.assertEqual(len(ag.retrain_fit_wall_times), fit_count)
        self.assertEqual(ag.skipped_trains, skipped_count + 1)
        self.assertEqual(ag.retrain_events[-1]["retrain_trigger"], "weight_only_change_skipped")
        self.assertEqual(ag.retrain_events[-1]["retrain_effective_start"], "skipped")
        self.assertEqual(ag.cold_change_count, count)

        for _ in range(5):
            if old_key in ag.replay.pruned:
                break
            ag.demo_counter += 1
            ag.replay.step(ag.demo_counter, ag.retrain_cycle)
        self.assertIn(old_key, ag.replay.pruned)

        fit_count_before_prune_refresh = len(ag.retrain_fit_wall_times)
        ag.refresh()

        self.assertEqual(len(ag.retrain_fit_wall_times), fit_count_before_prune_refresh + 1)
        self.assertEqual({entry.key for entry in ag.replay.active_items()}, {(recipe_id, new_h)})

    def test_full_agent_counts_only_additions_toward_the_cold_threshold(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=1,
                irl_warm_steps=1,
            )
        )
        actions = [
            "transfer (pot, from=storage, to=cooking_station)",
            "transfer (pan, from=storage, to=cooking_station)",
            "transfer (bowl, from=storage, to=prep_station)",
        ]

        ag.replay.register(
            "R0", "h0", (actions[0],), now=1, cycle=0, pin_latest=False,
            transitions=_trace(actions[0]),
        )
        ag._retrain()
        self.assertEqual(
            ag.retrain_events[-1]["retrain_effective_start"],
            "cold_fallback_no_model",
        )
        self.assertEqual(ag.cold_change_count, 0)

        ag.replay.register(
            "R1", "h1", (actions[1],), now=2, cycle=1, pin_latest=False,
            transitions=_trace(actions[1]),
        )
        ag._retrain()
        self.assertEqual(ag.retrain_events[-1]["retrain_effective_start"], "warm")
        self.assertEqual(ag.cold_change_count, 1)
        self.assertTrue(ag._fit_stats()["warm_start"])

        # A removal strictly shrinks the training set, so the incumbent weights
        # stay valid: warm start, and the cold-start counter does not advance.
        ag.replay.active.pop(("R1", "h1"))
        ag._retrain()
        self.assertEqual(
            ag.retrain_events[-1]["retrain_trigger"],
            "removal_only_membership_change",
        )
        self.assertEqual(ag.retrain_events[-1]["retrain_effective_start"], "warm")
        self.assertEqual(ag.retrain_events[-1]["active_removed_count"], 1)
        self.assertEqual(ag.retrain_events[-1]["active_added_count"], 0)
        self.assertEqual(ag.cold_change_count, 1)
        self.assertTrue(ag._fit_stats()["warm_start"])

        # Two further additions reach the threshold of three; the intervening
        # removal contributed nothing to it.
        ag.replay.register(
            "R2", "h2", (actions[2],), now=3, cycle=3, pin_latest=False,
            transitions=_trace(actions[2]),
        )
        ag._retrain()
        self.assertEqual(ag.retrain_events[-1]["retrain_effective_start"], "warm")
        self.assertEqual(ag.cold_change_count, 2)

        ag.replay.register(
            "R3", "h3", (actions[1],), now=4, cycle=4, pin_latest=False,
            transitions=_trace(actions[1]),
        )
        ag._retrain()
        self.assertEqual(
            ag.retrain_events[-1]["retrain_trigger"],
            "cumulative_addition_threshold_reached",
        )
        self.assertEqual(ag.retrain_events[-1]["retrain_effective_start"], "cold")
        self.assertEqual(ag.retrain_events[-1]["cold_threshold_basis"], "additions_only")
        self.assertEqual(ag.cold_change_count, 0)
        self.assertFalse(ag._fit_stats()["warm_start"])

        fits_before = len(ag.retrain_fit_wall_times)
        ag._retrain()
        self.assertEqual(ag.retrain_events[-1]["retrain_trigger"], "replay_unchanged")
        self.assertEqual(len(ag.retrain_fit_wall_times), fits_before)

    def test_assist_low_confidence_abstains_without_changing_mode_contract(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=30,
                irl_warm_steps=12,
            )
        )
        cls0 = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        recipe_id = cls0.recipe_id
        demos_before = ag.demo_counter
        retrain_before = len(ag.retrain_events)
        for obs in observe_actions(RECIPE_TOMATO_SOUP):
            ag.observe(obs, ground_truth_recipe="tomato_soup")
        cls = ag.end_demo()
        self.assertEqual(cls.kind, "known_recipe_uncertain")
        self.assertEqual(cls.recipe_id, recipe_id)
        self.assertFalse(ag._needs_observation)
        self.assertEqual(ag.last_commit_stats["decision"], "none")
        self.assertFalse(ag.last_commit_stats["commit_applied"])
        self.assertEqual(ag.demo_counter, demos_before + 1)
        # Rejected demos age memory; the fingerprint gate skips unchanged refits.
        self.assertEqual(len(ag.retrain_events), retrain_before + 1)
        self.assertEqual(set(ag.library.variants), {recipe_id})
        self.assertEqual({e.recipe_id for e in ag.replay.active_items()}, {recipe_id})

    def test_pair_gap_clock_includes_intervening_rejected_demo(self):
        ag = AdaptiveAgent(
            Settings(
                verbose=False,
                irl_cold_steps=2,
                irl_warm_steps=1,
            )
        )
        first = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        first_key = (
            first.recipe_id,
            ag.library.latest[first.recipe_id],
        )

        for obs in observe_actions(RECIPE_TOMATO_SOUP):
            ag.observe(obs)
        rejected = ag.end_demo()
        self.assertEqual(rejected.kind, "known_recipe_uncertain")
        self.assertFalse(ag._needs_observation)

        repeated = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)

        self.assertEqual(repeated.recipe_id, first.recipe_id)
        self.assertEqual(ag.demo_counter, 3)
        self.assertEqual(ag.replay.pair_history(first_key), [2])

    def test_empty_end_demo_does_not_advance_demo_clock(self):
        ag = AdaptiveAgent(Settings(verbose=False))
        ag.end_demo()
        self.assertEqual(ag.demo_counter, 0)

if __name__ == "__main__":
    unittest.main()
