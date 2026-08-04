import unittest

from src.adaptive_agent import AdaptiveHRCAgent, MODE_ONLINE
from src.memory import Classification, variant_hash
from src.representations import observations_from_actions
from src.environment import gen
from src.models import Config, top_k
from src.preferences import PRESET_PREFERENCES, WorkflowPreferenceModifier, materialize

RECIPE_TOMATO_ONION_SOUP = gen.recipe_library()["tomato_onion_soup_v1"]()
RECIPE_TOMATO_SOUP = gen.recipe_library()["tomato_soup"]()
RECIPE_BURGER = gen.recipe_library()["burger"]()


def _tomato_onion_preference():
    modifier = WorkflowPreferenceModifier()
    for name, pref in PRESET_PREFERENCES.items():
        if name == "identity":
            continue
        candidate = modifier.modify_recipe(RECIPE_TOMATO_ONION_SOUP, pref)
        if candidate != RECIPE_TOMATO_ONION_SOUP:
            return candidate
    return list(RECIPE_TOMATO_ONION_SOUP)


RECIPE_TOMATO_ONION_SOUP_PREF = _tomato_onion_preference()


class AdaptiveHRCAgentTests(unittest.TestCase):
    def _obs(self, agent, seq):
        agent.start_demo()
        for obs in observations_from_actions(seq):
            agent.observe_observation(obs)
        return agent.end_demo()

    def _online(self, agent, seq, gt):
        hits = []
        for obs in observations_from_actions(seq):
            hits.append(agent.observe_observation(obs, ground_truth_recipe=gt).correct)
        agent.end_demo()
        return hits

    def test_observe_registers_new_recipe(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        self.assertIn(cls.recipe_id, ag.memory.variants)
        self.assertEqual(ag.mode, MODE_ONLINE)

    def test_online_same_pref_high_accuracy(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        hits = self._online(ag, RECIPE_TOMATO_ONION_SOUP, "tomato_onion_soup_v1")
        self.assertGreaterEqual(sum(hits) / len(hits), 0.6)

    def test_online_prediction_skips_first_action_then_scores_before_registering_current_action(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        first_obs, second_obs = observations_from_actions(RECIPE_TOMATO_ONION_SOUP[:2])
        seen_during_predict = []
        prefixes = []

        def spy_predict(prefix=None):
            prefixes.append(tuple(prefix or ()))
            seen_during_predict.append(second_obs.action_vector in ag.action_vector_to_token)
            return {}

        ag.predict_next_tokens = spy_predict
        first = ag.observe_observation(first_obs, ground_truth_recipe="tomato_onion_soup_v1")
        second = ag.observe_observation(second_obs, ground_truth_recipe="tomato_onion_soup_v1")

        self.assertEqual(len(prefixes), 1)
        self.assertTrue(prefixes[0])
        self.assertEqual(seen_during_predict, [False])
        self.assertIn(first_obs.action_vector, ag.action_vector_to_token)
        self.assertIn(second_obs.action_vector, ag.action_vector_to_token)
        self.assertFalse(first.correct)
        self.assertFalse(second.correct)

    def test_action_policy_has_no_confidence_threshold_gate(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))

        self.assertFalse(hasattr(ag, "_action_gate_allows"))
        self.assertFalse(hasattr(ag.cfg, "ensemble_fallback_assist_threshold"))








    def test_retrain_rebuilds_predictors_from_active_set_only(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=2,
                maxent_iters_warm=1,
            )
        )
        a = ag._token_for_vector((1, 0))
        b = ag._token_for_vector((0, 1))
        c = ag._token_for_vector((1, 1))

        ag.decay.register("R1", "pref_a", (a, b), now=1, cycle=0)
        ag._retrain()
        self.assertEqual(ag.irl.normalizer.count, len(ag.irl.idx_to_state))

        ag.decay.active.clear()
        ag.decay.register("R2", "pref_b", (c,), now=2, cycle=1)
        ag._retrain()

        self.assertEqual(ag.irl.normalizer.count, len(ag.irl.idx_to_state))
        self.assertEqual(set(ag.markov.vocab), {c})

    def test_online_pref_shift_promotes_variant(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        self._online(ag, RECIPE_TOMATO_ONION_SOUP_PREF, "tomato_onion_soup_v1")
        self.assertGreaterEqual(len(ag.memory.variants[rid]), 2)
        latest = ag.memory.latest_variant(rid)
        # latest should match the preference transition-token ordering
        self.assertEqual(list(latest.ordering), ag._tokens_from_action_labels(RECIPE_TOMATO_ONION_SOUP_PREF))
        self.assertEqual(latest.last_seen_step, ag.step_counter)

    def test_online_multi_stage_reorganization_stays_assistive_for_known_recipe(self):
        ag = AdaptiveHRCAgent(Config(
            verbose=False,
            maxent_iters_cold=1,
            maxent_iters_warm=1,
        ))
        base = RECIPE_TOMATO_SOUP
        first = self._obs(ag, base)
        variant = materialize(base, "p12_multi_stage_reorganization")
        for observation in observations_from_actions(variant):
            ag.observe_observation(observation)
        cls = ag.end_demo()

        self.assertEqual(cls.kind, "preference_shift")
        self.assertEqual(cls.recipe_id, first.recipe_id)
        self.assertFalse(ag._needs_observation)


    def test_new_recipe_creates_entry(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        self.assertIn(cls_a.recipe_id, ag.memory.variants)
        self.assertIn(cls_b.recipe_id, ag.memory.variants)
        self.assertNotEqual(cls_a.recipe_id, cls_b.recipe_id)

    def test_decay_active_tracks_demos(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        ids = {e.recipe_id for e in ag.decay.active_entries()}
        self.assertEqual(ids, {cls_a.recipe_id, cls_b.recipe_id})

    def test_multiple_variants_remain_active_until_temporal_decay(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        ag._register_if_live("R1", ["a", "b"], step=1)
        ag._register_if_live("R1", ["b", "a"], step=2)
        ag._register_if_live("R1", ["c", "a"], step=3)
        memory_keys = {
            (rid, h)
            for rid, slot in ag.memory.variants.items()
            for h in slot
        }
        self.assertEqual(len(memory_keys), 3)
        self.assertEqual(set(ag.decay.active), memory_keys)
        self.assertIn(("R1", ag.memory.latest["R1"]), ag.decay.latest_keys)

    def test_discard_latest_requires_explicit_escape_hatch(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        v = ag._register_if_live("R1", ["a", "b"], step=1)
        with self.assertRaises(RuntimeError):
            ag.decay.discard("R1", v.variant_hash)
        ag.decay.discard("R1", v.variant_hash, allow_latest=True)
        self.assertNotIn(("R1", v.variant_hash), ag.decay.active)




    def test_refresh_after_prune_clears_stale_predictors(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_horizon_init=0,
                decay_horizon_floor=0,
                decay_after_grace_steps=1,
                prune_threshold=0.6,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        self.assertIsNotNone(ag.irl.theta)
        latest_h = ag.memory.latest[rid]
        ag.decay.unmark_latest(rid, latest_h)
        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        ag.refresh_model_from_memory()
        self.assertEqual(ag.decay.active_entries(), [])
        self.assertIsNone(ag.irl.theta)
        self.assertEqual(ag.predict_next([]), {})
        self.assertEqual(ag.evaluate_sequence(RECIPE_TOMATO_ONION_SOUP), 0.0)
        row = ag.observe_observation(observations_from_actions(RECIPE_TOMATO_ONION_SOUP[:1])[0])
        # Predictors are still cleared (no active variants -> no IRL output).
        self.assertIsNone(row.predicted)
        # Pruned-only recipes are absent during live prediction and can reenter
        # only after a completed online session.


    def test_pruned_variant_reenters_only_after_session_commit(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_horizon_init=0,
                decay_horizon_floor=0,
                decay_after_grace_steps=1,
                prune_threshold=0.6,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls0 = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls0.recipe_id
        latest_h = ag.memory.latest[rid]
        ag.decay.unmark_latest(rid, latest_h)
        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        ag.refresh_model_from_memory()

        rows = [
            ag.observe_observation(obs, ground_truth_recipe="tomato_onion_soup_v1")
            for obs in observations_from_actions(RECIPE_TOMATO_ONION_SOUP)
        ]
        self.assertTrue(all(row.predicted is None for row in rows))
        self.assertEqual(ag.decay.active_entries(), [])

        cls = ag.end_demo()
        self.assertEqual(cls.recipe_id, rid)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertEqual({entry.recipe_id for entry in ag.decay.active_entries()}, {rid})
        self.assertGreater(ag.evaluate_sequence(RECIPE_TOMATO_ONION_SOUP), 0.0)



    def test_latest_variant_is_never_decayed(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_horizon_init=0,
                decay_horizon_floor=0,
                decay_after_grace_steps=3,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        latest_h = ag.memory.latest[rid]
        for _ in range(5):
            ag.session_counter += 1
            ag.decay.step(ag.session_counter, ag.retrain_cycle)
        self.assertIn((rid, latest_h), ag.decay.active)
        self.assertAlmostEqual(ag.decay.active[(rid, latest_h)].weight, 1.0)

    def test_old_latest_decays_after_new_latest_registered(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_horizon_init=0,
                decay_horizon_floor=0,
                decay_after_grace_steps=3,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        old_h = ag.memory.latest[rid]
        self._online(ag, RECIPE_TOMATO_ONION_SOUP_PREF, "tomato_onion_soup_v1")
        new_h = ag.memory.latest[rid]
        self.assertNotEqual(old_h, new_h)

        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        weights = ag.decay.weights()
        self.assertLess(weights[(rid, old_h)], 1.0)
        self.assertAlmostEqual(weights[(rid, new_h)], 1.0)

    def test_weight_only_decay_does_not_refit_until_active_membership_changes(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_horizon_init=0,
                decay_horizon_floor=0,
                decay_after_grace_steps=3,
                prune_threshold=1e-9,
                maxent_iters_cold=5,
                maxent_iters_warm=2,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        old_h = ag.memory.latest[rid]
        self._obs(ag, RECIPE_TOMATO_ONION_SOUP_PREF)
        new_h = ag.memory.latest[rid]
        old_key = (rid, old_h)

        self.assertNotEqual(old_h, new_h)
        self.assertIn(old_key, ag.decay.active)
        fit_count = len(ag.retrain_fit_wall_times)
        skipped_count = ag.retrain_skipped_count

        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        self.assertLess(ag.decay.active[old_key].weight, 1.0)
        ag.refresh_model_from_memory()

        self.assertEqual(len(ag.retrain_fit_wall_times), fit_count)
        self.assertEqual(ag.retrain_skipped_count, skipped_count + 1)

        for _ in range(5):
            if old_key in ag.decay.pruned:
                break
            ag.session_counter += 1
            ag.decay.step(ag.session_counter, ag.retrain_cycle)
        self.assertIn(old_key, ag.decay.pruned)

        fit_count_before_prune_refresh = len(ag.retrain_fit_wall_times)
        ag.refresh_model_from_memory()

        self.assertEqual(len(ag.retrain_fit_wall_times), fit_count_before_prune_refresh + 1)
        self.assertEqual({entry.key for entry in ag.decay.active_entries()}, {(rid, new_h)})

    def test_online_novel_sequence_requires_observation_without_memory_mutation(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
            )
        )
        cls0 = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls0.recipe_id
        session_before = ag.session_counter
        retrain_before = len(ag.retrain_events)
        for obs in observations_from_actions(RECIPE_TOMATO_SOUP):
            ag.observe_observation(obs, ground_truth_recipe="tomato_soup")
        cls = ag.end_demo()
        self.assertEqual(cls.kind, "needs_observation")
        self.assertEqual(ag.session_counter, session_before)
        self.assertEqual(len(ag.retrain_events), retrain_before)
        self.assertEqual(set(ag.memory.variants), {rid})
        self.assertEqual({e.recipe_id for e in ag.decay.active_entries()}, {rid})



if __name__ == "__main__":
    unittest.main()
