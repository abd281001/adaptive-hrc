import unittest

from src.adaptive_agent import AdaptiveHRCAgent, MODE_ONLINE
from src.memory import _aligned_next_index
from src.representations import observations_from_actions
from src.environment import gen
from src.models import Config, top_k
from src.preferences import PRESET_PREFERENCES, WorkflowPreferenceModifier

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
                maxent_mc_rollouts=40,
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
                maxent_mc_rollouts=40,
            )
        )
        self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        hits = self._online(ag, RECIPE_TOMATO_ONION_SOUP, "tomato_onion_soup_v1")
        self.assertGreaterEqual(sum(hits) / len(hits), 0.6)

    def test_online_prediction_scores_before_registering_current_action(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        obs = observations_from_actions(RECIPE_TOMATO_ONION_SOUP[:1])[0]
        seen_during_predict = []

        def spy_predict(prefix=None, recipe_id=None, pref_hash=None):
            seen_during_predict.append(obs.action_vector in ag.action_vector_to_token)
            return {}

        ag.predict_next_tokens = spy_predict
        row = ag.observe_observation(obs, ground_truth_recipe="tomato_onion_soup_v1")

        self.assertEqual(seen_during_predict, [False])
        self.assertIn(obs.action_vector, ag.action_vector_to_token)
        self.assertFalse(row.correct)

    def test_aligned_next_index_advances_past_entire_prefix(self):
        ordering = ("a1", "a2", "a3", "a4")
        self.assertEqual(_aligned_next_index(("a1", "a2"), ordering), 2)
        self.assertEqual(_aligned_next_index(("a1", "a3"), ordering), 3)

    def test_action_gate_uses_calibrated_margin_entropy_score(self):
        ag = AdaptiveHRCAgent(Config(
            verbose=False,
            posterior_action_confidence_threshold=0.35,
            posterior_action_gate_temperature=1.0,
            posterior_action_gate_margin_weight=0.20,
            posterior_action_gate_entropy_weight=0.20,
            posterior_action_margin_threshold=0.10,
            posterior_action_margin_min_confidence=0.10,
            posterior_action_entropy_threshold=0.70,
        ))

        allowed, score, reason = ag._action_gate_allows(confidence=0.20, entropy=0.20, margin=0.20)
        self.assertTrue(allowed)
        self.assertGreaterEqual(score, 0.35)
        self.assertEqual(reason, "margin_entropy_gate")

        allowed, score, reason = ag._action_gate_allows(confidence=0.20, entropy=0.95, margin=0.0)
        self.assertFalse(allowed)
        self.assertLess(score, 0.35)
        self.assertIn(reason, {"low_action_confidence", "low_action_gate_score"})

    def test_predict_gates_final_ensemble_when_posterior_is_unavailable(self):
        ag = AdaptiveHRCAgent(Config(
            verbose=False,
            posterior_action_confidence_threshold=0.20,
            ensemble_fallback_assist_threshold=0.35,
        ))
        ag.irl.predict = lambda state: {"a": 0.90, "b": 0.10}
        ag.markov.predict = lambda state, prefix: {"a": 0.90, "b": 0.10}
        ag.graph.predict = lambda state, prefix: {"a": 0.90, "b": 0.10}

        dist = ag.predict_next_tokens([])
        gate = ag.assist_gate_stats()

        self.assertEqual(top_k(dist, 1)[0], "a")
        self.assertTrue(gate["assist_used"])
        self.assertEqual(gate["assist_source"], "ensemble_posterior_empty")
        self.assertIsNotNone(gate["final_action_confidence"])

    def test_predict_initializes_posterior_from_active_memory_at_episode_start(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        a = ag._token_for_vector((1, 0))
        b = ag._token_for_vector((0, 1))
        ag._register_if_live("R1", [a, b], step=1)
        ag.irl.predict = lambda state: {a: 0.50, b: 0.50}
        ag.markov.predict = lambda state, prefix: {a: 0.50, b: 0.50}
        ag.graph.predict = lambda state, prefix: {a: 0.50, b: 0.50}
        ag.posterior.reset()

        ag.predict_next_tokens([])

        self.assertTrue(ag.posterior.joint())

    def test_dynamic_blend_trusts_agreement_more_than_conflicting_ensemble(self):
        ag = AdaptiveHRCAgent(Config(
            verbose=False,
            posterior_assist_strength=0.70,
            posterior_assist_strength_min=0.15,
            posterior_assist_strength_max=0.90,
            posterior_assist_agreement_bonus=0.15,
            posterior_assist_disagreement_penalty=0.45,
        ))
        agree = ag._posterior_blend_strength({"a": 0.80, "b": 0.20}, {"a": 0.70, "b": 0.30}, 0.80, 0.72, 0.60, 0.70, 0.88, 0.40)
        disagree = ag._posterior_blend_strength({"b": 0.55, "a": 0.45}, {"a": 0.90, "b": 0.10}, 0.55, 0.99, 0.10, 0.90, 0.47, 0.80)

        self.assertGreater(agree, disagree)
        self.assertGreaterEqual(agree, 0.80)
        self.assertLess(disagree, 0.70)

    def test_preference_lock_boosts_locked_variant_frontier(self):
        ag = AdaptiveHRCAgent(Config(verbose=False, locked_variant_action_boost=0.70))
        a = ag._token_for_vector((1, 0))
        b = ag._token_for_vector((0, 1))
        c = ag._token_for_vector((1, 1))
        ag._register_if_live("R1", [a, b, c], step=1)
        locked = ag._register_if_live("R1", [a, c, b], step=2)
        ag.locked_variant_key = ("R1", locked.pref_hash)
        ag.locked_pref_id = ag.variant_pref_ids.get(ag.locked_variant_key)
        pid = ag.locked_pref_id or ag.last_pref_id or "P1"
        ag.posterior._joint = {("R1", pid): 1.0}

        dist, _confidence, _entropy, _margin = ag._action_marginal_distribution([a])

        self.assertGreater(dist.get(c, 0.0), dist.get(b, 0.0))
        self.assertEqual(top_k(dist, 1)[0], c)

    def test_retrain_rebuilds_predictors_from_active_set_only(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=2,
                maxent_iters_warm=1,
                maxent_mc_rollouts=2,
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
                maxent_mc_rollouts=40,
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

    def test_cross_recipe_preference_factor_guides_unseen_order(self):
        modifier = WorkflowPreferenceModifier()
        target = modifier.modify_recipe(RECIPE_TOMATO_ONION_SOUP, PRESET_PREFERENCES["p1_prep_first"])
        support = modifier.modify_recipe(RECIPE_BURGER, PRESET_PREFERENCES["p1_prep_first"])
        cfg = Config(verbose=False, maxent_iters_cold=2, maxent_iters_warm=1, maxent_mc_rollouts=2)

        ag_base = AdaptiveHRCAgent(cfg)
        self._obs(ag_base, RECIPE_TOMATO_ONION_SOUP)
        target_base = ag_base._tokens_from_action_labels(target)
        original_base = ag_base._tokens_from_action_labels(RECIPE_TOMATO_ONION_SOUP)
        prefix_base = target_base[:1]
        ag_base.posterior.update(prefix_base, ag_base._roles_for_tokens(prefix_base), ag_base.recipe_prototypes, ag_base.preference_prototypes, ag_base._memory_state_for_recipe)
        dist_base, _, _, _ = ag_base._action_marginal_distribution(prefix_base)

        ag_transfer = AdaptiveHRCAgent(cfg)
        self._obs(ag_transfer, RECIPE_TOMATO_ONION_SOUP)
        self._obs(ag_transfer, support)
        target_transfer = ag_transfer._tokens_from_action_labels(target)
        original_transfer = ag_transfer._tokens_from_action_labels(RECIPE_TOMATO_ONION_SOUP)
        prefix_transfer = target_transfer[:1]
        ag_transfer.posterior.update(prefix_transfer, ag_transfer._roles_for_tokens(prefix_transfer), ag_transfer.recipe_prototypes, ag_transfer.preference_prototypes, ag_transfer._memory_state_for_recipe)
        dist_transfer, _, _, _ = ag_transfer._action_marginal_distribution(prefix_transfer)

        self.assertGreater(dist_transfer.get(target_transfer[1], 0.0), dist_base.get(target_base[1], 0.0))
        self.assertGreater(dist_transfer.get(target_transfer[1], 0.0), dist_transfer.get(original_transfer[1], 0.0))
        self.assertEqual(top_k(dist_transfer, 1)[0], target_transfer[1])

    def test_new_recipe_creates_entry(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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
                maxent_mc_rollouts=40,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        ids = {e.recipe_id for e in ag.decay.active_entries()}
        self.assertEqual(ids, {cls_a.recipe_id, cls_b.recipe_id})

    def test_variant_lru_eviction_discards_decay_entry(self):
        ag = AdaptiveHRCAgent(Config(verbose=False, max_variants_per_recipe=2))
        ag._register_if_live("R1", ["a", "b"], step=1)
        ag._register_if_live("R1", ["b", "a"], step=2)
        ag._register_if_live("R1", ["c", "a"], step=3)
        memory_keys = {
            (rid, h)
            for rid, slot in ag.memory.variants.items()
            for h in slot
        }
        self.assertLessEqual(len(memory_keys), 2)
        self.assertTrue(set(ag.decay.active).issubset(memory_keys))

    def test_refresh_after_prune_clears_stale_predictors(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_init=0.5,
                decay_horizon_init=2,
                size_rescale_exponent=0.0,
                prune_threshold=0.6,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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
        self.assertNotIn(rid, ag.task_signatures)
        self.assertNotIn(rid, ag.recipe_prototypes.prototypes)
        row = ag.observe_observation(observations_from_actions(RECIPE_TOMATO_ONION_SOUP[:1])[0])
        # Predictors are still cleared (no active variants -> no IRL output).
        self.assertIsNone(row.predicted)
        # Pruned-only recipes are absent during live prediction and can reenter
        # only after a completed online session.
        self.assertIsNone(row.inferred_recipe)
        self.assertNotIn(rid, ag.recipe_prototypes.prototypes)

    def test_preference_lock_promotes_latest_only_at_end_demo(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        self._online(ag, RECIPE_TOMATO_ONION_SOUP_PREF, "tomato_onion_soup_v1")
        latest_after_preference = ag.memory.latest_variant(rid)
        self.assertEqual(latest_after_preference.last_seen_step, ag.step_counter)

        base_observations = observations_from_actions(RECIPE_TOMATO_ONION_SOUP)
        ag.observe_observation(base_observations[0], ground_truth_recipe="tomato_onion_soup_v1")
        ag.observe_observation(base_observations[1], ground_truth_recipe="tomato_onion_soup_v1")
        self.assertEqual(ag.memory.latest_variant(rid).pref_hash, latest_after_preference.pref_hash)

        for obs in base_observations[2:]:
            ag.observe_observation(obs, ground_truth_recipe="tomato_onion_soup_v1")
        ag.end_demo()
        self.assertEqual(list(ag.memory.latest_variant(rid).ordering), ag._tokens_from_action_labels(RECIPE_TOMATO_ONION_SOUP))
        self.assertEqual(ag.memory.latest_variant(rid).last_seen_step, ag.step_counter)

    def test_pruned_variant_reenters_only_after_session_commit(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_init=0.5,
                decay_horizon_init=2,
                size_rescale_exponent=0.0,
                prune_threshold=0.6,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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
        self.assertTrue(all(row.inferred_recipe is None for row in rows))
        self.assertEqual(ag.decay.active_entries(), [])

        cls = ag.end_demo()
        self.assertEqual(cls.recipe_id, rid)
        self.assertEqual(cls.kind, "reentry_from_pruned")
        self.assertEqual({entry.recipe_id for entry in ag.decay.active_entries()}, {rid})
        self.assertGreater(ag.evaluate_sequence(RECIPE_TOMATO_ONION_SOUP), 0.0)

    def test_pruned_influence_audit_separates_live_prediction_contract(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_init=0.5,
                decay_horizon_init=2,
                size_rescale_exponent=0.0,
                prune_threshold=0.6,
                maxent_iters_cold=5,
                maxent_iters_warm=2,
                maxent_mc_rollouts=5,
            )
        )
        cls = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid = cls.recipe_id
        latest_h = ag.memory.latest[rid]
        ag.decay.unmark_latest(rid, latest_h)
        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        ag.refresh_model_from_memory()
        cls2 = self._obs(ag, RECIPE_TOMATO_SOUP)
        self.assertIsNotNone(cls2.recipe_id)
        audit = ag.pruned_influence_audit(max_prefixes=4)
        self.assertIn("active_head_max_l1", audit)
        self.assertIn("live_prediction_max_l1", audit)
        self.assertTrue(audit["live_prediction_passed"])

    def test_active_task_signature_rebuilds_after_pruning(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_init=0.5,
                decay_horizon_init=2,
                size_rescale_exponent=0.0,
                prune_threshold=0.6,
                maxent_iters_cold=5,
                maxent_iters_warm=2,
                maxent_mc_rollouts=5,
            )
        )
        cls_a = self._obs(ag, RECIPE_TOMATO_ONION_SOUP)
        rid_a = cls_a.recipe_id
        cls_b = self._obs(ag, RECIPE_TOMATO_SOUP)
        rid_b = cls_b.recipe_id
        self.assertIn(rid_a, ag.task_signatures)
        self.assertIn(rid_b, ag.task_signatures)
        latest_h = ag.memory.latest[rid_a]
        ag.decay.unmark_latest(rid_a, latest_h)
        ag.session_counter += 1
        ag.decay.step(ag.session_counter, ag.retrain_cycle)
        ag.refresh_model_from_memory()
        self.assertNotIn(rid_a, ag.task_signatures)
        self.assertIn(rid_b, ag.task_signatures)

    def test_latest_variant_is_never_decayed(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                decay_init=0.1,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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
                decay_init=0.1,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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

    def test_online_novel_sequence_requires_observation_without_memory_mutation(self):
        ag = AdaptiveHRCAgent(
            Config(
                verbose=False,
                maxent_iters_cold=30,
                maxent_iters_warm=12,
                maxent_mc_rollouts=40,
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

    def test_recipe_scoring_uses_graph_consistency_weights(self):
        ag = AdaptiveHRCAgent(Config(verbose=False))
        self.assertTrue(hasattr(ag.cfg, "recipe_score_prefix_weight"))
        self.assertTrue(hasattr(ag.cfg, "recipe_score_graph_weight"))
        self.assertGreaterEqual(ag.cfg.recipe_score_prefix_weight, 0.0)
        self.assertGreaterEqual(ag.cfg.recipe_score_graph_weight, 0.0)

    def _patch_posterior(self, agent, joint):
        """Make `posterior.update` a no-op and force the joint snapshot."""
        agent.posterior._joint = dict(joint)
        agent.posterior.update = lambda **kwargs: dict(joint)

    def test_identity_hysteresis_blocks_one_step_flip(self):
        """A single-step argmax flip must NOT change inferred_recipe.
        Without hysteresis, prefix-collision scenarios thrash the committed
        identity; the gate requires k=2 consecutive agreements + log-margin."""
        ag = AdaptiveHRCAgent(Config(verbose=False))
        switch_agreement = int(getattr(ag.cfg, "posterior_switch_agreement", 2))
        ag.inferred_recipe = "A"
        self._patch_posterior(ag, {("B", "P1"): 0.9, ("A", "P1"): 0.1})
        ag._refresh_recipe_inference([])
        # First disagreement must NOT switch (pending_count = 1, K = 2).
        self.assertEqual(ag.inferred_recipe, "A", "single-step flip must not switch identity")
        self.assertEqual(ag._pending_argmax, "B")
        self.assertEqual(ag._pending_count, 1)
        # Second consecutive agreement: now switch.
        ag._refresh_recipe_inference([])
        if switch_agreement <= 2:
            self.assertEqual(ag.inferred_recipe, "B", "after K consecutive agreements identity should switch")

    def test_identity_hysteresis_resets_on_revert(self):
        """If the argmax reverts to the current committed recipe before K
        consecutive agreements accumulate, the pending counter must reset."""
        ag = AdaptiveHRCAgent(Config(verbose=False))
        ag.inferred_recipe = "A"
        self._patch_posterior(ag, {("B", "P1"): 0.9, ("A", "P1"): 0.1})
        ag._refresh_recipe_inference([])
        self.assertEqual(ag._pending_argmax, "B")
        # Revert to A
        self._patch_posterior(ag, {("A", "P1"): 0.9, ("B", "P1"): 0.1})
        ag._refresh_recipe_inference([])
        self.assertIsNone(ag._pending_argmax, "pending switch must reset when posterior agrees with current")
        self.assertEqual(ag._pending_count, 0)
        self.assertEqual(ag.inferred_recipe, "A")


if __name__ == "__main__":
    unittest.main()
