import unittest

from src.adaptive_agent import AdaptiveHRCAgent
from src.experiments import (
    LiveEpisodeRecord,
    LiveStepRecord,
    live_episode_metrics,
    preference_lock_purity,
)
from src.models import Config


class IdentitySpaceTests(unittest.TestCase):
    def test_agent_keeps_variant_hash_and_latent_pref_id_separate(self):
        agent = AdaptiveHRCAgent(Config(verbose=False, maxent_iters_cold=2, maxent_iters_warm=1, maxent_mc_rollouts=2))
        variant = agent._register_if_live("R1", ["a", "c", "b"], step=1)
        key = ("R1", variant.variant_hash)
        latent_pid = agent.variant_pref_ids[key]

        agent.inferred_recipe = "R1"
        agent.inferred_variant_hash = variant.variant_hash
        agent.inferred_latent_pref_id = latent_pid

        self.assertEqual(agent.inferred_variant_hash, variant.variant_hash)
        self.assertEqual(agent.inferred_latent_pref_id, latent_pid)
        self.assertNotEqual(agent.inferred_variant_hash, agent.inferred_latent_pref_id)
        self.assertEqual(agent.last_pref_id, latent_pid)

    def test_stale_locked_pref_id_is_cleared_after_active_cap_prunes_variant(self):
        agent = AdaptiveHRCAgent(Config(verbose=False, max_variants_per_recipe=1))
        old_variant = agent._register_if_live("R1", ["a", "b", "c"], step=1)
        old_key = ("R1", old_variant.variant_hash)
        agent.locked_variant_key = old_key
        agent.locked_pref_id = agent.variant_pref_ids[old_key]
        agent.inferred_recipe = "R1"
        agent.inferred_variant_hash = old_variant.variant_hash

        new_variant = agent._register_if_live("R1", ["a", "c", "b"], step=2)
        new_key = ("R1", new_variant.variant_hash)

        self.assertIn(old_key, agent.decay.pruned)
        self.assertIn(new_key, agent.decay.active)
        self.assertIsNone(agent.locked_variant_key)
        self.assertIsNone(agent.locked_pref_id)
        self.assertIsNone(agent.inferred_variant_hash)
        self.assertEqual(agent.last_pref_id, agent.variant_pref_ids[new_key])
        self.assertTrue(any(e.get("event") == "stale_preference_lock_cleared" for e in agent.prototype_events))

    def test_prototype_rebuild_reports_split_merge_and_rebuild_ari(self):
        agent = AdaptiveHRCAgent(Config(verbose=False))
        old = {
            "P_old": {("R1", "h1"), ("R2", "h2")},
        }
        split_new = {
            "P_a": {("R1", "h1")},
            "P_b": {("R2", "h2")},
        }
        split_report = agent._prototype_stability_report(old, split_new)

        self.assertEqual(split_report["prototype_split"], {"P_old": ["P_a", "P_b"]})
        self.assertEqual(split_report["prototype_split_count"], 1)
        self.assertIn("prototype_stability_ari", split_report)
        self.assertEqual(split_report["prototype_active_intersection"], 2)

        merge_old = {
            "P_a": {("R1", "h1")},
            "P_b": {("R2", "h2")},
        }
        merge_new = {
            "P_merged": {("R1", "h1"), ("R2", "h2")},
        }
        merge_report = agent._prototype_stability_report(merge_old, merge_new)

        self.assertEqual(merge_report["prototype_merge"], {"P_merged": ["P_a", "P_b"]})
        self.assertEqual(merge_report["prototype_merge_count"], 1)

    def test_prototype_continuity_reuses_old_id_on_active_member_overlap(self):
        agent = AdaptiveHRCAgent(Config(verbose=False))
        k1 = ("R1", "h1")
        k2 = ("R2", "h2")
        agent._active_pref_clusters = {"P_old": {k1, k2}}
        agent._active_pref_cluster_roles = {"P_old": {k1: ("prep", "cook"), k2: ("prep", "serve")}}

        reuse = agent._prototype_continuity_id_map(
            {"P_new": {k1, k2}},
            {"P_new": {k1: ("prep", "cook"), k2: ("prep", "serve")}},
        )

        self.assertEqual(reuse, {"P_new": "P_old"})

    def test_role_sequence_distance_is_order_distance_not_position_mismatch_rate(self):
        adjacent_swap = AdaptiveHRCAgent._role_sequence_distance(
            ("retrieve", "prep", "cook"),
            ("prep", "retrieve", "cook"),
        )
        reversed_order = AdaptiveHRCAgent._role_sequence_distance(
            ("retrieve", "prep", "cook"),
            ("cook", "prep", "retrieve"),
        )

        self.assertAlmostEqual(adjacent_swap, 1.0 / 3.0)
        self.assertAlmostEqual(reversed_order, 1.0)

    def test_live_metrics_do_not_mix_latent_ids_with_variant_hashes(self):
        steps = (
            LiveStepRecord(
                step=0,
                predicted="a",
                actual="a",
                correct_top1=True,
                correct_topk=True,
                topk=("a",),
                inferred_recipe="R1",
                inferred_variant_hash="variant_a",
                inferred_latent_pref_id="P1",
                would_assist_under_gate=True,
            ),
            LiveStepRecord(
                step=1,
                predicted="b",
                actual="b",
                correct_top1=True,
                correct_topk=True,
                topk=("b",),
                inferred_recipe="R1",
                inferred_variant_hash="variant_b",
                inferred_latent_pref_id="P1",
                would_assist_under_gate=True,
            ),
        )
        record = LiveEpisodeRecord(
            pair_label="R/p1",
            recipe_id="R1",
            recipe_name="R",
            preference_name="p1",
            memory_state="active_memory",
            mode="online",
            steps=steps,
            live_top1=1.0,
            live_topk=1.0,
            n=2,
            first_mismatch_robot_turn=None,
        )

        metrics = live_episode_metrics(record, true_rid="R1")
        self.assertEqual(metrics["posterior_pref_lock_consistency"], 1.0)
        self.assertEqual(metrics["posterior_variant_lock_consistency"], 0.5)
        self.assertEqual(preference_lock_purity([record], {"R/p1": "p1"}, use_latent_pref=True), 1.0)
        self.assertEqual(preference_lock_purity([record], {"R/p1": "p1"}, use_latent_pref=False), 0.5)


if __name__ == "__main__":
    unittest.main()
