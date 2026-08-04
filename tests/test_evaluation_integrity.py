"""Regression tests for evaluator-only labels and transfer scheduling."""
from __future__ import annotations

import unittest

from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.evaluation import (
    COMMIT_SENSITIVITY_SPECS,
    EvaluationConfig,
    PAPER_SEEDS,
    RecipePreferencePair,
    _memory_state,
    _pair_key,
    _preference_tags,
    aggregate_episode_metrics,
    assist_episode,
    base_config,
    build_deployment_random,
    build_ladder_heterogeneous,
    build_ladder_homogeneous,
    delayed_recurrence_audit_summary,
    frozen_summary,
    frozen_summary_by,
    materialize_pair,
    oracle_gap_summary,
    paired_bootstrap_summary,
    parse_args,
    reentry_stratification_summary,
)
from src.models import Config
from src.representations import observations_from_actions


class EvaluationIntegrityTests(unittest.TestCase):
    def test_audit_defaults_are_enabled_and_empty_summaries_are_explicit(self):
        cfg = EvaluationConfig()
        self.assertGreater(cfg.frozen_eval_period, 0)
        self.assertGreater(cfg.active_only_audit_period, 0)
        self.assertEqual(frozen_summary([])["status"], "not_run")

    def test_frozen_summary_reports_support_under_the_canonical_schema(self):
        summary = frozen_summary([
            {"checkpoint": "final", "top1": 0.75, "topk": 1.0, "human_correction_rate": 0.25},
            {"checkpoint": "final", "top1": 0.25, "topk": 0.5, "human_correction_rate": 0.75},
        ])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["n_checkpoints"], 1)
        self.assertEqual(summary["n_rows"], 2)
        self.assertEqual(summary["checkpoints"]["final"]["top1"], 0.5)

    def test_frozen_summary_by_condition_retains_only_tagged_probe_rows(self):
        summary = frozen_summary_by([
            {"condition": "delayed", "checkpoint": "pre_event_1", "top1": 0.75, "topk": 1.0, "human_correction_rate": 0.25},
            {"condition": "delayed", "checkpoint": "pre_event_2", "top1": 0.25, "topk": 0.5, "human_correction_rate": 0.75},
            {"checkpoint": "final", "top1": 1.0, "topk": 1.0, "human_correction_rate": 0.0},
        ], "condition")
        self.assertEqual(set(summary), {"delayed"})
        self.assertEqual(summary["delayed"]["n_rows"], 2)

    def test_delayed_recurrence_audit_requires_actual_memory_removal(self):
        summary = delayed_recurrence_audit_summary([
            {
                "delayed_recurrence_probe": True,
                "delayed_recurrence_target_active_before": True,
                "delayed_recurrence_target_is_latest_before": True,
                "delayed_recurrence_all_known_conflicts_pruned_before": True,
                "delayed_recurrence_known_conflicting_variant_count_before": 3,
                "delayed_recurrence_conflicting_active_count_before": 0,
                "delayed_recurrence_conflicting_pruned_count_before": 3,
                "delayed_recurrence_intervening_event_count": 4,
                "delayed_recurrence_required_intervening_event_count": 4,
                "delayed_recurrence_actual_grace_horizon_before": 7,
            },
            {
                "delayed_recurrence_probe": True,
                "delayed_recurrence_target_active_before": True,
                "delayed_recurrence_target_is_latest_before": False,
                "delayed_recurrence_all_known_conflicts_pruned_before": False,
                "delayed_recurrence_known_conflicting_variant_count_before": 3,
                "delayed_recurrence_conflicting_active_count_before": 3,
                "delayed_recurrence_conflicting_pruned_count_before": 0,
                "delayed_recurrence_intervening_event_count": 6,
                "delayed_recurrence_required_intervening_event_count": 6,
                "delayed_recurrence_actual_grace_horizon_before": 8,
            },
        ])
        self.assertEqual(summary["n_probes"], 2)
        self.assertEqual(summary["target_active_before_rate"], 1.0)
        self.assertEqual(summary["all_known_conflicts_pruned_before_rate"], 0.5)

    def test_paper_seed_protocol_is_the_default_for_api_and_cli(self):
        self.assertEqual(len(PAPER_SEEDS), 15)
        self.assertEqual(EvaluationConfig().seeds, PAPER_SEEDS)
        self.assertEqual(parse_args([]).seeds, PAPER_SEEDS)

    def test_commit_sensitivity_specs_only_override_named_config_fields(self):
        for _name, overrides in COMMIT_SENSITIVITY_SPECS:
            cfg = base_config(17, EvaluationConfig(model_overrides=overrides))
            for field, value in overrides.items():
                self.assertEqual(getattr(cfg, field), value)

    def test_generic_preference_probe_is_not_mislabeled_as_cross_recipe_transfer(self):
        pair = RecipePreferencePair(
            recipe_name="recipe_a",
            preference_name="p",
            actions=("opaque",),
        )
        self.assertNotIn("cross_recipe_transfer", _preference_tags(pair)["hypothesis_tags"])

    def _assert_phase_blocks(self, plan):
        by_phase = {}
        for event in plan.events:
            phase_id = event.tags.get("phase_id")
            self.assertIsNotNone(phase_id)
            by_phase.setdefault(phase_id, []).append(event)
        self.assertEqual(len(by_phase), len({event.tags["rung_idx"] for event in plan.events}))
        for phase_events in by_phase.values():
            climb = [event for event in phase_events if event.tags["phase_role"] == "climb"]
            settled = [event for event in phase_events if event.tags["phase_role"] == "settled"]
            self.assertTrue(climb)
            self.assertTrue(settled)
            self.assertEqual(phase_events, climb + settled)
            climb_labels = {event.pair.label for event in climb}
            self.assertTrue(all(event.pair.label in climb_labels for event in settled))
            self.assertTrue(all(
                event.tags["primary_probe"] is False
                or event.tags.get("event_type") == "random_ladder_delayed_recurrence_probe"
                for event in settled
            ))
            subgroup_size = climb[0].tags["subgroup_size"]
            self.assertEqual(len(climb), subgroup_size)
            self.assertGreaterEqual(len(settled), 2 * subgroup_size)
            self.assertLessEqual(len(settled), int(2.5 * subgroup_size))
            counts = {}
            for event in settled:
                counts[event.pair.label] = counts.get(event.pair.label, 0) + 1
            self.assertEqual(set(counts), climb_labels)
            if subgroup_size > 1:
                self.assertGreater(len(set(counts.values())), 1)

    def test_homogeneous_ladder_climbs_all_recipes_then_reuses_same_rung_pairs(self):
        plan = build_ladder_homogeneous(EvaluationConfig(n_recipes=4, ladder_rungs=3), seed=17)
        self._assert_phase_blocks(plan)
        for rung in range(3):
            events = [event for event in plan.events if event.tags["rung_idx"] == rung]
            climb = [event for event in events if event.tags["phase_role"] == "climb"]
            self.assertEqual({event.pair.recipe_name for event in climb}, set(plan.selected_recipes))
            self.assertEqual(len({event.pair.preference_name for event in climb}), 1)
            self.assertTrue(all(event.mode == ("observe" if rung == 0 else "assist") for event in climb))

    def test_heterogeneous_ladder_climbs_all_recipes_with_per_recipe_preference_changes(self):
        plan = build_ladder_heterogeneous(EvaluationConfig(n_recipes=4, ladder_rungs=3), seed=17)
        self._assert_phase_blocks(plan)
        pairs_by_recipe = {recipe: [] for recipe in plan.selected_recipes}
        for event in plan.events:
            if event.tags["phase_role"] == "climb":
                pairs_by_recipe[event.pair.recipe_name].append(event.pair.preference_name)
        self.assertTrue(all(len(values) == 3 and len(set(values)) == 3 for values in pairs_by_recipe.values()))
        for rung in range(3):
            climb = [
                event for event in plan.events
                if event.tags["rung_idx"] == rung and event.tags["phase_role"] == "climb"
            ]
            self.assertEqual({event.pair.recipe_name for event in climb}, set(plan.selected_recipes))
            self.assertTrue(all(event.mode == ("observe" if rung == 0 else "assist") for event in climb))

    def test_random_ladder_uses_variable_subgroups_and_phase_local_settling(self):
        plan = build_deployment_random(
            EvaluationConfig(n_recipes=8, ladder_rungs=5, random_ladder_min_recipes_per_rung=3),
            seed=1337,
        )
        self._assert_phase_blocks(plan)
        subgroup_sizes = {
            event.tags["subgroup_size"] for event in plan.events if event.tags["phase_role"] == "climb"
        }
        self.assertTrue(all(3 <= size <= 8 for size in subgroup_sizes))
        self.assertGreater(len(subgroup_sizes), 1)
        self.assertLess(min(subgroup_sizes), len(plan.selected_recipes))
        self.assertTrue(any(
            event.tags.get("event_type") in {
                "random_ladder_climb_preference_update",
                "random_ladder_climb_cross_recipe_transfer",
            }
            for event in plan.events
        ))

    def test_random_ladder_delayed_recurrence_probe_is_budget_matched_and_well_formed(self):
        config = EvaluationConfig(n_recipes=8, ladder_rungs=5, random_ladder_min_recipes_per_rung=3)
        plan = build_deployment_random(config, seed=1337)
        probes = [event for event in plan.events if event.tags.get("delayed_recurrence_probe")]
        self.assertEqual(len(probes), 3)

        for probe in probes:
            tags = probe.tags
            self.assertEqual(tags["event_type"], "random_ladder_delayed_recurrence_probe")
            self.assertEqual(tags["condition_family"], "delayed_recurrence_interference")
            self.assertTrue(tags["primary_probe"])
            self.assertGreaterEqual(
                tags["delayed_recurrence_historical_conflicting_preference_count"],
                config.random_ladder_recurrence_min_prior_conflicts,
            )
            self.assertGreaterEqual(
                tags["delayed_recurrence_intervening_event_count"],
                config.random_ladder_recurrence_min_intervening_events,
            )
            self.assertEqual(
                tags["delayed_recurrence_intervening_event_count"],
                tags["delayed_recurrence_required_intervening_event_count"],
            )
            self.assertEqual(
                tags["delayed_recurrence_required_intervening_event_count"],
                max(
                    config.random_ladder_recurrence_min_intervening_events,
                    max(
                        0,
                        tags["delayed_recurrence_predicted_grace_horizon_events"]
                        - tags["delayed_recurrence_current_reuse_gap_events"],
                    ) + tags["delayed_recurrence_prune_confirmation_events"],
                ),
            )

            phase_events = [
                event for event in plan.events if event.tags["phase_id"] == tags["phase_id"]
            ]
            climb = [event for event in phase_events if event.tags["phase_role"] == "climb"]
            settled = [event for event in phase_events if event.tags["phase_role"] == "settled"]
            self.assertEqual(climb[-1].pair.label, probe.pair.label)
            self.assertEqual(tags["phase_position"], tags["delayed_recurrence_intervening_event_count"])
            self.assertFalse(any(
                event.pair.label == probe.pair.label
                for event in settled[:tags["phase_position"]]
            ))
            self.assertEqual(len(settled), tags["settled_block_size"])

            earlier_conflicting_preferences = {
                event.pair.preference_name
                for event in plan.events
                if event.tags["rung_idx"] < tags["rung_idx"]
                and event.tags["phase_role"] == "climb"
                and event.pair.recipe_name == probe.pair.recipe_name
                and event.pair.preference_name != probe.pair.preference_name
            }
            self.assertGreaterEqual(
                len(earlier_conflicting_preferences),
                config.random_ladder_recurrence_min_prior_conflicts,
            )

    def test_reentry_metrics_separate_pruned_recovery_from_active_control(self):
        common = {
            "scheduled_reentry_probe": True,
            "hrc_robot_turn_count": 2,
            "hrc_robot_correct_count": 1,
            "hrc_robot_topk_hit_count": 2,
        }
        summary = reentry_stratification_summary([
            {
                **common,
                "mode": "assist",
                "reentry_probe_target_state_before": "pruned_exact_variant",
                "actual_reentry_from_pruned": True,
            },
            {
                **common,
                "mode": "assist",
                "reentry_probe_target_state_before": "active_exact_variant",
                "actual_reentry_from_pruned": False,
            },
            {
                **common,
                "mode": "observe",
                "reentry_probe_target_state_before": "pruned_exact_variant",
                "actual_reentry_from_pruned": False,
            },
        ])
        self.assertEqual(summary["n_scheduled_reentry_probes"], 3)
        self.assertEqual(summary["n_target_pruned_before_probe"], 2)
        self.assertEqual(summary["n_target_active_before_probe_control"], 1)
        self.assertEqual(summary["n_confirmed_reentry_from_pruned"], 1)
        self.assertEqual(summary["confirmed_reentry_from_pruned"]["live_top1"], 0.5)

    def test_paired_bootstrap_keeps_seed_pairs_and_metric_directions(self):
        def summary(seed, full_top1, baseline_top1, full_cost, baseline_cost):
            return {
                "scenario": "unit",
                "seed": seed,
                "per_baseline": {
                    "full": {"assist_only": {
                        "live_top1": full_top1,
                        "human_correction_rate": 1.0 - full_top1,
                        "testing_normalized_interaction_cost": full_cost,
                    }},
                    "adaptive_decay": {"assist_only": {
                        "live_top1": baseline_top1,
                        "human_correction_rate": 1.0 - baseline_top1,
                        "testing_normalized_interaction_cost": baseline_cost,
                    }},
                },
            }

        statistics = paired_bootstrap_summary({
            "unit/seed_1": summary(1, 0.8, 0.5, 1.1, 1.4),
            "unit/seed_2": summary(2, 0.6, 0.4, 1.2, 1.5),
        }, n_samples=200)
        comparison = statistics["by_scenario"]["unit"]["comparisons_against_full"]["adaptive_decay"]
        self.assertEqual(comparison["live_top1"]["n_paired_seeds"], 2)
        self.assertGreater(comparison["live_top1"]["mean"], 0.0)
        self.assertGreater(comparison["testing_normalized_interaction_cost"]["mean"], 0.0)
        self.assertEqual(set(comparison["live_top1"]["paired_seed_deltas_full_advantage"]), {"1", "2"})

    def test_topk_is_pooled_by_robot_turns_not_episode_count(self):
        rows = [
            {"hrc_robot_turn_count": 0, "hrc_robot_topk_hit_count": 0, "hrc_robot_correct_count": 0},
            {"hrc_robot_turn_count": 10, "hrc_robot_topk_hit_count": 8, "hrc_robot_correct_count": 8},
        ]
        metrics = aggregate_episode_metrics(rows)
        self.assertEqual(metrics["live_top1"], 0.8)
        self.assertEqual(metrics["live_topk"], 0.8)

    def test_oracle_summary_keeps_metrics_separate(self):
        summary = oracle_gap_summary([
            {"baseline": "full", "oracle_reference": "oracle", "metric": "live_top1", "scope": "assist_only", "oracle_advantage": 0.2, "regret_to_clairvoyant": 0.2},
            {"baseline": "full", "oracle_reference": "oracle", "metric": "human_correction_rate", "scope": "assist_only", "oracle_advantage": 0.1, "regret_to_clairvoyant": 0.1},
        ])
        by_metric = summary["by_baseline"]["full"]["by_metric"]
        self.assertEqual(set(by_metric), {"live_top1", "human_correction_rate"})
        self.assertEqual(by_metric["live_top1"]["mean_oracle_advantage"], 0.2)

    def test_memory_state_is_evaluated_from_pre_episode_active_or_pruned_membership(self):
        recipe_name, builder = next(iter(gen.recipe_library().items()))
        pair = materialize_pair(recipe_name, "identity", builder)
        agent = AdaptiveHRCAgent(Config(
            verbose=False,
            maxent_iters_cold=1,
            maxent_iters_warm=1,
        ))
        agent.start_demo()
        for observation in observations_from_actions(pair.actions):
            agent.observe_observation(observation)
        classification = agent.end_demo()
        self.assertIsNotNone(classification.recipe_id)
        name_to_rid = {recipe_name: classification.recipe_id}
        observed_pairs = {pair.label}
        observed_recipes = {recipe_name}
        key = _pair_key(agent, pair, name_to_rid)
        self.assertIsNotNone(key)
        assert key is not None
        self.assertEqual(
            _memory_state(agent, pair, name_to_rid, observed_pairs, observed_recipes),
            "active_memory",
        )

        entry = agent.decay.active[key]
        agent.decay._prune_entry(key, entry, now=agent.session_counter + 1, cycle=agent.retrain_cycle)
        self.assertEqual(
            _memory_state(agent, pair, name_to_rid, observed_pairs, observed_recipes),
            "pruned_memory",
        )

        row = assist_episode(
            agent,
            pair,
            name_to_rid,
            config=EvaluationConfig(),
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        )
        self.assertEqual(row["memory_state_before"], "pruned_memory")
        self.assertEqual(row["memory_state_after"], "active_memory")


if __name__ == "__main__":
    unittest.main()
