"""Smoke tests for the single-file evaluation harness."""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import src.evaluations as evaluations
from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.evaluations import (
    BASELINES,
    CORE_BASELINES,
    DEFAULT_MAIN_TESTS,
    EXPERIMENTS,
    LiveEpisodeRecord,
    LiveStepRecord,
    MEMORY_LABELS,
    PhaseBUserModelConfig,
    ScenarioEvent,
    aggregate_live_episodes,
    build_pairs,
    build_cross_product_transfer_spec,
    build_realistic_phase_b_events,
    derive_memory_state,
    execute_scenario_event,
    four_cell_live_metrics,
    main,
    preference_variant_quality,
    resolve_runtime_config,
    route_demo_by_memory_state,
    run_observation_demo,
    run_online_demo,
    run_online_demo_with_oracle,
    scenario_event_stream_signature,
    update_recipe_id_map,
    _make_pair,
)
from src.models import Config


class EvaluationsSmokeTest(unittest.TestCase):
    def test_recipe_library_is_single_source_of_truth(self):
        names = list(gen.recipe_library())
        self.assertGreaterEqual(len(names), 14)
        self.assertEqual(len(set(names)), len(names))

    def test_core_baselines(self):
        self.assertEqual(
            set(CORE_BASELINES),
            {
                "full", "adaptive", "fixed_decay", "no_decay",
                "latest_only", "experience_replay_bc", "bigram",
            },
        )
        for b in CORE_BASELINES:
            self.assertIn(b, BASELINES)

    def test_planned_groups_use_core_baseline_set(self):
        self.assertTrue(set(evaluations.BASELINES_2C).issubset(set(CORE_BASELINES)))
        self.assertTrue(set(evaluations.BASELINES_STRESS).issubset(set(CORE_BASELINES)))
        self.assertIn("experience_replay_bc", CORE_BASELINES)

    def test_experiments_registry_has_expected_entries(self):
        slugs = list(EXPERIMENTS.keys())
        self.assertEqual(
            slugs,
            [
                "test_1_short_term",
                "group_a_main_settled_use",
                "group_b_reentry_decay",
                "test_2_long_term",
                "test_3_decay_probes",
                # test_4a_mwr_window deleted (Phase 3 cleanup): redundant with
                # test_4a_frequency_sweep which already covers MWR-window dynamics.
                "test_4a_frequency_sweep",
                "test_4b_reentry_gap",
                "test_4c_disambig_threshold",
                "test_4d_anchor_sweep",
                # test_4e_cycle_width deleted (Phase 3 cleanup): cycle_width is
                # now a Phase-B mixture parameter inside test_2_long_term.
                "test_5_adversarial_stress",
                "test_6_sparse_settle",
                "test_7_zipfian_user_model",
                "test_8_cross_recipe_transfer",
                "test_9_axis_holdout",
                "test_10_novel_combo_holdout",
            ],
        )

    def test_default_main_tests_use_grouped_phase_a_b_run(self):
        self.assertIn("group_a_main_settled_use", DEFAULT_MAIN_TESTS)
        self.assertIn("group_b_reentry_decay", DEFAULT_MAIN_TESTS)
        self.assertNotIn("test_2_long_term", DEFAULT_MAIN_TESTS)
        self.assertNotIn("test_3_decay_probes", DEFAULT_MAIN_TESTS)
        self.assertNotIn("test_4c_disambig_threshold", DEFAULT_MAIN_TESTS)
        self.assertNotIn("test_4d_anchor_sweep", DEFAULT_MAIN_TESTS)
        self.assertNotIn("test_4b_reentry_gap", DEFAULT_MAIN_TESTS)

    def test_grouped_runs_keep_legacy_tests_debuggable(self):
        self.assertIn("test_3_decay_probes", EXPERIMENTS)
        self.assertIn("test_4b_reentry_gap", EXPERIMENTS)

    def test_memory_labels(self):
        self.assertEqual(
            set(MEMORY_LABELS),
            {"active_memory", "pruned_memory", "no_memory", "same_recipe_new_preference"},
        )

    def test_runtime_config_recomputes_derived_counts(self):
        # PREFS_PER_RECIPE = 7 after the workflow-preset unification.
        # All derived counts scale with len(PREFERENCE_NAMES).
        cfg = resolve_runtime_config({"num_recipes_per_seed": 4, "mwr_window": 20})
        self.assertEqual(cfg["short_term_demo_count"], 4)
        self.assertEqual(cfg["long_explore"], 4 * 7 * 2)        # = 56
        self.assertEqual(cfg["long_settle"], 4 * 7 * 3)         # = 84
        self.assertEqual(cfg["phase_b_long"], (4 * 7 * 3) * 2)  # = 168
        self.assertEqual(cfg["sparse_pool_size"], 4 * 7 // 2)   # = 14
        self.assertEqual(cfg["sparse_settle_size"], 4)
        self.assertEqual(cfg["cycle_width_sweep"], (1, 2, 4))

    def test_test_1_runs_end_to_end_on_one_seed(self):
        tmp = tempfile.mkdtemp(prefix="hrc_eval_smoke_")
        try:
            manifest = main(
                tests=["test_1_short_term"],
                seeds=[1337],
                workers=1,
                root=tmp,
                timestamp_subdir=False,
                quick=True,
            )
            self.assertIn("test_1_short_term", manifest["completed_tests"])
            test_dir = os.path.join(tmp, "test_1_short_term")
            self.assertTrue(os.path.isdir(test_dir))
            seed_dir = os.path.join(test_dir, "seed_1337")
            self.assertTrue(os.path.isdir(seed_dir))
            metrics_path = os.path.join(seed_dir, "metrics.json")
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path, "r", encoding="utf-8") as f:
                row = json.load(f)
            self.assertIn("wall_s", row)
            self.assertIn("memory_state_breakdown", row)
            for baseline in CORE_BASELINES:
                self.assertIn(f"{baseline}_top1", row, f"missing top1 for {baseline}")
                self.assertIn(f"{baseline}_top3", row, f"missing top3 for {baseline}")
            self.assertTrue(os.path.exists(os.path.join(seed_dir, "figure.png")))
            self.assertTrue(os.path.exists(os.path.join(test_dir, "aggregate", "aggregate.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "run_manifest.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class LiveOnlineHarnessTest(unittest.TestCase):
    """Phase 0A: live-evaluation harness primitives."""

    def _make_agent(self) -> AdaptiveHRCAgent:
        return AdaptiveHRCAgent(cfg=Config(verbose=False))

    def _two_pairs(self):
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        a = _make_pair(rname, "identity", fn)
        b = _make_pair(rname, "p1_prep_first", fn)
        return a, b

    def test_run_online_demo_does_not_call_start_demo(self):
        agent = self._make_agent()
        a, _ = self._two_pairs()
        run_observation_demo(agent, a)
        self.assertEqual(agent.mode, "online")
        rec = run_online_demo(agent, a, topk=3)
        # Mode must remain ONLINE throughout (no start_demo path was hit).
        self.assertEqual(agent.mode, "online")
        self.assertIsInstance(rec, LiveEpisodeRecord)
        self.assertEqual(rec.n, len(a.actions))
        # All step records present.
        self.assertEqual(len(rec.steps), rec.n)
        for st in rec.steps:
            self.assertIsInstance(st, LiveStepRecord)
            self.assertIsNotNone(st.actual)
            self.assertIsInstance(st.topk, tuple)

    def test_route_unseen_recipe_uses_observation(self):
        agent = self._make_agent()
        a, _ = self._two_pairs()
        state, rec = route_demo_by_memory_state(agent, a)
        self.assertEqual(state, "no_memory")
        self.assertIsNone(rec)
        self.assertGreater(len(agent.memory.variants), 0)

    def test_unseen_route_does_not_call_full_sequence_memory_probe(self):
        agent = self._make_agent()
        a, _ = self._two_pairs()
        original = evaluations.derive_memory_state

        def fail_if_called(*args, **kwargs):
            raise AssertionError("derive_memory_state should not decide unseen observe routing")

        evaluations.derive_memory_state = fail_if_called
        try:
            state, rec = evaluations.route_demo_by_memory_state(agent, a, name_to_rid={})
        finally:
            evaluations.derive_memory_state = original
        self.assertEqual(state, "no_memory")
        self.assertIsNone(rec)

    def test_route_known_recipe_uses_online(self):
        agent = self._make_agent()
        a, b = self._two_pairs()
        name_to_rid: dict = {}
        # First demo establishes the recipe; capture its agent-side rid.
        run_observation_demo(agent, a)
        update_recipe_id_map(agent, name_to_rid, a)
        # Same recipe + different preference should now route to online.
        state, rec = route_demo_by_memory_state(agent, b, name_to_rid=name_to_rid)
        self.assertIn(state, ("active_memory", "pruned_memory", "same_recipe_new_preference"))
        self.assertIsNotNone(rec)
        self.assertEqual(rec.mode, "online")
        self.assertEqual(rec.memory_state, state)
        self.assertGreaterEqual(rec.live_top1, 0.0)
        self.assertLessEqual(rec.live_top1, 1.0)

    def test_aggregate_live_episodes_smoke(self):
        agent = self._make_agent()
        a, b = self._two_pairs()
        run_observation_demo(agent, a)
        rec_a = run_online_demo(agent, a, topk=3)
        rec_b = run_online_demo(agent, b, topk=3)
        agg = aggregate_live_episodes([rec_a, rec_b])
        self.assertEqual(agg["n_episodes"], 2)
        self.assertEqual(agg["n"], rec_a.n + rec_b.n)
        self.assertGreaterEqual(agg["live_top1"], 0.0)
        self.assertLessEqual(agg["live_top1"], 1.0)

    def test_route_consistency_with_derive_memory_state(self):
        agent = self._make_agent()
        a, _ = self._two_pairs()
        name_to_rid: dict = {}
        run_observation_demo(agent, a)
        update_recipe_id_map(agent, name_to_rid, a)
        truth_state = derive_memory_state(agent, a.recipe_name, list(a.actions), name_to_rid)
        route_state, _ = route_demo_by_memory_state(agent, a, name_to_rid=name_to_rid)
        # The router must agree with the ground-truth memory-state probe at the
        # moment of routing (before mutation).
        self.assertEqual(route_state, truth_state)

    def test_explicit_scenario_event_controls_observe_vs_assist(self):
        agent = self._make_agent()
        a, b = self._two_pairs()
        name_to_rid: dict = {}
        observe_event = ScenarioEvent(a, "observe", "phase_a", "U1", {"event_type": "test_observe"})
        state, rec, metrics = execute_scenario_event(agent, observe_event, name_to_rid)
        self.assertEqual(state, "no_memory")
        self.assertIsNone(rec)
        self.assertEqual(metrics, {})
        self.assertIn(a.recipe_name, name_to_rid)

        assist_event = ScenarioEvent(b, "assist", "phase_b", "U2", {"event_type": "test_assist"})
        state, rec, metrics = execute_scenario_event(agent, assist_event, name_to_rid)
        self.assertIn(state, ("active_memory", "same_recipe_new_preference", "pruned_memory"))
        self.assertIsNotNone(rec)
        self.assertIn("live_top1", metrics)

    def test_phase_b_user_model_is_seed_reproducible(self):
        _, pairs = build_pairs(np.random.default_rng(123), n_recipes=4)
        rng1 = np.random.default_rng(456)
        rng2 = np.random.default_rng(456)
        phase_a = pairs[:8]
        cfg = PhaseBUserModelConfig(new_recipe_observation_rate=0.0)
        e1 = build_realistic_phase_b_events(pairs, phase_a, 12, rng1, cfg)
        e2 = build_realistic_phase_b_events(pairs, phase_a, 12, rng2, cfg)
        self.assertEqual([(e.pair.label, e.mode, e.user_id, e.tags["event_type"]) for e in e1],
                         [(e.pair.label, e.mode, e.user_id, e.tags["event_type"]) for e in e2])
        self.assertEqual(scenario_event_stream_signature(e1), scenario_event_stream_signature(e2))

    def test_phase_a_events_precede_phase_b_events(self):
        _, pairs = build_pairs(np.random.default_rng(123), n_recipes=4)
        phase_a = [
            ScenarioEvent(p, "observe", "phase_a", "U1", {"event_type": "phase_a"})
            for p in pairs[:6]
        ]
        phase_b = build_realistic_phase_b_events(pairs, [e.pair for e in phase_a], 5, np.random.default_rng(789))
        events = phase_a + phase_b
        boundary = len(phase_a)
        self.assertTrue(all(e.phase == "phase_a" for e in events[:boundary]))
        self.assertTrue(all(e.phase == "phase_b" for e in events[boundary:]))

    def test_all_baselines_can_share_identical_event_stream_signature(self):
        _, pairs = build_pairs(np.random.default_rng(321), n_recipes=4)
        phase_a = pairs[:6]
        events = build_realistic_phase_b_events(pairs, phase_a, 8, np.random.default_rng(654))
        signatures = {b: scenario_event_stream_signature(events) for b in CORE_BASELINES}
        self.assertEqual(len(set(signatures.values())), 1)

    def test_factorial_test8_offdiag_count(self):
        spec = build_cross_product_transfer_spec(np.random.default_rng(1337))
        self.assertIsNone(spec.get("error"))
        self.assertEqual(len(spec["diagonal"]), 7)
        self.assertEqual(len(spec["offdiag"]), 42)

    def test_recipe_preference_oracle_is_upper_bound_for_scored_episode(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False, ablation_oracle_recipe_and_preference_label=True))
        a, b = self._two_pairs()
        name_to_rid: dict = {}
        run_observation_demo(agent, a)
        update_recipe_id_map(agent, name_to_rid, a)
        pid = agent.last_pref_id
        rid = name_to_rid[a.recipe_name]
        rec = run_online_demo_with_oracle(agent, b, oracle_recipe_id=rid, oracle_pref_id=pid)
        self.assertEqual(rec.live_top1, 1.0)

    def test_duplicate_preference_variants_are_reported(self):
        a, _ = self._two_pairs()
        duplicate = type(a)(a.recipe_name, "duplicate_pref", f"{a.recipe_name}/duplicate_pref", a.actions, False)
        q = preference_variant_quality([a, duplicate])
        self.assertEqual(q["per_recipe"][a.recipe_name]["effective_preference_count"], 1)
        self.assertGreaterEqual(q["n_duplicate_pairs"], 1)

    def test_four_cell_outputs_include_support_counts(self):
        out = four_cell_live_metrics({})
        for cell in ("seen_seen", "seen_unseen", "unseen_seen", "unseen_unseen"):
            self.assertIn(f"n_episodes_{cell}", out)


if __name__ == "__main__":
    unittest.main()
