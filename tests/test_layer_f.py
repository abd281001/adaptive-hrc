"""Layer F: tests for the live-evaluation foundations.

These primitives back every per-test metric upgrade in the publication
suite. Lock them down before patching the workers.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.evaluations import (
    LiveEpisodeRecord,
    LiveStepRecord,
    _make_pair,
    assistance_coverage,
    binary_decision_metrics,
    bwt_fwt,
    compute_budget_recorder,
    live_episode_metrics,
    pareto_front,
    preference_lock_purity,
    route_demo_with_metrics,
    run_observation_demo,
    update_recipe_id_map,
)
from src.models import Config


# F1: live_episode_metrics
class LiveEpisodeMetricsTests(unittest.TestCase):
    def _record(self, n=4, all_correct=False, true_rid="R1", lock_pid="P1"):
        steps = []
        for k in range(n):
            steps.append(LiveStepRecord(
                step=k,
                predicted=("a" if all_correct else None),
                actual="a",
                correct_top1=all_correct,
                correct_top3=all_correct,
                topk=("a", "b", "c"),
                inferred_recipe=true_rid,
                inferred_pref=lock_pid,
                posterior_entropy=0.2,
                posterior_confidence=0.8,
                assist_used=True,
            ))
        first_mismatch = None if all_correct else 0
        return LiveEpisodeRecord(
            pair_label="R/x",
            recipe_id="R",
            preference_name="x",
            memory_state="active_memory",
            mode="online",
            steps=tuple(steps),
            live_top1=(1.0 if all_correct else 0.0),
            live_top3=(1.0 if all_correct else 0.0),
            n=n,
            first_mismatch_step=first_mismatch,
        )

    def test_perfect_episode_zero_steps_to_lock(self):
        m = live_episode_metrics(self._record(all_correct=True), true_rid="R1")
        self.assertEqual(m["steps_to_lock"], 0)
        self.assertEqual(m["live_top1"], 1.0)
        self.assertEqual(m["assistance_coverage"], 1.0)
        self.assertEqual(m["posterior_correct_recipe"], 1.0)
        self.assertEqual(m["posterior_pref_lock_consistency"], 1.0)

    def test_empty_episode_returns_zeros(self):
        empty = LiveEpisodeRecord(
            pair_label="x", recipe_id="x", preference_name="x",
            memory_state="no_memory", mode="online",
            steps=(), live_top1=0.0, live_top3=0.0, n=0,
            first_mismatch_step=None,
        )
        m = live_episode_metrics(empty, true_rid="R1")
        self.assertEqual(m["live_top1"], 0.0)
        self.assertEqual(m["steps_to_lock"], -1)

    def test_lock_consistency_under_pid_flip(self):
        # Build a record where the pid flips midway.
        steps = []
        for k in range(6):
            pid = "P1" if k < 4 else "P2"
            steps.append(LiveStepRecord(
                step=k, predicted="a", actual="a",
                correct_top1=True, correct_top3=True,
                topk=("a",), inferred_recipe="R1", inferred_pref=pid,
                posterior_entropy=0.5, posterior_confidence=0.5,
                assist_used=False,
            ))
        rec = LiveEpisodeRecord(
            pair_label="x", recipe_id="R1", preference_name="x",
            memory_state="active_memory", mode="online",
            steps=tuple(steps), live_top1=1.0, live_top3=1.0, n=6,
            first_mismatch_step=None,
        )
        m = live_episode_metrics(rec, true_rid="R1")
        # Modal pid was P1 (4 of 6); consistency = 4/6.
        self.assertAlmostEqual(m["posterior_pref_lock_consistency"], 4 / 6, places=5)

    def test_abstention_error_rate(self):
        steps = [
            LiveStepRecord(
                step=0, predicted="a", actual="b",
                correct_top1=False, correct_top3=False,
                topk=("a",), inferred_recipe="R1", inferred_pref="P1",
                posterior_entropy=0.5, posterior_confidence=0.2,
                assist_used=False,
            ),
            LiveStepRecord(
                step=1, predicted="b", actual="b",
                correct_top1=True, correct_top3=True,
                topk=("b",), inferred_recipe="R1", inferred_pref="P1",
                posterior_entropy=0.5, posterior_confidence=0.9,
                assist_used=True,
            ),
        ]
        rec = LiveEpisodeRecord(
            pair_label="x", recipe_id="R1", preference_name="x",
            memory_state="active_memory", mode="online",
            steps=tuple(steps), live_top1=0.5, live_top3=0.5, n=2,
            first_mismatch_step=0,
        )
        m = live_episode_metrics(rec, true_rid="R1")
        self.assertEqual(m["abstention_error_rate"], 1.0)


# F2: compute_budget_recorder
class ComputeBudgetTests(unittest.TestCase):
    def test_records_wall_and_retrains(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        a = _make_pair(rname, "identity", fn)
        with compute_budget_recorder(agent) as bag:
            run_observation_demo(agent, a)
        self.assertGreater(bag["wall_s"], 0.0)
        self.assertGreaterEqual(bag["n_retrains_called"], 1)
        self.assertIn("mean_inference_latency_us", bag)
        self.assertIn("peak_active_variants", bag)


# F3: binary_decision_metrics
class BinaryDecisionTests(unittest.TestCase):
    def test_perfect_classifier_auroc_one(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        m = binary_decision_metrics(scores, labels)
        self.assertAlmostEqual(m["auroc"], 1.0, places=5)
        self.assertGreater(m["auprc"], 0.99)

    def test_random_classifier_auroc_around_half(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        labels = rng.integers(0, 2, 200)
        m = binary_decision_metrics(scores, labels)
        self.assertGreater(m["auroc"], 0.30)
        self.assertLess(m["auroc"], 0.70)

    def test_calibration_returns_bins(self):
        scores = [0.1, 0.2, 0.8, 0.9, 0.5, 0.5, 0.6, 0.7]
        labels = [0, 0, 1, 1, 0, 1, 0, 1]
        m = binary_decision_metrics(scores, labels, n_bins=4)
        self.assertGreater(len(m["calibration"]), 0)


# F4: assistance_coverage
class CoverageMetricTests(unittest.TestCase):
    def test_zero_records_yields_zero(self):
        self.assertEqual(assistance_coverage([]), 0.0)

    def test_mixed_assist_use(self):
        steps_on = [LiveStepRecord(step=0, predicted=None, actual="a",
                                   correct_top1=True, correct_top3=True,
                                   topk=(), inferred_recipe=None, inferred_pref=None,
                                   assist_used=True)]
        steps_off = [LiveStepRecord(step=0, predicted=None, actual="a",
                                    correct_top1=True, correct_top3=True,
                                    topk=(), inferred_recipe=None, inferred_pref=None,
                                    assist_used=False)]
        rec_on = LiveEpisodeRecord(
            pair_label="x", recipe_id="x", preference_name="x",
            memory_state="active_memory", mode="online",
            steps=tuple(steps_on), live_top1=1.0, live_top3=1.0, n=1, first_mismatch_step=None,
        )
        rec_off = LiveEpisodeRecord(
            pair_label="x", recipe_id="x", preference_name="x",
            memory_state="active_memory", mode="online",
            steps=tuple(steps_off), live_top1=1.0, live_top3=1.0, n=1, first_mismatch_step=None,
        )
        # 1 of 2 steps gated open.
        self.assertAlmostEqual(assistance_coverage([rec_on, rec_off]), 0.5)


# F5: bwt_fwt
class BwtFwtTests(unittest.TestCase):
    def test_no_forgetting_no_transfer(self):
        # 3 tasks, accuracy stays the same after each new task -> BWT=0.
        M = np.array([
            [0.9, 0.9, 0.9],
            [0.0, 0.8, 0.8],
            [0.0, 0.0, 0.7],
        ])
        out = bwt_fwt(M)
        self.assertAlmostEqual(out["bwt"], 0.0, places=5)

    def test_forgetting_yields_negative_bwt(self):
        M = np.array([
            [0.9, 0.5, 0.3],
            [0.0, 0.8, 0.6],
            [0.0, 0.0, 0.7],
        ])
        out = bwt_fwt(M)
        self.assertLess(out["bwt"], 0.0)


# F6: pareto_front
class ParetoTests(unittest.TestCase):
    def test_dominated_point_excluded(self):
        # A dominates B (higher x, equal y); B should be excluded.
        points = {"A": (0.9, 0.5), "B": (0.5, 0.5), "C": (0.4, 0.9)}
        keep = set(pareto_front(points, maximize=(True, True)))
        self.assertIn("A", keep)
        self.assertIn("C", keep)
        self.assertNotIn("B", keep)


# F7: route_demo_with_metrics
class RouteWithMetricsTests(unittest.TestCase):
    def test_unseen_returns_empty_metrics(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        rname, fn = list(gen.recipe_library().items())[0]
        pair = _make_pair(rname, "identity", fn)
        state, rec, metrics = route_demo_with_metrics(agent, pair)
        self.assertEqual(state, "no_memory")
        self.assertIsNone(rec)
        self.assertEqual(metrics, {})

    def test_known_recipe_returns_live_metrics(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        rname, fn = list(gen.recipe_library().items())[0]
        a = _make_pair(rname, "identity", fn)
        b = _make_pair(rname, "p1_prep_first", fn)
        name_to_rid: dict = {}
        run_observation_demo(agent, a)
        update_recipe_id_map(agent, name_to_rid, a)
        state, rec, metrics = route_demo_with_metrics(agent, b, name_to_rid)
        self.assertNotEqual(state, "no_memory")
        self.assertIsNotNone(rec)
        # All headline keys are present in the metrics dict.
        for key in ("live_top1", "live_top3", "post_divergence_top1",
                    "first_mismatch_step", "steps_to_lock",
                    "posterior_correct_recipe", "posterior_pref_lock_consistency",
                    "assistance_coverage", "mean_posterior_entropy",
                    "mean_posterior_confidence"):
            self.assertIn(key, metrics)


# preference_lock_purity (cross-episode aggregator)
class LockPurityTests(unittest.TestCase):
    def test_purity_on_clean_grouping(self):
        # Two episodes, both labelled preset "x", both with pid "P1".
        steps = [LiveStepRecord(step=0, predicted=None, actual="a",
                                correct_top1=True, correct_top3=True,
                                topk=(), inferred_recipe=None, inferred_pref="P1",
                                assist_used=False)]
        rec = LiveEpisodeRecord(
            pair_label="R/x", recipe_id="R", preference_name="x",
            memory_state="active_memory", mode="online",
            steps=tuple(steps), live_top1=1.0, live_top3=1.0, n=1,
            first_mismatch_step=None,
        )
        purity = preference_lock_purity([rec, rec], {"R/x": "x"})
        self.assertAlmostEqual(purity, 1.0)


if __name__ == "__main__":
    unittest.main()
