"""Unit contracts for HRC turn diagnostics."""
from __future__ import annotations

import unittest
import numpy as np

from src.hrc_simulation import simulate_episode


class LaterReferenceActionDiagnosticTests(unittest.TestCase):
    def test_wrong_prediction_records_only_a_later_reference_action_match(self):
        actions = ("a", "b", "c", "b")
        prefix: list[str] = []
        predictions = ("a", "c", "c", "b")

        def predict_distribution(current_prefix):
            return {predictions[len(current_prefix)]: 1.0}

        def observe_ground_truth(observation, _distribution, _predicted):
            prefix.append(observation)

        trace = simulate_episode(
            observations=actions,
            actual_actions=actions,
            current_prefix=lambda: tuple(prefix),
            predict_distribution=predict_distribution,
            observe_ground_truth=observe_ground_truth,
            top_k=1,
            min_probability=1e-12,
        )

        wrong_turn = trace.robot_turns[0]
        self.assertFalse(wrong_turn.correct_top_1)
        self.assertTrue(
            wrong_turn.matches_later_action
        )
        self.assertEqual(wrong_turn.matching_action_offset, 1)
        self.assertEqual(trace.summary.robot_wrong_count, 1)
        self.assertEqual(trace.summary.later_action_errors, 1)

    def test_ties_are_counted_compactly_without_losing_distribution(self):
        actions = ("a", "b", "a", "b")
        prefix: list[str] = []

        def predict_distribution(_current_prefix):
            return {"a": 0.5, "b": 0.5}

        def observe_ground_truth(observation, _distribution, _predicted):
            prefix.append(observation)

        trace = simulate_episode(
            observations=actions,
            actual_actions=actions,
            current_prefix=lambda: tuple(prefix),
            predict_distribution=predict_distribution,
            observe_ground_truth=observe_ground_truth,
            top_k=1,
            min_probability=1e-12,
            tie_rng=np.random.default_rng(41),
        )

        predictions = [*trace.robot_turns, *trace.human_shadow_turns]
        self.assertEqual(trace.summary.teacher_forced_top_1_tie_count, 4)
        self.assertEqual(
            trace.summary.robot_top_1_tie_count,
            len(trace.robot_turns),
        )
        self.assertTrue(all(turn.distribution == {"a": 0.5, "b": 0.5} for turn in predictions))


if __name__ == "__main__":
    unittest.main()
