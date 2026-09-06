"""Contracts for opt-in component counterfactual instrumentation."""
from __future__ import annotations

import unittest

from src.evaluation import (
    _COUNTERFACTUAL_POLICY_NAMES,
    _counterfactual_policy_record,
    _validate_component_counterfactual_rows,
    build_plan,
    parse_args,
    run_stream,
    EvalSettings,
    ScheduleSettings,
)
from src.models import Settings


class ComponentDiagnosticTests(unittest.TestCase):
    def test_validated_lightweight_sequence_setting_is_the_default(self):
        self.assertEqual(Settings().latent_strategy_rank, 8)
        self.assertEqual(Settings().latent_strategy_knn, 3)
        self.assertEqual(Settings().latent_strategy_sequence_weight, 0.5)

    def test_cli_uses_a_separate_experiment_label(self):
        config = parse_args([
            "--component-counterfactuals",
            "--baselines", "full",
            "--no-oracle",
        ])
        self.assertTrue(config.component_counterfactuals)
        self.assertEqual(
            config.experiment, "component_counterfactual_diagnostic",
        )

    def test_policy_record_is_rng_free_and_tie_aware(self):
        record = _counterfactual_policy_record(
            {"b": 0.5, "a": 0.5}, "b", 1e-6,
        )
        self.assertEqual(record["argmax_action"], "a")
        self.assertEqual(record["top_tie_size"], 2)
        self.assertEqual(record["expected_top_1"], 0.5)
        self.assertAlmostEqual(record["actual_action_probability"], 0.5)

    def test_integrity_gate_rejects_missing_counterfactuals(self):
        with self.assertRaisesRegex(RuntimeError, "no counterfactual policies"):
            _validate_component_counterfactual_rows([{
                "semantic_gate_outcome": "exact_policy",
                "latent_gate_outcome": "blocked_confirmation",
            }])

    def test_integrity_gate_accepts_complete_row(self):
        policy = {
            "argmax_action": "a",
            "argmax_probability": 1.0,
            "actual_action_probability": 1.0,
            "actual_action_nll": 0.0,
            "actual_in_top_tie": True,
            "expected_top_1": 1.0,
            "top_tie_size": 1,
        }
        _validate_component_counterfactual_rows([{
            "semantic_gate_outcome": "exact_policy",
            "latent_gate_outcome": "applied",
            "counterfactual_policies": {
                name: dict(policy)
                for name in _COUNTERFACTUAL_POLICY_NAMES
            },
        }])


class ComponentCounterfactualParityTests(unittest.TestCase):
    """The diagnostic must be provably inert on the deployed policy.

    counterfactual_policies() and score_snapshot() are engineered to be
    state- and RNG-free (see the isolation contract in
    test_maxent_irl.py::test_counterfactual_policies_are_normalized_and_do_not_mutate_decision_state),
    but that unit test only checks one call in isolation. A future change to
    _feasible_policy or score_snapshot could reach back into shared gate
    state or the tie-break RNG without any single-call test noticing, because
    none of them run the same plan twice and diff the outcome across an
    entire stream. This does exactly that: identical plan, identical seed,
    only the flag differs.
    """

    @staticmethod
    def _config(component_counterfactuals: bool) -> EvalSettings:
        return EvalSettings(
            seeds=(19,),
            scenarios=("homogeneous",),
            baselines=("full",),
            include_oracle=False,
            recipe_count=2,
            schedule=ScheduleSettings(
                panel_size=2, phases=2, demos=42,
                min_recipes=1, max_recipes=2,
            ),
            audit_period=0,
            show_eta=False,
            component_counterfactuals=component_counterfactuals,
            model_settings={"irl_cold_steps": 1, "irl_warm_steps": 1},
        )

    def test_enabling_diagnostics_does_not_change_deployed_predictions_or_rng(self):
        plan = build_plan("homogeneous", self._config(False), 19)
        off = run_stream("full", plan, self._config(False))
        on = run_stream("full", plan, self._config(True))

        self.assertGreater(len(off.turn_rows), 0)
        self.assertEqual(len(off.turn_rows), len(on.turn_rows))
        self.assertTrue(any(
            row.get("counterfactual_policies") for row in on.turn_rows
        ))
        self.assertFalse(any(
            row.get("counterfactual_policies") for row in off.turn_rows
        ))

        tracked_fields = (
            "predicted", "actual", "correct_top_1", "correct_top_k",
            "margin", "confidence", "entropy", "raw_confidence",
            "final_confidence", "final_margin", "final_entropy",
            "semantic_fallback_used", "latent_strategy_used",
        )
        for off_turn, on_turn in zip(off.turn_rows, on.turn_rows):
            for field in tracked_fields:
                self.assertEqual(
                    off_turn.get(field), on_turn.get(field),
                    f"turn {off_turn.get('turn_index')} field {field!r} "
                    "differs with component_counterfactuals enabled",
                )

        # An extra draw inside the shadow branch would desynchronize every
        # tie-break decision after it; the isolation contract requires the
        # deployed RNG stream to be untouched, not merely its visible output.
        self.assertEqual(
            off.agent._tie_break_rng.bit_generator.state,
            on.agent._tie_break_rng.bit_generator.state,
        )


if __name__ == "__main__":
    unittest.main()
