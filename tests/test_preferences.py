"""Phase 1: tests for the new orthogonal-axis workflow preference generator."""
from __future__ import annotations

import unittest

from src.representations import observe_actions
from src.environment import recipe_builders, task_goal_signature, validate_ordering
from src.preferences import (
    PREFERENCE_AXES,
    PREFERENCE_VALUES,
    PREFERENCE_IDS,
    PREFERENCES,
    Preference,
    apply_preference,
    apply_preset_actions,
    apply_preset,
)


class AxisDefinitionTests(unittest.TestCase):
    def test_axes_match_plan(self):
        self.assertEqual(
            PREFERENCE_AXES,
            (
                "prep",
                "equipment",
                "serving",
                "loading",
                "shutdown",
                "cook_start",
                "cleanup",
                "serve_order",
            ),
        )

    def test_each_axis_has_two_values(self):
        for axis, values in PREFERENCE_VALUES.items():
            self.assertEqual(len(values), 2, f"{axis} should have exactly two values")

    def test_workflow_preference_rejects_invalid_axis_value(self):
        with self.assertRaises(ValueError):
            Preference(prep="invalid_value")

    def test_label_is_deterministic(self):
        p = Preference(
            prep="prep_first",
            equipment="just_in_time",
            serving="early",
            cleanup="when_free",
        )
        self.assertIn("prep_first", p.label)
        self.assertIn("just_in_time", p.label)
        self.assertIn("early", p.label)
        self.assertIn("when_free", p.label)


class ValidateGenerationTests(unittest.TestCase):
    """Every recipe/preset candidate must pass ordering validation."""

    def test_every_recipe_every_preset_validates(self):
        offenders = []
        for recipe_name, fn in recipe_builders().items():
            base = fn()
            for preference_id in PREFERENCE_IDS:
                actions = apply_preset_actions(base, preference_id)
                if not validate_ordering(list(actions)):
                    offenders.append((recipe_name, preference_id))
        self.assertEqual(offenders, [],
                         f"invalid orderings emitted for: {offenders[:5]}")

    def test_action_set_preserved(self):
        for recipe_name, fn in recipe_builders().items():
            base = fn()
            for preference_id in PREFERENCE_IDS:
                actions = apply_preset_actions(base, preference_id)
                self.assertEqual(
                    sorted(actions), sorted(base),
                    f"action set changed under {preference_id} for {recipe_name}",
                )

    def test_every_emitted_variant_is_effectful_and_goal_preserving(self):
        for recipe_name, fn in recipe_builders().items():
            base = fn()
            base_goal = task_goal_signature(base)
            for preference_id in PREFERENCE_IDS:
                actions = apply_preset_actions(base, preference_id)
                self.assertTrue(
                    validate_ordering(actions, expected_goal=base_goal),
                    f"{recipe_name}/{preference_id} is invalid or changes the task outcome",
                )
                self.assertFalse(
                    any(obs.state == obs.next_state for obs in observe_actions(actions)),
                    f"{recipe_name}/{preference_id} contains a no-effect action",
                )

    def test_some_axes_actually_change_ordering(self):
        meaningful_axes_changed = 0
        for recipe_name, fn in recipe_builders().items():
            base = fn()
            ident = apply_preset_actions(base, "default")
            for preference_id in PREFERENCE_IDS:
                if preference_id == "default":
                    continue
                modified = apply_preset_actions(base, preference_id)
                if list(modified) != list(ident):
                    meaningful_axes_changed += 1
                    break
        self.assertGreater(meaningful_axes_changed, 5)

    def test_cleanup_eager_moves_cleanup_before_service_when_possible(self):
        base = recipe_builders()["tomato_onion_soup"]()
        modified = apply_preset_actions(base, "cleanup_when_free")

        self.assertTrue(validate_ordering(modified))
        self.assertEqual(sorted(modified), sorted(base))
        self.assertLess(
            modified.index("wash (bowl, washing_station)"),
            modified.index("serve (plate, serving_station)"),
        )

    def test_equipment_just_in_time_defers_storage_setup_until_needed(self):
        base = recipe_builders()["tomato_onion_soup"]()
        modified = apply_preference(
            base,
            Preference(
                prep="serial",
                equipment="just_in_time",
                serving="just_in_time",
                cleanup="after_serving",
            ),
        ).actions

        pot_setup = "transfer (pot, from=storage, to=cooking_station)"
        pot_first_use = "load (mixture, pot, cooking_station)"
        self.assertTrue(validate_ordering(modified))
        self.assertGreater(modified.index(pot_setup), base.index(pot_setup))
        self.assertEqual(modified.index(pot_setup) + 1, modified.index(pot_first_use))


class PresetCompositionTests(unittest.TestCase):
    def test_prep_first_uses_mise_en_place(self):
        self.assertEqual(
            PREFERENCES["prep_first"].prep, "prep_first"
        )

    def test_cleanup_when_free_uses_eager_cleanup(self):
        self.assertEqual(
            PREFERENCES["cleanup_when_free"].cleanup, "when_free"
        )

    def test_prep_cleanup_combines_axes(self):
        p = PREFERENCES["prep_first_cleanup"]
        self.assertEqual(p.prep, "prep_first")
        self.assertEqual(p.cleanup, "when_free")

    def test_prep_serving_cleanup_combines_axes(self):
        p = PREFERENCES["prep_first_serving_cleanup"]
        self.assertEqual(p.prep, "prep_first")
        self.assertEqual(p.serving, "early")
        self.assertEqual(p.cleanup, "when_free")

    def test_prep_loading_serving_cleanup_combines_axes(self):
        p = PREFERENCES["prep_loading_serving_cleanup"]
        self.assertEqual(p.prep, "prep_first")
        self.assertEqual(p.serving, "early")
        self.assertEqual(p.cleanup, "when_free")
        self.assertEqual(p.loading, "just_in_time")

    def test_new_axis_presets_are_available(self):
        self.assertEqual(PREFERENCES["shutdown_late"].shutdown, "delayed")
        self.assertEqual(PREFERENCES["loading_jit"].loading, "just_in_time")
        self.assertEqual(PREFERENCES["cook_start_late"].cook_start, "delayed")
        self.assertEqual(PREFERENCES["cleanup_first"].serve_order, "cleanup_first")


class PreferenceApiTests(unittest.TestCase):
    """Current preference API contract."""

    def test_apply_preference_returns_report(self):
        recipe_name, fn = next(iter(recipe_builders().items()))
        report = apply_preference(fn(), PREFERENCES["prep_first"])
        self.assertIsInstance(report.actions, list)

    def test_modify_recipe_report_exposes_noop_axes(self):
        base = recipe_builders()["tomato_onion_soup"]()
        report = apply_preset(base, "prep_loading_serving_cleanup")
        self.assertIsInstance(report.actions, list)
        self.assertTrue(set(report.applied) | set(report.failed) | set(report.unchanged))
        self.assertEqual(set(report.values), set(PREFERENCE_AXES))

    def test_identity_report_preserves_generator_ordering_exactly(self):
        base = recipe_builders()["tomato_onion_soup"]()
        report = apply_preset(base, "default")
        self.assertEqual(report.actions, base)
        self.assertEqual(report.applied, [])
        self.assertEqual(report.failed, [])
        self.assertEqual(set(report.unchanged), set(PREFERENCE_AXES))

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            apply_preference(
                ["cut (tomato, prep_station)"],
                PREFERENCES["default"],
            )


if __name__ == "__main__":
    unittest.main()
