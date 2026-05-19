"""Phase 1: tests for the new orthogonal-axis workflow preference generator."""
from __future__ import annotations

import unittest

from src.representations import roles_from_actions
from src.environment import gen, validate_ordering
from src.posterior import role_order_features
from src.preferences import (
    AXES,
    AXIS_VALUES,
    PREFERENCE_NAMES,
    PRESET_PREFERENCES,
    WorkflowPreference,
    WorkflowPreferenceModifier,
    materialize,
    materialize_with_report,
)


class AxisDefinitionTests(unittest.TestCase):
    def test_axes_match_plan(self):
        self.assertEqual(
            AXES,
            ("ingredient_flow", "equipment_setup", "serving_setup", "cleanup_timing"),
        )

    def test_each_axis_has_two_values(self):
        for axis, values in AXIS_VALUES.items():
            self.assertEqual(len(values), 2, f"{axis} should have exactly two values")

    def test_workflow_preference_rejects_invalid_axis_value(self):
        with self.assertRaises(ValueError):
            WorkflowPreference(ingredient_flow="invalid_value")

    def test_label_is_deterministic(self):
        p = WorkflowPreference(
            ingredient_flow="prep_first",
            equipment_setup="frontloaded",
            serving_setup="frontloaded",
            cleanup_timing="as_soon_as_free",
        )
        self.assertIn("prep_first", p.label)
        self.assertIn("frontloaded", p.label)
        self.assertIn("as_soon_as_free", p.label)


class ValidateGenerationTests(unittest.TestCase):
    """Every emitted candidate must pass validate_ordering, for every recipe and
    every preset."""

    def test_every_recipe_every_preset_validates(self):
        offenders = []
        for recipe_name, fn in gen.recipe_library().items():
            base = fn()
            for preset_name in PREFERENCE_NAMES:
                actions = materialize(base, preset_name)
                if not validate_ordering(list(actions)):
                    offenders.append((recipe_name, preset_name))
        self.assertEqual(offenders, [],
                         f"invalid orderings emitted for: {offenders[:5]}")

    def test_action_set_preserved(self):
        # No insertions / deletions / rewrites — only reorderings.
        for recipe_name, fn in gen.recipe_library().items():
            base = fn()
            for preset_name in PREFERENCE_NAMES:
                actions = materialize(base, preset_name)
                self.assertEqual(
                    sorted(actions), sorted(base),
                    f"action set changed under {preset_name} for {recipe_name}",
                )

    def test_some_axes_actually_change_ordering(self):
        # The generator should produce at least one ordering different from
        # identity for each recipe with non-trivial structure (i.e. recipes
        # that have plate-staging or wash actions).
        meaningful_axes_changed = 0
        for recipe_name, fn in gen.recipe_library().items():
            base = fn()
            ident = materialize(base, "identity")
            for preset_name in PREFERENCE_NAMES:
                if preset_name == "identity":
                    continue
                modified = materialize(base, preset_name)
                if list(modified) != list(ident):
                    meaningful_axes_changed += 1
                    break
        # Out of ~15 recipes, the vast majority have wash or plate-staging
        # actions and should produce at least one non-identity ordering.
        self.assertGreater(meaningful_axes_changed, 5)

    def test_cleanup_eager_moves_cleanup_before_service_when_possible(self):
        base = gen.recipe_library()["tomato_onion_soup_v1"]()
        modified = materialize(base, "p3_clean_eager")

        self.assertTrue(validate_ordering(modified))
        self.assertEqual(sorted(modified), sorted(base))
        self.assertLess(
            modified.index("wash (bowl, washing_station)"),
            modified.index("serve (plate, serving_station)"),
        )
        self.assertLess(
            role_order_features(roles_from_actions(modified))["cleanup_delay_after_last_use"],
            role_order_features(roles_from_actions(base))["cleanup_delay_after_last_use"],
        )

    def test_equipment_just_in_time_defers_storage_setup_until_needed(self):
        base = gen.recipe_library()["tomato_onion_soup_v1"]()
        modified = WorkflowPreferenceModifier().modify_recipe(
            base,
            WorkflowPreference(
                ingredient_flow="serial",
                equipment_setup="just_in_time",
                serving_setup="just_in_time",
                cleanup_timing="after_service",
            ),
        )

        pot_setup = "transfer (pot, from=storage, to=cooking_station)"
        pot_first_use = "load (mixture, pot, cooking_station)"
        self.assertTrue(validate_ordering(modified))
        self.assertGreater(modified.index(pot_setup), base.index(pot_setup))
        self.assertEqual(modified.index(pot_setup) + 1, modified.index(pot_first_use))


class PresetCompositionTests(unittest.TestCase):
    def test_preset_p1_uses_prep_first(self):
        self.assertEqual(
            PRESET_PREFERENCES["p1_prep_first"].ingredient_flow, "prep_first"
        )

    def test_preset_p3_uses_eager_cleanup(self):
        self.assertEqual(
            PRESET_PREFERENCES["p3_clean_eager"].cleanup_timing, "as_soon_as_free"
        )

    def test_preset_p4_combines_axes(self):
        p = PRESET_PREFERENCES["p4_prep_clean"]
        self.assertEqual(p.ingredient_flow, "prep_first")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")

    def test_preset_p5_combines_prep_stage_and_cleanup(self):
        p = PRESET_PREFERENCES["p5_prep_stage_clean"]
        self.assertEqual(p.ingredient_flow, "prep_first")
        self.assertEqual(p.equipment_setup, "just_in_time")
        self.assertEqual(p.serving_setup, "frontloaded")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")

    def test_preset_p6_uses_all_non_default_axes(self):
        p = PRESET_PREFERENCES["p6_full_restructure"]
        self.assertEqual(p.ingredient_flow, "prep_first")
        self.assertEqual(p.equipment_setup, "frontloaded")
        self.assertEqual(p.serving_setup, "frontloaded")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")


class APICompatTests(unittest.TestCase):
    """Modifier surface mirrors the legacy module enough for consumer migration."""

    def test_modify_recipe_returns_list(self):
        modifier = WorkflowPreferenceModifier()
        recipe_name, fn = next(iter(gen.recipe_library().items()))
        out = modifier.modify_recipe(fn(), PRESET_PREFERENCES["p1_prep_first"])
        self.assertIsInstance(out, list)
        self.assertIsNotNone(modifier.last_result)

    def test_modify_recipe_report_exposes_noop_axes(self):
        base = gen.recipe_library()["tomato_onion_soup_v1"]()
        report = materialize_with_report(base, "p6_full_restructure")
        self.assertIsInstance(report.actions, list)
        self.assertTrue(set(report.applied_axes) | set(report.failed_axes) | set(report.unchanged_axes))
        self.assertEqual(set(report.axis_values), set(AXES))

    def test_invalid_input_raises(self):
        modifier = WorkflowPreferenceModifier()
        with self.assertRaises(ValueError):
            modifier.modify_recipe(
                ["cut (tomato, prep_station)"],
                PRESET_PREFERENCES["identity"],
            )


if __name__ == "__main__":
    unittest.main()
