"""Phase 1: tests for the new orthogonal-axis workflow preference generator."""
from __future__ import annotations

import unittest

from src.representations import observations_from_actions
from src.environment import gen, task_goal_signature, validate_ordering
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
            (
                "ingredient_flow",
                "equipment_setup",
                "serving_setup",
                "container_loading_style",
                "appliance_shutdown_timing",
                "cook_start_timing",
                "cleanup_timing",
                "serving_priority",
            ),
        )

    def test_each_axis_has_two_values(self):
        for axis, values in AXIS_VALUES.items():
            self.assertEqual(len(values), 2, f"{axis} should have exactly two values")

    def test_workflow_preference_rejects_invalid_axis_value(self):
        with self.assertRaises(ValueError):
            WorkflowPreference(ingredient_flow="invalid_value")

    def test_label_is_deterministic(self):
        p = WorkflowPreference(
            ingredient_flow="mise_en_place",
            equipment_setup="just_in_time",
            serving_setup="frontloaded",
            cleanup_timing="as_soon_as_free",
        )
        self.assertIn("mise_en_place", p.label)
        self.assertIn("just_in_time", p.label)
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

    def test_every_emitted_variant_is_effectful_and_goal_preserving(self):
        for recipe_name, fn in gen.recipe_library().items():
            base = fn()
            base_goal = task_goal_signature(base)
            for preset_name in PREFERENCE_NAMES:
                actions = materialize(base, preset_name)
                self.assertTrue(
                    validate_ordering(actions, expected_goal=base_goal),
                    f"{recipe_name}/{preset_name} is invalid or changes the task outcome",
                )
                self.assertFalse(
                    any(obs.state == obs.next_state for obs in observations_from_actions(actions)),
                    f"{recipe_name}/{preset_name} contains a no-effect action",
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
        modified = materialize(base, "p7_clean_eager")

        self.assertTrue(validate_ordering(modified))
        self.assertEqual(sorted(modified), sorted(base))
        self.assertLess(
            modified.index("wash (bowl, washing_station)"),
            modified.index("serve (plate, serving_station)"),
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
    def test_preset_p1_uses_mise_en_place(self):
        self.assertEqual(
            PRESET_PREFERENCES["p1_mise_en_place"].ingredient_flow, "mise_en_place"
        )

    def test_preset_p7_uses_eager_cleanup(self):
        self.assertEqual(
            PRESET_PREFERENCES["p7_clean_eager"].cleanup_timing, "as_soon_as_free"
        )

    def test_preset_p10_combines_axes(self):
        p = PRESET_PREFERENCES["p10_mise_en_place_clean"]
        self.assertEqual(p.ingredient_flow, "mise_en_place")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")

    def test_preset_p11_combines_mise_serving_and_cleanup(self):
        p = PRESET_PREFERENCES["p11_mise_en_place_serving_clean"]
        self.assertEqual(p.ingredient_flow, "mise_en_place")
        self.assertEqual(p.serving_setup, "frontloaded")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")

    def test_preset_p12_uses_multi_stage_axes(self):
        p = PRESET_PREFERENCES["p12_multi_stage_reorganization"]
        self.assertEqual(p.ingredient_flow, "mise_en_place")
        self.assertEqual(p.serving_setup, "frontloaded")
        self.assertEqual(p.cleanup_timing, "as_soon_as_free")
        self.assertEqual(p.container_loading_style, "just_in_time")

    def test_new_axis_presets_are_available(self):
        self.assertEqual(PRESET_PREFERENCES["p5_shutdown_late"].appliance_shutdown_timing, "deferred")
        self.assertEqual(PRESET_PREFERENCES["p4_load_just_in_time"].container_loading_style, "just_in_time")
        self.assertEqual(PRESET_PREFERENCES["p6_deferred_cook_start"].cook_start_timing, "deferred")
        self.assertEqual(PRESET_PREFERENCES["p8_cleanup_before_serve"].serving_priority, "cleanup_before_serve")


class WorkflowPreferenceAPITests(unittest.TestCase):
    """Current modifier API contract."""

    def test_modify_recipe_returns_list(self):
        modifier = WorkflowPreferenceModifier()
        recipe_name, fn = next(iter(gen.recipe_library().items()))
        out = modifier.modify_recipe(fn(), PRESET_PREFERENCES["p1_mise_en_place"])
        self.assertIsInstance(out, list)
        self.assertIsNotNone(modifier.last_result)

    def test_modify_recipe_report_exposes_noop_axes(self):
        base = gen.recipe_library()["tomato_onion_soup_v1"]()
        report = materialize_with_report(base, "p12_multi_stage_reorganization")
        self.assertIsInstance(report.actions, list)
        self.assertTrue(set(report.applied_axes) | set(report.failed_axes) | set(report.unchanged_axes))
        self.assertEqual(set(report.axis_values), set(AXES))

    def test_identity_report_preserves_generator_ordering_exactly(self):
        base = gen.recipe_library()["tomato_onion_soup_v1"]()
        report = materialize_with_report(base, "identity")
        self.assertEqual(report.actions, base)
        self.assertEqual(report.applied_axes, [])
        self.assertEqual(report.failed_axes, [])
        self.assertEqual(set(report.unchanged_axes), set(AXES))

    def test_invalid_input_raises(self):
        modifier = WorkflowPreferenceModifier()
        with self.assertRaises(ValueError):
            modifier.modify_recipe(
                ["cut (tomato, prep_station)"],
                PRESET_PREFERENCES["identity"],
            )


if __name__ == "__main__":
    unittest.main()
