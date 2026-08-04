"""Environment sanity checks.

These are intentionally light: the refactor moves the research
contributions (decay, memory, disambiguation, agent) above the PDDL
layer, so the environment is only required to (a) build a
StateTracker, (b) expose a recipe generator, (c) apply a handful of
actions without raising.
"""
import unittest

from src.environment import StateTracker, gen, validate_ordering


class EnvironmentTests(unittest.TestCase):
    def test_state_tracker_is_constructible(self):
        tracker = StateTracker()
        self.assertGreater(tracker.n_features, 0)

    def test_recipe_generator_has_recipes(self):
        recipe_methods = [name for name in dir(gen) if name.startswith("generate_")]
        self.assertGreaterEqual(len(recipe_methods), 10)

    def test_state_tracker_applies_basic_actions(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (pot, from=storage, to=cooking_station)")
        tracker.apply_action("transfer (tomato, from=storage, to=prep_station)")
        tracker.apply_action("cut (tomato, prep_station)")
        self.assertEqual(tracker.get_item_location("pot"), "cooking_station")
        self.assertEqual(tracker.get_item_location("tomato"), "prep_station")
        self.assertEqual(tracker.get_feature("tomato_cut"), 1)

    def test_move_container_keeps_contained_items_without_direct_location(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (bowl, from=storage, to=prep_station)")
        tracker.apply_action("transfer (tomato, from=storage, to=prep_station)")
        tracker.apply_action("load (tomato, bowl, prep_station)")
        tracker.apply_action("move_container (bowl, from=prep_station, to=plating_station)")
        self.assertEqual(tracker.get_item_location("bowl"), "plating_station")
        self.assertIsNone(tracker.get_item_location("tomato"))
        self.assertEqual(tracker.is_contained("tomato"), "bowl")

    def test_combine_marks_mixture_membership(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (bowl, from=storage, to=prep_station)")
        tracker.apply_action("transfer (tomato, from=storage, to=prep_station)")
        tracker.apply_action("transfer (onion, from=storage, to=prep_station)")
        tracker.apply_action("load (tomato, bowl, prep_station)")
        tracker.apply_action("load (onion, bowl, prep_station)")
        tracker.apply_action("combine (bowl, prep_station)")
        self.assertEqual(tracker.get_feature("bowl_contains_mixture"), 1)
        self.assertEqual(tracker.get_feature("tomato_in_mixture"), 1)
        self.assertEqual(tracker.get_feature("onion_in_mixture"), 1)
        self.assertEqual(tracker.get_feature("mixture_at_prep_station"), 0)

    def test_blend_requires_blender_on(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (glass, from=storage, to=blending_station)")
        tracker.apply_action("transfer (banana, from=storage, to=blending_station)")
        tracker.apply_action("load (banana, glass, blending_station)")
        with self.assertRaises(ValueError):
            tracker.apply_action("blend (glass, blending_station)", enforce_preconditions=True)
        tracker.apply_action("turn_on (blender, blending_station)")
        tracker.apply_action("blend (glass, blending_station)")
        self.assertEqual(tracker.get_feature("glass_contains_mixture"), 1)
        self.assertEqual(tracker.get_feature("banana_in_mixture"), 1)

    def test_cook_contents_requires_a_cookable_item(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (pot, from=storage, to=cooking_station)")
        tracker.apply_action("turn_on (stove, cooking_station)")
        with self.assertRaises(ValueError):
            tracker.apply_action("cook_contents (pot, cooking_station)", enforce_preconditions=True)
        self.assertFalse(validate_ordering([
            "transfer (pot, from=storage, to=cooking_station)",
            "turn_on (stove, cooking_station)",
            "cook_contents (pot, cooking_station)",
        ]))

    def test_season_container_marks_all_contents(self):
        tracker = StateTracker()
        tracker.apply_action("transfer (bowl, from=storage, to=prep_station)")
        tracker.apply_action("transfer (tomato, from=storage, to=prep_station)")
        tracker.apply_action("transfer (onion, from=storage, to=prep_station)")
        tracker.apply_action("load (tomato, bowl, prep_station)")
        tracker.apply_action("load (onion, bowl, prep_station)")
        tracker.apply_action("season_container (bowl, salt, prep_station)")
        self.assertEqual(tracker.get_feature("tomato_seasoned"), 1)
        self.assertEqual(tracker.get_feature("onion_seasoned"), 1)
        self.assertEqual(tracker.get_feature("tomato_seasoned_with_salt"), 1)
        self.assertEqual(tracker.get_feature("onion_seasoned_with_salt"), 1)

    def test_serve_sets_dish_served_flag_for_nonempty_plate(self):
        tracker = StateTracker()
        tracker.apply_action("load (tomato, plate, storage)", enforce_preconditions=True)
        tracker.apply_action("move_container (plate, from=storage, to=serving_station)", enforce_preconditions=True)
        tracker.apply_action("serve (plate, serving_station)", enforce_preconditions=True)
        self.assertEqual(tracker.get_feature("dish_served"), 1)

    def test_serve_accepts_nonempty_smoothie_glass(self):
        tracker = StateTracker()
        tracker.apply_action("load (milk, glass, storage)", enforce_preconditions=True)
        tracker.apply_action("move_container (glass, from=storage, to=serving_station)", enforce_preconditions=True)
        tracker.apply_action("serve (glass, serving_station)", enforce_preconditions=True)
        self.assertEqual(tracker.get_feature("dish_served"), 1)

    def test_serve_rejects_empty_or_non_serving_vessel(self):
        tracker = StateTracker()
        tracker.apply_action("move_container (plate, from=storage, to=serving_station)", enforce_preconditions=True)
        with self.assertRaises(ValueError):
            tracker.apply_action("serve (plate, serving_station)", enforce_preconditions=True)

        tracker = StateTracker()
        tracker.apply_action("load (tomato, plate, storage)", enforce_preconditions=True)
        tracker.apply_action("move_container (plate, from=storage, to=plating_station)", enforce_preconditions=True)
        with self.assertRaises(ValueError):
            tracker.apply_action("serve (plate, plating_station)", enforce_preconditions=True)

    def test_cooking_requires_pot_or_pan(self):
        tracker = StateTracker()
        tracker.apply_action("load (meat, bowl, storage)", enforce_preconditions=True)
        tracker.apply_action("move_container (bowl, from=storage, to=cooking_station)", enforce_preconditions=True)
        tracker.apply_action("turn_on (stove, cooking_station)", enforce_preconditions=True)
        with self.assertRaises(ValueError):
            tracker.apply_action("cook (meat, bowl, cooking_station)", enforce_preconditions=True)

    def test_season_requires_grounded_ingredient_location(self):
        tracker = StateTracker()
        with self.assertRaises(ValueError):
            tracker.apply_action("season (egg, salt)", enforce_preconditions=True)


if __name__ == "__main__":
    unittest.main()
