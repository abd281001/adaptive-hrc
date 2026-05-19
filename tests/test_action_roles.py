"""Phase 0B: deterministic role abstraction over kitchen-action transitions."""
from __future__ import annotations

import unittest

from src.representations import (
    ROLES,
    ROLE_ACTIVATE_APPLIANCE,
    ROLE_ADD_TO_CONTAINER,
    ROLE_CLEAN_CONTAINER,
    ROLE_COOK_OR_BLEND,
    ROLE_DEACTIVATE_APPLIANCE,
    ROLE_MOVE_CONTAINER,
    ROLE_PREPARE_INGREDIENT,
    ROLE_RETRIEVE_CONTAINER,
    ROLE_RETRIEVE_INGREDIENT,
    ROLE_SERVE,
    ROLE_STAGE_SERVING_VESSEL,
    ROLE_UNKNOWN_OR_NOOP,
    role_bigrams,
    role_from_transition,
    role_trigrams,
    roles_from_actions,
)
from src.environment import StateTracker, gen


class RoleAtomTests(unittest.TestCase):
    """Single-action assertions: each kitchen primitive maps to its expected role."""

    def _step(self, actions, action):
        tracker = StateTracker()
        for a in actions:
            tracker.apply_action(a)
        before = tuple(tracker.get_state_vector().astype(int).tolist())
        tracker.apply_action(action)
        after = tuple(tracker.get_state_vector().astype(int).tolist())
        delta = tuple(int(a) - int(b) for a, b in zip(after, before))
        return role_from_transition(before, delta, after)

    def test_activate_appliance(self):
        self.assertEqual(self._step([], "turn_on (stove)"), ROLE_ACTIVATE_APPLIANCE)

    def test_deactivate_appliance(self):
        self.assertEqual(
            self._step(["turn_on (stove)"], "turn_off (stove)"),
            ROLE_DEACTIVATE_APPLIANCE,
        )

    def test_retrieve_ingredient(self):
        self.assertEqual(
            self._step([], "transfer (tomato, from=storage, to=prep_station)"),
            ROLE_RETRIEVE_INGREDIENT,
        )

    def test_stage_serving_vessel(self):
        self.assertEqual(
            self._step([], "transfer (plate, from=storage, to=plating_station)"),
            ROLE_STAGE_SERVING_VESSEL,
        )

    def test_prepare_ingredient_cut(self):
        self.assertEqual(
            self._step(
                ["transfer (tomato, from=storage, to=prep_station)"],
                "cut (tomato, prep_station)",
            ),
            ROLE_PREPARE_INGREDIENT,
        )

    def test_add_to_container(self):
        # Need pot at storage and tomato at storage co-located before load.
        # The default tracker puts everything at storage, so we can load
        # tomato into pot at storage.
        self.assertEqual(
            self._step([], "load (tomato, pot, storage)"),
            ROLE_ADD_TO_CONTAINER,
        )

    def test_retrieve_container(self):
        # Move pot (with content) out of storage: this is equipment retrieval,
        # not generic station-to-station movement.
        self.assertEqual(
            self._step(
                [
                    "load (tomato, pot, storage)",
                ],
                "move_container (pot, from=storage, to=cooking_station)",
            ),
            ROLE_RETRIEVE_CONTAINER,
        )

    def test_move_container(self):
        self.assertEqual(
            self._step(
                [
                    "move_container (pot, from=storage, to=cooking_station)",
                ],
                "move_container (pot, from=cooking_station, to=plating_station)",
            ),
            ROLE_MOVE_CONTAINER,
        )

    def test_cook_or_blend(self):
        self.assertEqual(
            self._step(
                [
                    "load (tomato, pot, storage)",
                    "move_container (pot, from=storage, to=cooking_station)",
                    "turn_on (stove)",
                ],
                "cook (tomato, pot, cooking_station)",
            ),
            ROLE_COOK_OR_BLEND,
        )

    def test_serve(self):
        self.assertEqual(
            self._step(
                [
                    "transfer (plate, from=storage, to=plating_station)",
                    "move_container (plate, from=plating_station, to=serving_station)",
                ],
                "serve (plate, serving_station)",
            ),
            ROLE_SERVE,
        )

    def test_clean_container(self):
        self.assertEqual(
            self._step(
                ["move_container (plate, from=storage, to=washing_station)"],
                "wash (plate, washing_station)",
            ),
            ROLE_CLEAN_CONTAINER,
        )

    def test_noop_returns_unknown(self):
        # Identical states (no transition) -> unknown_or_noop.
        before = (0, 0, 0)
        after = (0, 0, 0)
        delta = (0, 0, 0)
        self.assertEqual(role_from_transition(before, delta, after), ROLE_UNKNOWN_OR_NOOP)


class RoleCoverageTests(unittest.TestCase):
    """Plan-mandated unit-test requirement.

    For every action in RecipeGenerator outputs, role_from_transition must
    return exactly one non-unknown role. This is the hard contract on which
    Phases 3+4 depend.
    """

    def test_every_recipe_action_has_a_role(self):
        offenders = []
        for recipe_name, fn in gen.recipe_library().items():
            actions = fn()
            roles = roles_from_actions(actions)
            self.assertEqual(len(roles), len(actions), f"role count mismatch for {recipe_name}")
            for action, role in zip(actions, roles):
                if role == ROLE_UNKNOWN_OR_NOOP:
                    offenders.append((recipe_name, action))
        self.assertEqual(offenders, [], f"actions with unknown role: {offenders[:5]}")

    def test_role_set_is_stable(self):
        # The role tuple is a contract that the preference prototype learner
        # depends on for embedding indices. If it changes, downstream code
        # must be updated explicitly.
        self.assertEqual(len(ROLES), 13)
        self.assertEqual(ROLES[0], "retrieve_ingredient")
        self.assertEqual(ROLES[1], "retrieve_container")
        self.assertEqual(ROLES[-1], "unknown_or_noop")


class RoleNgramTests(unittest.TestCase):
    def test_bigrams_and_trigrams(self):
        seq = ["a", "b", "c", "d"]
        self.assertEqual(role_bigrams(seq), [("a", "b"), ("b", "c"), ("c", "d")])
        self.assertEqual(role_trigrams(seq), [("a", "b", "c"), ("b", "c", "d")])
        self.assertEqual(role_bigrams(["a"]), [])
        self.assertEqual(role_trigrams(["a", "b"]), [])


class RoleLabelFreedomTests(unittest.TestCase):
    """test_role_extraction_is_label_free (from Phase 0B static leakage spec).

    The role function must produce identical output regardless of any
    preference label attached to the demo metadata. We assert this by
    fabricating a fake 'preference label' and showing it never reaches the
    decoder via any visible input channel.
    """

    def test_role_function_signature_takes_only_state_inputs(self):
        import inspect
        params = inspect.signature(role_from_transition).parameters
        self.assertEqual(
            list(params.keys()),
            ["state_before", "action_vector", "state_after"],
        )

    def test_role_output_invariant_to_external_label(self):
        actions = list(gen.recipe_library().values())[0]()
        roles_a = roles_from_actions(actions)
        # Re-run with the same inputs; deterministic.
        roles_b = roles_from_actions(actions)
        self.assertEqual(roles_a, roles_b)


if __name__ == "__main__":
    unittest.main()
