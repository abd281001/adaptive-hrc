"""Recipe DAGs and deterministic event-preference policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .catalog import (
    PREFERENCES,
    RecipeSpec,
    applicable_preferences,
    get_preference,
    get_recipe,
    preference_order,
)


PREFERENCE_NAMES = tuple(PREFERENCES)


@dataclass(frozen=True)
class CookingTaskGraph:
    recipe: RecipeSpec

    @classmethod
    def create(cls, recipe_id: str) -> "CookingTaskGraph":
        return cls(get_recipe(recipe_id))

    @property
    def actions(self) -> Tuple[str, ...]:
        return self.recipe.action_tokens

    def frontier(self, completed: Sequence[str]) -> Tuple[str, ...]:
        done = frozenset(map(str, completed))
        return tuple(
            action.token for action in self.recipe.actions
            if action.token not in done and action.requires <= done
        )

    def available_actions(
        self,
        completed: Sequence[str],
        physically_legal: Sequence[str] | None = None,
    ) -> Tuple[str, ...]:
        frontier = self.frontier(completed)
        if physically_legal is None:
            return frontier
        allowed = frozenset(map(str, physically_legal))
        return tuple(action for action in frontier if action in allowed)

    def is_complete(self, completed: Sequence[str]) -> bool:
        return len(frozenset(completed)) == len(self.actions)


@dataclass(frozen=True)
class CookingPreferencePolicy:
    name: str

    @classmethod
    def create(cls, name: str) -> "CookingPreferencePolicy":
        return cls(get_preference(name).name)

    def choose_action(
        self, legal_actions: Sequence[str], graph: CookingTaskGraph | None = None,
    ) -> str:
        candidates = tuple(dict.fromkeys(map(str, legal_actions)))
        if not candidates:
            raise ValueError("cannot choose from an empty legal frontier")
        if graph is None:
            raise ValueError("event-preference action selection requires a task graph")
        preference = get_preference(self.name)
        by_token = graph.recipe.action_by_token
        order = {action: index for index, action in enumerate(graph.actions)}
        if preference.early:
            key = lambda token: (
                preference.target_event not in by_token[token].events,
                order[token],
            )
        else:
            key = lambda token: (
                preference.target_event in by_token[token].events,
                order[token],
            )
        return min(candidates, key=key)

    def ordering(self, graph: CookingTaskGraph) -> Tuple[str, ...]:
        return preference_order(graph.recipe, get_preference(self.name))


def is_preference_discriminating(
    legal_actions: Sequence[str], graph: CookingTaskGraph,
) -> bool:
    legal = tuple(legal_actions)
    if len(legal) < 2:
        return False
    choices = {
        CookingPreferencePolicy.create(name).choose_action(legal, graph)
        for name in applicable_preferences(graph.recipe.recipe_id)
    }
    return len(choices) > 1


# Existing import names remain aliases, without retaining the old implementation.
BurritoTaskGraph = CookingTaskGraph
BurritoPreferencePolicy = CookingPreferencePolicy


__all__ = [
    "BurritoPreferencePolicy", "BurritoTaskGraph", "CookingPreferencePolicy",
    "CookingTaskGraph", "PREFERENCE_NAMES", "applicable_preferences",
    "is_preference_discriminating",
]
