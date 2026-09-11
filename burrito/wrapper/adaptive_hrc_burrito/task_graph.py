"""Recipe DAGs and deterministic event-preference policies."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    """Do the recipe's declared preferences disagree on this frontier?

    This is a property of the recipe, not of the episode: it ignores what the
    prefix already revealed.  It is the right denominator for a task-intrinsic
    count of choice points, and the wrong one for scoring a predictor -- see
    ``is_prefix_conditioned_discriminating``.
    """
    return _discriminating(graph.recipe.recipe_id, tuple(legal_actions))


@lru_cache(maxsize=None)
def _discriminating(recipe_id: str, legal: Tuple[str, ...]) -> bool:
    if len(legal) < 2:
        return False
    graph = CookingTaskGraph.create(recipe_id)
    choices = {
        CookingPreferencePolicy.create(name).choose_action(legal, graph)
        for name in applicable_preferences(recipe_id)
    }
    return len(choices) > 1


@lru_cache(maxsize=None)
def preferences_consistent_with_prefix(
    recipe_id: str, completed: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Applicable preferences that would themselves have produced this prefix.

    The executed action is always the active preference's choice -- a vetoed
    robot proposal is never carried out -- so a realized prefix is exactly some
    preference's ordering prefix, and every preference whose ordering starts
    the same way is still live.  Returns the full applicable set if nothing
    matches, which keeps an unexpected prefix scored rather than silently
    dropped.
    """
    recipe = get_recipe(recipe_id)
    depth = len(completed)
    names = applicable_preferences(recipe_id)
    consistent = tuple(
        name for name in names
        if preference_order(recipe, get_preference(name))[:depth] == completed
    )
    return consistent or names


def is_prefix_conditioned_discriminating(
    legal_actions: Sequence[str],
    graph: CookingTaskGraph,
    completed: Sequence[str],
) -> bool:
    """Is this frontier still ambiguous given what the prefix already revealed?

    ``is_preference_discriminating`` asks whether the recipe's preferences
    could ever disagree here.  That over-counts badly under ``human_first``:
    the human takes step 0, the widest frontier in the episode, so by the
    robot's first turn the surviving preference set is usually a singleton and
    the "choice" has one consistent answer.  Scoring a predictor on those
    decisions measures whether it can follow an already-determined ordering,
    not whether it tracks a preference.
    """
    return _prefix_conditioned(
        graph.recipe.recipe_id, tuple(completed), tuple(legal_actions),
    )


@lru_cache(maxsize=None)
def _prefix_conditioned(
    recipe_id: str, completed: Tuple[str, ...], legal: Tuple[str, ...],
) -> bool:
    if len(legal) < 2:
        return False
    graph = CookingTaskGraph.create(recipe_id)
    consistent = preferences_consistent_with_prefix(recipe_id, completed)
    choices = {
        CookingPreferencePolicy.create(name).choose_action(legal, graph)
        for name in consistent
    }
    return len(choices) > 1


# Existing import names remain aliases, without retaining the old implementation.
BurritoTaskGraph = CookingTaskGraph
BurritoPreferencePolicy = CookingPreferencePolicy


__all__ = [
    "BurritoPreferencePolicy", "BurritoTaskGraph", "CookingPreferencePolicy",
    "CookingTaskGraph", "PREFERENCE_NAMES", "applicable_preferences",
    "is_preference_discriminating", "is_prefix_conditioned_discriminating",
    "preferences_consistent_with_prefix",
]
