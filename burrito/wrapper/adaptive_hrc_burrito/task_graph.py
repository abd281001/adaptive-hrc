"""Physical Burrito task graph and partial-order human preferences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Sequence, Tuple

import numpy as np

from .domain import BurritoDomainAdapter, StateVector
from .macros import (
    COLLECT_AND_STAGE_COOKED_RICE,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    assemble_action,
    fetch_and_stage_action,
    macro_actions,
    prepare_and_stage_action,
    protein_name,
    serve_action,
    start_cooking_action,
)


PREFERENCE_NAMES: Tuple[str, ...] = (
    "protein_first",
    "rice_first",
    "plate_early",
    "plate_jit",
)

_PREFERENCE_ALIASES = {
    "plate_first": "plate_early",
    "plate_just_in_time": "plate_jit",
}


def preference_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _PREFERENCE_ALIASES.get(normalized, normalized)
    if normalized not in PREFERENCE_NAMES:
        raise ValueError(
            f"unknown Burrito preference {value!r}; expected one of "
            f"{PREFERENCE_NAMES}"
        )
    return normalized


@dataclass(frozen=True)
class BurritoTaskGraph:
    """A recipe DAG whose frontier is filtered by the physical state."""

    protein: str
    actions: Tuple[str, ...]
    dependencies: Mapping[str, FrozenSet[str]]

    @classmethod
    def create(cls, protein: str) -> "BurritoTaskGraph":
        normalized = protein_name(protein)
        fetch = fetch_and_stage_action(normalized)
        prepare = prepare_and_stage_action(normalized)
        start_protein = start_cooking_action(normalized)
        assemble = assemble_action(normalized)
        serve = serve_action(normalized)
        actions = macro_actions(normalized)
        return cls(
            protein=normalized,
            actions=actions,
            dependencies={
                fetch: frozenset(),
                prepare: frozenset((fetch,)),
                # Starting rice first is a preference-neutral safety
                # constraint: on the larger layout, protein can otherwise
                # burn while rice is still being fetched and plated. The
                # protein-first preference concerns *preparing* protein before
                # starting rice, so this does not erase that ordering choice.
                start_protein: frozenset((prepare, START_BOILING_RICE)),
                START_BOILING_RICE: frozenset(),
                STAGE_CLEAN_PLATE: frozenset(),
                COLLECT_AND_STAGE_COOKED_RICE: frozenset((
                    START_BOILING_RICE,
                    STAGE_CLEAN_PLATE,
                )),
                assemble: frozenset((
                    start_protein,
                    COLLECT_AND_STAGE_COOKED_RICE,
                )),
                serve: frozenset((assemble,)),
            },
        )

    def available_actions(
        self,
        state: StateVector,
        completed: Sequence[str],
        domain: BurritoDomainAdapter,
    ) -> Tuple[str, ...]:
        """Return the physically legal dependency frontier in stable order."""
        done = frozenset(map(str, completed))
        candidates = tuple(
            action for action in self.actions
            if action not in done and self.dependencies[action] <= done
        )
        return domain.legal_actions(state, candidates)

    def is_complete(self, completed: Sequence[str]) -> bool:
        return serve_action(self.protein) in frozenset(map(str, completed))


@dataclass(frozen=True)
class BurritoPreferencePolicy:
    """Preference-specific constraints over a task graph frontier."""

    name: str

    @classmethod
    def create(cls, name: str) -> "BurritoPreferencePolicy":
        return cls(preference_name(name))

    def acceptable_actions(
        self,
        graph: BurritoTaskGraph,
        state: StateVector,
        completed: Sequence[str],
        domain: BurritoDomainAdapter,
    ) -> Tuple[str, ...]:
        """Return all next actions consistent with state, DAG, and preference."""
        available = graph.available_actions(state, completed, domain)
        if not available:
            return ()
        done = frozenset(map(str, completed))
        prepare = prepare_and_stage_action(graph.protein)

        if self.name == "protein_first" and prepare not in done:
            return tuple(
                action for action in available
                if action != START_BOILING_RICE
            )

        if self.name == "rice_first" and START_BOILING_RICE not in done:
            return tuple(action for action in available if action != prepare)

        if self.name == "plate_early" and STAGE_CLEAN_PLATE not in done:
            return (
                (STAGE_CLEAN_PLATE,)
                if STAGE_CLEAN_PLATE in available else ()
            )

        if self.name == "plate_jit" and STAGE_CLEAN_PLATE not in done:
            cooked = (
                "chopped_steak"
                if graph.protein == "steak" else "fried_mushroom"
            )
            predicates = domain.functional_predicates(state)
            both_started = (
                predicates["protein_cooking"]
                or predicates["protein_ready"]
            ) and (
                predicates["starch_cooking"]
                or predicates["starch_ready"]
            )
            if both_started:
                return (
                    (STAGE_CLEAN_PLATE,)
                    if STAGE_CLEAN_PLATE in available else ()
                )
            return tuple(
                action for action in available
                if action != STAGE_CLEAN_PLATE
            )

        return available


def choose_acceptable_action(
    acceptable: Sequence[str], rng: np.random.Generator,
) -> str:
    """Seeded stochastic choice for semantically equivalent actions.

    Continuous standard-normal draws make exact ties probability-zero and
    avoid giving an action a privileged alphabetical or declaration position.
    """
    actions = tuple(dict.fromkeys(map(str, acceptable)))
    if not actions:
        raise ValueError("cannot choose from an empty acceptable action set")
    scores = rng.standard_normal(len(actions))
    return actions[int(np.argmax(scores))]


__all__ = [
    "BurritoPreferencePolicy",
    "BurritoTaskGraph",
    "PREFERENCE_NAMES",
    "choose_acceptable_action",
    "preference_name",
]
