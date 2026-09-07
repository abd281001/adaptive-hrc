"""Simulator-independent task-progress domain used by Adaptive-HRC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import FUNCTIONAL_ROLES, MAX_TASK_ACTIONS, RECIPES, get_recipe


StateVector = Tuple[int, ...]
REWARD_FEATURE_VERSION = "cooking_task_reward_v3"
SEMANTIC_FEATURE_VERSION = "cooking_task_semantic_v3"
STRATEGY_ROLE_VERSION = "cooking_functional_roles_v3"
# Keep the learner setting identical to Adaptive-HRC.  The environment adapter
# changes the representation, not the model hyperparameters.
SEMANTIC_FALLBACK_MAX_RMS_DISTANCE = 0.20

RECIPE_IDS = tuple(RECIPES)
RECIPE_CODE = {recipe_id: index + 1 for index, recipe_id in enumerate(RECIPE_IDS)}
CODE_RECIPE = {code: recipe_id for recipe_id, code in RECIPE_CODE.items()}


@dataclass(frozen=True)
class TaskState:
    recipe_id: str
    completed: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        recipe = get_recipe(self.recipe_id)
        unknown = set(self.completed) - set(recipe.action_tokens)
        if unknown:
            raise ValueError(f"unknown completed options: {sorted(unknown)}")

    def encode(self) -> StateVector:
        recipe = get_recipe(self.recipe_id)
        bits = [int(action in self.completed) for action in recipe.action_tokens]
        bits.extend([0] * (MAX_TASK_ACTIONS - len(bits)))
        return (RECIPE_CODE[self.recipe_id], *bits)

    @classmethod
    def decode(cls, state: Sequence[int]) -> "TaskState":
        values = tuple(int(value) for value in state)
        if len(values) != 1 + MAX_TASK_ACTIONS:
            raise ValueError(
                f"cooking task state has width {len(values)}, expected "
                f"{1 + MAX_TASK_ACTIONS}"
            )
        recipe_id = CODE_RECIPE.get(values[0])
        if recipe_id is None:
            raise ValueError(f"unknown recipe code {values[0]!r}")
        if any(bit not in (0, 1) for bit in values[1:]):
            raise ValueError("task completion flags must be binary")
        recipe = get_recipe(recipe_id)
        if any(values[1 + len(recipe.actions):]):
            raise ValueError("unused task completion flags must be zero")
        return cls(
            recipe_id,
            frozenset(
                action.token for action, complete
                in zip(recipe.actions, values[1:]) if complete
            ),
        )


class CookingDomainAdapter:
    """Expose recipe-local progress and cross-recipe semantic roles."""

    name = "overcooked_burrito_task_options"
    action_representation = "completion_checked_task_options"
    reward_feature_version = REWARD_FEATURE_VERSION
    semantic_feature_version = SEMANTIC_FEATURE_VERSION
    strategy_roles = FUNCTIONAL_ROLES

    def __init__(self, recipe_id: str = "burrito_steak_burrito"):
        self._recipe_id = get_recipe(recipe_id).recipe_id

    @property
    def recipe_id(self) -> str:
        return self._recipe_id

    def begin_task(self, recipe_id: str) -> StateVector:
        self._recipe_id = get_recipe(recipe_id).recipe_id
        return self.initial_state()

    def initial_state(self) -> StateVector:
        return TaskState(self._recipe_id).encode()

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        del actor_id
        if isinstance(state, TaskState):
            return state.encode()
        if isinstance(state, (tuple, list, np.ndarray)):
            return TaskState.decode(state).encode()
        raise TypeError(
            "learning state must be TaskState or an encoded tuple; raw "
            "simulator geometry belongs to the physical executor"
        )

    def recipe_of_state(self, state: Any) -> str:
        """Recover the catalog recipe a state belongs to.

        The encoded state carries its recipe in slot zero, so a stored replay
        transition identifies its own task.  ``AdaptiveAgent`` labels recipes
        with the identifiers it allocates during observation (``R0``, ``R1``,
        ...), which this catalog does not know; scoping the domain by that
        label raised on every audit.  Reading the recipe back out of the state
        keeps the learner's naming private to the learner.
        """
        return TaskState.decode(self.state_key(state)).recipe_id

    def state_from_completed(
        self, recipe_id: str, completed: Sequence[str],
    ) -> StateVector:
        return TaskState(recipe_id, frozenset(map(str, completed))).encode()

    def replay_transition(self, state: StateVector, action: str) -> StateVector:
        successor = self.successor(state, action)
        if successor is None:
            raise ValueError(f"cannot replay illegal cooking option {action!r}")
        return successor

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        state = self.initial_state()
        for action in actions:
            state = self.replay_transition(state, str(action))
        return state

    def successor(
        self, state: StateVector, action: str,
    ) -> Optional[StateVector]:
        decoded = TaskState.decode(self.state_key(state))
        label = str(action).strip().upper()
        if label not in self.legal_actions(state, (label,)):
            return None
        return TaskState(decoded.recipe_id, decoded.completed | {label}).encode()

    def legal_actions(
        self, state: StateVector, actions: Sequence[str],
    ) -> Tuple[str, ...]:
        decoded = TaskState.decode(self.state_key(state))
        recipe = get_recipe(decoded.recipe_id)
        by_token = recipe.action_by_token
        legal = []
        for action in actions:
            label = str(action).strip().upper()
            spec = by_token.get(label)
            if (
                spec is not None
                and label not in decoded.completed
                and spec.requires <= decoded.completed
            ):
                legal.append(label)
        return tuple(dict.fromkeys(legal))

    def canonical_action(self, action: str) -> str:
        label = str(action).strip().upper()
        known = {
            item.token for recipe in RECIPES.values() for item in recipe.actions
        }
        if label not in known:
            raise ValueError(f"unknown cooking task option {action!r}")
        return label

    def reward_features(
        self,
        state_vectors: Mapping[int, StateVector],
        known_mean: Optional[np.ndarray] = None,
        known_scale: Optional[np.ndarray] = None,
        normalizer: Any = None,
        update_normalizer: bool = False,
        *,
        feature_mode: str = "engineered",
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_features(
            state_vectors,
            known_mean=known_mean,
            known_scale=known_scale,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
            feature_mode=feature_mode,
            normalize=normalize,
        )

    def semantic_features(
        self,
        state_vectors: Mapping[int, StateVector],
        *,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_features(
            state_vectors, feature_mode="semantic", normalize=normalize,
        )

    def build_features(
        self,
        state_vectors: Mapping[int, StateVector],
        known_mean: Optional[np.ndarray] = None,
        known_scale: Optional[np.ndarray] = None,
        normalizer: Any = None,
        update_normalizer: bool = False,
        *,
        feature_mode: str = "engineered",
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if feature_mode not in {"semantic", "engineered", "raw_state"}:
            raise ValueError("unsupported feature mode")
        states = [
            TaskState.decode(self.state_key(state_vectors[index]))
            for index in range(len(state_vectors))
        ]
        semantic = np.asarray(
            [self._semantic_row(state) for state in states], dtype=np.float32
        )
        if feature_mode == "raw_state":
            # One-hot the recipe, never its integer code: the code is an
            # arbitrary enumeration index, and feeding it as a scalar asserts
            # that recipe 7 lies between recipe 6 and recipe 8.
            raw = np.asarray([
                [float(state.recipe_id == recipe_id) for recipe_id in RECIPE_IDS]
                + list(state.encode()[1:])
                for state in states
            ], dtype=np.float32)
        elif feature_mode == "semantic":
            raw = semantic
        else:
            identity = np.asarray([
                [float(state.recipe_id == recipe_id) for recipe_id in RECIPE_IDS]
                for state in states
            ], dtype=np.float32)
            raw = np.concatenate((semantic, identity), axis=1)
        return self._normalize(
            raw,
            known_mean=known_mean,
            known_scale=known_scale,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
            normalize=normalize,
        )

    @staticmethod
    def _semantic_row(state: TaskState) -> Tuple[float, ...]:
        recipe = get_recipe(state.recipe_id)
        role_counts = {role: [0, 0] for role in FUNCTIONAL_ROLES}
        for action in recipe.actions:
            role_counts[action.role][1] += 1
            role_counts[action.role][0] += int(action.token in state.completed)
        progress = tuple(
            completed / total if total else 0.0
            for completed, total in (role_counts[role] for role in FUNCTIONAL_ROLES)
        )
        return progress + (len(state.completed) / len(recipe.actions),)

    @staticmethod
    def _normalize(
        raw: np.ndarray,
        *,
        known_mean: Optional[np.ndarray],
        known_scale: Optional[np.ndarray],
        normalizer: Any,
        update_normalizer: bool,
        normalize: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if raw.ndim != 2:
            raw = raw.reshape((len(raw), -1))
        width = raw.shape[1]
        if not normalize:
            return raw, np.zeros(width, np.float32), np.ones(width, np.float32)
        if normalizer is not None:
            if update_normalizer:
                normalizer.update(raw)
            if normalizer.mean is not None:
                mean = np.asarray(normalizer.mean, np.float32)
                scale = np.asarray(normalizer.scale, np.float32)
                return normalizer.transform(raw), mean, scale
        elif known_mean is not None and known_scale is not None:
            mean = np.asarray(known_mean, np.float32)
            scale = np.asarray(known_scale, np.float32)
            if mean.shape != (width,) or scale.shape != (width,):
                raise ValueError("known feature statistics have the wrong width")
            return (raw - mean) / np.where(abs(scale) > 1e-8, scale, 1), mean, scale
        mean = raw.mean(axis=0).astype(np.float32)
        scale = raw.std(axis=0).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        return (raw - mean) / scale, mean, scale

    def functional_predicates(self, state: StateVector) -> Mapping[str, float]:
        decoded = TaskState.decode(self.state_key(state))
        row = self._semantic_row(decoded)
        return {
            role: row[index] for index, role in enumerate(FUNCTIONAL_ROLES)
        }

    def action_role(self, action: str) -> str:
        label = self.canonical_action(action)
        for recipe in RECIPES.values():
            spec = recipe.action_by_token.get(label)
            if spec is not None:
                return spec.role
        raise AssertionError(label)

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        """Preference-neutral task signature, matching the symbolic adapter.

        This must identify the recipe, not its shape.  A multiset of functional
        roles does not: it maps all three two-ingredient Overcooked recipes to
        one signature, all four three-ingredient ones to another, and each pair
        of Burrito recipes to a third -- eleven recipes onto four signatures.
        ``SymbolicDomainAdapter.goal_signature`` returns the action set, and the
        open-set separation guard in the protocol assumes signatures separate
        recipes, so return the action set here too.
        """
        return tuple(sorted({self.canonical_action(action) for action in actions}))

    def role_signature(self, actions: Sequence[str]) -> Hashable:
        """Role-level shape, deliberately shared across analogous recipes."""
        return tuple(sorted(self.action_role(action) for action in actions))


BurritoDomainAdapter = CookingDomainAdapter


__all__ = [
    "BurritoDomainAdapter", "CookingDomainAdapter", "FUNCTIONAL_ROLES",
    "REWARD_FEATURE_VERSION", "SEMANTIC_FALLBACK_MAX_RMS_DISTANCE",
    "SEMANTIC_FEATURE_VERSION", "STRATEGY_ROLE_VERSION", "StateVector",
    "TaskState",
]
