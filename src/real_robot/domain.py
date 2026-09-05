"""Recipe-independent task state exposed to the Adaptive-HRC learner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import LabConfig


StateVector = Tuple[int, ...]


@dataclass(frozen=True)
class PhysicalObservation:
    """A completion-checked real-world action transition."""

    state: StateVector
    action: str
    next_state: StateVector


class PhysicalTaskDomain:
    """Encode completed physical options without exposing recipe identity.

    The state is a fixed bit vector over the configured action catalog.  A
    recipe label is deliberately absent: the learner must infer the task and
    preference from observed actions, just as it does in the symbolic domain.
    """

    name = "physical_marker_kitchen"
    action_representation = "completion_checked_marker_transfers"

    def __init__(self, config: LabConfig):
        self.config = config
        self.actions = tuple(config.actions)
        self.action_index = {action: index for index, action in enumerate(self.actions)}
        self.strategy_roles = tuple(dict.fromkeys(
            config.actions[action].role for action in self.actions
        ))
        if not self.strategy_roles:
            raise ValueError("physical domain requires at least one action role")

    def initial_state(self) -> StateVector:
        return (0,) * len(self.actions)

    def predicate_names(self) -> Tuple[str, ...]:
        """Name each completion flag, for predictors that read states as text."""
        return tuple(f"{action}_done" for action in self.actions)

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        del actor_id
        values = tuple(int(value) for value in state)
        if len(values) != len(self.actions):
            raise ValueError(
                f"physical state has width {len(values)}, expected {len(self.actions)}"
            )
        if any(value not in (0, 1) for value in values):
            raise ValueError("physical task completion flags must be binary")
        return values

    def canonical_action(self, action: str) -> str:
        token = str(action).strip().upper()
        if token not in self.action_index:
            raise ValueError(f"unknown physical task action {action!r}")
        return token

    def successor(self, state: StateVector, action: str) -> Optional[StateVector]:
        values = list(self.state_key(state))
        try:
            token = self.canonical_action(action)
        except ValueError:
            return None
        index = self.action_index[token]
        if values[index]:
            return None
        spec = self.config.actions[token]
        if any(not values[self.action_index[required]] for required in spec.requires):
            return None
        values[index] = 1
        return tuple(values)

    def replay_transition(self, state: StateVector, action: str) -> StateVector:
        after = self.successor(state, action)
        if after is None:
            raise ValueError(f"illegal physical replay transition: {action!r}")
        return after

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        state = self.initial_state()
        for action in actions:
            state = self.replay_transition(state, str(action))
        return state

    def legal_actions(
        self, state: StateVector, actions: Sequence[str],
    ) -> Tuple[str, ...]:
        values = self.state_key(state)
        legal = []
        for action in actions:
            try:
                token = self.canonical_action(action)
            except ValueError:
                continue
            if self.successor(values, token) is not None and token not in legal:
                legal.append(token)
        return tuple(legal)

    def action_role(self, action: str) -> str:
        return self.config.actions[self.canonical_action(action)].role

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        return tuple(sorted({self.canonical_action(action) for action in actions}))

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
        states = [self.state_key(state_vectors[index]) for index in range(len(state_vectors))]
        bits = np.asarray(states, dtype=np.float32)
        if not len(states):
            bits = np.zeros((0, len(self.actions)), dtype=np.float32)
        semantic = np.asarray([self._semantic_row(state) for state in states], dtype=np.float32)
        if not len(states):
            semantic = np.zeros((0, len(self.strategy_roles) + 1), dtype=np.float32)
        if feature_mode == "raw_state":
            raw = bits
        elif feature_mode == "semantic":
            raw = semantic
        else:
            raw = np.concatenate((bits, semantic), axis=1)
        return self._normalize(
            raw,
            known_mean=known_mean,
            known_scale=known_scale,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
            normalize=normalize,
        )

    def _semantic_row(self, state: StateVector) -> Tuple[float, ...]:
        values = self.state_key(state)
        progress = []
        for role in self.strategy_roles:
            indices = [
                self.action_index[token] for token in self.actions
                if self.config.actions[token].role == role
            ]
            progress.append(sum(values[index] for index in indices) / len(indices))
        progress.append(sum(values) / max(1, len(values)))
        return tuple(float(value) for value in progress)

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
            safe_scale = np.where(abs(scale) > 1e-8, scale, 1.0)
            return (raw - mean) / safe_scale, mean, safe_scale
        mean = raw.mean(axis=0).astype(np.float32)
        scale = raw.std(axis=0).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        return (raw - mean) / scale, mean, scale
