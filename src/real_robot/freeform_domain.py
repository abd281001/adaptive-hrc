"""Open-ended marker-location domain for the physical three-box HRC demo."""
from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import LabConfig


StateVector = Tuple[int, ...]

LOCATION_TOKENS = (
    "S0_LEFT",
    "S0_CENTER",
    "S0_RIGHT",
    "S2",
    "S3",
    "S4",
)

TOKEN_TO_SCENE = {
    "S0_LEFT": "S0_left",
    "S0_CENTER": "S0_center",
    "S0_RIGHT": "S0_right",
    "S2": "S2",
    "S3": "S3",
    "S4": "S4",
}
SCENE_TO_TOKEN = {value: key for key, value in TOKEN_TO_SCENE.items()}


class FreeformPhysicalDomain:
    """State = current physical location of each tagged box.

    Actions mean "move this tagged box to this calibrated location".
    The same action token may occur again later after the box has moved away.
    """

    name = "physical_marker_freeform"
    action_representation = "marker_box_to_location"

    def __init__(self, config: LabConfig):
        self.config = config

        self.object_ids = tuple(sorted(
            config.objects,
            key=lambda object_id: int(config.objects[object_id].marker_id),
        ))
        if len(self.object_ids) != 3:
            raise ValueError(
                "freeform physical mode currently requires exactly three tagged objects"
            )

        self.object_index = {
            object_id: index
            for index, object_id in enumerate(self.object_ids)
        }
        self.location_index = {
            location: index
            for index, location in enumerate(LOCATION_TOKENS)
        }

        self.marker_to_object = {
            int(spec.marker_id): object_id
            for object_id, spec in config.objects.items()
        }

        self.action_map = {}
        actions = []
        for object_id in self.object_ids:
            object_token = object_id.upper()
            for location in LOCATION_TOKENS:
                token = f"MOVE_{object_token}_TO_{location}"
                self.action_map[token] = (object_id, location)
                actions.append(token)

        self.actions = tuple(actions)

        # Temporary valid default. Every real episode replaces this from the
        # marker-observed baseline before any learner observation.
        self._initial_state = tuple(
            self.location_index[location]
            for location in ("S0_LEFT", "S0_CENTER", "S0_RIGHT")
        )

    def set_initial_scene(self, scene: Mapping[str, Mapping[str, Any]]) -> StateVector:
        found: dict[str, str] = {}

        for physical_location, row in scene.items():
            tag_id = row.get("tag_id")
            if tag_id is None:
                continue

            object_id = self.marker_to_object.get(int(tag_id))
            if object_id is None:
                continue

            location_token = SCENE_TO_TOKEN.get(str(physical_location))
            if location_token is None:
                raise ValueError(
                    f"scene contains unsupported location {physical_location!r}"
                )

            if object_id in found:
                raise ValueError(
                    f"{object_id} appears at multiple physical locations"
                )

            found[object_id] = location_token

        missing = set(self.object_ids) - set(found)
        if missing:
            raise ValueError(
                f"scene baseline is missing tagged objects: {sorted(missing)}"
            )

        values = tuple(
            self.location_index[found[object_id]]
            for object_id in self.object_ids
        )

        if len(set(values)) != len(values):
            raise ValueError("two boxes occupy the same calibrated location")

        self._initial_state = values
        return values

    def initial_state(self) -> StateVector:
        return tuple(self._initial_state)

    def predicate_names(self) -> Tuple[str, ...]:
        return tuple(
            f"{object_id}_location"
            for object_id in self.object_ids
        )

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        del actor_id
        values = tuple(int(value) for value in state)

        if len(values) != len(self.object_ids):
            raise ValueError(
                f"freeform physical state has width {len(values)}, "
                f"expected {len(self.object_ids)}"
            )

        if any(
            value < 0 or value >= len(LOCATION_TOKENS)
            for value in values
        ):
            raise ValueError("freeform physical state has invalid location index")

        if len(set(values)) != len(values):
            raise ValueError(
                "freeform physical state places multiple boxes in one location"
            )

        return values

    def canonical_action(self, action: str) -> str:
        token = str(action).strip().upper()
        if token not in self.action_map:
            raise ValueError(f"unknown freeform physical action {action!r}")
        return token

    def action_info(self, action: str) -> tuple[str, str]:
        return self.action_map[self.canonical_action(action)]

    def action_label(self, action: str) -> str:
        object_id, location = self.action_info(action)
        return f"Move {object_id.upper()} to {location.replace('_', ' ')}"

    def successor(
        self,
        state: StateVector,
        action: str,
    ) -> Optional[StateVector]:
        values = list(self.state_key(state))

        try:
            object_id, target_location = self.action_info(action)
        except ValueError:
            return None

        object_index = self.object_index[object_id]
        target_index = self.location_index[target_location]

        if values[object_index] == target_index:
            return None

        # Every calibrated location can hold only one box.
        for index, value in enumerate(values):
            if index != object_index and value == target_index:
                return None

        values[object_index] = target_index
        return tuple(values)

    def replay_transition(
        self,
        state: StateVector,
        action: str,
    ) -> StateVector:
        after = self.successor(state, action)
        if after is None:
            raise ValueError(
                f"illegal freeform physical replay transition: {action!r}"
            )
        return after

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        state = self.initial_state()
        for action in actions:
            state = self.replay_transition(state, str(action))
        return state

    def legal_actions(
        self,
        state: StateVector,
        actions: Sequence[str],
    ) -> Tuple[str, ...]:
        legal = []
        for action in actions:
            try:
                token = self.canonical_action(action)
            except ValueError:
                continue

            if self.successor(state, token) is not None and token not in legal:
                legal.append(token)

        return tuple(legal)

    def action_role(self, action: str) -> str:
        # Physical relocation destinations are represented explicitly in the
        # action token and location state. They are not cooking workflow roles.
        # "other" is the learner's supported neutral role for external-domain
        # operations that do not map onto the symbolic kitchen role taxonomy.
        self.canonical_action(action)
        return "other"

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        # Preserve repeated-action counts while removing preference/order.
        counts = Counter(self.canonical_action(action) for action in actions)
        return tuple(sorted(counts.items()))

    def describe_state(self, state: StateVector) -> Mapping[str, str]:
        values = self.state_key(state)
        return {
            object_id: LOCATION_TOKENS[values[index]]
            for index, object_id in enumerate(self.object_ids)
        }

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
    ):
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
    ):
        return self.build_features(
            state_vectors,
            feature_mode="semantic",
            normalize=normalize,
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
    ):
        if feature_mode not in {"semantic", "engineered", "raw_state"}:
            raise ValueError("unsupported feature mode")

        states = [
            self.state_key(state_vectors[index])
            for index in range(len(state_vectors))
        ]

        state_array = np.asarray(states, dtype=np.float32)
        if not len(states):
            state_array = np.zeros(
                (0, len(self.object_ids)),
                dtype=np.float32,
            )

        one_hot = np.zeros(
            (
                len(states),
                len(self.object_ids) * len(LOCATION_TOKENS),
            ),
            dtype=np.float32,
        )

        occupancy = np.zeros(
            (len(states), len(LOCATION_TOKENS)),
            dtype=np.float32,
        )

        for row_index, state in enumerate(states):
            for object_index, location_index in enumerate(state):
                one_hot[
                    row_index,
                    object_index * len(LOCATION_TOKENS) + location_index,
                ] = 1.0
                occupancy[row_index, location_index] = 1.0

        if feature_mode == "raw_state":
            raw = state_array
        elif feature_mode == "semantic":
            # Identity-masked location occupancy.
            raw = occupancy
        else:
            raw = np.concatenate((one_hot, occupancy), axis=1)

        return self._normalize(
            raw,
            known_mean=known_mean,
            known_scale=known_scale,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
            normalize=normalize,
        )

    @staticmethod
    def _normalize(
        raw: np.ndarray,
        *,
        known_mean: Optional[np.ndarray],
        known_scale: Optional[np.ndarray],
        normalizer: Any,
        update_normalizer: bool,
        normalize: bool,
    ):
        if raw.ndim != 2:
            raw = raw.reshape((len(raw), -1))

        width = raw.shape[1]

        if not normalize:
            return (
                raw,
                np.zeros(width, np.float32),
                np.ones(width, np.float32),
            )

        if normalizer is not None:
            if update_normalizer:
                normalizer.update(raw)
            if normalizer.mean is not None:
                mean = np.asarray(normalizer.mean, np.float32)
                scale = np.asarray(normalizer.scale, np.float32)
                return normalizer.transform(raw), mean, scale

        if known_mean is not None and known_scale is not None:
            mean = np.asarray(known_mean, np.float32)
            scale = np.asarray(known_scale, np.float32)
            if mean.shape != (width,) or scale.shape != (width,):
                raise ValueError(
                    "known feature statistics have the wrong width"
                )
            safe_scale = np.where(abs(scale) > 1e-8, scale, 1.0)
            return (raw - mean) / safe_scale, mean, safe_scale

        if not len(raw):
            return (
                raw,
                np.zeros(width, np.float32),
                np.ones(width, np.float32),
            )

        mean = raw.mean(axis=0).astype(np.float32)
        scale = raw.std(axis=0).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)

        return (raw - mean) / scale, mean, scale
