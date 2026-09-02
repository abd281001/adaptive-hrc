"""Burrito task-option adapter for the existing Adaptive-HRC learner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .macros import (
    COLLECT_AND_STAGE_COOKED_RICE,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    action_protein,
    assemble_action,
    fetch_and_stage_action,
    prepare_and_stage_action,
    serve_action,
    start_cooking_action,
)


StateVector = Tuple[int, ...]


ITEM_NAMES: Tuple[str, ...] = (
    "dirty_plate", "clean_plate", "meat", "mushroom", "rice",
    "tortilla", "chopped_meat", "chopped_mushroom",
    "chopped_steak", "fried_mushroom", "boiled_rice",
    "boiled_rice-plate", "tortilla-plate", "fried_mushroom-plate",
    "chopped_steak-plate", "boiled_rice-tortilla-plate",
    "chopped_steak-tortilla-plate", "fried_mushroom-tortilla-plate",
    "chopped_steak-boiled_rice-plate",
    "fried_mushroom-boiled_rice-plate", "steak_burrito",
    "mushroom_burrito", "charcoal", "charcoal-plate", "fire_ext",
)
ITEM_INDEX = {name: index for index, name in enumerate(ITEM_NAMES)}

PROCESS_NAMES: Tuple[str, ...] = (
    "dirty_plate", "chopped_meat", "chopped_mushroom",
    "chopped_steak", "fried_mushroom", "boiled_rice",
)
PROCESS_INDEX = {name: index for index, name in enumerate(PROCESS_NAMES)}
PROCESS_WIDTH = 7

GENERIC_ITEM_GROUPS: Tuple[str, ...] = (
    "none", "dirty_plate", "clean_plate", "raw_protein",
    "chopped_protein", "cooked_protein", "raw_rice", "cooked_rice",
    "tortilla", "partial_plate", "finished_burrito", "safety",
    "other",
)
GENERIC_INDEX = {name: index for index, name in enumerate(GENERIC_ITEM_GROUPS)}

REWARD_FEATURE_VERSION = "burrito_relational_reward_v1"
SEMANTIC_FEATURE_VERSION = "burrito_functional_semantic_v1"
STRATEGY_ROLE_VERSION = "burrito_workflow_roles_v2"
# Frozen from identity-matched steak/mushroom workflow states. Equivalent
# pairs have RMS 0.0; the closest distinct workflow states have RMS 0.1961.
SEMANTIC_FALLBACK_MAX_RMS_DISTANCE = 0.10
FUNCTIONAL_PREDICATES: Tuple[str, ...] = (
    "plate_staged",
    "protein_fetched",
    "protein_prepared",
    "protein_cooking",
    "protein_ready",
    "starch_cooking",
    "starch_ready",
    "starch_collected",
    "burrito_assembled",
    "order_delivered",
    "prep_station_available",
    "protein_cooker_available",
    "rice_pot_available",
)


def _generic_item(name: str | None) -> str:
    if name is None:
        return "none"
    value = str(name)
    if value == "dirty_plate":
        return "dirty_plate"
    if value == "clean_plate":
        return "clean_plate"
    if value in {"meat", "mushroom"}:
        return "raw_protein"
    if value in {"chopped_meat", "chopped_mushroom"}:
        return "chopped_protein"
    if value in {"chopped_steak", "fried_mushroom"}:
        return "cooked_protein"
    if value == "rice":
        return "raw_rice"
    if value == "boiled_rice":
        return "cooked_rice"
    if value == "tortilla":
        return "tortilla"
    if value in {"steak_burrito", "mushroom_burrito"}:
        return "finished_burrito"
    if value.endswith("-plate"):
        return "partial_plate"
    if value in {"charcoal", "charcoal-plate", "fire_ext"}:
        return "safety"
    return "other"


def _orientation_code(value: Any) -> int:
    return {
        (0, -1): 0,
        (1, 0): 1,
        (0, 1): 2,
        (-1, 0): 3,
    }.get(tuple(value), 4)


@dataclass(frozen=True)
class StateSchema:
    player_offset: int = 0
    player_width: int = 4
    player_count: int = 2

    @property
    def count_offset(self) -> int:
        return self.player_offset + self.player_width * self.player_count

    @property
    def process_offset(self) -> int:
        return self.count_offset + len(ITEM_NAMES)

    @property
    def task_offset(self) -> int:
        return self.process_offset + len(PROCESS_NAMES) * PROCESS_WIDTH

    @property
    def width(self) -> int:
        # num_plates, max_plates, completed, consecutive, steak_active,
        # mushroom_active, next_recipe, remaining_orders, decision_actor,
        # prep_available, protein_cooker_available, rice_pot_available
        return self.task_offset + 12


SCHEMA = StateSchema()


class BurritoDomainAdapter:
    """Expose task-level physical state without copying any learning code.

    State identity retains physical positions and ingredient identity. Reward
    features omit navigation details, and semantic-neighbour features collapse
    steak/mushroom identities while preserving workflow progress.
    """

    name = "burrito_task_options"
    action_representation = "completion_checked_task_options"
    reward_feature_version = REWARD_FEATURE_VERSION
    semantic_feature_version = SEMANTIC_FEATURE_VERSION
    strategy_roles = (
        "retrieve", "prepare", "start_cook", "stage", "collect",
        "assemble", "serve",
    )

    def __init__(
        self,
        initial_state: Any | None = None,
        *,
        terrain_positions: Mapping[str, Sequence[Tuple[int, int]]] | None = None,
    ):
        self._initial_state: StateVector | None = None
        self._transitions: Dict[Tuple[StateVector, str], StateVector] = {}
        self._station_positions = {
            str(kind): frozenset(tuple(map(int, position)) for position in positions)
            for kind, positions in (terrain_positions or {}).items()
        }
        if initial_state is not None:
            self.bind_initial_state(initial_state)

    def bind_initial_state(
        self, state: Any, *, actor_id: int = 0,
    ) -> StateVector:
        encoded = self.state_key(state, actor_id=actor_id)
        self._initial_state = encoded
        return encoded

    def initial_state(self) -> StateVector:
        if self._initial_state is None:
            raise RuntimeError("bind an initial Burrito state before replay")
        return self._initial_state

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        actor = int(actor_id)
        if actor not in (0, 1):
            raise ValueError("actor_id must be 0 or 1")
        if isinstance(state, tuple):
            if len(state) != SCHEMA.width:
                raise ValueError(
                    f"Burrito state key has width {len(state)}, expected "
                    f"{SCHEMA.width}"
                )
            return tuple(int(value) for value in state)
        if not hasattr(state, "players") or not hasattr(state, "objects"):
            raise TypeError("state must be a BurritoState or encoded tuple")
        row = [0] * SCHEMA.width
        for player_index, player in enumerate(state.players):
            offset = SCHEMA.player_offset + player_index * SCHEMA.player_width
            row[offset] = int(player.position[0])
            row[offset + 1] = int(player.position[1])
            row[offset + 2] = _orientation_code(player.orientation)
            held = player.held_object
            held_name = None if held is None else str(held.name)
            row[offset + 3] = (
                ITEM_INDEX.get(held_name, len(ITEM_NAMES)) + 1
                if held_name is not None else 0
            )

        objects = tuple(state.objects.values())
        for obj in objects:
            name = str(obj.name)
            item_index = ITEM_INDEX.get(name)
            if item_index is not None:
                row[SCHEMA.count_offset + item_index] += 1

        for process_name, process_index in PROCESS_INDEX.items():
            process_objects = [
                obj for obj in objects if str(obj.name) == process_name
            ]
            if not process_objects:
                continue
            offset = SCHEMA.process_offset + process_index * PROCESS_WIDTH
            row[offset] = max(
                int(getattr(obj, "_cooking_tick", -1))
                for obj in process_objects
            )
            row[offset + 1] = max(
                int(getattr(obj, "_cook_time", 0) or 0)
                for obj in process_objects
            )
            row[offset + 2] = max(
                int(getattr(obj, "_warning_tick", -1))
                for obj in process_objects
            )
            row[offset + 3] = max(
                int(getattr(obj, "_waiting_tick", -1))
                for obj in process_objects
            )
            row[offset + 4] = int(any(
                bool(getattr(obj, "is_ready", False))
                for obj in process_objects
            ))
            row[offset + 5] = int(any(
                bool(getattr(obj, "is_warning", False))
                for obj in process_objects
            ))
            row[offset + 6] = int(any(
                bool(getattr(obj, "is_burnt", False))
                for obj in process_objects
            ))

        orders = tuple(getattr(state, "order_list", ()))
        order_names = tuple(
            str(order[0] if isinstance(order, (tuple, list)) else order)
            for order in orders
        )
        task = SCHEMA.task_offset
        row[task] = int(getattr(state, "num_plates", 0))
        row[task + 1] = int(getattr(state, "max_plates", 0))
        row[task + 2] = len(getattr(state, "_complete_orders", ()))
        row[task + 3] = int(getattr(state, "consecutive_orders", 0))
        row[task + 4] = sum("steak_burrito" in name for name in order_names)
        row[task + 5] = sum("mushroom_burrito" in name for name in order_names)
        row[task + 6] = int(getattr(state, "next_recipe", 0))
        row[task + 7] = len(orders)
        row[task + 8] = actor
        self._update_station_bits(row, raw_state=state)
        return tuple(row)

    def _update_station_bits(
        self, row: list[int], *, raw_state: Any | None = None,
    ) -> None:
        state = tuple(row)
        task = SCHEMA.task_offset
        if raw_state is not None and self._station_positions:
            occupied = {
                tuple(map(int, obj.position))
                for obj in raw_state.objects.values()
            }
            row[task + 9] = int(any(
                position not in occupied
                for position in self._station_positions.get("B", ())
            ))
            row[task + 10] = int(any(
                position not in occupied
                for position in self._station_positions.get("G", ())
            ))
            row[task + 11] = int(any(
                position not in occupied
                for position in self._station_positions.get("P", ())
            ))
            return
        # Projected successors have no raw geometry. Under the restricted
        # single-order task graph, component occupancy is the conservative
        # functional equivalent of station capacity.
        prep_capacity = max(1, len(self._station_positions.get("B", ())))
        cooker_capacity = max(1, len(self._station_positions.get("G", ())))
        rice_capacity = max(1, len(self._station_positions.get("P", ())))
        row[task + 9] = int(sum(
            self._count(state, name)
            for name in ("chopped_meat", "chopped_mushroom")
        ) < prep_capacity)
        row[task + 10] = int(sum(
            self._count(state, name)
            for name in ("chopped_steak", "fried_mushroom")
        ) < cooker_capacity)
        row[task + 11] = int(
            self._count(state, "boiled_rice") < rice_capacity
        )

    def record_transition(
        self, state: Any, action: str, next_state: Any,
    ) -> Tuple[StateVector, StateVector]:
        before = self.state_key(state)
        after = self.state_key(next_state)
        self._transitions[(before, str(action))] = after
        return before, after

    def replay_transition(self, state: StateVector, action: str) -> StateVector:
        key = (self.state_key(state), str(action))
        observed = self._transitions.get(key)
        if observed is not None:
            return observed
        successor = self.successor(key[0], key[1])
        if successor is None:
            raise ValueError(f"cannot replay illegal Burrito option {action!r}")
        return successor

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        state = self.initial_state()
        for action in actions:
            state = self.replay_transition(state, str(action))
        return state

    def successor(
        self, state: StateVector, action: str,
    ) -> Optional[StateVector]:
        encoded = self.state_key(state)
        label = str(action)
        observed = self._transitions.get((encoded, label))
        if observed is not None:
            return observed
        if label not in self.legal_actions(encoded, (label,)):
            return None
        return self._project_successor(encoded, label)

    def legal_actions(
        self, state: StateVector, actions: Sequence[str],
    ) -> Tuple[str, ...]:
        encoded = self.state_key(state)
        return tuple(dict.fromkeys(
            str(action) for action in actions
            if str(action) != "stop"
            and self._encoded_option_is_legal(encoded, str(action))
        ))

    def _count(self, state: StateVector, name: str) -> int:
        return int(state[SCHEMA.count_offset + ITEM_INDEX[name]])

    def _ready(self, state: StateVector, name: str) -> bool:
        process = PROCESS_INDEX.get(name)
        if process is None:
            return self._count(state, name) > 0
        return bool(
            state[
                SCHEMA.process_offset + process * PROCESS_WIDTH + 4
            ]
        )

    def object_ready(self, state: StateVector, name: str) -> bool:
        """Expose process readiness without leaking encoded-state offsets."""
        return self._ready(self.state_key(state), str(name))

    def _encoded_option_is_legal(
        self, state: StateVector, action: str,
    ) -> bool:
        label = str(action).upper()
        # Every declared option boundary is empty-handed by construction.
        if any(
            int(state[SCHEMA.player_offset + i * SCHEMA.player_width + 3])
            for i in range(SCHEMA.player_count)
        ):
            return False
        if label == STAGE_CLEAN_PLATE:
            return (
                self._count(state, "clean_plate") == 0
                and self._count(state, "boiled_rice-plate") == 0
                and self._count(state, "dirty_plate") > 0
            )
        if label == START_BOILING_RICE:
            return (
                self._count(state, "boiled_rice") == 0
                and self._count(state, "boiled_rice-plate") == 0
            )
        if label == COLLECT_AND_STAGE_COOKED_RICE:
            return (
                self._count(state, "clean_plate") > 0
                and self._count(state, "boiled_rice") > 0
                and self._ready(state, "boiled_rice")
                and self._count(state, "boiled_rice-plate") == 0
            )
        protein = action_protein(label)
        if protein is None:
            return False
        chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
        cooked = "chopped_steak" if protein == "steak" else "fried_mushroom"
        if label == fetch_and_stage_action(protein):
            return (
                self._count(state, chopped) == 0
                and self._count(state, cooked) == 0
            )
        if label == prepare_and_stage_action(protein):
            return self._count(state, chopped) > 0 and not self._ready(
                state, chopped,
            )
        if label == start_cooking_action(protein):
            return self._count(state, chopped) > 0 and self._ready(
                state, chopped,
            ) and self._count(state, cooked) == 0
        if label == assemble_action(protein):
            return (
                self._count(state, "boiled_rice-plate") > 0
                and self._count(state, cooked) > 0
                and self._ready(state, cooked)
            )
        if label == serve_action(protein):
            return self._count(state, f"{protein}_burrito") > 0
        return False

    def _set_count(self, row: list[int], name: str, value: int) -> None:
        row[SCHEMA.count_offset + ITEM_INDEX[name]] = max(0, int(value))

    def _set_process(
        self, row: list[int], name: str, *, ready: bool,
    ) -> None:
        offset = SCHEMA.process_offset + PROCESS_INDEX[name] * PROCESS_WIDTH
        total = max(1, int(row[offset + 1]) or (3 if name in {
            "dirty_plate", "chopped_meat", "chopped_mushroom",
        } else 80))
        row[offset] = total if ready else 0
        row[offset + 1] = total
        row[offset + 2] = -1
        row[offset + 3] = 0 if ready else -1
        row[offset + 4] = int(ready)
        row[offset + 5] = 0
        row[offset + 6] = 0

    def _project_successor(
        self, state: StateVector, action: str,
    ) -> StateVector:
        row = list(state)
        label = str(action).upper()
        if label == STAGE_CLEAN_PLATE:
            self._set_count(row, "dirty_plate", self._count(state, "dirty_plate") - 1)
            self._set_count(row, "clean_plate", self._count(state, "clean_plate") + 1)
        elif label == START_BOILING_RICE:
            self._set_count(row, "boiled_rice", 1)
            self._set_process(row, "boiled_rice", ready=False)
        elif label == COLLECT_AND_STAGE_COOKED_RICE:
            self._set_count(row, "clean_plate", self._count(state, "clean_plate") - 1)
            self._set_count(row, "boiled_rice", self._count(state, "boiled_rice") - 1)
            self._set_count(row, "boiled_rice-plate", 1)
        else:
            protein = action_protein(label)
            if protein is None:
                return tuple(row)
            chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
            cooked = "chopped_steak" if protein == "steak" else "fried_mushroom"
            if label == fetch_and_stage_action(protein):
                self._set_count(row, chopped, 1)
                self._set_process(row, chopped, ready=False)
            elif label == prepare_and_stage_action(protein):
                self._set_process(row, chopped, ready=True)
            elif label == start_cooking_action(protein):
                self._set_count(row, chopped, self._count(state, chopped) - 1)
                self._set_count(row, cooked, 1)
                self._set_process(row, cooked, ready=False)
            elif label == assemble_action(protein):
                for name in ("boiled_rice-plate", cooked):
                    self._set_count(row, name, self._count(state, name) - 1)
                self._set_count(row, f"{protein}_burrito", 1)
            elif label == serve_action(protein):
                self._set_count(row, f"{protein}_burrito", 0)
                row[SCHEMA.task_offset + 2] += 1
        self._update_station_bits(row)
        return tuple(row)

    def canonical_action(self, action: str) -> str:
        label = str(action).strip().upper()
        known = {
            candidate
            for protein in ("steak", "mushroom")
            for candidate in (
                fetch_and_stage_action(protein),
                prepare_and_stage_action(protein),
                start_cooking_action(protein),
                assemble_action(protein),
                serve_action(protein),
            )
        } | {
            START_BOILING_RICE,
            STAGE_CLEAN_PLATE,
            COLLECT_AND_STAGE_COOKED_RICE,
        }
        if label not in known:
            raise ValueError(f"unknown Burrito macro {action!r}")
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if feature_mode not in {"semantic", "engineered", "raw_state"}:
            raise ValueError(
                "feature_mode must be 'semantic', 'engineered', or 'raw_state'"
            )
        encoded_rows = [
            self.state_key(state_vectors[index])
            for index in range(len(state_vectors))
        ]
        if feature_mode == "raw_state":
            raw = np.asarray(encoded_rows, dtype=np.float32)
        elif feature_mode == "engineered":
            raw = np.asarray([
                self._engineered_features(state) for state in encoded_rows
            ], dtype=np.float32)
        else:
            raw = np.asarray([
                self._semantic_features(state) for state in encoded_rows
            ], dtype=np.float32)
        return self._normalize(
            raw,
            known_mean=known_mean,
            known_scale=known_scale,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
            normalize=normalize,
        )

    def _engineered_features(self, state: StateVector) -> Tuple[float, ...]:
        # Reward learning receives relational workflow features, plus a small
        # grounded suffix so it can distinguish the two public recipes. Grid
        # coordinates and orientations never enter this representation.
        row = list(self._semantic_features(state))
        for name in (
            "chopped_meat", "chopped_mushroom",
            "chopped_steak", "fried_mushroom",
            "steak_burrito", "mushroom_burrito",
        ):
            row.append(float(self._count(state, name)))
        row.extend((
            float(self._ready(state, "chopped_steak")),
            float(self._ready(state, "fried_mushroom")),
        ))
        return tuple(row)

    def functional_predicates(
        self, state: StateVector,
    ) -> Mapping[str, float]:
        encoded = self.state_key(state)
        protein_fetched = any(
            self._count(encoded, name) > 0
            for name in ("chopped_meat", "chopped_mushroom")
        )
        protein_prepared = any(
            self._ready(encoded, name)
            for name in ("chopped_meat", "chopped_mushroom")
        )
        cooked_present = any(
            self._count(encoded, name) > 0
            for name in ("chopped_steak", "fried_mushroom")
        )
        protein_ready = any(
            self._ready(encoded, name)
            for name in ("chopped_steak", "fried_mushroom")
        )
        rice_present = self._count(encoded, "boiled_rice") > 0
        rice_ready = self._ready(encoded, "boiled_rice")
        task = SCHEMA.task_offset
        values = {
            "plate_staged": float(
                self._count(encoded, "clean_plate") > 0
                or self._count(encoded, "boiled_rice-plate") > 0
            ),
            "protein_fetched": float(protein_fetched),
            "protein_prepared": float(protein_prepared),
            "protein_cooking": float(cooked_present and not protein_ready),
            "protein_ready": float(protein_ready),
            "starch_cooking": float(rice_present and not rice_ready),
            "starch_ready": float(
                rice_ready or self._count(encoded, "boiled_rice-plate") > 0
            ),
            "starch_collected": float(
                self._count(encoded, "boiled_rice-plate") > 0
            ),
            "burrito_assembled": float(
                self._count(encoded, "steak_burrito") > 0
                or self._count(encoded, "mushroom_burrito") > 0
            ),
            "order_delivered": float(encoded[task + 2] > 0),
            "prep_station_available": float(encoded[task + 9]),
            "protein_cooker_available": float(encoded[task + 10]),
            "rice_pot_available": float(encoded[task + 11]),
        }
        return {name: values[name] for name in FUNCTIONAL_PREDICATES}

    def _semantic_features(self, state: StateVector) -> Tuple[float, ...]:
        row: list[float] = []
        for player in range(SCHEMA.player_count):
            held_code = int(
                state[
                    SCHEMA.player_offset + player * SCHEMA.player_width + 3
                ]
            )
            held_name = (
                None if held_code == 0 else
                ITEM_NAMES[held_code - 1]
                if held_code - 1 < len(ITEM_NAMES) else "unknown"
            )
            group = GENERIC_INDEX[_generic_item(held_name)]
            row.extend(
                float(index == group)
                for index in range(len(GENERIC_ITEM_GROUPS))
            )
        group_counts = [0.0] * len(GENERIC_ITEM_GROUPS)
        for name in ITEM_NAMES:
            group_counts[GENERIC_INDEX[_generic_item(name)]] += float(
                self._count(state, name)
            )
        row.extend(group_counts)

        process_groups = (
            ("dirty_plate",),
            ("chopped_meat", "chopped_mushroom"),
            ("chopped_steak", "fried_mushroom"),
            ("boiled_rice",),
        )
        for names in process_groups:
            present = [name for name in names if self._count(state, name) > 0]
            if not present:
                row.extend((0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            progress = []
            flags = [0.0, 0.0, 0.0]
            for name in present:
                offset = (
                    SCHEMA.process_offset
                    + PROCESS_INDEX[name] * PROCESS_WIDTH
                )
                progress.append(
                    max(0.0, float(state[offset]))
                    / max(1.0, float(state[offset + 1]))
                )
                for index in range(3):
                    flags[index] = max(
                        flags[index], float(state[offset + 4 + index]),
                    )
            row.extend((1.0, float(np.mean(progress)), *flags))
        task = SCHEMA.task_offset
        row.extend((
            float(state[task]),
            float(state[task + 1]),
            float(state[task + 2]),
            float(state[task + 3]),
            float(state[task + 4] + state[task + 5]),
            float(state[task + 7]),
        ))
        predicates = self.functional_predicates(state)
        row.extend(float(predicates[name]) for name in FUNCTIONAL_PREDICATES)
        return tuple(row)

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
            return (
                raw.astype(np.float32),
                np.zeros(width, dtype=np.float32),
                np.ones(width, dtype=np.float32),
            )
        if normalizer is not None:
            if update_normalizer:
                normalizer.update(raw)
            if normalizer.mean is None:
                mean = np.zeros(width, dtype=np.float32)
                scale = np.ones(width, dtype=np.float32)
                features = raw
            else:
                mean = np.asarray(normalizer.mean, dtype=np.float32)
                scale = np.asarray(normalizer.scale, dtype=np.float32)
                features = normalizer.transform(raw)
        elif known_mean is not None and known_scale is not None:
            mean = np.asarray(known_mean, dtype=np.float32)
            scale = np.asarray(known_scale, dtype=np.float32)
            if mean.shape != (width,) or scale.shape != (width,):
                raise ValueError("known feature statistics have the wrong width")
            features = (raw - mean) / np.where(
                np.abs(scale) > 1e-8, scale, 1.0,
            )
        else:
            mean = raw.mean(axis=0).astype(np.float32)
            scale = raw.std(axis=0).astype(np.float32)
            scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
            features = (raw - mean) / scale
        return (
            np.asarray(features, dtype=np.float32),
            np.asarray(mean, dtype=np.float32),
            np.asarray(scale, dtype=np.float32),
        )

    def action_role(self, action: str) -> str:
        label = str(action).upper()
        if label.startswith("FETCH_AND_STAGE_"):
            return "retrieve"
        if label.startswith("PREPARE_AND_STAGE_"):
            return "prepare"
        if label.startswith("START_COOKING_") or label == START_BOILING_RICE:
            return "start_cook"
        if label == STAGE_CLEAN_PLATE:
            return "stage"
        if label == COLLECT_AND_STAGE_COOKED_RICE:
            return "collect"
        if label.startswith("ASSEMBLE_"):
            return "assemble"
        if label.startswith("SERVE_"):
            return "serve"
        return "other"

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        proteins = {
            protein for protein in map(action_protein, actions)
            if protein is not None
        }
        return (
            tuple(sorted(proteins)),
            tuple(sorted(self.action_role(action) for action in actions)),
        )


__all__ = [
    "BurritoDomainAdapter",
    "FUNCTIONAL_PREDICATES",
    "GENERIC_ITEM_GROUPS",
    "ITEM_NAMES",
    "PROCESS_NAMES",
    "REWARD_FEATURE_VERSION",
    "SCHEMA",
    "SEMANTIC_FALLBACK_MAX_RMS_DISTANCE",
    "SEMANTIC_FEATURE_VERSION",
    "STRATEGY_ROLE_VERSION",
]
