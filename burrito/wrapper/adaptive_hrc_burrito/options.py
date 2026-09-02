"""Reproducible task-level options over Burrito's high-level planner.

The upstream actions are navigation commands whose completion flag does not
always imply completion of the kitchen subtask (for example, one invocation
may perform only part of chopping or washing).  Adaptive-HRC therefore makes
decisions over the options below.  Each option verifies its physical outcome
before returning and leaves the acting player empty-handed at a neutral
parking cell.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

from .runtime import BurritoRuntime
from .macros import (
    COLLECT_AND_STAGE_COOKED_RICE,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    action_protein,
    assemble_action,
    fetch_and_stage_action,
    macro_actions,
    prepare_and_stage_action,
    protein_name,
    serve_action,
    start_cooking_action,
)


_UPSTREAM_STATE_LOCK = threading.RLock()

CONTROLLED_PARKING_POSITIONS: Mapping[
    str, Mapping[int, Tuple[int, int]]
] = {
    # These handoff cells were validated across both actors, recipes, and all
    # four preference policies. They avoid single-access station tiles.
    "burrito_1-2_2p": {0: (1, 1), 1: (8, 1)},
    "burrito": {0: (2, 2), 1: (8, 2)},
}


def _held_name(state: Any, actor_id: int) -> str | None:
    held = state.players[int(actor_id)].held_object
    return None if held is None else str(held.name)


def _objects(state: Any, name: str) -> Tuple[Any, ...]:
    return tuple(
        obj for obj in state.objects.values() if str(obj.name) == str(name)
    )


def _has_object(state: Any, name: str) -> bool:
    return bool(_objects(state, name))


def _is_ready(obj: Any) -> bool:
    return bool(getattr(obj, "is_ready", False))


def _has_ready(state: Any, name: str) -> bool:
    return any(_is_ready(obj) for obj in _objects(state, name))


def macro_is_legal(state: Any, action: str, actor_id: int = 0) -> bool:
    """Preference-neutral macro preconditions at a handoff boundary."""
    label = str(action).upper()
    if _held_name(state, actor_id) is not None:
        return False
    if label == STAGE_CLEAN_PLATE:
        return (
            not _has_object(state, "clean_plate")
            and not _has_object(state, "boiled_rice-plate")
            and _has_object(state, "dirty_plate")
        )
    if label == START_BOILING_RICE:
        return (
            not _has_object(state, "boiled_rice")
            and not _has_object(state, "boiled_rice-plate")
        )
    if label == COLLECT_AND_STAGE_COOKED_RICE:
        return (
            _has_object(state, "clean_plate")
            and _has_ready(state, "boiled_rice")
            and not _has_object(state, "boiled_rice-plate")
        )

    protein = action_protein(label)
    if protein is None:
        return False
    chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
    cooked = "chopped_steak" if protein == "steak" else "fried_mushroom"
    if label == fetch_and_stage_action(protein):
        return not _has_object(state, chopped) and not _has_object(state, cooked)
    if label == prepare_and_stage_action(protein):
        return bool(_objects(state, chopped)) and not _has_ready(state, chopped)
    if label == start_cooking_action(protein):
        return _has_ready(state, chopped) and not _has_object(state, cooked)
    if label == assemble_action(protein):
        return (
            _has_object(state, "boiled_rice-plate")
            and _has_ready(state, cooked)
        )
    if label == serve_action(protein):
        return _has_object(state, f"{protein}_burrito")
    return False


@dataclass(frozen=True)
class OptionExecution:
    action: str
    actor_id: int
    low_level_ticks: int
    primitive_calls: Tuple[str, ...]
    primitive_ticks: Tuple[int, ...]
    sparse_reward: float
    deliveries_before: int
    deliveries_after: int


class OptionExecutionError(RuntimeError):
    """Raised when a task option cannot establish its declared postcondition."""


class BurritoOptionExecutor:
    """Execute completion-checked task options with two physical players."""

    def __init__(
        self,
        runtime: BurritoRuntime,
        *,
        layout: str = "burrito_1-2_2p",
        horizon: int = 1200,
        seed: int = 0,
        parking_positions: Mapping[int, Tuple[int, int]] | None = None,
    ):
        self.runtime = runtime
        self.layout = str(layout)
        self.seed = int(seed)
        self.env = runtime.create_environment(
            self.layout,
            horizon=int(horizon),
            player_types=("H", "H"),
            restrict_capability=False,
            info_level=0,
        )
        if int(self.env.mdp.num_players) != 2:
            raise ValueError("Adaptive-HRC Burrito protocol requires two players")
        self._gridworld, self._environment, self.actions = runtime.upstream_types()
        from overcooked_ai_py.mdp.actions import Action

        self._low_level_action = Action
        self._low_level_stay = int(Action.ACTION_TO_INDEX[Action.STAY])
        self._episode_index = -1
        self._option_index = 0
        self.execution_log: list[OptionExecution] = []
        self.parking_positions = dict(
            parking_positions or self._default_parking_positions()
        )
        self.reset()

    @property
    def state(self) -> Any:
        return self.env.state

    def reset(self) -> Any:
        self._episode_index += 1
        self._option_index = 0
        with _UPSTREAM_STATE_LOCK:
            self._shared_complete_orders().clear()
            self.env.reset()
            self._shared_complete_orders().clear()
        # BurritoState's upstream constructor and deepcopy use one shared
        # mutable default for complete_orders. Keep an executor-owned list and
        # synchronize it around every upstream transition in ``_step``.
        self._episode_complete_orders: list[Any] = []
        self.env.state._complete_orders = self._episode_complete_orders
        self.env.state._bonus_orders = []
        self.env.state.consecutive_orders = 0
        self.execution_log = []
        self.environment_wait_ticks = 0
        # The fixed parking configuration removes station blocking from the
        # option boundary while retaining all physical navigation inside it.
        self._park(0)
        self._park(1)
        return self.env.state

    def _shared_complete_orders(self) -> list[Any]:
        defaults = type(self.env.state).__init__.__defaults__ or ()
        if len(defaults) < 3 or not isinstance(defaults[2], list):
            raise RuntimeError(
                "pinned BurritoState complete_orders default changed"
            )
        return defaults[2]

    def _step(self, joint_action: Sequence[int]) -> Any:
        """Call upstream while isolating its shared completion-history list."""
        with _UPSTREAM_STATE_LOCK:
            shared = self._shared_complete_orders()
            shared.clear()
            shared.extend(self._episode_complete_orders)
            transition = self.env.step(list(joint_action))
            if transition is not None:
                state = transition[0]
                self._episode_complete_orders[:] = list(
                    getattr(state, "_complete_orders", ())
                )
                state._complete_orders = self._episode_complete_orders
            shared.clear()
            return transition

    def legal_actions(
        self, candidates: Sequence[str], *, actor_id: int = 0,
    ) -> Tuple[str, ...]:
        # Preserve the task graph's declared order.  Ordering a set here would
        # silently introduce an alphabetical tie-break into the learner.
        return tuple(dict.fromkeys(
            str(action) for action in candidates
            if macro_is_legal(self.state, str(action), actor_id=actor_id)
        ))

    def execute(self, action: str, *, actor_id: int) -> OptionExecution:
        actor = int(actor_id)
        if actor not in (0, 1):
            raise ValueError("actor_id must be 0 or 1")
        label = str(action).upper()
        if not macro_is_legal(self.state, label, actor_id=actor):
            raise OptionExecutionError(
                f"option {label} is not legal for actor {actor}"
            )
        before_tick = int(self.state.timestep)
        before_deliveries = self._delivery_count()
        before_reward = np.asarray(
            self.env.game_stats["cumulative_sparse_rewards_by_agent"],
            dtype=float,
        ).sum()
        primitives: list[str] = []
        self._current_primitive_ticks: list[int] = []
        with self._deterministic_upstream_rng():
            if label == STAGE_CLEAN_PLATE:
                self._stage_clean_plate(actor, primitives)
            elif label == START_BOILING_RICE:
                self._start_rice(actor, primitives)
            elif label == COLLECT_AND_STAGE_COOKED_RICE:
                self._collect_and_stage_rice(actor, primitives)
            else:
                protein = action_protein(label)
                if protein is None:
                    raise OptionExecutionError(f"unknown option {label!r}")
                if label == fetch_and_stage_action(protein):
                    self._fetch_and_stage(actor, protein, primitives)
                elif label == prepare_and_stage_action(protein):
                    self._prepare_and_stage(actor, protein, primitives)
                elif label == start_cooking_action(protein):
                    self._start_protein(actor, protein, primitives)
                elif label == assemble_action(protein):
                    self._assemble_and_stage(actor, protein, primitives)
                elif label == serve_action(protein):
                    self._serve(actor, protein, primitives)
                else:
                    raise OptionExecutionError(f"unknown option {label!r}")
            if _held_name(self.state, actor) is not None:
                raise OptionExecutionError(
                    f"option {label} left actor {actor} holding "
                    f"{_held_name(self.state, actor)!r}"
                )
            self._park(actor)
        after_reward = np.asarray(
            self.env.game_stats["cumulative_sparse_rewards_by_agent"],
            dtype=float,
        ).sum()
        execution = OptionExecution(
            action=label,
            actor_id=actor,
            low_level_ticks=int(self.state.timestep) - before_tick,
            primitive_calls=tuple(primitives),
            primitive_ticks=tuple(self._current_primitive_ticks),
            sparse_reward=float(after_reward - before_reward),
            deliveries_before=before_deliveries,
            deliveries_after=self._delivery_count(),
        )
        self.execution_log.append(execution)
        self._option_index += 1
        return execution

    @contextmanager
    def _deterministic_upstream_rng(self) -> Iterator[None]:
        state = np.random.get_state()
        # The simulator planner is a controlled executor, not an experimental
        # variable. Its fixed seed deterministically derives a route stream;
        # learner/ground-truth seeds remain independent.
        child = np.random.SeedSequence([
            self.seed & 0xFFFFFFFF,
            self._episode_index & 0xFFFFFFFF,
            self._option_index & 0xFFFFFFFF,
        ])
        np.random.seed(int(child.generate_state(1, dtype=np.uint32)[0]))
        try:
            yield
        finally:
            np.random.set_state(state)

    def _primitive_valid(self, actor: int, primitive: str) -> bool:
        mask = self.actions.agent_action_mask(
            actor,
            self.env.planner.grid_distances,
            self.env.planner.mdp.terrain_pos_dict,
            self.env.planner.terrain_mtx,
            self.state,
        )
        action = getattr(self.actions, primitive)
        return bool(mask[int(action.action_index)])

    def _run_primitive(
        self, actor: int, primitive: str, primitives: list[str],
        *, max_ticks: int = 200, retries: int = 4,
    ) -> int:
        action = getattr(self.actions, primitive)
        action_index = int(action.action_index)
        if not self._primitive_valid(actor, primitive):
            raise OptionExecutionError(
                f"upstream primitive {primitive} is invalid for actor {actor}"
            )
        elapsed = 0
        for _attempt in range(max(1, int(retries))):
            # Both physical players participate in upstream collision-aware
            # planning. The non-acting player receives the upstream STAY
            # option; only ``actor`` receives the task primitive.
            roles = ["A", "A"]
            self.env.setup_planner(
                self.layout, roles, restrict_capability=False,
            )
            joint = [self._low_level_stay, self._low_level_stay]
            joint[actor] = action_index
            for _ in range(max(1, int(max_ticks))):
                transition = self._step(joint)
                if transition is None:
                    raise OptionExecutionError(
                        f"BurritoEnv suppressed an exception during {primitive}"
                    )
                _state, _reward, done, info = transition
                elapsed += 1
                if done:
                    raise OptionExecutionError(
                        f"episode ended during primitive {primitive}"
                    )
                status = info.get("action_status", ())[actor]
                if bool(status["status"]):
                    if int(status["prev_action"]) == action_index:
                        primitives.append(primitive)
                        self._current_primitive_ticks.append(elapsed)
                        return elapsed
                    break
        raise OptionExecutionError(
            f"primitive {primitive} did not complete after {retries} attempts"
        )

    def _run_primitive_until(
        self,
        actor: int,
        primitive: str,
        predicate: Any,
        primitives: list[str],
        *,
        max_ticks: int = 200,
        retries: int = 4,
    ) -> int:
        """Stop a native macro on the first physical postcondition frame."""
        action = getattr(self.actions, primitive)
        action_index = int(action.action_index)
        if not self._primitive_valid(actor, primitive):
            raise OptionExecutionError(
                f"upstream primitive {primitive} is invalid for actor {actor}"
            )
        elapsed = 0
        for _attempt in range(max(1, int(retries))):
            roles = ["A", "A"]
            self.env.setup_planner(
                self.layout, roles, restrict_capability=False,
            )
            joint = [self._low_level_stay, self._low_level_stay]
            joint[actor] = action_index
            for _ in range(max(1, int(max_ticks))):
                transition = self._step(joint)
                if transition is None:
                    raise OptionExecutionError(
                        f"BurritoEnv suppressed an exception during {primitive}"
                    )
                _state, _reward, done, info = transition
                elapsed += 1
                if done:
                    raise OptionExecutionError(
                        f"episode ended during primitive {primitive}"
                    )
                if bool(predicate()):
                    primitives.append(primitive)
                    self._current_primitive_ticks.append(elapsed)
                    return elapsed
                status = info.get("action_status", ())[actor]
                if bool(status["status"]):
                    break
        raise OptionExecutionError(
            f"primitive {primitive} ended before its physical postcondition"
        )

    def _repeat_until(
        self,
        actor: int,
        primitive: str,
        predicate: Any,
        primitives: list[str],
        *,
        max_calls: int = 12,
    ) -> None:
        for _ in range(max(1, int(max_calls))):
            if bool(predicate()):
                return
            self._run_primitive(actor, primitive, primitives)
        if not bool(predicate()):
            raise OptionExecutionError(
                f"{primitive} completed without reaching its option outcome"
            )

    def _stage_clean_plate(self, actor: int, primitives: list[str]) -> None:
        self._repeat_until(
            actor, "GRAB_DIRTY_PLATE",
            lambda: _held_name(self.state, actor) == "dirty_plate",
            primitives,
        )
        self._repeat_until(
            actor, "GO_TO_SINK_TO_WASH_PLATE",
            lambda: _has_object(self.state, "clean_plate"),
            primitives,
        )

    def _fetch_and_stage(
        self, actor: int, protein: str, primitives: list[str],
    ) -> None:
        raw = "meat" if protein == "steak" else "mushroom"
        chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
        grab = "GRAB_MEAT" if protein == "steak" else "GRAB_MUSHROOM"
        self._repeat_until(
            actor, grab, lambda: _held_name(self.state, actor) == raw,
            primitives,
        )
        self._run_primitive_until(
            actor, "GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT",
            lambda: bool(_objects(self.state, chopped)), primitives,
        )
        staged = _objects(self.state, chopped)
        if not staged or _has_ready(self.state, chopped):
            raise OptionExecutionError(
                f"fetch macro did not leave unprepared {chopped} staged"
            )

    def _prepare_and_stage(
        self, actor: int, protein: str, primitives: list[str],
    ) -> None:
        chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
        self._repeat_until(
            actor, "GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT",
            lambda: _has_ready(self.state, chopped), primitives,
        )

    def _start_rice(self, actor: int, primitives: list[str]) -> None:
        self._repeat_until(
            actor, "GRAB_RICE",
            lambda: _held_name(self.state, actor) == "rice", primitives,
        )
        self._run_primitive(
            actor, "GO_TO_POT_AND_BOIL_RICE", primitives,
        )
        if not _has_object(self.state, "boiled_rice"):
            raise OptionExecutionError("rice option did not start the pot")

    def _start_protein(
        self, actor: int, protein: str, primitives: list[str],
    ) -> None:
        chopped = "chopped_meat" if protein == "steak" else "chopped_mushroom"
        cooked = "chopped_steak" if protein == "steak" else "fried_mushroom"
        grab = (
            "GRAB_CHOPPED_MEAT" if protein == "steak"
            else "GRAB_CHOPPED_MUSHROOM"
        )
        self._repeat_until(
            actor, grab, lambda: _held_name(self.state, actor) == chopped,
            primitives,
        )
        self._run_primitive(
            actor, "GO_TO_PAN_AND_FRY_INGREDIENT", primitives,
        )
        if not _has_object(self.state, cooked):
            raise OptionExecutionError(
                f"protein option did not start {cooked}"
            )

    def _collect_and_stage_rice(
        self, actor: int, primitives: list[str],
    ) -> None:
        self._repeat_until(
            actor, "GRAB_CLEAN_PLATE",
            lambda: _held_name(self.state, actor) == "clean_plate",
            primitives,
        )
        self._repeat_until(
            actor, "GET_BOILED_RICE",
            lambda: _held_name(self.state, actor) == "boiled_rice-plate",
            primitives,
        )
        self._repeat_until(
            actor, "PUT_DOWN_OBJECT",
            lambda: (
                _held_name(self.state, actor) is None
                and _has_object(self.state, "boiled_rice-plate")
            ),
            primitives,
        )

    def _assemble_and_stage(
        self, actor: int, protein: str, primitives: list[str],
    ) -> None:
        cooked = "chopped_steak" if protein == "steak" else "fried_mushroom"
        get_protein = (
            "GET_STEAK_FROM_GRILLER" if protein == "steak"
            else "GET_MURHSOOM_FROM_GRILLER"
        )
        final = f"{protein}_burrito"
        self._repeat_until(
            actor, "GRAB_ONE_INGREDIENT_PLATE",
            lambda: _held_name(self.state, actor) == "boiled_rice-plate",
            primitives,
        )
        self._repeat_until(
            actor, get_protein,
            lambda: cooked in (_held_name(self.state, actor) or ""),
            primitives,
        )
        self._repeat_until(
            actor, "GRAB_TORTILLA",
            lambda: _held_name(self.state, actor) == final,
            primitives,
        )
        self._repeat_until(
            actor, "PUT_DOWN_OBJECT",
            lambda: (
                _held_name(self.state, actor) is None
                and _has_object(self.state, final)
            ),
            primitives,
        )

    def _serve(
        self, actor: int, protein: str, primitives: list[str],
    ) -> None:
        final = f"{protein}_burrito"
        grab = (
            "GRAB_STEAK_BURRITO_PLATE" if protein == "steak"
            else "GRAB_MURHSOOM_BURRITO_PLATE"
        )
        self._repeat_until(
            actor, grab,
            lambda: _held_name(self.state, actor) == final,
            primitives,
        )
        before = self._delivery_count()
        self._run_primitive(actor, "GO_TO_SERVE_DISH", primitives)
        if _held_name(self.state, actor) is not None or self._delivery_count() <= before:
            raise OptionExecutionError(
                f"serve macro did not deliver a {protein} burrito"
            )

    def advance_environment(self, ticks: int = 1) -> int:
        """Advance passive cooking frames without creating preference actions."""
        requested = max(0, int(ticks))
        self.env.setup_planner(
            self.layout, ["H", "H"], restrict_capability=False,
        )
        for _ in range(requested):
            transition = self._step([
                self._low_level_stay, self._low_level_stay,
            ])
            if transition is None:
                raise OptionExecutionError(
                    "BurritoEnv suppressed an exception while waiting"
                )
            _state, _reward, done, _info = transition
            if done:
                raise OptionExecutionError("episode ended while cooking")
        self.environment_wait_ticks += requested
        return requested

    def _delivery_count(self) -> int:
        return int(sum(
            len(events)
            for events in self.env.game_stats.get("dish_delivery", ())
        ))

    def _walkable_cells(self) -> set[Tuple[int, int]]:
        terrain = np.asarray(self.env.mdp.terrain_mtx)
        return {
            (int(x), int(y))
            for y in range(terrain.shape[0])
            for x in range(terrain.shape[1])
            if str(terrain[y, x]) == " "
        }

    def _default_parking_positions(self) -> Dict[int, Tuple[int, int]]:
        cells = self._walkable_cells()
        if len(cells) < 2:
            raise ValueError("layout has no walkable floor cells")
        controlled = CONTROLLED_PARKING_POSITIONS.get(self.layout)
        if controlled is not None:
            resolved = {
                int(actor): tuple(position)
                for actor, position in controlled.items()
            }
            if set(resolved) != {0, 1} or any(
                position not in cells for position in resolved.values()
            ):
                raise ValueError(
                    f"controlled parking is invalid for layout {self.layout!r}"
                )
            return resolved
        # Boundary cells can be the sole interaction tile for a dispenser or
        # station (notably the mushroom dispenser in ``burrito``). Park on
        # maximally connected interior cells, choosing the farthest pair so a
        # stationary teammate neither blocks a task goal nor crowds the actor.
        degree = {
            cell: sum(
                (cell[0] + dx, cell[1] + dy) in cells
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            )
            for cell in cells
        }
        candidates = sorted(cells)
        first, second = max(
            (
                (left, right)
                for index, left in enumerate(candidates)
                for right in candidates[index + 1:]
            ),
            key=lambda pair: (
                min(degree[pair[0]], degree[pair[1]]),
                degree[pair[0]] + degree[pair[1]],
                abs(pair[0][0] - pair[1][0])
                + abs(pair[0][1] - pair[1][1]),
                pair,
            ),
        )
        return {0: first, 1: second}

    def _shortest_path(
        self, start: Tuple[int, int], goal: Tuple[int, int],
        blocked: Iterable[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], ...]:
        if start == goal:
            return (start,)
        walkable = self._walkable_cells() - set(blocked)
        walkable.add(start)
        queue = deque([start])
        parent: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}
        while queue:
            current = queue.popleft()
            for delta in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nxt = (current[0] + delta[0], current[1] + delta[1])
                if nxt not in walkable or nxt in parent:
                    continue
                parent[nxt] = current
                if nxt == goal:
                    path = [nxt]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return tuple(reversed(path))
                queue.append(nxt)
        raise OptionExecutionError(
            f"no collision-free path from {start} to parking cell {goal}"
        )

    def _park(self, actor: int) -> None:
        if _held_name(self.state, actor) is not None:
            raise OptionExecutionError("cannot park an actor holding an object")
        goal = tuple(self.parking_positions[actor])
        start = tuple(self.state.players[actor].position)
        other = tuple(self.state.players[1 - actor].position)
        path = self._shortest_path(start, goal, (other,))
        if len(path) <= 1:
            return
        self.env.setup_planner(
            self.layout, ["H", "H"], restrict_capability=False,
        )
        for current, nxt in zip(path, path[1:]):
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            joint = [self._low_level_stay, self._low_level_stay]
            joint[actor] = int(
                self._low_level_action.ACTION_TO_INDEX[direction]
            )
            transition = self._step(joint)
            if transition is None:
                raise OptionExecutionError(
                    "BurritoEnv suppressed an exception while parking"
                )
            state, _reward, done, _info = transition
            if done or tuple(state.players[actor].position) != nxt:
                raise OptionExecutionError(
                    f"actor {actor} failed to reach parking path cell {nxt}"
                )


__all__ = [
    "BurritoOptionExecutor",
    "COLLECT_AND_STAGE_COOKED_RICE",
    "CONTROLLED_PARKING_POSITIONS",
    "OptionExecution",
    "OptionExecutionError",
    "STAGE_CLEAN_PLATE",
    "START_BOILING_RICE",
    "action_protein",
    "assemble_action",
    "fetch_and_stage_action",
    "macro_actions",
    "macro_is_legal",
    "prepare_and_stage_action",
    "serve_action",
    "start_cooking_action",
]
