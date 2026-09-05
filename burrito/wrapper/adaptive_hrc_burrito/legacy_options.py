"""Audited compatibility executor for Burrito's four declared legacy dishes.

The pinned repository still advertises these dishes and layouts, but its
current interaction handler only cooks rice and chopped burrito proteins.  We
therefore leave the vendored source untouched and restore only the missing
raw-steak/raw-chicken station transitions here.  Every restored transition is
reported in ``compatibility_calls``.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
import threading
from typing import Any, Iterator, Sequence, Tuple
import zlib

import numpy as np

from .catalog import get_recipe
from .options import OptionExecution, OptionExecutionError, seed_upstream_rng
from .runtime import BurritoRuntime


_LEGACY_LOCK = threading.RLock()

# Legacy dishes the pinned interaction handler still advertises but no longer
# cooks.  The wrapper both creates and ticks them; every such call is reported
# through ``compatibility_calls``.
_COMPATIBILITY_COOKED = frozenset({"steak", "boiled_chicken"})


def _held_name(state: Any, actor: int) -> str | None:
    held = state.players[int(actor)].held_object
    return None if held is None else str(held.name)


class LegacyBurritoOptionExecutor:
    """Physical navigation plus isolated legacy station compatibility."""

    def __init__(
        self,
        runtime: BurritoRuntime,
        recipe_id: str,
        *,
        horizon: int = 1200,
        seed: int = 0,
    ):
        self.runtime = runtime
        self.horizon = int(horizon)
        self.seed = int(seed)
        self._build(recipe_id)

    @property
    def state(self) -> Any:
        return self.env.state

    @property
    def compatibility_dynamics(self) -> bool:
        return True

    def _build(self, recipe_id: str) -> None:
        recipe = get_recipe(recipe_id)
        if recipe.environment != "burrito" or not recipe.compatibility_dynamics:
            raise ValueError(f"{recipe_id!r} is not a legacy Burrito recipe")
        self.recipe = recipe
        self.env = self.runtime.create_environment(
            recipe.layout,
            horizon=self.horizon,
            player_types=("H", "H"),
            restrict_capability=False,
        )
        self.env.planner = None
        from burrito.mdp.burrito_mdp import (
            Burrito_Recipe, ChickenState, IdObjectState, SteakState,
        )
        from overcooked_ai_py.mdp.actions import Action

        self.Action = Action
        self.ChickenState = ChickenState
        self.IdObjectState = IdObjectState
        self.SteakState = SteakState
        self._recipe_type = Burrito_Recipe
        self.stage_counters = self._select_stage_counters(3)
        self.parking_positions = self._select_parking_positions()
        self.reset()

    def reset(self, recipe_id: str | None = None) -> Any:
        if recipe_id is not None and recipe_id != self.recipe.recipe_id:
            self._build(recipe_id)
            return self.state
        with _LEGACY_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
            self._shared_orders().clear()
            self.env.reset(regen_mdp=False)
            self._shared_orders().clear()
        self._episode_complete_orders: list[Any] = []
        self.env.state._complete_orders = self._episode_complete_orders
        target_order = [self.recipe.upstream_dish, self.horizon, self.horizon]
        self.env.mdp.all_recipes = [list(target_order) for _ in range(4)]
        self.env.state.order_list = [list(target_order) for _ in range(3)]
        self.env.state._order_display_list = self.env.state.order_list
        self.env.state.next_recipe = 0
        self.execution_log: list[OptionExecution] = []
        self.compatibility_calls: list[str] = []
        self.environment_wait_ticks = 0
        self._compatibility_cook_ticks = 0
        return self.state

    def _shared_orders(self) -> list[Any]:
        defaults = type(self.env.state).__init__.__defaults__ or ()
        if len(defaults) < 3 or not isinstance(defaults[2], list):
            raise RuntimeError("pinned Burrito completion-history contract changed")
        return defaults[2]

    def _step(self, actor: int, action: Any) -> Tuple[Any, float, bool, Any]:
        joint = [
            int(self.Action.ACTION_TO_INDEX[self.Action.STAY]),
            int(self.Action.ACTION_TO_INDEX[self.Action.STAY]),
        ]
        joint[int(actor)] = int(self.Action.ACTION_TO_INDEX[action])
        with _LEGACY_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
            shared = self._shared_orders()
            shared[:] = self._episode_complete_orders
            self._tick_compatibility_cooking()
            transition = self.env.step(joint)
            if transition is None:
                raise OptionExecutionError("BurritoEnv suppressed a legacy-step error")
            state = transition[0]
            self._episode_complete_orders[:] = list(state._complete_orders)
            state._complete_orders = self._episode_complete_orders
            shared.clear()
        return transition

    def _plans_to(self, feature: Tuple[int, int], actor: int):
        start = tuple(self.state.players[actor].position)
        start_orientation = tuple(self.state.players[actor].orientation)
        walkable = set(map(tuple, self.env.mdp.get_valid_player_positions()))
        blocked = {tuple(self.state.players[1 - actor].position)}
        candidates = []
        for direction in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            goal = (feature[0] - direction[0], feature[1] - direction[1])
            if goal not in walkable or goal in blocked:
                continue
            queue = deque([start])
            parent = {start: None}
            while queue and goal not in parent:
                current = queue.popleft()
                for delta in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nxt = (current[0] + delta[0], current[1] + delta[1])
                    if nxt in parent or nxt in blocked or nxt not in walkable:
                        continue
                    parent[nxt] = current
                    queue.append(nxt)
            if goal not in parent:
                continue
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            actions = [
                (right[0] - left[0], right[1] - left[1])
                for left, right in zip(path, path[1:])
            ]
            orientation = actions[-1] if actions else start_orientation
            if orientation != direction:
                actions.append(direction)
            actions.append(self.Action.INTERACT)
            candidates.append((len(actions), tuple(actions)))
        return sorted(candidates, key=lambda item: (item[0], repr(item[1])))

    def _navigate(self, actor: int, feature: Tuple[int, int], *, interact: bool) -> int:
        plans = self._plans_to(feature, actor)
        if not plans:
            raise OptionExecutionError(f"actor {actor} cannot reach {feature}")
        actions = plans[0][1]
        if not interact:
            if not actions or actions[-1] != self.Action.INTERACT:
                raise OptionExecutionError("motion planner omitted final interaction")
            actions = actions[:-1]
        start = int(self.state.timestep)
        for action in actions:
            _state, _reward, done, _info = self._step(actor, action)
            if done:
                raise OptionExecutionError("legacy Burrito episode ended")
        return int(self.state.timestep) - start

    def _feature(self, terrain: str, actor: int) -> Tuple[int, int]:
        candidates = []
        for position in self.env.mdp.terrain_pos_dict.get(terrain, ()):
            plans = self._plans_to(tuple(position), actor)
            if plans:
                candidates.append((plans[0][0], tuple(position)))
        if not candidates:
            raise OptionExecutionError(f"actor {actor} cannot reach station {terrain}")
        return min(candidates)[1]

    def _select_stage_counters(self, count: int) -> Tuple[Tuple[int, int], ...]:
        candidates = []
        for position in self.env.mdp.get_counter_locations():
            costs = []
            for actor in (0, 1):
                plans = self._plans_to(tuple(position), actor)
                if not plans:
                    break
                costs.append(plans[0][0])
            if len(costs) == 2:
                candidates.append((max(costs), sum(costs), tuple(position)))
        selected = tuple(item[2] for item in sorted(candidates)[:count])
        if len(selected) < count:
            raise OptionExecutionError("legacy layout lacks shared staging counters")
        return selected

    def _object_position(self, name: str) -> Tuple[int, int] | None:
        for position, obj in self.state.objects.items():
            if str(obj.name) == name:
                return tuple(position)
        return None

    def _counter_for(self, name: str) -> Tuple[int, int]:
        mapping = {"meat": 0, "chicken": 0, "onion": 1, "final": 2}
        return self.stage_counters[mapping[name]]

    def legal_actions(
        self, candidates: Sequence[str], *, actor_id: int = 0,
    ) -> Tuple[str, ...]:
        if _held_name(self.state, actor_id) is not None:
            return ()
        legal = []
        for candidate in candidates:
            label = str(candidate).upper()
            if label == "FETCH_MEAT":
                allowed = self._object_position("meat") is None
            elif label == "FETCH_CHICKEN":
                allowed = self._object_position("chicken") is None
            elif label == "FETCH_ONION":
                allowed = self._object_position("onion") is None
            elif label == "COOK_STEAK":
                allowed = self._object_position("meat") is not None
            elif label == "BOIL_CHICKEN":
                allowed = self._object_position("chicken") is not None
            elif label == "CHOP_ONION":
                allowed = self._object_position("onion") is not None
            elif label == "ASSEMBLE_STEAK_ONION":
                allowed = self._is_ready_dish("steak") and self._ready_garnish()
            elif label == "ASSEMBLE_CHICKEN_ONION":
                allowed = self._is_ready_dish("boiled_chicken") and self._ready_garnish()
            elif label.startswith("SERVE_"):
                final = {
                    "SERVE_STEAK": "steak",
                    "SERVE_BOILED_CHICKEN": "boiled_chicken",
                    "SERVE_STEAK_ONION": "steak_onion",
                    "SERVE_CHICKEN_ONION": "boiled_chicken_onion",
                }.get(label)
                allowed = final is not None and self._object_position(final) is not None
            else:
                allowed = False
            if allowed:
                legal.append(label)
        return tuple(dict.fromkeys(legal))

    def _tick_compatibility_cooking(self) -> None:
        """Advance the legacy dishes the pinned transition does not tick."""
        for obj in self.state.objects.values():
            if str(obj.name) not in _COMPATIBILITY_COOKED:
                continue
            if getattr(obj, "is_cooking", False):
                obj.cook()
                self._compatibility_cook_ticks += 1

    def _is_ready_dish(self, name: str) -> bool:
        position = self._object_position(name)
        return bool(
            position is not None
            and getattr(self.state.get_object(position), "is_ready", False)
        )

    def _ready_garnish(self) -> bool:
        position = self._object_position("garnish")
        return bool(
            position is not None
            and getattr(self.state.get_object(position), "is_ready", False)
        )

    def _pickup_and_stage(
        self, actor: int, terrain: str, name: str, calls: list[str], ticks: list[int],
    ) -> None:
        ticks.append(self._navigate(actor, self._feature(terrain, actor), interact=True))
        calls.append(f"PICKUP_{name.upper()}")
        if name == "chicken":
            raw = self.state.players[actor].remove_object()
            wrapped = self.ChickenState(
                raw.id, "chicken", tuple(self.state.players[actor].position),
                ingredients=[], cooking_tick=-1,
            )
            self.state.players[actor].set_object(wrapped)
            self.compatibility_calls.append("COMPAT_STAGEABLE_CHICKEN")
            calls.append("COMPAT_STAGEABLE_CHICKEN")
        target = self._counter_for(name)
        ticks.append(self._navigate(actor, target, interact=True))
        calls.append(f"STAGE_{name.upper()}")

    def _pickup_staged(
        self, actor: int, name: str, calls: list[str], ticks: list[int],
    ) -> None:
        position = self._object_position(name)
        if position is None:
            raise OptionExecutionError(f"no staged {name}")
        ticks.append(self._navigate(actor, position, interact=True))
        calls.append(f"PICKUP_STAGED_{name.upper()}")

    def _compat_cook(
        self, actor: int, raw_name: str, terrain: str, cooked_name: str,
        calls: list[str], ticks: list[int],
    ) -> None:
        self._pickup_staged(actor, raw_name, calls, ticks)
        station = self._feature(terrain, actor)
        ticks.append(self._navigate(actor, station, interact=False))
        self._step(actor, self.Action.STAY)
        raw = self.state.players[actor].remove_object()
        object_id = int(self.state.obj_count)
        if cooked_name == "steak":
            cooked = self.SteakState(
                object_id, cooked_name, station, [raw], cooking_tick=-1,
            )
        else:
            cooked = self.ChickenState(
                object_id, cooked_name, station, [raw], cooking_tick=-1,
            )
        # The pinned transition only ticks chopped_steak / fried_mushroom /
        # boiled_rice, so these legacy dishes are never advanced by the
        # environment.  The first integration worked around that with
        # auto_finish(), which made legacy cooking instantaneous and left
        # low_level_ticks incomparable with the natively executed recipes.
        # Start the cook instead and advance it on the wrapper's own clock, at
        # the layout's configured duration for this protein.
        cooked.begin_cooking()
        self.state.add_object(cooked, station)
        self.state.obj_count = object_id + 1
        marker = f"COMPAT_{raw_name.upper()}_AT_{terrain}"
        self.compatibility_calls.append(marker)
        calls.append(marker)
        ticks.append(1)

    def _take_cooked(
        self, actor: int, name: str, calls: list[str], ticks: list[int],
    ) -> None:
        position = self._object_position(name)
        if position is None:
            raise OptionExecutionError(f"no cooked {name}")
        ticks.append(self._navigate(actor, position, interact=False))
        self._step(actor, self.Action.STAY)
        obj = self.state.remove_object(position)
        self.state.players[actor].set_object(obj)
        marker = f"COMPAT_PICKUP_{name.upper()}"
        self.compatibility_calls.append(marker)
        calls.append(marker)
        ticks.append(1)

    def execute(self, action: str, *, actor_id: int) -> OptionExecution:
        actor = int(actor_id)
        label = str(action).upper()
        if label not in self.legal_actions((label,), actor_id=actor):
            raise OptionExecutionError(f"legacy option {label} is not legal")
        before_tick = int(self.state.timestep)
        before_deliveries = self._delivery_count()
        before_reward = float(np.asarray(
            self.env.game_stats["cumulative_sparse_rewards_by_agent"]
        ).sum())
        calls: list[str] = []
        ticks: list[int] = []
        with self._deterministic_rng(label, actor):
            if label == "FETCH_MEAT":
                self._pickup_and_stage(actor, "M", "meat", calls, ticks)
            elif label == "FETCH_CHICKEN":
                self._pickup_and_stage(actor, "C", "chicken", calls, ticks)
            elif label == "FETCH_ONION":
                self._pickup_and_stage(actor, "O", "onion", calls, ticks)
            elif label == "COOK_STEAK":
                self._compat_cook(actor, "meat", "G", "steak", calls, ticks)
            elif label == "BOIL_CHICKEN":
                self._compat_cook(actor, "chicken", "P", "boiled_chicken", calls, ticks)
            elif label == "CHOP_ONION":
                self._pickup_staged(actor, "onion", calls, ticks)
                board = self._feature("B", actor)
                ticks.append(self._navigate(actor, board, interact=True))
                calls.append("START_CHOP_ONION")
                for _ in range(20):
                    if self._ready_garnish():
                        break
                    self._step(actor, self.Action.STAY)
                    self.state.get_object(board).chop()
                if not self._ready_garnish():
                    raise OptionExecutionError("onion garnish did not finish chopping")
                calls.append("COMPAT_CHOP_ONION")
                self.compatibility_calls.append("COMPAT_CHOP_ONION")
            elif label.startswith("ASSEMBLE_"):
                cooked = "steak" if "STEAK" in label else "boiled_chicken"
                self._take_cooked(actor, cooked, calls, ticks)
                board = self._object_position("garnish")
                if board is None:
                    raise OptionExecutionError("no garnish to assemble")
                ticks.append(self._navigate(actor, board, interact=True))
                calls.append(label)
                if "CHICKEN" in label:
                    assembled = self.state.players[actor].remove_object()
                    wrapped = self.ChickenState(
                        assembled.id,
                        "boiled_chicken_onion",
                        tuple(self.state.players[actor].position),
                        ingredients=[],
                        cooking_tick=-1,
                    )
                    self.state.players[actor].set_object(wrapped)
                    calls.append("COMPAT_STAGEABLE_CHICKEN_ONION")
                    self.compatibility_calls.append(
                        "COMPAT_STAGEABLE_CHICKEN_ONION"
                    )
                ticks.append(self._navigate(actor, self._counter_for("final"), interact=True))
                calls.append("STAGE_FINAL")
            elif label.startswith("SERVE_"):
                cooked = {
                    "SERVE_STEAK": "steak",
                    "SERVE_BOILED_CHICKEN": "boiled_chicken",
                    "SERVE_STEAK_ONION": "steak_onion",
                    "SERVE_CHICKEN_ONION": "boiled_chicken_onion",
                }[label]
                position = self._object_position(cooked)
                terrain = self.env.mdp.get_terrain_type_at_pos(position)
                if terrain in {"G", "P"}:
                    self._take_cooked(actor, cooked, calls, ticks)
                else:
                    self._pickup_staged(actor, cooked, calls, ticks)
                ticks.append(self._navigate(actor, self._feature("S", actor), interact=True))
                calls.append(label)
            if _held_name(self.state, actor) is not None:
                raise OptionExecutionError(f"legacy option {label} ended with held object")
            self._park(actor)
        after_reward = float(np.asarray(
            self.env.game_stats["cumulative_sparse_rewards_by_agent"]
        ).sum())
        execution = OptionExecution(
            action=label,
            actor_id=actor,
            low_level_ticks=int(self.state.timestep) - before_tick,
            primitive_calls=tuple(calls),
            primitive_ticks=tuple(ticks),
            sparse_reward=after_reward - before_reward,
            deliveries_before=before_deliveries,
            deliveries_after=self._delivery_count(),
        )
        self.execution_log.append(execution)
        return execution

    @contextmanager
    def _deterministic_rng(self, action: str, actor: int) -> Iterator[None]:
        state = np.random.get_state()
        seed_upstream_rng(self.seed, action, actor)
        try:
            yield
        finally:
            np.random.set_state(state)

    def advance_environment(self, ticks: int = 1) -> int:
        count = max(0, int(ticks))
        for _ in range(count):
            _state, _reward, done, _info = self._step(0, self.Action.STAY)
            if done:
                raise OptionExecutionError("legacy episode ended while waiting")
        self.environment_wait_ticks += count
        return count

    def _select_parking_positions(self) -> dict:
        """Validated parking cells for the legacy layout.

        Derived and checked against the layout instead of hardcoded, so a
        layout change fails here with a clear message rather than as an opaque
        mid-episode routing error.
        """
        walkable = sorted(map(tuple, self.env.mdp.get_valid_player_positions()))
        if len(walkable) < 2:
            raise OptionExecutionError(
                f"layout {self.recipe.layout!r} has fewer than two floor cells"
            )
        stations = {
            tuple(position)
            for terrain in ("M", "C", "O", "B", "G", "P", "S")
            for position in self.env.mdp.terrain_pos_dict.get(terrain, ())
        }
        reserved = set(self.stage_counters)

        def approach_load(cell):
            return sum(
                (cell[0] + dx, cell[1] + dy) in stations | reserved
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            )

        first, second = max(
            (
                (left, right)
                for index, left in enumerate(walkable)
                for right in walkable[index + 1:]
            ),
            key=lambda pair: (
                -(approach_load(pair[0]) + approach_load(pair[1])),
                abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]),
                pair,
            ),
        )
        return {0: first, 1: second}

    def _park(self, actor: int) -> None:
        goal = self.parking_positions[actor]
        start = tuple(self.state.players[actor].position)
        if start == goal:
            return
        walkable = set(map(tuple, self.env.mdp.get_valid_player_positions()))
        blocked = {tuple(self.state.players[1 - actor].position)}
        queue = deque([start])
        parent = {start: None}
        while queue and goal not in parent:
            current = queue.popleft()
            for delta in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nxt = (current[0] + delta[0], current[1] + delta[1])
                if nxt in parent or nxt in blocked or nxt not in walkable:
                    continue
                parent[nxt] = current
                queue.append(nxt)
        if goal not in parent:
            raise OptionExecutionError(f"actor {actor} cannot reach parking cell")
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        for left, right in zip(path, path[1:]):
            direction = (right[0] - left[0], right[1] - left[1])
            _state, _reward, done, _info = self._step(actor, direction)
            if done or tuple(self.state.players[actor].position) != right:
                raise OptionExecutionError(f"actor {actor} failed to park")

    def _delivery_count(self) -> int:
        return int(sum(
            len(events) for events in self.env.game_stats.get("dish_delivery", ())
        ))


__all__ = ["LegacyBurritoOptionExecutor"]
