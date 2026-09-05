"""Completion-checked task options for standard Overcooked."""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
import threading
from typing import Any, Dict, Iterator, Sequence, Tuple
import zlib

import numpy as np

from .catalog import RecipeSpec, get_recipe
from .options import OptionExecution, OptionExecutionError, seed_upstream_rng
from .runtime import BurritoRuntime


_OVERCOOKED_LOCK = threading.RLock()


def _held_name(state: Any, actor: int) -> str | None:
    held = state.players[int(actor)].held_object
    return None if held is None else str(held.name)


class OvercookedOptionExecutor:
    """Physically execute one of the nine onion/tomato recipes."""

    def __init__(
        self,
        runtime: BurritoRuntime,
        recipe_id: str,
        *,
        horizon: int = 800,
        seed: int = 0,
    ):
        self.runtime = runtime
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.execution_log: list[OptionExecution] = []
        self._build(recipe_id)

    def _build(self, recipe_id: str) -> None:
        recipe = get_recipe(recipe_id)
        if recipe.environment != "overcooked":
            raise ValueError(f"{recipe_id!r} is not a standard Overcooked recipe")
        self.recipe = recipe
        self.env = self.runtime.create_overcooked_environment(
            recipe.layout, recipe.ingredients, horizon=self.horizon,
        )
        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.mdp.overcooked_mdp import Recipe
        from overcooked_ai_py.planning.planners import MotionPlanner

        self.Action = Action
        self._recipe_type = Recipe
        self.motion_planner = MotionPlanner(
            self.env.mdp, self.env.mdp.get_counter_locations(),
        )
        self.stage_counter = self._select_stage_counter()
        self.parking_positions = self._select_parking_positions()
        self.reset()

    @property
    def state(self) -> Any:
        return self.env.state

    @property
    def compatibility_dynamics(self) -> bool:
        return False

    def reset(self, recipe_id: str | None = None) -> Any:
        if recipe_id is not None and recipe_id != self.recipe.recipe_id:
            self._build(recipe_id)
            return self.state
        with _OVERCOOKED_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
            self.env.reset(regen_mdp=False)
        self.env._mp = self.motion_planner
        self.execution_log = []
        self.environment_wait_ticks = 0
        self._park(0)
        self._park(1)
        return self.state

    def _select_stage_counter(self) -> Tuple[int, int]:
        candidates = []
        for counter in self.env.mdp.get_counter_locations():
            costs = []
            for player in self.env.state.players:
                plans = self._plans_to(counter, player.pos_and_or)
                if not plans:
                    break
                costs.append(plans[0][0])
            if len(costs) == 2:
                candidates.append((max(costs), sum(costs), tuple(counter)))
        if not candidates:
            raise OptionExecutionError("no counter is reachable by both players")
        return min(candidates)[2]

    def _plans_to(
        self, feature: Tuple[int, int], start: Any, *, avoid_players: bool = True,
    ) -> list[Tuple[int, Tuple[Any, ...], Any]]:
        plans = []
        blocked = ({
            tuple(player.position) for player in self.state.players
            if player.pos_and_or != start
        } if avoid_players else set())
        for goal in self.motion_planner.motion_goals_for_pos[tuple(feature)]:
            try:
                actions, path, cost = self.motion_planner.get_plan(start, goal)
            except (KeyError, ValueError):
                continue
            if any(tuple(step[0]) in blocked for step in path):
                continue
            plans.append((int(cost), tuple(actions), goal))
        return sorted(plans, key=lambda item: (item[0], repr(item[1])))

    def _step(self, actor: int, action: Any) -> Tuple[Any, float, bool, Any]:
        joint = [self.Action.STAY, self.Action.STAY]
        joint[int(actor)] = action
        with _OVERCOOKED_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
            return self.env.step(joint)

    def _interact(self, actor: int, feature: Tuple[int, int]) -> int:
        plans = self._plans_to(feature, self.state.players[actor].pos_and_or)
        if not plans:
            self._clear_route(actor, feature)
            plans = self._plans_to(feature, self.state.players[actor].pos_and_or)
        if not plans:
            raise OptionExecutionError(
                f"actor {actor} cannot reach feature {feature}"
            )
        start_tick = int(self.state.timestep)
        actions, goal = plans[0][1], plans[0][2]
        for action in actions:
            _state, _reward, done, _info = self._step(actor, action)
            if done:
                raise OptionExecutionError("standard Overcooked episode ended")
        if self.state.players[actor].pos_and_or != goal:
            raise OptionExecutionError("collision changed a planned interaction route")
        return int(self.state.timestep) - start_tick

    def _feature(self, terrain: str, actor: int) -> Tuple[int, int]:
        locations = tuple(self.env.mdp.terrain_pos_dict.get(terrain, ()))
        if not locations:
            raise OptionExecutionError(f"layout has no {terrain!r} station")
        reachable = []
        for location in locations:
            plans = self._plans_to(
                tuple(location), self.state.players[actor].pos_and_or,
                avoid_players=False,
            )
            if plans:
                reachable.append((plans[0][0], tuple(location)))
        if not reachable:
            raise OptionExecutionError(f"no reachable {terrain!r} station")
        return min(reachable)[1]

    def _pot_object(self) -> Any | None:
        for position in self.env.mdp.get_pot_locations():
            if self.state.has_object(position):
                return self.state.get_object(position)
        return None

    def legal_actions(
        self, candidates: Sequence[str], *, actor_id: int = 0,
    ) -> Tuple[str, ...]:
        if _held_name(self.state, actor_id) is not None:
            return ()
        # ``pot.is_ready`` resolves cook time through the global ``Recipe``
        # class, which every executor reconfigures on each step.  legal_actions
        # is called by the runner outside _step, so pin the configuration here
        # too rather than relying on whichever executor stepped last.
        with _OVERCOOKED_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
            return self._legal_actions(candidates, actor_id=actor_id)

    def _legal_actions(
        self, candidates: Sequence[str], *, actor_id: int = 0,
    ) -> Tuple[str, ...]:
        pot = self._pot_object()
        ingredients = () if pot is None else tuple(getattr(pot, "ingredients", ()))
        stage = (
            self.state.get_object(self.stage_counter)
            if self.state.has_object(self.stage_counter) else None
        )
        legal = []
        for candidate in candidates:
            action = str(candidate).upper()
            if action.startswith("ADD_ONION_"):
                allowed = ingredients.count("onion") < self.recipe.ingredients.count("onion")
            elif action.startswith("ADD_TOMATO_"):
                allowed = ingredients.count("tomato") < self.recipe.ingredients.count("tomato")
            elif action == "START_COOKING_SOUP":
                allowed = (
                    pot is not None
                    and len(ingredients) == len(self.recipe.ingredients)
                    and bool(getattr(pot, "is_idle", False))
                )
            elif action == "STAGE_DISH":
                allowed = stage is None
            elif action == "PLATE_SOUP":
                allowed = (
                    stage is not None and stage.name == "dish"
                    and pot is not None and bool(getattr(pot, "is_ready", False))
                )
            elif action == "SERVE_SOUP":
                allowed = stage is not None and stage.name == "soup"
            else:
                allowed = False
            if allowed:
                legal.append(action)
        return tuple(dict.fromkeys(legal))

    def execute(self, action: str, *, actor_id: int) -> OptionExecution:
        actor = int(actor_id)
        label = str(action).upper()
        if label not in self.legal_actions((label,), actor_id=actor):
            raise OptionExecutionError(f"option {label} is not physically legal")
        before_tick = int(self.state.timestep)
        before_deliveries = self._delivery_count()
        before_reward = float(np.asarray(
            self.env.game_stats["cumulative_sparse_rewards_by_agent"]
        ).sum())
        calls: list[str] = []
        ticks: list[int] = []
        with self._deterministic_rng(label, actor):
            if label.startswith("ADD_"):
                ingredient = "onion" if "ONION" in label else "tomato"
                terrain = "O" if ingredient == "onion" else "T"
                ticks.append(self._interact(actor, self._feature(terrain, actor)))
                calls.append(f"PICKUP_{ingredient.upper()}")
                ticks.append(self._interact(actor, self._feature("P", actor)))
                calls.append(f"POT_{ingredient.upper()}")
            elif label == "START_COOKING_SOUP":
                # Starting the pot is a decision, not a side effect of the last
                # ingredient.  Bundling it into ADD_* hid the only late-episode
                # ordering choice standard Overcooked offers, which is what made
                # "container just in time" unobservable.
                ticks.append(self._interact(actor, self._feature("P", actor)))
                calls.append("START_COOKING_SOUP")
                pot = self._pot_object()
                if pot is None or bool(getattr(pot, "is_idle", False)):
                    raise OptionExecutionError("pot did not begin cooking")
            elif label == "STAGE_DISH":
                ticks.append(self._interact(actor, self._feature("D", actor)))
                calls.append("PICKUP_DISH")
                ticks.append(self._interact(actor, self.stage_counter))
                calls.append("STAGE_DISH")
            elif label == "PLATE_SOUP":
                ticks.append(self._interact(actor, self.stage_counter))
                calls.append("PICKUP_STAGED_DISH")
                ticks.append(self._interact(actor, self._feature("P", actor)))
                calls.append("PLATE_SOUP")
                ticks.append(self._interact(actor, self.stage_counter))
                calls.append("STAGE_PLATED_SOUP")
            elif label == "SERVE_SOUP":
                ticks.append(self._interact(actor, self.stage_counter))
                calls.append("PICKUP_STAGED_SOUP")
                ticks.append(self._interact(actor, self._feature("S", actor)))
                calls.append("DELIVER_SOUP")
            if _held_name(self.state, actor) is not None:
                raise OptionExecutionError(f"option {label} did not end empty-handed")
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
                raise OptionExecutionError("episode ended while soup cooked")
        self.environment_wait_ticks += count
        return count

    def _walkable(self) -> set[Tuple[int, int]]:
        return set(map(tuple, self.env.mdp.get_valid_player_positions()))

    def _move_player(self, actor: int, goal: Tuple[int, int]) -> None:
        start = tuple(self.state.players[actor].position)
        if start == goal:
            return
        blocked = {tuple(self.state.players[1 - actor].position)}
        queue = deque([start])
        parent = {start: None}
        while queue and goal not in parent:
            current = queue.popleft()
            for delta in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nxt = (current[0] + delta[0], current[1] + delta[1])
                if nxt in parent or nxt in blocked or nxt not in self._walkable():
                    continue
                parent[nxt] = current
                queue.append(nxt)
                if nxt == goal:
                    break
        if goal not in parent:
            raise OptionExecutionError(f"actor {actor} cannot reach relocation cell")
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        for current, nxt in zip(path, path[1:]):
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            _state, _reward, done, _info = self._step(actor, direction)
            if done or tuple(self.state.players[actor].position) != nxt:
                raise OptionExecutionError(f"actor {actor} failed to relocate")

    def _clear_route(self, actor: int, feature: Tuple[int, int]) -> None:
        all_plans = self._plans_to(
            feature, self.state.players[actor].pos_and_or, avoid_players=False,
        )
        if not all_plans:
            return
        other = 1 - actor
        occupied_by_route = set()
        position = tuple(self.state.players[actor].position)
        for action in all_plans[0][1]:
            if action in self.Action.MOTION_ACTIONS and action != self.Action.STAY:
                position = (position[0] + action[0], position[1] + action[1])
            occupied_by_route.add(position)
        candidates = sorted(
            self._walkable() - occupied_by_route - {
                tuple(self.state.players[actor].position)
            },
            key=lambda cell: (
                -abs(cell[0] - feature[0]) - abs(cell[1] - feature[1]), cell
            ),
        )
        for candidate in candidates:
            try:
                self._move_player(other, candidate)
                return
            except OptionExecutionError:
                continue
        raise OptionExecutionError("could not relocate passive player off route")

    def _park(self, actor: int) -> None:
        self._move_player(actor, self.parking_positions[actor])

    def _select_parking_positions(self) -> Dict[int, Tuple[int, int]]:
        """Two mutually reachable floor cells that are not station approaches.

        Validated against the layout rather than hardcoded, so a layout change
        fails here with a clear message instead of surfacing later as an
        opaque "cannot reach relocation cell" mid-episode.
        """
        walkable = sorted(self._walkable())
        if len(walkable) < 2:
            raise OptionExecutionError(
                f"layout {self.recipe.layout!r} has fewer than two floor cells"
            )
        stations = {
            tuple(position)
            for terrain in ("P", "D", "S", "O", "T")
            for position in self.env.mdp.terrain_pos_dict.get(terrain, ())
        }
        def approach_load(cell: Tuple[int, int]) -> int:
            return sum(
                (cell[0] + dx, cell[1] + dy) in stations
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

    def _delivery_count(self) -> int:
        return int(sum(
            len(events)
            for events in self.env.game_stats.get("soup_delivery", ())
        ))


__all__ = ["OvercookedOptionExecutor"]
