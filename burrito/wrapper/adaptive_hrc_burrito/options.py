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
import re
import threading
from typing import Any, Dict, FrozenSet, Iterable, Iterator, Mapping, Sequence, Tuple
import zlib

import numpy as np

from .catalog import get_recipe
from .runtime import BurritoRuntime
from .macros import (
    PLATE_COMPONENTS,
    PLATE_RICE,
    PLATE_TORTILLA,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    action_protein,
    fetch_and_stage_action,
    macro_actions,
    plate_protein_action,
    prepare_and_stage_action,
    protein_name,
    serve_action,
    start_cooking_action,
)


_UPSTREAM_STATE_LOCK = threading.RLock()

CONTROLLED_PARKING_POSITIONS: Mapping[
    str, Mapping[int, Tuple[int, int]]
] = {
    # These handoff cells are validated across both actors and native Burrito
    # recipes. They avoid single-access station tiles.
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


COOKED_PROTEIN = {"steak": "chopped_steak", "mushroom": "fried_mushroom"}
CHOPPED_PROTEIN = {"steak": "chopped_meat", "mushroom": "chopped_mushroom"}
RAW_PROTEIN = {"steak": "meat", "mushroom": "mushroom"}
_GRAB_PLATE_BY_COMPONENT_COUNT = {
    0: "GRAB_CLEAN_PLATE",
    1: "GRAB_ONE_INGREDIENT_PLATE",
    2: "GRAB_TWO_INTREDIENT_PLATE",
}


def plate_components(name: str | None) -> FrozenSet[str] | None:
    """Decompose an upstream plate object name into burrito components.

    Returns ``None`` when the name is not a plate the recipe can build on.
    The pinned environment encodes plate contents in the object name, e.g.
    ``clean_plate`` -> {}, ``boiled_rice-tortilla-plate`` -> {rice, tortilla},
    ``steak_burrito`` -> every component.

    Parsing is deliberately protein-agnostic.  Resolving contents *against* a
    protein is ambiguous -- ``tortilla-plate`` parses validly under either one
    -- which previously let a mushroom episode be read as a steak episode.
    """
    if name is None:
        return None
    label = str(name)
    if label in {f"{protein}_burrito" for protein in COOKED_PROTEIN}:
        return frozenset(PLATE_COMPONENTS)
    if label == "clean_plate":
        return frozenset()
    if not label.endswith("-plate"):
        return None
    parts = set(label.split("-"))
    components = set()
    if "boiled_rice" in parts:
        components.add("rice")
    if "tortilla" in parts:
        components.add("tortilla")
    if parts & set(COOKED_PROTEIN.values()):
        components.add("protein")
    return frozenset(components)


def _staged_plate(state: Any) -> Tuple[Any, FrozenSet[str]] | None:
    """The single plate the recipe is currently building.

    Only one plate is ever in play: the pinned sink does not yield a second
    addressable ``clean_plate``, and the upstream grab primitives select their
    target by object name, so two same-named plates could not be told apart.
    Multi-order recipes therefore serialise assembly and parallelise the
    protein and rice branches instead.
    """
    for obj in state.objects.values():
        components = plate_components(getattr(obj, "name", None))
        if components is not None:
            return obj, components
    return None


def _held_components(state: Any, actor_id: int) -> FrozenSet[str] | None:
    return plate_components(_held_name(state, actor_id))


ORDINAL_SUFFIX = re.compile(r"_(\d+)$")


def base_token(action: str) -> str:
    """Strip the ordinal that distinguishes repeats of one physical option.

    A two-order recipe needs two ``STAGE_CLEAN_PLATE`` decisions that the task
    graph must keep apart but the executor performs identically.
    """
    return ORDINAL_SUFFIX.sub("", str(action).upper())


def _component_primitive(component: str, protein: str | None) -> str:
    if component == "rice":
        return "GET_BOILED_RICE"
    if component == "tortilla":
        return "GRAB_TORTILLA"
    return (
        "GET_STEAK_FROM_GRILLER" if protein == "steak"
        else "GET_MURHSOOM_FROM_GRILLER"
    )


def _has_ready(state: Any, name: str) -> bool:
    return any(_is_ready(obj) for obj in _objects(state, name))


def seed_upstream_rng(
    seed: int, action: str, actor_id: int, attempt: int = 0,
) -> None:
    """Seed global ``np.random`` for one planner invocation.

    The pinned Burrito planner samples from the global NumPy stream, so the
    route a macro takes is only reproducible if the stream is pinned per
    (option, actor, attempt).  Every executor uses this one implementation.
    """
    child = np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        zlib.crc32(str(action).encode("utf-8")) & 0xFFFFFFFF,
        int(actor_id) & 0xFFFFFFFF,
        int(attempt) & 0xFFFFFFFF,
    ])
    np.random.seed(int(child.generate_state(1, dtype=np.uint32)[0]))


def macro_is_legal(
    state: Any,
    action: str,
    actor_id: int = 0,
    *,
    pot_capacity: int = 1,
) -> bool:
    """Preference-neutral macro preconditions at a handoff boundary.

    These are *physical* possibility checks only.  Single-firing of each task
    option is guaranteed by the task graph, which the runner intersects with
    this mask; encoding it here as well previously made ``FETCH_AND_STAGE_X``
    illegal once the *other* protein had been plated.
    """
    label = base_token(action)
    if _held_name(state, actor_id) is not None:
        return False
    staged = _staged_plate(state)
    components = frozenset() if staged is None else staged[1]

    if label == STAGE_CLEAN_PLATE:
        return staged is None and _has_object(state, "dirty_plate")
    if label == START_BOILING_RICE:
        return len(_objects(state, "boiled_rice")) < max(1, int(pot_capacity))
    if label == PLATE_RICE:
        return (
            staged is not None
            and "rice" not in components
            and _has_ready(state, "boiled_rice")
        )
    if label == PLATE_TORTILLA:
        return staged is not None and "tortilla" not in components

    protein = action_protein(label)
    if protein is None:
        return False
    chopped = CHOPPED_PROTEIN[protein]
    cooked = COOKED_PROTEIN[protein]
    if label == fetch_and_stage_action(protein):
        return not _has_object(state, chopped) and not _has_object(state, cooked)
    if label == prepare_and_stage_action(protein):
        return bool(_objects(state, chopped)) and not _has_ready(state, chopped)
    if label == start_cooking_action(protein):
        return _has_ready(state, chopped) and not _has_object(state, cooked)
    if label == plate_protein_action(protein):
        return (
            staged is not None
            and "protein" not in components
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
        recipe_id: str = "burrito_steak_burrito",
        *,
        horizon: int = 1200,
        seed: int = 0,
        parking_positions: Mapping[int, Tuple[int, int]] | None = None,
    ):
        recipe = get_recipe(recipe_id)
        if recipe.environment != "burrito" or recipe.compatibility_dynamics:
            raise ValueError(f"{recipe_id!r} is not a native Burrito recipe")
        proteins = tuple(dict.fromkeys(
            action_protein(token) for token in recipe.action_tokens
            if action_protein(token) is not None
        ))
        if not proteins:
            raise ValueError(f"{recipe_id!r} names no burrito protein")
        self.recipe = recipe
        self.proteins = proteins
        self.protein = proteins[0]
        self.runtime = runtime
        self.layout = str(recipe.layout)
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
        from burrito.mdp.burrito_mdp import Burrito_Recipe
        from overcooked_ai_py.mdp.actions import Action

        self._recipe_type = Burrito_Recipe
        self._low_level_action = Action
        self._low_level_stay = int(Action.ACTION_TO_INDEX[Action.STAY])
        self.execution_log: list[OptionExecution] = []
        self.pot_capacity = max(
            1, len(self.env.mdp.terrain_pos_dict.get("P", ()) or ())
        )
        self.parking_positions = dict(
            parking_positions or self._default_parking_positions()
        )
        self.reset()

    @property
    def state(self) -> Any:
        return self.env.state

    @property
    def compatibility_dynamics(self) -> bool:
        return False

    def reset(self, recipe_id: str | None = None) -> Any:
        if recipe_id is not None and recipe_id != self.recipe.recipe_id:
            raise ValueError(
                f"this executor is bound to {self.recipe.recipe_id!r}, "
                f"not {recipe_id!r}"
            )
        with _UPSTREAM_STATE_LOCK:
            self._recipe_type.configure(self.env.mdp.recipe_config)
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
        self._current_primitive_ticks: list[int] = []
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
            self._recipe_type.configure(self.env.mdp.recipe_config)
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
            if macro_is_legal(
                self.state, str(action), actor_id=actor_id,
                pot_capacity=self.pot_capacity,
            )
        ))

    def execute(self, action: str, *, actor_id: int) -> OptionExecution:
        actor = int(actor_id)
        if actor not in (0, 1):
            raise ValueError("actor_id must be 0 or 1")
        # Dispatch on the ordinal-stripped token: a two-order recipe repeats
        # one physical option under distinct task-graph tokens.  The execution
        # log keeps the original token so it lines up with the task graph.
        token = str(action).upper()
        label = base_token(token)
        if not macro_is_legal(
            self.state, label, actor_id=actor, pot_capacity=self.pot_capacity,
        ):
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
        with self._deterministic_upstream_rng(label, actor):
            if label == STAGE_CLEAN_PLATE:
                self._stage_clean_plate(actor, primitives)
            elif label == START_BOILING_RICE:
                self._start_rice(actor, primitives)
            elif label == PLATE_RICE:
                self._add_plate_component(actor, None, "rice", primitives)
            elif label == PLATE_TORTILLA:
                self._add_plate_component(actor, None, "tortilla", primitives)
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
                elif label == plate_protein_action(protein):
                    self._add_plate_component(
                        actor, protein, "protein", primitives,
                    )
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
            action=token,
            actor_id=actor,
            low_level_ticks=int(self.state.timestep) - before_tick,
            primitive_calls=tuple(primitives),
            primitive_ticks=tuple(self._current_primitive_ticks),
            sparse_reward=float(after_reward - before_reward),
            deliveries_before=before_deliveries,
            deliveries_after=self._delivery_count(),
        )
        self.execution_log.append(execution)
        return execution

    @contextmanager
    def _deterministic_upstream_rng(
        self, action: str, actor_id: int,
    ) -> Iterator[None]:
        state = np.random.get_state()
        seed_upstream_rng(self.seed, action, actor_id, 0)
        try:
            yield
        finally:
            np.random.set_state(state)

    def _reseed_attempt(self, primitive: str, actor_id: int, attempt: int) -> None:
        """Give each retry its own reproducible planner stream.

        The upstream planner draws from global ``np.random`` (it shuffles
        ``high_level_action_priority`` and samples interaction counts), so
        replaying an attempt without advancing the stream reproduces the same
        route and the same failure.  Retries were therefore identical replays
        and ``retries`` bought nothing; mixing the attempt index in keeps the
        run reproducible while making a retry actually explore a new route.
        """
        seed_upstream_rng(self.seed, primitive, actor_id, attempt)

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
        for attempt in range(max(1, int(retries))):
            self._reseed_attempt(primitive, actor, attempt)
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
        for attempt in range(max(1, int(retries))):
            self._reseed_attempt(primitive, actor, attempt)
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

    def _grab_staged_plate(self, actor: int, primitives: list[str]) -> None:
        staged = _staged_plate(self.state)
        if staged is None:
            raise OptionExecutionError("no plate is staged for assembly")
        obj, components = staged
        target = str(obj.name)
        primitive = _GRAB_PLATE_BY_COMPONENT_COUNT.get(len(components))
        if primitive is None:
            raise OptionExecutionError(f"{target} is already a finished burrito")
        self._repeat_until(
            actor, primitive,
            lambda: _held_name(self.state, actor) == target,
            primitives,
        )

    def _add_plate_component(
        self, actor: int, protein: str | None, component: str,
        primitives: list[str],
    ) -> None:
        """Add exactly one component to the staged plate, then re-stage it.

        The pinned environment accepts rice, tortilla and the cooked protein in
        any order, so each addition is an independently schedulable option.
        Re-staging keeps the handoff invariant: every option ends with the
        acting player empty-handed.
        """
        self._grab_staged_plate(actor, primitives)
        self._repeat_until(
            actor, _component_primitive(component, protein),
            lambda: component in (
                _held_components(self.state, actor) or frozenset()
            ),
            primitives,
        )
        assembled = _held_name(self.state, actor)
        if assembled is None:
            raise OptionExecutionError(
                f"adding {component} consumed the staged plate"
            )
        self._repeat_until(
            actor, "PUT_DOWN_OBJECT",
            lambda: (
                _held_name(self.state, actor) is None
                and _has_object(self.state, assembled)
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
    "CONTROLLED_PARKING_POSITIONS",
    "COOKED_PROTEIN",
    "OptionExecution",
    "OptionExecutionError",
    "PLATE_COMPONENTS",
    "PLATE_RICE",
    "PLATE_TORTILLA",
    "STAGE_CLEAN_PLATE",
    "START_BOILING_RICE",
    "action_protein",
    "fetch_and_stage_action",
    "macro_actions",
    "macro_is_legal",
    "plate_components",
    "plate_protein_action",
    "prepare_and_stage_action",
    "serve_action",
    "start_cooking_action",
]
