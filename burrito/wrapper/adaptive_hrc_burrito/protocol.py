"""Adaptive-HRC observation/assist protocol in the physical Burrito domain."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .domain import BurritoDomainAdapter, StateVector
from .macros import macro_actions, protein_name
from .options import BurritoOptionExecutor
from .task_graph import (
    BurritoPreferencePolicy,
    BurritoTaskGraph,
    choose_acceptable_action,
    preference_name,
)


OBSERVE = "observe"
ASSIST = "assist"


@dataclass(frozen=True)
class BurritoTask:
    """One recipe/preference episode; its action order is state-generated."""

    recipe_id: str
    protein: str
    preference: str

    @classmethod
    def create(
        cls, protein: str, preference: str = "plate_early",
        *, recipe_id: str | None = None,
    ) -> "BurritoTask":
        normalized = protein_name(protein)
        return cls(
            recipe_id=recipe_id or f"{normalized}_burrito",
            protein=normalized,
            preference=preference_name(preference),
        )

    @property
    def action_space(self) -> Tuple[str, ...]:
        return macro_actions(self.protein)


@dataclass(frozen=True)
class BurritoObservation:
    """Duck-typed equivalent of src.representations.Observation."""

    state: StateVector
    action: str
    next_state: StateVector


@dataclass(frozen=True)
class BurritoDecision:
    recipe_step: int
    mode: str
    scheduled_actor: str
    physical_actor_id: int
    executed_by: str
    legal_actions: Tuple[str, ...]
    acceptable_actions: Tuple[str, ...]
    reference_action: str
    actual: str
    predicted: Optional[str]
    correct_top_1: bool
    exact_reference_match: bool
    reference_probability: float
    reference_nll: float
    acceptable_probability_mass: float
    acceptable_nll: float
    invalid_prediction: bool
    prediction_wall_s: float
    prediction_stats: Mapping[str, Any]
    distribution: Mapping[str, float]
    human_corrected: bool
    proposal_executed: bool
    low_level_ticks: int
    passive_wait_ticks_before: int
    primitive_calls: Tuple[str, ...]
    primitive_ticks: Tuple[int, ...]


@dataclass(frozen=True)
class BurritoEpisodeResult:
    task: BurritoTask
    mode: str
    decisions: Tuple[BurritoDecision, ...]
    observations: Tuple[BurritoObservation, ...]
    match: Any
    robot_turns: int
    robot_top_1_hits: int
    robot_exact_reference_hits: int
    human_shadow_top_1_hits: int
    human_shadow_exact_reference_hits: int
    corrections: int
    human_actions: int
    robot_actions: int
    invalid_predictions: int
    prediction_wall_s: float
    low_level_ticks: int
    passive_wait_ticks: int
    sparse_reward: float
    deliveries: int
    memory_age_delta: int

    @property
    def robot_top_1(self) -> float:
        """Acceptable-set top-1 accuracy on scheduled robot turns."""
        return (
            self.robot_top_1_hits / self.robot_turns
            if self.robot_turns else 0.0
        )

    @property
    def robot_exact_reference_top_1(self) -> float:
        """Diagnostic match to the sampled human fallback, not correctness."""
        return (
            self.robot_exact_reference_hits / self.robot_turns
            if self.robot_turns else 0.0
        )

    @property
    def human_shadow_top_1(self) -> float:
        human_turns = sum(
            decision.scheduled_actor == "human"
            for decision in self.decisions
            if decision.mode == ASSIST
        )
        return (
            self.human_shadow_top_1_hits / human_turns
            if human_turns else 0.0
        )

    @property
    def preference_sequence(self) -> Tuple[str, ...]:
        return tuple(decision.actual for decision in self.decisions)


class BurritoHrcRunner:
    """Run dynamic ground truth with the existing learner and memory clock.

    A wrong robot proposal is scored but never applied. The human performs a
    sampled acceptable action and the next decision remains a robot turn.
    Native movement and cooking frames never enter the preference sequence.
    """

    def __init__(
        self,
        agent: Any,
        executor: BurritoOptionExecutor,
        domain: BurritoDomainAdapter,
        *,
        seed: int | None = None,
        max_passive_wait_ticks: int = 400,
    ):
        if getattr(agent, "domain", None) is not domain:
            raise ValueError(
                "agent and BurritoHrcRunner must share one domain adapter"
            )
        self.agent = agent
        self.executor = executor
        self.domain = domain
        self.max_passive_wait_ticks = max(1, int(max_passive_wait_ticks))
        resolved_seed = (
            int(seed) if seed is not None
            else int(getattr(getattr(agent, "settings", None), "seed", 0))
        )
        self._ground_truth_rng = np.random.default_rng(
            np.random.SeedSequence((resolved_seed, 0x42555252))
        )
        self._observed_recipes: set[str] = set()

    @property
    def observed_recipes(self) -> Tuple[str, ...]:
        return tuple(sorted(self._observed_recipes))

    def run_task(
        self, task: BurritoTask, *, force_mode: str | None = None,
    ) -> BurritoEpisodeResult:
        graph = BurritoTaskGraph.create(task.protein)
        policy = BurritoPreferencePolicy.create(task.preference)
        natural_mode = (
            OBSERVE if task.recipe_id not in self._observed_recipes else ASSIST
        )
        mode = natural_mode if force_mode is None else str(force_mode)
        if mode not in {OBSERVE, ASSIST}:
            raise ValueError("mode must be 'observe' or 'assist'")
        if force_mode == OBSERVE and task.recipe_id in self._observed_recipes:
            raise ValueError(
                "normal evaluation forbids a second observation of a recipe"
            )
        if mode == ASSIST and task.recipe_id not in self._observed_recipes:
            raise ValueError(
                "a recipe must receive exactly one observation before assist"
            )

        self.executor.reset()
        self.domain.bind_initial_state(self.executor.state, actor_id=0)
        if mode == OBSERVE:
            self.agent.start_demo()

        decisions: list[BurritoDecision] = []
        observations: list[BurritoObservation] = []
        completed: list[str] = []
        robot_turn_next = False
        robot_turns = robot_hits = robot_reference_hits = 0
        human_shadow_hits = human_shadow_reference_hits = corrections = 0
        human_actions = robot_actions = macro_ticks = passive_wait_ticks = 0
        invalid_predictions = 0
        prediction_wall_s = 0.0
        reward = 0.0
        deliveries_before = self._delivery_count()
        demo_counter_before = int(self.agent.demo_counter)

        while not graph.is_complete(completed):
            if len(completed) >= len(graph.actions):
                raise RuntimeError("task graph exhausted without a delivery")
            scheduled_actor = (
                "human" if mode == OBSERVE or not robot_turn_next else "robot"
            )
            decision_actor_id = 0 if scheduled_actor == "human" else 1
            waited = 0
            while True:
                state = self.domain.state_key(
                    self.executor.state, actor_id=decision_actor_id,
                )
                acceptable = policy.acceptable_actions(
                    graph, state, completed, self.domain,
                )
                if acceptable:
                    legal = graph.available_actions(
                        state, completed, self.domain,
                    )
                    break
                if waited >= self.max_passive_wait_ticks:
                    raise RuntimeError(
                        "no acceptable task-graph action became physically "
                        "legal before the passive-wait limit"
                    )
                self.executor.advance_environment(1)
                waited += 1
            passive_wait_ticks += waited
            reference = choose_acceptable_action(
                acceptable, self._ground_truth_rng,
            )

            prefix = tuple(
                self.agent.pending_demo
                if mode == OBSERVE else self.agent.current_prefix
            )
            distribution: Dict[str, float] = {}
            predicted: Optional[str] = None
            acceptable_hit = False
            reference_hit = False
            acceptable_mass = 0.0
            acceptable_nll = math.nan
            reference_probability = 0.0
            reference_nll = math.nan
            prediction_elapsed = 0.0
            invalid_prediction = False
            prediction_stats: Dict[str, Any] = {}
            if mode == ASSIST:
                prediction_started = time.perf_counter()
                distribution = dict(self.agent.predict_actions(
                    prefix,
                    state=self.executor.state,
                    actor_id=decision_actor_id,
                    # The recipe DAG is the task-level feasibility mask;
                    # physical preconditions alone would permit starting a
                    # second burrito after assembly.
                    action_universe=legal,
                ))
                ranked = self.agent.rank_actions(distribution, k=1)
                prediction_elapsed = time.perf_counter() - prediction_started
                prediction_wall_s += prediction_elapsed
                predicted = ranked[0] if ranked else None
                prediction_stats = dict(self.agent.policy_stats())
                invalid_prediction = bool(
                    predicted is not None and predicted not in legal
                )
                invalid_predictions += int(invalid_prediction)
                acceptable_hit = predicted in acceptable
                reference_hit = predicted == reference
                reference_probability = max(
                    0.0, float(distribution.get(reference, 0.0)),
                )
                reference_nll = -math.log(max(reference_probability, 1e-12))
                acceptable_mass = float(sum(
                    max(0.0, float(distribution.get(action, 0.0)))
                    for action in acceptable
                ))
                acceptable_nll = -math.log(max(acceptable_mass, 1e-12))

            corrected = False
            if scheduled_actor == "robot":
                robot_turns += 1
                robot_hits += int(acceptable_hit)
                robot_reference_hits += int(reference_hit)
                if acceptable_hit and predicted is not None:
                    executed_action = predicted
                    physical_actor = 1
                    executed_by = "robot"
                    robot_actions += 1
                    robot_turn_next = False
                else:
                    executed_action = reference
                    physical_actor = 0
                    executed_by = "human_correction"
                    corrected = True
                    corrections += 1
                    human_actions += 1
                    robot_turn_next = True
            else:
                executed_action = reference
                physical_actor = 0
                executed_by = "human"
                human_actions += 1
                if mode == ASSIST:
                    human_shadow_hits += int(acceptable_hit)
                    human_shadow_reference_hits += int(reference_hit)
                    robot_turn_next = True

            before = self.domain.state_key(
                self.executor.state, actor_id=decision_actor_id,
            )
            execution = self.executor.execute(
                executed_action, actor_id=physical_actor,
            )
            macro_ticks += execution.low_level_ticks
            reward += execution.sparse_reward
            completed.append(executed_action)
            next_actor_id = (
                0 if mode == OBSERVE or not robot_turn_next else 1
            )
            after = self.domain.state_key(
                self.executor.state, actor_id=next_actor_id,
            )
            self.domain.record_transition(before, executed_action, after)
            observation = BurritoObservation(before, executed_action, after)
            observations.append(observation)
            self.agent.observe(
                observation,
                ground_truth_recipe=task.recipe_id,
                precomputed_distribution=(
                    distribution if mode == ASSIST else None
                ),
                precomputed_prediction=(
                    predicted if mode == ASSIST else None
                ),
            )
            decisions.append(BurritoDecision(
                recipe_step=len(completed) - 1,
                mode=mode,
                scheduled_actor=scheduled_actor,
                physical_actor_id=physical_actor,
                executed_by=executed_by,
                legal_actions=tuple(legal),
                acceptable_actions=tuple(acceptable),
                reference_action=reference,
                actual=executed_action,
                predicted=predicted,
                correct_top_1=bool(acceptable_hit),
                exact_reference_match=bool(reference_hit),
                reference_probability=reference_probability,
                reference_nll=reference_nll,
                acceptable_probability_mass=acceptable_mass,
                acceptable_nll=acceptable_nll,
                invalid_prediction=invalid_prediction,
                prediction_wall_s=prediction_elapsed,
                prediction_stats=prediction_stats,
                distribution=distribution,
                human_corrected=corrected,
                proposal_executed=bool(
                    scheduled_actor != "robot" or acceptable_hit
                ),
                low_level_ticks=execution.low_level_ticks,
                passive_wait_ticks_before=waited,
                primitive_calls=execution.primitive_calls,
                primitive_ticks=execution.primitive_ticks,
            ))

        match = self.agent.end_demo()
        memory_age_delta = int(self.agent.demo_counter) - demo_counter_before
        if memory_age_delta != 1:
            raise RuntimeError(
                "one completed Burrito recipe must age memory exactly once; "
                f"observed delta {memory_age_delta}"
            )
        if mode == OBSERVE:
            self._observed_recipes.add(task.recipe_id)
        deliveries = self._delivery_count() - deliveries_before
        if deliveries != 1:
            raise RuntimeError(
                f"Burrito task should deliver exactly one dish, got {deliveries}"
            )
        return BurritoEpisodeResult(
            task=task,
            mode=mode,
            decisions=tuple(decisions),
            observations=tuple(observations),
            match=match,
            robot_turns=robot_turns,
            robot_top_1_hits=robot_hits,
            robot_exact_reference_hits=robot_reference_hits,
            human_shadow_top_1_hits=human_shadow_hits,
            human_shadow_exact_reference_hits=human_shadow_reference_hits,
            corrections=corrections,
            human_actions=human_actions,
            robot_actions=robot_actions,
            invalid_predictions=invalid_predictions,
            prediction_wall_s=prediction_wall_s,
            low_level_ticks=macro_ticks + passive_wait_ticks,
            passive_wait_ticks=passive_wait_ticks,
            sparse_reward=reward,
            deliveries=deliveries,
            memory_age_delta=memory_age_delta,
        )

    def run_stream(
        self, tasks: Sequence[BurritoTask],
    ) -> Tuple[BurritoEpisodeResult, ...]:
        return tuple(self.run_task(task) for task in tasks)

    def _delivery_count(self) -> int:
        return int(sum(
            len(events)
            for events in self.executor.env.game_stats.get(
                "dish_delivery", (),
            )
        ))


__all__ = [
    "ASSIST",
    "OBSERVE",
    "BurritoDecision",
    "BurritoEpisodeResult",
    "BurritoHrcRunner",
    "BurritoObservation",
    "BurritoTask",
]
