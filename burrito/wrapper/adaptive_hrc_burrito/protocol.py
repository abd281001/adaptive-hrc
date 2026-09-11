"""Natural-stream human/robot protocol for Overcooked and Burrito."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .catalog import get_preference, get_recipe
from .domain import CookingDomainAdapter, StateVector
from .physical import create_executor
from .runtime import BurritoRuntime
from .task_graph import (
    CookingPreferencePolicy,
    CookingTaskGraph,
    is_preference_discriminating,
    is_prefix_conditioned_discriminating,
)


OBSERVE = "observe"
ASSIST = "assist"

# Which actor takes the opening move of an assist episode.  ``human_first`` is
# the Adaptive-HRC protocol and stays the default.  It has a consequence worth
# naming: the opening move is never a robot turn, and in the shallower cooking
# task graphs it is sometimes the only preference-discriminating decision, so
# those cells are scored for prediction but never for assistance.
# ``counterbalanced`` alternates the lead across a recipe's assist exposures so
# the opening move is also exercised as a robot decision.
HUMAN_FIRST = "human_first"
COUNTERBALANCED = "counterbalanced"
LEAD_ACTOR_POLICIES = (HUMAN_FIRST, COUNTERBALANCED)


@dataclass(frozen=True)
class CookingTask:
    """One naturally scheduled recipe exposure in a ladder stream."""

    recipe_id: str
    preference: str
    scenario: str
    phase: int
    schedule_step: int
    phase_role: str
    lifecycle: str
    preference_changed: bool
    exposure_after_change: int
    strategy: str
    adaptation_id: Optional[str]
    holdout_target: bool

    @classmethod
    def create(
        cls,
        recipe_id: str,
        preference: str,
        *,
        scenario: str = "manual",
        phase: int = 0,
        schedule_step: int = 0,
        phase_role: str = "ordinary",
        lifecycle: str = "retain",
        preference_changed: bool = False,
        exposure_after_change: int = 0,
        strategy: str = "manual",
        adaptation_id: Optional[str] = None,
        holdout_target: bool = False,
    ) -> "CookingTask":
        recipe = get_recipe(recipe_id)
        selected = get_preference(preference)
        return cls(
            recipe.recipe_id,
            selected.name,
            str(scenario),
            int(phase),
            int(schedule_step),
            str(phase_role),
            str(lifecycle),
            bool(preference_changed),
            max(0, int(exposure_after_change)),
            str(strategy),
            None if adaptation_id is None else str(adaptation_id),
            bool(holdout_target),
        )

    @property
    def action_space(self) -> Tuple[str, ...]:
        return get_recipe(self.recipe_id).action_tokens


@dataclass(frozen=True)
class CookingObservation:
    state: StateVector
    action: str
    next_state: StateVector


@dataclass(frozen=True)
class CookingDecision:
    recipe_step: int
    mode: str
    scheduled_actor: str
    physical_actor_id: int
    executed_by: str
    legal_actions: Tuple[str, ...]
    ground_truth_action: str
    actual: str
    predicted: Optional[str]
    correct_top_1: bool
    correct_top_k: bool
    ground_truth_probability: float
    ground_truth_nll: float
    preference_discriminating: bool
    prefix_conditioned_discriminating: bool
    prediction_available: bool
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
class CookingEpisodeResult:
    task: CookingTask
    mode: str
    decisions: Tuple[CookingDecision, ...]
    observations: Tuple[CookingObservation, ...]
    match: Any
    robot_turns: int
    robot_top_1_hits: int
    robot_top_k_hits: int
    corrections: int
    human_actions: int
    robot_actions: int
    invalid_predictions: int
    # Teacher-forced totals span every scored decision -- robot turns, human
    # turns and the opening move -- so a preference whose only discriminating
    # decision is step 0 is still measured.
    scored_turns: int
    scored_top_1_hits: int
    scored_top_k_hits: int
    scored_discriminating_decisions: int
    scored_discriminating_top_1_hits: int
    # Prediction availability is reported, never used to trim a denominator.
    scored_available_decisions: int
    robot_available_decisions: int
    prediction_wall_s: float
    low_level_ticks: int
    passive_wait_ticks: int
    sparse_reward: float
    deliveries: int
    memory_age_delta: int
    commit_applied: bool
    active_rehearsal: bool
    retrain_executed: bool
    retrain_correctly_skipped: bool
    retrain_events: Tuple[Mapping[str, Any], ...]
    compatibility_dynamics: bool
    compatibility_calls: Tuple[str, ...]
    lead_actor_policy: str
    # The floor actually applied to this episode's losses.  Reading it back off
    # the agent's settings ignores an explicit runner override and records a
    # convention that was not the one in force.
    nll_probability_floor: float

    @property
    def robot_top_1(self) -> float:
        return self.robot_top_1_hits / self.robot_turns if self.robot_turns else 0.0

    @property
    def robot_top_k(self) -> float:
        return self.robot_top_k_hits / self.robot_turns if self.robot_turns else 0.0

    @property
    def scored_discriminating_top_1(self) -> Optional[float]:
        if not self.scored_discriminating_decisions:
            return None
        return (
            self.scored_discriminating_top_1_hits
            / self.scored_discriminating_decisions
        )

    @property
    def preference_sequence(self) -> Tuple[str, ...]:
        return tuple(decision.actual for decision in self.decisions)

    @property
    def post_update_recurrence(self) -> bool:
        return self.task.exposure_after_change >= 2


class CookingHrcRunner:
    """Execute a stream with player 0 as human and player 1 as robot.

    On an incorrect robot proposal, the proposal is vetoed, player 0 performs
    the ground-truth option, and the next scheduled turn remains with the
    robot.  A recipe's first occurrence is a natural human demonstration;
    later occurrences are assistance episodes.
    """

    def __init__(
        self,
        agent: Any,
        runtime: BurritoRuntime,
        domain: CookingDomainAdapter,
        *,
        horizon: int = 1600,
        planner_seed: int = 0,
        max_passive_wait_ticks: int = 400,
        top_k: int = 3,
        memory_updates_enabled: bool = True,
        require_shift_update: bool = True,
        lead_actor_policy: str = HUMAN_FIRST,
        nll_probability_floor: float | None = None,
    ):
        if getattr(agent, "domain", None) is not domain:
            raise ValueError("agent and runner must share one domain adapter")
        self.agent = agent
        self.runtime = runtime
        self.domain = domain
        self.horizon = int(horizon)
        self.planner_seed = int(planner_seed)
        self.max_passive_wait_ticks = max(1, int(max_passive_wait_ticks))
        self.top_k = max(1, int(top_k))
        # Matches src.evaluation, which floors at
        # max(agent.settings.min_probability, 1e-12); taking it from the agent
        # keeps one convention across both environments instead of two
        # constants that happen to differ by six orders of magnitude.
        self.nll_probability_floor = max(
            float(
                getattr(getattr(agent, "settings", None), "min_probability", 1e-6)
                if nll_probability_floor is None else nll_probability_floor
            ),
            1e-12,
        )
        self.memory_updates_enabled = bool(memory_updates_enabled)
        self.require_shift_update = bool(require_shift_update)
        if lead_actor_policy not in LEAD_ACTOR_POLICIES:
            raise ValueError(
                f"lead_actor_policy must be one of {LEAD_ACTOR_POLICIES}"
            )
        self.lead_actor_policy = str(lead_actor_policy)
        self._assist_exposures: Dict[str, int] = {}
        self._observed_recipes: set[str] = set()
        self._learner_recipe_by_task: Dict[str, str] = {}
        self._executors: Dict[str, Any] = {}

    @property
    def observed_recipes(self) -> Tuple[str, ...]:
        return tuple(sorted(self._observed_recipes))

    @property
    def learner_recipe_by_task(self) -> Mapping[str, str]:
        """Read-only external-to-internal identity map for evaluator audits."""
        return dict(self._learner_recipe_by_task)

    def _executor(self, recipe_id: str) -> Any:
        if recipe_id not in self._executors:
            self._executors[recipe_id] = create_executor(
                self.runtime,
                recipe_id,
                horizon=self.horizon,
                seed=self.planner_seed,
            )
        return self._executors[recipe_id]

    def run_task(self, task: CookingTask) -> CookingEpisodeResult:
        graph = CookingTaskGraph.create(task.recipe_id)
        policy = CookingPreferencePolicy.create(task.preference)
        mode = OBSERVE if task.recipe_id not in self._observed_recipes else ASSIST
        executor = self._executor(task.recipe_id)
        executor.reset()
        self.domain.begin_task(task.recipe_id)
        if mode == OBSERVE:
            self.agent.start_demo()

        decisions: list[CookingDecision] = []
        observations: list[CookingObservation] = []
        completed: list[str] = []
        exposure = self._assist_exposures.get(task.recipe_id, 0)
        robot_turn_next = bool(
            mode == ASSIST
            and self.lead_actor_policy == COUNTERBALANCED
            and exposure % 2 == 1
        )
        robot_turns = robot_hits = robot_top_k_hits = corrections = 0
        scored_turns = scored_hits = scored_top_k_hits = 0
        scored_discriminating = scored_discriminating_hits = 0
        scored_available = robot_available = 0
        human_actions = robot_actions = macro_ticks = passive_wait_ticks = 0
        invalid_predictions = 0
        prediction_wall_s = reward = 0.0
        deliveries_before = executor._delivery_count()
        demo_counter_before = int(self.agent.demo_counter)
        train_before = len(self.agent.retrain_events)

        while not graph.is_complete(completed):
            scheduled_actor = (
                "human" if mode == OBSERVE or not robot_turn_next else "robot"
            )
            decision_actor = 0 if scheduled_actor == "human" else 1
            # A preference ranges over the task-graph frontier, not over
            # whatever happens to be cooked yet.  "Plate the protein first"
            # means waiting for the protein; picking greedily from the
            # currently-legal subset instead lets readiness timing dictate the
            # order, which collapsed every assembly preference into one
            # realized behaviour.
            legal = graph.frontier(completed)
            ground_truth = policy.choose_action(legal, graph)
            # The candidate list is the structural frontier and nothing else.
            # It used to be the physically-legal subset at the first tick where
            # the *ground truth* became executable, which made the predictor's
            # candidate list a function of the answer: competing options that
            # were not ready yet silently disappeared, and 670 of one seed's
            # 45,641 forced robot decisions were forced only by that waiting.
            # Readiness is now waited out after the decision, against whichever
            # option is actually executed.
            state = self.domain.state_from_completed(task.recipe_id, completed)

            distribution: Dict[str, float] = {}
            predicted: Optional[str] = None
            correct = False
            correct_top_k = False
            ground_truth_probability = 0.0
            ground_truth_nll = math.nan
            prediction_elapsed = 0.0
            invalid_prediction = False
            prediction_stats: Dict[str, Any] = {}
            # Every assist decision is scored, including the opening move and
            # the human's own turns.  ``src.hrc_simulation.simulate_episode``
            # records exactly these shadow predictions (it calls ``_predict``
            # on human turns, index 0 included, with an empty prefix), and
            # skipping them here was not parity: in these task graphs the first
            # decision is the single most preference-informative move, so for
            # several recipe/preference pairs it was the *only* discriminating
            # decision and no reported metric could see it.  Shadow predictions
            # never control execution.
            prediction_available = False
            if mode == ASSIST:
                started = time.perf_counter()
                distribution = dict(self.agent.predict_actions(
                    tuple(self.agent.current_prefix),
                    state=state,
                    action_universe=legal,
                ))
                ranked = self.agent.rank_actions(distribution, k=self.top_k)
                prediction_stats = dict(self.agent.policy_stats())
                prediction_elapsed = time.perf_counter() - started
                prediction_wall_s += prediction_elapsed
                predicted = ranked[0] if ranked else None
                prediction_available = predicted is not None
                invalid_prediction = predicted is not None and predicted not in legal
                invalid_predictions += int(invalid_prediction)
                correct = predicted == ground_truth
                correct_top_k = ground_truth in ranked
                ground_truth_probability = max(
                    0.0, float(distribution.get(ground_truth, 0.0))
                )
                # One scoring convention, shared with the symbolic evaluator:
                # floor the ground-truth probability at settings.min_probability
                # (src.evaluation uses max(min_probability, 1e-12) for exactly
                # this) and take the loss.  An arm that emitted nothing has
                # ground-truth probability zero and is charged the floor, the
                # same as an arm that emitted a distribution excluding the
                # truth -- because they are the same event for a scoring rule.
                #
                # A uniform-over-candidates fallback was tried here and is
                # wrong twice over: it silently swaps in a distribution the
                # predictor never emitted, and on a single-candidate frontier
                # it scores an arm that said nothing at zero loss, which is a
                # perfect score for silence.  Availability is reported
                # separately instead.
                ground_truth_nll = -math.log(
                    max(ground_truth_probability, self.nll_probability_floor)
                )

            discriminating = is_preference_discriminating(legal, graph)
            # Two denominators, deliberately both recorded.  ``discriminating``
            # is task-intrinsic: could this recipe's preferences ever disagree
            # here.  ``conditioned`` is what the predictor actually faced: do
            # the preferences still consistent with this episode's prefix
            # disagree.  Under human_first the human takes the widest frontier
            # of every episode, so the two differ by a large factor and only
            # the second is an accuracy denominator.
            conditioned = discriminating and is_prefix_conditioned_discriminating(
                legal, graph, completed,
            )
            if mode == ASSIST:
                scored_turns += 1
                scored_hits += int(correct)
                scored_top_k_hits += int(correct_top_k)
                scored_available += int(prediction_available)
                if discriminating:
                    scored_discriminating += 1
                    scored_discriminating_hits += int(correct)

            corrected = False
            if scheduled_actor == "robot":
                robot_turns += 1
                robot_hits += int(correct)
                robot_top_k_hits += int(correct_top_k)
                robot_available += int(prediction_available)
                if correct and predicted is not None:
                    executed_action = predicted
                    physical_actor = 1
                    executed_by = "robot"
                    robot_actions += 1
                    robot_turn_next = False
                else:
                    executed_action = ground_truth
                    physical_actor = 0
                    executed_by = "human_correction"
                    corrected = True
                    corrections += 1
                    human_actions += 1
                    robot_turn_next = True
            else:
                executed_action = ground_truth
                physical_actor = 0
                executed_by = "human"
                human_actions += 1
                if mode == ASSIST:
                    robot_turn_next = True

            before = state
            # Readiness waiting belongs here, after the decision: the option
            # being executed is already chosen, so waiting for the pot cannot
            # feed back into the candidate list.  Every structural
            # precondition in this catalog is satisfied by elapsed time alone
            # (a plate action only enters the frontier once its pot or grill
            # has been started), so this terminates; max_passive_wait_ticks
            # remains the backstop.
            waited = 0
            while executed_action not in executor.legal_actions(
                (executed_action,), actor_id=physical_actor,
            ):
                if waited >= self.max_passive_wait_ticks:
                    raise RuntimeError(
                        f"{task.recipe_id}: selected option {executed_action} "
                        "never became physically legal"
                    )
                executor.advance_environment(1)
                waited += 1
            passive_wait_ticks += waited
            execution = executor.execute(executed_action, actor_id=physical_actor)
            macro_ticks += execution.low_level_ticks
            reward += execution.sparse_reward
            completed.append(executed_action)
            after = self.domain.state_from_completed(task.recipe_id, completed)
            if self.domain.successor(before, executed_action) != after:
                raise RuntimeError("task transition disagrees with physical option")
            observation = CookingObservation(before, executed_action, after)
            observations.append(observation)
            self.agent.observe(
                observation,
                ground_truth_recipe=task.recipe_id,
                precomputed_distribution=(distribution or None),
                precomputed_prediction=predicted,
            )
            decisions.append(CookingDecision(
                recipe_step=len(completed) - 1,
                mode=mode,
                scheduled_actor=scheduled_actor,
                physical_actor_id=physical_actor,
                executed_by=executed_by,
                legal_actions=tuple(legal),
                ground_truth_action=ground_truth,
                actual=executed_action,
                predicted=predicted,
                correct_top_1=correct,
                correct_top_k=correct_top_k,
                ground_truth_probability=ground_truth_probability,
                ground_truth_nll=ground_truth_nll,
                preference_discriminating=discriminating,
                prefix_conditioned_discriminating=conditioned,
                prediction_available=prediction_available,
                invalid_prediction=invalid_prediction,
                prediction_wall_s=prediction_elapsed,
                prediction_stats=prediction_stats,
                distribution=distribution,
                human_corrected=corrected,
                proposal_executed=scheduled_actor == "robot" and correct,
                low_level_ticks=execution.low_level_ticks,
                passive_wait_ticks_before=waited,
                primitive_calls=execution.primitive_calls,
                primitive_ticks=execution.primitive_ticks,
            ))

        match = self.agent.end_demo()
        learned_recipe = getattr(match, "recipe_id", None)
        if mode == OBSERVE and learned_recipe is not None:
            owner = next((
                recipe_id for recipe_id, internal_id
                in self._learner_recipe_by_task.items()
                if internal_id == learned_recipe and recipe_id != task.recipe_id
            ), None)
            if owner is not None:
                raise RuntimeError(
                    f"open-set learner merged {task.recipe_id} with {owner} "
                    f"as {learned_recipe}"
                )
            self._learner_recipe_by_task[task.recipe_id] = str(learned_recipe)
        elif mode == ASSIST and self.memory_updates_enabled:
            expected_recipe = self._learner_recipe_by_task.get(task.recipe_id)
            if (
                expected_recipe is not None
                and learned_recipe is not None
                and str(learned_recipe) != expected_recipe
            ):
                raise RuntimeError(
                    f"assist episode for {task.recipe_id} was committed to "
                    f"{learned_recipe}, expected {expected_recipe}"
                )
        new_retrain_events = tuple(
            dict(event) for event in self.agent.retrain_events[train_before:]
        )
        memory_age_delta = int(self.agent.demo_counter) - demo_counter_before
        deployment_locked = bool(getattr(self.agent, "_deployment_locked", False))
        expected_delta = int(
            (mode == OBSERVE and not deployment_locked)
            or (mode == ASSIST and self.memory_updates_enabled)
        )
        if memory_age_delta != expected_delta:
            raise RuntimeError(
                f"unexpected memory age delta {memory_age_delta}; expected {expected_delta}"
            )
        if mode == OBSERVE:
            self._observed_recipes.add(task.recipe_id)
        else:
            self._assist_exposures[task.recipe_id] = exposure + 1

        match_key = (
            (match.recipe_id, match.variant_id)
            if getattr(match, "recipe_id", None) is not None
            and getattr(match, "variant_id", None) is not None else None
        )
        commit_applied = bool(
            (mode == OBSERVE and memory_age_delta == 1)
            or self.agent.last_commit_stats.get("commit_applied", False)
        )
        active_rehearsal = bool(
            match_key is not None and match_key in self.agent.replay.active
        )
        retrain_executed = any(
            not bool(event.get("skipped", False)) for event in new_retrain_events
        )
        # A retrain can be correctly skipped: if the shifted variant was
        # already resident in active replay (active_rehearsal=True) from an
        # earlier occurrence, TrainPolicy.decide() sees no addition/removal/
        # weight change *this step* and returns "skip", "replay_unchanged".
        # That's a legitimate no-op, not a missed retrain -- only treat the
        # skip as a failure if the active set actually changed underneath it.
        retrain_correctly_skipped = bool(new_retrain_events) and all(
            bool(event.get("skipped", False))
            and int(event.get("active_added_count", 0) or 0) == 0
            and int(event.get("active_removed_count", 0) or 0) == 0
            and int(event.get("active_weight_changed_count", 0) or 0) == 0
            for event in new_retrain_events
        )
        if (
            mode == ASSIST
            and task.preference_changed
            and task.exposure_after_change == 1
            and self.require_shift_update
            and self.memory_updates_enabled
            and not (
                commit_applied
                and active_rehearsal
                and (retrain_executed or retrain_correctly_skipped)
            )
        ):
            # Name the term that failed. All three are recomputed above from
            # separate agent state, and a bare assertion left the next run as
            # the only way to learn which one was False.
            raise RuntimeError(
                "first natural post-shift exposure did not commit, rehearse, "
                f"and retrain (commit_applied={commit_applied}, "
                f"active_rehearsal={active_rehearsal}, "
                f"retrain_executed={retrain_executed}, "
                f"retrain_correctly_skipped={retrain_correctly_skipped}, "
                f"match_key={match_key!r}, "
                f"retrain_events={new_retrain_events!r})"
            )

        deliveries = executor._delivery_count() - deliveries_before
        expected_deliveries = get_recipe(task.recipe_id).expected_deliveries
        if deliveries != expected_deliveries:
            raise RuntimeError(
                f"{task.recipe_id} should deliver {expected_deliveries} dish(es), "
                f"got {deliveries}"
            )
        return CookingEpisodeResult(
            task=task,
            mode=mode,
            decisions=tuple(decisions),
            observations=tuple(observations),
            match=match,
            robot_turns=robot_turns,
            robot_top_1_hits=robot_hits,
            robot_top_k_hits=robot_top_k_hits,
            corrections=corrections,
            human_actions=human_actions,
            robot_actions=robot_actions,
            invalid_predictions=invalid_predictions,
            scored_turns=scored_turns,
            scored_top_1_hits=scored_hits,
            scored_top_k_hits=scored_top_k_hits,
            scored_discriminating_decisions=scored_discriminating,
            scored_discriminating_top_1_hits=scored_discriminating_hits,
            scored_available_decisions=scored_available,
            robot_available_decisions=robot_available,
            prediction_wall_s=prediction_wall_s,
            low_level_ticks=macro_ticks + passive_wait_ticks,
            passive_wait_ticks=passive_wait_ticks,
            sparse_reward=reward,
            deliveries=deliveries,
            memory_age_delta=memory_age_delta,
            commit_applied=commit_applied,
            active_rehearsal=active_rehearsal,
            retrain_executed=retrain_executed,
            retrain_correctly_skipped=retrain_correctly_skipped,
            retrain_events=new_retrain_events,
            compatibility_dynamics=bool(executor.compatibility_dynamics),
            compatibility_calls=tuple(getattr(executor, "compatibility_calls", ())),
            lead_actor_policy=self.lead_actor_policy,
            nll_probability_floor=self.nll_probability_floor,
        )

    def run_stream(
        self, tasks: Sequence[CookingTask],
    ) -> Tuple[CookingEpisodeResult, ...]:
        return tuple(self.run_task(task) for task in tasks)


BurritoTask = CookingTask
BurritoObservation = CookingObservation
BurritoDecision = CookingDecision
BurritoEpisodeResult = CookingEpisodeResult
BurritoHrcRunner = CookingHrcRunner


__all__ = [
    "ASSIST", "COUNTERBALANCED", "HUMAN_FIRST", "LEAD_ACTOR_POLICIES", "OBSERVE", "BurritoDecision", "BurritoEpisodeResult",
    "BurritoHrcRunner", "BurritoObservation", "BurritoTask",
    "CookingDecision", "CookingEpisodeResult", "CookingHrcRunner",
    "CookingObservation", "CookingTask",
]
