"""HRC turn-taking, correction, and time-accounting policies."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

from .models import top_actions, top_probability_tie_size


@dataclass(frozen=True)
class Timing:
    human_action_time: float = 4.0
    robot_correct_action_time: float = 4.0
    robot_wrong_action_time: float = 2.0
    # Includes recognition, rollback, and the correct human action.
    human_correction_time: float = 8.0


DEFAULT_TIMING = Timing()


@dataclass(frozen=True)
class Prediction:
    recipe_step: int
    prefix: Tuple[str, ...]
    actual: str
    distribution: Mapping[str, float]
    ranked: Tuple[str, ...]
    predicted: Optional[str]
    correct_top_1: bool
    correct_top_k: bool
    top_1_tie_size: int
    prediction_time: float


@dataclass(frozen=True)
class RobotDecision(Prediction):
    robot_turn_index: int
    matches_later_action: bool
    matching_action_offset: Optional[int]


@dataclass(frozen=True)
class RobotTurn(RobotDecision):
    scheduled_actor: str
    executed_by: str
    human_corrected: bool
    step_time: float
    elapsed_time: float
    human_actions: int
    human_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanTurn(Prediction):
    human_turn_index: int
    scheduled_actor: str = "human"
    executed_by: str = "human"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSummary:
    recipe_steps: int
    robot_turn_count: int
    human_turn_count: int
    human_correction_count: int
    robot_correct_count: int
    robot_wrong_count: int
    later_action_errors: int
    top_1_hits: int
    top_k_hits: int
    robot_top_1_tie_count: int
    teacher_forced_top_1_tie_count: int
    log_loss: float
    first_mismatch_robot_turn: int
    first_mismatch_recipe_step: int
    total_time: float
    human_only_time: float
    correct_robot_time: float
    wrong_robot_time: float
    human_action_time: float
    correction_time: float
    human_time: float


@dataclass(frozen=True)
class EpisodeTrace:
    robot_turns: Tuple[RobotTurn, ...]
    human_shadow_turns: Tuple[HumanTurn, ...]
    summary: EpisodeSummary


def _predict(recipe_step: int, prefix: Tuple[str, ...], actual: str, predict_distribution: Callable[[Sequence[str]], Mapping[str, float]], top_k: int, tie_rng: np.random.Generator) -> Prediction:
    started = time.perf_counter()
    distribution = dict(predict_distribution(prefix))
    elapsed = time.perf_counter() - started
    ranked = tuple(top_actions(distribution, k=max(1, int(top_k)), rng=tie_rng) if distribution else ())
    return Prediction(recipe_step=recipe_step, prefix=prefix, actual=actual, distribution=distribution, ranked=ranked, predicted=ranked[0] if ranked else None, correct_top_1=bool(ranked and ranked[0] == actual),
        correct_top_k=actual in ranked, top_1_tie_size=top_probability_tie_size(distribution), prediction_time=float(elapsed))


def simulate_episode(*, observations: Sequence[Any], actual_actions: Sequence[str], current_prefix: Callable[[], Sequence[str]], predict_distribution: Callable[[Sequence[str]], Mapping[str, float]],
    observe_ground_truth: Callable[[Any, Optional[Mapping[str, float]], Optional[str]], None], top_k: int, min_probability: float, tie_rng: Optional[np.random.Generator] = None, timing: Timing = DEFAULT_TIMING,
    capture_prediction_metadata: Optional[Callable[[Prediction], Mapping[str, Any]]] = None,
    capture_robot_metadata: Optional[Callable[[RobotDecision], Mapping[str, Any]]] = None, on_robot_feedback: Optional[Callable[[RobotDecision], None]] = None) -> EpisodeTrace:
    """Run the HRC protocol: human first, then alternating robot/correction turns."""
    robot_turn_next = False
    robot_turn_index = human_turn_count = human_correction_count = 0
    robot_correct_count = robot_wrong_count = later_action_errors = 0
    top_1_hits = top_k_hits = 0
    robot_top_1_tie_count = teacher_forced_top_1_tie_count = 0
    log_loss = 0.0
    first_mismatch = first_mismatch_recipe_step = -1
    total_time = correct_robot_time = wrong_robot_time = 0.0
    human_action_time = correction_time = 0.0
    robot_turns = []
    human_turns = []
    floor = max(float(min_probability), 1e-12)
    reference_positions: Dict[str, list[int]] = defaultdict(list)
    decision_rng = tie_rng if tie_rng is not None else np.random.default_rng(0)
    for position, action in enumerate(actual_actions): reference_positions[action].append(position)

    for index, (observation, actual) in enumerate(zip(observations, actual_actions)):
        prefix = tuple(current_prefix())
        if not robot_turn_next:
            human_turn_count += 1
            # Shadow predictions never control execution. Together with the robot-turn predictions they score every ground-truth prefix.
            prediction = _predict(index, prefix, actual, predict_distribution, top_k, decision_rng)
            teacher_forced_top_1_tie_count += int(prediction.top_1_tie_size > 1)
            metadata = (dict(capture_prediction_metadata(prediction) or {}) if capture_prediction_metadata is not None else {})
            human_turns.append(HumanTurn(**vars(prediction), human_turn_index=human_turn_count - 1, metadata=metadata))
            human_action_time += timing.human_action_time
            total_time += timing.human_action_time
            observe_ground_truth(observation, prediction.distribution, prediction.predicted)
            robot_turn_next = True
            continue

        prediction = _predict(index, prefix, actual, predict_distribution, top_k, decision_rng)
        tied = int(prediction.top_1_tie_size > 1)
        robot_top_1_tie_count += tied
        teacher_forced_top_1_tie_count += tied
        positions = reference_positions.get(prediction.predicted or "", ())
        next_position_index = bisect_right(positions, index)
        next_position = (positions[next_position_index] if next_position_index < len(positions) else None)
        matching_reference_offset = (next_position - index if next_position is not None else None)
        matches_later_reference_action = bool(not prediction.correct_top_1 and matching_reference_offset is not None)
        decision = RobotDecision(**vars(prediction), robot_turn_index=robot_turn_index, matches_later_action=(matches_later_reference_action), matching_action_offset=matching_reference_offset)
        top_1_hits += int(decision.correct_top_1)
        top_k_hits += int(decision.correct_top_k)
        if first_mismatch < 0 and not decision.correct_top_1:
            first_mismatch = robot_turn_index
            first_mismatch_recipe_step = index
        log_loss -= math.log(max(float(decision.distribution.get(actual, floor)), floor))

        if decision.correct_top_1:
            step_time = timing.robot_correct_action_time
            correct_robot_time += timing.robot_correct_action_time
            robot_correct_count += 1
            executed_by = "robot"
            robot_turn_next = False
        else:
            step_time = timing.robot_wrong_action_time + timing.human_correction_time
            wrong_robot_time += timing.robot_wrong_action_time
            correction_time += timing.human_correction_time
            human_correction_count += 1
            robot_wrong_count += 1
            later_action_errors += int(matches_later_reference_action)
            executed_by = "human_correction"
            robot_turn_next = True

        metadata = (dict(capture_prediction_metadata(decision) or {}) if capture_prediction_metadata is not None else {})
        if capture_robot_metadata is not None:
            metadata.update(dict(capture_robot_metadata(decision) or {}))
        if on_robot_feedback is not None: on_robot_feedback(decision)
        total_time += step_time
        human_actions = human_turn_count + human_correction_count
        human_effort = (human_turn_count * timing.human_action_time + human_correction_count * timing.human_correction_time)
        robot_turns.append(RobotTurn(**vars(decision), scheduled_actor="robot", executed_by=executed_by, human_corrected=not decision.correct_top_1, step_time=float(step_time), elapsed_time=float(total_time),
                        human_actions=human_actions, human_time=float(human_effort), metadata=metadata))
        robot_turn_index += 1
        observe_ground_truth(observation, decision.distribution, decision.predicted)

    human_only_time = len(actual_actions) * timing.human_action_time
    human_time = (human_turn_count * timing.human_action_time + human_correction_count * timing.human_correction_time)
    return EpisodeTrace(robot_turns=tuple(robot_turns), human_shadow_turns=tuple(human_turns), summary=EpisodeSummary(recipe_steps=len(actual_actions), robot_turn_count=robot_turn_index, human_turn_count=human_turn_count,
            human_correction_count=human_correction_count, robot_correct_count=robot_correct_count, robot_wrong_count=robot_wrong_count, later_action_errors=later_action_errors, top_1_hits=top_1_hits, top_k_hits=top_k_hits,
            robot_top_1_tie_count=robot_top_1_tie_count, teacher_forced_top_1_tie_count=teacher_forced_top_1_tie_count, log_loss=float(log_loss), first_mismatch_robot_turn=first_mismatch, first_mismatch_recipe_step=first_mismatch_recipe_step,
            total_time=float(total_time), human_only_time=float(human_only_time), correct_robot_time=float(correct_robot_time), wrong_robot_time=float(wrong_robot_time), human_action_time=float(human_action_time),
            correction_time=float(correction_time), human_time=float(human_time)))
