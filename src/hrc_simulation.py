"""Human-robot collaboration rollout policies.

This module owns interaction semantics such as turn-taking, correction, and
time accounting. Experiment harnesses can attach logging/metric hooks, but the
policy itself should not live in experiment setup code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .models import top_k


@dataclass(frozen=True)
class HRCTimingConfig:
    human_action_time: float = 4.0
    robot_correct_action_time: float = 6.0
    robot_wrong_action_time: float = 2.0
    # Intervention/correction effort: noticing the error, interrupting current
    # activity, and executing the correct action costs more than a pre-planned
    # human step.  Equal to human_action_time collapses human_effort_fraction
    # to human_action_fraction, making them redundant (identical floats).
    human_correction_time: float = 5.0
    wrong_action_extra_penalty: float = 3.0


DEFAULT_HRC_TIMING = HRCTimingConfig()


@dataclass(frozen=True)
class HRCDecisionContext:
    recipe_step: int
    robot_turn_idx: int
    prefix: Tuple[str, ...]
    actual: str
    actual_action: str
    distribution: Mapping[str, float]
    ranked: Tuple[str, ...]
    predicted: Optional[str]
    correct_top1: bool
    correct_topk: bool
    future_valid_wrong: bool
    predicted_future_offset: Optional[int]
    prediction_wall_s: float


@dataclass(frozen=True)
class HRCRobotTurn:
    recipe_step: int
    robot_turn_idx: int
    prefix: Tuple[str, ...]
    actual: str
    actual_action: str
    distribution: Dict[str, float]
    ranked: Tuple[str, ...]
    predicted: Optional[str]
    correct_top1: bool
    correct_topk: bool
    future_valid_wrong: bool
    predicted_future_offset: Optional[int]
    prediction_wall_s: float
    scheduled_actor: str
    executed_by: str
    human_corrected: bool
    hrc_step_time: float
    hrc_total_time_so_far: float
    human_actions_so_far: int
    human_effort_time_so_far: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HRCInteractionSummary:
    n_recipe_steps: int
    n_robot_turns: int
    human_turn_count: int
    human_correction_count: int
    robot_correct_count: int
    robot_wrong_count: int
    future_valid_wrong_count: int
    hits1: int
    hitsk: int
    nll: float
    first_mismatch_robot_turn: int
    first_mismatch_recipe_step: int
    hrc_total_time: float
    human_only_time: float
    hrc_robot_correct_time: float
    hrc_robot_wrong_time: float
    hrc_human_action_time: float
    hrc_human_correction_time: float
    hrc_wrong_action_extra_penalty_time: float
    human_action_fraction: float
    human_effort_time: float
    human_effort_fraction: float


@dataclass(frozen=True)
class HRCInteractionTrace:
    robot_turns: Tuple[HRCRobotTurn, ...]
    summary: HRCInteractionSummary


def _action_distribution_stats(dist: Mapping[str, float], ranked: Sequence[str], predicted: Optional[str]) -> Dict[str, Optional[float]]:
    predicted_confidence = float(dist.get(predicted, 0.0)) if predicted is not None else None
    ranked_probs = [float(dist.get(token, 0.0)) for token in ranked]
    margin = None
    if ranked_probs:
        margin = ranked_probs[0] - (ranked_probs[1] if len(ranked_probs) > 1 else 0.0)
    positive_probs = [float(p) for p in dist.values() if float(p) > 0.0]
    if len(positive_probs) > 1:
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in positive_probs) / math.log(len(positive_probs))
    elif len(positive_probs) == 1:
        entropy = 0.0
    else:
        entropy = None
    return {
        "confidence": predicted_confidence,
        "margin": margin,
        "entropy": entropy,
    }


def policy_distribution_diagnostics(dist: Mapping[str, float], ranked: Sequence[str], predicted: Optional[str]) -> Dict[str, Any]:
    """Return policy-distribution diagnostics for a scheduled robot turn."""
    stats = _action_distribution_stats(dist, ranked, predicted)
    confidence = stats["confidence"]
    margin = stats["margin"]
    entropy = stats["entropy"]
    return {
        "policy_confidence": confidence,
        "policy_margin": margin,
        "policy_entropy": entropy,
    }


def run_alternating_hrc_episode(
    *,
    observations: Sequence[Any],
    actual_tokens: Sequence[str],
    actual_actions: Sequence[str],
    current_prefix: Callable[[], Sequence[str]],
    predict_distribution: Callable[[Sequence[str]], Mapping[str, float]],
    observe_ground_truth: Callable[[Any, Optional[Mapping[str, float]]], None],
    topk: int,
    prob_floor: float,
    timing: HRCTimingConfig = DEFAULT_HRC_TIMING,
    capture_robot_metadata: Optional[Callable[[HRCDecisionContext], Mapping[str, Any]]] = None,
) -> HRCInteractionTrace:
    """Run the intended HRC protocol: human first, then robot turns with correction."""
    robot_turn_next = False
    robot_turn_idx = 0
    human_turn_count = 0
    human_correction_count = 0
    robot_correct_count = 0
    robot_wrong_count = 0
    future_valid_wrong_count = 0
    hits1 = 0
    hitsk = 0
    nll = 0.0
    first_mismatch = -1
    first_mismatch_recipe_step = -1
    hrc_total_time = 0.0
    hrc_robot_correct_time = 0.0
    hrc_robot_wrong_time = 0.0
    hrc_human_action_time = 0.0
    hrc_human_correction_time = 0.0
    hrc_extra_penalty_time = 0.0
    robot_turns = []
    floor = max(float(prob_floor), 1e-12)

    for idx, (obs, actual, action_label) in enumerate(zip(observations, actual_tokens, actual_actions)):
        if not robot_turn_next:
            human_turn_count += 1
            hrc_human_action_time += float(timing.human_action_time)
            hrc_total_time += float(timing.human_action_time)
            observe_ground_truth(obs, None)
            robot_turn_next = True
            continue

        prefix = tuple(current_prefix())
        t_pred = time.perf_counter()
        raw_dist = predict_distribution(prefix)
        prediction_wall_s = time.perf_counter() - t_pred
        dist = dict(raw_dist)
        ranked = tuple(top_k(dist, k=max(1, int(topk))) if dist else ())
        predicted = ranked[0] if ranked else None
        correct1 = bool(ranked and ranked[0] == actual)
        correctk = bool(actual in ranked)
        future_offsets = [
            offset + 1
            for offset, future_actual in enumerate(actual_tokens[idx + 1:])
            if predicted is not None and predicted == future_actual
        ]
        future_valid_wrong = bool((not correct1) and future_offsets)
        hits1 += int(correct1)
        hitsk += int(correctk)
        if first_mismatch < 0 and not correct1:
            first_mismatch = robot_turn_idx
            first_mismatch_recipe_step = idx
        nll -= math.log(max(float(dist.get(actual, floor)), floor))

        if correct1:
            step_time = float(timing.robot_correct_action_time)
            hrc_robot_correct_time += float(timing.robot_correct_action_time)
            robot_correct_count += 1
            executed_by = "robot"
            robot_turn_next = False
        else:
            step_time = (
                float(timing.robot_wrong_action_time)
                + float(timing.human_correction_time)
                + float(timing.wrong_action_extra_penalty)
            )
            hrc_robot_wrong_time += float(timing.robot_wrong_action_time)
            hrc_human_correction_time += float(timing.human_correction_time)
            hrc_extra_penalty_time += float(timing.wrong_action_extra_penalty)
            human_correction_count += 1
            robot_wrong_count += 1
            future_valid_wrong_count += int(future_valid_wrong)
            executed_by = "human_correction"
            robot_turn_next = True

        context = HRCDecisionContext(
            recipe_step=idx,
            robot_turn_idx=robot_turn_idx,
            prefix=prefix,
            actual=actual,
            actual_action=action_label,
            distribution=dist,
            ranked=ranked,
            predicted=predicted,
            correct_top1=correct1,
            correct_topk=correctk,
            future_valid_wrong=future_valid_wrong,
            predicted_future_offset=future_offsets[0] if future_offsets else None,
            prediction_wall_s=float(prediction_wall_s),
        )
        metadata = dict(capture_robot_metadata(context) or {}) if capture_robot_metadata is not None else {}
        hrc_total_time += step_time
        human_actions_so_far = human_turn_count + human_correction_count
        human_effort_time_so_far = (
            human_turn_count * float(timing.human_action_time)
            + human_correction_count * float(timing.human_correction_time)
        )
        robot_turns.append(HRCRobotTurn(
            recipe_step=idx,
            robot_turn_idx=robot_turn_idx,
            prefix=prefix,
            actual=actual,
            actual_action=action_label,
            distribution=dist,
            ranked=ranked,
            predicted=predicted,
            correct_top1=correct1,
            correct_topk=correctk,
            future_valid_wrong=future_valid_wrong,
            predicted_future_offset=future_offsets[0] if future_offsets else None,
            prediction_wall_s=float(prediction_wall_s),
            scheduled_actor="robot",
            executed_by=executed_by,
            human_corrected=not correct1,
            hrc_step_time=float(step_time),
            hrc_total_time_so_far=float(hrc_total_time),
            human_actions_so_far=int(human_actions_so_far),
            human_effort_time_so_far=float(human_effort_time_so_far),
            metadata=metadata,
        ))
        robot_turn_idx += 1
        observe_ground_truth(obs, dist)

    human_only_time = float(len(actual_tokens) * float(timing.human_action_time))
    human_action_fraction = float((human_turn_count + human_correction_count) / max(1, len(actual_tokens)))
    human_effort_time = float(
        human_turn_count * float(timing.human_action_time)
        + human_correction_count * float(timing.human_correction_time)
    )
    human_effort_fraction = float(human_effort_time / max(human_only_time, 1e-12)) if human_only_time > 0.0 else 0.0
    return HRCInteractionTrace(
        robot_turns=tuple(robot_turns),
        summary=HRCInteractionSummary(
            n_recipe_steps=len(actual_tokens),
            n_robot_turns=robot_turn_idx,
            human_turn_count=human_turn_count,
            human_correction_count=human_correction_count,
            robot_correct_count=robot_correct_count,
            robot_wrong_count=robot_wrong_count,
            future_valid_wrong_count=future_valid_wrong_count,
            hits1=hits1,
            hitsk=hitsk,
            nll=float(nll),
            first_mismatch_robot_turn=first_mismatch,
            first_mismatch_recipe_step=first_mismatch_recipe_step,
            hrc_total_time=float(hrc_total_time),
            human_only_time=human_only_time,
            hrc_robot_correct_time=float(hrc_robot_correct_time),
            hrc_robot_wrong_time=float(hrc_robot_wrong_time),
            hrc_human_action_time=float(hrc_human_action_time),
            hrc_human_correction_time=float(hrc_human_correction_time),
            hrc_wrong_action_extra_penalty_time=float(hrc_extra_penalty_time),
            human_action_fraction=human_action_fraction,
            human_effort_time=human_effort_time,
            human_effort_fraction=human_effort_fraction,
        ),
    )
